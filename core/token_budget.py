"""
Token 与成本控制，适配到金融舆情问答场景：
1. **Token溢出阈值的动态设定**：不用模型极限做阈值，用极限的 TOKEN_BUDGET_RATIO（默认80%），
   预留空间给系统提示词和模型输出（输出Token与输入共享总预算）。
2. **智能截断兜底策略**：淘汰顺序 = 最早的普通对话轮次 > 最早的RAG检索片段。
   永远保留 System Prompt 和当前最新一轮用户问题。
3. **关键信息锚定**：用轻量规则（正则）从当前问题里提取"关键实体"（公司名/股票代码），
   强制保留包含这些实体的对话轮次和检索片段，即使它们本应被淘汰。
   （公司名/股票代码是强模式，用正则即可，不必引入BERT）
"""
import re

from config.settings import settings


# ---------- Token 估算 ----------

def _is_cjk(ch: str) -> bool:
    return "一" <= ch <= "鿿"


def estimate_tokens(text) -> int:
    """
    启发式估算token数（未引入tiktoken，避免额外依赖）：
    - 中文字符按 1 字符 ≈ 1 token 估
    - 其余字符（英文/数字/符号）按 4 字符 ≈ 1 token 估
    估算偏保守一点，宁可多留余量，不可超限。
    """
    if not text:
        return 0
    text = str(text)
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    return int(cjk + other / 4) + 1


def fit_text(text: str, budget: int) -> str:
    """把文本截断到不超过budget token（二分求最大前缀），超出部分用…标记"""
    if estimate_tokens(text) <= budget:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + "…"


# ---------- 上下文窗口与预算 ----------

def get_context_limit(model_name: str = None) -> int:
    """已知模型的上下文窗口（token）。Claude家族一般200K，GPT-4o-mini为128K。"""
    if settings.MODEL_CONTEXT_LIMIT > 0:
        return settings.MODEL_CONTEXT_LIMIT
    name = (model_name or settings.ANTHROPIC_MODEL or settings.OPENAI_MODEL or "").lower()
    if "claude" in name:
        return 200_000
    if "gpt-4o-mini" in name:
        return 128_000
    return 128_000


def get_token_budget(model_name: str = None) -> int:
    """输入预算 = 极限值 × TOKEN_BUDGET_RATIO（默认80%），预留系统提示词和输出空间"""
    return int(get_context_limit(model_name) * settings.TOKEN_BUDGET_RATIO)


def plan_budget(query: str, model_name: str = None) -> tuple:
    """
    把可用预算拆成 对话历史预算 / RAG文档预算 两部分，返回 (conv_budget, docs_budget)。
    先扣掉系统提示词预留 + 当前问题的token，剩下的按 TOKEN_CONV_SHARE 分给对话历史，
    其余给检索文档（文档是本轮回答的核心依据，占比更大）。
    """
    budget = get_token_budget(model_name)
    avail = max(budget - settings.TOKEN_SYSTEM_RESERVED - estimate_tokens(query), 2000)
    conv_budget = int(avail * settings.TOKEN_CONV_SHARE)
    docs_budget = avail - conv_budget
    return conv_budget, docs_budget


# ---------- 关键信息锚定 ----------

# 中文公司名：Xxx公司/集团/股份/银行/证券/控股/科技/能源等常见A股公司名后缀。
# 前缀限制2-6字——中文公司名的前缀一般2-4字（海天味业=4、宁德=2），
# 限制长度避免把"海天味业集团和宁德时代"这种并列公司整串误判成一个名字；
# 开头加断言禁止以连词（和/与/及/或/同/标点）开头，避免"和宁德时代"把"和"吃进来。
# 启发式规则，覆盖大部分常见公司名；个别不带后缀的知名公司（如比亚迪）锚定不到，
# 只影响"强制保留"这一个优化，不影响正常检索，属可接受降级。
_COMPANY_RE = re.compile(
    r"(?![和与及或同、，,])[一-鿿]{2,6}"
    r"(?:公司|集团|股份|控股|银行|证券|保险|基金|科技|能源|电子|医药|生物|"
    r"医疗|汽车|地产|材料|传媒|食品|家电|航空|时代|智能)"
)
# A股股票代码：6位数字，可选 .SH/.SZ/.BJ 后缀。
# 用 (?<!\d)/(?!\d) 而不是 \b 判断边界——\b 在 ASCII字母/数字 与 CJK 之间不成立，
# 会导致 "600519.SH怎么样"（后跟中文）匹配不到。
_STOCK_RE = re.compile(r"(?<!\d)\d{6}(?:\.(?:SH|SZ|BJ))?(?!\d)")


def extract_anchor_entities(query: str) -> list:
    """从当前问题提取关键实体（公司名/股票代码），用于强制保留相关上下文"""
    if not query:
        return []
    entities = _COMPANY_RE.findall(query) + _STOCK_RE.findall(query)
    # 去重且去掉太短的噪声
    return list(dict.fromkeys(e for e in entities if len(e) >= 2))


# ---------- 动态截断 ----------

def trim_turns(turns: list, budget: int, anchor_entities: list = None) -> list:
    """
    截断对话轮次：淘汰最老的轮次，保留最近的；锚定实体命中的轮次强制保留；
    最新一轮永远保留。返回仍按时间顺序排列的轮次列表。
    """
    if not turns:
        return []
    anchor_entities = anchor_entities or []

    def turn_text(t):
        return f"用户：{t['user']}\n助手：{t['assistant']}"

    def is_anchored(t):
        return any(e and (e in t["user"] or e in t["assistant"]) for e in anchor_entities)

    latest = turns[-1]  # 最新一轮永远保留（当前问题相关的对话）
    others = turns[:-1]

    anchored = [t for t in others if is_anchored(t)]
    total = estimate_tokens(turn_text(latest)) + sum(estimate_tokens(turn_text(t)) for t in anchored)

    kept = list(anchored)
    for t in reversed(others):
        if t in anchored:
            continue
        cost = estimate_tokens(turn_text(t))
        if total + cost > budget:
            continue  # 从最老的开始淘汰
        kept.append(t)
        total += cost

    kept.append(latest)
    order = {id(t): i for i, t in enumerate(turns)}
    kept.sort(key=lambda t: order[id(t)])
    return kept


def trim_docs(docs: list, budget: int, anchor_entities: list = None) -> list:
    """
    截断RAG检索文档：保留检索序靠前的文档（相关性高），超预算时从尾部开始
    截断文档正文；锚定实体命中的文档强制保留（即使超出预算——它们通常很少）。
    返回截断后的Document列表。
    """
    if not docs:
        return docs
    anchor_entities = anchor_entities or []

    anchored_ids = {
        id(d) for d in docs
        if any(e and e in d.page_content for e in anchor_entities)
    }

    kept = []
    total = 0
    for d in docs:
        cost = estimate_tokens(d.page_content)
        if id(d) in anchored_ids:
            kept.append(d)
            total += cost
            continue
        if total + cost > budget:
            remaining = budget - total
            if remaining > 30:  # 至少给一条能看的内容
                from langchain_core.documents import Document
                kept.append(Document(
                    page_content=fit_text(d.page_content, remaining),
                    metadata=d.metadata,
                ))
            break
        kept.append(d)
        total += cost
    return kept
