"""
幻觉控制：RAG回答与检索资料的事实一致性程序化校验。
三层校验（每层独立、整体可开关）：
  第1层 引用核查：校验回答中 [编号] 引用是否真实存在（防"编造出处"）；
  第2层 数值/实体忠实性：抽取回答中的数字与公司名逐个回查资料原文；
  第3层 LLM蕴含复核：NLI式判断"回答是否完全被资料支持"，兜住语义级编造。
关键工程决策：校验是"风险信号"不是"硬拦截"——触发带警告的重新生成、加"未经核实"
标注、写入state供调用方审计，绝不阻断主流程；模块自身全try/except包裹，任何一层失败
都降级为"不标记"，校验挂了不能让问答也挂。开关为环境变量 QA_VERIFY（1/0，默认1，
动态读取）。如实拒答（"现有资料有限"）不算幻觉：拒答回答跳过第2/3层，避免把
"正确地承认不知道"误判成风险。用法：verify_answer(query, answer, docs) 只校验，verify_with_regenerate(...) 校验+受限重生成。
"""
import json
import os
import re

# ---------- 开关 ----------

def verify_enabled() -> bool:
    """动态读取开关（不缓存在settings里，支持评测时同进程内切baseline/verified对比）"""
    return os.getenv("QA_VERIFY", "1").strip().lower() not in ("0", "false", "no", "off")


# ---------- 拒答识别（拒答=正确行为，不算幻觉） ----------

# 强拒答语：出现即声明"答不了"。校准口径：
# 真拒答（库外问题）一律在开头就声明无法回答；而库内的实质性回答经常
# 以"根据现有资料……"开头、把"现有资料有限/未提及X"作为结尾的边界说明——
# 若按"全文含任一关键词即拒答"判定，库内回答会被大量误判为拒答。
_STRONG_REFUSAL_RE = re.compile(
    r"无法[^，。]{0,12}(回答|给出|提供|判断|确认|核实|预测|估算)|"
    r"没有关于|没有任何|未检索到|没有检索到|查无"
)


def is_refusal(answer: str) -> bool:
    """规则判定：回答是否属于'如实说明资料不足'类拒答（拒答=正确行为，不算幻觉）。
    判定口径：强拒答语出现在前120字符内 → 拒答（开头即声明答不了，后文只是解释原因）；
    回答很短（<80字符）且含强拒答语 → 拒答（简短兜底句）；"现有资料有限/未提及"这类
    **局限性说明不算拒答**（通常跟在实质性回答后面，按拒答处理会跳过第2/3层校验并
    污染"过度拒答率"指标）。误杀率优先往低了调——宁可漏判（多查一层）不可错杀。"""
    if not answer:
        return False
    if _STRONG_REFUSAL_RE.search(answer[:120]):
        return True
    return len(answer) < 80 and bool(_STRONG_REFUSAL_RE.search(answer))


# ---------- 第1层：引用编号核查 ----------

_CITE_RE = re.compile(r"\[(\d{1,2})\]")


def check_citations(answer: str, n_docs: int) -> dict:
    """校验回答中的[编号]引用。要求：有引用、且编号都在资料范围内。
    注意：引用编号本身会被第2层的数字抽取误伤，抽取前先移除引用标记。"""
    refs = [int(m) for m in _CITE_RE.findall(answer or "")]
    invalid = sorted({r for r in refs if r < 1 or r > n_docs})
    return {
        "has_refs": len(refs) > 0,
        "total_refs": len(refs),
        "invalid_refs": invalid,
    }


# ---------- 第2层：数值/实体忠实性 ----------

# 数字+可选量纲。量纲表覆盖本项目语料的高频单位（金额/百分比/销量/产能/日期粒度）
# 量纲表刻意不含"年/月/日/天"：模型回答常补写"来源：XX，2026-09-25"这类
# 取自文档元数据的日期，而日期格式在回答里又会被改写（"9月8日" vs "2026-09-08"），
# 逐字回查必然系统性误报（评测发现的风险几乎全是日期误报）。
# 日期属低危幻觉，交由第3层LLM蕴含复核兜底。
_NUM_RE = re.compile(
    r"\d+(?:\.\d+)?(?:\s*[万亿千百])?(?:\s*(?:%|元|美元|倍|座|辆|吨|个|人|家|款|GW|MW|Wh|kg|km|基点))?"
)
# 引用标记（先移除再抽数字，避免[1][2]被当成数字）
_MARK_RE = re.compile(r"\[\d{1,2}\]")
# 行首列表序号（"1. xxx"的1不是事实数字）
_LIST_NUM_RE = re.compile(r"(?m)^\s*\d{1,2}[\.、\)]\s*")


def _normalize(s: str) -> str:
    return (s or "").replace(",", "").replace("，", "").replace(" ", "")


def check_numbers(answer: str, docs_text: str) -> list:
    """抽取回答中带量纲的数字，逐个回查资料原文。返回资料中找不到的数字列表。
    抑制误报规则：引用标记[n]与行首列表序号先移除；纯1~2位整数且无量纲的跳过
    （几乎全是序号/枚举，带量纲的个位数如"下跌3%"会保留）；语料侧同样做
    逗号/空格归一后再比对。
    """
    text = _LIST_NUM_RE.sub("", _MARK_RE.sub("", answer or ""))
    corpus = _normalize(docs_text)
    unfaithful = []
    for m in _NUM_RE.finditer(text):
        tok = _normalize(m.group())
        if not tok:
            continue
        # 纯1~2位整数且无任何量纲 → 大概率是序号/枚举，跳过
        if tok.isdigit() and len(tok) <= 2:
            continue
        # 纯4位数字=年份（"2026"），跳过（理由见上方量纲表注释）
        if tok.isdigit() and len(tok) == 4:
            continue
        if tok not in corpus:
            unfaithful.append(tok)
    # 去重保序
    seen, out = set(), []
    for t in unfaithful:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def check_companies(answer: str, docs_text: str) -> list:
    """回答中提到的种子库公司，若其canonical名与别名都不在任何一份资料里，视为幻觉信号。
    （种子库=knowledge_graph/seed_data.py，是全项目公司信息的唯一数据源）"""
    from knowledge_graph.seed_data import COMPANIES  # 唯一数据源，轻量无副作用

    unknown = []
    for canonical, info in COMPANIES.items():
        aliases = [canonical] + [a for a in info.get("aliases", []) if a]
        mentioned = next((a for a in aliases if a and a in (answer or "")), None)
        if mentioned and canonical not in docs_text and mentioned not in docs_text:
            unknown.append(mentioned)
    return unknown


# ---------- 第3层：LLM蕴含复核 ----------

_ENTAILMENT_PROMPT = """你是一名严格的事实核查员。下面是检索到的编号资料和一段基于资料的回答。
请判断回答中的事实性论断是否完全被资料支持。

资料：
{context}

回答：
{answer}

判定规则：
- 只要回答里出现资料中不存在的具体数字、日期、事件或结论，supported就是false，并把它们列入unsupported_claims
- 回答如果只是如实说明"现有资料有限，无法回答"，supported=true（正确的拒答不是幻觉）
- 回答对资料的合理概括、同义改写不算编造
- 只输出JSON，不要输出其他任何内容：
{{"supported": true或false, "unsupported_claims": ["资料中不存在的具体论断"], "reason": "一句话理由"}}"""


def llm_entailment(query: str, answer: str, docs: list) -> dict:
    """LLM蕴含复核。任何失败（调用异常/JSON解析失败）都返回 supported=None，
    即该层降级为不参与风险判定——校验挂了不能影响问答主流程。"""
    # 惰性导入避免循环依赖（qa_chain -> faithfulness模块级互引）
    from chains.qa_chain import get_llm, format_docs_numbered

    context = format_docs_numbered(docs)
    prompt = _ENTAILMENT_PROMPT.format(context=context, answer=answer)
    try:
        resp = get_llm().invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {"supported": None, "unsupported_claims": [], "reason": "LLM输出无法解析为JSON，本层降级"}
        data = json.loads(m.group())
        supported = data.get("supported")
        if not isinstance(supported, bool):
            supported = None
        return {
            "supported": supported,
            "unsupported_claims": [str(c) for c in data.get("unsupported_claims", [])][:5],
            "reason": str(data.get("reason", ""))[:200],
        }
    except Exception as e:
        return {"supported": None, "unsupported_claims": [], "reason": f"LLM复核失败降级: {type(e).__name__}"}


# ---------- 汇总 ----------

HALLUCINATION_WARNING = "⚠️ 系统核查提示：以下回答中部分内容未能通过事实一致性校验（可能是模型编造），请谨慎采信、以原始资料为准。\n\n"


def verify_answer(query: str, answer: str, docs: list, run_llm_check: bool = True) -> dict:
    """
    三层校验汇总。返回dict：
      risk(bool 是否幻觉风险，唯一对外关心的顶层信号)、refusal(bool 如实拒答，拒答跳过2/3层)、
      citations(dict 第1层结果)、unfaithful_numbers/unknown_companies(list 第2层结果)、
      llm(dict 第3层结果，supported=True/False/None)、layers_triggered(list 触发的层名)、
      summary(str 风险摘要，作为strict重生成的原因提示)。
    """
    # 问题原文并入回查范围：答案复述问题中的数字/公司（如"这13家公司"）是
    # 给定信息不是生成内容，不能算编造。
    docs_text = "\n".join(d.page_content for d in (docs or [])) + "\n" + (query or "")
    refusal = is_refusal(answer)

    citations = check_citations(answer, len(docs or []))
    unfaithful_numbers = [] if refusal else check_numbers(answer, docs_text)
    unknown_companies = [] if refusal else check_companies(answer, docs_text)

    llm_res = {"supported": None, "unsupported_claims": [], "reason": "拒答回答跳过LLM复核"}
    if run_llm_check and not refusal:
        llm_res = llm_entailment(query, answer, docs)

    layers = []
    if citations["invalid_refs"]:
        layers.append("citation_invalid")
    if unfaithful_numbers:
        layers.append("number_unfaithful")
    if unknown_companies:
        layers.append("company_unknown")
    if llm_res["supported"] is False:
        layers.append("llm_not_supported")

    parts = []
    if citations["invalid_refs"]:
        parts.append(f"引用了不存在的资料编号{citations['invalid_refs']}")
    if unfaithful_numbers:
        parts.append(f"出现了资料中不存在的数字：{', '.join(unfaithful_numbers[:5])}")
    if unknown_companies:
        parts.append(f"提及了检索资料中没有的公司：{', '.join(unknown_companies[:5])}")
    if llm_res["supported"] is False:
        parts.append(f"LLM复核判定存在无依据论断：{llm_res['reason']}")

    return {
        "risk": bool(layers),
        "refusal": refusal,
        "citations": citations,
        "unfaithful_numbers": unfaithful_numbers,
        "unknown_companies": unknown_companies,
        "llm": llm_res,
        "layers_triggered": layers,
        "summary": "；".join(parts) if parts else "",
    }


def verify_with_regenerate(query: str, answer: str, docs: list,
                           conversation_history: str = "", max_regen: int = 1) -> tuple:
    """
    校验 + 受限重生成（生产verify_node与评测脚本共用）：1.校验当前回答；2.有风险则
    带具体风险原因(strict_reason)重新生成一次（上限max_regen次）；3.重生成结果风险层
    更少或无风险才采纳，否则保留原回答（避免越改越差）；4.最终仍有风险则在回答前加
    "未经核实"警告标注，不删除内容（保留审计线索）。返回 (final_answer, verification_result)。
    """
    from chains.qa_chain import generate_answer  # 惰性导入避免循环依赖

    res = verify_answer(query, answer, docs)
    regen = 0
    while res["risk"] and regen < max_regen:
        regen += 1
        answer2 = generate_answer(
            query, docs, conversation_history=conversation_history,
            strict=True, strict_reason=res["summary"],
        )
        res2 = verify_answer(query, answer2, docs)
        if (not res2["risk"]) or len(res2["layers_triggered"]) <= len(res["layers_triggered"]):
            answer, res = answer2, res2
    res["regen_attempts"] = regen
    if res["risk"]:
        answer = HALLUCINATION_WARNING + (answer or "")
    return answer, res


# ================= 异步版（README问题①异步化改造，2026-10-07） =================
# 校验/重生成的判定逻辑与同步版完全一致（规则层本就是纯CPU直接复用）；
# 区别仅在LLM调用（蕴含复核/重生成）走 ainvoke，等待期间不阻塞事件循环。

async def llm_entailment_async(query: str, answer: str, docs: list) -> dict:
    """llm_entailment 的异步版，降级语义一致：任何失败返回 supported=None。"""
    from chains.qa_chain import get_llm, format_docs_numbered

    context = format_docs_numbered(docs)
    prompt = _ENTAILMENT_PROMPT.format(context=context, answer=answer)
    try:
        resp = await get_llm().ainvoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {"supported": None, "unsupported_claims": [], "reason": "LLM输出无法解析为JSON，本层降级"}
        data = json.loads(m.group())
        supported = data.get("supported")
        if not isinstance(supported, bool):
            supported = None
        return {
            "supported": supported,
            "unsupported_claims": [str(c) for c in data.get("unsupported_claims", [])][:5],
            "reason": str(data.get("reason", ""))[:200],
        }
    except Exception as e:
        return {"supported": None, "unsupported_claims": [], "reason": f"LLM复核失败降级: {type(e).__name__}"}


async def verify_answer_async(query: str, answer: str, docs: list, run_llm_check: bool = True) -> dict:
    """verify_answer 的异步版。规则三层为纯CPU计算直接执行；LLM蕴含层 await 异步调用。"""
    docs_text = "\n".join(d.page_content for d in (docs or [])) + "\n" + (query or "")
    refusal = is_refusal(answer)

    citations = check_citations(answer, len(docs or []))
    unfaithful_numbers = [] if refusal else check_numbers(answer, docs_text)
    unknown_companies = [] if refusal else check_companies(answer, docs_text)

    llm_res = {"supported": None, "unsupported_claims": [], "reason": "拒答回答跳过LLM复核"}
    if run_llm_check and not refusal:
        llm_res = await llm_entailment_async(query, answer, docs)

    layers = []
    if citations["invalid_refs"]:
        layers.append("citation_invalid")
    if unfaithful_numbers:
        layers.append("number_unfaithful")
    if unknown_companies:
        layers.append("company_unknown")
    if llm_res["supported"] is False:
        layers.append("llm_not_supported")

    parts = []
    if citations["invalid_refs"]:
        parts.append(f"引用了不存在的资料编号{citations['invalid_refs']}")
    if unfaithful_numbers:
        parts.append(f"出现了资料中不存在的数字：{', '.join(unfaithful_numbers[:5])}")
    if unknown_companies:
        parts.append(f"提及了检索资料中没有的公司：{', '.join(unknown_companies[:5])}")
    if llm_res["supported"] is False:
        parts.append(f"LLM复核判定存在无依据论断：{llm_res['reason']}")

    return {
        "risk": bool(layers),
        "refusal": refusal,
        "citations": citations,
        "unfaithful_numbers": unfaithful_numbers,
        "unknown_companies": unknown_companies,
        "llm": llm_res,
        "layers_triggered": layers,
        "summary": "；".join(parts) if parts else "",
    }


async def verify_with_regenerate_async(query: str, answer: str, docs: list,
                                       conversation_history: str = "", max_regen: int = 1) -> tuple:
    """verify_with_regenerate 的异步版，采纳/兜底判定逻辑完全一致。"""
    from chains.qa_chain import generate_answer_async  # 惰性导入避免循环依赖

    res = await verify_answer_async(query, answer, docs)
    regen = 0
    while res["risk"] and regen < max_regen:
        regen += 1
        answer2 = await generate_answer_async(
            query, docs, conversation_history=conversation_history,
            strict=True, strict_reason=res["summary"],
        )
        res2 = await verify_answer_async(query, answer2, docs)
        if (not res2["risk"]) or len(res2["layers_triggered"]) <= len(res["layers_triggered"]):
            answer, res = answer2, res2
    res["regen_attempts"] = regen
    if res["risk"]:
        answer = HALLUCINATION_WARNING + (answer or "")
    return answer, res


if __name__ == "__main__":
    # 离线自测：不调LLM（run_llm_check=False），只验证规则层
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from langchain_core.documents import Document

    docs = [
        Document(page_content="贵州茅台三季报净利润同比增长15%，直销渠道占比提升至45%。", metadata={}),
        Document(page_content="招商银行不良率5.2%，拨备覆盖率430%。", metadata={}),
    ]
    a_ok = "茅台净利润同比增长15% [1]，招行不良率5.2% [2]。"
    a_bad = "茅台净利润同比增长18% [1]，洋河股份也发布了财报 [2][9]。"
    a_refuse = "现有资料有限，无法回答该问题。"
    a_hedge = "根据资料，茅台净利润同比增长15% [1]。现有资料有限，未提供更细分数据。"
    a_dates = "招行不良率5.2% [2]。来源：财新，2026-09-25，9月8日数据。"

    for name, ans in (("正常回答", a_ok), ("编造回答", a_bad), ("如实拒答", a_refuse),
                      ("带局限说明的回答", a_hedge), ("含日期的回答", a_dates)):
        r = verify_answer("测试问题", ans, docs, run_llm_check=False)
        print(f"{name}: risk={r['risk']} refusal={r['refusal']} layers={r['layers_triggered']} summary={r['summary']}")
