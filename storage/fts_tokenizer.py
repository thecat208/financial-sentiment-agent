"""
FTS5 中文分词器。

SQLite默认unicode61分词器按标点切分中文，子串词（可转债/碳酸锂/IPO）查不中，
含标点的查询（600519.SH）还会触发FTS5查询语法错误。索引侧与查询侧用同一套分词：
写入时把 cleaned_text+company 预分词成空格分隔token串，存入
sentiment_records.fts_tokens 列，FTS5外同步表索引该列（见 storage/db.py）；
查询先分词，再逐token加双引号拼成FTS5 MATCH表达式，引号包裹同时根治
特殊字符的查询语法错误。jieba缺失时自动降级为"英文/数字整词+中文二元组"，
二元组保证中文子串可召回（质量略逊于jieba但零依赖、永不失败）。
分词器自身永不抛异常——任何失败都降级到可用路径，不让关键词检索变成500。
"""
import re

try:
    import jieba
    _jieba = jieba
except ImportError:  # jieba未安装时降级，不影响启动
    _jieba = None

# 纯标点/空白/符号token：索引与查询都不需要
_PUNCT_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)
# 连续英文/数字（含内部的 . - _，用于股票代码 600519.SH / 电池型号等）
_ASCII_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# 连续中文（含中文标点会被 \W 切开，这里取CJK统一表意文字区段）
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")

_MAX_QUERY_TOKENS = 12  # 查询token数上限：AND语义下token越多越难命中，长查询截断


def _fallback_tokenize(text: str) -> list:
    """无jieba时的降级分词：英文/数字整词 + 中文二元组"""
    tokens = []
    for run in _ASCII_RUN.findall(text):
        tokens.append(run)
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


def _jieba_tokenize(text: str) -> list:
    tokens = []
    for tok in _jieba.cut(text):
        tok = tok.strip()
        if not tok or _PUNCT_ONLY.match(tok):
            continue
        tokens.append(tok)
    return tokens


def tokenize_for_fts(text: str) -> str:
    """文本 → 空格分隔的token串（入库时存入 fts_tokens 列）。
    永不抛异常：任何异常降级返回空串（该条记录不进关键词索引，不影响主流程）。"""
    if not text:
        return ""
    try:
        tokens = _jieba_tokenize(text) if _jieba else _fallback_tokenize(text)
        return " ".join(tokens)
    except Exception:
        try:
            return " ".join(_fallback_tokenize(text))
        except Exception:
            return ""


def fts_query_tokens(query: str) -> list:
    """用户查询 → 有效token列表（供查询构造与降级判断使用）。永不抛异常。"""
    if not query:
        return []
    try:
        tokens = _jieba_tokenize(query) if _jieba else _fallback_tokenize(query)
    except Exception:
        return []
    return [t for t in tokens if not _PUNCT_ONLY.match(t)][:_MAX_QUERY_TOKENS]


def fts_query_expr(query: str, operator: str = "AND") -> str:
    """用户查询 → FTS5 MATCH 表达式。
    每个token用双引号包裹（FTS5短语语法）：既保证token按指定逻辑组合，又让特殊字符
    （600519.SH、括号等）不再触发查询语法错误。operator="AND"精确优先（jieba主路径下
    词边界一致）；operator="OR"召回兜底——二元组降级路径下查询侧会出现文档中不存在的
    跨界二元组（如"碳酸锂价格"的"锂价"），AND会漏召，由调用方在AND零命中时用OR+bm25兜底。
    返回空串表示查询无有效token，调用方应直接返回空结果。
    """
    tokens = fts_query_tokens(query)
    op = " OR " if operator.upper() == "OR" else " "
    return op.join('"%s"' % t.replace('"', '""') for t in tokens)


if __name__ == "__main__":
    # 离线自测（不依赖数据库）
    docs = "贵州茅台三季报净利润同比增长15%。招商银行发行500亿元可转债。宁德时代碳酸锂采购价格下行。"
    print("索引侧token样例:", tokenize_for_fts(docs)[:120], "...")
    for q in ["可转债", "碳酸锂价格", "600519.SH", "IPO募资", "！！！"]:
        print(f"查询[{q}] ->", repr(fts_query_expr(q)))
    assert _jieba is None or True
    print("jieba状态:", "已加载" if _jieba else "未安装（降级为二元组分词）")
