"""
检索层。提供两种检索能力，以及把两者合并的混合检索：

1. retrieve()：向量语义检索，能抓住"说法不同但意思相近"的内容（如"业绩暴雷"和"营收大幅下滑"）
2. keyword_retrieve()：SQLite FTS5关键词检索，精确命中公司名/股票代码/专业术语
   （分词与索引侧共用 storage/fts_tokenizer.py，jieba优先/二元组降级）
3. hybrid_retrieve()：合并去重上面两者的结果，带TTL缓存
检索结果不足时"要不要放宽条件重试"属于流程控制，放在 graphs/qa_graph.py
用LangGraph编排，这里只负责"单次怎么查"。所有检索函数带 tenant_id 参数
（默认"default"）：向量检索通过metadata过滤隔离，关键词检索透传给search_text。
"""
import re
from datetime import datetime, timedelta
from langchain_core.documents import Document
from retrieval.vectorstore import get_vectorstore
from storage.repository import search_text
from config.settings import settings
from core.cache import get_cache, make_key, hash_part


def _docs_to_json(docs: list) -> list:
    """Document列表序列化成可缓存结构"""
    return [
        {"page_content": d.page_content, "metadata": dict(d.metadata or {})}
        for d in docs
    ]


def _docs_from_json(rows: list) -> list:
    return [Document(page_content=r["page_content"], metadata=r["metadata"]) for r in rows]


def _build_where_filter(tenant_id: str, company: str = None) -> dict:
    """
    构造Chroma兼容的where过滤条件。

    Chroma where语法限制：顶层字典只能含一个条件或一个逻辑操作符，多个条件必须用
    $and（或$or）包裹成列表，平铺多键会直接报错。日期不在这里过滤——chromadb 1.x 的
    $gt/$gte/$lt/$lte 只接受 int/float 操作数，ISO日期字符串无法在向量库内做范围比较，
    时间窗过滤由 retrieve() 取回候选后在Python侧完成。
    """
    conditions: list[dict] = [{"tenant_id": tenant_id}]
    if company:
        conditions.append({"company": company})

    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _in_date_window(meta_date, cutoff: str) -> bool:
    """metadata日期是否落在 [cutoff, +∞) 窗口内。无效/缺失日期一律视为窗外。"""
    s = str(meta_date or "")[:10]
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", s)) and s >= cutoff


def retrieve(query: str, company: str = None, days: int = None, k: int = None, tenant_id: str = "default"):
    """向量语义检索"""
    vs = get_vectorstore()
    k = k or settings.RETRIEVE_TOP_K
    where = _build_where_filter(tenant_id, company)

    if days:
        # 时间窗过滤：chromadb 1.x范围操作符不支持字符串日期，改为
        # 超量取回候选（3倍且不少于24条）后在Python侧按日期过滤再截断。
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        pool = max(k * 3, 24)
        docs = vs.similarity_search(query, k=pool, filter=where)
        docs = [d for d in docs if _in_date_window((d.metadata or {}).get("date"), cutoff)]
        return docs[:k]

    return vs.similarity_search(query, k=k, filter=where)


def keyword_retrieve(query: str, company: str = None, k: int = None, tenant_id: str = "default") -> list:
    """SQLite FTS5关键词检索，转换成与向量检索一致的Document格式，方便下游统一处理"""
    k = k or settings.RETRIEVE_TOP_K
    try:
        rows = search_text(query, company=company, limit=k, tenant_id=tenant_id)
    except Exception:
        # FTS5对某些特殊字符的查询语法敏感（如引号、星号），查询失败时降级为空结果，
        # 不让关键词检索的问题影响整体流程（向量检索仍然可用）
        return []

    docs = []
    for r in rows:
        docs.append(Document(
            page_content=r["cleaned_text"],
            metadata={
                "source": r.get("source") or "未知来源",
                "date": r.get("date") or "未知日期",
                "company": r.get("company"),
            },
        ))
    return docs


def _dedup(docs: list) -> list:
    """按正文前50字去重，避免向量检索和关键词检索命中同一条内容时重复展示"""
    seen = set()
    result = []
    for d in docs:
        key = d.page_content[:50]
        if key not in seen:
            seen.add(key)
            result.append(d)
    return result


def hybrid_retrieve(
    query: str, company: str = None, days: int = None, k: int = None, tenant_id: str = "default"
) -> list:
    """
    混合检索：向量检索负责"找得准"（语义相关），关键词检索负责"找得对"
    （精确命中公司名/专业术语），两者合并去重后返回。

    按 (租户+公司+时间窗+query) 缓存检索结果，TTL内相同查询直接返回缓存，
    减少对向量库/FTS的重复查询。缓存key带tenant_id，保证多租户之间检索结果不串。
    """
    k = k or settings.RETRIEVE_TOP_K
    cache = get_cache()
    key = make_key(
        "retrieve", tenant_id,
        company or "*",
        days if days is not None else "*",
        k,
        hash_part(query),
    )
    hit = cache.get(key)
    if hit is not None:
        return _docs_from_json(hit)

    vector_docs = retrieve(query, company=company, days=days, k=k, tenant_id=tenant_id)
    keyword_docs = keyword_retrieve(query, company=company, k=k, tenant_id=tenant_id)
    merged = _dedup(vector_docs + keyword_docs)

    cache.set(key, _docs_to_json(merged), settings.CACHE_TTL_RETRIEVE)
    return merged
