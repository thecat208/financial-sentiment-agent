"""
知识图谱统一入口。

- 后端选择：KG_PROVIDER=auto 时优先 Neo4j，连不上自动降级 NetworkX；也可强制指定
- 懒构建：首次查询时若图未构建，自动 build_graph()（种子知识库 + 舆情记录）
- 对外能力：
    is_kg_query(query)  -> 判断是否为图谱类问题（关系型/事实型）
    kg_retrieve(query)  -> 返回图谱事实列表（字符串），供混合路由并入QA上下文
    stats()             -> 图规模统计
"""
from config.settings import settings
from knowledge_graph.builder import build_graph_data
from knowledge_graph.seed_data import INDUSTRIES
from knowledge_graph.networkx_store import NetworkXStore
from knowledge_graph.neo4j_store import Neo4jStore

_store = None
_backend = None

# 图谱类问题的关键词（命中其一且能识别出实体，才判定为图谱查询）
KG_QUERY_KEYWORDS = [
    "属于", "行业", "关系", "联系", "关联", "之间", "供应链", "竞争", "供货",
    "上游", "下游", "有哪些", "是什么", "什么关系", "路径", "谁", "概念股",
    "同行", "供应商",
]


def get_store():
    """返回选定的图存储后端（懒加载）。"""
    global _store, _backend
    if _store is not None:
        return _store
    if settings.KG_PROVIDER in ("auto", "neo4j"):
        nb = Neo4jStore()
        if nb.is_available():
            _store, _backend = nb, "neo4j"
    if _store is None:
        _store, _backend = NetworkXStore(), "networkx"
    return _store


def backend() -> str:
    return _backend


def build_graph(force: bool = False, load_seed: bool = None) -> dict:
    """构建图（种子+舆情记录）。NetworkX 优先读持久化文件；Neo4j 即写即存。"""
    store = get_store()
    if not force and store.is_built():
        return {"built": False, "backend": backend(), "reason": "already_built"}

    if isinstance(store, NetworkXStore) and not force:
        if store.load_from_file():
            store.mark_built()
            return {"built": True, "backend": backend(), "loaded_from_file": True}

    if load_seed is None:
        load_seed = settings.KG_LOAD_SEED
    data = build_graph_data(load_seed=load_seed)
    store.load_nodes(data["nodes"])
    store.load_edges(data["edges"])
    store.save()
    store.mark_built()
    return {"built": True, "backend": backend(), "nodes": len(data["nodes"]), "edges": len(data["edges"])}


def _ensure_built():
    store = get_store()
    if not store.is_built():
        build_graph()
    return store


def is_kg_query(query: str) -> bool:
    """图谱类问题：
    1) 问题里同时提到两个已知实体 → 极可能是关系/比较类问题，直接判为图谱查询
    2) 提到一个实体 且 含图谱关键词（属于/行业/关系/有哪些…）→ 图谱查询
    否则返回 False，走纯RAG。
    """
    if not query:
        return False
    store = _ensure_built()
    entities = store.find_entities(query)
    if not entities:
        return False
    if len(entities) >= 2:
        return True
    return any(k in query for k in KG_QUERY_KEYWORDS)


def kg_retrieve(query: str, limit: int = 15) -> list:
    """
    图谱查询：返回给LLM的事实列表（字符串）。非图谱类问题返回空列表。
    - 单实体：返回其直接邻居关系（属于/发行/供货/竞争/共现…）；若是行业则返回成员公司
    - 双实体：返回两者间的关系 及 最短关系路径
    """
    if not query:
        return []
    store = _ensure_built()
    if not store.is_available():
        return []

    entities = store.find_entities(query)
    if not entities:
        # 兜底：行业清单里的词即使还没建节点，也按行业查询处理
        for ind in INDUSTRIES:
            if ind in query:
                entities = [("industry", ind)]
                break
    if not entities:
        return []

    facts = []
    if len(entities) >= 2:
        n1, n2 = entities[0][1], entities[1][1]
        for _rel, label, _props in store.get_relations_between(n1, n2):
            facts.append(f"{n1} -[{label}]-> {n2}")
        path = store.find_path(n1, n2)
        if path:
            facts.append(f"关系路径：{' → '.join(path)}")

    ntype, name = entities[0]
    for _rel, label, target, _props in store.get_neighbors(name):
        facts.append(f"{name} -[{label}]-> {target}")
        if len(facts) >= limit:
            break

    if ntype == "industry":
        members = store.get_industry_members(name)
        if members:
            facts.append(f"行业「{name}」成员公司：{'、'.join(members[:15])}")

    return list(dict.fromkeys(facts))[:limit]


def stats() -> dict:
    return _ensure_built().stats()
