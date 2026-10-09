"""
知识图谱构建器。

把"内置种子知识库 + SQLite舆情记录"组织成统一的 节点/边 列表，
交给选定的后端（Neo4j / NetworkX）落库。
关系构成：
- 公司 → 发行股票 / 属于行业（种子）
- 公司/人 → 竞争/供货/任职（种子）
- 公司 → 涉及 → 事件类型（来自舆情记录的 event_type）
- 公司 → 舆情共现 → 公司（同一条舆情正文里同时提到两个已知公司）
"""
from knowledge_graph import data_model as dm
from knowledge_graph.seed_data import COMPANIES, INDUSTRIES, PEOPLE, SEED_RELATIONS
from storage.repository import get_all_records


def _known_key(name):
    """把种子里的公司名/人名映射到节点key；未知名字直接按原样当key处理"""
    if name in COMPANIES:
        return dm.node_key("company", name)
    if name in PEOPLE:
        return dm.node_key("person", name)
    if name in INDUSTRIES:
        return dm.node_key("industry", name)
    if name.startswith(("company:", "person:", "industry:", "stock:", "event_type:")):
        return name
    return name


def build_graph_data(load_seed: bool = True) -> dict:
    """返回 {"nodes": [...], "edges": [...]}，节点/边用统一字典结构"""
    nodes, edges = [], []
    node_keys, edge_keys = set(), set()

    def add_node(key, ntype, label, aliases=None):
        if key not in node_keys:
            node_keys.add(key)
            nodes.append({"key": key, "type": ntype, "label": label, "aliases": aliases or []})

    def add_edge(src, rel, dst, props=None, extra=""):
        ek = (src, rel, dst, extra)
        if ek in edge_keys:
            return
        edge_keys.add(ek)
        edges.append({"src": src, "rel": rel, "dst": dst, "props": props or {}})

    if load_seed:
        for canonical, info in COMPANIES.items():
            ckey = dm.node_key("company", canonical)
            add_node(ckey, "company", canonical, info["aliases"])
            ikey = dm.node_key("industry", info["industry"])
            add_node(ikey, "industry", info["industry"])
            add_edge(ckey, "belongs_to", ikey)
            if info.get("stock"):
                skey = dm.node_key("stock", info["stock"])
                add_node(skey, "stock", info["stock"])
                add_edge(ckey, "issues", skey)
        for person, info in PEOPLE.items():
            add_node(dm.node_key("person", person), "person", person, info["aliases"])
        for src, rel, dst, props in SEED_RELATIONS:
            add_edge(_known_key(src), rel, _known_key(dst), props)

    # 从舆情记录自动补充：公司节点、公司-事件、公司-种子公司共现
    try:
        records = get_all_records()
    except Exception:
        records = []

    for r in records:
        company = r.get("company")
        if not company or company in ("未知", ""):
            continue
        ckey = dm.node_key("company", company)
        add_node(ckey, "company", company)

        event = r.get("event_type")
        if event and event not in ("解析失败", ""):
            ekey = dm.node_key("event_type", event)
            add_node(ekey, "event_type", event)
            add_edge(ckey, "involved_in", ekey)

        # 共现：该记录正文里同时提到种子公司 -> 建 company-共现-种子公司 边
        text = (r.get("raw_text") or "") + (r.get("cleaned_text") or "")
        for seed in COMPANIES:
            if seed != company and seed in text:
                add_edge(ckey, "co_occurs", dm.node_key("company", seed), {"count": 1})

    return {"nodes": nodes, "edges": edges}
