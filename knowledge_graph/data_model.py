"""
知识图谱数据模型定义

定义节点类型与关系类型（含中文标签），Neo4j 与 NetworkX 两套后端共用这套定义，
保证两边查询结果的中文展示一致。
"""

# ---- 节点类型 ----
NODE_TYPES = {"company", "industry", "stock", "person", "event_type"}

# ---- 关系类型 -> 中文标签（用于查询结果展示）----
RELATION_LABELS = {
    "belongs_to": "属于行业",
    "issues": "发行股票",
    "works_at": "任职于",
    "supplies": "供货给",
    "competes_with": "与…竞争",
    "invests_in": "投资/持股",
    "co_occurs": "舆情共现",
    "involved_in": "涉及事件",
    "related": "相关",
}


def node_key(ntype: str, name: str) -> str:
    """节点在存储层的唯一key，格式为 type:name"""
    return f"{ntype}:{name}"
