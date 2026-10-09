"""
图谱检索桥接。

把知识图谱查询结果转成 RAG 问答流程能直接消费的 Document 列表：
- is_kg_query(query) 判定为图谱类问题（关系型/事实型）时，返回知识图谱事实
- 否则返回空列表，完全不干扰原有 RAG 检索

图谱事实以 [知识图谱] 前缀标识，作为额外的"资料"与 RAG 检索结果一起喂给LLM，
来源标记为"知识图谱"，方便溯源。
"""
from langchain_core.documents import Document

from knowledge_graph.graph_manager import kg_retrieve, is_kg_query


def kg_retrieve_docs(query: str) -> list:
    """图谱类问题 -> 知识图谱事实Document列表；否则返回[]（RAG不受影响）"""
    if not is_kg_query(query):
        return []
    facts = kg_retrieve(query)
    return [
        Document(
            page_content=f"[知识图谱] {f}",
            metadata={"source": "知识图谱", "date": ""},
        )
        for f in facts
    ]
