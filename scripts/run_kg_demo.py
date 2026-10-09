"""
知识图谱演示（知识图谱与RAG工程）。
运行方式：python scripts/run_kg_demo.py

展示：图构建规模、当前后端（Neo4j/NetworkX）、实体识别、
典型图谱查询（行业成员 / 两实体关系 / 关系路径）、以及"图谱类问题"判定（混合路由）。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_graph.graph_manager import build_graph, stats, kg_retrieve, is_kg_query, backend

if __name__ == "__main__":
    print("构建知识图谱...")
    info = build_graph()
    print(f"构建结果：{info}")
    print(f"当前后端：{backend()}，图规模：{stats()}\n")

    queries = [
        "宁德时代属于哪个行业？",
        "有哪些券商？",
        "宁德时代和比亚迪是什么关系？",
        "宁德时代和特斯拉之间有什么联系？",
        "示例公司最近的舆情怎么样？",  # 非图谱问题，应返回空走RAG
    ]
    for q in queries:
        print(f"Q: {q}")
        print(f"   是否图谱类问题：{is_kg_query(q)}")
        facts = kg_retrieve(q)
        if facts:
            for f in facts:
                print(f"   - {f}")
        else:
            print("   （无图谱事实，正常走RAG检索）")
        print()
