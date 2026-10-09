"""
手动重建向量索引。从SQLite全部记录重建Chroma索引，
写入新集合并原子切换活跃指针，旧集合自动清理。运行方式：python scripts/rebuild_vector_index.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.rebuild import rebuild_vector_index

if __name__ == "__main__":
    print("开始重建向量索引...（记录较多时首次Embedding可能较慢）")
    result = rebuild_vector_index()
    if result.get("rebuild"):
        print(f"重建完成：新集合={result['new_collection']}，共 {result['count']} 条记录，"
              f"旧集合 {result['old_collection']} 已清理。")
    else:
        print(f"未重建：{result.get('reason')}")
