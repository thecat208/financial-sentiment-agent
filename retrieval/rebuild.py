"""
向量索引定期重建。

策略：从SQLite读全部记录 → 写入一个全新的Chroma集合（{base}_v{时间戳}）→
原子切换"活跃集合"指针 → 删除旧集合。切换指针是写一个小JSON文件，重建期间
旧集合持续对外服务，做到服务不中断（把 Milvus/Weaviate 的别名切换思想适配到
Chroma 的集合切换）。

调用方式：手动脚本 scripts/rebuild_vector_index.py、调度器每日凌晨任务、
或通过任务队列 submit_vector_rebuild() 异步执行。
"""
import json
import os
import time

from config.settings import settings
from storage.repository import get_all_records
from retrieval.vectorstore import get_embeddings, get_active_collection_name, reset_vectorstore


def rebuild_vector_index(tenant_id: str = None) -> dict:
    """从SQLite重建向量索引。返回重建结果统计。"""
    records = get_all_records(tenant_id)
    if not records:
        return {"rebuild": False, "reason": "没有可重建的舆情数据", "count": 0}

    texts = [r.get("cleaned_text") or r.get("raw_text") or "" for r in records]
    metadatas = [
        {
            "source": r.get("source") or "未知来源",
            "date": r.get("date") or "",
            "company": r.get("company") or "未知",
            "tenant_id": r.get("tenant_id") or "default",
        }
        for r in records
    ]

    from langchain_chroma import Chroma
    new_name = f"{settings.CHROMA_COLLECTION_NAME}_v{int(time.time())}"
    new_vs = Chroma(
        collection_name=new_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
    new_vs.add_texts(texts=texts, metadatas=metadatas)

    # 切换活跃集合指针（原子写文件，期间旧集合继续服务）
    old_name = get_active_collection_name()
    pointer = settings.CHROMA_ACTIVE_POINTER
    os.makedirs(os.path.dirname(pointer) or ".", exist_ok=True)
    with open(pointer, "w", encoding="utf-8") as f:
        json.dump({"active": new_name, "count": len(texts)}, f, ensure_ascii=False)
    reset_vectorstore()

    # 删除旧集合（失败不影响切换结果，留下待清理集合而已）
    try:
        if old_name and old_name != new_name:
            old_vs = Chroma(
                collection_name=old_name,
                embedding_function=get_embeddings(),
                persist_directory=settings.CHROMA_PERSIST_DIR,
            )
            old_vs.delete_collection()
    except Exception:
        pass

    return {"rebuild": True, "new_collection": new_name, "old_collection": old_name, "count": len(texts)}
