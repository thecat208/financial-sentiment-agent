"""
向量库封装。负责 Embedding 模型的选择、Chroma 的初始化，以及写入接口。
用单例模式避免每次调用都重新加载模型（本地Embedding模型加载较慢）。

支持"活跃集合切换"：重建时写入全新的Chroma集合，再原子切换活跃集合指针
（见 retrieval/rebuild.py），重建期间旧集合持续服务，服务不中断。
"""
import json
import os
import threading

from config.settings import settings

_embeddings = None
_vectorstore = None
_vectorstore_name = None

# 初始化锁：bench/生产API并发首轮会有多个线程同时首次构建客户端，
# Chroma PersistentClient并发初始化同一目录存在竞态（sqlite系统表初始化
# 冲突），部分线程会被包装成"Could not connect to tenant default_tenant"。
# 用RLock是因为get_vectorstore持锁后会调用get_embeddings再取同一把锁。
_init_lock = threading.RLock()


def get_active_collection_name() -> str:
    """当前生效的Chroma集合名。重建切换指针后，这里读到的是新集合名"""
    pointer = settings.CHROMA_ACTIVE_POINTER
    try:
        with open(pointer, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("active"):
            return data["active"]
    except Exception:
        pass
    return settings.CHROMA_COLLECTION_NAME


def reset_vectorstore():
    """清空向量库客户端缓存（重建切换集合后调用，让get_vectorstore重新按指针建客户端）"""
    global _vectorstore, _vectorstore_name
    _vectorstore = None
    _vectorstore_name = None


def get_embeddings():
    """按配置返回本地或云端Embedding模型，全局只加载一次"""
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    with _init_lock:
        if _embeddings is not None:
            return _embeddings

        if settings.EMBEDDING_PROVIDER == "openai":
            from langchain_openai import OpenAIEmbeddings
            _embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            _embeddings = HuggingFaceEmbeddings(model_name=settings.LOCAL_EMBEDDING_MODEL)

    return _embeddings


def get_vectorstore():
    """返回Chroma向量库实例。活跃集合名变化时自动重建客户端（配合向量重建的集合切换）"""
    global _vectorstore, _vectorstore_name
    active = get_active_collection_name()
    if _vectorstore is not None and active == _vectorstore_name:
        return _vectorstore

    with _init_lock:
        # 双检：等锁期间可能已有线程建好客户端，避免重复构建/再次竞态
        if _vectorstore is not None and active == _vectorstore_name:
            return _vectorstore

        from langchain_chroma import Chroma
        _vectorstore = Chroma(
            collection_name=active,
            embedding_function=get_embeddings(),
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )
        _vectorstore_name = active

    return _vectorstore


def add_documents(texts: list[str], metadatas: list[dict] = None, ids: list[str] = None):
    """
    写入文本到向量库。

    metadatas 建议包含：
      - company: 公司名称，用于精确过滤
      - source: 数据来源（如"新浪财经"）
      - date: 发布日期，格式 YYYY-MM-DD，用于时间范围过滤
    """
    vs = get_vectorstore()
    return vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)


if __name__ == "__main__":
    # 简单自测：写入一条测试数据并检索
    add_documents(
        texts=["示例公司发布三季度财报，营收同比增长15%，市场反应积极。"],
        metadatas=[{"company": "示例公司", "source": "测试", "date": "2026-08-01"}],
    )
    vs = get_vectorstore()
    results = vs.similarity_search("示例公司业绩怎么样", k=1)
    for r in results:
        print(r.page_content, r.metadata)
