"""
用量追踪上下文。

问题：LLM客户端（chains/qa_chain.get_llm()）是全局共享单例，5个chain（analyze/qa/
report/alert_review/intent）复用同一个客户端；但"这次调用是哪个租户、什么场景"只有
调用方（graphs/*.py）知道，若把tenant_id当参数一路传进chain内部要改动5个文件，
还会连带影响 core/cache.py 的缓存key（缓存目前跨租户共享，不去动它）。
方案：用 contextvars——graphs/*.py 调用chain前 set_usage_context()，回调
（core/llm_usage_callback.py）在LLM调用结束时 get_usage_context() 记账，中间5个
chain文件完全不用改。本项目LangGraph都是线性同步执行，不涉及跨线程/跨协程调用，
contextvars 设置一次即可在同步调用链里正确传递。
"""
import contextvars

_current_tenant = contextvars.ContextVar("usage_tenant_id", default="default")
_current_operation = contextvars.ContextVar("usage_operation", default="unknown")


def set_usage_context(tenant_id: str, operation: str) -> None:
    _current_tenant.set(tenant_id or "default")
    _current_operation.set(operation or "unknown")


def get_usage_context() -> tuple:
    return _current_tenant.get(), _current_operation.get()
