"""
LLM用量追踪回调。

挂在 chains/qa_chain.get_llm() 构造出的LLM客户端上（单例，全局只挂一次），
每次LLM调用结束时自动读取token消耗、结合 core/usage_context.py 当前设置的
租户/场景，写进 storage/usage_tracker.py。chains/ 下的5个业务chain文件
完全不用改代码。token数从 message.usage_metadata 取——LangChain统一的
跨provider字段，比解析 response_metadata 里provider专有字段名更稳定。
拿不到usage_metadata（某些不支持返回用量的模型/mock场景）就不记token数
但不影响流程——统计不到不代表调用失败，不应因拿不到token数让调用报错。
"""
from langchain_core.callbacks import BaseCallbackHandler
from core.usage_context import get_usage_context
from storage.usage_tracker import record_llm_usage
from config.settings import settings


class UsageTrackingCallback(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs) -> None:
        tenant_id, operation = get_usage_context()
        model = settings.OPENAI_MODEL if settings.LLM_PROVIDER == "openai" else settings.ANTHROPIC_MODEL

        try:
            for generation_list in response.generations:
                for gen in generation_list:
                    message = getattr(gen, "message", None)
                    usage = getattr(message, "usage_metadata", None) if message else None
                    if not usage:
                        continue
                    record_llm_usage(
                        tenant_id=tenant_id, operation=operation, model=model,
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                    )
        except Exception:
            # 用量统计本身出错不应该影响主流程——LLM调用已经成功返回了结果，
            # 记账这一步失败只是"这次没统计到"，不该往上抛异常打断业务逻辑。
            pass


# 全局单例，chains/qa_chain.get_llm() 构造LLM客户端时传入这一个实例即可
usage_callback = UsageTrackingCallback()
