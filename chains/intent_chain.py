"""
意图识别Chain。只负责"给定用户输入，判断意图类型+提取公司/日期"，
不负责该意图具体怎么处理（那是 graphs/router_graph.py 的职责）。

相同问句按内容hash缓存意图结果，不重复调LLM。解析失败或LLM返回异常值时，
兜底为"qa"——走最通用的问答路径，不让意图识别异常阻塞整个流程；兜底结果不缓存。
"""
from chains.prompts.intent_prompt import intent_prompt, IntentResult
from chains.structured_output import invoke_structured
from core.cache import cached, hash_part
from core.logger import get_logger
from core.retry import with_retry, with_retry_async
from config.settings import settings

VALID_INTENTS = {"qa", "report", "alert_stats", "other"}

logger = get_logger("chain.intent")


@with_retry()
@cached("llm:intent", settings.CACHE_TTL_LLM, key_parts_fn=lambda query: (hash_part(query),))
def _classify_uncached(query: str) -> dict:
    """实际LLM意图识别。返回非法意图值视为失败（抛异常），避免缓存异常结果"""
    # 结构化输出的多provider兼容见 chains/structured_output.py
    output = invoke_structured(IntentResult, intent_prompt.format(query=query))

    result_dict = output.dict() if hasattr(output, "dict") else dict(output)
    if result_dict.get("intent") not in VALID_INTENTS:
        raise ValueError(f"意图识别返回非法值：{result_dict.get('intent')}")
    return result_dict


def classify_intent(query: str) -> dict:
    """
    返回 {"intent": "qa"/"report"/"alert_stats"/"other", "company": str, "date": str}

    命中缓存直接返回；解析失败或LLM返回异常值时，兜底为"qa"——
    走最通用的问答路径，不让意图识别本身的异常阻塞整个流程
    （对应清单里"超时/异常自动降级"的思路，这里降级不是重试，而是退到风险最低的默认分支）。
    """
    try:
        return _classify_uncached(query)
    except Exception as e:
        # 兜底不中断流程，但真因要留痕，否则意图识别失效时外部只看到"什么都是qa"
        logger.warning("意图识别调用失败，兜底为qa（%s: %s）", type(e).__name__, e)
        return {"intent": "qa", "company": "", "date": ""}


# ================= 异步版 =================
# 判定/缓存/重试/兜底逻辑与同步版完全一致（缓存key相同，两条路径命中同一份缓存）；
# 区别仅在LLM调用走 ainvoke_structured，重试退避不阻塞事件循环。

@with_retry_async()
@cached("llm:intent", settings.CACHE_TTL_LLM, key_parts_fn=lambda query: (hash_part(query),))
async def _classify_uncached_async(query: str) -> dict:
    """实际LLM意图识别（异步版）。返回非法意图值视为失败（抛异常），避免缓存异常结果"""
    from chains.structured_output import ainvoke_structured
    output = await ainvoke_structured(IntentResult, intent_prompt.format(query=query))

    result_dict = output.dict() if hasattr(output, "dict") else dict(output)
    if result_dict.get("intent") not in VALID_INTENTS:
        raise ValueError(f"意图识别返回非法值：{result_dict.get('intent')}")
    return result_dict


async def classify_intent_async(query: str) -> dict:
    """classify_intent 的异步版，降级语义一致：任何异常兜底为"qa"。"""
    try:
        return await _classify_uncached_async(query)
    except Exception as e:
        logger.warning("意图识别调用失败，兜底为qa（%s: %s）", type(e).__name__, e)
        return {"intent": "qa", "company": "", "date": ""}
