"""
预警复核Chain。规则初筛（关键词/情感阈值）命中后，交给LLM二次判断是否真的值得推送，
避免"命中关键词就推送"造成误报刷屏。

按 (文本|情感标签|情感分) hash 缓存复核结果，同一文本重复命中预警时不再重复花LLM调用；
失败/解析失败的结果不缓存。
"""
from chains.prompts.alert_review_prompt import alert_review_prompt, ReviewResult
from chains.structured_output import invoke_structured, ainvoke_structured
from core.cache import cached, hash_part
from core.logger import get_logger
from core.retry import with_retry, with_retry_async
from config.settings import settings

logger = get_logger("chain.alert_review")


@with_retry()
@cached(
    "llm:review", settings.CACHE_TTL_LLM,
    key_parts_fn=lambda text, label, score: (hash_part(f"{text}|{label}|{score}"),),
)
def _review_uncached(text: str, sentiment_label: str, sentiment_score: float) -> dict:
    """实际LLM复核。失败/解析失败时抛异常，由装饰器决定不缓存本次结果"""
    prompt_value = alert_review_prompt.format(
        text=text, sentiment_label=sentiment_label, sentiment_score=sentiment_score
    )
    # 结构化输出的多provider兼容见 chains/structured_output.py
    response = invoke_structured(ReviewResult, prompt_value)

    result = response.dict() if hasattr(response, "dict") else dict(response)
    return result


def review_alert(text: str, sentiment_label: str, sentiment_score: float) -> dict:
    """命中缓存直接返回；LLM失败/解析失败时默认放行（不缓存兜底结果），宁可误报也不漏报"""
    try:
        return _review_uncached(text, sentiment_label, sentiment_score)
    except Exception as e:
        logger.warning("预警复核调用失败，默认放行（%s: %s）", type(e).__name__, e)
        return {"is_valid": True,
                "reason": f"复核解析失败（{type(e).__name__}: {e}），默认放行（本次结果未缓存）"}


# ================= 异步版 =================
# 语义与同步版完全一致（缓存key/重试判定/兜底"默认放行"全部相同），仅LLM调用走ainvoke。

@with_retry_async()
@cached(
    "llm:review", settings.CACHE_TTL_LLM,
    key_parts_fn=lambda text, label, score: (hash_part(f"{text}|{label}|{score}"),),
)
async def _review_uncached_async(text: str, sentiment_label: str, sentiment_score: float) -> dict:
    """异步版实际LLM复核。失败/解析失败时抛异常，由装饰器决定不缓存本次结果"""
    prompt_value = alert_review_prompt.format(
        text=text, sentiment_label=sentiment_label, sentiment_score=sentiment_score
    )
    response = await ainvoke_structured(ReviewResult, prompt_value)

    result = response.dict() if hasattr(response, "dict") else dict(response)
    return result


async def review_alert_async(text: str, sentiment_label: str, sentiment_score: float) -> dict:
    """review_alert() 的异步版：失败/解析失败同样默认放行（不缓存兜底结果）"""
    try:
        return await _review_uncached_async(text, sentiment_label, sentiment_score)
    except Exception as e:
        logger.warning("预警复核调用失败，默认放行（%s: %s）", type(e).__name__, e)
        return {"is_valid": True,
                "reason": f"复核解析失败（{type(e).__name__}: {e}），默认放行（本次结果未缓存）"}
