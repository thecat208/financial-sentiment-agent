"""
情感分析 + 实体抽取Chain。一次LLM调用同时完成两件事，对应pipeline_graph里的analyze_node。

- LLM调用按文本hash缓存（1小时）；失败/解析失败不缓存，避免把错误答案缓存成正确答案。
- 四个维度分数组装成 {"业绩","管理层","行业前景","合规风险"} 的dict返回，存库时序列化成JSON；
  event_type 经 normalize_event_type 兜底到受控词表（跨模型/provider时结构化输出严格度不一致）。
- _finalize_result() 是纯函数转换，单独拆出便于在无API Key环境下对解析逻辑做单元测试。
- analyze_text_gated() 是生产管线入口：先走轻量模型前置筛选（chains/lightweight_sentiment.py），
  置信度够高、识别出已知公司、不含风险关键词时直接用轻量结果，否则走 analyze_text() 完整LLM路径。
- 多provider适配：结构化输出统一走 chains/structured_output.py，官方OpenAI端点用 json_schema，
  仅支持 response_format=json_object 的网关（DeepSeek等）自动降级 json_mode；兜底结果带真实异常便于排查。
"""
from chains.prompts.analysis_prompt import analysis_prompt, AnalysisResult
from chains.structured_output import invoke_structured, ainvoke_structured
from chains.lightweight_sentiment import classify as lightweight_classify
from core.cache import cached, hash_part
from core.logger import get_logger
from core.retry import with_retry, with_retry_async
from core.taxonomy import (
    DIMENSIONS, normalize_event_type, clip_dimension_score, default_dimension_scores,
    guess_event_type_by_keywords,
)
from config.settings import settings
from knowledge_graph.seed_data import find_known_company_in_text
from alert.rules import contains_risk_keyword

logger = get_logger("chain.analysis")

# 兜底结果的reason前缀。scripts/evaluate_phrasebank.py 靠它判断"这条是不是失败兜底"，
# 所以要改措辞的话，评测脚本里的判断要一起改，否则兜底会被当成正常预测计入准确率。
FALLBACK_REASON_PREFIX = "LLM调用/解析失败"


def _finalize_result(raw: dict) -> dict:
    """
    把AnalysisResult.dict()的扁平字段，组装成对外统一的返回结构：
    - dimensions: {"业绩": x, "管理层": x, "行业前景": x, "合规风险": x}
    - event_type: 归一化到受控词表内
    - sentiment_score: 转float，防止上游传字符串
    """
    dimensions = {
        "业绩": clip_dimension_score(raw.get("score_performance", 0.0)),
        "管理层": clip_dimension_score(raw.get("score_management", 0.0)),
        "行业前景": clip_dimension_score(raw.get("score_industry", 0.0)),
        "合规风险": clip_dimension_score(raw.get("score_compliance", 0.0)),
    }
    return {
        "sentiment_label": raw.get("sentiment_label", "中性"),
        "sentiment_score": float(raw.get("sentiment_score", 0.0)),
        "dimensions": dimensions,
        "company": raw.get("company") or "未知",
        "event_type": normalize_event_type(raw.get("event_type")),
        "reason": raw.get("reason", ""),
    }


@with_retry()
@cached("llm:analysis", settings.CACHE_TTL_LLM, key_parts_fn=lambda text: (hash_part(text),))
def _analyze_uncached(text: str) -> dict:
    """实际LLM调用。失败/解析失败时抛异常，由装饰器决定不缓存本次结果"""
    prompt_value = analysis_prompt.format(text=text)
    # 结构化输出的多provider兼容（json_schema -> json_mode 自动降级）见 chains/structured_output.py
    response = invoke_structured(AnalysisResult, prompt_value)

    raw = response.dict() if hasattr(response, "dict") else dict(response)
    return _finalize_result(raw)


def analyze_text(text: str) -> dict:
    """
    返回字典：sentiment_label, sentiment_score, dimensions, company, event_type, reason。
    dimensions 是 {"业绩":分数,"管理层":分数,"行业前景":分数,"合规风险":分数} 的多维度打分。
    命中缓存直接返回；LLM失败/解析失败时返回保守的兜底结果（兜底结果不缓存），避免流程中断。

    注意：这是纯LLM路径，不经过轻量模型前置筛选；生产管线（graphs/pipeline_graph.py）
    调用的是 analyze_text_gated()，本函数保留给"需要LLM完整能力"的场景
    （如评测脚本要测LLM本身的准确率，不应被轻量模型分流影响）。
    """
    try:
        return _analyze_uncached(text)
    except Exception as e:
        # 兜底是"不让主流程中断"，但真因必须落在日志和reason里，
        # 否则跑批时只会看到一堆中性，完全看不出是Key失效、超时还是格式不支持
        logger.warning("情感分析调用失败，返回保守兜底（%s: %s）", type(e).__name__, e)
        return {
            "sentiment_label": "中性",
            "sentiment_score": 0.0,
            "dimensions": default_dimension_scores(),
            "company": "未知",
            "event_type": "其他",
            "reason": f"{FALLBACK_REASON_PREFIX}，已返回保守兜底"
                      f"（{type(e).__name__}: {e}）（本次结果未缓存）",
        }


def _try_lightweight_fastpath(text: str) -> dict | None:
    """
    尝试用轻量模型独立处理这条文本，返回结果字典（附加source=lightweight用于统计）；
    满足不了"能独立处理"的条件时返回None，调用方应该转去走完整LLM路径。

    能独立处理的三个必要条件（任一不满足就升级LLM，宁可多调LLM也不要用不靠谱的结果）：
      1. 轻量模型的情感判断置信度达到阈值（不是模糊/边界case）
      2. 文本里能识别出种子库已登记的已知公司（轻量模型没有通用NER能力）
      3. 文本不包含预警风险关键词（风险类文本的判断后果重，宁可保守一点都走LLM，
         这和 alert/rules.py 的"规则初筛，宁可错杀不放过"是同一个态度）
    """
    if contains_risk_keyword(text):
        return None

    company = find_known_company_in_text(text)
    if not company:
        return None

    sentiment = lightweight_classify(text)
    if sentiment["confidence"] < settings.LIGHTWEIGHT_CONFIDENCE_THRESHOLD:
        return None

    return {
        "sentiment_label": sentiment["label"],
        "sentiment_score": sentiment["score"],
        # 快速通道没有多维度拆解能力，这是设计上的已知取舍（见模块docstring），
        # 不用整体分数去伪造四个维度，全零更诚实——展示层看到全零维度会视为"未细分"。
        "dimensions": default_dimension_scores(),
        "company": company,
        "event_type": guess_event_type_by_keywords(text),
        "reason": f"轻量模型前置筛选独立处理（backend={sentiment['backend']}，"
                  f"置信度={sentiment['confidence']}，命中词={sentiment.get('hits') or '无'}）",
        "source": "lightweight",
    }


def analyze_text_gated(text: str) -> dict:
    """
    轻量模型前置筛选 + 大模型兜底。生产管线（graphs/pipeline_graph.py的
    analyze_node）调用这个函数，而不是直接调 analyze_text()。

    settings.ENABLE_LIGHTWEIGHT_PREFILTER=False 时等价于直接调 analyze_text()
    （用于关闭前置筛选做AB对比，衡量"轻量模型到底省了多少LLM调用"）。

    返回结构和 analyze_text() 完全一致，多一个 source 字段（"lightweight"或"llm"），
    调用方不关心这个字段也不影响使用；scripts/evaluate_lightweight_prefilter.py
    用这个字段统计"轻量模型独立处理比例"。
    """
    if not settings.ENABLE_LIGHTWEIGHT_PREFILTER:
        result = analyze_text(text)
        result["source"] = "llm"
        return result

    fastpath = _try_lightweight_fastpath(text)
    if fastpath is not None:
        return fastpath

    result = analyze_text(text)
    result["source"] = "llm"
    return result


# ================= 异步版 =================
# 语义与同步版完全一致：重试判定/缓存key/兜底结构/返回字段全部相同；
# 区别仅在LLM调用走 ainvoke_structured（等待期间不阻塞事件循环）、
# 重试退避用 await asyncio.sleep。同步版保留原样，生产管线/评测脚本零变化。

@with_retry_async()
@cached("llm:analysis", settings.CACHE_TTL_LLM, key_parts_fn=lambda text: (hash_part(text),))
async def _analyze_uncached_async(text: str) -> dict:
    """异步版实际LLM调用。失败/解析失败时抛异常，由装饰器决定不缓存本次结果"""
    prompt_value = analysis_prompt.format(text=text)
    response = await ainvoke_structured(AnalysisResult, prompt_value)

    raw = response.dict() if hasattr(response, "dict") else dict(response)
    return _finalize_result(raw)


async def analyze_text_async(text: str) -> dict:
    """analyze_text() 的异步版：命中缓存直接返回；失败/解析失败返回同样的保守兜底（不缓存）"""
    try:
        return await _analyze_uncached_async(text)
    except Exception as e:
        logger.warning("情感分析调用失败，返回保守兜底（%s: %s）", type(e).__name__, e)
        return {
            "sentiment_label": "中性",
            "sentiment_score": 0.0,
            "dimensions": default_dimension_scores(),
            "company": "未知",
            "event_type": "其他",
            "reason": f"{FALLBACK_REASON_PREFIX}，已返回保守兜底"
                      f"（{type(e).__name__}: {e}）（本次结果未缓存）",
        }


async def analyze_text_gated_async(text: str) -> dict:
    """analyze_text_gated() 的异步版。

    轻量模型前置筛选是本地CPU推理（lexicon词典/transformer小模型，无LLM API），
    CPU密集会占住事件循环，用 asyncio.to_thread 丢进线程池执行。
    """
    import asyncio

    if not settings.ENABLE_LIGHTWEIGHT_PREFILTER:
        result = await analyze_text_async(text)
        result["source"] = "llm"
        return result

    fastpath = await asyncio.to_thread(_try_lightweight_fastpath, text)
    if fastpath is not None:
        return fastpath

    result = await analyze_text_async(text)
    result["source"] = "llm"
    return result
