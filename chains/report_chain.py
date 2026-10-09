"""
日报小节生成。只负责"给定一个公司的统计数据+样本原文，生成一段小结文字"，
不负责数据从哪来（SQLite查询）、也不负责多公司怎么拼成完整日报（graphs/report_graph.py）。
"""
from chains.prompts.report_prompt import report_prompt
from chains.qa_chain import get_llm
from core.retry import with_retry, with_retry_async

_report_llm_chain = None


def get_report_chain():
    global _report_llm_chain
    if _report_llm_chain is None:
        _report_llm_chain = report_prompt | get_llm()
    return _report_llm_chain


@with_retry()
def _invoke_summary(stats_text: str, samples_text: str) -> str:
    """实际调用LLM生成单家公司小结。失败抛异常交给with_retry重试"""
    chain = get_report_chain()
    result = chain.invoke({"stats": stats_text, "samples": samples_text})
    return result.content if hasattr(result, "content") else str(result)


def summarize_company(company: str, stats: dict, samples: list) -> str:
    avg_score = round(stats.get("avg_score") or 0, 2)
    stats_text = (
        f"公司：{company}，舆情条数：{stats.get('cnt', 0)}，"
        f"平均情感分：{avg_score}，负面条数：{stats.get('negative_cnt', 0)}"
    )
    samples_text = "\n".join(f"- {s[:100]}" for s in samples[:3]) or "（无代表性原文）"

    try:
        return _invoke_summary(stats_text, samples_text)
    except Exception as e:
        # 单家公司小结失败不拖垮整份日报，降级为占位说明
        return f"（该公司舆情小结生成失败：{type(e).__name__}，可稍后手动重跑日报）"


# ================= 异步版 =================
# LCEL链对象同时支持invoke/ainvoke，直接复用get_report_chain()的同一个链实例；
# 重试判定复用is_retryable，退避用await asyncio.sleep。
# 日报的多家公司小结由 report_graph 的异步节点用 asyncio.gather 并发调用——
# 这是日报链路异步化最大的收益点（N家公司的小结LLM调用互相穿插等待）。

async def _invoke_summary_async(stats_text: str, samples_text: str) -> str:
    """异步版实际LLM小结。失败抛异常交给with_retry_async重试"""
    chain = get_report_chain()
    result = await chain.ainvoke({"stats": stats_text, "samples": samples_text})
    return result.content if hasattr(result, "content") else str(result)


async def summarize_company_async(company: str, stats: dict, samples: list) -> str:
    """summarize_company() 的异步版：单家失败同样降级为占位说明，不拖垮整份日报"""
    avg_score = round(stats.get("avg_score") or 0, 2)
    stats_text = (
        f"公司：{company}，舆情条数：{stats.get('cnt', 0)}，"
        f"平均情感分：{avg_score}，负面条数：{stats.get('negative_cnt', 0)}"
    )
    samples_text = "\n".join(f"- {s[:100]}" for s in samples[:3]) or "（无代表性原文）"

    try:
        return await _invoke_summary_async(stats_text, samples_text)
    except Exception as e:
        return f"（该公司舆情小结生成失败：{type(e).__name__}，可稍后手动重跑日报）"
