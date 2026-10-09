"""
舆情数据提交接口。复用 graphs/pipeline_graph.process()（同步，等结果）和
ingestion/async_jobs.submit_ingest_item()（异步，扔进后台队列立即返回任务ID）
——两条路径和 scripts/run_pipeline.py、ingestion/scheduler.py 走的是同一套处理逻辑
（清洗→情感分析→事件分类→预警判断→入库→触发预警推送）。
"""
from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_tenant_id, require_api_key, rate_limit
from api.schemas import IngestRequest, IngestSyncResponse, IngestAsyncResponse
from graphs.pipeline_graph import process_async
from ingestion.async_jobs import submit_ingest_item

router = APIRouter(prefix="/api/v1", tags=["数据提交"], dependencies=[Depends(require_api_key)])


@router.post(
    "/ingest",
    summary="提交一条舆情文本（可选同步等结果 / 异步立即返回任务ID）",
    response_model=None,
)
async def ingest(req: IngestRequest, tenant_id: str = Depends(get_tenant_id)):
    """
    同步模式（默认）：等完整跑完清洗→分析→入库→预警才返回，适合调用方需要立刻拿到
    分析结果的场景；耗时取决于LLM响应速度，量大时建议用异步模式。

    异步模式（async_mode=true）：立即返回task_id，实际处理在后台线程池队列完成，
    结果要通过 /api/v1/companies/{company}/records 或直接查库拿。

    需要完整的LLM/向量库依赖（ANTHROPIC_API_KEY等）配置好才能跑通完整链路，
    仅装了 requirements.txt 里的轻量依赖、没配LLM Key时这个接口会报错——
    这是预期行为，不是接口本身的bug。
    """
    rate_limit(tenant_id)

    if req.async_mode:
        # 后台队列模式：内部是线程池worker（ingestion/async_jobs.py），保持原样
        task_id = submit_ingest_item(
            {"raw_text": req.raw_text, "source": req.source, "date": req.date,
             "source_type": req.source_type},
            tenant_id=tenant_id,
        )
        return IngestAsyncResponse(task_id=task_id)

    try:
        # 同步等结果模式走异步图（LLM等待不占线程池worker）
        result = await process_async(
            raw_text=req.raw_text, source=req.source, date=req.date,
            tenant_id=tenant_id, source_type=req.source_type,
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"分析流程执行失败（请检查LLM API Key/向量库等依赖是否配置好）：{e}",
        )

    return IngestSyncResponse(
        company=result.get("company"),
        event_type=result.get("event_type"),
        sentiment_label=result.get("sentiment_label"),
        sentiment_score=result.get("sentiment_score"),
        dimension_scores=result.get("dimension_scores"),
        need_alert=result.get("need_alert"),
        db_id=result.get("db_id"),
    )
