"""
智能入口 / RAG 问答接口。直接复用 graphs/router_graph 和 graphs/qa_graph 的异步入口，
和 Streamlit「🧭 智能入口」「RAG 问答」标签页调用的是同一份代码，行为完全一致。

两个路由为 async def（handle_query_async / ask_async）：LLM 等待期间事件循环
可以处理其他并发请求，不再像同步版那样一个请求占死一个线程池线程干等。
语义与同步版完全一致。
"""
from fastapi import APIRouter, Depends

from api.deps import get_tenant_id, require_api_key, rate_limit
from api.schemas import QueryRequest, QueryResponse, AskRequest
from graphs.router_graph import handle_query_async
from graphs.qa_graph import ask_async

router = APIRouter(prefix="/api/v1", tags=["问答"], dependencies=[Depends(require_api_key)])


@router.post("/query", response_model=QueryResponse, summary="智能入口：一句话自动路由到问答/日报/预警统计")
async def query(req: QueryRequest, tenant_id: str = Depends(get_tenant_id)):
    """
    对应 Streamlit「🧭 智能入口」标签页。不需要自己判断该走哪个功能，
    系统会先做意图识别，再自动路由到问答/日报/预警统计里最合适的一个。
    """
    rate_limit(tenant_id)
    result = await handle_query_async(req.query, tenant_id=tenant_id, session_id=req.session_id)
    return QueryResponse(**result)


@router.post("/qa", response_model=QueryResponse, summary="RAG问答：直接走检索问答流程，不做意图路由")
async def qa(req: AskRequest, tenant_id: str = Depends(get_tenant_id)):
    """对应 Streamlit「RAG 问答」标签页。适合明确知道自己要问答、想跳过意图识别这一步的调用方。"""
    rate_limit(tenant_id)
    result = await ask_async(
        req.query, company=req.company, tenant_id=tenant_id,
        session_id=req.session_id, rate_limit=False,  # 已经在上面统一限流过，避免重复计次
    )
    return QueryResponse(intent="qa", text=result.get("answer"), payload=result)
