"""
舆情数据处理主流程：ingest_node → analyze_node（情感分析+实体抽取，一次LLM调用）
→ embed_node → save_node（落库）→ alert_check_node（条件路由：规则初筛）；
触发时走 alert_review_node（LLM复核，避免误报）→ 属实则 push_node 推送，误报则 END。

注意：analyze_node 必须在 embed_node 之前——只有分析出 company 之后，
写入向量库的 metadata 才能带上 company 字段，否则 RAG问答里"按公司过滤检索"会失效。
"""
from datetime import datetime
from langgraph.graph import StateGraph, END
from graphs.state_schemas import PipelineState
from processing.cleaner import clean_text
from retrieval.vectorstore import add_documents
from chains.analysis_chain import analyze_text_gated
from chains.alert_review_chain import review_alert
from alert.rules import rule_triggered
from alert.push import push_alert
from storage.repository import insert_record, update_alert_review, mark_pushed
from core.taxonomy import normalize_source_type
from core.usage_context import set_usage_context


def ingest_node(state: PipelineState) -> PipelineState:
    state["cleaned_text"] = clean_text(state["raw_text"])
    return state


def analyze_node(state: PipelineState) -> PipelineState:
    set_usage_context(state.get("tenant_id", "default"), "analyze")  # 用量归因
    result = analyze_text_gated(state["cleaned_text"])
    state["sentiment_label"] = result["sentiment_label"]
    state["sentiment_score"] = result["sentiment_score"]
    state["dimension_scores"] = result["dimensions"]
    state["company"] = result["company"]
    state["event_type"] = result["event_type"]
    state["analysis_reason"] = result["reason"]
    return state


def embed_node(state: PipelineState) -> PipelineState:
    add_documents(
        texts=[state["cleaned_text"]],
        metadatas=[{
            "source": state.get("source", "未知来源"),
            "date": state.get("date", ""),
            "company": state.get("company") or "未知",
            "tenant_id": state.get("tenant_id", "default"),
        }],
    )
    return state


def save_node(state: PipelineState) -> PipelineState:
    """落库。此时need_alert还未判定，先按False写入，后续节点按需回填预警字段"""
    record_id = insert_record({
        "raw_text": state["raw_text"],
        "cleaned_text": state["cleaned_text"],
        "source": state.get("source"),
        "date": state.get("date"),
        "company": state.get("company"),
        "event_type": state.get("event_type"),
        "sentiment_label": state.get("sentiment_label"),
        "sentiment_score": state.get("sentiment_score"),
        "dimension_scores": state.get("dimension_scores"),
        "analysis_reason": state.get("analysis_reason"),
        "need_alert": False,
        "media_type": state.get("media_type", "text"),
        "media_path": state.get("media_path"),
        "tenant_id": state.get("tenant_id", "default"),
        "source_type": state.get("source_type", "新闻资讯"),
    })
    state["db_id"] = record_id
    return state


def alert_check_node(state: PipelineState) -> str:
    """条件路由：规则初筛是否命中。

    纯净函数——只读state、只返回下一跳节点名。langgraph规定条件边路由函数
    内的状态修改不会被保存（旧版langgraph侥幸生效，1.x起一律丢弃），
    所以need_alert的写入移到真正执行的节点：触发路径在alert_review_node里
    置True并经update_alert_review写库，未触发路径沿用初始值False。
    """
    if rule_triggered(state["sentiment_score"], state["cleaned_text"]):
        return "alert_review_node"
    return END


def alert_review_node(state: PipelineState) -> PipelineState:
    set_usage_context(state.get("tenant_id", "default"), "alert_review")  # 用量归因
    # 能进入本节点=规则初筛已命中（见alert_check_node），在此置位才有持久化效果
    state["need_alert"] = True
    review = review_alert(
        state["cleaned_text"], state["sentiment_label"], state["sentiment_score"]
    )
    state["alert_is_valid"] = review["is_valid"]
    state["alert_review_reason"] = review["reason"]

    # 回填复核结果到数据库，无论最终是否推送，都保留复核记录方便后续统计误报率
    if state.get("db_id"):
        update_alert_review(state["db_id"], review["is_valid"], review["reason"])
    return state


def route_after_review(state: PipelineState) -> str:
    return "push_node" if state["alert_is_valid"] else END


def push_node(state: PipelineState) -> PipelineState:
    push_alert(
        company=state.get("company") or "未知公司",
        event_type=state.get("event_type") or "未知事件",
        reason=state.get("analysis_reason") or "",
        source_text=state["cleaned_text"],
    )
    if state.get("db_id"):
        mark_pushed(state["db_id"])
    return state


def build_pipeline_graph(async_mode: bool = False):
    """构建舆情处理主流程图。async_mode=True时注册异步节点（配合 graph.ainvoke 使用）。

    异步化的关键与qa_graph一致：LLM等待期间事件循环处理其他请求；
    同步阻塞的DB/向量库操作丢 asyncio.to_thread；条件路由函数是纯内存判断，
    同步实现可直接复用（langgraph在异步图里支持同步路由函数）。
    同步版保留原样（采集脚本/定时任务/worker继续用），行为零变化。
    """
    import asyncio

    if not async_mode:
        graph = StateGraph(PipelineState)
        graph.add_node("ingest_node", ingest_node)
        graph.add_node("embed_node", embed_node)
        graph.add_node("analyze_node", analyze_node)
        graph.add_node("save_node", save_node)
        graph.add_node("alert_review_node", alert_review_node)
        graph.add_node("push_node", push_node)
    else:
        from chains.analysis_chain import analyze_text_gated_async
        from chains.alert_review_chain import review_alert_async

        async def ingest_node_async(state: PipelineState) -> PipelineState:
            state["cleaned_text"] = clean_text(state["raw_text"])
            return state

        async def analyze_node_async(state: PipelineState) -> PipelineState:
            set_usage_context(state.get("tenant_id", "default"), "analyze")
            result = await analyze_text_gated_async(state["cleaned_text"])
            state["sentiment_label"] = result["sentiment_label"]
            state["sentiment_score"] = result["sentiment_score"]
            state["dimension_scores"] = result["dimensions"]
            state["company"] = result["company"]
            state["event_type"] = result["event_type"]
            state["analysis_reason"] = result["reason"]
            return state

        async def embed_node_async(state: PipelineState) -> PipelineState:
            # 向量化（本地bge模型CPU推理）+ Chroma写入是同步阻塞重操作，丢线程池
            await asyncio.to_thread(
                add_documents,
                texts=[state["cleaned_text"]],
                metadatas=[{
                    "source": state.get("source", "未知来源"),
                    "date": state.get("date", ""),
                    "company": state.get("company") or "未知",
                    "tenant_id": state.get("tenant_id", "default"),
                }],
            )
            return state

        async def save_node_async(state: PipelineState) -> PipelineState:
            record_id = await asyncio.to_thread(insert_record, {
                "raw_text": state["raw_text"],
                "cleaned_text": state["cleaned_text"],
                "source": state.get("source"),
                "date": state.get("date"),
                "company": state.get("company"),
                "event_type": state.get("event_type"),
                "sentiment_label": state.get("sentiment_label"),
                "sentiment_score": state.get("sentiment_score"),
                "dimension_scores": state.get("dimension_scores"),
                "analysis_reason": state.get("analysis_reason"),
                "need_alert": False,
                "media_type": state.get("media_type", "text"),
                "media_path": state.get("media_path"),
                "tenant_id": state.get("tenant_id", "default"),
                "source_type": state.get("source_type", "新闻资讯"),
            })
            state["db_id"] = record_id
            return state

        async def alert_review_node_async(state: PipelineState) -> PipelineState:
            set_usage_context(state.get("tenant_id", "default"), "alert_review")
            # 同alert_review_node：进入本节点即规则初筛已命中，在真实节点内置位
            state["need_alert"] = True
            review = await review_alert_async(
                state["cleaned_text"], state["sentiment_label"], state["sentiment_score"]
            )
            state["alert_is_valid"] = review["is_valid"]
            state["alert_review_reason"] = review["reason"]

            if state.get("db_id"):
                await asyncio.to_thread(
                    update_alert_review, state["db_id"], review["is_valid"], review["reason"])
            return state

        async def push_node_async(state: PipelineState) -> PipelineState:
            await asyncio.to_thread(
                push_alert,
                company=state.get("company") or "未知公司",
                event_type=state.get("event_type") or "未知事件",
                reason=state.get("analysis_reason") or "",
                source_text=state["cleaned_text"],
            )
            if state.get("db_id"):
                await asyncio.to_thread(mark_pushed, state["db_id"])
            return state

        graph = StateGraph(PipelineState)
        graph.add_node("ingest_node", ingest_node_async)
        graph.add_node("embed_node", embed_node_async)
        graph.add_node("analyze_node", analyze_node_async)
        graph.add_node("save_node", save_node_async)
        graph.add_node("alert_review_node", alert_review_node_async)
        graph.add_node("push_node", push_node_async)

    graph.set_entry_point("ingest_node")
    # 注意：add_edge/add_conditional_edges 的节点引用必须是名字字符串（见qa_graph注释）
    graph.add_edge("ingest_node", "analyze_node")
    graph.add_edge("analyze_node", "embed_node")
    graph.add_edge("embed_node", "save_node")
    graph.add_conditional_edges("save_node", alert_check_node)
    graph.add_conditional_edges("alert_review_node", route_after_review)
    graph.add_edge("push_node", END)

    return graph.compile()


pipeline_app = build_pipeline_graph()
# 异步版图实例：api /ingest 同步等结果模式用（采集脚本/定时任务仍走同步pipeline_app）
pipeline_app_async = build_pipeline_graph(async_mode=True)


def process(
    raw_text: str,
    source: str = "未知来源",
    date: str = None,
    media_type: str = "text",
    media_path: str = None,
    tenant_id: str = "default",
    source_type: str = None,
) -> dict:
    """
    对外统一入口，供采集脚本/定时任务/多模态normalizer调用。
    date不传时默认取当天，避免落库后日期为空影响后续按日统计。

    media_type/media_path：图片/视频经 ingestion/multimodal/normalizer.py
    转成文本后传入这里，raw_text此时已经是提取好的文字内容，本函数不关心它来自哪种模态。

    tenant_id：企业级多租户隔离，个人/单租户场景下留默认值"default"即可。

    source_type：标注数据来源类型（机构公告/新闻资讯/研报/社交媒体/其他，
    受控词表见 core/taxonomy.py）。不传则按"新闻资讯"处理——RSS新闻是
    本项目最早的数据源，默认值保证旧调用方不用改代码。
    """
    initial_state: PipelineState = {
        "raw_text": raw_text,
        "source": source,
        "date": date or datetime.now().strftime("%Y-%m-%d"),
        "media_type": media_type,
        "media_path": media_path,
        "tenant_id": tenant_id,
        "source_type": normalize_source_type(source_type),
        "cleaned_text": "",
        "sentiment_label": None,
        "sentiment_score": None,
        "dimension_scores": None,
        "company": None,
        "event_type": None,
        "analysis_reason": None,
        "need_alert": False,
        "alert_is_valid": None,
        "alert_review_reason": None,
        "db_id": None,
    }
    return pipeline_app.invoke(initial_state)


async def process_async(
    raw_text: str,
    source: str = "未知来源",
    date: str = None,
    media_type: str = "text",
    media_path: str = None,
    tenant_id: str = "default",
    source_type: str = None,
) -> dict:
    """process() 的异步版：初值构造/语义/返回结构完全一致，图执行走 pipeline_app_async.ainvoke。

    供 async def 路由（/ingest 同步等结果模式）调用；采集脚本/定时任务继续用同步版。
    """
    initial_state: PipelineState = {
        "raw_text": raw_text,
        "source": source,
        "date": date or datetime.now().strftime("%Y-%m-%d"),
        "media_type": media_type,
        "media_path": media_path,
        "tenant_id": tenant_id,
        "source_type": normalize_source_type(source_type),
        "cleaned_text": "",
        "sentiment_label": None,
        "sentiment_score": None,
        "dimension_scores": None,
        "company": None,
        "event_type": None,
        "analysis_reason": None,
        "need_alert": False,
        "alert_is_valid": None,
        "alert_review_reason": None,
        "db_id": None,
    }
    return await pipeline_app_async.ainvoke(initial_state)


if __name__ == "__main__":
    sample = "某公司今日公告称，因涉嫌信息披露违规被证监会立案调查，股价开盘跌停。"
    res = process(sample, source="测试", date="2026-08-06")
    print("情感：", res["sentiment_label"], res["sentiment_score"])
    print("多维度打分：", res.get("dimension_scores"))
    print("公司/事件：", res["company"], res["event_type"])
    print("是否触发预警初筛：", res["need_alert"])
    if res["need_alert"]:
        print("复核结果：", res["alert_is_valid"], res["alert_review_reason"])
