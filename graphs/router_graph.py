"""
查询路由 / Supervisor：intent_node（意图识别） → 条件路由 →
  qa          → qa_node，调用 graphs/qa_graph.ask()
  report      → report_node，调用 graphs/report_graph.generate_report()
  alert_stats → alert_node，调用 storage.repository.get_alert_stats()
  other       → other_node，直接返回引导语，不进入任何子流程

设计意图：用户只需输入一句自然语言，系统自动判断该走哪条已有的LangGraph流程；
这一层不重新实现问答/日报/预警逻辑，只做"意图识别+分发"。
"""
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from chains.intent_chain import classify_intent
from graphs.qa_graph import ask
from graphs.report_graph import generate_report
from storage.repository import get_alert_stats
from memory.memory_manager import record_turn
from core.rate_limiter import allow
from core.compliance import contains_sensitive
from core.usage_context import set_usage_context


class RouterState(TypedDict):
    query: str
    tenant_id: str
    session_id: Optional[str]   # 多轮追问记忆：问答分支透传给 ask()，便于理解指代
    intent: Optional[str]
    company: Optional[str]
    date: Optional[str]
    result_text: Optional[str]
    result_payload: Optional[dict]


def intent_node(state: RouterState) -> RouterState:
    set_usage_context(state.get("tenant_id", "default"), "intent")  # 用量归因
    parsed = classify_intent(state["query"])
    state["intent"] = parsed["intent"]
    state["company"] = parsed.get("company") or None
    state["date"] = parsed.get("date") or None
    return state


def route_by_intent(state: RouterState) -> str:
    return {
        "qa": "qa_node",
        "report": "report_node",
        "alert_stats": "alert_node",
    }.get(state["intent"], "other_node")


def qa_node(state: RouterState) -> RouterState:
    # rate_limit=False：handle_query入口已经做过限流，避免同一请求被计两次
    result = ask(
        state["query"],
        company=state.get("company"),
        tenant_id=state.get("tenant_id", "default"),
        session_id=state.get("session_id"),
        rate_limit=False,
    )
    state["result_text"] = result["answer"]
    state["result_payload"] = result
    return state


def report_node(state: RouterState) -> RouterState:
    result = generate_report(state.get("date"), tenant_id=state.get("tenant_id", "default"))
    state["result_text"] = result["report_text"]
    state["result_payload"] = result
    return state


def alert_node(state: RouterState) -> RouterState:
    stats = get_alert_stats(days=30, tenant_id=state.get("tenant_id", "default"))
    total = stats["total"] or 0
    triggered = stats["triggered"] or 0
    invalid = stats["invalid"] or 0
    rate = round(invalid / triggered * 100, 1) if triggered else 0
    state["result_text"] = (
        f"最近30天共监测到 {total} 条舆情，规则初筛触发预警 {triggered} 次，"
        f"其中经LLM复核判定为误报的约占 {rate}%。"
    )
    state["result_payload"] = stats
    return state


def other_node(state: RouterState) -> RouterState:
    state["result_text"] = "抱歉，我目前只能回答舆情问答、日报查看、预警统计相关的问题，换个说法试试？"
    state["result_payload"] = None
    return state


def build_router_graph(async_mode: bool = False):
    """构建路由图。async_mode=True时注册异步节点（配合 graph.ainvoke 使用）。
    异步版节点语义与同步版完全一致；阻塞操作（DB/LLM）按阻塞类型分别处理：
    LLM调用原生await，同步阻塞的DB/日报生成丢进线程池。同步版保留原样零变化。"""
    import asyncio

    if not async_mode:
        graph = StateGraph(RouterState)
        graph.add_node("intent_node", intent_node)
        graph.add_node("qa_node", qa_node)
        graph.add_node("report_node", report_node)
        graph.add_node("alert_node", alert_node)
        graph.add_node("other_node", other_node)
    else:
        from chains.intent_chain import classify_intent_async
        from graphs.qa_graph import ask_async
        from graphs.report_graph import generate_report_async

        async def intent_node_async(state: RouterState) -> RouterState:
            set_usage_context(state.get("tenant_id", "default"), "intent")
            parsed = await classify_intent_async(state["query"])
            state["intent"] = parsed["intent"]
            state["company"] = parsed.get("company") or None
            state["date"] = parsed.get("date") or None
            return state

        async def qa_node_async(state: RouterState) -> RouterState:
            result = await ask_async(
                state["query"],
                company=state.get("company"),
                tenant_id=state.get("tenant_id", "default"),
                session_id=state.get("session_id"),
                rate_limit=False,
            )
            state["result_text"] = result["answer"]
            state["result_payload"] = result
            return state

        async def report_node_async(state: RouterState) -> RouterState:
            # 日报生成走异步图（多公司小结gather并发，LLM等待不阻塞事件循环）
            result = await generate_report_async(
                state.get("date"), tenant_id=state.get("tenant_id", "default"))
            state["result_text"] = result["report_text"]
            state["result_payload"] = result
            return state

        async def alert_node_async(state: RouterState) -> RouterState:
            stats = await asyncio.to_thread(
                get_alert_stats, days=30, tenant_id=state.get("tenant_id", "default"))
            total = stats["total"] or 0
            triggered = stats["triggered"] or 0
            invalid = stats["invalid"] or 0
            rate = round(invalid / triggered * 100, 1) if triggered else 0
            state["result_text"] = (
                f"最近30天共监测到 {total} 条舆情，规则初筛触发预警 {triggered} 次，"
                f"其中经LLM复核判定为误报的约占 {rate}%。"
            )
            state["result_payload"] = stats
            return state

        async def other_node_async(state: RouterState) -> RouterState:
            state["result_text"] = "抱歉，我目前只能回答舆情问答、日报查看、预警统计相关的问题，换个说法试试？"
            state["result_payload"] = None
            return state

        graph = StateGraph(RouterState)
        graph.add_node("intent_node", intent_node_async)
        graph.add_node("qa_node", qa_node_async)
        graph.add_node("report_node", report_node_async)
        graph.add_node("alert_node", alert_node_async)
        graph.add_node("other_node", other_node_async)

    graph.set_entry_point("intent_node")
    graph.add_conditional_edges("intent_node", route_by_intent)
    graph.add_edge("qa_node", END)
    graph.add_edge("report_node", END)
    graph.add_edge("alert_node", END)
    graph.add_edge("other_node", END)

    return graph.compile()


router_app = build_router_graph()
# 异步版图实例：api/routers/query.py 的 async def 路由用（Streamlit/脚本仍走同步router_app）
router_app_async = build_router_graph(async_mode=True)


def handle_query(query: str, tenant_id: str = "default", session_id: str = None) -> dict:
    """对外统一入口：给一句自然语言，自动识别意图并返回处理结果。

    session_id：多轮追问记忆。传入时问答分支会带上该会话的短期记忆上下文，
    无论路由到哪个分支，本轮 用户输入/结果 都会记入记忆，保证跨意图的对话连贯。

    限流：入口处先过 全局令牌桶+单租户限流，被拦截直接返回提示，不进任何子流程，
    防止恶意刷接口烧Token（意图识别本身也是一次LLM调用，必须挡在它前面）。
    """
    if not allow(tenant_id):
        return {"intent": "rate_limited", "text": "请求过于频繁，请稍后再试。", "payload": None}

    # 输入防火墙：敏感词拦截，不进意图识别等任何子流程
    if contains_sensitive(query):
        return {
            "intent": "blocked",
            "text": "输入包含不合规内容，已拦截。请勿输入与舆情分析无关的敏感内容。",
            "payload": None,
        }

    initial_state: RouterState = {
        "query": query,
        "tenant_id": tenant_id,
        "session_id": session_id,
        "intent": None,
        "company": None,
        "date": None,
        "result_text": None,
        "result_payload": None,
    }
    result = router_app.invoke(initial_state)
    if session_id:
        record_turn(tenant_id, session_id, user=query, assistant=result["result_text"] or "")
    return {
        "intent": result["intent"],
        "text": result["result_text"],
        "payload": result["result_payload"],
    }


async def handle_query_async(query: str, tenant_id: str = "default", session_id: str = None) -> dict:
    """handle_query 的异步版，限流/防火墙/记忆语义完全一致，供 async def 路由调用。"""
    import asyncio

    if not allow(tenant_id):
        return {"intent": "rate_limited", "text": "请求过于频繁，请稍后再试。", "payload": None}

    if contains_sensitive(query):
        return {
            "intent": "blocked",
            "text": "输入包含不合规内容，已拦截。请勿输入与舆情分析无关的敏感内容。",
            "payload": None,
        }

    initial_state: RouterState = {
        "query": query,
        "tenant_id": tenant_id,
        "session_id": session_id,
        "intent": None,
        "company": None,
        "date": None,
        "result_text": None,
        "result_payload": None,
    }
    result = await router_app_async.ainvoke(initial_state)
    if session_id:
        await asyncio.to_thread(
            record_turn, tenant_id, session_id, user=query, assistant=result["result_text"] or "")
    return {
        "intent": result["intent"],
        "text": result["result_text"],
        "payload": result["result_payload"],
    }


if __name__ == "__main__":
    q = input("请输入你的问题（如：示例公司最近舆情怎么样 / 今天的日报 / 最近预警误报率高吗）：").strip()
    res = handle_query(q)
    print(f"\n[识别意图：{res['intent']}]\n")
    print(res["text"])
