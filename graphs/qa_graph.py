"""
RAG问答流程：retrieve_node --(条件路由：结果是否充足？)-->
  ├─ 充足 或 已达最大重试次数 → generate_answer_node → verify_node → END
  └─ 不充足 → 放宽检索条件 → 回到 retrieve_node（最多重试 RETRIEVE_MAX_RETRY 次）

放宽策略随 retry_count 递增：0=精确匹配（指定公司+近30天），1=放宽时间范围，>=2=全局语义检索。
每步都用 hybrid_retrieve（向量+SQLite关键词合并去重）：向量负责语义相关，关键词负责精确命中。

幻觉控制：generate_answer_node 之后由 verify_node 用 core/faithfulness.py 做三层校验
（引用核查+数值/实体忠实性+LLM蕴含复核），检出风险时带具体原因受限重生成一次（上限1次），
仍不过则加"未经核实"警告。开关：QA_VERIFY（默认开）；校验失败一律降级放行，不阻断主流程。
"""
from langgraph.graph import StateGraph, END
from graphs.state_schemas import QAState
from retrieval.retriever import hybrid_retrieve
from retrieval.kg_retrieve import kg_retrieve_docs
from chains.qa_chain import generate_answer
from core.faithfulness import verify_enabled, verify_answer, verify_with_regenerate
from config.settings import settings
from memory.memory_manager import build_context, record_turn
from core.token_budget import plan_budget, extract_anchor_entities, trim_docs
from core.rate_limiter import allow
from core.compliance import contains_sensitive
from core.usage_context import set_usage_context


def retrieve_node(state: QAState) -> QAState:
    retry = state["retry_count"]
    company = state.get("company")
    tenant_id = state.get("tenant_id", "default")

    if retry == 0:
        docs = hybrid_retrieve(state["query"], company=company, days=30, tenant_id=tenant_id)
        desc = "混合检索：精确匹配（指定公司 + 近30天）"
    elif retry == 1:
        docs = hybrid_retrieve(state["query"], company=company, days=None, tenant_id=tenant_id)
        desc = "混合检索：放宽时间范围（指定公司，不限时间）"
    else:
        docs = hybrid_retrieve(state["query"], company=None, days=None, tenant_id=tenant_id)
        desc = "混合检索：全局检索（不限公司、不限时间）"

    # 图谱与RAG混合路由——图谱类问题（关系型/事实型）附加知识图谱事实，
    # 与RAG检索结果融合。kg_retrieve_docs 对非图谱类问题返回[]，不干扰原有检索。
    kg_docs = kg_retrieve_docs(state["query"])
    if kg_docs:
        docs = list(docs) + kg_docs
        desc += " + 知识图谱关系查询"

    state["retrieved_docs"] = docs
    state["retrieve_desc"] = desc
    # 重试计数必须在节点内自增：条件边路由函数（check_sufficiency）里改state
    # 不会被langgraph保存（路由函数应保持纯净），靠路由函数自增会导致
    # retry_count永远停在0、检索不充足时无限循环（GraphRecursionError）。
    # 自增后retry_count语义="已完成的检索轮数"，check_sufficiency据其判断是否封顶。
    state["retry_count"] = retry + 1
    return state


def check_sufficiency(state: QAState) -> str:
    """条件路由：判断是否需要继续放宽重试。

    纯净函数——只读state、只返回下一跳节点名，禁止修改state
    （langgraph条件边函数内的状态修改不会被保存）。
    retrieve_node每跑完一轮已把retry_count自增，故这里的封顶条件是">"
    （retry_count=3表示已跑完0/1/2三档策略，即最多重试MAX_RETRY次）。
    """
    enough = len(state["retrieved_docs"]) >= settings.RETRIEVE_MIN_DOCS
    reached_limit = state["retry_count"] > settings.RETRIEVE_MAX_RETRY

    if enough or reached_limit:
        return "generate_answer_node"

    return "retrieve_node"


def generate_answer_node(state: QAState) -> QAState:
    query = state["query"]
    # Token控制：按预算截断检索文档（从尾部开始），锚定实体命中的文档强制保留
    docs = trim_docs(
        state["retrieved_docs"],
        state.get("docs_budget", 8000),
        state.get("anchor_entities"),
    )
    answer = generate_answer(
        query, docs,
        conversation_history=state.get("conversation_history", ""),
    )
    state["answer"] = answer
    state["used_docs"] = docs  # 校验层(verify_node)以实际送入LLM的文档为准核对事实
    state["sources"] = [
        {
            "content_preview": d.page_content[:80],
            "source": d.metadata.get("source", "未知来源"),
            "date": d.metadata.get("date", "未知日期"),
        }
        for d in docs
    ]
    return state


def verify_node(state: QAState) -> QAState:
    """三层事实一致性校验 + 受限重生成 + 警告标注。

    QA_VERIFY=0 时直接放行（写enabled=False便于审计区分）。
    校验/重生成全流程包在try/except里：校验本身出错不能让问答挂掉。
    """
    if not verify_enabled():
        state["verification"] = {"enabled": False, "risk": False}
        return state

    try:
        answer, res = verify_with_regenerate(
            state["query"],
            state.get("answer") or "",
            state.get("used_docs") or [],
            conversation_history=state.get("conversation_history", ""),
            max_regen=1,
        )
        state["answer"] = answer
        state["verification"] = res
    except Exception as e:
        # 兜底中的兜底：校验流程本身崩溃时放行原回答，只留审计标记
        state["verification"] = {"enabled": True, "risk": False, "error": f"{type(e).__name__}: {e}"}
    return state


def build_qa_graph(async_mode: bool = False):
    """构建QA流程图。async_mode=True时注册异步节点（配合 graph.ainvoke 使用）。

    异步化的关键：检索/LLM等待期间事件循环可以去处理其他并发请求，
    而不是像同步版那样每个请求占死一个线程干等LLM返回。
    同步版保留原样（Streamlit脚本/评测脚本/任务队列worker继续用），行为零变化。
    """
    import asyncio

    from chains.qa_chain import generate_answer_async
    from core.faithfulness import verify_with_regenerate_async

    if not async_mode:
        graph = StateGraph(QAState)
        graph.add_node("retrieve_node", retrieve_node)
        graph.add_node("generate_answer_node", generate_answer_node)
        graph.add_node("verify_node", verify_node)
    else:
        async def retrieve_node_async(state: QAState) -> QAState:
            retry = state["retry_count"]
            company = state.get("company")
            tenant_id = state.get("tenant_id", "default")
            # 检索层是同步阻塞实现（SQLite+Chroma），丢进线程池执行，不阻塞事件循环
            if retry == 0:
                docs = await asyncio.to_thread(
                    hybrid_retrieve, state["query"], company=company, days=30, tenant_id=tenant_id)
                desc = "混合检索：精确匹配（指定公司 + 近30天）"
            elif retry == 1:
                docs = await asyncio.to_thread(
                    hybrid_retrieve, state["query"], company=company, days=None, tenant_id=tenant_id)
                desc = "混合检索：放宽时间范围（指定公司，不限时间）"
            else:
                docs = await asyncio.to_thread(
                    hybrid_retrieve, state["query"], company=None, days=None, tenant_id=tenant_id)
                desc = "混合检索：全局检索（不限公司、不限时间）"

            kg_docs = await asyncio.to_thread(kg_retrieve_docs, state["query"])
            if kg_docs:
                docs = list(docs) + kg_docs
                desc += " + 知识图谱关系查询"

            state["retrieved_docs"] = docs
            state["retrieve_desc"] = desc
            # 同retrieve_node：重试计数在节点内自增（路由函数改state不生效）
            state["retry_count"] = retry + 1
            return state

        async def generate_answer_node_async(state: QAState) -> QAState:
            docs = trim_docs(
                state["retrieved_docs"],
                state.get("docs_budget", 8000),
                state.get("anchor_entities"),
            )
            answer = await generate_answer_async(
                state["query"], docs,
                conversation_history=state.get("conversation_history", ""),
            )
            state["answer"] = answer
            state["used_docs"] = docs
            state["sources"] = [
                {
                    "content_preview": d.page_content[:80],
                    "source": d.metadata.get("source", "未知来源"),
                    "date": d.metadata.get("date", "未知日期"),
                }
                for d in docs
            ]
            return state

        async def verify_node_async(state: QAState) -> QAState:
            if not verify_enabled():
                state["verification"] = {"enabled": False, "risk": False}
                return state
            try:
                answer, res = await verify_with_regenerate_async(
                    state["query"],
                    state.get("answer") or "",
                    state.get("used_docs") or [],
                    conversation_history=state.get("conversation_history", ""),
                    max_regen=1,
                )
                state["answer"] = answer
                state["verification"] = res
            except Exception as e:
                state["verification"] = {"enabled": True, "risk": False, "error": f"{type(e).__name__}: {e}"}
            return state

        graph = StateGraph(QAState)
        graph.add_node("retrieve_node", retrieve_node_async)
        graph.add_node("generate_answer_node", generate_answer_node_async)
        graph.add_node("verify_node", verify_node_async)

    graph.set_entry_point("retrieve_node")
    graph.add_conditional_edges("retrieve_node", check_sufficiency)
    # 注意：add_edge 的两端必须是节点名字符串（传函数对象在新版langgraph下validate会报
    # "unknown node"，旧版属于侥幸兼容）
    graph.add_edge("generate_answer_node", "verify_node")
    graph.add_edge("verify_node", END)

    return graph.compile()


qa_app = build_qa_graph()
# 异步版图实例：api/routers/query.py 的 async def 路由用（Streamlit/worker仍走同步qa_app）
qa_app_async = build_qa_graph(async_mode=True)


def ask(
    query: str,
    company: str = None,
    tenant_id: str = "default",
    session_id: str = None,
    rate_limit: bool = True,
) -> dict:
    """
    对外统一入口，供 Streamlit / 脚本调用。tenant_id用于企业多租户数据隔离。

    session_id：多轮追问记忆。传入时系统会带上该会话的短期记忆上下文（长期摘要+最近几轮），
    回答后自动把这一轮对话记入记忆；不传则和之前一样是独立单轮问答。

    Token控制：按模型上下文窗口的80%算输入预算，对话历史与检索文档各自截断；
    从当前问题提取关键实体（公司名/股票代码），强制保留包含这些实体的上下文。

    限流：rate_limit=True（默认）时先过 全局令牌桶+单租户限流，被拦截返回提示语；
    router_graph 内部调用时传 rate_limit=False，避免同一个用户请求被计两次。
    """
    if rate_limit and not allow(tenant_id):
        return {
            "answer": "请求过于频繁，请稍后再试。",
            "sources": [],
            "retrieve_desc": "rate_limited",
        }

    set_usage_context(tenant_id, "qa")  # 用量归因

    # 输入防火墙：敏感词拦截，不进入LLM
    if contains_sensitive(query):
        return {
            "answer": "输入包含不合规内容，已拦截。请勿输入与舆情分析无关的敏感内容。",
            "sources": [],
            "retrieve_desc": "sensitive_blocked",
        }

    conv_budget, docs_budget = plan_budget(query)
    anchor_entities = extract_anchor_entities(query)
    conversation_history = build_context(
        tenant_id, session_id,
        token_budget=conv_budget,
        anchor_entities=anchor_entities,
    ) if session_id else ""
    initial_state: QAState = {
        "query": query,
        "company": company,
        "tenant_id": tenant_id,
        "retrieved_docs": [],
        "retrieve_desc": "",
        "retry_count": 0,
        "answer": None,
        "sources": [],
        "conversation_history": conversation_history,
        "docs_budget": docs_budget,
        "anchor_entities": anchor_entities,
        # 幻觉控制新增字段的初值
        "used_docs": [],
        "verification": None,
        "verify_retries": 0,
    }
    result = qa_app.invoke(initial_state)
    if session_id:
        record_turn(tenant_id, session_id, user=query, assistant=result["answer"] or "")
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "retrieve_desc": result["retrieve_desc"],
        "verification": result.get("verification"),
    }


async def ask_async(
    query: str,
    company: str = None,
    tenant_id: str = "default",
    session_id: str = None,
    rate_limit: bool = True,
) -> dict:
    """ask() 的异步版，判定/限流/防火墙/记忆语义完全一致，供 async def 路由调用。

    差异点：
    - 图执行走 qa_app_async.ainvoke（异步节点，LLM等待不占线程不阻塞事件循环）；
    - 同步阻塞的记忆读写（SQLite）用 asyncio.to_thread 丢进线程池；
    - 限流/敏感词/预算规划是内存级微秒操作，直接同步调用。
    """
    import asyncio

    if rate_limit and not allow(tenant_id):
        return {
            "answer": "请求过于频繁，请稍后再试。",
            "sources": [],
            "retrieve_desc": "rate_limited",
        }

    set_usage_context(tenant_id, "qa")

    if contains_sensitive(query):
        return {
            "answer": "输入包含不合规内容，已拦截。请勿输入与舆情分析无关的敏感内容。",
            "sources": [],
            "retrieve_desc": "sensitive_blocked",
        }

    conv_budget, docs_budget = plan_budget(query)
    anchor_entities = extract_anchor_entities(query)
    conversation_history = ""
    if session_id:
        conversation_history = await asyncio.to_thread(
            build_context, tenant_id, session_id,
            token_budget=conv_budget,
            anchor_entities=anchor_entities,
        )
    initial_state: QAState = {
        "query": query,
        "company": company,
        "tenant_id": tenant_id,
        "retrieved_docs": [],
        "retrieve_desc": "",
        "retry_count": 0,
        "answer": None,
        "sources": [],
        "conversation_history": conversation_history,
        "docs_budget": docs_budget,
        "anchor_entities": anchor_entities,
        "used_docs": [],
        "verification": None,
        "verify_retries": 0,
    }
    result = await qa_app_async.ainvoke(initial_state)
    if session_id:
        await asyncio.to_thread(
            record_turn, tenant_id, session_id, user=query, assistant=result["answer"] or "")
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "retrieve_desc": result["retrieve_desc"],
        "verification": result.get("verification"),
    }


if __name__ == "__main__":
    # 简单自测（需先运行 retrieval/vectorstore.py 写入至少一条测试数据）
    res = ask("示例公司最近业绩怎么样？", company="示例公司")
    print("回答：", res["answer"])
    print("检索方式：", res["retrieve_desc"])
    print("来源：", res["sources"])
    print("校验：", res.get("verification"))
