"""
每日简报生成流程：schedule_trigger → fetch_stats_node（从SQLite按公司分组统计）
→ summarize_each_node（对每个公司调用report_chain生成小结）
→ compose_report_node（拼成完整Markdown日报）→ save_report_node（落盘到 reports/日期.md，可选推送）→ END。

复用了SQLite落库阶段建的 get_daily_company_stats / get_daily_records，
不需要额外的数据管道。
"""
import os
import json
from datetime import datetime
from core.usage_context import set_usage_context
from langgraph.graph import StateGraph, END
from graphs.state_schemas import ReportState
from storage.repository import get_daily_company_stats, get_daily_records
from chains.report_chain import summarize_company
from alert.push import push_all
from core.compliance import append_disclaimer

REPORTS_DIR = os.getenv("REPORTS_DIR", "./reports")


def fetch_stats_node(state: ReportState) -> ReportState:
    state["company_stats"] = get_daily_company_stats(state["date"], tenant_id=state.get("tenant_id", "default"))
    return state


def summarize_each_node(state: ReportState) -> ReportState:
    if not state["company_stats"]:
        state["company_summaries"] = []
        return state

    all_records = get_daily_records(state["date"], tenant_id=state.get("tenant_id", "default"))
    summaries = []
    for stat in state["company_stats"]:
        company = stat["company"] or "未知公司"
        samples = [r["cleaned_text"] for r in all_records if r["company"] == stat["company"]]
        summary = summarize_company(company, stat, samples)
        summaries.append({"company": company, "summary": summary, "stats": stat})

    state["company_summaries"] = summaries
    return state


def compose_report_node(state: ReportState) -> ReportState:
    lines = [f"# 金融舆情日报 {state['date']}", ""]

    if not state["company_summaries"]:
        lines.append("今日暂无相关舆情数据。")
    else:
        total = sum(s["stats"]["cnt"] for s in state["company_summaries"])
        lines.append(f"共监测到 {len(state['company_summaries'])} 家公司、{total} 条相关舆情。\n")
        for item in state["company_summaries"]:
            s = item["stats"]
            avg_score = round(s.get("avg_score") or 0, 2)
            lines.append(f"## {item['company']}")
            lines.append(f"条数：{s['cnt']} | 平均情感分：{avg_score} | 负面条数：{s['negative_cnt']}")
            lines.append("")
            lines.append(item["summary"])
            lines.append("")

    # 输出防火墙：日报末尾强制拼接免责声明（幂等）
    state["report_text"] = append_disclaimer("\n".join(lines))
    return state


def save_report_node(state: ReportState) -> ReportState:
    _save_report_and_push(state)
    return state


def build_report_graph(async_mode: bool = False):
    """构建日报生成流程图。async_mode=True时注册异步节点（配合 graph.ainvoke 使用）。

    异步版最大收益点：summarize_each_node_async 对多家公司的小结LLM调用用
    asyncio.gather 并发执行——同步版是for循环逐家串行等待，N家公司总耗时≈N倍
    单次LLM延迟，异步版≈最慢一家（受服务商并发配额约束）。
    DB聚合/落盘/推送是同步阻塞操作，丢 asyncio.to_thread；
    compose_report_node是纯CPU字符串拼接（微秒级），同步实现两版共用。
    同步版保留原样（定时任务/脚本继续用），行为零变化。
    """
    import asyncio

    if not async_mode:
        graph = StateGraph(ReportState)
        graph.add_node("fetch_stats_node", fetch_stats_node)
        graph.add_node("summarize_each_node", summarize_each_node)
        graph.add_node("compose_report_node", compose_report_node)
        graph.add_node("save_report_node", save_report_node)
    else:
        from chains.report_chain import summarize_company_async

        async def fetch_stats_node_async(state: ReportState) -> ReportState:
            state["company_stats"] = await asyncio.to_thread(
                get_daily_company_stats, state["date"],
                tenant_id=state.get("tenant_id", "default"))
            return state

        async def summarize_each_node_async(state: ReportState) -> ReportState:
            if not state["company_stats"]:
                state["company_summaries"] = []
                return state

            all_records = await asyncio.to_thread(
                get_daily_records, state["date"],
                tenant_id=state.get("tenant_id", "default"))

            async def one(stat):
                company = stat["company"] or "未知公司"
                samples = [r["cleaned_text"] for r in all_records if r["company"] == stat["company"]]
                summary = await summarize_company_async(company, stat, samples)
                return {"company": company, "summary": summary, "stats": stat}

            # 多家公司小结并发执行：LLM等待期间事件循环处理其他公司的小结
            state["company_summaries"] = list(await asyncio.gather(
                *[one(stat) for stat in state["company_stats"]]))
            return state

        async def save_report_node_async(state: ReportState) -> ReportState:
            # 落盘+推送是同步阻塞操作，整体丢线程池（与同步版save_report_node逻辑一致）
            await asyncio.to_thread(_save_report_and_push, state)
            return state

        graph = StateGraph(ReportState)
        graph.add_node("fetch_stats_node", fetch_stats_node_async)
        graph.add_node("summarize_each_node", summarize_each_node_async)
        graph.add_node("compose_report_node", compose_report_node)
        graph.add_node("save_report_node", save_report_node_async)

    graph.set_entry_point("fetch_stats_node")
    graph.add_edge("fetch_stats_node", "summarize_each_node")
    graph.add_edge("summarize_each_node", "compose_report_node")
    graph.add_edge("compose_report_node", "save_report_node")
    graph.add_edge("save_report_node", END)

    return graph.compile()


report_app = build_report_graph()
# 异步版图实例：async def 路由（/reports/daily）与router_graph异步report节点用
report_app_async = build_report_graph(async_mode=True)


def _save_report_and_push(state: ReportState) -> None:
    """落盘+JSON侧车+推送。从save_report_node拆出来的纯同步实现，
    供同步节点直接调用、异步节点经asyncio.to_thread调用，两版逻辑只有一份。"""
    tenant_id = state.get("tenant_id", "default")
    tenant_dir = os.path.join(REPORTS_DIR, tenant_id)
    os.makedirs(tenant_dir, exist_ok=True)
    path = os.path.join(tenant_dir, f"{state['date']}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(state["report_text"])
    state["report_path"] = path

    # 结构化数据（公司统计+LLM小结）另存一份JSON，供 export/report_export.py
    # 导出Excel/PDF时直接读取，不用为了导出而重新调一次LLM生成小结
    json_path = os.path.join(tenant_dir, f"{state['date']}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "date": state["date"],
            "tenant_id": tenant_id,
            "company_stats": state.get("company_stats") or [],
            "company_summaries": state.get("company_summaries") or [],
        }, f, ensure_ascii=False, indent=2)

    # 顺带尝试推送到已配置的渠道（未配置时各渠道降级为打印，不影响落盘结果）
    push_all(f"舆情日报 {state['date']}", state["report_text"])


def generate_report(date: str = None, tenant_id: str = "default") -> dict:
    date = date or datetime.now().strftime("%Y-%m-%d")
    set_usage_context(tenant_id, "report")  # 用量归因，覆盖本次报告涉及的所有公司小结LLM调用
    initial_state: ReportState = {
        "date": date,
        "tenant_id": tenant_id,
        "company_stats": [],
        "company_summaries": [],
        "report_text": None,
        "report_path": None,
    }
    return report_app.invoke(initial_state)


async def generate_report_async(date: str = None, tenant_id: str = "default") -> dict:
    """generate_report() 的异步版：语义/返回结构完全一致，图执行走 report_app_async.ainvoke。

    多家公司的小结LLM调用在图内 asyncio.gather 并发（详见build_report_graph注释）。
    用法归因说明：contextvars 在 asyncio.gather 创建子任务时会被复制继承，
    这里先 set 再进图，所有小结调用的用量都会归到 (tenant_id, "report")。
    """
    date = date or datetime.now().strftime("%Y-%m-%d")
    set_usage_context(tenant_id, "report")
    initial_state: ReportState = {
        "date": date,
        "tenant_id": tenant_id,
        "company_stats": [],
        "company_summaries": [],
        "report_text": None,
        "report_path": None,
    }
    return await report_app_async.ainvoke(initial_state)


if __name__ == "__main__":
    date = input("生成哪天的日报？（YYYY-MM-DD，直接回车用今天）：").strip() or None
    result = generate_report(date)
    print("\n" + result["report_text"])
    print(f"\n已保存到：{result['report_path']}")
