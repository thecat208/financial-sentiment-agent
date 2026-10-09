"""
用量统计。

记两类事件到同一张 usage_events 表：
  - llm_call：每次LLM调用的token消耗（谁调的、调了哪个操作、耗了多少token）
  - api_call：每次 /api/* 请求的调用次数（不涉及token，count就是1次）

这是"计费/用量统计"的数据基础设施，本身不做计费——这里只保证"每个租户
用了多少"这件事有准确、可查询的记录，计费规则可以在这份数据之上另起一层去算。
"""
from datetime import datetime, timedelta

from storage.db import get_conn


def record_llm_usage(tenant_id: str, operation: str, model: str,
                      input_tokens: int = 0, output_tokens: int = 0) -> None:
    """
    记一次LLM调用的token消耗。operation是调用场景标识，比如"analyze"/"qa"/
    "report"/"alert_review"/"intent"（对应 chains/ 下各个chain的职责），
    不是任意字符串——统计的时候要能按场景分组，operation取值太随意就没法分组了。
    """
    now = datetime.now()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO usage_events
               (tenant_id, event_kind, operation, model, input_tokens, output_tokens, total_tokens, date)
               VALUES (?, 'llm_call', ?, ?, ?, ?, ?, ?)""",
            (tenant_id, operation, model, input_tokens, output_tokens,
             input_tokens + output_tokens, now.strftime("%Y-%m-%d")),
        )


def record_api_call(tenant_id: str, endpoint: str) -> None:
    """记一次API端点调用。供 api/deps.py 的中间件/依赖项调用，不涉及token"""
    now = datetime.now()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO usage_events
               (tenant_id, event_kind, operation, date)
               VALUES (?, 'api_call', ?, ?)""",
            (tenant_id, endpoint, now.strftime("%Y-%m-%d")),
        )


def get_usage_stats(tenant_id: str, days: int = 30) -> dict:
    """
    某租户近N天的用量汇总，返回dict：{"days", "llm_calls", "input_tokens",
    "output_tokens", "total_tokens", "by_operation":[{"operation","calls","total_tokens"}],
    "api_calls", "api_calls_by_endpoint":[{"endpoint","calls"}]}。
    """
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    with get_conn() as conn:
        llm_totals = conn.execute(
            """SELECT COUNT(*) AS calls, COALESCE(SUM(input_tokens),0) AS input_tokens,
                      COALESCE(SUM(output_tokens),0) AS output_tokens, COALESCE(SUM(total_tokens),0) AS total_tokens
               FROM usage_events
               WHERE tenant_id = ? AND event_kind = 'llm_call' AND date >= ?""",
            (tenant_id, cutoff),
        ).fetchone()

        by_operation = conn.execute(
            """SELECT operation, COUNT(*) AS calls, COALESCE(SUM(total_tokens),0) AS total_tokens
               FROM usage_events
               WHERE tenant_id = ? AND event_kind = 'llm_call' AND date >= ?
               GROUP BY operation ORDER BY total_tokens DESC""",
            (tenant_id, cutoff),
        ).fetchall()

        api_total = conn.execute(
            """SELECT COUNT(*) AS calls FROM usage_events
               WHERE tenant_id = ? AND event_kind = 'api_call' AND date >= ?""",
            (tenant_id, cutoff),
        ).fetchone()

        by_endpoint = conn.execute(
            """SELECT operation AS endpoint, COUNT(*) AS calls FROM usage_events
               WHERE tenant_id = ? AND event_kind = 'api_call' AND date >= ?
               GROUP BY operation ORDER BY calls DESC""",
            (tenant_id, cutoff),
        ).fetchall()

    return {
        "days": days,
        "llm_calls": llm_totals["calls"] or 0,
        "input_tokens": llm_totals["input_tokens"] or 0,
        "output_tokens": llm_totals["output_tokens"] or 0,
        "total_tokens": llm_totals["total_tokens"] or 0,
        "by_operation": [dict(r) for r in by_operation],
        "api_calls": api_total["calls"] or 0,
        "api_calls_by_endpoint": [dict(r) for r in by_endpoint],
    }
