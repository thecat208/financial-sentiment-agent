"""
操作审计日志。

记录"谁在什么时间对哪条数据做了什么操作"，金融行业强监管场景的常见合规要求
——比日志脱敏更进一步，是结构化、可查询的操作记录，不是散落在日志文件里的文本。

重要局限：本项目目前没有真正的用户认证体系，所以这里的 user_id 是调用方
自报的标识（Streamlit里手填的"操作人"文本框，或API的 X-User-Id 请求头），
不是经过身份验证的登录态，任何人都可以填任意的user_id（与租户ID处境相同）。
真正做到"审计记录不可抵赖"必须先有SSO校验身份；这里先把审计记录的数据结构
和记录时机打好地基，等SSO接入后user_id换成真实登录态即可，调用方式不需要变。
"""
from datetime import datetime, timedelta

from storage.db import get_conn

UNKNOWN_USER = "unknown"


def record_audit(tenant_id: str, action: str, user_id: str = None,
                  target_type: str = None, target_id: str = None, detail: str = None) -> int:
    """
    记一条审计日志，返回自增id。user_id未传时记为UNKNOWN_USER而不是留空，
    这样"这条操作到底是谁做的都不知道"这件事本身在数据里是显式可见的，
    不会和"记录本身缺失"混淆。
    """
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO audit_log (tenant_id, user_id, action, target_type, target_id, detail)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (tenant_id, user_id or UNKNOWN_USER, action, target_type, target_id, detail),
        )
        return cur.lastrowid


def get_audit_log(tenant_id: str, days: int = 30, action: str = None) -> list:
    """查某租户近N天的审计记录，可选按action过滤，按时间倒序（最新的在前）"""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    query = "SELECT * FROM audit_log WHERE tenant_id = ? AND created_at >= ?"
    params = [tenant_id, cutoff]
    if action:
        query += " AND action = ?"
        params.append(action)
    query += " ORDER BY created_at DESC"

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
