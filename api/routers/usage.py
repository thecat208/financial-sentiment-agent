"""
用量统计 + 审计日志 + 预警人工复核接口。

用量统计/审计日志：复用 storage/usage_tracker.py 和 storage/audit_log.py，
和 Streamlit「📋 用量与审计」标签页读的是同一份数据。

人工复核覆盖预警判定：复用 storage/repository.manual_override_alert()，
和 Streamlit「预警看板」标签页的"标记为属实/误报"按钮走的是同一个函数，
调用后会自动记一条审计日志（谁在什么时间把哪条预警改成了什么状态）。
"""
from fastapi import APIRouter, Depends

from api.deps import get_tenant_id, get_user_id, require_api_key
from api.schemas import ManualOverrideRequest
from storage.usage_tracker import get_usage_stats
from storage.audit_log import get_audit_log
from storage.repository import manual_override_alert

router = APIRouter(prefix="/api/v1", tags=["用量与审计"], dependencies=[Depends(require_api_key)])


@router.get("/usage/stats", summary="近N天用量统计：LLM调用/Token消耗、API调用次数")
def usage_stats(days: int = 30, tenant_id: str = Depends(get_tenant_id)):
    return get_usage_stats(tenant_id, days=days)


@router.get("/audit-log", summary="近N天审计日志，可选按action过滤")
def audit_log(days: int = 30, action: str = None, tenant_id: str = Depends(get_tenant_id)):
    return {"records": get_audit_log(tenant_id, days=days, action=action)}


@router.post("/alerts/{record_id}/override", summary="人工复核覆盖LLM的预警判定（会自动记审计日志）")
def override_alert(
    record_id: int,
    req: ManualOverrideRequest,
    tenant_id: str = Depends(get_tenant_id),
    header_user_id: str = Depends(get_user_id),
):
    """
    user_id优先取请求体里的字段，请求体没传才退回请求头X-User-Id。
    优先用请求体是因为操作人姓名很可能是中文，HTTP请求头按规范只能是ASCII，
    直接把中文塞进header在不少HTTP客户端（比如httpx）里会直接报编码错误。
    X-User-Id依然保留，给用ASCII工号/用户名（常见于企业SSO场景）的调用方
    一个更符合HTTP习惯的选项。
    """
    manual_override_alert(
        record_id, is_valid=req.is_valid, tenant_id=tenant_id,
        user_id=req.user_id or header_user_id, note=req.note,
    )
    return {"status": "ok", "record_id": record_id, "is_valid": req.is_valid}
