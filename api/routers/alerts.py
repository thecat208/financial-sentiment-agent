"""预警统计/记录接口，复用 storage/repository.py 里已有的查询函数。"""
from fastapi import APIRouter, Depends

from api.deps import get_tenant_id, require_api_key
from api.schemas import AlertStats
from storage.repository import get_alert_stats, get_alert_records

router = APIRouter(prefix="/api/v1/alerts", tags=["预警"], dependencies=[Depends(require_api_key)])


@router.get("/stats", response_model=AlertStats, summary="近N天预警统计（触发数/误报数/推送数）")
def alert_stats(days: int = 30, tenant_id: str = Depends(get_tenant_id)):
    return get_alert_stats(days=days, tenant_id=tenant_id)


@router.get("/records", summary="近N天预警记录明细")
def alert_records(days: int = 30, only_triggered: bool = True, tenant_id: str = Depends(get_tenant_id)):
    return {"records": get_alert_records(days=days, only_triggered=only_triggered, tenant_id=tenant_id)}
