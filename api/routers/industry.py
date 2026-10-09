"""行业舆情热度榜接口，复用 storage/repository.get_industry_heat_ranking()。"""
from fastapi import APIRouter, Depends

from api.deps import get_tenant_id, require_api_key
from storage.repository import get_industry_heat_ranking

router = APIRouter(prefix="/api/v1/industry", tags=["行业热度"], dependencies=[Depends(require_api_key)])


@router.get("/ranking", summary="行业舆情热度榜：按行业聚合舆情量/平均情感分排行")
def ranking(days: int = 7, tenant_id: str = Depends(get_tenant_id)):
    return {"days": days, "ranking": get_industry_heat_ranking(days=days, tenant_id=tenant_id)}
