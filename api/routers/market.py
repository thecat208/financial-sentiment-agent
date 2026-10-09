"""
舆情-行情关联分析与回测接口。复用 analysis/backtest.py 里的函数，
返回结果统一带 disclaimer 字段——这是**研究工具**，不构成投资建议，
调用方展示给最终用户时请原样带上这个字段，不要省略。
"""
from fastapi import APIRouter, Depends

from api.deps import get_tenant_id, require_api_key
from api.schemas import BacktestRequest
from analysis.backtest import get_sentiment_price_overlay, run_alert_backtest

router = APIRouter(prefix="/api/v1/market", tags=["舆情-行情联动"], dependencies=[Depends(require_api_key)])


@router.get("/{company}/overlay", summary="情感趋势与股价走势叠加+相关系数")
def overlay(company: str, days: int = 30, tenant_id: str = Depends(get_tenant_id)):
    return get_sentiment_price_overlay(company, days=days, tenant_id=tenant_id)


@router.post("/backtest", summary="预警信号回测：持有N个交易日后的涨跌幅统计")
def backtest(req: BacktestRequest, tenant_id: str = Depends(get_tenant_id)):
    return run_alert_backtest(
        company=req.company, tenant_id=tenant_id, days=req.days,
        holding_days=req.holding_days, alert_scope=req.alert_scope,
    )
