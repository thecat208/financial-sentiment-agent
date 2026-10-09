"""
舆情-行情关联分析与回测框架。

只做两件事，都是**研究工具**，不给投资建议：
  1. overlay：把某公司的情感趋势和股价曲线按日期对齐，供用户自己观察走势是否相关，
     顺带算一个皮尔逊相关系数作为参考（相关系数本身也不构成任何判断依据）。
  2. backtest：统计"预警触发后N个交易日"的平均涨跌幅与正负案例占比——这仍是
     描述性统计而非"信号有效"的结论，样本量小时尤其要小心。
所有对外返回结果都带 `disclaimer` 字段（core.compliance.BACKTEST_DISCLAIMER），
上层UI/API必须原样展示，不能省略。
"""
from datetime import datetime, timedelta

import pandas as pd

from core.compliance import BACKTEST_DISCLAIMER
from config.settings import settings
from market_data.provider import get_daily_prices, resolve_stock_code, get_price_n_trading_days_after
from storage.repository import get_company_daily_trend, get_alert_records


def get_sentiment_price_overlay(company: str, days: int = 30, tenant_id: str = "default") -> dict:
    """
    情感趋势 + 股价曲线按日期对齐，供图表叠加展示。
    返回：{"company", "code", "is_mock", "points":[{"date","avg_score","cnt",
    "negative_cnt","close","pct_change"}], "correlation"(情感均分vs当日涨跌幅的
    皮尔逊相关系数，样本<3时为None), "sample_size", "disclaimer"}。
    """
    sentiment_trend = get_company_daily_trend(company, days=days, tenant_id=tenant_id)
    if not sentiment_trend:
        return {
            "company": company, "code": resolve_stock_code(company), "is_mock": None,
            "points": [], "correlation": None, "sample_size": 0,
            "disclaimer": BACKTEST_DISCLAIMER,
        }

    start_date = sentiment_trend[0]["date"]
    end_date = min(
        (datetime.strptime(sentiment_trend[-1]["date"], "%Y-%m-%d") + timedelta(days=5)).strftime("%Y-%m-%d"),
        datetime.now().strftime("%Y-%m-%d"),
    )
    price_result = get_daily_prices(company, start_date, end_date)
    price_by_date = {row["date"]: row for row in price_result["prices"]}

    points = []
    for s in sentiment_trend:
        p = price_by_date.get(s["date"])
        points.append({
            "date": s["date"],
            "avg_score": s["avg_score"],
            "cnt": s["cnt"],
            "negative_cnt": s["negative_cnt"],
            "close": p["close"] if p else None,
            "pct_change": p["pct_change"] if p else None,
        })

    matched = [pt for pt in points if pt["pct_change"] is not None]
    correlation = None
    if len(matched) >= 3:
        df = pd.DataFrame(matched)
        corr_val = df["avg_score"].corr(df["pct_change"])
        correlation = round(float(corr_val), 3) if pd.notna(corr_val) else None

    return {
        "company": company,
        "code": price_result["code"],
        "is_mock": price_result["is_mock"],
        "points": points,
        "correlation": correlation,
        "sample_size": len(matched),
        "disclaimer": BACKTEST_DISCLAIMER,
    }


def run_alert_backtest(company: str = None, tenant_id: str = "default", days: int = 90,
                        holding_days: int = None, alert_scope: str = "triggered") -> dict:
    """
    预警信号回测："预警触发后N个交易日的平均涨跌幅、正负案例占比"。
    参数：company不传则统计该租户下所有公司；holding_days默认取
    settings.BACKTEST_DEFAULT_HOLDING_DAYS；alert_scope="triggered"统计规则初筛触发的
    全部记录（need_alert=1），"valid_only"只统计LLM复核判为属实的记录（样本更少但信号更"干净"）。
    返回：{"samples","avg_return_pct","win_rate","positive_count","negative_count",
    "holding_days","details","disclaimer"}，details为[{"date","company","entry_close","exit_close","return_pct"}]。
    """
    if holding_days is None:
        holding_days = settings.BACKTEST_DEFAULT_HOLDING_DAYS

    records = get_alert_records(days=days, only_triggered=True, tenant_id=tenant_id)
    if company:
        records = [r for r in records if r.get("company") == company]
    if alert_scope == "valid_only":
        records = [r for r in records if r.get("alert_is_valid") == 1]

    details = []
    for r in records:
        rec_company = r.get("company")
        rec_date = r.get("date")
        if not rec_company or not rec_date:
            continue
        entry = get_price_n_trading_days_after(rec_company, rec_date, 0)
        exit_ = get_price_n_trading_days_after(rec_company, rec_date, holding_days)
        if not entry or not exit_:
            continue  # 行情数据覆盖不到（比如太新还没有足够的后续交易日），跳过该样本
        ret_pct = round((exit_["close"] - entry["close"]) / entry["close"] * 100, 2)
        details.append({
            "date": rec_date,
            "company": rec_company,
            "entry_close": entry["close"],
            "exit_close": exit_["close"],
            "return_pct": ret_pct,
        })

    samples = len(details)
    positive = sum(1 for d in details if d["return_pct"] > 0)
    negative = sum(1 for d in details if d["return_pct"] < 0)
    avg_return = round(sum(d["return_pct"] for d in details) / samples, 2) if samples else None
    win_rate = round(positive / samples * 100, 1) if samples else None

    return {
        "samples": samples,
        "avg_return_pct": avg_return,
        "win_rate": win_rate,
        "positive_count": positive,
        "negative_count": negative,
        "holding_days": holding_days,
        "details": details,
        "low_sample_warning": samples < settings.BACKTEST_MIN_SAMPLES,
        "disclaimer": BACKTEST_DISCLAIMER,
    }
