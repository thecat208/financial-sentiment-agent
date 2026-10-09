"""
验收测试：多公司横向对比与行业热度榜。

不依赖真实LLM/akshare——只测存储层的聚合查询逻辑和行业归属解析，
对应README的验收标准：
  1. 能同时选择2-5家公司看对比趋势图（底层：get_company_daily_trend 按公司分别查询）
  2. 能看到一个按行业排序的舆情热度榜单（get_industry_heat_ranking）

用法：
    SQLITE_DB_PATH=/tmp/p11_test.db python scripts/test_p11_industry.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.repository import (
    insert_record,
    get_company_window_stats,
    get_industry_heat_ranking,
    get_company_daily_trend,
)
from knowledge_graph.seed_data import resolve_industry

TENANT = "p11_test_tenant"  # 用独立租户跑测试，不和真实数据混在一起


def _seed_data():
    from datetime import datetime, timedelta
    today = datetime.now()
    d0 = today.strftime("%Y-%m-%d")
    d1 = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    rows = [
        ("贵州茅台", "正面", 0.6, d1), ("贵州茅台", "负面", -0.4, d0),
        ("五粮液", "正面", 0.5, d1),
        ("宁德时代", "正面", 0.7, d1), ("宁德时代", "正面", 0.3, d0),
        ("比亚迪", "负面", -0.6, d0),
        ("未收录的公司ABC", "中性", 0.0, d1),
    ]
    for company, label, score, date in rows:
        insert_record({
            "raw_text": "t", "cleaned_text": "t", "source": "test", "date": date,
            "company": company, "sentiment_label": label, "sentiment_score": score,
            "need_alert": False, "tenant_id": TENANT,
        })


def test_resolve_industry():
    """行业归属解析：已收录公司归到对应行业，未收录归到兜底类目"""
    assert resolve_industry("贵州茅台") == "白酒"
    assert resolve_industry("宁德时代") == "新能源"
    assert resolve_industry("从没听说过的公司") == "其他行业"
    assert resolve_industry(None) == "其他行业"
    print("✅ test_resolve_industry 通过")


def test_company_window_stats():
    """按公司聚合近N天统计，条数/均分/负面数都正确"""
    stats = {s["company"]: s for s in get_company_window_stats(days=30, tenant_id=TENANT)}
    assert stats["贵州茅台"]["cnt"] == 2
    assert stats["贵州茅台"]["negative_cnt"] == 1
    assert abs(stats["贵州茅台"]["avg_score"] - 0.1) < 1e-6
    assert stats["宁德时代"]["cnt"] == 2
    assert stats["宁德时代"]["negative_cnt"] == 0
    print("✅ test_company_window_stats 通过：", stats["贵州茅台"], stats["宁德时代"])


def test_industry_heat_ranking():
    """行业热度榜：按行业正确聚合，total_cnt降序排列，未收录公司归入兜底行业"""
    ranking = get_industry_heat_ranking(days=30, tenant_id=TENANT)
    by_industry = {r["industry"]: r for r in ranking}

    assert by_industry["白酒"]["company_cnt"] == 2   # 茅台+五粮液
    assert by_industry["白酒"]["total_cnt"] == 3      # 2+1条
    assert by_industry["白酒"]["negative_cnt"] == 1
    assert by_industry["新能源"]["total_cnt"] == 2
    assert by_industry["其他行业"]["total_cnt"] == 1  # 未收录公司落到兜底类目，没有丢数据

    # 按舆情量（热度）降序：白酒(3) 排在 新能源(2) 前面
    industries_in_order = [r["industry"] for r in ranking]
    assert industries_in_order.index("白酒") < industries_in_order.index("新能源")
    print("✅ test_industry_heat_ranking 通过：", ranking)


def test_multi_company_trend_alignable():
    """多公司对比的底层数据：每家公司的日趋势可以按日期对齐叠加（Streamlit用pd.concat做这件事）"""
    import pandas as pd
    trends = {}
    for c in ["贵州茅台", "宁德时代"]:
        t = get_company_daily_trend(c, days=30, tenant_id=TENANT)
        assert len(t) >= 1
        trends[c] = pd.DataFrame(t)[["date", "avg_score"]].set_index("date")

    merged = pd.concat([trends[c].rename(columns={"avg_score": c}) for c in trends], axis=1)
    assert set(merged.columns) == {"贵州茅台", "宁德时代"}
    print("✅ test_multi_company_trend_alignable 通过，合并后列：", list(merged.columns))


if __name__ == "__main__":
    _seed_data()

    test_resolve_industry()
    test_company_window_stats()
    test_industry_heat_ranking()
    test_multi_company_trend_alignable()

    print("\n全部验收测试通过 ✅")
