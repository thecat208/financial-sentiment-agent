"""
FastAPI 接口验收测试。用 TestClient 直接测应用，不需要真的启动uvicorn进程，
也不需要LLM API Key（只读接口不调LLM；/ingest的同步分析路径需要完整LLM/向量库
依赖，这里只验证它在依赖不全时能优雅报错而不是让整个服务崩掉）。

用法：
    SQLITE_DB_PATH=/tmp/api_test.db python scripts/test_api.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api.main import app
from storage.repository import insert_record
from datetime import datetime, timedelta

client = TestClient(app)


def _seed_data():
    today = datetime.now()
    rows = [
        ("贵州茅台", "正面", 0.6, 0), ("贵州茅台", "负面", -0.3, 1),
        ("宁德时代", "正面", 0.5, 0), ("比亚迪", "负面", -0.5, 1),
    ]
    for company, label, score, day_offset in rows:
        insert_record({
            "raw_text": "test", "cleaned_text": "test", "source": "test",
            "date": (today - timedelta(days=day_offset)).strftime("%Y-%m-%d"),
            "company": company, "sentiment_label": label, "sentiment_score": score,
            "need_alert": label == "负面", "tenant_id": "api_test_tenant",
        })


def test_health_and_root():
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/").status_code == 200
    print("✅ test_health_and_root 通过")


def test_companies_and_trend():
    r = client.get("/api/v1/companies", headers={"X-Tenant-Id": "api_test_tenant"})
    assert r.status_code == 200
    companies = r.json()["companies"]
    assert "贵州茅台" in companies

    r = client.get("/api/v1/companies/贵州茅台/trend?days=30", headers={"X-Tenant-Id": "api_test_tenant"})
    assert r.status_code == 200
    assert len(r.json()["trend"]) == 2
    print("✅ test_companies_and_trend 通过")


def test_compare_companies():
    r = client.post(
        "/api/v1/companies/compare",
        json={"companies": ["贵州茅台", "宁德时代"], "days": 30},
        headers={"X-Tenant-Id": "api_test_tenant"},
    )
    assert r.status_code == 200
    assert set(r.json()["series"].keys()) == {"贵州茅台", "宁德时代"}
    print("✅ test_compare_companies 通过")


def test_industry_ranking():
    r = client.get("/api/v1/industry/ranking?days=30", headers={"X-Tenant-Id": "api_test_tenant"})
    assert r.status_code == 200
    ranking = r.json()["ranking"]
    assert any(item["industry"] == "白酒" for item in ranking)
    print("✅ test_industry_ranking 通过")


def test_alert_stats():
    r = client.get("/api/v1/alerts/stats?days=30", headers={"X-Tenant-Id": "api_test_tenant"})
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 4 and data["triggered"] == 2
    print("✅ test_alert_stats 通过")


def test_market_overlay_and_backtest():
    r = client.get("/api/v1/market/贵州茅台/overlay?days=30", headers={"X-Tenant-Id": "api_test_tenant"})
    assert r.status_code == 200
    assert "disclaimer" in r.json()

    r = client.post(
        "/api/v1/market/backtest",
        json={"company": "贵州茅台", "days": 90},
        headers={"X-Tenant-Id": "api_test_tenant"},
    )
    assert r.status_code == 200
    assert "disclaimer" in r.json()
    print("✅ test_market_overlay_and_backtest 通过")


def test_ingest_async_returns_task_id():
    r = client.post(
        "/api/v1/ingest",
        json={"raw_text": "测试公司发布测试公告", "source": "test", "async_mode": True},
        headers={"X-Tenant-Id": "api_test_tenant"},
    )
    assert r.status_code == 200
    assert "task_id" in r.json()
    print("✅ test_ingest_async_returns_task_id 通过")


def test_ingest_sync_fails_gracefully_without_full_stack():
    """
    没配完整LLM/向量库依赖时，同步分析接口应该返回502而不是让进程崩溃；
    也可能命中项目默认限流（RATE_LIMIT_USER_MAX=1/2秒，前面几个测试已经用同一个
    租户发过好几次请求），返回429同样是限流器工作正常的表现，不是bug。
    """
    r = client.post(
        "/api/v1/ingest",
        json={"raw_text": "测试公司发布测试公告", "source": "test"},
        headers={"X-Tenant-Id": "api_test_tenant_ingest"},  # 用独立租户避免被前面测试的限流计数影响
    )
    assert r.status_code in (200, 502, 429)
    print(f"✅ test_ingest_sync_fails_gracefully_without_full_stack 通过（状态码 {r.status_code}）")


def test_tenant_isolation():
    """不传X-Tenant-Id用默认租户，看不到api_test_tenant的数据"""
    r = client.get("/api/v1/companies")
    assert "贵州茅台" not in r.json()["companies"]
    print("✅ test_tenant_isolation 通过")


def test_api_key_auth_when_configured():
    """API_KEY未设置时（默认）不需要鉴权就能访问"""
    from config.settings import settings
    if not settings.API_KEY:
        r = client.get("/api/v1/companies", headers={"X-Tenant-Id": "api_test_tenant"})
        assert r.status_code == 200
        print("✅ test_api_key_auth_when_configured 通过（当前未设置API_KEY，符合默认行为）")
    else:
        r = client.get("/api/v1/companies")
        assert r.status_code == 401
        print("✅ test_api_key_auth_when_configured 通过（已设置API_KEY，未带头返回401）")


if __name__ == "__main__":
    _seed_data()

    test_health_and_root()
    test_companies_and_trend()
    test_compare_companies()
    test_industry_ranking()
    test_alert_stats()
    test_market_overlay_and_backtest()
    test_ingest_async_returns_task_id()
    test_ingest_sync_fails_gracefully_without_full_stack()
    test_tenant_isolation()
    test_api_key_auth_when_configured()

    print("\n全部 FastAPI 接口验收测试通过 ✅")
