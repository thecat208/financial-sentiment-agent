"""
验收测试：SaaS化基础能力（用量统计/审计留痕）。

用法：
    SQLITE_DB_PATH=/tmp/p14_test.db python scripts/test_p14_saas.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TENANT = "p14_test_tenant"


def test_usage_context_roundtrip():
    """set_usage_context设置后，get_usage_context能读到同样的值（同步调用链内）"""
    from core.usage_context import set_usage_context, get_usage_context

    set_usage_context("tenant_x", "analyze")
    assert get_usage_context() == ("tenant_x", "analyze")

    set_usage_context(None, None)  # 传None应该落回默认值，不应该把None原样存进去
    assert get_usage_context() == ("default", "unknown")
    print("✅ test_usage_context_roundtrip 通过")


def test_record_and_query_llm_usage():
    from storage.usage_tracker import record_llm_usage, get_usage_stats

    record_llm_usage(TENANT, operation="analyze", model="claude-test", input_tokens=100, output_tokens=50)
    record_llm_usage(TENANT, operation="analyze", model="claude-test", input_tokens=200, output_tokens=80)
    record_llm_usage(TENANT, operation="qa", model="claude-test", input_tokens=300, output_tokens=120)

    stats = get_usage_stats(TENANT, days=30)
    assert stats["llm_calls"] == 3
    assert stats["input_tokens"] == 600
    assert stats["output_tokens"] == 250
    assert stats["total_tokens"] == 850

    by_op = {row["operation"]: row for row in stats["by_operation"]}
    assert by_op["analyze"]["calls"] == 2
    assert by_op["qa"]["calls"] == 1
    print("✅ test_record_and_query_llm_usage 通过：", stats["total_tokens"], "tokens,", stats["llm_calls"], "次调用")


def test_record_and_query_api_calls():
    from storage.usage_tracker import record_api_call, get_usage_stats

    record_api_call(TENANT, "/api/v1/companies")
    record_api_call(TENANT, "/api/v1/companies")
    record_api_call(TENANT, "/api/v1/industry/ranking")

    stats = get_usage_stats(TENANT, days=30)
    assert stats["api_calls"] == 3
    by_endpoint = {row["endpoint"]: row["calls"] for row in stats["api_calls_by_endpoint"]}
    assert by_endpoint["/api/v1/companies"] == 2
    assert by_endpoint["/api/v1/industry/ranking"] == 1
    print("✅ test_record_and_query_api_calls 通过")


def test_manual_override_writes_audit_log():
    """人工复核覆盖预警判定，应该正确改数据、且自动记一条带操作人/时间的审计日志"""
    from storage.repository import insert_record, manual_override_alert
    from storage.audit_log import get_audit_log

    rid = insert_record({
        "raw_text": "t", "cleaned_text": "t", "source": "test", "date": "2026-09-11",
        "company": "P14测试公司", "sentiment_label": "负面", "sentiment_score": -0.6,
        "need_alert": True, "tenant_id": TENANT,
    })

    manual_override_alert(rid, is_valid=True, tenant_id=TENANT, user_id="测试操作员", note="人工确认属实")

    records = get_audit_log(TENANT, days=30, action="alert_manual_override")
    matched = [r for r in records if r["target_id"] == str(rid)]
    assert len(matched) == 1
    assert matched[0]["user_id"] == "测试操作员"
    assert "人工确认属实" in matched[0]["detail"]
    assert matched[0]["created_at"]  # 时间戳非空
    print("✅ test_manual_override_writes_audit_log 通过：",
          matched[0]["user_id"], matched[0]["created_at"])


def test_audit_log_records_unknown_user_explicitly():
    """不传user_id时应该显式记为UNKNOWN_USER，而不是留空/静默跳过"""
    from storage.audit_log import record_audit, get_audit_log, UNKNOWN_USER

    record_audit(TENANT, action="test_action_no_user", target_type="x", target_id="1")
    records = get_audit_log(TENANT, days=30, action="test_action_no_user")
    assert records[0]["user_id"] == UNKNOWN_USER
    print(f"✅ test_audit_log_records_unknown_user_explicitly 通过（记为'{UNKNOWN_USER}'而非空值）")


def test_api_middleware_records_usage_and_excludes_health():
    """API中间件应该自动记录/api/*请求，且不把/health这类系统端点计入用量"""
    from fastapi.testclient import TestClient
    from api.main import app
    from storage.usage_tracker import get_usage_stats

    client = TestClient(app)
    tenant = "p14_middleware_test"

    client.get("/health")
    client.get("/health")
    client.get("/api/v1/companies", headers={"X-Tenant-Id": tenant})

    stats = get_usage_stats(tenant, days=30)
    assert stats["api_calls"] == 1  # 只有/api/v1/companies这一次，/health不计入
    print("✅ test_api_middleware_records_usage_and_excludes_health 通过")


def test_api_override_endpoint_accepts_non_ascii_user_id_in_body():
    """中文操作人姓名必须通过请求体传递（HTTP请求头仅支持ASCII，不能承载中文）"""
    from fastapi.testclient import TestClient
    from api.main import app
    from storage.repository import insert_record
    from storage.audit_log import get_audit_log

    client = TestClient(app)
    tenant = "p14_override_test"

    rid = insert_record({
        "raw_text": "t", "cleaned_text": "t", "source": "test", "date": "2026-09-11",
        "company": "覆盖测试公司", "sentiment_label": "负面", "sentiment_score": -0.6,
        "need_alert": True, "tenant_id": tenant,
    })

    r = client.post(
        f"/api/v1/alerts/{rid}/override",
        json={"is_valid": True, "note": "API人工确认", "user_id": "李四"},
        headers={"X-Tenant-Id": tenant},
    )
    assert r.status_code == 200

    records = get_audit_log(tenant, days=30, action="alert_manual_override")
    assert any(rec["user_id"] == "李四" for rec in records)
    print("✅ test_api_override_endpoint_accepts_non_ascii_user_id_in_body 通过")


if __name__ == "__main__":
    test_usage_context_roundtrip()
    test_record_and_query_llm_usage()
    test_record_and_query_api_calls()
    test_manual_override_writes_audit_log()
    test_audit_log_records_unknown_user_explicitly()
    test_api_middleware_records_usage_and_excludes_health()
    test_api_override_endpoint_accepts_non_ascii_user_id_in_body()

    print("\n全部验收测试通过 ✅")
