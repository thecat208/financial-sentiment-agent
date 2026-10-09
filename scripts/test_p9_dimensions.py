"""
验收测试：多维度情感与标准化事件分类。

不依赖真实LLM API Key——直接测试"LLM结构化输出 -> 对外dict -> 落库 -> 读回"
这条链路的解析和存储逻辑是否正确，对应README的验收标准：
  1. 一条舆情能输出多维度分数（如 {"业绩": 0.6, "管理层": -0.3, "行业前景": 0.2}）
  2. 事件类型能归到预定义类目里而不是自由文本

用法：
    SQLITE_DB_PATH=/tmp/p9_test.db python scripts/test_p9_dimensions.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chains.analysis_chain import _finalize_result
from core.taxonomy import EVENT_CATEGORIES, DIMENSIONS, normalize_event_type
from storage.repository import insert_record, get_records_by_company, get_dimension_avg, get_event_type_distribution


def test_finalize_result_matches_matrix_signal():
    """模拟"业绩超预期但管理层被质疑"这种矛盾信号，验证维度分数不会被互相抵消"""
    raw_llm_output = {
        "sentiment_label": "中性",
        "sentiment_score": 0.1,
        "score_performance": 0.6,
        "score_management": -0.3,
        "score_industry": 0.2,
        "score_compliance": 0.0,
        "company": "示例公司",
        "event_type": "业绩预告",
        "reason": "业绩超预期但管理层被质疑挪用资金，业绩和管理层维度方向相反",
    }
    result = _finalize_result(raw_llm_output)

    assert result["dimensions"] == {"业绩": 0.6, "管理层": -0.3, "行业前景": 0.2, "合规风险": 0.0}, \
        f"维度分数不符合预期：{result['dimensions']}"
    assert result["event_type"] == "业绩预告"
    assert set(result["dimensions"].keys()) == set(DIMENSIONS)
    print("✅ test_finalize_result_matches_matrix_signal 通过")
    print("   ", result["dimensions"])


def test_event_type_normalization():
    """LLM偶尔跑出受控词表之外的类目时，能兜底归一化到"其他\""""
    assert normalize_event_type("并购重组") == "并购重组"
    assert normalize_event_type("这是个从没见过的乱七八糟类目") == "其他"
    assert normalize_event_type(None) == "其他"
    print("✅ test_event_type_normalization 通过")


def test_score_clipping():
    """维度分数超出[-1,1]范围时被裁剪，不会污染统计"""
    raw = {
        "sentiment_label": "负面", "sentiment_score": -0.8,
        "score_performance": -5.0,   # 故意超范围
        "score_management": 3.0,     # 故意超范围
        "score_industry": 0.0, "score_compliance": -0.9,
        "company": "示例公司", "event_type": "监管处罚", "reason": "test",
    }
    result = _finalize_result(raw)
    assert result["dimensions"]["业绩"] == -1.0
    assert result["dimensions"]["管理层"] == 1.0
    print("✅ test_score_clipping 通过")


def test_storage_roundtrip():
    """插入一条带多维度打分的记录，读回后dimension_scores应该是dict，不是JSON字符串"""
    record_id = insert_record({
        "raw_text": "测试原文", "cleaned_text": "示例公司业绩超预期但管理层被质疑",
        "source": "test", "date": "2026-08-01", "company": "示例公司",
        "event_type": "业绩预告",
        "sentiment_label": "中性", "sentiment_score": 0.1,
        "dimension_scores": {"业绩": 0.6, "管理层": -0.3, "行业前景": 0.2, "合规风险": 0.0},
        "analysis_reason": "test", "need_alert": False,
        "tenant_id": "default", "media_type": "text",
    })
    assert record_id is not None

    records = get_records_by_company("示例公司", days=3650, tenant_id="default")
    matched = [r for r in records if r["id"] == record_id]
    assert len(matched) == 1, "读回的记录数不对"
    r = matched[0]
    assert isinstance(r["dimension_scores"], dict), f"应该是dict，实际是{type(r['dimension_scores'])}"
    assert r["dimension_scores"]["业绩"] == 0.6
    assert r["event_type"] == "业绩预告"
    print("✅ test_storage_roundtrip 通过：", r["dimension_scores"])

    dim_avg = get_dimension_avg("示例公司", days=3650, tenant_id="default")
    assert dim_avg["业绩"] == 0.6
    print("✅ get_dimension_avg 通过：", dim_avg)

    dist = get_event_type_distribution(days=3650, company="示例公司", tenant_id="default")
    assert any(d["event_type"] == "业绩预告" for d in dist)
    print("✅ get_event_type_distribution 通过：", dist)


if __name__ == "__main__":
    print(f"受控事件类型词表（{len(EVENT_CATEGORIES)}类）：{EVENT_CATEGORIES}")
    print(f"情感维度词表（{len(DIMENSIONS)}类）：{DIMENSIONS}\n")

    test_finalize_result_matches_matrix_signal()
    test_event_type_normalization()
    test_score_clipping()
    test_storage_roundtrip()

    print("\n全部验收测试通过 ✅")
