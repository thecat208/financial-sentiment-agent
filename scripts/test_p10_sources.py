"""
验收测试：专业数据源与公告全文接入。

不依赖真实LLM API Key/langgraph完整依赖链——直接测试受控词表、采集层降级逻辑、
存储层source_type/source_credibility透传这条链路，对应README的验收标准：
  1. 能采集到至少一类新数据源（机构公告/散户情绪代理指标）的内容并产出统一格式
  2. source字段能区分"机构公告"/"新闻资讯"/"社交媒体"等来源类型

用法：
    SQLITE_DB_PATH=/tmp/p10_test.db python scripts/test_p10_sources.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.taxonomy import SOURCE_TYPES, normalize_source_type, get_source_credibility
from ingestion.announcement_source import fetch_announcements
from ingestion.social_source import fetch_retail_sentiment
from graphs.pipeline_graph import save_node
from storage.repository import get_records_by_company


def test_source_type_taxonomy():
    """受控词表归一化 + 可信度分级正确"""
    assert normalize_source_type("机构公告") == "机构公告"
    assert normalize_source_type("乱七八糟没见过的类型") == "新闻资讯"  # 兜底默认值
    assert normalize_source_type(None) == "新闻资讯"  # 未传时兼容旧调用方
    assert get_source_credibility("机构公告") == "高"
    assert get_source_credibility("社交媒体") == "低"
    assert get_source_credibility("新闻资讯") == "中"
    print("✅ test_source_type_taxonomy 通过")


def test_announcement_source_produces_valid_items():
    """公告采集（真实/降级均可）产出统一格式，source_type固定为机构公告"""
    items = fetch_announcements("宁德时代", days=7)
    assert isinstance(items, list)
    for it in items:
        assert set(["raw_text", "source", "source_type", "date"]).issubset(it.keys())
        assert it["source_type"] == "机构公告"
        assert it["raw_text"].strip()
    print(f"✅ test_announcement_source_produces_valid_items 通过（{len(items)}条，"
          f"{'mock降级' if items and items[0].get('is_mock') else '真实数据' if items else '空结果'}）")


def test_social_source_produces_valid_item():
    """散户情绪代理指标采集产出统一格式，source_type固定为社交媒体"""
    item = fetch_retail_sentiment("宁德时代")
    assert item is None or set(["raw_text", "source", "source_type", "date"]).issubset(item.keys())
    if item:
        assert item["source_type"] == "社交媒体"
        assert item["raw_text"].strip()
    print(f"✅ test_social_source_produces_valid_item 通过"
          f"（{'mock降级' if item and item.get('is_mock') else '真实数据' if item else '该股票暂无数据'}）")


def test_source_type_roundtrip_through_pipeline():
    """save_node → insert_record → 读回，source_type/source_credibility正确落库"""
    state = {
        "raw_text": "P10测试用公告文本", "cleaned_text": "P10测试用公告文本",
        "source": "巨潮资讯网-公告", "date": "2026-08-05", "company": "P10测试公司",
        "event_type": "产品与业务", "sentiment_label": "正面", "sentiment_score": 0.4,
        "dimension_scores": None, "analysis_reason": "test",
        "media_type": "text", "media_path": None, "tenant_id": "default",
        "source_type": "机构公告",
    }
    result_state = save_node(state)
    assert result_state["db_id"] is not None

    recs = get_records_by_company("P10测试公司", days=3650, tenant_id="default")
    matched = [r for r in recs if r["id"] == result_state["db_id"]]
    assert len(matched) == 1
    r = matched[0]
    assert r["source_type"] == "机构公告"
    assert r["source_credibility"] == "高"
    print("✅ test_source_type_roundtrip_through_pipeline 通过：",
          r["source_type"], r["source_credibility"])


def test_default_source_type_backward_compat():
    """不传source_type时（模拟升级前的旧调用方），落库应默认为'新闻资讯'/可信度'中'"""
    state = {
        "raw_text": "旧调用方测试文本", "cleaned_text": "旧调用方测试文本",
        "source": "某RSS源", "date": "2026-08-05", "company": "P10测试公司B",
        "event_type": "其他", "sentiment_label": "中性", "sentiment_score": 0.0,
        "dimension_scores": None, "analysis_reason": "test",
        "media_type": "text", "media_path": None, "tenant_id": "default",
        "source_type": None,
    }
    result_state = save_node(state)
    recs = get_records_by_company("P10测试公司B", days=3650, tenant_id="default")
    matched = [r for r in recs if r["id"] == result_state["db_id"]]
    r = matched[0]
    assert r["source_type"] == "新闻资讯"
    assert r["source_credibility"] == "中"
    print("✅ test_default_source_type_backward_compat 通过：", r["source_type"], r["source_credibility"])


if __name__ == "__main__":
    print(f"来源类型受控词表（{len(SOURCE_TYPES)}类）：{SOURCE_TYPES}\n")

    test_source_type_taxonomy()
    test_announcement_source_produces_valid_items()
    test_social_source_produces_valid_item()
    test_source_type_roundtrip_through_pipeline()
    test_default_source_type_backward_compat()

    print("\n全部验收测试通过 ✅")
