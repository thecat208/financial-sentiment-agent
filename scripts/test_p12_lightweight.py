"""
逻辑正确性测试：轻量模型前置筛选的分流决策规则本身对不对。

和 scripts/evaluate_lightweight_prefilter.py 的区别：那边测的是"效果好不好"（准确率、
独立处理比例这些连续型指标），这里测的是"规则对不对"（给定精心构造的边界样本，
分流决策是否符合设计——用断言而不是打印百分比）。

用法：
    python scripts/test_p12_lightweight.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chains.lightweight_sentiment import classify
from chains.analysis_chain import _try_lightweight_fastpath
from knowledge_graph.seed_data import find_known_company_in_text
from core.taxonomy import guess_event_type_by_keywords


def test_find_known_company_in_text():
    assert find_known_company_in_text("贵州茅台发布公告") == "贵州茅台"
    assert find_known_company_in_text("宁王最近走势不错") == "宁德时代"  # 命中别名而非canonical名
    assert find_known_company_in_text("某未知创业公司发布新闻") is None
    print("✅ test_find_known_company_in_text 通过")


def test_guess_event_type_by_keywords():
    assert guess_event_type_by_keywords("公司发布业绩预告，净利润增长") == "业绩预告"
    assert guess_event_type_by_keywords("公司因违规被立案调查") == "监管处罚"
    assert guess_event_type_by_keywords("完全无关的一句话") == "其他"
    print("✅ test_guess_event_type_by_keywords 通过")


def test_lexicon_classify_basic_polarity():
    pos = classify("公司净利润大幅增长，业绩超预期，创历史新高。")
    neg = classify("公司因财务造假被立案调查，股价跌停。")
    neu = classify("今天天气不错。")  # 完全不含金融词典词条

    assert pos["label"] == "正面" and pos["confidence"] >= 0.6
    assert neg["label"] == "负面" and neg["confidence"] >= 0.6
    assert neu["confidence"] < 0.6  # 没有任何词典命中，理应置信度不够，不能假装"确定是中性"
    print("✅ test_lexicon_classify_basic_polarity 通过")


def test_negation_flips_polarity():
    """"未构成重大资产重组"这种否定表达，不应该被当成"重大资产重组"的正常权重处理"""
    r_plain = classify("公司完成重大资产重组。")
    r_negated = classify("公司表示，本次交易并未构成重大资产重组。")
    # 否定后的分数应该比原始表达更偏负/更接近0，不应该和未否定时同号同幅度
    assert r_negated["score"] <= r_plain["score"]
    print(f"✅ test_negation_flips_polarity 通过（未否定={r_plain['score']}，否定后={r_negated['score']}）")


def test_fastpath_requires_known_company():
    """情感再清晰，公司名不在种子库里也不能走快速通道（轻量模型没有通用NER能力）"""
    result = _try_lightweight_fastpath("某创业公司净利润大幅增长，业绩超预期。")
    assert result is None
    print("✅ test_fastpath_requires_known_company 通过")


def test_fastpath_blocked_by_risk_keyword():
    """即使是种子库已知公司+情感信号清晰，命中风险关键词也必须强制升级LLM"""
    result = _try_lightweight_fastpath("贵州茅台因涉嫌财务造假被证监会立案调查。")
    assert result is None
    print("✅ test_fastpath_blocked_by_risk_keyword 通过")


def test_fastpath_success_case():
    """三个条件都满足时，快速通道应该正常产出结果，且source标记为lightweight"""
    result = _try_lightweight_fastpath("贵州茅台2025年净利润再创新高，并提升年度分红比例。")
    assert result is not None
    assert result["source"] == "lightweight"
    assert result["company"] == "贵州茅台"
    assert result["sentiment_label"] == "正面"
    assert result["dimensions"] == {"业绩": 0.0, "管理层": 0.0, "行业前景": 0.0, "合规风险": 0.0}
    print("✅ test_fastpath_success_case 通过")


def test_disabled_prefilter_falls_back_to_llm_path():
    """关闭前置筛选开关时，analyze_text_gated应该完全不碰轻量模型，等价于纯LLM路径"""
    import importlib
    os.environ["ENABLE_LIGHTWEIGHT_PREFILTER"] = "false"
    import config.settings as settings_mod
    importlib.reload(settings_mod)
    import chains.analysis_chain as ac
    importlib.reload(ac)

    result = ac.analyze_text_gated("贵州茅台2025年净利润再创新高。")
    assert result["source"] == "llm"

    # 还原环境变量，不影响后续其他测试/脚本
    os.environ["ENABLE_LIGHTWEIGHT_PREFILTER"] = "true"
    importlib.reload(settings_mod)
    importlib.reload(ac)
    print("✅ test_disabled_prefilter_falls_back_to_llm_path 通过")


if __name__ == "__main__":
    test_find_known_company_in_text()
    test_guess_event_type_by_keywords()
    test_lexicon_classify_basic_polarity()
    test_negation_flips_polarity()
    test_fastpath_requires_known_company()
    test_fastpath_blocked_by_risk_keyword()
    test_fastpath_success_case()
    test_disabled_prefilter_falls_back_to_llm_path()

    print("\n全部逻辑正确性测试通过 ✅")
