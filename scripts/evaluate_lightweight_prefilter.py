"""
效果评测：轻量模型前置筛选。每次运行都重新计算，结果随环境和配置变化。

两部分测试：

Part A：轻量模型本身的情感判断质量
  用 data/Sentences_AllAgree.txt（financial phrasebank 英文原版，与
  scripts/evaluate_phrasebank.py 的 LLM 评测同源）seed=42 固定抽样 100 条，
  测轻量模型的整体情感准确率，可与 LLM 评测结果直接对比。
  注意：语料为英文，LIGHTWEIGHT_HF_MODEL 需配置为支持英文的模型
  （如 cardiffnlp/twitter-roberta-base-sentiment-latest，三分类）。

Part B：完整前置筛选流程的独立处理率
  用一份手工构造的小测试集（覆盖清晰正负面、模糊、含风险关键词、未知公司
  五种场景，标注期望路由结果和期望情感标签），测 analyze_text_gated() 的
  路由决策与独立处理时的情感判断；以"每条都调LLM"为基线，独立处理率即
  LLM调用量下降比例。

运行方式：
    python scripts/evaluate_lightweight_prefilter.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chains.lightweight_sentiment import classify
from chains.analysis_chain import _try_lightweight_fastpath
from config.settings import settings

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

_LABEL_MAP = {"positive": "正面", "neutral": "中性", "negative": "负面"}


def _norm_label(label: str) -> str:
    """归一化情感标签，需同时兼容英文（positive/neutral/negative）与中文（正面/中性/负面）两种格式。"""
    label = label.strip()
    if label in ("正面", "中性", "负面"):
        return label
    return _LABEL_MAP.get(label.lower(), "未知")


def _load_phrasebank(path: str) -> list[tuple[str, str]]:
    items = []
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        # phrasebank 原始文件非UTF-8（含Latin-1字符，如公司名中的重音字母），降级解码
        with open(path, encoding="latin-1") as f:
            lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        sentence, label = line.rsplit("@", 1)
        items.append((sentence.strip(), _norm_label(label)))
    return items


def part_a_sentiment_quality():
    print("=" * 70)
    print("Part A：轻量模型整体情感准确率（data/Sentences_AllAgree.txt，financial phrasebank）")
    print("=" * 70)

    # 中文翻译版 Sentences_100.txt 已不在 data/ 目录；Part A 改用与
    # evaluate_phrasebank.py（LLM 评测）同源的 AllAgree 英文原版抽样，
    # 两条评测链路数据口径一致，准确率可直接对比。
    # 注意：语料是英文，LIGHTWEIGHT_HF_MODEL 需配置为支持英文的模型
    # （如 cardiffnlp/twitter-roberta-base-sentiment-latest，三分类）；
    # 中文模型（如默认的 JD 评论模型）跑英文语料的结果无效。
    path = os.path.join(DATA_DIR, "Sentences_AllAgree.txt")
    if not os.path.exists(path):
        print(f"跳过：找不到数据集文件 {path}")
        return

    all_items = _load_phrasebank(path)
    sample_n = min(100, len(all_items))
    items = random.Random(42).sample(all_items, sample_n)  # 固定种子，结果可复现
    n = len(items)

    backend_seen = set()
    per_class = {}  # 标签 -> [判断正确数, 样本总数]
    overall_correct = 0

    for text, expected in items:
        r = classify(text)
        backend_seen.add(r["backend"])
        c = per_class.setdefault(expected, [0, 0])
        c[1] += 1
        if r["label"] == expected:
            overall_correct += 1
            c[0] += 1

    overall_acc = overall_correct / n if n else 0
    print(f"样本数         : {n}（AllAgree 全量 {len(all_items)} 条，seed=42 固定抽样）")
    print(f"实际使用的后端 : {'/'.join(sorted(backend_seen))}")
    print(f"整体情感准确率 : {overall_acc:.1%}  ({overall_correct}/{n})")
    for label in ("负面", "中性", "正面"):
        if label in per_class:
            ok, total = per_class[label]
            print(f"  {label}: {ok}/{total}")
    if backend_seen == {"unavailable"}:
        print("→ 警告：transformer 后端不可用（依赖未装/模型加载失败），本次结果无效。")
    print()


# Part B 测试集：手工构造，覆盖五种场景，标注期望路由结果(expected_route)
# 和期望情感标签(expected_label)。公司名都取自 knowledge_graph/seed_data.py 的种子库，
# 保证"已知公司"这个条件在这份测试集里是可控的。
PART_B_CASES = [
    # (文本, 期望路由: fastpath/llm, 期望情感标签)
    ("贵州茅台2025年净利润再创新高，并提升年度分红比例。", "fastpath", "正面"),
    ("宁德时代与特斯拉签署长期供货协议，订单量大幅增长。", "fastpath", "正面"),
    ("招商银行公布半年度报告，净利润同比增长，业绩超预期。", "fastpath", "正面"),
    ("五粮液发布公告，宣布提高年度分红比例，回购部分股份。", "fastpath", "正面"),
    ("中国平安评级上调，机构普遍看好后市表现。", "fastpath", "正面"),
    ("比亚迪第三季度营收大幅增长，净利润创历史新高。", "fastpath", "正面"),
    ("隆基绿能业绩预减，净利润同比大幅下滑。", "fastpath", "负面"),
    ("恒瑞医药部分产品遭遇集采降价，短期利空。", "fastpath", "负面"),
    ("迈瑞医疗评级下调，机构预期转为谨慎。", "llm", "负面"),  # "评级下调"命中风险词表，强制升级（不是置信度问题）
    ("比亚迪因涉嫌财务造假被证监会立案调查。", "llm", "负面"),  # "财务造假""立案调查"命中风险词表，强制升级
    ("中信证券被曝涉嫌违规，交易所已介入调查。", "fastpath", "负面"),  # "违规"命中负面词典但不在风险词表里，正常走快速通道
    ("东方财富宣布更换董事会秘书，属于常规人事变动。", "llm", "中性"),  # 无明确情感词，置信度不够
    ("工商银行召开年度股东大会，审议多项议案。", "llm", "中性"),  # 无明确情感词，置信度不够
    ("某科技公司发布新产品，市场反应热烈，订单大幅增长。", "llm", "正面"),  # 公司名不在种子库
    ("某新能源企业遭遇供应链危机，被迫停产。", "llm", "负面"),  # 公司名不在种子库
    ("特斯拉据传将扩大在华产能，但公司尚未正式回应。", "llm", "中性"),  # 信号模糊，"尚未"构成否定，混合信号
]


def part_b_full_gating():
    print("=" * 70)
    print("Part B：完整前置筛选流程 —— 独立处理率 / LLM调用量下降比例")
    print("=" * 70)

    n = len(PART_B_CASES)
    fastpath_cnt = 0
    route_correct = 0
    fastpath_label_correct = 0

    rows = []
    for text, expected_route, expected_label in PART_B_CASES:
        fast_result = _try_lightweight_fastpath(text)
        actual_route = "fastpath" if fast_result is not None else "llm"

        if actual_route == expected_route:
            route_correct += 1
        if actual_route == "fastpath":
            fastpath_cnt += 1
            if fast_result["sentiment_label"] == expected_label:
                fastpath_label_correct += 1

        rows.append((text[:28], expected_route, actual_route, expected_label,
                      fast_result["sentiment_label"] if fast_result else "-"))

    for text, exp_r, act_r, exp_l, act_l in rows:
        mark = "✓" if exp_r == act_r else "✗"
        print(f"  [{mark}] 期望路由={exp_r:<8} 实际路由={act_r:<8} "
              f"期望情感={exp_l} 实际情感={act_l:<3} | {text}")

    bypass_rate = fastpath_cnt / n
    route_acc = route_correct / n
    fastpath_label_acc = fastpath_label_correct / fastpath_cnt if fastpath_cnt else 0

    print()
    print(f"测试集规模                         : {n} 条")
    print(f"路由决策正确率（该走哪条路判断对了）: {route_acc:.1%}  ({route_correct}/{n})")
    print(f"轻量模型独立处理比例（bypass_rate） : {bypass_rate:.1%}  ({fastpath_cnt}/{n})")
    print(f"独立处理样本的情感判断准确率        : {fastpath_label_acc:.1%}  "
          f"({fastpath_label_correct}/{fastpath_cnt})")
    print()
    print(f"→ 以\"每条都调LLM\"为基线，本测试集上LLM调用量下降约 {bypass_rate:.1%}"
          f"（{fastpath_cnt}/{n} 条完全不需要调用LLM）。")
    print("  这个数字只代表这份16条测试集的情况，实际生产环境的比例取决于"
          "真实流量里\"清晰明确\"文本的占比，会随着词典/种子公司库的扩充而变化。")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="轻量前置筛选评测")
    parser.add_argument("--part", choices=["a", "b", "all"], default="all",
                        help="只跑 Part A（情感准确率）/ Part B（门控路由）/ 全部（默认）")
    args = parser.parse_args()

    print(f"当前配置：ENABLE_LIGHTWEIGHT_PREFILTER={settings.ENABLE_LIGHTWEIGHT_PREFILTER}, "
          f"BACKEND={settings.LIGHTWEIGHT_SENTIMENT_BACKEND}, "
          f"MODEL={settings.LIGHTWEIGHT_HF_MODEL}, "
          f"CONFIDENCE_THRESHOLD={settings.LIGHTWEIGHT_CONFIDENCE_THRESHOLD}\n")

    if args.part in ("a", "all"):
        part_a_sentiment_quality()
    if args.part in ("b", "all"):
        part_b_full_gating()
