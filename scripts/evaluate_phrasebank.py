"""
用 HuggingFace financial_phrasebank 数据集评测系统情感分析准确率。

与 scripts/evaluate_sentiment.py 的区别：那边是内置的十几条手写样例，
这里是引入外部真实标注数据集（Malo et al. 2014，金融新闻句子+正/中/负标注），
规模可控（默认抽样100条），作为更可信的量化指标来源。

- 数据集：financial_phrasebank，本地 TXT 文件放在项目 data/ 目录下。
  文件格式：每行 "<sentence>@<label>"，label 为 positive/neutral/negative。
  默认使用 Sentences_AllAgree.txt（标注质量最高，约2264条）。
  可用 --file 切换为 Sentences_75Agree.txt / 66Agree / 50Agree。
- 评测流程：抽样 → 逐条调用 chains.analysis_chain.analyze_text（与生产情感分析
  完全同一套逻辑，也享受LLM缓存）→ 预测标签 vs 数据集标注 → 统计
    解析成功率   LLM输出能被结构化解析的比例（解析失败走"中性"兜底，不算命中）
    情感准确率   预测标签==标注标签的比例（核心指标）
    混淆矩阵     按标注/预测的 负面-中性-正面 交叉计数，便于看错在哪个方向
- 失败可见：没解析成功的那条会把真实原因（异常类型+消息）打出来，最后汇总失败原因分布。
  "连续3条失败且零成功"会直接终止并提示去查 .env（Key/接入地址/模型名），
  避免整轮100条都撞在同一个配置错误上白花额度。
- 只评测不写库：本脚本不调用 pipeline_graph，不会把评测数据写进 SQLite/向量库，
  不污染生产语料和 RAG 检索结果。
- 数据源加载：自动识别 data/ 目录下的 Sentences_*.txt 文件，零配置。

运行方式：
    python scripts/evaluate_phrasebank.py            # 默认用 AllAgree，抽100条
    python scripts/evaluate_phrasebank.py --n 500    # 换抽样量
    python scripts/evaluate_phrasebank.py --file Sentences_75Agree.txt  # 换文件
    python scripts/evaluate_phrasebank.py --dry-run  # 只加载并打印抽样样本，不调LLM
前置：已按 APIKey配置指南.md 配置好 LLM API Key；已将数据集放入 data/ 目录。
"""
import argparse
import glob
import os
import random
import sys

# 项目根目录 = scripts/ 的上一级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Windows 控制台默认GBK，脚本输出中文容易乱码，强制UTF-8（非Windows环境自动跳过）
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from chains.analysis_chain import analyze_text, FALLBACK_REASON_PREFIX

# data 目录路径
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 数据集的原始标签 → 系统情感标签 的映射
_LABEL_MAP = {
    "positive": "正面",
    "neutral":  "中性",
    "negative": "负面",
}


def _norm_label(label) -> str:
    """将字符串标签（positive/neutral/negative）统一成系统的中文标签"""
    if isinstance(label, str):
        return _LABEL_MAP.get(label.strip().lower(), "未知")
    return "未知"


def _discover_files() -> list[str]:
    """扫描 data/ 目录下所有 Sentences_*.txt 文件"""
    pattern = os.path.join(DATA_DIR, "Sentences_*.txt")
    files = sorted(glob.glob(pattern))
    return files


def _select_file(preferred: str | None) -> str | None:
    """
    选择要加载的文件：
    1. 如果指定了 --file，按文件名匹配（支持全称或简称如 75Agree）
    2. 否则优先选 Sentences_AllAgree.txt
    3. 再否则取字母序第一个
    """
    files = _discover_files()
    if not files:
        return None

    if preferred:
        # 精确匹配
        for f in files:
            if os.path.basename(f) == preferred:
                return f
        # 模糊匹配：--file 75Agree → Sentences_75Agree.txt
        for f in files:
            if preferred.replace(".txt", "") in os.path.basename(f):
                return f

    # 默认优先 AllAgree
    for f in files:
        if "AllAgree" in os.path.basename(f):
            return f

    return files[0]


def load_phrasebank(file_path: str) -> list[tuple[str, str]]:
    """
    从本地 TXT 文件加载数据集。
    文件格式：每行 "<sentence>@<label>"，编码为 iso-8859-1。
    返回 [(文本, 标准情感标签), ...]
    """
    sentences, labels = [], []
    with open(file_path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 用 rsplit 防止句子本身包含 @ 符号
            sentence, label = line.rsplit("@", 1)
            sentences.append(sentence.strip())
            labels.append(_norm_label(label.strip()))

    items = list(zip(sentences, labels))
    print(f"[evaluate_phrasebank] 已加载：{os.path.basename(file_path)}，共 {len(items)} 条")
    return items


def compute_metrics(pairs: list[tuple[str, str, bool]]) -> tuple[float, float, dict, int]:
    """纯函数，方便单独测试。
    pairs: [(标注标签, 预测标签, 是否解析成功), ...]
    返回 (解析成功率, 情感准确率, 混淆矩阵, 样本数)。
    """
    labels = ["负面", "中性", "正面"]
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, {}, 0

    parsed = sum(1 for _, _, ok in pairs if ok)
    hits = sum(1 for exp, pred, ok in pairs if ok and pred == exp)

    cm = {e: {p: 0 for p in labels} for e in labels}
    for exp, pred, ok in pairs:
        if ok and exp in cm and pred in cm:
            cm[exp][pred] += 1

    return parsed / n, hits / n, cm, n


def main():
    parser = argparse.ArgumentParser(description="用 financial_phrasebank 评测情感分析准确率")
    parser.add_argument("--file", type=str, default=None,
                        help="选择数据集文件（如 Sentences_75Agree.txt 或简写 75Agree），"
                             "默认自动选 Sentences_AllAgree.txt")
    parser.add_argument("--n", type=int, default=100, help="抽样条数（默认100）")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子，固定便于复现")
    parser.add_argument("--dry-run", action="store_true", help="只加载并打印抽样样本，不调用LLM（零成本）")
    args = parser.parse_args()

    # ---- 自动发现并选择文件 ----
    selected = _select_file(args.file)

    if selected is None:
        print(f"[evaluate_phrasebank] 在 {DATA_DIR} 下未找到任何 Sentences_*.txt 文件。")
        print("")
        print("请按以下步骤操作：")
        print("  1. 下载数据集压缩包：")
        print("     https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip")
        print("  2. 解压后你会看到 4 个文件：")
        print("     Sentences_AllAgree.txt  (2264条, 100%标注一致, 推荐)")
        print("     Sentences_75Agree.txt   (3453条, ≥75%一致)")
        print("     Sentences_66Agree.txt   (4217条)")
        print("     Sentences_50Agree.txt   (4846条)")
        print(f"  3. 把它们全部放进项目的 data/ 目录：{DATA_DIR}/")
        print("  4. 重新运行本脚本即可，无需任何额外参数。")
        return

    # 列出可选文件（方便用户知道还有哪些可用）
    available = _discover_files()
    if len(available) > 1:
        others = [os.path.basename(f) for f in available if f != selected]
        print(f"[evaluate_phrasebank] 检测到 {len(available)} 个数据集文件，"
              f"当前使用：{os.path.basename(selected)}，"
              f"可选：{', '.join(others)}（用 --file 切换）")

    items = load_phrasebank(selected)
    if not items:
        return

    n = min(args.n, len(items))
    sampled = [items[i] for i in random.Random(args.seed).sample(range(len(items)), n)]
    print(f"抽样 {n} 条（seed={args.seed}），标注分布：")
    dist = {}
    for _, label in sampled:
        dist[label] = dist.get(label, 0) + 1
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(dist.items())))

    if args.dry_run:
        print("\n[--dry-run] 不调用LLM，前5条样本如下：")
        for text, label in sampled[:5]:
            print(f"  [{label}] {text[:60]}")
        return

    print(f"\n评测开始：{n} 条，调用 analyze_text（每条一次LLM调用，命中缓存不重复计费）…\n")
    pairs = []
    fail_reasons = {}
    consecutive_failures = 0
    for i, (text, exp_label) in enumerate(sampled, 1):
        try:
            pred = analyze_text(text)
        except Exception as e:
            pred = {"sentiment_label": "中性", "sentiment_score": 0.0,
                    "company": "", "event_type": "其他",
                    "reason": f"{FALLBACK_REASON_PREFIX}（未捕获异常：{type(e).__name__}: {e}）"}

        pred_label = pred.get("sentiment_label", "中性")
        reason = str(pred.get("reason", ""))
        # 兜底结果的reason带固定前缀（见 chains/analysis_chain.py），据此区分
        # "真的判断成中性"和"这条根本没调通/没解析出来"
        is_parsed = not reason.startswith(FALLBACK_REASON_PREFIX)
        pairs.append((exp_label, pred_label, is_parsed))

        mark = "✓" if (is_parsed and pred_label == exp_label) else "✗"
        print(f"[{i:>3}] 标注={exp_label} 预测={pred_label}({pred.get('sentiment_score')}) {mark} | {text[:40]}…")
        if is_parsed:
            consecutive_failures = 0
        else:
            print(f"      ↳ 本条未解析成功：{reason[:200]}")
            fail_reasons[reason[:200]] = fail_reasons.get(reason[:200], 0) + 1
            consecutive_failures += 1
            # 连续失败且至今零成功，基本是Key/接入地址/模型名这类系统性问题：
            # 继续跑只会把剩下样本全撞在同一个错误上，还白花额度，直接停下来说清楚
            if consecutive_failures >= 3 and not any(ok for _, _, ok in pairs):
                print("\n[提前终止] 连续3条都调用/解析失败，且没有任何一条成功，"
                      "大概率是LLM接入配置问题，不是评测数据的问题。")
                print(f"最后一条的失败原因：{reason}")
                print("排查顺序：1) .env 的 LLM_PROVIDER 和对应Key 是否匹配且有效；")
                print("          2) OPENAI_BASE_URL 是否指向Key所属服务商"
                      "（用DeepSeek的Key就要填 https://api.deepseek.com/v1）；")
                print("          3) OPENAI_MODEL 是否为该服务商真实存在的模型名。")
                sys.exit(1)

    parse_rate, acc, cm, total = compute_metrics(pairs)

    print("\n" + "=" * 60)
    print("评测汇总")
    print("=" * 60)
    print(f"样本数       : {total} 条（{os.path.basename(selected)} 抽样）")
    print(f"解析成功率   : {parse_rate:.1%}")
    print(f"情感准确率   : {acc:.1%}")
    print("\n混淆矩阵（行=标注，列=预测）：")
    labels = ["负面", "中性", "正面"]
    header = f"{'标注预测':<10}" + "".join(f"{l:>8}" for l in labels)
    print(header)
    for e in labels:
        row = f"{e:<10}" + "".join(f"{cm.get(e, {}).get(p, 0):>8}" for p in labels)
        print(row)
    if fail_reasons:
        print("\n未解析成功的原因分布（按出现次数排序）：")
        for reason, cnt in sorted(fail_reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {cnt:>4} 次：{reason}")
    print("=" * 60)
    print("说明：financial_phrasebank 的 neutral 同时涵盖中性/混合情绪，且为英文文本，")
    print("     准确率会低于中文测试集，可与 scripts/evaluate_sentiment.py 的结果对照参考。")


if __name__ == "__main__":
    main()
