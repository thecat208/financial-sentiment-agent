"""
模型质量离线评测（情感分析任务）。运行方式：python scripts/evaluate_sentiment.py
前置：已按 APIKey配置指南.md 配置好 LLM API Key。

说明：
- 内置一个金融舆情小测试集（TEST_SET，含期望情感标签与期望公司），
  逐条调用 chains.analysis_chain.analyze_text 得到预测结果。
- 统计四类指标：
    parse_success_rate  解析成功率（LLM输出能被结构化解析的比例，失败会走"中性/解析失败"兜底）
    sentiment_accuracy  情感准确率（预测标签==期望标签的比例，是衡量专业性的核心指标）
    company_hit_rate    公司识别命中率（期望公司出现在预测company里）
    avg_score           平均情感分（观察分布是否合理，-1~1）
- 结果会打印明细表 + 汇总指标，可写入报告或用于前后对比（如微调前后）。
- 注意：这只是"情感分析"单任务的评测。RAG问答的"回答质量"建议再抽一批问答对，
  人工或LLM-judge按 相关度/引用准确/幻觉 打分（见 项目介绍与答辩指南.md 第5节）。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chains.analysis_chain import analyze_text

# (文本, 期望情感标签, 期望公司)
TEST_SET = [
    ("示例公司发布三季度财报，营收同比增长15%，市场反应积极。", "正面", "示例公司"),
    ("某公司因涉嫌信息披露违规被证监会立案调查，股价开盘跌停。", "负面", "某公司"),
    ("某公司发布公告，宣布将于下月召开年度股东大会。", "中性", "某公司"),
    ("某公司遭遇供应商断供，核心生产线被迫停产。", "负面", "某公司"),
    ("宁德时代与特斯拉签署长期供货协议，订单量大幅增长。", "正面", "宁德时代"),
    ("招商银行公布2025年半年度报告，净利润平稳增长。", "正面", "招商银行"),
    ("某公司董事长因涉嫌内幕交易被立案调查。", "负面", "某公司"),
    ("贵州茅台2025年净利润再创新高，并提升年度分红比例。", "正面", "贵州茅台"),
    ("某公司宣布更换董事会秘书，属常规人事变动。", "中性", "某公司"),
    ("某公司被曝产品质量问题，宣布召回大批产品。", "负面", "某公司"),
]


def main():
    if not TEST_SET:
        print("测试集为空，无法评估")
        return

    print(f"评测开始：共 {len(TEST_SET)} 条，调用 analyze_text ...\n")
    rows = []
    parse_ok = 0
    senti_hit = 0
    company_hit = 0

    for text, exp_label, exp_company in TEST_SET:
        try:
            pred = analyze_text(text)
        except Exception as e:
            pred = {"sentiment_label": "解析失败", "sentiment_score": 0.0, "company": "", "reason": str(e)}

        label = pred.get("sentiment_label")
        company = pred.get("company") or ""
        score = pred.get("sentiment_score")

        is_parsed = label not in ("解析失败", "") and score is not None
        if is_parsed:
            parse_ok += 1
        if label == exp_label:
            senti_hit += 1
        if company and exp_company and exp_company in str(company):
            company_hit += 1

        rows.append((text[:24], exp_label, label, score, exp_company, company))
        print(f"[{len(rows):>2}] 期望={exp_label} 预测={label}({score}) 公司={company or '-'} | {text[:24]}...")

    n = len(TEST_SET)
    print("\n" + "=" * 60)
    print("评测汇总")
    print("=" * 60)
    print(f"测试集规模             : {n} 条")
    print(f"解析成功率             : {parse_ok / n:.1%}  ({parse_ok}/{n})")
    print(f"情感准确率             : {senti_hit / n:.1%}  ({senti_hit}/{n})")
    print(f"公司识别命中率         : {company_hit / n:.1%}  ({company_hit}/{n})")
    print("=" * 60)
    print("说明：准确率高低与所选LLM型号/测试集难度有关；可更换模型或补充样本后对比。")


if __name__ == "__main__":
    main()
