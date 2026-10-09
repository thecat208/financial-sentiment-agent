# -*- coding: utf-8 -*-
"""用校准后的规则(is_refusal/check_numbers)对已存评测答案离线重算指标并重新生成报告。
复用 eval_hallu_chroma 已建索引与 eval_hallu_sentiment.db（不重建、不调LLM），跑完即删。"""
import io
import json
import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

EVAL_DIR = os.path.join(PROJECT_ROOT, "scripts", "eval_results")
SRC_CHROMA = os.path.join(EVAL_DIR, "eval_hallu_chroma")
RO_CHROMA = os.path.join(EVAL_DIR, "eval_hallu_chroma_ro")
# 原目录可能被残留进程独占锁，统一用副本
if os.path.exists(RO_CHROMA):
    shutil.rmtree(RO_CHROMA)
shutil.copytree(SRC_CHROMA, RO_CHROMA)
os.environ["SQLITE_DB_PATH"] = os.path.join(EVAL_DIR, "eval_hallu_sentiment.db")
os.environ["CHROMA_PERSIST_DIR"] = RO_CHROMA
os.environ["CHROMA_ACTIVE_POINTER"] = os.path.join(EVAL_DIR, "eval_hallu_active.json")
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from hallucination_corpus import QUESTIONS  # noqa: E402
from config.settings import settings  # noqa: E402
from retrieval.retriever import hybrid_retrieve  # noqa: E402
from core.faithfulness import is_refusal, check_numbers, check_companies, check_citations  # noqa: E402

detail_path = os.path.join(EVAL_DIR, "hallucination_detail.json")
data = json.load(open(detail_path, encoding="utf-8"))
det = data["detail"]

print("== 重新检索 30 题（不调LLM）==")
docs_map = {}
for i, q in enumerate(QUESTIONS, 1):
    docs_map[q["id"]] = hybrid_retrieve(q["q"], k=settings.RETRIEVE_TOP_K)
print(f"   检索完成 {len(docs_map)} 题")

# ---- 用校准规则重算各标记（与 verify_answer 同口径：问题原文并入回查范围）----
changes = {"refusal": [], "numbers": []}
for q in QUESTIONS:
    qid = q["id"]
    docs = docs_map[qid]
    docs_text = "\n".join(d.page_content for d in docs) + "\n" + q["q"]
    n_docs = len(docs)
    for mode in ("baseline", "verified"):
        r = det[qid][mode]
        ans = r["answer"]
        new_refusal = is_refusal(ans)
        if new_refusal != r["refusal"]:
            changes["refusal"].append((qid, mode, r["refusal"], new_refusal))
        r["refusal"] = new_refusal
        new_unfaith = check_numbers(ans, docs_text)
        if new_unfaith != r["unfaithful_numbers"]:
            changes["numbers"].append((qid, mode, r["unfaithful_numbers"], new_unfaith))
        r["unfaithful_numbers"] = new_unfaith
        r["unknown_companies"] = check_companies(ans, docs_text)
        cite = check_citations(ans, n_docs)
        r["invalid_refs"] = cite["invalid_refs"]
        if mode == "verified":
            r["total_refs"] = cite["total_refs"]
            layers = []
            if cite["invalid_refs"]:
                layers.append("citation_invalid")
            if r["unfaithful_numbers"]:
                layers.append("number_unfaithful")
            if r["unknown_companies"]:
                layers.append("company_unknown")
            if r.get("llm_supported") is False:
                layers.append("llm_not_supported")
            r["layers"] = layers
            r["risk"] = bool(layers)

print("\n== 重算变化明细 ==")
print(f"refusal 变化 {len(changes['refusal'])} 处: " +
      ", ".join(f"{c[0]}/{c[1][:4]}:{int(c[2])}->{int(c[3])}" for c in changes["refusal"]))
print(f"unfaithful_numbers 变化 {len(changes['numbers'])} 处:")
for c in changes["numbers"]:
    print(f"   {c[0]} {c[1]}: {c[2]} -> {c[3]}")

# ---- 汇总（与 eval_hallucination.py 的 agg 公式一致）----
def agg(mode, qtype):
    rows = [det[q["id"]][mode] for q in QUESTIONS if q["type"] == qtype]
    n = len(rows)
    return {
        "rule_refusal_rate": sum(r["refusal"] for r in rows) / n,
        "rule_fabrication_rate": sum(bool(r["unfaithful_numbers"] or r["unknown_companies"]) for r in rows) / n,
        "judge_refusal_rate": sum(
            (r["judge"]["corpus_has_answer"] is False and not r["judge"]["fabricated"]) for r in rows) / n,
        "judge_fabrication_rate": sum(bool(r["judge"]["fabricated"]) for r in rows) / n,
        "judge_faithful_rate": sum(r["judge"]["faithful"] is True for r in rows) / n,
        "n": n,
    }

summary = {}
for mode in ("baseline", "verified"):
    in_rows = [det[q["id"]][mode] for q in QUESTIONS if q["type"] == "in"]
    entry = {"out": agg(mode, "out"), "in": agg(mode, "in")}
    if mode == "verified":
        n = len(in_rows)
        entry["in_extra_citation_coverage"] = sum(r["total_refs"] > 0 for r in in_rows) / n
        entry["in_extra_avg_refs"] = sum(r["total_refs"] for r in in_rows) / n
        entry["in_extra_risk_rate"] = sum(r["risk"] for r in in_rows) / n
    summary[mode] = entry
data["summary"] = summary

# ---- 重新生成报告 ----
b, v = summary["baseline"]["out"], summary["verified"]["out"]
bi, vi = summary["baseline"]["in"], summary["verified"]["in"]
meta = data["meta"]
md = []
md.append("# 幻觉评测报告（README问题③：RAG问答幻觉工程化控制）\n")
md.append(f"- 日期：{meta['date']}（2026-10-05 指标口径校准后离线重算）")
md.append(f"- 语料：{meta['docs']} 篇（与问题②评测同源），问题：{meta['questions']} 条（in 15 / out 15）")
md.append(f"- LLM：{meta['llm_model']}（真实API调用，全程 {meta['llm_calls']} 次，耗时 {meta['cost_seconds']}s）")
md.append("- 对比：baseline（原版Prompt，无校验） vs verified（引用标注+三层程序化校验+受限重生成）\n")
md.append("## 总体结果\n")
md.append("| 指标 | baseline | verified | 说明 |")
md.append("|---|---|---|---|")
md.append(f"| out拒答率（规则） | {b['rule_refusal_rate']:.1%} | {v['rule_refusal_rate']:.1%} | 越高越好 |")
md.append(f"| out拒答率（judge） | {b['judge_refusal_rate']:.1%} | {v['judge_refusal_rate']:.1%} | 越高越好 |")
md.append(f"| out编造率（规则） | {b['rule_fabrication_rate']:.1%} | {v['rule_fabrication_rate']:.1%} | 越低越好 |")
md.append(f"| out编造率（judge） | {b['judge_fabrication_rate']:.1%} | {v['judge_fabrication_rate']:.1%} | 越低越好 |")
md.append(f"| out忠实率（judge） | {b['judge_faithful_rate']:.1%} | {v['judge_faithful_rate']:.1%} | 越高越好 |")
md.append(f"| in正常回答率（规则） | {1 - bi['rule_refusal_rate']:.1%} | {1 - vi['rule_refusal_rate']:.1%} | 校验不应引入过度拒答 |")
md.append(f"| in忠实率（judge） | {bi['judge_faithful_rate']:.1%} | {vi['judge_faithful_rate']:.1%} | judge判定回答忠于资料 |")
md.append(f"| in引用标注覆盖率 | — | {summary['verified']['in_extra_citation_coverage']:.1%} | verified专属能力 |")
md.append(f"| in校验风险率 | — | {summary['verified']['in_extra_risk_rate']:.1%} | 库内问题的误报水平 |")
md.append("""
## 指标口径校准说明（2026-10-05）

首跑后发现并修正两处规则缺陷，上表为校准后对全部已存答案**离线重算**的结果（LLM回答与judge结果不变）：

1. **拒答判定校准**：旧口径"全文含'资料有限/未提及'等词即拒答"会把库内
   "先答已知、再说明边界"的合格回答误判为拒答（首跑误判10/15）；校准后仅在
   回答开头出现强拒答语（"无法回答/没有关于/无法判断…"）才判拒答。
2. **数字忠实性校准**：旧口径把回答补写的"来源：XX，2026-09-25"中的日期当事实
   数字回查正文，格式不一致导致首跑 verified 风险 5/5 全为日期误报；校准后日期
   不参与数字核对（由 LLM 蕴含层兜底），且问题原文并入回查范围（复述问题中的
   数字不算编造）。
3. 已知残留：首跑 verified 阶段 5 条库内回答因日期误报触发过重生成并带警告头，
   原始答案文本保留在 detail.json；若以校准后逻辑重跑，这些不会触发。
""")
md.append("## 库外问题逐条明细（幻觉主战场）\n")
md.append("| id | 问题 | base拒答 | base编造(规则) | base编造(judge) | ver拒答 | ver编造(规则) | ver编造(judge) |")
md.append("|---|---|---|---|---|---|---|---|")
for q in QUESTIONS:
    if q["type"] != "out":
        continue
    rb, rv = det[q["id"]]["baseline"], det[q["id"]]["verified"]
    md.append(
        f"| {q['id']} | {q['q'][:30]} "
        f"| {'是' if rb['refusal'] else '否'} "
        f"| {','.join((rb['unfaithful_numbers'] + rb['unknown_companies'])[:2]) or '无'} "
        f"| {'是' if rb['judge']['fabricated'] else '否'} "
        f"| {'是' if rv['refusal'] else '否'} "
        f"| {','.join((rv['unfaithful_numbers'] + rv['unknown_companies'])[:2]) or '无'} "
        f"| {'是' if rv['judge']['fabricated'] else '否'} |")

with open(detail_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
print(f"\n明细已重写: {detail_path}")

report_path = os.path.join(EVAL_DIR, "hallucination_eval_report.md")
io.open(report_path, "w", encoding="utf-8", newline="").write("\n".join(md))
print(f"报告已重写: {report_path}")
print("\n== 校准后关键指标 ==")
print(f"out拒答率(规则): base {b['rule_refusal_rate']:.0%} / ver {v['rule_refusal_rate']:.0%}")
print(f"in正常回答率(规则): base {1-bi['rule_refusal_rate']:.1%} / ver {1-vi['rule_refusal_rate']:.1%}")
print(f"in忠实率(judge): base {bi['judge_faithful_rate']:.1%} / ver {vi['judge_faithful_rate']:.1%}")
print(f"in校验风险率(校准后): {summary['verified']['in_extra_risk_rate']:.1%}")
print(f"in引用覆盖率: {summary['verified']['in_extra_citation_coverage']:.1%}")
