"""
幻觉评测：RAG问答的 幻觉拒答率 + 事实一致性，baseline vs verified 对比。

对应 README 6.1 问题3（LLM幻觉只有Prompt软约束，无程序化校验）。

评测设计：
- 语料：eval_corpus.py 的104篇（灌库+向量索引，与问题②评测同一套环境隔离方案）
- 问题：hallucination_corpus.py 30条（in库内可回答15 / out库外不可回答15）
- 对比两种模式（同一进程内切换，检索结果共用保证公平）：
  baseline：QA_VERIFY=0，原版prompt（历史行为）
  verified：QA_VERIFY=1，引用标注prompt + faithfulness三层校验 + 受限重生成
- 指标：
  out库外问题：拒答率（规则判定 + LLM judge双口径）、编造率（规则 + judge）
  in库内问题：正常回答率（检测校验是否引入"过度拒答"副作用）、引用标注覆盖
- judge：deepseek-v4-flash 对每个回答做一次"资料是否支持"判定（60次调用）

运行：python scripts/eval_hallucination.py  （需真实 DeepSeek API Key，约135+次调用）
产出：scripts/eval_results/hallucination_eval_report.md + hallucination_detail.json
"""
import json
import os
import re
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ---- 环境隔离 + 配置覆盖（必须在导入项目模块之前）----
EVAL_DIR = os.path.join(PROJECT_ROOT, "scripts", "eval_results")
DB_PATH = os.path.join(EVAL_DIR, "eval_hallu_sentiment.db")
CHROMA_DIR = os.path.join(EVAL_DIR, "eval_hallu_chroma")
os.makedirs(EVAL_DIR, exist_ok=True)
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
if os.path.exists(CHROMA_DIR):
    import shutil
    shutil.rmtree(CHROMA_DIR)
os.environ["SQLITE_DB_PATH"] = DB_PATH
os.environ["CHROMA_PERSIST_DIR"] = CHROMA_DIR
os.environ["CHROMA_ACTIVE_POINTER"] = os.path.join(EVAL_DIR, "eval_hallu_active.json")
os.environ["ANONYMIZED_TELEMETRY"] = "False"
# QA_VERIFY 不在此设置：脚本内分阶段动态切换（baseline=0 / verified=1）

from eval_corpus import DOCS  # noqa: E402
from hallucination_corpus import QUESTIONS  # noqa: E402

llm_calls = {"n": 0}


def judge_answer(query: str, answer: str, docs: list, format_docs_numbered) -> dict:
    """LLM judge：判定 资料是否足以回答 / 回答是否忠实 / 是否编造。失败降级为None字段。"""
    from chains.qa_chain import get_llm

    prompt = (
        "你是一名严格的事实核查员。下面是检索到的编号资料、用户问题和系统的回答。\n\n"
        f"资料：\n{format_docs_numbered(docs)}\n\n"
        f"用户问题：{query}\n\n"
        f"系统回答：\n{answer}\n\n"
        "请判定（判定只依据上面的资料，不要用你自己的知识）：\n"
        '只输出JSON，不要输出其他内容：\n'
        '{"corpus_has_answer": true或false,'
        ' "faithful": true或false,'
        ' "fabricated": ["回答中资料不存在的具体论断或数字"],'
        ' "reason": "一句话理由"}\n'
        "判定规则：\n"
        "- corpus_has_answer：资料里的信息是否足以回答该问题\n"
        "- faithful：回答是否忠于资料。回答中出现资料没有的具体数字/日期/事件 => false；"
        "如实说明\"资料有限，无法回答\" => true；对资料的合理概括改写 => true\n"
        "- fabricated：列出回答中资料不存在的具体论断；没有则空数组"
    )
    try:
        llm_calls["n"] += 1
        resp = get_llm().invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {"corpus_has_answer": None, "faithful": None, "fabricated": [], "reason": "judge输出解析失败"}
        data = json.loads(m.group())
        return {
            "corpus_has_answer": data.get("corpus_has_answer") if isinstance(data.get("corpus_has_answer"), bool) else None,
            "faithful": data.get("faithful") if isinstance(data.get("faithful"), bool) else None,
            "fabricated": [str(x) for x in data.get("fabricated", [])][:5],
            "reason": str(data.get("reason", ""))[:200],
        }
    except Exception as e:
        return {"corpus_has_answer": None, "faithful": None, "fabricated": [], "reason": f"judge失败降级: {type(e).__name__}"}


def main():
    t0 = time.time()

    print("== 1. 灌库 + 重建向量索引（与问题②评测同一套生产链路）==")
    from storage.db import init_db
    from storage import repository
    init_db()
    for d in DOCS:
        repository.insert_record({
            "raw_text": d["text"], "cleaned_text": d["text"],
            "source": d["source"], "date": d["date"], "company": d["company"],
            "source_type": "研报", "tenant_id": "default",
        })
    print(f"   已写入 {len(DOCS)} 条")
    from retrieval.rebuild import rebuild_vector_index
    res = rebuild_vector_index()
    assert res.get("rebuild"), f"向量索引重建失败: {res}"
    print(f"   向量索引重建完成: {res['count']}条")

    from retrieval.retriever import hybrid_retrieve
    from chains import qa_chain
    from chains.qa_chain import format_docs_numbered
    from core.faithfulness import verify_answer, verify_with_regenerate, is_refusal, verify_enabled
    from config.settings import settings

    print(f"== 2. 检索阶段（两种模式共用同一批docs保证对比公平）==")
    retrieved = {}
    for i, q in enumerate(QUESTIONS, 1):
        retrieved[q["id"]] = hybrid_retrieve(q["q"], k=settings.RETRIEVE_TOP_K)
        print(f"   [{i:>2}/{len(QUESTIONS)}] {q['id']} 检索到 {len(retrieved[q['id']])} 篇")

    results = {q["id"]: {"q": q, "docs_n": len(retrieved[q["id"]])} for q in QUESTIONS}

    # ---- Phase A: baseline（QA_VERIFY=0，原版prompt）----
    print("\n== 3. Phase A baseline（原版prompt，无校验）==")
    os.environ["QA_VERIFY"] = "0"
    qa_chain.reset_chains()
    assert not verify_enabled()
    for i, q in enumerate(QUESTIONS, 1):
        docs = retrieved[q["id"]]
        llm_calls["n"] += 1
        t = time.perf_counter()
        ans = qa_chain.generate_answer(q["q"], docs)
        # baseline回答也做离线规则校验（run_llm_check=False，只用于测量，不干预生成）
        m = verify_answer(q["q"], ans, docs, run_llm_check=False)
        results[q["id"]]["baseline"] = {
            "answer": ans, "latency_s": round(time.perf_counter() - t, 1),
            "refusal": is_refusal(ans),
            "unfaithful_numbers": m["unfaithful_numbers"],
            "unknown_companies": m["unknown_companies"],
            "invalid_refs": m["citations"]["invalid_refs"],
        }
        print(f"   [{i:>2}/{len(QUESTIONS)}] {q['id']} {q['type'].upper()} 生成完毕 ({results[q['id']]['baseline']['latency_s']}s)")

    # ---- Phase B: verified（QA_VERIFY=1，引用prompt+三层校验+受限重生成）----
    print("\n== 4. Phase B verified（引用标注+三层校验）==")
    os.environ["QA_VERIFY"] = "1"
    qa_chain.reset_chains()
    assert verify_enabled()
    for i, q in enumerate(QUESTIONS, 1):
        docs = retrieved[q["id"]]
        llm_calls["n"] += 1
        t = time.perf_counter()
        ans = qa_chain.generate_answer(q["q"], docs)
        ans2, res = verify_with_regenerate(q["q"], ans, docs)
        results[q["id"]]["verified"] = {
            "answer": ans2, "latency_s": round(time.perf_counter() - t, 1),
            "regen_attempts": res.get("regen_attempts", 0),
            "risk": res["risk"], "layers": res["layers_triggered"],
            "refusal": is_refusal(ans2),
            "unfaithful_numbers": res["unfaithful_numbers"],
            "unknown_companies": res["unknown_companies"],
            "invalid_refs": res["citations"]["invalid_refs"],
            "total_refs": res["citations"]["total_refs"],
            "llm_supported": res["llm"]["supported"],
        }
        flag = "RISK" if res["risk"] else "ok"
        print(f"   [{i:>2}/{len(QUESTIONS)}] {q['id']} {q['type'].upper()} {flag} "
              f"layers={res['layers_triggered']} regen={res.get('regen_attempts', 0)}")

    # ---- Phase C: judge（对两个模式的全部回答做LLM判定）----
    print("\n== 5. Phase C LLM judge（每回答一次判定）==")
    for mode in ("baseline", "verified"):
        for i, q in enumerate(QUESTIONS, 1):
            r = results[q["id"]][mode]
            r["judge"] = judge_answer(q["q"], r["answer"], retrieved[q["id"]], format_docs_numbered)
        print(f"   {mode}: {len(QUESTIONS)} 条判定完成")

    # ---- 汇总 ----
    def agg(mode, qtype):
        rows = [results[q["id"]][mode] for q in QUESTIONS if q["type"] == qtype]
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

    def agg_in_extra(mode):
        rows = [results[q["id"]][mode] for q in QUESTIONS if q["type"] == "in"]
        n = len(rows)
        out = {"n": n}
        if mode == "verified":
            with_refs = [r for r in rows if r["total_refs"] > 0]
            out["citation_coverage"] = len(with_refs) / n
            out["avg_refs"] = sum(r["total_refs"] for r in rows) / n
            out["risk_rate"] = sum(r["risk"] for r in rows) / n
        return out

    summary = {
        mode: {
            "out": agg(mode, "out"),
            "in": agg(mode, "in"),
            **{f"in_extra_{k}": v for k, v in agg_in_extra(mode).items()},
        }
        for mode in ("baseline", "verified")
    }

    print("\n== 6. 汇总对比 ==")
    print(f"{'指标':<26}{'baseline':>12}{'verified':>12}")
    print("-" * 50)
    b, v = summary["baseline"]["out"], summary["verified"]["out"]
    print(f"{'out拒答率(规则)':<24}{b['rule_refusal_rate']:>12.1%}{v['rule_refusal_rate']:>12.1%}")
    print(f"{'out拒答率(judge)':<23}{b['judge_refusal_rate']:>12.1%}{v['judge_refusal_rate']:>12.1%}")
    print(f"{'out编造率(规则)':<24}{b['rule_fabrication_rate']:>12.1%}{v['rule_fabrication_rate']:>12.1%}")
    print(f"{'out编造率(judge)':<23}{b['judge_fabrication_rate']:>12.1%}{v['judge_fabrication_rate']:>12.1%}")
    print(f"{'out忠实率(judge)':<23}{b['judge_faithful_rate']:>12.1%}{v['judge_faithful_rate']:>12.1%}")
    bi, vi = summary["baseline"]["in"], summary["verified"]["in"]
    print(f"{'in正常回答率(规则)':<23}{1 - bi['rule_refusal_rate']:>12.1%}{1 - vi['rule_refusal_rate']:>12.1%}"
          f"  ←（越高越好：应回答不拒答）")
    print(f"{'in忠实率(judge)':<24}{bi['judge_faithful_rate']:>12.1%}{vi['judge_faithful_rate']:>12.1%}")
    if "in_extra_citation_coverage" in summary["verified"]:
        print(f"{'in引用标注覆盖率':<23}{'—':>12}{summary['verified']['in_extra_citation_coverage']:>12.1%}")
        print(f"{'in平均引用数':<24}{'—':>12}{summary['verified']['in_extra_avg_refs']:>12.1f}")
        print(f"{'in校验风险率':<25}{'—':>12}{summary['verified']['in_extra_risk_rate']:>12.1%}")
    print(f"\nLLM调用总数: {llm_calls['n']}次, 总耗时 {time.time() - t0:.0f}s")

    # ---- 保存 ----
    detail = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M"), "docs": len(DOCS), "questions": len(QUESTIONS),
            "embedding": os.getenv("LOCAL_EMBEDDING_MODEL", "local"),
            "llm_model": os.getenv("OPENAI_MODEL", ""),
            "llm_calls": llm_calls["n"], "cost_seconds": round(time.time() - t0, 0),
        },
        "summary": summary,
        "detail": {
            qid: {
                "question": results[qid]["q"]["q"],
                "qtype": results[qid]["q"]["type"],
                "trap": results[qid]["q"].get("trap", ""),
                **{k: v for k, v in results[qid].items() if k != "q"},
            } for qid in results
        },
    }
    out_json = os.path.join(EVAL_DIR, "hallucination_detail.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2, default=str)
    print(f"明细已保存: {out_json}")

    # ---- markdown 报告 ----
    md = []
    md.append("# 幻觉评测报告（README问题③：RAG问答幻觉工程化控制）\n")
    md.append(f"- 日期：{time.strftime('%Y-%m-%d %H:%M')}")
    md.append(f"- 语料：{len(DOCS)} 篇（与问题②评测同源），问题：30 条（in 15 / out 15）")
    md.append(f"- LLM：{os.getenv('OPENAI_MODEL', '')}（真实API调用，全程 {llm_calls['n']} 次）")
    md.append(f"- 对比：baseline（原版Prompt，无校验） vs verified（引用标注+三层程序化校验+受限重生成）\n")
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
    if "in_extra_citation_coverage" in summary["verified"]:
        md.append(f"| in引用标注覆盖率 | — | {summary['verified']['in_extra_citation_coverage']:.1%} | verified专属能力 |")
        md.append(f"| in校验风险率 | — | {summary['verified']['in_extra_risk_rate']:.1%} | 库内问题的误报水平 |")
    md.append(f"\nFTS/向量检索环境与问题②评测一致（本地bge-large-zh-v1.5）。"
              f"judge与生成使用同一模型（{os.getenv('OPENAI_MODEL', '')}），存在自评偏差可能，已用规则口径交叉。\n")
    md.append("## 库外问题逐条明细（幻觉主战场）\n")
    md.append("| id | 问题 | base拒答 | base编造(规则) | base编造(judge) | ver拒答 | ver编造(规则) | ver编造(judge) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for q in QUESTIONS:
        if q["type"] != "out":
            continue
        rb, rv = results[q["id"]]["baseline"], results[q["id"]]["verified"]
        md.append(
            f"| {q['id']} | {q['q'][:30]} "
            f"| {'是' if rb['refusal'] else '否'} "
            f"| {','.join((rb['unfaithful_numbers'] + rb['unknown_companies'])[:2]) or '无'} "
            f"| {'是' if rb['judge']['fabricated'] else '否'} "
            f"| {'是' if rv['refusal'] else '否'} "
            f"| {','.join((rv['unfaithful_numbers'] + rv['unknown_companies'])[:2]) or '无'} "
            f"| {'是' if rv['judge']['fabricated'] else '否'} |")
    out_md = os.path.join(EVAL_DIR, "hallucination_eval_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"报告已保存: {out_md}")


if __name__ == "__main__":
    main()
