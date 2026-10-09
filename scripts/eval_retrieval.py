"""
检索质量评测：向量(Chroma+OpenAI Embedding) vs FTS5关键词 vs 混合检索。

对应 README 6.1 问题2（向量检索从未真实跑通、召回质量未验证）。

严格复用项目自身的检索栈，评测的是"生产路径"而非旁路实现：
- 灌库走 storage.repository.insert_record（FTS触发器自动建索引）
- 向量索引走 retrieval.rebuild.rebuild_vector_index（生产重建链路，含tenant_id元数据）
- 三种检索走 retrieval.retriever 的 retrieve/keyword_retrieve/hybrid_retrieve

指标（K=5，与 settings.RETRIEVE_TOP_K 一致）：
- Recall@5：标准答案被捞回的比例（多文档归集类查询的主指标）
- MRR：第一个命中答案的排名倒数（单答案查询的主指标）
- nDCG@5：按位次折算的加权命中质量

运行：venv python scripts/eval_retrieval.py
产出：scripts/eval_results/retrieval_eval_report.md + detail.json
"""
import json
import math
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ---- 环境隔离 + 配置覆盖（必须在导入项目模块之前）----
EVAL_DIR = os.path.join(PROJECT_ROOT, "scripts", "eval_results")
DB_PATH = os.path.join(EVAL_DIR, "eval_sentiment.db")
CHROMA_DIR = os.path.join(EVAL_DIR, "eval_chroma")
os.makedirs(EVAL_DIR, exist_ok=True)
for p in (DB_PATH,):
    if os.path.exists(p):
        os.remove(p)
if os.path.exists(CHROMA_DIR):
    import shutil
    shutil.rmtree(CHROMA_DIR)

os.environ["SQLITE_DB_PATH"] = DB_PATH
os.environ["CHROMA_PERSIST_DIR"] = CHROMA_DIR
os.environ["CHROMA_ACTIVE_POINTER"] = os.path.join(EVAL_DIR, "eval_active.json")
os.environ["ANONYMIZED_TELEMETRY"] = "False"
# EMBEDDING_PROVIDER 不在此覆盖：跟随 .env 的生产配置（local: D:\test_model\bge-large-zh-v1.5）
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from eval_corpus import DOCS, QUERIES  # noqa: E402

K = 5           # 与 settings.RETRIEVE_TOP_K 对齐
TOPN = 10       # 评测窗口


def metrics_for(ranked: list[str], relevant: list[str]) -> dict:
    """ranked: 按排名顺序的doc id列表; relevant: 标准答案id列表"""
    top5, top10 = ranked[:K], ranked[:TOPN]
    rel = set(relevant)
    hit5 = rel.intersection(top5)
    hit10 = rel.intersection(top10)
    recall5 = len(hit5) / len(rel) if rel else 0.0
    mrr = 0.0
    for i, d in enumerate(top10, start=1):
        if d in rel:
            mrr = 1.0 / i
            break
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(top5, start=1) if d in rel)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), K) + 1))
    ndcg5 = dcg / idcg if idcg else 0.0
    return {"recall5": recall5, "mrr": mrr, "ndcg5": ndcg5,
            "hit10": bool(hit10)}


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "recall5": sum(r["recall5"] for r in rows) / n,
        "mrr": sum(r["mrr"] for r in rows) / n,
        "ndcg5": sum(r["ndcg5"] for r in rows) / n,
        "hit10_rate": sum(r["hit10"] for r in rows) / n,
        "n": n,
    }


def main():
    t0 = time.time()
    print("== 1. 灌库（走项目生产写入链路，FTS触发器自动建索引）==")
    from storage.db import init_db
    from storage import repository
    init_db()
    text2id = {}
    for d in DOCS:
        repository.insert_record({
            "raw_text": d["text"], "cleaned_text": d["text"],
            "source": d["source"], "date": d["date"], "company": d["company"],
            "source_type": "研报", "tenant_id": "default",
        })
        text2id[d["text"]] = d["id"]
    print(f"   已写入 {len(DOCS)} 条")

    print("== 2. 重建向量索引（走项目生产重建链路，OpenAI Embedding）==")
    from retrieval.rebuild import rebuild_vector_index
    res = rebuild_vector_index()
    print(f"   {res}")
    assert res.get("rebuild"), "向量索引重建失败"

    from retrieval.retriever import retrieve, keyword_retrieve, hybrid_retrieve
    from storage.db import get_conn
    from storage.repository import search_text

    def ranked_ids(docs: list) -> list[str]:
        out = []
        for doc in docs:
            did = text2id.get(doc.page_content)
            out.append(did)
        return [d for d in out if d]

    print(f"== 3. 执行 {len(QUERIES)} 条查询 × 3种检索 ==")
    all_rows = {"vector": [], "fts5": [], "hybrid": []}
    fts_syntax_errors = 0
    latency = {"vector": [], "fts5": [], "hybrid": []}

    for i, q in enumerate(QUERIES, start=1):
        query = q["q"]
        # FTS5语法错误单独计数（keyword_retrieve会静默吞掉异常返回[]，这里显式探测）
        try:
            search_text(query, limit=3)
        except Exception:
            fts_syntax_errors += 1

        t = time.perf_counter()
        v_docs = retrieve(query, k=TOPN)
        latency["vector"].append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        f_docs = keyword_retrieve(query, k=TOPN)
        latency["fts5"].append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        h_docs = hybrid_retrieve(query, k=TOPN)
        latency["hybrid"].append((time.perf_counter() - t) * 1000)

        m = {
            "vector": metrics_for(ranked_ids(v_docs), q["relevant"]),
            "fts5": metrics_for(ranked_ids(f_docs), q["relevant"]),
            "hybrid": metrics_for(ranked_ids(h_docs), q["relevant"]),
        }
        for k_name, v in m.items():
            all_rows[k_name].append({**v, "type": q["type"], "q": query})
        flag = "OK " if m["vector"]["hit10"] or m["fts5"]["hit10"] else "MISS"
        print(f"   [{flag}] {i:>2}/40 {query}  "
              f"vec R@5={m['vector']['recall5']:.2f} fts R@5={m['fts5']['recall5']:.2f}")

    print("\n== 4. 汇总 ==")
    report = {"K": K, "queries": len(QUERIES), "docs": len(DOCS), "retrievers": {}}
    header = f"{'检索器':<10}{'Recall@5':>10}{'MRR':>8}{'nDCG@5':>9}{'Hit@10':>8}{'P50延迟ms':>11}"
    print("  " + header)
    print("  " + "-" * len(header))
    for name in ("vector", "fts5", "hybrid"):
        overall = aggregate(all_rows[name])
        lat = sorted(latency[name])
        p50 = lat[len(lat) // 2]
        print(f"  {name:<10}{overall['recall5']:>10.3f}{overall['mrr']:>8.3f}"
              f"{overall['ndcg5']:>9.3f}{overall['hit10_rate']:>8.2%}{p50:>11.1f}")
        report["retrievers"][name] = {"overall": overall}
    report["fts_syntax_errors"] = fts_syntax_errors

    # 分查询类型
    print("\n  分查询类型 Recall@5：")
    type_report = {}
    for tname in ("semantic", "keyword", "multi"):
        type_report[tname] = {}
        for name in ("vector", "fts5", "hybrid"):
            rows = [r for r in all_rows[name] if r["type"] == tname]
            if rows:
                type_report[tname][name] = aggregate(rows)
        line = "  " + f"{tname:<10}"
        for name in ("vector", "fts5", "hybrid"):
            if name in type_report[tname]:
                line += f"  {name}={type_report[tname][name]['recall5']:.3f}"
        print(line)
    report["by_type"] = type_report
    report["detail"] = all_rows
    report["latency_p50_ms"] = {k: sorted(v)[len(v) // 2] for k, v in latency.items()}
    report["cost_seconds"] = round(time.time() - t0, 1)

    out_json = os.path.join(EVAL_DIR, "detail.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n明细已保存: {out_json}")
    print(f"总耗时 {report['cost_seconds']}s, FTS语法错误查询数: {fts_syntax_errors}")

    # ---- 生成 markdown 报告 ----
    md = []
    md.append("# 检索质量评测报告（README问题②：向量检索真实环境验证）\n")
    md.append(f"- 日期：{time.strftime('%Y-%m-%d %H:%M')}")
    md.append(f"- 语料：{len(DOCS)} 篇手写中文金融新闻（13家公司×8篇，事件类型分散）")
    md.append(f"- 查询：{len(QUERIES)} 条标注查询（semantic 24 / keyword 8 / multi 8，标注关系58条）")
    md.append(f"- 指标窗口：K={K}（与 RETRIEVE_TOP_K 一致），评测窗口 Top{TOPN}")
    _prov = os.getenv("EMBEDDING_PROVIDER", "local")
    _model = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5") if _prov == "local" else "OpenAI text-embedding"
    md.append(f"- Embedding：{_prov} ({_model})，跟随项目 .env 生产配置")
    md.append("- 检索栈：全部走项目生产链路（retrieval/retriever + rebuild + repository），非旁路实现\n")
    md.append("## 总体结果\n")
    md.append("| 检索器 | Recall@5 | MRR | nDCG@5 | Hit@10 | P50延迟(ms) |")
    md.append("|---|---|---|---|---|---|")
    for name, label in (("vector", "向量检索(Chroma)"), ("fts5", "FTS5关键词"), ("hybrid", "混合检索")):
        o = report["retrievers"][name]["overall"]
        md.append(f"| {label} | {o['recall5']:.3f} | {o['mrr']:.3f} | {o['ndcg5']:.3f} "
                  f"| {o['hit10_rate']:.2%} | {report['latency_p50_ms'][name]:.1f} |")
    md.append("\n## 分查询类型 Recall@5\n")
    md.append("| 查询类型 | 向量 | FTS5 | 混合 |")
    md.append("|---|---|---|---|")
    for tname, label in (("semantic", "语义改写(24)"), ("keyword", "精确关键词(8)"), ("multi", "多文档归集(8)")):
        r = report["by_type"].get(tname, {})
        md.append(f"| {label} | {r.get('vector', {}).get('recall5', 0):.3f} "
                  f"| {r.get('fts5', {}).get('recall5', 0):.3f} "
                  f"| {r.get('hybrid', {}).get('recall5', 0):.3f} |")
    md.append(f"\nFTS5查询语法错误（被降级吞掉返回空结果）的查询数：{fts_syntax_errors}\n")
    out_md = os.path.join(EVAL_DIR, "retrieval_eval_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"报告已保存: {out_md}")


if __name__ == "__main__":
    main()
