"""
QA链路并发基准测试（README问题①异步化改造验收）。

对比同一QA链路（检索→生成→三层校验）在两种并发模型下的吞吐：
  sync  ：线程池并发（原架构——每请求占一个线程，线程在LLM等待期间空转）
  async ：asyncio事件循环并发（新架构——LLM等待期间事件循环处理其他请求）

两者调用同一份生产代码（ask / ask_async），同一批问题、同一套数据，
唯一变量是并发模型——差值就是异步化改造的净收益。

数据环境：复用 scripts/eval_results/ 下的评测库（eval_hallu_sentiment.db +
eval_hallu_chroma，104篇语料，与幻觉评测同源）；不存在时自动从语料重建。

运行（ctm_kg环境，项目根目录）：
  python scripts/bench_concurrency.py                 # 默认并发5，QA路径
  python scripts/bench_concurrency.py --concurrency 8 --rounds 2
  python scripts/bench_concurrency.py --router        # 走意图路由路径(/query等价)
  python scripts/bench_concurrency.py --smoke         # 快速冒烟：单请求对比两种模式

说明：
- LLM调用约 2×并发数×轮数 次（生成+校验judge），费用可忽略；建议并发5~10
- 两种模式的执行先后会有先后偏差（服务商负载波动），数字看量级不看个位数
"""
import argparse
import asyncio
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

# ---- 数据环境（与幻觉评测同一套隔离库；存在即复用，避免重复建索引）----
EVAL_DIR = os.path.join(PROJECT_ROOT, "scripts", "eval_results")
DB_PATH = os.path.join(EVAL_DIR, "eval_hallu_sentiment.db")
CHROMA_DIR = os.path.join(EVAL_DIR, "eval_hallu_chroma")
os.makedirs(EVAL_DIR, exist_ok=True)
os.environ["SQLITE_DB_PATH"] = DB_PATH
os.environ["CHROMA_PERSIST_DIR"] = CHROMA_DIR
os.environ["CHROMA_ACTIVE_POINTER"] = os.path.join(EVAL_DIR, "eval_hallu_active.json")
os.environ["ANONYMIZED_TELEMETRY"] = "False"


def ensure_data():
    """评测库不存在/为空时，从语料灌库+重建向量索引（与eval脚本同一生产链路）"""
    import sqlite3
    need_build = True
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            n = conn.execute("SELECT COUNT(*) FROM sentiment_records").fetchone()[0]
            conn.close()
            need_build = n == 0
        except sqlite3.OperationalError:
            need_build = True
    if not need_build:
        print(f"数据环境就绪：{DB_PATH}（复用现有索引）")
        return
    print("评测库为空，正在灌库+重建向量索引（首次约1~2分钟）...")
    from eval_corpus import DOCS
    from storage.repository import insert_record
    from retrieval.rebuild import rebuild_vector_index
    for d in DOCS:
        insert_record(dict(d))
    rebuild_vector_index()
    print(f"灌库完成：{len(DOCS)}条")


def _pick_questions(n: int) -> list:
    """从标注查询集取n条真实问题（问题互不相同，避免缓存命中使对比失真）"""
    from eval_corpus import QUERIES
    qs = []
    for q in QUERIES:
        text = q["q"] if isinstance(q, dict) else q
        if text not in qs:
            qs.append(text)
        if len(qs) >= n:
            break
    return qs


def run_sync(questions: list) -> list:
    """线程池并发（原架构模型）：每请求一个线程，LLM等待期间线程空转"""
    from graphs.qa_graph import ask
    t0 = time.perf_counter()

    def one(q):
        start = time.perf_counter()
        r = ask(q, rate_limit=False)
        return r, start - t0, time.perf_counter() - start

    with ThreadPoolExecutor(max_workers=len(questions)) as pool:
        futures = [pool.submit(one, q) for q in questions]
        out = [f.result() for f in futures]
    wall = time.perf_counter() - t0
    results = [r for r, _, _ in out]
    timing = [(off, lat) for _, off, lat in out]  # (启动偏移, 单请求耗时)
    return results, wall, timing


async def run_async(questions: list) -> list:
    """事件循环并发（新架构模型）：协程挂起等待LLM，事件循环处理其他请求"""
    from graphs.qa_graph import ask_async

    t0 = time.perf_counter()

    async def one(q):
        start = time.perf_counter()
        r = await ask_async(q, rate_limit=False)
        return r, start - t0, time.perf_counter() - start

    out = await asyncio.gather(*[one(q) for q in questions])
    wall = time.perf_counter() - t0
    results = [r for r, _, _ in out]
    timing = [(off, lat) for _, off, lat in out]
    return results, wall, timing


def run_router_sync(questions: list) -> list:
    from graphs.router_graph import handle_query
    t0 = time.perf_counter()

    def one(q):
        start = time.perf_counter()
        r = handle_query(q, rate_limit=False)
        return r, start - t0, time.perf_counter() - start

    with ThreadPoolExecutor(max_workers=len(questions)) as pool:
        futures = [pool.submit(one, q) for q in questions]
        out = [f.result() for f in futures]
    wall = time.perf_counter() - t0
    results = [r for r, _, _ in out]
    timing = [(off, lat) for _, off, lat in out]
    return results, wall, timing


async def run_router_async(questions: list) -> list:
    from graphs.router_graph import handle_query_async

    t0 = time.perf_counter()

    async def one(q):
        start = time.perf_counter()
        r = await handle_query_async(q, rate_limit=False)
        return r, start - t0, time.perf_counter() - start

    out = await asyncio.gather(*[one(q) for q in questions])
    wall = time.perf_counter() - t0
    results = [r for r, _, _ in out]
    timing = [(off, lat) for _, off, lat in out]
    return results, wall, timing


def _report(mode: str, wall: float, n: int, results: list, timing: list = None):
    ok = sum(1 for r in results if r.get("answer"))
    print(f"  {mode:6s}: 总耗时 {wall:6.1f}s | 吞吐 {n / wall:4.2f} 请求/秒 | "
          f"成功 {ok}/{n} | 平均单请求 {wall / n:5.1f}s")
    if timing:
        detail = ", ".join(
            f"[启动+{off:.1f}s 耗时{lat:.1f}s]" for off, lat in sorted(timing)
        )
        lats = sorted(lat for _, lat in timing)
        p50 = lats[len(lats) // 2]
        print(f"         每请求: {detail}")
        print(f"         单请求p50: {p50:.1f}s")
        print("         判读: 启动都≈0s → 客户端真并发（无串行化）；"
              "耗时远大于正常单请求值 → 服务商侧排队")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=5, help="并发请求数（默认5）")
    ap.add_argument("--rounds", type=int, default=1, help="每个模式跑几轮（默认1，多轮取看稳定性）")
    ap.add_argument("--router", action="store_true", help="测意图路由路径（/query等价，默认测QA路径）")
    ap.add_argument("--smoke", action="store_true", help="冒烟模式：单请求各跑一次，验证两条路径都通")
    args = ap.parse_args()

    ensure_data()
    n = max(1, args.concurrency)

    if args.smoke:
        questions = _pick_questions(1)
        print(f"\n== 冒烟：问题「{questions[0]}」 ==")
        rs, ws, _ = (run_router_sync if args.router else run_sync)(questions)
        ra, wa, _ = asyncio.run((run_router_async if args.router else run_async)(questions))
        print(f"  sync : {ws:.1f}s | 回答: {rs[0].get('answer', '')[:60]}...")
        print(f"  async: {wa:.1f}s | 回答: {ra[0].get('answer', '')[:60]}...")
        print("  两条路径均返回非空回答即通过")
        return

    questions_all = _pick_questions(min(n * args.rounds * 2, len(_pick_questions(999))))
    print(f"\n== QA链路并发基准：并发={n}，每模式{args.rounds}轮，路径={'路由' if args.router else 'QA'} ==")
    print("（同批问题在两种模式下重复使用——生成无缓存，不影响对比公平性）\n")

    sync_walls, async_walls = [], []
    for i in range(args.rounds):
        qs = questions_all[i * n:(i + 1) * n] or questions_all[:n]
        run_s = run_router_sync if args.router else run_sync
        run_a = run_router_async if args.router else run_async
        rs, ws, s_timing = run_s(qs)
        print(f"第{i + 1}轮 ", end="")
        _report("sync", ws, len(qs), rs, s_timing)
        sync_walls.append((ws, len(qs)))
        ra, wa, a_timing = asyncio.run(run_a(qs))
        _report("async", wa, len(qs), ra, a_timing)
        async_walls.append((wa, len(qs)))

    s_n = sum(c for _, c in sync_walls)
    a_n = sum(c for _, c in async_walls)
    s_t = sum(t for t, _ in sync_walls)
    a_t = sum(t for t, _ in async_walls)
    print("\n== 汇总 ==")
    print(f"  sync : {s_n / s_t:.2f} 请求/秒（累计{s_n}请求 / {s_t:.1f}s）")
    print(f"  async: {a_n / a_t:.2f} 请求/秒（累计{a_n}请求 / {a_t:.1f}s）")
    if s_t > 0:
        print(f"  异步加速比: {s_t / a_t:.2f}x")
    print("\n判读参考：两种模式的请求都是同时发出的（每请求启动偏移≈0），")
    print("总耗时差异主要来自服务商侧的并发配额与排队调度，单轮波动大（实测1.2x~2.1x），")
    print("结论看多轮趋势；并发场景下 async 的单请求p50延迟优势通常比吞吐比更稳定。")


if __name__ == "__main__":
    main()
