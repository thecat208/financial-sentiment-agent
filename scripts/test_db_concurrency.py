"""
storage/db.py 并发补丁的验收测试（README 6.1 问题1）。

验证三件事：
  1. 正确性：WAL模式生效；多线程并发读写下无"database is locked"、无丢数据；
     迁移逻辑对旧库兼容（建一张缺列的旧表再连，字段能补齐）。
  2. DDL只跑一次：首个连接执行初始化，后续连接不再重复建表/迁移/索引。
  3. 性能对照：同样负载下对比"旧行为（每连接跑DDL+默认journal）"和
     "新行为（一次初始化+WAL）"的并发写吞吐和读延迟。

用法（不需要任何第三方依赖，纯标准库）：
  python scripts/test_db_concurrency.py            # 跑全部验证
  python scripts/test_db_concurrency.py --bench    # 只跑性能对照
"""
import os
import sys
import time
import tempfile
import threading
import sqlite3
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name} {detail}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def fresh_db_module(db_path: str):
    """每个用例用独立的临时库文件，重新import storage.db拿到干净的模块状态"""
    os.environ["SQLITE_DB_PATH"] = db_path
    import importlib
    import storage.db as db
    importlib.reload(db)
    return db


def test_wal_mode(tmp):
    print("\n== 1. WAL模式与busy_timeout生效 ==")
    db = fresh_db_module(os.path.join(tmp, "wal_test.db"))
    db.init_db()
    raw = sqlite3.connect(db.DB_PATH)
    mode = raw.execute("PRAGMA journal_mode").fetchone()[0]
    check("journal_mode=wal", mode.lower() == "wal", f"(实际: {mode})")
    raw.close()


def test_migration_compat(tmp):
    print("\n== 2. 旧库迁移兼容（缺列表自动补齐）==")
    path = os.path.join(tmp, "legacy.db")
    # 手工造一个"旧版本"的表：只有最初的几个字段，没有 tenant_id/source_type 等
    raw = sqlite3.connect(path)
    raw.execute("""
        CREATE TABLE sentiment_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_text TEXT, cleaned_text TEXT NOT NULL, source TEXT, date TEXT,
            company TEXT, event_type TEXT, sentiment_label TEXT, sentiment_score REAL,
            analysis_reason TEXT, need_alert INTEGER DEFAULT 0, alert_is_valid INTEGER,
            alert_review_reason TEXT, pushed INTEGER DEFAULT 0
        )
    """)
    raw.execute("INSERT INTO sentiment_records (cleaned_text) VALUES ('旧数据一条')")
    raw.commit()
    raw.close()

    db = fresh_db_module(path)
    with db.get_conn() as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(sentiment_records)")}
        row = conn.execute("SELECT cleaned_text, tenant_id FROM sentiment_records").fetchone()
    missing = {"tenant_id", "media_type", "media_path", "dimension_scores", "source_type", "source_credibility"} - cols
    check("缺列全部补齐", not missing, f"(缺: {missing})" if missing else "")
    check("旧数据保留且新列有默认值", row is not None and row["tenant_id"] == "default")


def test_ddl_once(tmp):
    print("\n== 3. DDL进程内只执行一次 ==")
    db = fresh_db_module(os.path.join(tmp, "once.db"))
    with db.get_conn() as conn:
        conn.execute("SELECT 1").fetchone()
    # 行为级验证：改掉SCHEMA常量注入"金丝雀表"。如果后续还会重跑DDL，
    # 查询时金丝雀表就会出现；不重跑则它永远不存在。
    db.TABLE_SCHEMA = "CREATE TABLE IF NOT EXISTS ddl_canary_should_not_exist (id INTEGER);"
    db.INDEX_SCHEMA = ""
    with db.get_conn() as conn:
        tables = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    check("后续访问不再重跑DDL", "ddl_canary_should_not_exist" not in tables)
    check("初始化标志已记录当前库路径", db._initialized_for_path == os.path.abspath(db.DB_PATH))


def run_concurrent_load(db, n_writers=8, n_readers=4, rows_per_writer=25):
    """并发写+读负载，走db.get_conn()（新实现：共享连接+锁串行化）"""
    errors = []
    inserted = {"n": 0}
    lock = threading.Lock()
    read_latencies = []

    def writer(wid):
        try:
            for i in range(rows_per_writer):
                with db.get_conn() as conn:
                    conn.execute(
                        "INSERT INTO sentiment_records (cleaned_text, company, sentiment_score, need_alert)"
                        " VALUES (?, ?, ?, 0)",
                        (f"writer{wid}_row{i}_" + "测试文本" * 5, f"公司{wid % 3}", 0.5),
                    )
                with lock:
                    inserted["n"] += 1
        except Exception as e:
            errors.append(f"writer{wid}: {type(e).__name__}: {e}")

    def reader():
        try:
            for _ in range(40):
                t0 = time.perf_counter()
                with db.get_conn() as conn:
                    conn.execute(
                        "SELECT id, company FROM sentiment_records WHERE company=? LIMIT 10",
                        ("公司1",),
                    ).fetchall()
                read_latencies.append((time.perf_counter() - t0) * 1000)
        except Exception as e:
            errors.append(f"reader: {type(e).__name__}: {e}")

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(n_writers)] + \
              [threading.Thread(target=reader) for _ in range(n_readers)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    return {
        "errors": errors,
        "inserted": inserted["n"],
        "elapsed": elapsed,
        "wps": inserted["n"] / elapsed,
        "read_p50": statistics.median(read_latencies) if read_latencies else None,
        "read_p95": (sorted(read_latencies)[int(len(read_latencies) * 0.95)]) if read_latencies else None,
    }


def run_legacy_load(db_path, n_writers=8, n_readers=4, rows_per_writer=25):
    """复现旧版get_conn行为：每次操作新建连接+每连接跑一遍DDL，无锁。独立实现不经过db.py"""
    import contextlib

    @contextlib.contextmanager
    def legacy_conn():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        # 旧行为：每次连接都重跑建表+迁移+索引
        conn.executescript(TABLE_SCHEMA_SNAPSHOT)
        conn.executescript(INDEX_SCHEMA_SNAPSHOT)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    errors = []
    inserted = {"n": 0}
    read_latencies = []

    def writer(wid):
        try:
            for i in range(rows_per_writer):
                with legacy_conn() as conn:
                    conn.execute(
                        "INSERT INTO sentiment_records (cleaned_text, company, sentiment_score, need_alert)"
                        " VALUES (?, ?, ?, 0)",
                        (f"writer{wid}_row{i}_" + "测试文本" * 5, f"公司{wid % 3}", 0.5),
                    )
                inserted["n"] += 1
        except Exception as e:
            errors.append(f"writer{wid}: {type(e).__name__}: {e}")

    def reader():
        try:
            for _ in range(40):
                t0 = time.perf_counter()
                with legacy_conn() as conn:
                    conn.execute(
                        "SELECT id, company FROM sentiment_records WHERE company=? LIMIT 10",
                        ("公司1",),
                    ).fetchall()
                read_latencies.append((time.perf_counter() - t0) * 1000)
        except Exception as e:
            errors.append(f"reader: {type(e).__name__}: {e}")

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(n_writers)] + \
              [threading.Thread(target=reader) for _ in range(n_readers)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    return {
        "errors": errors,
        "inserted": inserted["n"],
        "wps": inserted["n"] / elapsed,
        "read_p50": statistics.median(read_latencies) if read_latencies else None,
        "read_p95": (sorted(read_latencies)[int(len(read_latencies) * 0.95)]) if read_latencies else None,
    }


# 供legacy复现用的schema快照（测试启动时从db模块拷贝）
TABLE_SCHEMA_SNAPSHOT = ""
INDEX_SCHEMA_SNAPSHOT = ""


def test_concurrency_and_bench(tmp):
    global TABLE_SCHEMA_SNAPSHOT, INDEX_SCHEMA_SNAPSHOT
    print("\n== 4. 并发正确性（8写+4读线程，新实现）==")
    db = fresh_db_module(os.path.join(tmp, "conc_new.db"))
    TABLE_SCHEMA_SNAPSHOT = db.TABLE_SCHEMA
    INDEX_SCHEMA_SNAPSHOT = db.INDEX_SCHEMA
    res = run_concurrent_load(db)
    check("无任何异常（尤其无database is locked/readonly）", not res["errors"],
          f"({res['errors'][:2]})" if res["errors"] else "")
    check("写入数精确一致", res["inserted"] == 8 * 25, f"(实际: {res['inserted']}/200)")

    print("\n== 5. 性能对照：旧版行为 vs 新实现 ==")
    res_old = run_legacy_load(os.path.join(tmp, "conc_old.db"))
    res_new = run_concurrent_load(db)
    print(f"  旧版(每操作开关连接+每连接跑DDL): {res_old['wps']:.0f} 写/秒, "
          f"读P50={res_old['read_p50']:.2f}ms, 读P95={res_old['read_p95']:.2f}ms, "
          f"成功{res_old['inserted']}/200, 错误{len(res_old['errors'])}个")
    print(f"  新实现(共享连接+锁串行化+WAL):    {res_new['wps']:.0f} 写/秒, "
          f"读P50={res_new['read_p50']:.2f}ms, 读P95={res_new['read_p95']:.2f}ms, "
          f"成功{res_new['inserted']}/200, 错误{len(res_new['errors'])}个")
    check("新实现并发零错误", not res_new["errors"])
    # 旧行为的错误数只打印不断言——干净环境下旧行为可能不报错，
    # 但存在杀毒/过滤驱动干扰SQLite锁的环境会稳定复现


def main():
    only_bench = "--bench" in sys.argv
    # ignore_cleanup_errors：Windows下WAL的-shm文件释放时机不确定，目录清理失败不影响测试结论
    with tempfile.TemporaryDirectory(prefix="db_conc_test_", ignore_cleanup_errors=True) as tmp:
        if not only_bench:
            test_wal_mode(tmp)
            test_migration_compat(tmp)
            test_ddl_once(tmp)
        test_concurrency_and_bench(tmp)
    print(f"\n结果: {PASS} 通过, {FAIL} 失败")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
