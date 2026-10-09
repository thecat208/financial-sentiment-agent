# -*- coding: utf-8 -*-
"""FTS5中文分词修复的离线冒烟测试（降级二元组分词路径，不依赖jieba/网络）。
覆盖：a.新库全链路（建库→入库→子串词/代码/纯标点查询）
     b.老库迁移（旧FTS结构自动重建+存量回填）
     c.并发验收脚本回归（确认db.py改动不破坏问题①的8/8）"""
import os
import shutil
import sqlite3
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

passed = failed = 0


def check(name, cond, extra=""):
    global passed, failed
    mark = "PASS" if cond else "FAIL"
    if cond:
        passed += 1
    else:
        failed += 1
    print(f"  [{mark}] {name}" + (f"  ({extra})" if extra else ""))


docs = [
    {"cleaned_text": "贵州茅台三季报净利润同比增长15%，直销渠道占比提升。", "company": "贵州茅台", "source": "新闻资讯", "date": "2026-10-01"},
    {"cleaned_text": "招商银行发行500亿元可转债，用于补充核心资本。", "company": "招商银行", "source": "新闻资讯", "date": "2026-10-02"},
    {"cleaned_text": "宁德时代碳酸锂采购价格下行，毛利率改善。", "company": "宁德时代", "source": "研报", "date": "2026-10-03"},
    {"cleaned_text": "贵州茅台酒销售有限公司完成IPO上市辅导备案。", "company": "贵州茅台", "source": "公告", "date": "2026-10-04"},
]

print("== A. 新库全链路（降级二元组分词）==")
tmp = tempfile.mkdtemp(prefix="fts_test_")
os.environ["SQLITE_DB_PATH"] = os.path.join(tmp, "new.db")
for m in [m for m in list(sys.modules) if m.startswith(("storage", "config"))]:
    del sys.modules[m]
from storage import db as db_mod
from storage.repository import insert_record, search_text
db_mod.init_db()
for d in docs:
    insert_record(d)

r = search_text("可转债")
check("子串词[可转债]命中招行", len(r) == 1 and r[0]["company"] == "招商银行", f"命中{len(r)}条")
r = search_text("碳酸锂价格")
check("子串词[碳酸锂价格]命中宁德", len(r) >= 1 and r[0]["company"] == "宁德时代", f"命中{len(r)}条")
r = search_text("600519.SH")
check("含标点查询[600519.SH]不报错(空结果)", isinstance(r, list))
r = search_text("IPO")
check("英文词[IPO]命中", len(r) == 1 and "IPO" in r[0]["cleaned_text"], f"命中{len(r)}条")
r = search_text("茅台 净利润")
check("多词AND[茅台 净利润]命中茅台财报", len(r) == 1 and r[0]["company"] == "贵州茅台", f"命中{len(r)}条")
r = search_text("！！！")
check("纯标点查询返回空列表", r == [])
r = search_text("")
check("空查询返回空列表", r == [])

print("== B. 老库迁移（旧FTS结构→新结构+回填）==")
old_db = os.path.join(tmp, "old.db")
# 老库的行数据可能还在-WAL里没checkpoint，必须连sidecar一起拷贝，
# 否则拷出去的主库是空的，测不了"存量回填"
for suffix in ["", "-wal", "-shm"]:
    src = "data/sentiment.db" + suffix
    if os.path.exists(src):
        shutil.copy2(src, old_db + suffix)
old_cnt = sqlite3.connect(old_db).execute("SELECT COUNT(*) FROM sentiment_records").fetchone()[0]
if old_cnt == 0:
    # 真实库sentiment_records当前为空，无法测"存量回填"——插入模拟旧数据：
    # 直连raw INSERT（不走insert_record），fts_tokens保持NULL，等价于升级前的老库
    legacy = sqlite3.connect(old_db)
    legacy.execute("INSERT INTO sentiment_records (cleaned_text, company, tenant_id) VALUES ('老数据：宁德时代发行可转债获批', '宁德时代', 'default')")
    legacy.execute("INSERT INTO sentiment_records (cleaned_text, company, tenant_id) VALUES ('老数据：贵州茅台发布三季报', '贵州茅台', 'default')")
    legacy.commit()
    legacy.close()
    old_cnt = 2
os.environ["SQLITE_DB_PATH"] = old_db
for m in [m for m in list(sys.modules) if m.startswith(("storage", "config"))]:
    del sys.modules[m]
import importlib
db_old = importlib.import_module("storage.db")
db_old.init_db()
conn = sqlite3.connect(old_db)
conn.row_factory = sqlite3.Row
fts_sql = conn.execute("SELECT sql FROM sqlite_master WHERE name='sentiment_fts'").fetchone()["sql"]
check("旧FTS表已重建为fts_tokens结构", "fts_tokens" in fts_sql)
n_tok = conn.execute("SELECT COUNT(*) FROM sentiment_records WHERE fts_tokens IS NOT NULL").fetchone()[0]
check(f"存量{old_cnt}条记录token全部回填", n_tok == old_cnt, f"回填{n_tok}/{old_cnt}")
n_fts = conn.execute("SELECT COUNT(*) FROM sentiment_fts").fetchone()[0]
check("重建后的倒排索引非空", n_fts > 0, f"索引{n_fts}条")
conn.close()
# 迁移后的库必须能直接检索（模拟升级后立即使用的场景）
r_mig = db_old_search = None
for m in [m for m in list(sys.modules) if m.startswith(("storage", "config"))]:
    del sys.modules[m]
from storage.repository import search_text as _st
r_mig = _st("可转债")
check("迁移后的老库可召回子串词[可转债]", len(r_mig) == 1 and r_mig[0]["company"] == "宁德时代", f"命中{len(r_mig)}条")

print("== C. 并发验收回归（问题①不回退）==")
import subprocess
r = subprocess.run([sys.executable, "scripts/test_db_concurrency.py"],
                   capture_output=True, text=True, cwd=ROOT, timeout=300,
                   env={**os.environ, "SQLITE_DB_PATH": os.path.join(tmp, "conc.db")})
if r.returncode != 0:
    print("  ---- 并发脚本完整输出 ----")
    print(r.stdout[-3000:])
    print(r.stderr[-1500:])
check("并发验收脚本退出码为0", r.returncode == 0)

shutil.rmtree(tmp, ignore_errors=True)
print(f"\n结果: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
