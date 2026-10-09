"""
SQLite存储层。
并发方案：进程内单共享连接 + RLock串行化 + WAL只在首次初始化时设置一次 + busy_timeout。
原因：本环境下多连接并发访问不稳定（疑似杀软实时扫描对库文件/-wal/-shm的锁占用干扰了
SQLite锁语义），而SQLite单条操作是微秒级，串行化代价可忽略；journal_mode是数据库文件
的持久属性，重复执行切换会因锁冲突把连接打成readonly，故只设一次。
get_conn()的with块内只放纯数据库操作，不要包耗时业务逻辑。
FTS5中文检索：fts_tokens列存预分词结果（storage/fts_tokenizer.py，jieba优先/二元组降级），
FTS外同步表索引该列，老库自动检测旧结构→DROP重建→回填→rebuild。
扩展字段：tenant_id多租户隔离、media_type/media_path素材溯源、dimension_scores四维打分JSON
（见core/taxonomy.py）、source_type/source_credibility来源词表与可信度（写入时算好落库）。
"""
import os
import sqlite3
import threading
from contextlib import contextmanager

DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/sentiment.db")

# 写锁忙等超时（毫秒）：写锁被其他连接占用时最多等多久。
# 环境变量可覆盖，方便压测时调整观察锁行为。
_BUSY_TIMEOUT_MS = int(os.getenv("SQLITE_BUSY_TIMEOUT_MS", 5000))

# FTS5虚表DDL单独拎出来：_ensure_fts_schema()要做"旧结构检测→DROP→按新结构重建"，
# 需要在两处（建表脚本+迁移逻辑）用同一份定义。
_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS sentiment_fts USING fts5(
    fts_tokens, company,
    content='sentiment_records', content_rowid='id'
);
"""

TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sentiment_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_text TEXT,
    cleaned_text TEXT NOT NULL,
    source TEXT,
    date TEXT,
    company TEXT,
    event_type TEXT,
    sentiment_label TEXT,
    sentiment_score REAL,
    analysis_reason TEXT,
    need_alert INTEGER DEFAULT 0,
    alert_is_valid INTEGER,
    alert_review_reason TEXT,
    pushed INTEGER DEFAULT 0,
    tenant_id TEXT DEFAULT 'default',
    media_type TEXT DEFAULT 'text',
    media_path TEXT,
    dimension_scores TEXT,
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);

""" + _FTS_DDL + """
-- 触发器必须每次DROP+重建（CREATE TRIGGER没有IF NOT EXISTS的更新语义），
-- 保证旧库上的旧触发器（索引cleaned_text原文）一定会被替换成新逻辑（索引fts_tokens）。
DROP TRIGGER IF EXISTS sentiment_ai;
CREATE TRIGGER sentiment_ai AFTER INSERT ON sentiment_records BEGIN
    INSERT INTO sentiment_fts(rowid, fts_tokens, company)
    VALUES (new.id, new.fts_tokens, new.company);
END;

-- 用量统计。event_kind区分"llm_call"（LLM调用，记录token消耗）
-- 和"api_call"（API端点调用次数，不涉及token），统一放一张表方便一次查询看全貌。
-- 按date列做天粒度统计（多数计费/配额场景按天/按月汇总，不需要精确到秒级查询）。
CREATE TABLE IF NOT EXISTS usage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    event_kind TEXT NOT NULL,
    operation TEXT NOT NULL,
    model TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    date TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);

-- 操作审计日志。user_id是调用方自报的标识（比如Streamlit里手填的
-- "操作人"，或API的X-User-Id请求头），不是经过身份验证的登录态——本项目目前
-- 没有真正的用户认证体系，这里先把审计记录的数据结构打好地基，
-- 等SSO接入后user_id换成真实登录态即可，表结构不用改。
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    user_id TEXT,
    action TEXT NOT NULL,
    target_type TEXT,
    target_id TEXT,
    detail TEXT,
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);

CREATE INDEX IF NOT EXISTS idx_usage_tenant_date ON usage_events(tenant_id, date);
CREATE INDEX IF NOT EXISTS idx_audit_tenant_date ON audit_log(tenant_id, created_at);
"""

# 兼容早期批次生成的数据库文件（当时表里还没有这几个字段）。
# SQLite不支持"ADD COLUMN IF NOT EXISTS"，需要先查已有列名再决定要不要补。
_MIGRATIONS = [
    ("tenant_id", "TEXT DEFAULT 'default'"),
    ("media_type", "TEXT DEFAULT 'text'"),
    ("media_path", "TEXT"),
    # 多维度情感打分存成JSON字符串
    # （{"业绩":x,"管理层":x,"行业前景":x,"合规风险":x}），旧记录该列为NULL，
    # 展示层按"该记录还没有维度打分"处理，不影响原有 sentiment_label/sentiment_score 的读取。
    ("dimension_scores", "TEXT"),
    # 来源类型受控词表（机构公告/新闻资讯/研报/社交媒体/其他），见 core/taxonomy.py；
    # 旧记录默认按"新闻资讯"处理（历史数据均来自RSS新闻）。
    ("source_type", "TEXT DEFAULT '新闻资讯'"),
    # 按source_type自动推导的可信度分级（高/中/低），写入时算好落库，见上方模块docstring。
    ("source_credibility", "TEXT DEFAULT '中'"),
    # FTS5中文分词修复：入库时由 repository.insert_record() 写入分词结果
    # （cleaned_text+company经 storage/fts_tokenizer.tokenize_for_fts() 预分词），
    # FTS5外同步表索引该列而非原文。旧记录该列为NULL，初始化时自动回填（见_backfill_fts_tokens）。
    ("fts_tokens", "TEXT"),
]

INDEX_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_company ON sentiment_records(company);
CREATE INDEX IF NOT EXISTS idx_date ON sentiment_records(date);
CREATE INDEX IF NOT EXISTS idx_tenant ON sentiment_records(tenant_id);
"""

# 进程内初始化标志：记录已初始化的DB绝对路径。用锁保护"检查+执行"，
# 避免多线程首次并发连接时同时跑DDL。
_initialized_for_path = None

# 进程内单共享连接 + 可重入锁（见模块docstring）：
# 所有数据库访问通过get_conn()串行化。RLock保证同线程嵌套使用不死锁。
_db_lock = threading.RLock()
_shared_conn = None


def _run_migrations(conn):
    existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(sentiment_records)").fetchall()}
    for col_name, col_def in _MIGRATIONS:
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE sentiment_records ADD COLUMN {col_name} {col_def}")


def _ensure_fts_schema(conn) -> bool:
    """老库迁移：FTS虚表若还是旧结构（索引cleaned_text原文、无fts_tokens）则DROP重建。
    返回是否发生了重建（重建后必须对索引做全量rebuild）。"""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='sentiment_fts'"
    ).fetchone()
    if row and row["sql"] and "fts_tokens" not in row["sql"]:
        conn.execute("DROP TABLE IF EXISTS sentiment_fts")
        conn.execute(_FTS_DDL)
        return True
    return False


def _backfill_fts_tokens(conn) -> int:
    """给存量记录补算 fts_tokens（分词结果），返回回填条数。
    分词器自带降级（jieba缺失→二元组），这里不依赖任何可选包是否可用。"""
    rows = conn.execute(
        "SELECT id, cleaned_text, company FROM sentiment_records WHERE fts_tokens IS NULL"
    ).fetchall()
    if not rows:
        return 0
    # 延迟导入：避免"只想要数据库连接、不碰全文检索"的场景背上分词器开销
    from storage.fts_tokenizer import tokenize_for_fts
    for r in rows:
        text = (r["cleaned_text"] or "") + "\n" + (r["company"] or "")
        conn.execute(
            "UPDATE sentiment_records SET fts_tokens=? WHERE id=?",
            (tokenize_for_fts(text), r["id"]),
        )
    return len(rows)


def _ensure_shared_conn():
    """惰性创建进程内唯一共享连接：建表→迁移→索引→FTS迁移/回填→开启WAL，全部只执行一次。

    journal_mode必须在这里设（而不是每个连接都设）：它是数据库文件的持久属性，
    一次设置永久生效；切换动作需要瞬间独占锁，若有其他连接活跃使用数据库，
    切换失败会把执行切换的那条连接留在readonly状态。
    首次初始化发生在进程刚启动、锁保护下的单连接时刻，是安全的。
    """
    global _initialized_for_path, _shared_conn
    abs_path = os.path.abspath(DB_PATH)
    if _initialized_for_path == abs_path and _shared_conn is not None:
        return
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    _shared_conn = sqlite3.connect(DB_PATH, timeout=_BUSY_TIMEOUT_MS / 1000.0,
                                   check_same_thread=False)
    _shared_conn.row_factory = sqlite3.Row
    _shared_conn.execute("PRAGMA journal_mode=WAL")
    _shared_conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    _shared_conn.execute("PRAGMA synchronous=NORMAL")
    _shared_conn.executescript(TABLE_SCHEMA)
    _run_migrations(_shared_conn)
    _shared_conn.executescript(INDEX_SCHEMA)
    # FTS5中文分词修复：老库结构迁移 + 存量token回填，任一发生都全量重建倒排索引
    # （外同步FTS表只认"INSERT时触发器写入"的旧数据，结构变了/补了token都必须rebuild）
    fts_rebuilt = _ensure_fts_schema(_shared_conn)
    n_backfilled = _backfill_fts_tokens(_shared_conn)
    if fts_rebuilt or n_backfilled:
        _shared_conn.execute("INSERT INTO sentiment_fts(sentiment_fts) VALUES('rebuild')")
    _shared_conn.commit()
    _initialized_for_path = abs_path


@contextmanager
def get_conn():
    """
    共享连接版。对外用法：
        with get_conn() as conn:
            conn.execute(...)
    连接是进程内唯一的一条（所有线程共用），with块由可重入锁串行化，
    退出时commit（异常时rollback），连接永不关闭、随进程释放。
    with块内请只放纯数据库操作，不要包耗时业务逻辑。
    """
    with _db_lock:
        _ensure_shared_conn()
        try:
            yield _shared_conn
            _shared_conn.commit()
        except Exception:
            _shared_conn.rollback()
            raise


def init_db():
    """保留独立入口，方便显式初始化/检查数据库文件位置"""
    with get_conn():
        pass


if __name__ == "__main__":
    init_db()
    mode = sqlite3.connect(DB_PATH).execute("PRAGMA journal_mode").fetchone()[0]
    print(f"数据库已初始化：{os.path.abspath(DB_PATH)}（journal_mode={mode}）")
