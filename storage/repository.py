"""
数据读写操作。pipeline_graph 写入分析结果，日报生成/Streamlit历史查询从这里读。

所有读函数都带 tenant_id 参数，默认"default"——单租户场景不用管；企业多租户场景
按当前登录用户所属租户传入，不同租户的数据在SQL层面被WHERE条件天然隔离，
不需要应用层再做过滤，避免漏加过滤条件导致跨租户看到别人数据。
dimension_scores 落库存成JSON字符串，读出时统一反序列化成dict（_row_to_dict里做），
反序列化失败（脏数据/损坏）时给None，不让一条脏记录影响整批查询结果。
source_type 为来源类型受控词表（见 core/taxonomy.py），source_credibility 由
insert_record() 按 source_type 自动推导落库，调用方不需要自己算。
"""
import json
import sqlite3
from datetime import datetime, timedelta
from storage.db import get_conn
from storage.audit_log import record_audit
from core.taxonomy import normalize_source_type, get_source_credibility


def _row_to_dict(row) -> dict:
    """sqlite3.Row -> dict，顺带把dimension_scores从JSON字符串还原成dict"""
    d = dict(row)
    raw = d.get("dimension_scores")
    if raw:
        try:
            d["dimension_scores"] = json.loads(raw)
        except (TypeError, ValueError):
            d["dimension_scores"] = None
    return d


def insert_record(data: dict) -> int:
    """写入一条分析记录，返回自增id（后续更新预警复核结果要用到）"""
    dimension_scores = data.get("dimension_scores")
    source_type = normalize_source_type(data.get("source_type"))
    source_credibility = get_source_credibility(source_type)
    # FTS5中文分词修复：入库时预分词存入fts_tokens列，FTS5外同步表索引该列
    # （unicode61直接索引中文原文会导致子串词完全查不中，见 storage/fts_tokenizer.py）
    from storage.fts_tokenizer import tokenize_for_fts
    fts_tokens = tokenize_for_fts(
        (data.get("cleaned_text") or "") + "\n" + (data.get("company") or ""))
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO sentiment_records
               (raw_text, cleaned_text, source, date, company, event_type,
                sentiment_label, sentiment_score, analysis_reason, need_alert,
                tenant_id, media_type, media_path, dimension_scores,
                source_type, source_credibility, fts_tokens)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                data.get("raw_text"),
                data.get("cleaned_text"),
                data.get("source"),
                data.get("date"),
                data.get("company"),
                data.get("event_type"),
                data.get("sentiment_label"),
                data.get("sentiment_score"),
                data.get("analysis_reason"),
                int(bool(data.get("need_alert"))),
                data.get("tenant_id", "default"),
                data.get("media_type", "text"),
                data.get("media_path"),
                json.dumps(dimension_scores, ensure_ascii=False) if dimension_scores else None,
                source_type,
                source_credibility,
                fts_tokens,
            ),
        )
        return cur.lastrowid


def update_alert_review(record_id: int, is_valid: bool, reason: str):
    """预警复核完成后回填结果。

    注意：本函数只在"规则初筛触发→进入LLM复核"路径上被调用，因此这里
    顺带把need_alert置1——规则触发标记必须在真实节点里写库，不能依赖
    条件边路由函数改state（langgraph规定路由函数内的状态修改不会被保存，
    见graphs/pipeline_graph.py的alert_check_node注释）。
    """
    with get_conn() as conn:
        conn.execute(
            "UPDATE sentiment_records SET need_alert=1, alert_is_valid=?, alert_review_reason=? WHERE id=?",
            (int(bool(is_valid)), reason, record_id),
        )


def mark_pushed(record_id: int):
    """推送成功后标记，便于统计"""
    with get_conn() as conn:
        conn.execute("UPDATE sentiment_records SET pushed=1 WHERE id=?", (record_id,))


def manual_override_alert(record_id: int, is_valid: bool, tenant_id: str = "default",
                           user_id: str = None, note: str = None) -> None:
    """
    人工复核覆盖LLM的预警判定。比如LLM误判了一条"监管处罚"为无关误报，
    人工确认后可以手动改回"属实"，反之亦然。
    和 update_alert_review() 的区别：那个是LLM复核结果的首次回填，这个是"人工事后覆盖"，
    会同时写一条 audit_log（见 storage/audit_log.py）留痕改动前后对比——
    alert_review_reason字段本身会被覆盖，审计记录里留一份改动前的值，
    避免"之前LLM是怎么判断的"这个信息彻底丢失。
    """
    with get_conn() as conn:
        before = conn.execute(
            "SELECT alert_is_valid, alert_review_reason FROM sentiment_records WHERE id=?",
            (record_id,),
        ).fetchone()
        conn.execute(
            "UPDATE sentiment_records SET alert_is_valid=?, alert_review_reason=? WHERE id=?",
            (int(bool(is_valid)), f"[人工复核] {note or ''}".strip(), record_id),
        )

    before_desc = "无记录" if before is None else f"is_valid={before['alert_is_valid']}, reason={before['alert_review_reason']}"
    record_audit(
        tenant_id=tenant_id, action="alert_manual_override", user_id=user_id,
        target_type="sentiment_record", target_id=str(record_id),
        detail=f"改动前：{before_desc}；改动后：is_valid={int(bool(is_valid))}，备注：{note or '（无）'}",
    )


def get_records_by_company(company: str, days: int = 7, tenant_id: str = "default") -> list[dict]:
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM sentiment_records
               WHERE company=? AND date>=? AND tenant_id=?
               ORDER BY date DESC""",
            (company, cutoff, tenant_id),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_all_records(tenant_id: str = None) -> list[dict]:
    """取全部记录。向量索引重建、知识图谱构建（公司/事件/共现抽取）用。
    tenant_id=None 返回所有租户（重建向量索引时需要）；否则按租户过滤。"""
    with get_conn() as conn:
        if tenant_id:
            rows = conn.execute(
                "SELECT * FROM sentiment_records WHERE tenant_id=?", (tenant_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM sentiment_records").fetchall()
        return [_row_to_dict(r) for r in rows]


def get_daily_records(date: str, tenant_id: str = "default") -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sentiment_records WHERE date=? AND tenant_id=? ORDER BY company",
            (date, tenant_id),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_daily_company_stats(date: str, tenant_id: str = "default") -> list[dict]:
    """按公司分组统计当日情感均值/条数/负面条数，供日报生成使用"""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT company,
                      COUNT(*) as cnt,
                      AVG(sentiment_score) as avg_score,
                      SUM(CASE WHEN sentiment_label='负面' THEN 1 ELSE 0 END) as negative_cnt
               FROM sentiment_records
               WHERE date=? AND tenant_id=?
               GROUP BY company
               ORDER BY cnt DESC""",
            (date, tenant_id),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_alert_stats(days: int = 30, tenant_id: str = "default") -> dict:
    """
    预警统计：初筛触发数、复核通过（属实）数、复核判定误报数、实际推送数。
    用于评估"规则初筛"和"LLM复核"这两道关卡各自的效果。
    """
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    with get_conn() as conn:
        row = conn.execute(
            """SELECT
                   COUNT(*) as total,
                   SUM(CASE WHEN need_alert=1 THEN 1 ELSE 0 END) as triggered,
                   SUM(CASE WHEN need_alert=1 AND alert_is_valid=1 THEN 1 ELSE 0 END) as valid,
                   SUM(CASE WHEN need_alert=1 AND alert_is_valid=0 THEN 1 ELSE 0 END) as invalid,
                   SUM(CASE WHEN pushed=1 THEN 1 ELSE 0 END) as pushed
               FROM sentiment_records
               WHERE date>=? AND tenant_id=?""",
            (cutoff, tenant_id),
        ).fetchone()
        return dict(row)


def get_alert_records(days: int = 30, only_triggered: bool = True, tenant_id: str = "default") -> list[dict]:
    """预警明细列表，用于看板下钻查看具体是哪些记录触发了预警/被判定误报"""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    with get_conn() as conn:
        sql = "SELECT * FROM sentiment_records WHERE date>=? AND tenant_id=?"
        params = [cutoff, tenant_id]
        if only_triggered:
            sql += " AND need_alert=1"
        sql += " ORDER BY date DESC"
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_company_daily_trend(company: str, days: int = 30, tenant_id: str = "default") -> list[dict]:
    """按天分组统计某公司情感均值/条数/负面条数，用于趋势图"""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT date,
                      COUNT(*) as cnt,
                      AVG(sentiment_score) as avg_score,
                      SUM(CASE WHEN sentiment_label='负面' THEN 1 ELSE 0 END) as negative_cnt
               FROM sentiment_records
               WHERE company=? AND date>=? AND tenant_id=?
               GROUP BY date
               ORDER BY date ASC""",
            (company, cutoff, tenant_id),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_all_companies(tenant_id: str = "default") -> list[str]:
    """已入库的公司列表，供Streamlit下拉选择用"""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT DISTINCT company FROM sentiment_records
               WHERE company IS NOT NULL AND company != '' AND company != '未知' AND tenant_id=?
               ORDER BY company""",
            (tenant_id,),
        ).fetchall()
        return [r["company"] for r in rows]


def search_text(keyword: str, company: str = None, limit: int = 10, tenant_id: str = "default") -> list[dict]:
    """FTS5关键词全文检索，作为向量语义检索之外的精确匹配补充。

    查询先用与索引侧相同的分词器切token，逐token双引号包裹拼成MATCH表达式：
    中文按词命中子串词（可转债/碳酸锂）；600519.SH之类含特殊字符的查询不再触发
    FTS5语法错误。两阶段匹配：先AND（精确优先）；零命中且多token时降级OR+bm25排序
    （二元组降级分词会产生文档中不存在的跨界二元组，AND会漏召）。
    无有效token或匹配异常时返回空列表，不抛错。
    """
    from storage.fts_tokenizer import fts_query_expr, fts_query_tokens
    tokens = fts_query_tokens(keyword or "")
    if not tokens:
        return []
    with get_conn() as conn:
        sql_base = """SELECT sentiment_records.*
                      FROM sentiment_fts
                      JOIN sentiment_records ON sentiment_fts.rowid = sentiment_records.id
                      WHERE sentiment_fts MATCH ? AND sentiment_records.tenant_id = ?"""
        if company:
            sql_base += " AND sentiment_records.company = ?"

        def _run(match_expr: str, order_by_rank: bool):
            sql, params = sql_base, [match_expr, tenant_id]
            if company:
                params.append(company)
            if order_by_rank:  # OR召回时按bm25相关性排序，命中最多的排前
                sql += " ORDER BY rank"
            sql += " LIMIT ?"
            params.append(limit)
            return conn.execute(sql, params).fetchall()

        try:
            # 第一阶段：AND，精确优先
            rows = _run(fts_query_expr(keyword, "AND"), order_by_rank=False)
            # 第二阶段：AND零命中且查询有多个token → OR+rank兜底
            if not rows and len(tokens) > 1:
                rows = _run(fts_query_expr(keyword, "OR"), order_by_rank=True)
        except sqlite3.OperationalError:
            # 极端情况下的最后防线：分词后仍触发FTS语法问题（如未来SQLite版本行为变化），
            # 关键词检索降级为空结果，绝不让检索层异常打断主流程
            return []
        return [_row_to_dict(r) for r in rows]


def get_event_type_distribution(days: int = 30, company: str = None, tenant_id: str = "default") -> list[dict]:
    """
    按事件类型统计条数（event_type改为受控词表后才能可靠GROUP BY）。
    company不传则统计该租户下全部公司，用于"最近哪类事件最多"这种跨公司宏观视角。
    """
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    with get_conn() as conn:
        sql = """SELECT event_type, COUNT(*) as cnt
                 FROM sentiment_records
                 WHERE date>=? AND tenant_id=?"""
        params = [cutoff, tenant_id]
        if company:
            sql += " AND company=?"
            params.append(company)
        sql += " GROUP BY event_type ORDER BY cnt DESC"
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def get_dimension_avg(company: str, days: int = 30, tenant_id: str = "default") -> dict:
    """
    某公司近N天的多维度情感均值。dimension_scores存的是JSON字符串，
    SQLite没有跨版本可靠的JSON聚合语法，这里取出记录后在Python里算均值；
    数据量大了以后可迁移到PostgreSQL用jsonb聚合函数优化。
    """
    records = get_records_by_company(company, days=days, tenant_id=tenant_id)
    from core.taxonomy import DIMENSIONS
    sums = {d: 0.0 for d in DIMENSIONS}
    counts = {d: 0 for d in DIMENSIONS}
    for r in records:
        scores = r.get("dimension_scores")
        if not scores:
            continue
        for d in DIMENSIONS:
            if d in scores:
                sums[d] += scores[d]
                counts[d] += 1
    return {
        d: (round(sums[d] / counts[d], 3) if counts[d] else None)
        for d in DIMENSIONS
    }


def get_company_window_stats(days: int = 7, tenant_id: str = "default") -> list[dict]:
    """
    按公司分组统计近N天的情感均值/条数/负面条数。
    和 get_daily_company_stats() 的区别：那个是"某一天"按公司分组，
    这个是"近N天整体"按公司分组——行业热度榜和多公司对比都要用到"一段时间的汇总"，
    不是"某一天的快照"，所以单独提供这个函数而不是复用 get_daily_company_stats。
    """
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT company,
                      COUNT(*) as cnt,
                      AVG(sentiment_score) as avg_score,
                      SUM(CASE WHEN sentiment_label='负面' THEN 1 ELSE 0 END) as negative_cnt
               FROM sentiment_records
               WHERE date>=? AND tenant_id=? AND company IS NOT NULL AND company != '' AND company != '未知'
               GROUP BY company
               ORDER BY cnt DESC""",
            (cutoff, tenant_id),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_industry_heat_ranking(days: int = 7, tenant_id: str = "default") -> list[dict]:
    """
    行业舆情热度榜：按行业聚合舆情量/平均情感分排行。
    行业归属复用知识图谱种子库的 resolve_industry()（与 market_data.provider 的代码
    解析共用同一份COMPANIES别名表），不查knowledge_graph的图存储——种子表是权威
    数据源，图存储里的"公司-行业"关系边也是从这份种子表建的，直接查表结果一致且少绕一层。
    返回：[{"industry","company_cnt","total_cnt","avg_score","negative_cnt"}, ...]，
    按 total_cnt 降序排列（"热度"=舆情条数）。
    """
    from knowledge_graph.seed_data import resolve_industry

    company_stats = get_company_window_stats(days=days, tenant_id=tenant_id)
    if not company_stats:
        return []

    by_industry: dict = {}
    for cs in company_stats:
        industry = resolve_industry(cs["company"])
        bucket = by_industry.setdefault(industry, {
            "industry": industry, "company_cnt": 0, "total_cnt": 0,
            "negative_cnt": 0, "_score_sum": 0.0, "_score_weight": 0,
        })
        bucket["company_cnt"] += 1
        bucket["total_cnt"] += cs["cnt"]
        bucket["negative_cnt"] += cs["negative_cnt"] or 0
        if cs["avg_score"] is not None:
            bucket["_score_sum"] += cs["avg_score"] * cs["cnt"]
            bucket["_score_weight"] += cs["cnt"]

    ranking = []
    for b in by_industry.values():
        avg_score = round(b["_score_sum"] / b["_score_weight"], 3) if b["_score_weight"] else None
        ranking.append({
            "industry": b["industry"],
            "company_cnt": b["company_cnt"],
            "total_cnt": b["total_cnt"],
            "negative_cnt": b["negative_cnt"],
            "avg_score": avg_score,
        })
    ranking.sort(key=lambda x: x["total_cnt"], reverse=True)
    return ranking
