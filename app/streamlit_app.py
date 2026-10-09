"""
运行方式：streamlit run app/streamlit_app.py
（需在项目根目录下运行，确保能正确import config/retrieval/chains/graphs/storage）

企业级多租户：顶部加了一个"当前组织/租户"选择框，
选中的租户会传给下面所有的问答/趋势图/日报/预警看板查询，实现数据隔离的
演示效果。这里只是最简单的"手选租户"，不是真正的登录鉴权——
生产环境需要接入企业的SSO/账号体系，从登录态里拿到tenant_id，
而不是让用户自己在下拉框里选（否则等于没做权限控制）。
这部分TODO在README的"企业级 & 多模态升级路线图"里有详细说明。
"""
import sys
import os
import uuid

# 确保能从项目根目录导入其他模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from graphs.qa_graph import ask
from graphs.router_graph import handle_query
from memory.memory_manager import get_status, clear_memory
from storage.repository import (
    get_all_companies,
    get_company_daily_trend,
    get_records_by_company,
    get_alert_stats,
    get_alert_records,
    get_dimension_avg,
    get_event_type_distribution,
    get_industry_heat_ranking,
    manual_override_alert,
)
from storage.usage_tracker import get_usage_stats
from storage.audit_log import get_audit_log
from analysis.backtest import get_sentiment_price_overlay, run_alert_backtest
from export.report_export import export_to_excel, export_to_pdf, load_report_json

st.set_page_config(
    page_title="金融舆情分析系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== 全局视觉样式（纯 CSS，不碰业务逻辑） ==========
st.markdown("""
<style>
/* 全局字体 */
html, body, [class*="css"], button, input {
    font-family: "PingFang SC", "Microsoft YaHei", -apple-system, "Segoe UI", sans-serif;
}
.stApp { background: #0d1017; }

/* 顶栏 hero 横幅 */
.hero {
    display: flex; align-items: center; gap: 16px;
    padding: 22px 28px; margin-bottom: 6px;
    background: linear-gradient(120deg, #1a2030 0%, #231a2e 55%, #2a1520 100%);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 16px;
}
.hero-icon {
    width: 52px; height: 52px; border-radius: 14px; flex: none;
    display: flex; align-items: center; justify-content: center;
    font-size: 28px;
    background: linear-gradient(135deg, #f43f5e, #b91c5c);
    box-shadow: 0 6px 18px rgba(244,63,94,.35);
}
.hero-title { font-size: 24px; font-weight: 700; color: #f5f7fb; letter-spacing: .5px; }
.hero-sub { font-size: 13px; color: #8b93a7; margin-top: 4px; }

/* 侧边栏 */
[data-testid="stSidebar"] {
    background: #11151f;
    border-right: 1px solid rgba(255,255,255,.06);
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h1 { color: #f5f7fb; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.08); }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #6d7689; font-size: 12px; }

/* 标签页 */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid rgba(255,255,255,.08); }
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 9px 9px 0 0;
    padding: 8px 14px; color: #8b93a7; font-weight: 500;
}
.stTabs [data-baseweb="tab"]:hover { background: rgba(255,255,255,.04); color: #e6eaf2; }
.stTabs [aria-selected="true"] { background: rgba(244,63,94,.13) !important; color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #f43f5e !important; height: 3px; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* 主按钮：渐变红 */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #f43f5e, #dc2648);
    border: none; border-radius: 10px;
    padding: .4rem 1.5rem; font-weight: 600; color: #fff;
    box-shadow: 0 4px 14px rgba(244,63,94,.3);
    transition: filter .15s ease, transform .15s ease;
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.12); transform: translateY(-1px); }
.stButton > button[kind="primary"]:active { transform: translateY(0); }

/* 次要按钮：示例 chip 胶囊 */
.stButton > button[kind="secondary"], .stButton > button:not([kind="primary"]) {
    background: #1a2030; color: #c3cbd9;
    border: 1px solid rgba(255,255,255,.09);
    border-radius: 999px; padding: .3rem .9rem;
    transition: all .15s ease;
}
.stButton > button[kind="secondary"]:hover, .stButton > button:not([kind="primary"]):hover {
    border-color: rgba(244,63,94,.55); color: #fff; background: rgba(244,63,94,.1);
}

/* 输入框 */
.stTextInput input, .stTextArea textarea {
    background: #161b26 !important;
    border: 1px solid rgba(255,255,255,.1) !important;
    border-radius: 10px !important;
    color: #e6eaf2 !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #f43f5e !important;
    box-shadow: 0 0 0 2px rgba(244,63,94,.22) !important;
}

/* 指标卡 */
[data-testid="stMetric"] {
    background: #151a26;
    border: 1px solid rgba(255,255,255,.06);
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,.25);
}
[data-testid="stMetricValue"] { color: #f5f7fb; }

/* 折叠面板 */
[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,.07) !important;
    border-radius: 10px !important;
    background: #12161f;
}
[data-testid="stExpander"] summary:hover { color: #fff; }

/* 提示框 / 表格圆角 */
[data-testid="stAlert"] { border-radius: 10px; }
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 10px; overflow: hidden;
}

/* 结果卡片 */
.result-card {
    background: #12161f;
    border: 1px solid rgba(255,255,255,.07);
    border-left: 3px solid #f43f5e;
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">📈</div>
        <div>
            <div class="hero-title">金融舆情分析系统</div>
            <div class="hero-sub">多租户舆情采集 · RAG 问答 · 日报生成 · 预警复核 · 舆情-行情联动</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("当前组织")
    CURRENT_TENANT_ID = st.text_input(
        "租户ID（企业多租户演示，生产环境应来自登录态）", value="default", key="tenant_id_input"
    ).strip() or "default"
    st.caption("个人/单租户场景保持默认值即可，不用管这个框")
    st.divider()

    # ---- 多轮追问记忆 ----
    # 用一个进程内会话ID串起同一浏览器会话的多轮对话；点"清空记忆"会换新ID
    st.header("对话记忆")
    if "conv_session_id" not in st.session_state:
        st.session_state.conv_session_id = uuid.uuid4().hex
    CURRENT_SESSION_ID = st.session_state.conv_session_id
    mem_status = get_status(CURRENT_TENANT_ID, CURRENT_SESSION_ID)
    st.caption(
        f"会话 {CURRENT_SESSION_ID[:8]}… | 短期 {mem_status['turn_count']} 轮 | "
        f"长期摘要 {'有' if mem_status['has_summary'] else '无'}"
    )
    if st.button("🧹 清空对话记忆"):
        clear_memory(CURRENT_TENANT_ID, CURRENT_SESSION_ID)
        st.session_state.conv_session_id = uuid.uuid4().hex
        st.rerun()
    st.divider()

tab_smart, tab_qa, tab_history, tab_report, tab_alert, tab_market, tab_compare, tab_usage = st.tabs(
    ["🧭 智能入口", "RAG 问答", "历史舆情 / 趋势图", "日报查看", "预警看板", "📊 舆情-行情联动", "🏭 多公司对比/行业热度", "📋 用量与审计"]
)

# ========== Tab 0: 智能入口（意图识别路由） ==========
with tab_smart:
    st.caption("不用挑标签页，直接说你想问的：问舆情、看日报、查预警统计，系统自动识别并路由到对应流程")

    # 示例问题：点击即查
    examples = ["查看今天的日报", "示例公司最近的舆情怎么样", "最近预警误报率高吗", "各公司舆情对比"]
    chip_clicked = None
    chip_cols = st.columns(len(examples))
    for col, ex in zip(chip_cols, examples):
        if col.button(ex, key=f"chip_{ex}"):
            chip_clicked = ex

    smart_query = st.text_input(
        "输入你的问题", placeholder="例如：示例公司最近舆情怎么样 / 看看今天的日报 / 最近预警误报率高吗",
        key="smart_query",
        label_visibility="collapsed",
    )
    smart_submit = st.button("发送", type="primary", key="smart_submit")

    submitted_query = smart_query.strip() if smart_submit and smart_query.strip() else chip_clicked

    if submitted_query:
        with st.spinner("识别意图并处理中..."):
            try:
                result = handle_query(submitted_query, tenant_id=CURRENT_TENANT_ID, session_id=CURRENT_SESSION_ID)
            except Exception as e:
                st.error(f"出错了：{e}")
                st.stop()

        intent_label = {
            "qa": "📚 问答检索", "report": "📰 日报", "alert_stats": "🚨 预警统计", "other": "❓ 未识别",
        }.get(result["intent"], result["intent"])
        st.markdown(
            f"""
            <div class="result-card">
                <span style="color:#8b93a7;font-size:12px;">识别到的意图</span>
                <span style="background:rgba(244,63,94,.15);color:#fb7185;border-radius:999px;
                              padding:2px 12px;font-size:12px;margin-left:8px;">{intent_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(result["text"])
    elif smart_submit:
        st.warning("请输入问题")

# ========== Tab 1: RAG 问答 ==========
with tab_qa:
    with st.sidebar:
        st.header("问答检索设置")
        qa_company = st.text_input("公司名称（可选，用于精确过滤）", "", key="qa_company")
        st.caption("留空则不限公司，直接做全局语义检索")

    query = st.text_input("请输入你的问题", placeholder="例如：示例公司最近的舆情怎么样？")
    submit = st.button("提问", type="primary")

    if submit and query.strip():
        with st.spinner("检索并生成回答中..."):
            try:
                result = ask(
                    query,
                    company=qa_company.strip() or None,
                    tenant_id=CURRENT_TENANT_ID,
                    session_id=CURRENT_SESSION_ID,
                )
            except Exception as e:
                st.error(f"出错了：{e}")
                st.stop()

        st.subheader("回答")
        st.write(result["answer"])
        st.caption(f"本次检索方式：{result['retrieve_desc']}")

        st.subheader("信息来源")
        if result["sources"]:
            for i, src in enumerate(result["sources"], 1):
                with st.expander(f"来源 {i}：{src['source']}（{src['date']}）"):
                    st.write(src["content_preview"] + " ...")
        else:
            st.info("未检索到相关舆情资料")
    elif submit:
        st.warning("请输入问题")

# ========== Tab 2: 历史舆情 / 趋势图 ==========
with tab_history:
    companies = get_all_companies(tenant_id=CURRENT_TENANT_ID)

    if not companies:
        st.info("目前SQLite里还没有已识别公司的舆情数据，先跑几次 scripts/run_pipeline.py 产生一些数据。")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_company = st.selectbox("选择公司", companies)
        with col2:
            days = st.slider("查看最近N天", min_value=7, max_value=90, value=30, step=7)

        trend = get_company_daily_trend(selected_company, days=days, tenant_id=CURRENT_TENANT_ID)

        if not trend:
            st.info(f"{selected_company} 在最近 {days} 天内没有舆情记录。")
        else:
            df = pd.DataFrame(trend)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

            st.subheader(f"{selected_company} 情感趋势（近{days}天）")
            st.line_chart(df["avg_score"], height=280)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("累计条数", int(df["cnt"].sum()))
            col_b.metric("平均情感分", round(df["avg_score"].mean(), 2))
            col_c.metric("负面条数", int(df["negative_cnt"].sum()))

            st.subheader("每日明细")
            st.dataframe(
                df.reset_index().rename(columns={
                    "date": "日期", "cnt": "条数",
                    "avg_score": "平均情感分", "negative_cnt": "负面条数",
                }),
                use_container_width=True,
            )

            st.subheader("多维度情感均值（近{}天）".format(days))
            dim_avg = get_dimension_avg(selected_company, days=days, tenant_id=CURRENT_TENANT_ID)
            dim_cols = st.columns(len(dim_avg))
            for col, (dim_name, dim_val) in zip(dim_cols, dim_avg.items()):
                col.metric(dim_name, dim_val if dim_val is not None else "暂无数据")
            st.caption("把总体情感分拆成业绩/管理层/行业前景/合规风险四个维度，"
                       "矛盾信号（如业绩正面但合规风险高）不会被互相抵消掉")

            event_dist = get_event_type_distribution(days=days, company=selected_company, tenant_id=CURRENT_TENANT_ID)
            if event_dist:
                st.caption("事件类型分布：" + "，".join(f"{e['event_type']} {e['cnt']}条" for e in event_dist))

            st.subheader("原始记录")
            records = get_records_by_company(selected_company, days=days, tenant_id=CURRENT_TENANT_ID)
            for r in records[:30]:  # 避免一次渲染太多
                label = r["sentiment_label"] or "未知"
                score = r["sentiment_score"]
                media_tag = {"image": "🖼️图片", "video": "🎬视频"}.get(r.get("media_type"), "")
                source_type_tag = {"机构公告": "📄公告", "社交媒体": "💬社交", "研报": "📊研报"}.get(
                    r.get("source_type"), ""
                )
                title = f"[{r['date']}] {label}（{score}）| {r['event_type'] or ''}"
                if source_type_tag:
                    title += f" | {source_type_tag}"
                if media_tag:
                    title += f" | {media_tag}"
                with st.expander(title):
                    st.write(r["cleaned_text"])
                    if r.get("source_type") and r.get("source_type") != "新闻资讯":
                        st.caption(f"来源：{r.get('source') or '未知'} "
                                   f"| 来源类型：{r['source_type']}（可信度：{r.get('source_credibility') or '中'}）")
                    if r.get("dimension_scores"):
                        st.caption(
                            "多维度打分：" + "，".join(f"{k} {v}" for k, v in r["dimension_scores"].items())
                        )
                    if r["need_alert"]:
                        st.caption(
                            f"曾触发预警初筛 | 复核有效={r['alert_is_valid']} | "
                            f"已推送={bool(r['pushed'])}"
                        )
            if len(records) > 30:
                st.caption(f"共 {len(records)} 条，仅展示最近30条")

# ========== Tab 3: 日报查看 ==========
with tab_report:
    reports_dir = os.path.join(os.getenv("REPORTS_DIR", "./reports"), CURRENT_TENANT_ID)

    if not os.path.isdir(reports_dir):
        st.info("还没有生成过日报，先运行 scripts/run_daily_report.py")
    else:
        files = sorted(
            [f for f in os.listdir(reports_dir) if f.endswith(".md")], reverse=True
        )
        if not files:
            st.info("还没有生成过日报，先运行 scripts/run_daily_report.py")
        else:
            selected_file = st.selectbox("选择日期", files)
            with open(os.path.join(reports_dir, selected_file), encoding="utf-8") as f:
                st.markdown(f.read())

            report_date = selected_file.replace(".md", "")
            report_json = load_report_json(report_date, tenant_id=CURRENT_TENANT_ID)

            if report_json is None:
                st.caption("（该日期的日报没有结构化数据，暂不支持导出；重新生成日报后即可导出）")
            else:
                exp_col1, exp_col2 = st.columns(2)
                with exp_col1:
                    if st.button("📊 导出为 Excel", key="export_xlsx"):
                        xlsx_path = export_to_excel(report_json)
                        with open(xlsx_path, "rb") as f:
                            st.download_button(
                                "下载 Excel 文件", f.read(),
                                file_name=f"舆情日报_{report_date}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="download_xlsx",
                            )
                with exp_col2:
                    if st.button("📄 导出为 PDF", key="export_pdf"):
                        pdf_path = export_to_pdf(report_json)
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                "下载 PDF 文件", f.read(),
                                file_name=f"舆情日报_{report_date}.pdf",
                                mime="application/pdf",
                                key="download_pdf",
                            )

# ========== Tab 4: 预警看板 ==========
with tab_alert:
    alert_days = st.slider("统计最近N天", min_value=7, max_value=90, value=30, step=7, key="alert_days")
    stats = get_alert_stats(days=alert_days, tenant_id=CURRENT_TENANT_ID)

    total = stats["total"] or 0
    triggered = stats["triggered"] or 0
    valid = stats["valid"] or 0
    invalid = stats["invalid"] or 0
    pushed = stats["pushed"] or 0

    if total == 0:
        st.info(f"最近{alert_days}天还没有舆情数据。")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总舆情条数", total)
        col2.metric("规则初筛触发数", triggered, help="命中关键词或情感分低于阈值的条数")
        col3.metric("LLM复核判定属实", valid)
        col4.metric("LLM复核判定误报", invalid)

        if triggered > 0:
            false_positive_rate = round(invalid / triggered * 100, 1)
            valid_rate = round(valid / triggered * 100, 1)
            st.subheader("规则初筛的误报率")
            st.write(
                f"规则初筛命中 **{triggered}** 条，经LLM复核后：**{valid_rate}%** 判定属实值得关注，"
                f"**{false_positive_rate}%** 判定为误报（如关键词命中但实际是正面消息）。"
            )
            st.caption(f"其中最终实际推送 {pushed} 条。误报率越高，说明规则初筛的关键词/阈值需要调整。")
        else:
            st.info("这段时间内规则初筛没有触发过预警。")

        st.subheader("预警触发明细")
        operator_name = st.text_input(
            "操作人（用于人工复核留痕，未登录系统时自行填写）", value="", key="alert_operator_name",
            help="人工覆盖LLM判定时会记入审计日志。本项目暂无真实登录鉴权，"
                 "这里填的名字不做身份校验，只是自报标识。",
        )
        records = get_alert_records(days=alert_days, only_triggered=True, tenant_id=CURRENT_TENANT_ID)
        if not records:
            st.info("没有触发过预警的记录。")
        else:
            for r in records[:30]:
                valid_flag = r["alert_is_valid"]
                status = "✅ 属实" if valid_flag == 1 else ("❌ 误报" if valid_flag == 0 else "⏳ 未复核")
                pushed_flag = "已推送" if r["pushed"] else "未推送"
                with st.expander(f"[{r['date']}] {r['company'] or '未知公司'} | {status} | {pushed_flag}"):
                    st.write(r["cleaned_text"])
                    st.caption(f"情感：{r['sentiment_label']}（{r['sentiment_score']}） | "
                               f"复核理由：{r['alert_review_reason'] or '（无）'}")

                    st.caption("人工复核（覆盖后会记入审计日志，可在下方\"用量与审计\"标签页查看）")
                    ov_col1, ov_col2, ov_col3 = st.columns([1, 1, 2])
                    with ov_col1:
                        confirm_valid = st.button("✅ 标记为属实", key=f"confirm_{r['id']}")
                    with ov_col2:
                        confirm_invalid = st.button("❌ 标记为误报", key=f"reject_{r['id']}")
                    with ov_col3:
                        note = st.text_input("备注（可选）", key=f"note_{r['id']}", label_visibility="collapsed",
                                              placeholder="备注（可选）")
                    if confirm_valid or confirm_invalid:
                        manual_override_alert(
                            r["id"], is_valid=confirm_valid, tenant_id=CURRENT_TENANT_ID,
                            user_id=operator_name or None, note=note,
                        )
                        st.success("已记录人工复核结果，刷新后生效")
                        st.rerun()
            if len(records) > 30:
                st.caption(f"共 {len(records)} 条，仅展示最近30条")

# ========== Tab 6.5: 用量与审计 ==========
with tab_usage:
    st.caption("用量统计和审计日志是SaaS化对外收费/满足强监管审计要求前的基础设施，本身不做计费。")

    usage_days = st.slider("统计最近N天", min_value=7, max_value=90, value=30, step=7, key="usage_days")
    usage_stats = get_usage_stats(CURRENT_TENANT_ID, days=usage_days)

    st.subheader("LLM 用量")
    uc1, uc2, uc3, uc4 = st.columns(4)
    uc1.metric("LLM调用次数", usage_stats["llm_calls"])
    uc2.metric("输入Token", usage_stats["input_tokens"])
    uc3.metric("输出Token", usage_stats["output_tokens"])
    uc4.metric("总Token", usage_stats["total_tokens"])

    if usage_stats["by_operation"]:
        st.caption("按场景拆分（analyze=情感分析，qa=RAG问答，report=日报生成，alert_review=预警复核，intent=意图识别）")
        st.dataframe(pd.DataFrame(usage_stats["by_operation"]).rename(
            columns={"operation": "场景", "calls": "调用次数", "total_tokens": "总Token"}
        ), use_container_width=True)
    else:
        st.info("这段时间内没有LLM调用记录（可能是还没配置API Key，或者没有触发过需要LLM的操作）。")

    st.subheader("API 调用统计")
    st.metric("API调用总次数", usage_stats["api_calls"])
    if usage_stats["api_calls_by_endpoint"]:
        st.dataframe(pd.DataFrame(usage_stats["api_calls_by_endpoint"]).rename(
            columns={"endpoint": "端点", "calls": "调用次数"}
        ), use_container_width=True)
    else:
        st.caption("这段时间内没有通过 /api/* 接口的调用记录（Streamlit本身的操作不计入这里）。")

    st.divider()
    st.subheader("审计日志")
    st.caption("目前只记录\"人工覆盖预警判定\"这一类关键操作；user_id是调用方自报的标识，"
               "不是经过身份验证的登录态（详见 storage/audit_log.py 顶部说明）。")
    audit_records = get_audit_log(CURRENT_TENANT_ID, days=usage_days)
    if not audit_records:
        st.info("这段时间内没有审计记录。")
    else:
        st.dataframe(pd.DataFrame(audit_records)[["created_at", "user_id", "action", "target_id", "detail"]].rename(
            columns={"created_at": "时间", "user_id": "操作人", "action": "操作类型",
                     "target_id": "目标记录ID", "detail": "详情"}
        ), use_container_width=True)

# ========== Tab 5: 舆情-行情联动 ==========
with tab_market:
    st.caption(
        "研究工具：把舆情信号和真实历史行情摆在一起，供你自己判断有没有参考价值，"
        "**不是投资建议**，也不构成任何买卖信号。"
    )
    companies_m = get_all_companies(tenant_id=CURRENT_TENANT_ID)

    if not companies_m:
        st.info("目前还没有已识别公司的舆情数据，先跑几次 scripts/run_pipeline.py 产生一些数据。")
    else:
        st.subheader("① 情感趋势 与 股价走势 叠加")
        col1, col2 = st.columns([2, 1])
        with col1:
            market_company = st.selectbox("选择公司", companies_m, key="market_company")
        with col2:
            market_days = st.slider("查看最近N天", min_value=14, max_value=180, value=60, step=7, key="market_days")

        overlay = get_sentiment_price_overlay(market_company, days=market_days, tenant_id=CURRENT_TENANT_ID)

        if overlay["is_mock"]:
            st.warning(
                f"⚠️ 当前展示的是**演示行情数据**（{overlay['code']} 未接入真实数据源或akshare未安装/网络不通），"
                "不代表真实走势，仅用于功能演示。"
            )

        if not overlay["points"]:
            st.info(f"{market_company} 在最近 {market_days} 天内没有舆情记录。")
        else:
            df_overlay = pd.DataFrame(overlay["points"]).set_index("date")
            oc1, oc2 = st.columns(2)
            with oc1:
                st.caption("情感均分趋势")
                st.line_chart(df_overlay["avg_score"], height=240)
            with oc2:
                st.caption(f"股价走势（{overlay['code']}，收盘价）")
                if df_overlay["close"].notna().any():
                    st.line_chart(df_overlay["close"], height=240)
                else:
                    st.info("该时间段内没有匹配到行情数据")

            if overlay["correlation"] is not None:
                st.metric("情感均分 vs 当日涨跌幅 相关系数", overlay["correlation"],
                           help=f"样本数={overlay['sample_size']}，仅供参考，不构成任何结论")
            else:
                st.caption("有效样本不足（需要至少3个匹配日期），暂不计算相关系数")

        st.divider()
        st.subheader("② 预警信号回测")
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            backtest_scope_label = st.selectbox(
                "统计范围", ["仅当前公司", "全部公司（当前租户）"], key="backtest_scope"
            )
        with bc2:
            backtest_holding = st.slider("持有交易日数", min_value=1, max_value=20, value=5, key="backtest_holding")
        with bc3:
            backtest_alert_scope_label = st.selectbox(
                "预警样本", ["规则初筛触发（全部）", "仅LLM复核判定属实"], key="backtest_alert_scope"
            )

        if st.button("运行回测", key="run_backtest"):
            backtest_company = market_company if backtest_scope_label == "仅当前公司" else None
            backtest_alert_scope = "valid_only" if backtest_alert_scope_label == "仅LLM复核判定属实" else "triggered"

            with st.spinner("拉取行情数据并计算中..."):
                bt = run_alert_backtest(
                    company=backtest_company,
                    tenant_id=CURRENT_TENANT_ID,
                    days=365,
                    holding_days=backtest_holding,
                    alert_scope=backtest_alert_scope,
                )

            if bt["samples"] == 0:
                st.info("没有找到可回测的样本（可能是没有触发过预警，或行情数据覆盖不到这些日期）。")
            else:
                if bt["low_sample_warning"]:
                    st.warning(f"⚠️ 样本量仅 {bt['samples']} 条，统计结果参考价值有限。")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("样本数", bt["samples"])
                mc2.metric(f"平均涨跌幅（持有{bt['holding_days']}日）", f"{bt['avg_return_pct']}%")
                mc3.metric("正案例胜率", f"{bt['win_rate']}%")
                mc4.metric("正/负案例数", f"{bt['positive_count']} / {bt['negative_count']}")

                st.caption("样本明细")
                st.dataframe(
                    pd.DataFrame(bt["details"]).rename(columns={
                        "date": "预警日期", "company": "公司", "entry_close": "触发日收盘价",
                        "exit_close": f"持有{bt['holding_days']}日后收盘价", "return_pct": "涨跌幅(%)",
                    }),
                    use_container_width=True,
                )

        st.caption(overlay.get("disclaimer", "").strip("*\n-") if companies_m else "")

# ========== Tab 6: 多公司对比 / 行业热度榜 ==========
with tab_compare:
    companies_c = get_all_companies(tenant_id=CURRENT_TENANT_ID)

    if not companies_c:
        st.info("目前还没有已识别公司的舆情数据，先跑几次 scripts/run_pipeline.py 产生一些数据。")
    else:
        st.subheader("① 多公司情感趋势对比")
        compare_companies = st.multiselect(
            "选择2-5家公司", companies_c,
            default=companies_c[: min(3, len(companies_c))],
            max_selections=5, key="compare_companies",
        )
        compare_days = st.slider("查看最近N天", min_value=7, max_value=180, value=30, step=7, key="compare_days")

        if len(compare_companies) < 2:
            st.info("至少选择2家公司才能做对比。")
        else:
            trend_frames = []
            record_counts = {}
            for c in compare_companies:
                trend = get_company_daily_trend(c, days=compare_days, tenant_id=CURRENT_TENANT_ID)
                record_counts[c] = len(trend)
                if trend:
                    df_c = pd.DataFrame(trend)[["date", "avg_score"]].set_index("date")
                    df_c.columns = [c]
                    trend_frames.append(df_c)

            if not trend_frames:
                st.info("所选公司在这个时间范围内都没有舆情记录。")
            else:
                merged = pd.concat(trend_frames, axis=1).sort_index()
                st.caption("情感均分趋势叠加（每条线一家公司）")
                st.line_chart(merged, height=320)

                st.caption("同期汇总对比")
                st.dataframe(
                    pd.DataFrame([
                        {
                            "公司": c,
                            "条数": record_counts[c],
                            "情感均分": round(merged[c].mean(), 3) if c in merged.columns and merged[c].notna().any() else None,
                        }
                        for c in compare_companies
                    ]),
                    use_container_width=True,
                )

        st.divider()
        st.subheader("② 行业舆情热度榜")
        ranking_days = st.slider("统计窗口（天）", min_value=7, max_value=90, value=7, step=7, key="ranking_days")
        ranking = get_industry_heat_ranking(days=ranking_days, tenant_id=CURRENT_TENANT_ID)

        if not ranking:
            st.info("这个时间范围内没有数据。")
        else:
            df_ranking = pd.DataFrame(ranking).rename(columns={
                "industry": "行业", "company_cnt": "公司数", "total_cnt": "舆情条数",
                "negative_cnt": "负面条数", "avg_score": "情感均分",
            })
            rc1, rc2 = st.columns([2, 1])
            with rc1:
                st.caption("按舆情条数（热度）排序")
                st.dataframe(df_ranking, use_container_width=True, hide_index=True)
            with rc2:
                st.caption("热度分布")
                st.bar_chart(df_ranking.set_index("行业")["舆情条数"], height=280)

            st.caption("说明：行业归属来自内置的公司-行业种子库（knowledge_graph/seed_data.py），"
                       "覆盖范围有限，未收录的公司会归入\"其他行业\"。")
