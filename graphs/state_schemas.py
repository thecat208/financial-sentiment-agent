from typing import TypedDict, List, Optional
from langchain_core.documents import Document


class QAState(TypedDict):
    """RAG问答流程的状态"""
    query: str
    company: Optional[str]
    tenant_id: str              # 企业级多租户隔离，个人单租户场景下固定为"default"
    retrieved_docs: List[Document]
    retrieve_desc: str          # 记录本次实际用的检索策略，便于调试/展示
    retry_count: int
    answer: Optional[str]
    sources: List[dict]
    conversation_history: str     # 多轮追问的短期记忆上下文，来自 memory.memory_manager.build_context
    docs_budget: int              # Token控制：检索文档可用的token预算（ask()里按模型上下文算出）
    anchor_entities: List[str]    # 关键信息锚定：从当前问题提取的公司/股票代码等实体
    # ---- 幻觉控制 ----
    used_docs: List[Document]     # 实际送入LLM的文档（trim后），校验层以此为准核对事实
    verification: Optional[dict]  # core/faithfulness.verify_answer 的校验结果（审计/展示用）
    verify_retries: int           # 幻觉重生成已用次数（上限1次，防止无限循环）


class PipelineState(TypedDict):
    """舆情数据处理主流程（入库 → 分析 → 预警）的状态"""
    raw_text: str
    source: str
    date: str
    tenant_id: str              # 企业级多租户隔离，个人单租户场景下固定为"default"
    media_type: str             # "text" / "image" / "video"，标记原始输入的媒体类型
    media_path: Optional[str]   # 原始图片/视频文件路径，text类型为None，便于溯源审计
    source_type: str            # 来源类型，受控词表见 core/taxonomy.py（机构公告/新闻资讯/研报/社交媒体/其他）
    cleaned_text: str
    sentiment_label: Optional[str]     # 正面/中性/负面（总体情感，兼容原有下游）
    sentiment_score: Optional[float]   # -1 到 1（总体情感强度，兼容原有下游）
    dimension_scores: Optional[dict]   # 多维度情感：{"业绩":x,"管理层":x,"行业前景":x,"合规风险":x}
    company: Optional[str]
    event_type: Optional[str]          # 受控词表内的标准化类目，见 core/taxonomy.py
    analysis_reason: Optional[str]
    need_alert: bool
    alert_is_valid: Optional[bool]
    alert_review_reason: Optional[str]
    db_id: Optional[int]       # SQLite记录id，写入后由update_review_node/push_node回填后续字段用


class ReportState(TypedDict):
    """每日简报生成流程的状态"""
    date: str
    tenant_id: str              # 企业级多租户隔离，个人单租户场景下固定为"default"
    company_stats: List[dict]        # 按公司分组的统计（条数/均分/负面数），来自SQLite
    company_summaries: List[dict]    # 每个公司的LLM生成小结 [{company, summary, stats}]
    report_text: Optional[str]
    report_path: Optional[str]
