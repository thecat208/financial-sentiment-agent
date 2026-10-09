"""
请求/响应数据模型。只做请求校验和响应结构约束，不包含任何业务逻辑
——具体字段含义参考对应的 storage/repository.py、graphs/*.py 里的函数注释。
"""
from typing import Optional, Any
from pydantic import BaseModel, Field


# ---------- 通用 ----------

class ErrorResponse(BaseModel):
    detail: str


# ---------- 智能入口 / 问答 ----------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="自然语言问题，如'贵州茅台最近舆情怎么样'")
    session_id: Optional[str] = Field(None, description="多轮追问的会话ID，不传则单轮问答不带上下文")


class QueryResponse(BaseModel):
    intent: str
    text: Optional[str]
    payload: Optional[dict] = None


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    company: Optional[str] = Field(None, description="指定公司可以缩小检索范围，不传则全局检索")
    session_id: Optional[str] = None


# ---------- 舆情数据提交 ----------

class IngestRequest(BaseModel):
    raw_text: str = Field(..., min_length=1, description="原始文本内容")
    source: str = Field("API提交", description="具体来源名称，如'某新闻站'")
    date: Optional[str] = Field(None, description="YYYY-MM-DD，不传则取当天")
    source_type: Optional[str] = Field(
        None, description="来源类型：机构公告/新闻资讯/研报/社交媒体/其他，不传则按'新闻资讯'处理"
    )
    async_mode: bool = Field(
        False, description="True=提交到后台队列异步处理并立即返回任务ID；False=同步处理并等待分析结果"
    )


class IngestSyncResponse(BaseModel):
    company: Optional[str]
    event_type: Optional[str]
    sentiment_label: Optional[str]
    sentiment_score: Optional[float]
    dimension_scores: Optional[dict]
    need_alert: Optional[bool]
    db_id: Optional[int]


class IngestAsyncResponse(BaseModel):
    task_id: str
    status: str = "submitted"


# ---------- 公司与趋势 ----------

class CompanyTrendPoint(BaseModel):
    date: str
    cnt: int
    avg_score: Optional[float]
    negative_cnt: int


class CompanyCompareRequest(BaseModel):
    companies: list[str] = Field(..., min_length=2, max_length=5, description="2-5家公司名称")
    days: int = Field(30, ge=1, le=365)


# ---------- 预警 ----------

class AlertStats(BaseModel):
    total: int
    triggered: int
    valid: int
    invalid: int
    pushed: int


# ---------- 行情联动 / 回测 ----------

class BacktestRequest(BaseModel):
    company: Optional[str] = Field(None, description="不传则统计该租户下所有公司")
    days: int = Field(90, ge=1, le=730)
    holding_days: Optional[int] = Field(None, ge=1, le=60)
    alert_scope: str = Field("triggered", pattern="^(triggered|valid_only)$")


# ---------- 用量统计 / 审计----------

class ManualOverrideRequest(BaseModel):
    is_valid: bool = Field(..., description="True=标记为属实，False=标记为误报")
    note: Optional[str] = Field(None, description="备注，会记入审计日志")
    user_id: Optional[str] = Field(
        None, description="操作人标识，会记入审计日志。放在请求体而不是请求头，"
                           "是因为HTTP请求头按规范只能是ASCII，中文操作人姓名放header会导致"
                           "很多HTTP客户端报编码错误。"
    )
