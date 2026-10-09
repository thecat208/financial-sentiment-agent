from typing import Literal
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from core.taxonomy import EVENT_CATEGORIES

# Literal类型需要一个固定的tuple，供pydantic生成JSON Schema的enum约束，
# 让LLM结构化输出时只能从受控词表里选，选不出来对应类目时用"其他"兜底。
_EventType = Literal[tuple(EVENT_CATEGORIES)]


class AnalysisResult(BaseModel):
    """
    情感分析结果：多维度情感 + 标准化事件分类。

    sentiment_label / sentiment_score 保留"总体"情感判断，预警规则（alert/rules.py）、
    历史趋势图等下游都还在用这两个字段。新增的四个维度分数评估"业绩""管理层""行业前景"
    "合规风险"，让"业绩超预期但管理层被质疑"这种矛盾信号能被完整保留；event_type 为
    受控词表（Literal约束），方便跨公司/跨时间的标准化统计。
    """
    sentiment_label: str = Field(description="总体情感倾向，只能是：正面、中性、负面 三者之一")
    sentiment_score: float = Field(description="总体情感强度，-1到1之间的小数，越负越负面，越正越正面")

    score_performance: float = Field(
        description="「业绩」维度打分，-1到1之间：营收利润、业绩预期兑现情况的正负面程度；"
                    "文本未涉及该维度则给0"
    )
    score_management: float = Field(
        description="「管理层」维度打分，-1到1之间：高管/董监事可信度、稳定性、治理层面的正负面程度；"
                    "文本未涉及该维度则给0"
    )
    score_industry: float = Field(
        description="「行业前景」维度打分，-1到1之间：所在行业/赛道景气度、政策环境、竞争格局的正负面程度；"
                    "文本未涉及该维度则给0"
    )
    score_compliance: float = Field(
        description="「合规风险」维度打分，-1到1之间：监管/法律/诉讼/信息披露等合规层面的风险程度，"
                    "风险越高越负；文本未涉及该维度则给0"
    )

    company: str = Field(description="文本中提及的主要相关公司名称，没有明确公司则填 无")
    event_type: _EventType = Field(description=f"事件类型，必须从以下类目中选一个：{'/'.join(EVENT_CATEGORIES)}")
    reason: str = Field(description="判断依据，一句话说明，如果各维度打分不一致（比如业绩正面但合规负面），"
                                     "在这里简要说明是哪几个维度在拉扯")


ANALYSIS_TEMPLATE = """你是一名专业的金融舆情分析助手，请分析以下文本，完成情感判断和关键信息抽取。

需要注意金融领域的特殊表达：
- 反讽或调侃语气（如某些自媒体标题）应结合实际内容判断真实情感，不要被表面词语误导
- 出现"卖出评级""下调评级"等专业术语时，这是分析师的观点陈述，需结合上下文判断是否代表实质负面消息

多维度打分说明（重要）：
- 一条舆情经常同时涉及多个维度，且方向可能矛盾，比如"业绩超预期但管理层被质疑挪用资金"——
  这种情况请如实给"业绩"打正分、"管理层"打负分，不要因为要凑一个总体印象而互相抵消打成0
- 文本完全没涉及某个维度就给0分，不要强行推测
- sentiment_label/sentiment_score 是总体印象，可以和某个单一维度分数不同
  （比如业绩正面但合规风险很高，总体仍可能判断为负面）

事件类型必须从给定的受控词表里选一个最贴切的，选不出来就用"其他"，不要自己发明新类目。

文本：
{text}
"""

analysis_prompt = PromptTemplate(
    template=ANALYSIS_TEMPLATE,
    input_variables=["text"],
)
