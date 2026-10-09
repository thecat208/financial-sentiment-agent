from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class ReviewResult(BaseModel):
    """预警复核结果"""
    is_valid: bool = Field(description="是否值得推送预警，只能填 true 或 false")
    reason: str = Field(description="判断理由，一句话说明")


REVIEW_TEMPLATE = """以下是一条被规则初筛标记为"可能需要预警"的金融舆情文本，请复核判断是否真的值得推送预警。

以下情况不应判定为有效预警（is_valid=false）：
- 只是转发/引用旧闻，没有新增实质风险信息
- 关键词命中但实际语境是正面或中性（例如"顺利通过监管审查"包含"监管"一词但并非负面）
- 泛泛的行业评论，未针对具体公司的实质性风险

文本：
{text}

初步情感分析结果：{sentiment_label}（强度：{sentiment_score}）
"""

alert_review_prompt = PromptTemplate(
    template=REVIEW_TEMPLATE,
    input_variables=["text", "sentiment_label", "sentiment_score"],
)
