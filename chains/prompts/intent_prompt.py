"""
意图识别Prompt。金融舆情场景的需求是"自然语言查询的意图识别"：
判断用户到底想问答、看日报，还是查预警统计。
"""
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class IntentResult(BaseModel):
    """意图识别结果"""
    intent: str = Field(
        description="取值仅限以下之一：qa（问答检索）/ report（请求查看或生成日报/简报）/ alert_stats（询问预警统计信息）/ other（无法归类或闲聊）"
    )
    company: str = Field(description="用户提到的公司名称，没有明确提及则返回空字符串")
    date: str = Field(description="用户提到的具体日期，格式YYYY-MM-DD，没有则返回空字符串")


INTENT_TEMPLATE = """你是金融舆情系统的意图识别助手，需要判断用户输入属于哪种类型，并提取其中提到的公司名和日期。

用户输入：{query}
"""

intent_prompt = PromptTemplate(
    template=INTENT_TEMPLATE,
    input_variables=["query"],
)
