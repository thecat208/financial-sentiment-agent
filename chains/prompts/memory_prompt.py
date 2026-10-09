"""
记忆压缩Prompt。当对话累计轮数超过阈值时，用LLM把"旧长期摘要 + 新一批对话"合并压缩成
新的长期摘要，防止关键信息（用户跟踪的公司/事件、追问过的未答问题、表达过的偏好）
在短期窗口淘汰时丢失。
"""
from langchain_core.prompts import PromptTemplate

MEMORY_SUMMARY_TEMPLATE = """你是金融舆情分析系统的长期记忆助手。请把一段多轮对话压缩成"长期记忆摘要"，
只保留对后续继续对话仍有价值的信息，例如：
- 用户一直在跟踪的公司、行业或具体事件
- 用户问过但还没得到明确答案的问题
- 用户表达过的偏好（如"只看负面舆情"、"关注预警"）

要求：
1. 用简洁的中文要点列出，不要复述对话细节，不要加markdown标题
2. 如果已有旧摘要，请把旧摘要与新对话合并、去重，不要重复列出相同信息
3. 摘要要能让完全没看过这些对话的人读懂上下文

旧摘要：
{prev_summary}

新对话：
{dialogue}
"""

memory_summary_prompt = PromptTemplate(
    template=MEMORY_SUMMARY_TEMPLATE,
    input_variables=["prev_summary", "dialogue"],
)
