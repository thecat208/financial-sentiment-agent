from langchain_core.prompts import ChatPromptTemplate

REPORT_SYSTEM_PROMPT = """你是金融舆情日报撰写助手。请根据某公司当日的舆情统计数据和代表性原文，写一段简报小节。

要求：
1. 先给出情感倾向的整体判断（结合平均分和负面条数）
2. 结合1-2条代表性内容说明具体原因，不要空泛地说"舆情较多"
3. 语言简洁客观，2-4句话即可
4. 不给出任何投资建议，只做事实性归纳
"""

report_prompt = ChatPromptTemplate.from_messages([
    ("system", REPORT_SYSTEM_PROMPT),
    ("human", "统计数据：{stats}\n\n代表性内容：\n{samples}\n\n请生成该公司的简报小节。"),
])
