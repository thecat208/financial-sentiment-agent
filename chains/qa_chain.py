"""
RAG问答Chain。只负责"给定检索到的文档，生成回答"，不负责检索本身（在 retrieval/retriever.py），
也不负责"检索不足要不要重试"（那是 graphs/qa_graph.py 的职责）。

- LLM客户端配置 timeout，失败按指数退避重试（见 core/retry.py），全部失败后降级返回提示语，
  不让单个LLM故障打崩整个问答流程。
- get_ll() 每次比对环境变量Key/接入地址/模型名与已建客户端，任一变化自动重建（支持运行时换Key）；
  OPENAI_BASE_URL 可指向兼容OpenAI协议的第三方服务（如DeepSeek），留空用官方端点。
- 幻觉控制（core/faithfulness.py）：QA_VERIFY=1 时用带编号资料+引用标注的 qa_prompt_cited
  供程序化校验；校验出风险后带具体风险原因受限重生成（strict参数），编排见 faithfulness.verify_with_regenerate()。
"""
import os

from langchain_core.output_parsers import StrOutputParser
from chains.prompts.qa_prompt import qa_prompt, qa_prompt_cited
from config.settings import settings
from core.retry import with_retry, with_retry_async
from core.compliance import append_disclaimer
from core.llm_usage_callback import usage_callback
from core.faithfulness import verify_enabled

_llm = None
_llm_key = None
# 两种模式的chain分别缓存（baseline=原prompt / cited=引用标注prompt）
_combine_chains = {"plain": None, "cited": None}


def get_llm():
    global _llm, _llm_key
    # 从环境变量实时取Key，支持运行时轮换（对应"敏感API密钥动态轮转"的轻量版）
    key = (
        os.getenv("OPENAI_API_KEY", "") if settings.LLM_PROVIDER == "openai"
        else os.getenv("ANTHROPIC_API_KEY", "")
    )
    # 客户端指纹：Key / 接入地址 / 模型名 任一变化都要重建，
    # 否则改了 .env（比如从官方OpenAI切到DeepSeek网关）进程内还在用旧客户端
    fingerprint = (
        settings.LLM_PROVIDER, key,
        settings.OPENAI_BASE_URL, settings.OPENAI_MODEL, settings.ANTHROPIC_MODEL,
    )
    if _llm is not None and fingerprint == _llm_key:
        return _llm

    if settings.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=key,
            base_url=settings.OPENAI_BASE_URL or None,   # 兼容OpenAI协议的第三方服务地址
            temperature=0,
            timeout=settings.LLM_TIMEOUT,   # 单次调用超时
            max_retries=0,                  # 关闭客户端内置重试，统一走 with_retry 精确控制次数
            callbacks=[usage_callback],     # 用量统计，见 core/llm_usage_callback.py
        )
    else:
        from langchain_anthropic import ChatAnthropic
        _llm = ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            api_key=key,
            temperature=0,
            timeout=settings.LLM_TIMEOUT,
            callbacks=[usage_callback],     # 用量统计，见 core/llm_usage_callback.py
        )
    _llm_key = fingerprint
    return _llm


def format_docs_numbered(docs: list) -> str:
    """把Document列表拼装成编号文本：[1] 正文（来源：xx，日期）。cited模式的资料格式，
    编号与回答里的[1][2]引用标注一一对应，供faithfulness第1层核查。"""
    lines = []
    for i, d in enumerate(docs or [], start=1):
        meta = getattr(d, "metadata", None) or {}
        lines.append(
            f"[{i}] {d.page_content}（来源：{meta.get('source', '未知来源')}，{meta.get('date', '未知日期')}）"
        )
    return "\n\n".join(lines)


def get_qa_combine_chain():
    """按当前模式（QA_VERIFY开=引用标注版 / 关=原版）返回生成chain，各自缓存。
    模式切换由评测脚本配合 reset_chains() 使用。"""
    mode = "cited" if verify_enabled() else "plain"
    if _combine_chains[mode] is None:
        llm = get_llm()
        prompt = qa_prompt_cited if mode == "cited" else qa_prompt
        _combine_chains[mode] = prompt | llm | StrOutputParser()
    return _combine_chains[mode]


def reset_chains():
    """清空chain缓存（评测脚本切换QA_VERIFY模式后调用，强制按新模式重建）"""
    global _combine_chains
    _combine_chains = {"plain": None, "cited": None}


@with_retry()
def _invoke_answer(query: str, docs: list, conversation_history: str = "",
                   strict: bool = False, strict_reason: str = "") -> str:
    """实际生成回答。失败抛异常交给with_retry重试；docs为空直接返回提示，不触发LLM"""
    if not docs:
        return "抱歉，未检索到与该问题相关的舆情资料，暂时无法回答。"

    user_input = query
    if strict:
        # 幻觉控制（问题③）：verify_node检出风险后的受限重生成。
        # 把"具体哪里编造了"写进约束，比泛泛的"不要编造"有效得多。
        user_input = (
            f"{query}\n\n"
            f"（系统提示：你上一次的回答被事实一致性校验拦截，问题：{strict_reason or '包含资料中没有的信息'}。"
            f"请严格只依据资料中明确出现的内容回答；资料无法支撑的部分，明确说明\"现有资料有限\"。）"
        )

    chain = get_qa_combine_chain()
    result = chain.invoke({
        "input": user_input,
        # cited模式下传预格式化的编号文本；plain模式保持原行为（Document列表）
        "context": format_docs_numbered(docs) if verify_enabled() else docs,
        "conversation_history": conversation_history or "",
    })
    # 输出防火墙：回答末尾强制拼接免责声明（幂等，失败/未检索提示不加）
    return append_disclaimer(result)


def generate_answer(query: str, docs: list, conversation_history: str = "",
                    strict: bool = False, strict_reason: str = "") -> str:
    """
    query: 用户问题
    docs: 已检索到的 Document 列表（由 retriever 提供）
    conversation_history: 多轮追问的短期记忆上下文（长期摘要+最近几轮），没有则传空串
    strict/strict_reason: 幻觉校验检出风险后的受限重生成（问题③）

    重试耗尽后降级返回提示语（降级兜底，不阻塞主流程）。
    """
    try:
        return _invoke_answer(query, docs, conversation_history, strict, strict_reason)
    except Exception as e:
        return f"抱歉，回答生成暂时不可用（服务繁忙或超时），请稍后再试。（{type(e).__name__}）"


# ================= 异步版 =================
# 判定/重试/兜底语义与同步版完全一致；区别仅在chain调用走 ainvoke +
# 重试退避用 with_retry_async（await asyncio.sleep，等待期间不阻塞事件循环）。
# LCEL chain（prompt|llm|parser）原生支持 ainvoke，HTTP等待期间事件循环
# 可以处理其他并发请求——这是异步化并发收益的核心来源。


@with_retry_async()
async def _ainvoke_answer(query: str, docs: list, conversation_history: str = "",
                          strict: bool = False, strict_reason: str = "") -> str:
    """实际生成回答（异步版）。失败抛异常交给with_retry_async重试；docs为空直接返回提示，不触发LLM"""
    if not docs:
        return "抱歉，未检索到与该问题相关的舆情资料，暂时无法回答。"

    user_input = query
    if strict:
        user_input = (
            f"{query}\n\n"
            f"（系统提示：你上一次的回答被事实一致性校验拦截，问题：{strict_reason or '包含资料中没有的信息'}。"
            f"请严格只依据资料中明确出现的内容回答；资料无法支撑的部分，明确说明\"现有资料有限\"。）"
        )

    chain = get_qa_combine_chain()
    result = await chain.ainvoke({
        "input": user_input,
        "context": format_docs_numbered(docs) if verify_enabled() else docs,
        "conversation_history": conversation_history or "",
    })
    return append_disclaimer(result)


async def generate_answer_async(query: str, docs: list, conversation_history: str = "",
                                strict: bool = False, strict_reason: str = "") -> str:
    """generate_answer 的异步版，降级语义一致：重试耗尽返回提示语，不抛异常。"""
    try:
        return await _ainvoke_answer(query, docs, conversation_history, strict, strict_reason)
    except Exception as e:
        return f"抱歉，回答生成暂时不可用（服务繁忙或超时），请稍后再试。（{type(e).__name__}）"
