"""
结构化输出的兼容层（多provider / 多网关适配）。

各Chain原本直接 `get_llm().with_structured_output(Schema)`，但大量"兼容OpenAI协议"的第三方网关
只支持 response_format=json_object，遇到 json_schema 直接返回400；function_calling 在 DeepSeek
思考（thinking）模式下也返回400。这类400原先被上层兜底吞掉、难以定位，故统一收口到这里：

- invoke_structured(schema, prompt) 按 settings.LLM_STRUCTURED_METHOD 调用，默认 "auto"：先按
  最严格的 json_schema 调，确认是"方式不被支持"类400时自动降级 json_mode 重试一次，成功方式记在
  进程内（后续不再浪费试探请求）。json_mode 不校验schema，会在提示词末尾补上字段格式说明。
- 非"方式不支持"的错误（超时/鉴权失败/限流/模型名错）一律原样抛出，交给 core/retry.py 重试或
  链路兜底，避免把"Key失效"误判成"格式不支持"而掩盖真正的故障。
"""
from langchain_core.output_parsers import PydanticOutputParser

from chains.qa_chain import get_llm
from config.settings import settings
from core.logger import get_logger

logger = get_logger("chain.structured")

# 进程内记住第一次探测成功的方式：一次降级试探即可，同进程后续调用直接用
_resolved_method: str | None = None

# 支持显式指定的结构化输出方式
VALID_METHODS = ("json_schema", "function_calling", "json_mode")

# auto模式下的尝试顺序：先严格后宽松。json_schema 能被服务端校验schema、约束最硬；
# 不支持时退到 json_mode（只要求输出是合法JSON对象，字段靠提示词约束）。
AUTO_ORDER = ("json_schema", "json_mode")

# 判定"服务端不支持这种结构化输出方式"的关键词（配合状态码400一起判断）
_UNSUPPORTED_MARKERS = (
    "response_format",       # DeepSeek: This response_format type is unavailable now
    "json_schema",           # 网关只支持json_object时的各种改述
    "tool_choice",           # DeepSeek思考模式: Thinking mode does not support this tool_choice
    "structured output",
)


def _configured_method() -> str:
    """读配置，非法值一律按auto处理（配置写错不该把主流程打挂）"""
    return (settings.LLM_STRUCTURED_METHOD or "auto").strip().lower()


def _candidate_methods() -> list[str]:
    """本次调用要依次尝试的方式列表"""
    if _resolved_method:
        return [_resolved_method]
    configured = _configured_method()
    if configured in VALID_METHODS:
        return [configured]
    return list(AUTO_ORDER)


def _status_code(exc: Exception) -> int | None:
    """取HTTP状态码。openai/anthropic的异常都带status_code属性"""
    return getattr(exc, "status_code", None)


def is_unsupported_method_error(exc: Exception) -> bool:
    """
    这个异常是不是"服务端不支持当前结构化输出方式"造成的400？
    只有这种错误才允许自动降级；其余错误（超时/鉴权/限流/模型名错）都要原样抛出，
    否则会把一次真实故障悄悄降级成"换种方式再试一次"，掩盖问题。
    """
    if _status_code(exc) != 400:
        return False
    message = str(exc).lower()
    return any(marker in message for marker in _UNSUPPORTED_MARKERS)


def _invoke_once(schema, prompt: str, method: str):
    """用指定方式调一次LLM并把返回解析成schema实例（解析失败抛异常）"""
    llm = get_llm().with_structured_output(schema, method=method)
    if method == "json_mode":
        # json_mode只保证"输出是合法JSON"，不保证字段名对得上，必须把schema规则写进提示词
        prompt = prompt + "\n\n" + PydanticOutputParser(pydantic_object=schema).get_format_instructions()
    return llm.invoke(prompt)


def invoke_structured(schema, prompt: str):
    """
    schema: pydantic模型类；prompt: 已渲染好的提示词（字符串）。

    返回 schema 的实例；调用失败/解析失败时抛异常，由调用方的兜底逻辑决定怎么降级
    （本函数只负责"换一种结构化输出方式再试一次"，不负责重试网络错误——那是 core/retry.py 的事）。
    """
    global _resolved_method
    candidates = _candidate_methods()
    last_exc: Exception | None = None

    for method in candidates:
        try:
            result = _invoke_once(schema, prompt, method)
        except Exception as e:
            if not is_unsupported_method_error(e):
                raise
            last_exc = e
            logger.warning("服务端不支持结构化输出方式 %s，自动降级重试：%s", method, e)
            continue
        if _resolved_method != method:
            logger.info("结构化输出方式确定：%s（模型=%s）", method, settings.OPENAI_MODEL)
        _resolved_method = method
        return result

    if len(candidates) == 1 and _configured_method() in VALID_METHODS:
        logger.error(
            "LLM_STRUCTURED_METHOD=%s 在当前服务端不可用，把它改成 auto（自动降级到json_mode）",
            _configured_method(),
        )
    raise last_exc


def reset_method_cache():
    """清掉"已探测成功方式"的进程内缓存。切换模型/网关后需要重新探测时调用（也方便测试）"""
    global _resolved_method
    _resolved_method = None


# ================= 异步版 =================
# 逻辑与同步版完全一致（共用 _resolved_method 探测缓存，避免两条路径各试一遍）；
# 区别仅在 LLM 调用走 ainvoke——等待期间不阻塞事件循环，并发请求才能互相穿插。

async def _ainvoke_once(schema, prompt: str, method: str):
    """异步版_invoke_once：用指定方式调一次LLM并把返回解析成schema实例"""
    llm = get_llm().with_structured_output(schema, method=method)
    if method == "json_mode":
        prompt = prompt + "\n\n" + PydanticOutputParser(pydantic_object=schema).get_format_instructions()
    return await llm.ainvoke(prompt)


async def ainvoke_structured(schema, prompt: str):
    """
    invoke_structured 的异步版。schema/参数/异常语义完全一致：
    返回 schema 实例；"方式不支持"自动降级重试，其余异常原样抛出交给上层重试/兜底。
    """
    global _resolved_method
    candidates = _candidate_methods()
    last_exc: Exception | None = None

    for method in candidates:
        try:
            result = await _ainvoke_once(schema, prompt, method)
        except Exception as e:
            if not is_unsupported_method_error(e):
                raise
            last_exc = e
            logger.warning("服务端不支持结构化输出方式 %s，自动降级重试：%s", method, e)
            continue
        if _resolved_method != method:
            logger.info("结构化输出方式确定：%s（模型=%s）", method, settings.OPENAI_MODEL)
        _resolved_method = method
        return result

    if len(candidates) == 1 and _configured_method() in VALID_METHODS:
        logger.error(
            "LLM_STRUCTURED_METHOD=%s 在当前服务端不可用，把它改成 auto（自动降级到json_mode）",
            _configured_method(),
        )
    raise last_exc
