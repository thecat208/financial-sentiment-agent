"""
记忆管理器。

混合记忆池策略：
  1. 短期滑动窗口：保留最近 N 轮对话原文，直接拼进QA Prompt，供LLM理解指代
  2. 长期摘要记忆：当累计对话达到阈值轮数时，用LLM把（旧摘要 + 待压缩轮次）
     压缩成新的长期摘要，然后清空被压缩的轮次
对外接口：
  - build_context(tenant_id, session_id)：拼出给LLM的上下文文本（长期摘要 + 最近几轮），无记忆时返回空串
  - record_turn(tenant_id, session_id, user, assistant)：记录一轮对话，触发压缩判断
  - get_status(tenant_id, session_id) / clear_memory(tenant_id, session_id)：查看记忆状态 / 清空会话记忆
"""
from memory.session_store import InMemorySessionStore, RedisSessionStore
from chains.prompts.memory_prompt import memory_summary_prompt
from chains.qa_chain import get_llm
from core.token_budget import trim_turns, fit_text
from core.retry import with_retry
from config.settings import settings

_store = None


def get_store():
    """
    返回会话存储。优先Redis（跨进程共享 + 热会话TTL滑动续期），
    Redis不可用/未启动时自动降级为进程内内存存储，不阻塞主流程。
    """
    global _store
    if _store is None:
        try:
            import redis
            client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
            client.ping()
            _store = RedisSessionStore(client, ttl=settings.SESSION_TTL_HOT)
        except Exception:
            _store = InMemorySessionStore()
    return _store


def build_context(
    tenant_id: str,
    session_id: str,
    max_recent_turns: int = None,
    token_budget: int = None,
    anchor_entities: list = None,
) -> str:
    """
    组装给LLM的对话上下文，格式：
        [前期对话摘要] ...（如有）
        [最近对话]
        用户：...
        助手：...

    1. 长期摘要：如果有，直接拼进上下文；否则为空
    2. 最近对话：保留最近 N 轮对话原文，直接拼进QA Prompt，供LLM理解指代
    3. 如果有token预算限制，则使用token预算限制来裁剪对话；否则使用max_recent_turns来限制最近对话轮数
    4. 如果有锚实体列表，则使用锚实体列表来裁剪对话；否则使用token预算限制来裁剪对话
    """
    max_recent = max_recent_turns or settings.MEMORY_CONTEXT_RECENT_TURNS
    sess = get_store().get_session(tenant_id, session_id)

    summary = sess.get("summary") or ""
    if summary:
        summary = fit_text(summary, 400)  # 长期摘要本身也不允许无限膨胀

    turns = sess["turns"]
    if token_budget:
        turns = trim_turns(turns, token_budget, anchor_entities)
    elif max_recent > 0:
        turns = turns[-max_recent:]

    parts = []
    if summary:
        parts.append(f"[前期对话摘要] {summary}")

    if turns:
        lines = []
        for t in turns:
            lines.append(f"用户：{t['user']}")
            lines.append(f"助手：{t['assistant']}")
        parts.append("[最近对话]\n" + "\n".join(lines))

    return "\n\n".join(parts)


def record_turn(tenant_id: str, session_id: str, user: str, assistant: str):
    """记录一轮对话；累计轮数达到阈值时触发LLM压缩（被淘汰的轮次并入长期摘要）"""
    sess = get_store().add_turn(tenant_id, session_id, user, assistant)
    if len(sess["turns"]) >= settings.MEMORY_SUMMARY_TRIGGER_TURNS:
        _summarize_and_trim(tenant_id, session_id)


def _summarize_and_trim(tenant_id: str, session_id: str):
    """把"短期窗口去掉最近KEEP_TURNS轮之外"的对话交给LLM压缩，并保留最近几轮原文"""
    sess = get_store().get_session(tenant_id, session_id)
    keep_n = settings.MEMORY_TRIM_KEEP_TURNS

    to_compress = sess["turns"][:-keep_n] if keep_n > 0 else list(sess["turns"])
    keep = sess["turns"][-keep_n:] if keep_n > 0 else []

    if not to_compress:
        return

    new_summary = _llm_compress(sess.get("summary") or "", to_compress)
    get_store().set_summary(tenant_id, session_id, new_summary)
    get_store().replace_turns(tenant_id, session_id, keep)


@with_retry()
def _invoke_compress(prev_summary: str, dialogue: str) -> str:
    """实际调用LLM压缩记忆。失败抛异常交给with_retry重试"""
    llm = get_llm()
    prompt_value = memory_summary_prompt.format(prev_summary=prev_summary, dialogue=dialogue)
    resp = llm.invoke(prompt_value)
    return resp.content if hasattr(resp, "content") else str(resp)


def _llm_compress(prev_summary: str, turns: list) -> str:
    """调用LLM把旧摘要 + 新对话合并成新的长期摘要。重试耗尽后降级为保留原文，不阻塞主流程"""
    dialogue = "\n".join(
        f"用户：{t['user']}\n助手：{t['assistant']}" for t in turns
    )
    try:
        return _invoke_compress(prev_summary, dialogue)
    except Exception:
        return prev_summary or dialogue


def get_status(tenant_id: str, session_id: str) -> dict:
    """查看当前记忆状态，供前端展示/调试"""
    sess = get_store().get_session(tenant_id, session_id)
    return {
        "turn_count": len(sess["turns"]),
        "has_summary": bool(sess.get("summary")),
    }


def clear_memory(tenant_id: str, session_id: str):
    get_store().clear_session(tenant_id, session_id)
