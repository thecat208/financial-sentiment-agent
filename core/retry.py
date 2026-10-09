"""
通用重试工具：超时 + 指数退避。

- 每次调用设置超时（LLM调用在客户端配置 timeout，本模块负责失败重试的编排）
- 调用失败触发指数退避重试，最多 max_attempts 次
- 退避间隔 = base * 2^attempt + 随机抖动，避免大规模失败时形成重试风暴（防雪崩）
- 最后一次失败后抛出异常，由业务链路的兜底逻辑降级处理，不阻塞主流程

说明：这里用同步指数退避重试——对本项目单机/单进程场景足够且实现简单可靠；
引入任务队列后，可把重试任务丢进队列实现真正异步。
"""
import time
import random
import functools

from config.settings import settings

# 不该重试的HTTP状态码：这些是"再试一次结果也一样"的确定性失败，
# 重试只会白等退避时间、还可能重复计费（典型场景：Key失效的401被重试3次，
# 每次都要等1~3秒，跑批时100条文本就是几百次无意义的失败请求）。
# 429（限流）和5xx（服务端故障）不在其中，仍然值得重试。
_NON_RETRYABLE_STATUS = {400, 401, 403, 404, 405, 409, 422}

# 各SDK里"连接/超时"类异常的名字（不想为判断异常类型额外依赖openai/httpx的导入）
_RETRYABLE_EXC_NAMES = {
    "APITimeoutError", "APIConnectionError", "APIConnectionTimeoutError",
    "RateLimitError", "InternalServerError", "APIStatusError",
    "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout",
    "ConnectError", "ReadError", "RemoteProtocolError",
    "TimeoutError", "ConnectionError", "ConnectionResetError", "OSError",
}

_NON_RETRYABLE_EXC_NAMES = {
    "AuthenticationError", "PermissionDeniedError", "BadRequestError",
    "NotFoundError", "UnprocessableEntityError", "APIResponseValidationError",
}


def is_retryable(exc: Exception) -> bool:
    """
    这个异常值不值得重试？
    - 有HTTP状态码：400/401/403/404/405/409/422 这类确定性失败→不重试；
      429限流、5xx服务端故障→重试
    - 没有状态码：连接/超时类异常→重试；鉴权/参数类异常名→不重试
    - 认不出来的异常保持原有行为（重试），避免把现有链路的容错能力改小
    """
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if isinstance(status, int):
        if status in _NON_RETRYABLE_STATUS:
            return False
        return True

    name = type(exc).__name__
    if name in _NON_RETRYABLE_EXC_NAMES:
        return False
    if name in _RETRYABLE_EXC_NAMES:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return True


def with_retry(max_attempts: int = None, base_delay: float = None):
    """
    重试装饰器：被装饰函数抛异常时按指数退避+随机抖动重试，
    达到 max_attempts 次后抛出最后一次异常（由调用方兜底）。
    遇到确定性失败（鉴权失败/参数错误/模型名不存在等4xx）不重试，立即抛出，
    避免把"再试一次也一样"的错误也拖满退避时间。

    用法：@with_retry()  或  @with_retry(max_attempts=5, base_delay=2.0)
    """
    if max_attempts is None:
        max_attempts = settings.LLM_MAX_RETRIES
    if base_delay is None:
        base_delay = settings.RETRY_BASE_DELAY
    max_attempts = max(1, int(max_attempts))

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if not is_retryable(e):
                        break
                    if attempt < max_attempts - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
                        time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator


def with_retry_async(max_attempts: int = None, base_delay: float = None):
    """
    异步版重试装饰器：重试判定与退避策略与 with_retry 完全一致（复用is_retryable），
    区别仅在退避等待用 await asyncio.sleep 而非 time.sleep——
    time.sleep 会阻塞整个事件循环，异步化后并发收益的前提就是等待期间
    让事件循环去处理其他协程的请求。
    用法：@with_retry_async() 装饰 async def 函数。
    """
    import asyncio

    if max_attempts is None:
        max_attempts = settings.LLM_MAX_RETRIES
    if base_delay is None:
        base_delay = settings.RETRY_BASE_DELAY
    max_attempts = max(1, int(max_attempts))

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if not is_retryable(e):
                        break
                    if attempt < max_attempts - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
                        await asyncio.sleep(delay)
            raise last_exc
        return wrapper
    return decorator
