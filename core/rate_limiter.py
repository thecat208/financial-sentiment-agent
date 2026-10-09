"""
多级限流。

1. 全局令牌桶：限制整体QPS，防止服务被打爆
2. 单用户固定窗口：按 tenant_id/user_id 每窗口限制调用次数，防止恶意刷接口消耗Token
   （固定窗口实现更简单，语义与漏桶一致）

后端：Redis（分布式，INCR+EXPIRE，多实例共享）；Redis不可用时降级为进程内实现（单机够用）。

对外接口：allow(identity) -> bool，True放行 / False限流拦截。
"""
import time
import threading

from config.settings import settings

try:
    import redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class TokenBucket:
    """令牌桶：以 rate/s 补充令牌，容量 capacity，取令牌失败返回 False"""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.updated = time.time()
        self.lock = threading.Lock()

    def try_acquire(self, n: int = 1) -> bool:
        with self.lock:
            now = time.time()
            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False


class FixedWindowLimiter:
    """固定窗口限流（内存版）：每 identity 在 window 秒内最多 max_calls 次"""

    def __init__(self, window: int, max_calls: int):
        self.window = window
        self.max_calls = max_calls
        self._data = {}  # identity -> [start_time, count]
        self.lock = threading.Lock()

    def allow(self, identity: str) -> bool:
        with self.lock:
            now = time.time()
            start, count = self._data.get(identity, (now, 0))
            if now - start >= self.window:
                start, count = now, 0
            if count >= self.max_calls:
                return False
            self._data[identity] = (start, count + 1)
            return True


class RedisFixedWindowLimiter:
    """固定窗口限流（Redis版）：INCR + EXPIRE，多实例共享"""

    def __init__(self, client, window: int, max_calls: int):
        self._client = client
        self.window = window
        self.max_calls = max_calls

    def allow(self, identity: str) -> bool:
        key = f"rl:{identity}"
        count = self._client.incr(key)
        if count == 1:
            self._client.expire(key, self.window)
        return count <= self.max_calls


_global_bucket = None
_user_limiter = None
_redis_user_limiter = None


def allow(identity: str) -> bool:
    """全局令牌桶 + 单用户限流，两者都放行才返回 True"""
    global _global_bucket, _user_limiter, _redis_user_limiter

    if _global_bucket is None:
        _global_bucket = TokenBucket(settings.RATE_LIMIT_GLOBAL_QPS, settings.RATE_LIMIT_GLOBAL_BURST)
    if not _global_bucket.try_acquire():
        return False

    if _user_limiter is None and _redis_user_limiter is None:
        if _REDIS_AVAILABLE:
            try:
                client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
                client.ping()
                _redis_user_limiter = RedisFixedWindowLimiter(
                    client, settings.RATE_LIMIT_USER_WINDOW_SEC, settings.RATE_LIMIT_USER_MAX
                )
            except Exception:
                _user_limiter = FixedWindowLimiter(
                    settings.RATE_LIMIT_USER_WINDOW_SEC, settings.RATE_LIMIT_USER_MAX
                )
        else:
            _user_limiter = FixedWindowLimiter(
                settings.RATE_LIMIT_USER_WINDOW_SEC, settings.RATE_LIMIT_USER_MAX
            )

    limiter = _redis_user_limiter or _user_limiter
    return limiter.allow(identity)
