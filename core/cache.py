"""
统一缓存层。

两个后端，对外接口一致，业务代码不关心用的是哪个：
  - RedisCache：优先，TTL/滑动续期由 Redis 原生支持，可跨进程共享（多组件复用同一份缓存）
  - InMemoryCache：Redis 不可用/未安装时的降级实现（进程内 dict + 过期时间），不阻塞主流程

值统一走 JSON 序列化，字符串/数字/列表/字典都可以存。
"""
import json
import time
import hashlib
import threading
import functools

try:
    import redis
    _REDIS_AVAILABLE = True
except ImportError:  # 未安装 redis 包时降级为内存缓存
    _REDIS_AVAILABLE = False


def make_key(prefix: str, *parts) -> str:
    """拼缓存key：prefix:part1:part2:...，None/空部分自动跳过，避免key里出现多余分隔符"""
    kept = [str(p) for p in parts if p is not None and str(p) != ""]
    return ":".join([prefix] + kept)


def hash_part(text: str) -> str:
    """长内容（query/正文）先hash成短key，避免Redis key过长"""
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:12]


class RedisCache:
    """Redis后端。TTL原生支持，get时不续期（需要滑动续期的会话场景由专门的store处理）"""

    def __init__(self, client):
        self._client = client

    def get(self, key: str):
        raw = self._client.get(key)
        return json.loads(raw) if raw is not None else None

    def set(self, key: str, value, ttl: int):
        self._client.set(key, json.dumps(value, ensure_ascii=False), ex=ttl)

    def delete(self, key: str):
        self._client.delete(key)


class InMemoryCache:
    """Redis不可用时的降级后端：进程内 dict + 过期时间，加锁保证线程安全"""

    def __init__(self):
        self._data = {}
        self._expire = {}
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            exp = self._expire.get(key)
            if exp is not None and time.time() > exp:
                self._data.pop(key, None)
                self._expire.pop(key, None)
                return None
            return self._data.get(key)

    def set(self, key: str, value, ttl: int):
        with self._lock:
            self._data[key] = value
            self._expire[key] = time.time() + ttl

    def delete(self, key: str):
        with self._lock:
            self._data.pop(key, None)
            self._expire.pop(key, None)


_cache = None


def get_cache():
    """返回全局缓存实例。首次调用时探测Redis，不可用则降级为内存缓存"""
    global _cache
    if _cache is None:
        if _REDIS_AVAILABLE:
            try:
                from config.settings import settings
                client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
                client.ping()
                _cache = RedisCache(client)
            except Exception:
                _cache = InMemoryCache()
        else:
            _cache = InMemoryCache()
    return _cache


def cached(key_prefix: str, ttl: int, key_parts_fn=None):
    """
    通用缓存装饰器：缓存函数返回值（要求返回值可JSON序列化）。
    key_parts_fn(*args, **kwargs) 返回参与拼key的元组；缺省用全部位置参数。
    结果不能缓存（None）时不写入缓存。
    兼容 async 函数：装饰协程函数时自动生成 async wrapper（await后缓存结果），
    供异步链路（如 classify_intent_async）复用同一套缓存。
    """
    import inspect

    def decorator(fn):
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                parts = key_parts_fn(*args, **kwargs) if key_parts_fn else args
                key = make_key(key_prefix, *parts)
                cache = get_cache()
                hit = cache.get(key)
                if hit is not None:
                    return hit
                result = await fn(*args, **kwargs)
                if result is not None:
                    cache.set(key, result, ttl)
                return result
            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            parts = key_parts_fn(*args, **kwargs) if key_parts_fn else args
            key = make_key(key_prefix, *parts)
            cache = get_cache()
            hit = cache.get(key)
            if hit is not None:
                return hit
            result = fn(*args, **kwargs)
            if result is not None:
                cache.set(key, result, ttl)
            return result
        return wrapper
    return decorator
