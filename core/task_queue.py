"""
异步任务队列。

把耗时操作（日报生成、RSS批量采集、报告生成）从同步请求剥离：
submit() 立即返回 task_id，后台worker消费队列异步执行，get_status() 查询结果。
后端：
- Redis（rpush + BRPOP + HSET状态）：可跨进程共享，scheduler/API/前端共用同一队列（生产推荐）
- 内存（queue.Queue + 后台线程）：Redis不可用时降级，进程内共享（单机部署够用）
使用约定：
- 任务函数必须是模块级可导入的函数（Redis后端用pickle按模块路径序列化，便于跨进程解析）
- 任务失败时 status=error 并记录错误信息，不阻塞队列里其他任务
"""
import queue
import pickle
import threading
import uuid

from config.settings import settings

try:
    import redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class InMemoryTaskQueue:
    """进程内任务队列：queue.Queue + 后台消费线程 + 任务状态表（线程安全）"""

    def __init__(self):
        self._q = queue.Queue()
        self._status = {}
        self._lock = threading.Lock()
        self._started = False

    def submit(self, fn, args, kwargs) -> str:
        task_id = uuid.uuid4().hex
        with self._lock:
            self._status[task_id] = {"status": "pending", "result": None, "error": None}
        self._q.put((task_id, fn, args, kwargs))
        return task_id

    def start(self, concurrency: int = 1):
        with self._lock:
            if self._started:
                return
            self._started = True
        for _ in range(concurrency):
            threading.Thread(target=self._worker_loop, daemon=True).start()

    def _worker_loop(self):
        while True:
            task_id, fn, args, kwargs = self._q.get()
            self._set_status(task_id, "running")
            try:
                result = fn(*args, **kwargs)
                self._set_status(task_id, "done", result=result)
            except Exception as e:
                self._set_status(task_id, "error", error=f"{type(e).__name__}: {e}")

    def _set_status(self, task_id, status, result=None, error=None):
        with self._lock:
            self._status[task_id] = {"status": status, "result": result, "error": error}

    def get_status(self, task_id) -> dict:
        with self._lock:
            return dict(self._status.get(task_id, {"status": "unknown", "result": None, "error": None}))


class RedisTaskQueue:
    """Redis任务队列：rpush任务 + BRPOP消费，状态存hash（带TTL），可跨进程共享"""

    def __init__(self, client, queue_name="tasks:queue"):
        self._client = client
        self._queue_name = queue_name
        self._started = False
        self._lock = threading.Lock()

    def submit(self, fn, args, kwargs) -> str:
        task_id = uuid.uuid4().hex
        payload = pickle.dumps((fn, args, kwargs))
        self._client.hset(
            f"tasks:status:{task_id}",
            mapping={"status": "pending", "result": "", "error": ""},
        )
        self._client.expire(f"tasks:status:{task_id}", 86400)  # 任务状态保留1天
        self._client.rpush(self._queue_name, f"{task_id}:{payload.hex()}")
        return task_id

    def start(self, concurrency: int = 1):
        with self._lock:
            if self._started:
                return
            self._started = True
        for _ in range(concurrency):
            threading.Thread(target=self._worker_loop, daemon=True).start()

    def _worker_loop(self):
        while True:
            raw = self._client.brpop(self._queue_name, timeout=5)
            if raw is None:
                continue
            task_id, payload_hex = raw[1].split(":", 1)
            self._set_status(task_id, "running")
            try:
                fn, args, kwargs = pickle.loads(bytes.fromhex(payload_hex))
                result = fn(*args, **kwargs)
                self._set_status(task_id, "done", result=result)
            except Exception as e:
                self._set_status(task_id, "error", error=f"{type(e).__name__}: {e}")

    def _set_status(self, task_id, status, result=None, error=None):
        mapping = {"status": status, "error": error or ""}
        if result is not None:
            try:
                mapping["result"] = pickle.dumps(result).hex()
            except Exception:
                mapping["result"] = f"!(unpicklable){result}"
        self._client.hset(f"tasks:status:{task_id}", mapping=mapping)

    def get_status(self, task_id) -> dict:
        data = self._client.hgetall(f"tasks:status:{task_id}")
        if not data:
            return {"status": "unknown", "result": None, "error": None}
        result = data.get("result", "") or None
        if result and isinstance(result, str):
            if result.startswith("!(unpicklable)"):
                result = result[len("!(unpicklable)"):]
            else:
                try:
                    result = pickle.loads(bytes.fromhex(result))
                except Exception:
                    result = str(result)
        return {"status": data.get("status"), "result": result, "error": data.get("error")}


_queue = None


def get_task_queue():
    """返回全局任务队列。首次调用探测Redis，不可用则降级为内存队列"""
    global _queue
    if _queue is None:
        if _REDIS_AVAILABLE:
            try:
                client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
                client.ping()
                _queue = RedisTaskQueue(client, queue_name=settings.TASK_QUEUE_NAME)
            except Exception:
                _queue = InMemoryTaskQueue()
        else:
            _queue = InMemoryTaskQueue()
    return _queue
