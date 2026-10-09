"""
会话存储。

"混合记忆池"两层结构：
  - 短期：turns 列表，保存最近若干轮 用户/助手 对话原文
  - 长期：summary 字段，累计轮数超阈值时由LLM把历史压缩成摘要，防止关键信息丢失
两个后端，对外接口一致：
  - RedisSessionStore：优先。key = session:{tenant_id}:{session_id}:data，热会话TTL=30分钟
    并滑动续期（每次读取刷新），"清空记忆"主动失效（删除key）；可跨进程共享，重启后记忆不丢。
  - InMemorySessionStore：Redis不可用时的降级实现（进程内字典），不阻塞主流程。
Key 都拼了 tenant_id，不同租户的会话状态天然隔离（多租户隔离在存储层做）。
"""
import json
from typing import Dict, List

try:
    import redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class RedisSessionStore:
    """Redis会话存储。value为JSON {turns, summary}，读时滑动续期TTL"""

    def __init__(self, client, ttl: int = 1800):
        self._client = client
        self._ttl = ttl

    def _key(self, tenant_id: str, session_id: str) -> str:
        return f"session:{tenant_id}:{session_id}:data"

    def _save(self, key: str, data: dict):
        self._client.set(key, json.dumps(data, ensure_ascii=False), ex=self._ttl)

    def get_session(self, tenant_id: str, session_id: str) -> dict:
        key = self._key(tenant_id, session_id)
        raw = self._client.get(key)
        if raw is None:
            data = {"turns": [], "summary": ""}
            self._save(key, data)
        else:
            data = json.loads(raw)
            self._client.expire(key, self._ttl)  # 滑动续期：每次访问刷新TTL
        return data

    def add_turn(self, tenant_id: str, session_id: str, user: str, assistant: str) -> dict:
        sess = self.get_session(tenant_id, session_id)
        sess["turns"].append({"user": user, "assistant": assistant})
        self._save(self._key(tenant_id, session_id), sess)
        return sess

    def set_summary(self, tenant_id: str, session_id: str, summary: str):
        key = self._key(tenant_id, session_id)
        sess = self.get_session(tenant_id, session_id)
        sess["summary"] = summary
        self._save(key, sess)

    def replace_turns(self, tenant_id: str, session_id: str, turns: list):
        key = self._key(tenant_id, session_id)
        sess = self.get_session(tenant_id, session_id)
        sess["turns"] = turns
        self._save(key, sess)

    def clear_session(self, tenant_id: str, session_id: str):
        """冷数据主动失效：会话结束后立即删除，仅保留DB记录"""
        self._client.delete(self._key(tenant_id, session_id))

    def total_sessions(self) -> int:
        return len(list(self._client.scan_iter("session:*:data")))


class InMemorySessionStore:
    """进程内会话存储。key = f"{tenant_id}:{session_id}"，value 含 turns + summary"""

    def __init__(self):
        self._sessions: Dict[str, dict] = {}

    def _key(self, tenant_id: str, session_id: str) -> str:
        return f"{tenant_id}:{session_id}"

    def get_session(self, tenant_id: str, session_id: str) -> dict:
        key = self._key(tenant_id, session_id)
        if key not in self._sessions:
            self._sessions[key] = {"turns": [], "summary": ""}
        return self._sessions[key]

    def add_turn(self, tenant_id: str, session_id: str, user: str, assistant: str) -> dict:
        """追加一轮对话，返回更新后的会话"""
        sess = self.get_session(tenant_id, session_id)
        sess["turns"].append({"user": user, "assistant": assistant})
        return sess

    def set_summary(self, tenant_id: str, session_id: str, summary: str):
        self.get_session(tenant_id, session_id)["summary"] = summary

    def replace_turns(self, tenant_id: str, session_id: str, turns: list):
        """压缩完成后，用保留的最近几轮替换短期窗口"""
        self.get_session(tenant_id, session_id)["turns"] = turns

    def clear_session(self, tenant_id: str, session_id: str):
        key = self._key(tenant_id, session_id)
        self._sessions.pop(key, None)

    def total_sessions(self) -> int:
        return len(self._sessions)
