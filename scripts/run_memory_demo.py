"""
手动测试多轮追问记忆（上下文与记忆管理）。
运行方式：python scripts/run_memory_demo.py

用法：在一个会话里连续提问，第二个问题可以用指代词（如"那它呢？"），
看系统是否结合上一轮对话理解。输入 quit/exit/退出 结束。
"""
import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphs.qa_graph import ask
from memory.memory_manager import get_status, clear_memory

if __name__ == "__main__":
    session_id = uuid.uuid4().hex
    print(f"本次会话ID：{session_id}")
    print("（连续提问，系统会记住上下文；输入 quit / exit / 退出 结束）")

    while True:
        q = input("\n你的问题：").strip()
        if not q:
            continue
        if q.lower() in ("quit", "exit", "退出"):
            break

        res = ask(q, tenant_id="default", session_id=session_id)
        print("\n[回答]")
        print(res["answer"])
        print(f"[检索方式] {res['retrieve_desc']}")

        status = get_status("default", session_id)
        print(f"[记忆] 短期轮数={status['turn_count']} 长期摘要={'有' if status['has_summary'] else '无'}")

    clear_memory("default", session_id)
    print("\n已清空会话记忆。")
