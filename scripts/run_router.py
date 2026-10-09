"""
手动测试意图路由。运行方式：python scripts/run_router.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphs.router_graph import handle_query

if __name__ == "__main__":
    print("试试输入：'示例公司最近舆情怎么样'、'看看今天的日报'、'最近预警误报率高不高'")
    q = input("请输入你的问题：").strip()
    res = handle_query(q)
    print(f"\n[识别意图：{res['intent']}]\n")
    print(res["text"])
