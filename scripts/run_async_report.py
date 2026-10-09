"""
演示异步任务队列：提交日报生成后立即返回task_id，轮询查看完成状态。

对比 graphs/report_graph.generate_report() 的同步调用——同步版会一直阻塞到
所有公司的LLM小结生成完才返回；这里 submit 立刻返回，由后台worker异步执行。

运行方式：python scripts/run_async_report.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.async_jobs import ensure_worker, submit_daily_report, get_task_status

if __name__ == "__main__":
    ensure_worker()
    date = input("生成哪天的日报？（YYYY-MM-DD，直接回车用今天）：").strip() or None

    task_id = submit_daily_report(date)
    print(f"已提交日报生成任务：{task_id}")
    print("任务在后台异步执行（LLM逐个生成公司小结），本脚本轮询等待完成...\n")

    status = None
    while True:
        status = get_task_status(task_id)
        print(f"  状态：{status['status']}")
        if status["status"] in ("done", "error"):
            break
        time.sleep(1)

    if status["status"] == "done":
        result = status["result"]
        path = result.get("report_path") if isinstance(result, dict) else result
        print(f"\n完成！日报已保存：{path}")
    else:
        print(f"\n任务失败：{status.get('error')}")
