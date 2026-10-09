"""
手动生成某天的日报，方便调试。
运行方式：python scripts/run_daily_report.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphs.report_graph import generate_report

if __name__ == "__main__":
    date = input("生成哪天的日报？（YYYY-MM-DD，直接回车用今天）：").strip() or None
    result = generate_report(date)
    print("\n" + result["report_text"])
    print(f"\n已保存到：{result['report_path']}")
