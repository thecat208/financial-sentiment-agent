"""
查看SQLite里已落库的舆情记录，方便验证 pipeline_graph 是否正确写入。
运行方式：python scripts/check_records.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.db import DB_PATH
from storage.repository import get_daily_records
from datetime import datetime

if __name__ == "__main__":
    print(f"数据库位置：{os.path.abspath(DB_PATH)}")

    date = input("查询日期（YYYY-MM-DD，直接回车用今天）：").strip()
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    records = get_daily_records(date)
    print(f"\n{date} 共 {len(records)} 条记录：\n")
    for r in records:
        print(f"[{r['id']}] {r['company']} | {r['sentiment_label']}({r['sentiment_score']}) "
              f"| {r['event_type']} | 预警={bool(r['need_alert'])} "
              f"复核有效={r['alert_is_valid']} 已推送={bool(r['pushed'])}")
        print(f"    {r['cleaned_text'][:60]}...")
