"""
演示脚本：专业数据源接入。

拉取指定公司的机构公告 + 千股千评市场热度数据，跑完整的
"清洗→分析→入库→预警"链路，验证 source_type 能正确区分来源类型。

用法：
    python scripts/run_multi_source_demo.py 宁德时代
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.announcement_source import fetch_announcements
from ingestion.social_source import fetch_retail_sentiment
from graphs.pipeline_graph import process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("company", help="公司名称，如 宁德时代")
    parser.add_argument("--days", type=int, default=7, help="公告回溯天数")
    args = parser.parse_args()

    print(f"===== 机构公告：{args.company}（近{args.days}天）=====")
    announcements = fetch_announcements(args.company, days=args.days)
    print(f"共 {len(announcements)} 条")
    for item in announcements:
        tag = "（演示数据）" if item.get("is_mock") else ""
        print(f"  [{item['date']}]{tag} {item['raw_text'][:50]}")
        result = process(
            raw_text=item["raw_text"], source=item["source"], date=item["date"],
            source_type=item["source_type"],
        )
        print(f"    → 情感：{result['sentiment_label']}({result['sentiment_score']}) "
              f"事件：{result['event_type']} 触发预警：{result['need_alert']}")

    print(f"\n===== 市场热度（千股千评）：{args.company} =====")
    social = fetch_retail_sentiment(args.company)
    if not social:
        print("暂无该股票的千股千评数据")
    else:
        tag = "（演示数据）" if social.get("is_mock") else ""
        print(f"{tag}{social['raw_text']}")
        result = process(
            raw_text=social["raw_text"], source=social["source"], date=social["date"],
            source_type=social["source_type"],
        )
        print(f"  → 情感：{result['sentiment_label']}({result['sentiment_score']})")

    print("\n提示：用 Streamlit「历史舆情/趋势图」标签页查看这些记录，"
          "机构公告带📄标记，社交媒体来源带💬标记，和普通新闻区分开来。")


if __name__ == "__main__":
    main()
