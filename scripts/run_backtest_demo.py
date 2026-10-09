"""
演示脚本：舆情-行情关联分析 + 预警信号回测。

运行前提：先跑过 scripts/run_pipeline.py 产生一些舆情数据（最好包含触发过预警的记录）。

用法：
    python scripts/run_backtest_demo.py 公司名 [--days 90] [--holding 5]
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.backtest import get_sentiment_price_overlay, run_alert_backtest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("company", help="公司名称，如 宁德时代")
    parser.add_argument("--days", type=int, default=90, help="统计窗口天数")
    parser.add_argument("--holding", type=int, default=5, help="回测持有交易日数")
    parser.add_argument("--tenant", default="default", help="租户ID")
    args = parser.parse_args()

    print(f"===== 情感-行情关联分析：{args.company}（近{args.days}天）=====")
    overlay = get_sentiment_price_overlay(args.company, days=args.days, tenant_id=args.tenant)
    if overlay["is_mock"]:
        print(f"⚠️  行情数据源：mock（演示数据，非真实行情），股票代码解析为 {overlay['code']}")
    else:
        print(f"行情数据源：akshare（真实数据），股票代码 {overlay['code']}")
    print(f"匹配样本数：{overlay['sample_size']}，相关系数：{overlay['correlation']}")

    print(f"\n===== 预警信号回测：持有{args.holding}个交易日 =====")
    bt = run_alert_backtest(
        company=args.company, tenant_id=args.tenant, days=args.days, holding_days=args.holding,
    )
    print(f"样本数：{bt['samples']}")
    if bt["samples"] == 0:
        print("没有可回测的样本（该公司近期没有触发过预警，或行情数据覆盖不到）。")
    else:
        print(f"平均涨跌幅：{bt['avg_return_pct']}%")
        print(f"正案例胜率：{bt['win_rate']}%（正{bt['positive_count']} / 负{bt['negative_count']}）")
        if bt["low_sample_warning"]:
            print("⚠️  样本量较少，统计结果仅供参考")

    print(bt["disclaimer"])


if __name__ == "__main__":
    main()
