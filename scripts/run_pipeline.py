"""
手动测试主流程：输入一段文本，跑一遍完整的"清洗→入库→分析→预警"链路。
不依赖RSS采集，方便在还没配置数据源前先验证LLM分析和预警逻辑是否work。
运行方式：python scripts/run_pipeline.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphs.pipeline_graph import process

if __name__ == "__main__":
    text = input("输入一段舆情文本：\n").strip()
    if not text:
        text = "某公司今日公告称，因涉嫌信息披露违规被证监会立案调查，股价开盘跌停。"
        print(f"（未输入，使用示例文本：{text}）")

    result = process(text, source="手动输入")

    print("\n===== 处理结果 =====")
    print("情感：", result["sentiment_label"], result["sentiment_score"])
    print("公司：", result["company"])
    print("事件类型：", result["event_type"])
    print("判断依据：", result["analysis_reason"])
    print("是否触发预警初筛：", result["need_alert"])
    if result["need_alert"]:
        print("复核是否有效：", result["alert_is_valid"])
        print("复核理由：", result["alert_review_reason"])
