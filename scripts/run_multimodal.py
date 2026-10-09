"""
手动测试多模态输入处理。运行方式：python scripts/run_multimodal.py

用法示例：
  - 纯文本：直接回车跳过文件路径，输入文本内容
  - 图片：提供图片路径（如研报截图），可选配一句说明文字
  - 视频：提供视频路径（如业绩说明会录像），当前ASR部分是TODO骨架，
          实际转写内容会是空的，只有关键帧OCR和说明文字生效，
          等 video_processor._transcribe_audio() 接入具体ASR方案后才完整可用
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.multimodal.normalizer import process_input

if __name__ == "__main__":
    file_path = input("图片/视频文件路径（纯文本测试直接回车跳过）：").strip() or None
    text = input("文本内容 / 说明文字：").strip()

    result = process_input(text=text, file_path=file_path, source="手动测试")

    if result.get("skipped"):
        print(f"\n跳过处理：{result['reason']}")
    else:
        print("\n情感：", result.get("sentiment_label"), result.get("sentiment_score"))
        print("公司/事件：", result.get("company"), result.get("event_type"))
        print("是否触发预警初筛：", result.get("need_alert"))
