"""
多模态统一入口。所有采集脚本/API接口都应该调用这里的 process_input()，
而不是分别调用 image_processor / video_processor / pipeline_graph——
这一层把"识别输入类型 → 转文本 → 送入主流程"的逻辑收敛到一处，
避免每个调用方都要自己判断"这是图片还是视频还是文本"。
"""
import os
from ingestion.multimodal import image_processor, video_processor
from graphs.pipeline_graph import process as pipeline_process

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def detect_media_type(file_path: str = None) -> str:
    """file_path为空则认为是纯文本输入"""
    if not file_path:
        return "text"
    ext = os.path.splitext(file_path)[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return "text"


def process_input(
    text: str = "",
    file_path: str = None,
    source: str = "未知来源",
    date: str = None,
    tenant_id: str = "default",
    source_type: str = None,
) -> dict:
    """
    统一入口：text和file_path至少提供一个（file_path对应图片/视频，
    text此时作为该图片/视频的说明文字，会和提取出的内容拼接）。

    source_type：P10新增，标注数据来源类型（机构公告/新闻资讯/研报/社交媒体/其他），
    不传则按"新闻资讯"处理，见 graphs/pipeline_graph.process()。

    内部流程：
      1. 识别media_type（text/image/video）
      2. 图片/视频转成统一的raw_text（分别调用image_processor/video_processor）
      3. 调用现有的pipeline_graph.process()，后续逻辑完全复用，不做任何改动
    """
    media_type = detect_media_type(file_path)

    if media_type == "image":
        raw_text = image_processor.build_raw_text(file_path, caption=text)
    elif media_type == "video":
        raw_text = video_processor.build_raw_text(file_path, caption=text)
    else:
        raw_text = text

    if not raw_text.strip():
        return {"skipped": True, "reason": "未提取到任何文字内容（OCR/ASR可能未正确配置或识别失败）"}

    return pipeline_process(
        raw_text=raw_text,
        source=source,
        date=date,
        media_type=media_type,
        media_path=file_path,
        tenant_id=tenant_id,
        source_type=source_type,
    )
