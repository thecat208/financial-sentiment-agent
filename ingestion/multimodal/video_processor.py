"""
视频输入处理（本地ASR已接入）。
设计原则同 image_processor.py：产出统一的raw_text，复用 graphs/pipeline_graph.py；
典型场景：业绩说明会录像、财经新闻视频、路演视频。
三路信息合并：1.音频转写：ASR（默认本地faster-whisper）；2.关键帧OCR：抽取PPT/字幕
画面用 image_processor 做OCR（关键信息常在PPT里）；3.视频说明文字（caption，调用方传入）。
依赖与降级约定（增强能力失败不阻断主流程）：ffmpeg缺失时音轨/关键帧两路为空，
仅剩caption；faster-whisper未安装/加载失败时语音转写为空但不报错，且进程内
不再重复尝试（懒加载单例+失败标记）；云端ASR（ASR_PROVIDER="cloud"）为预留接入点；
whisper懒加载用双检锁单例（模式同 retrieval/vectorstore.py）。事件循环注意：whisper推理是CPU/GPU密集型会阻塞事件循环，当前调用方均在同步上下文无需处理，未来若新增async路由须用 asyncio.to_thread 包裹。
"""
import os
import subprocess
import tempfile
import threading

from config.settings import settings
from ingestion.multimodal.image_processor import extract_text_from_image

FRAME_INTERVAL_SECONDS = settings.VIDEO_FRAME_INTERVAL

# whisper模型懒加载单例（进程级，只初始化一次）
_whisper_model = None
_whisper_checked = False
_whisper_lock = threading.Lock()


def _get_whisper_model():
    """
    懒加载faster-whisper模型（双检锁单例）。

    首次调用时初始化WhisperModel（会触发权重下载，若本地无缓存）；
    依赖未安装或加载失败时返回None并置失败标记，进程内不再重试——
    避免每条视频都重复走一遍注定失败的加载路径。
    """
    global _whisper_model, _whisper_checked
    if _whisper_checked:
        return _whisper_model
    with _whisper_lock:
        if _whisper_checked:
            return _whisper_model
        try:
            from faster_whisper import WhisperModel

            _whisper_model = WhisperModel(
                settings.ASR_WHISPER_MODEL_SIZE,
                device=settings.ASR_WHISPER_DEVICE,
            )
        except Exception:
            _whisper_model = None
        finally:
            _whisper_checked = True
        return _whisper_model


def _extract_audio(video_path: str, output_dir: str) -> str:
    """用ffmpeg从视频中提取音轨（16kHz单声道wav，whisper推荐输入格式）。
    需要系统安装ffmpeg；失败返回空字符串，由调用方降级处理。"""
    audio_path = os.path.join(output_dir, "audio.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", audio_path],
            check=True, capture_output=True,
        )
        return audio_path
    except Exception:
        return ""


def _transcribe_audio(audio_path: str) -> str:
    """
    按配置分发到具体ASR方案（settings.ASR_PROVIDER）。约定（所有后端必须遵守，保证未来换云端API上游零改动）：
    入参=音频文件路径（wav），返回值=转写纯文本，失败/不可用返回空字符串。
    local（默认）走本地faster-whisper（见 _transcribe_audio_local）；
    cloud 为云端ASR预留接入点：实现 _transcribe_audio_cloud(audio_path)->str（候选：阿里云智能语音
    交互/腾讯云ASR/OpenAI Whisper API，按数据敏感性与成本选型），在本函数追加 elif 分发，
    凭证走 .env + config/settings.py，上游调用方零改动。其他值视为未配置，返回空串（不报错）。
    """
    if not audio_path or not os.path.exists(audio_path):
        return ""
    provider = (settings.ASR_PROVIDER or "local").lower()
    if provider == "local":
        return _transcribe_audio_local(audio_path)
    # cloud 接入点：实现 _transcribe_audio_cloud(audio_path) 后在此分发
    return ""


def _transcribe_audio_local(audio_path: str) -> str:
    """本地faster-whisper转写。模型/依赖不可用时返回空串（降级，不抛异常）。"""
    model = _get_whisper_model()
    if model is None:
        return ""
    try:
        language = None if settings.ASR_LANGUAGE.lower() in ("auto", "") else settings.ASR_LANGUAGE
        segments, _info = model.transcribe(
            audio_path,
            language=language,
            vad_filter=True,  # 跳过静音段：业绩说明会常有长时间停顿，避免空转写碎片
        )
        return "".join(seg.text for seg in segments).strip()
    except Exception:
        return ""


def _extract_keyframes(video_path: str, output_dir: str) -> list:
    """
    按固定间隔抽取关键帧，用ffmpeg截图。返回帧图片路径列表。
    间隔时间由 settings.VIDEO_FRAME_INTERVAL 控制，默认30秒一帧。
    """
    pattern = os.path.join(output_dir, "frame_%03d.jpg")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps=1/{FRAME_INTERVAL_SECONDS}", pattern],
            check=True, capture_output=True,
        )
        return sorted(
            os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith("frame_")
        )
    except Exception:
        return []


def build_raw_text(video_path: str, caption: str = "") -> str:
    """
    视频转文本的统一入口，合并 音频转写 + 关键帧OCR + 视频说明，
    产出和文本采集一样的raw_text，供 pipeline_graph.process() 使用。
    """
    if not os.path.exists(video_path):
        return caption.strip()

    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = _extract_audio(video_path, tmp_dir)
        transcript = _transcribe_audio(audio_path) if audio_path else ""

        frames = _extract_keyframes(video_path, tmp_dir)
        frame_texts = [extract_text_from_image(f) for f in frames]
        frame_text = "\n".join(t for t in frame_texts if t)

    parts = [p for p in [caption.strip(), transcript.strip(), frame_text.strip()] if p]
    return "\n".join(parts)
