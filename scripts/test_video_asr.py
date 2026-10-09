"""
视频ASR冒烟测试（ingestion/multimodal/video_processor.py 验收）。

两种模式：

1. 机制测试（默认，无需真实语音素材，可离线跑）：
   用ffmpeg合成一段3秒测试视频（蓝色画面+440Hz正弦音，无语音），
   验证 抽音轨→转写→抽关键帧→OCR→三路合并→raw_text 全链路不抛异常、
   返回字符串、caption保留。ASR对正弦音转出空串属预期（无语音内容）。

   python scripts/test_video_asr.py

2. 真实素材测试（验收转写质量，推荐）：
   python scripts/test_video_asr.py --video 业绩说明会.mp4
   打印 build_raw_text 的产出与各阶段耗时，人工核对语音内容是否被转出、
   PPT画面的OCR是否命中。ASR质量无自动评测口径，以人工抽测为准。

环境自检（两种模式都会先跑）：
  - ffmpeg 是否在 PATH（缺失则音轨/关键帧两路不可用，属预期降级而非报错）
  - faster-whisper 是否可导入（可选依赖，缺失时语音转写降级为空）
  - whisper模型是否已缓存/能否初始化（首次会触发下载，约460MB/small）
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings  # noqa: E402
from ingestion.multimodal.video_processor import (  # noqa: E402
    _extract_audio,
    _transcribe_audio,
    _get_whisper_model,
    build_raw_text,
)


def check_env() -> dict:
    """环境自检，返回各项状态供打印。任何一项缺失都不算失败——只影响对应链路。"""
    status = {}
    status["ffmpeg"] = shutil.which("ffmpeg") is not None
    try:
        import faster_whisper  # noqa: F401
        status["faster-whisper"] = True
    except ImportError:
        status["faster-whisper"] = False
    if status["faster-whisper"]:
        t0 = time.time()
        status["whisper模型"] = _get_whisper_model() is not None
        status["模型加载耗时"] = f"{time.time() - t0:.1f}s"
    return status


def make_synth_video(dir_path: str) -> str:
    """ffmpeg合成3秒测试视频：蓝色画面+正弦音（无语音，用于机制验证）。"""
    out = os.path.join(dir_path, "test_synth.mp4")
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=3",
         "-shortest", "-pix_fmt", "yuv420p", out],
        check=True, capture_output=True,
    )
    return out


def run_case(video_path: str, caption: str, label: str):
    """跑一次完整 build_raw_text，分阶段打印耗时与结果。"""
    print(f"\n----- {label} -----")
    with tempfile.TemporaryDirectory() as tmp_dir:
        t0 = time.time()
        audio = _extract_audio(video_path, tmp_dir)
        t1 = time.time()
        transcript = _transcribe_audio(audio) if audio else ""
        t2 = time.time()
        raw = build_raw_text(video_path, caption=caption)
        t3 = time.time()

    print(f"抽音轨: {'OK' if audio else '失败(降级)'} ({t1 - t0:.2f}s)")
    print(f"ASR转写: {'OK' if transcript else '空(无语音/依赖缺失/降级)'} ({t2 - t1:.2f}s)")
    if transcript:
        print(f"  转写内容({len(transcript)}字): {transcript[:200]}")
    print(f"build_raw_text 总耗时: {t3 - t0:.2f}s")
    print(f"raw_text({len(raw)}字):\n{raw}")
    assert isinstance(raw, str), "build_raw_text 必须返回字符串"
    assert caption.strip() in raw, "caption 必须保留在 raw_text 中"
    print("断言通过")


def main():
    parser = argparse.ArgumentParser(description="视频ASR冒烟测试")
    parser.add_argument("--video", default=None, help="真实视频文件路径（不传则用合成视频做机制测试）")
    args = parser.parse_args()

    print("== 环境自检 ==")
    status = check_env()
    for k, v in status.items():
        print(f"  {k}: {'OK' if v is True else v if v is not False else '缺失(降级)'}")
    print(f"  ASR_PROVIDER={settings.ASR_PROVIDER} MODEL={settings.ASR_WHISPER_MODEL_SIZE} "
          f"DEVICE={settings.ASR_WHISPER_DEVICE} LANG={settings.ASR_LANGUAGE}")

    if args.video:
        if not os.path.exists(args.video):
            print(f"视频不存在: {args.video}")
            sys.exit(1)
        run_case(args.video, caption="", label=f"真实素材测试: {args.video}")
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            video = make_synth_video(tmp_dir)
            run_case(video, caption="测试视频说明文字", label="机制测试(合成视频,无语音)")

    print("\n== 全部通过 ==")


if __name__ == "__main__":
    main()
