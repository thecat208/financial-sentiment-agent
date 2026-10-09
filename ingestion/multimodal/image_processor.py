"""
图片输入处理。

设计原则：图片处理只负责"提取出文字内容"，产出和文本采集完全一样的raw_text格式，
之后直接复用 graphs/pipeline_graph.py，不需要为图片输入单独设计下游的分析/预警/入库逻辑。
典型场景：研报截图、公告PDF页面截图、财经海报、K线图配文字说明。
依赖：pytesseract + Pillow（本地OCR，免费但准确率一般）。需要更高准确率时，可把
extract_text_from_image 内部实现换成云OCR API（阿里云/腾讯云/百度OCR等），外部调用
接口保持不变，不影响 pipeline_graph 或其他调用方。
"""
import os

try:
    import pytesseract
    from PIL import Image
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False


def extract_text_from_image(image_path: str, lang: str = "chi_sim+eng") -> str:
    """
    对单张图片做OCR，返回提取出的文字。lang默认中英混合识别（chi_sim=简体中文），
    需要本地安装对应的tesseract语言包（apt-get install tesseract-ocr-chi-sim）。
    OCR失败或依赖未安装时返回空字符串，不抛异常——上游 normalizer.py 会据此判断
    是否有内容可处理，避免因为一张图片识别失败就中断整批数据的处理。
    """
    if not _OCR_AVAILABLE:
        return ""
    if not os.path.exists(image_path):
        return ""

    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang=lang)
        return text.strip()
    except Exception:
        return ""


def build_raw_text(image_path: str, caption: str = "") -> str:
    """
    把OCR结果和图片说明（如果有，比如社交媒体配图的文字说明）
    拼接成统一的raw_text，供 pipeline_graph.process() 使用。
    """
    ocr_text = extract_text_from_image(image_path)
    parts = [p for p in [caption.strip(), ocr_text] if p]
    return "\n".join(parts)
