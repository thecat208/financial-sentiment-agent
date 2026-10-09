import re

from core.compliance import mask_sensitive


def clean_text(text: str) -> str:
    """基础文本清洗：去HTML标签、合并多余空白；入库前做敏感词打码"""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = mask_sensitive(text)
    return text
