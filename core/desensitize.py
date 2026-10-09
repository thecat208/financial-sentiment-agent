"""
日志/输出脱敏。

用正则识别 手机号/身份证/银行卡/邮箱 等个人信息，做部分打码（保留少量头尾便于核对，
其余替换为 ***），防止在日志、推送、界面展示中泄露用户隐私。

注意：脱敏用于"日志/展示"层；数据库落库的明文加密属于更深一层，
是企业环境的后续工作。
"""
import re


def _mask_phone(m):
    s = m.group(0)
    return s[:3] + "****" + s[-4:] if len(s) == 11 else "***"


def _mask_id_card(m):
    s = m.group(0)
    return s[:4] + "**********" + s[-4:] if len(s) == 18 else "***"


def _mask_bankcard(m):
    s = m.group(0)
    return s[:4] + "****" + s[-4:] if len(s) >= 16 else "***"


def _mask_email(m):
    s = m.group(0)
    if "@" in s:
        local, domain = s.split("@", 1)
        head = local[:1] + "***" if local else "***"
        return f"{head}@{domain}"
    return "***"


# 隐私信息识别规则：正则 -> 打码函数
_PII_RULES = [
    # 手机号（11位，1开头）
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), _mask_phone),
    # 18位身份证（末位可为数字或X）
    (re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), _mask_id_card),
    # 银行卡（16-19位）
    (re.compile(r"(?<!\d)\d{16,19}(?!\d)"), _mask_bankcard),
    # 邮箱
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), _mask_email),
]


def desensitize(text: str) -> str:
    """对文本中的个人信息做打码，返回脱敏后的文本"""
    if not text:
        return text
    for pattern, repl in _PII_RULES:
        text = pattern.sub(repl, text)
    return text
