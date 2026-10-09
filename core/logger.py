"""
规范化 + 脱敏日志。

get_logger() 返回的 logger 在输出时自动对整条日志做 PII 脱敏
（手机号/身份证/银行卡/邮箱），避免把用户隐私打进日志/链路追踪；
同时统一用 logging 模块替代散落的 print，输出带时间戳和级别。
"""
import logging

from core.desensitize import desensitize


class DesensitizeFormatter(logging.Formatter):
    """格式化后再对整个日志文本做一遍脱敏，简单可靠地覆盖 msg + args"""

    def format(self, record):
        return desensitize(super().format(record))


def get_logger(name: str = "app") -> logging.Logger:
    """返回带脱敏输出、规范化格式的 logger（全局只配置一次）"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            DesensitizeFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
