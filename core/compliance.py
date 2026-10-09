"""
安全与合规。

1. 敏感词过滤：输入防火墙——用户问句含敏感词直接拦截；入库正文含敏感词自动打码。
   内置清单只是演示用种子，生产环境请通过 SENSITIVE_WORDS_FILE 维护自己的词库
   （每行一个词）。金融场景注意别把"内幕交易/操纵市场"这类业务词加进去——
   那是本系统要监测的事件，不属于需要拦截的违规内容。
2. 免责声明：输出防火墙——金融回答/日报末尾强制拼接"不构成投资建议"免责声明。
"""
from config.settings import settings

# 内置默认敏感词清单（演示种子，覆盖赌博/毒品/色情等明显违规内容类别）。
# 生产环境请用 SENSITIVE_WORDS_FILE 指向自己的词库，替换掉这个内置清单。
_DEFAULT_SENSITIVE_WORDS = [
    "赌博", "赌球", "博彩", "赌场",
    "毒品", "冰毒", "海洛因", "大麻", "摇头丸",
    "卖淫", "嫖娼", "裸聊", "约炮",
    "枪支", "弹药", "军火",
]

# 免责声明：金融场景强制拼接，不构成投资建议
DISCLAIMER = (
    "\n\n---\n"
    "*免责声明：以上内容基于公开舆情资料自动生成，仅供信息参考，"
    "不构成任何投资建议。据此操作风险自负。*"
)

# 舆情-行情关联分析/回测专用免责声明：
# 比通用DISCLAIMER更进一步，明确点出"统计工具"和"历史不代表未来"，
# 避免回测结果被误读为可执行的投资信号。
BACKTEST_DISCLAIMER = (
    "\n\n---\n"
    "*说明：以上是基于历史公开数据的统计研究工具，用于观察舆情信号与历史行情走势"
    "是否存在统计关联，不构成、也不应被理解为任何投资建议或买卖信号。"
    "历史统计结果不代表未来表现，市场存在不确定性，据此操作风险自负。*"
)

_sensitive_words = None


def _get_sensitive_words() -> list:
    """返回生效的敏感词清单（内置 + 外部词库文件）"""
    global _sensitive_words
    if _sensitive_words is not None:
        return _sensitive_words
    words = list(_DEFAULT_SENSITIVE_WORDS)
    words_file = settings.SENSITIVE_WORDS_FILE
    if words_file:
        try:
            with open(words_file, encoding="utf-8") as f:
                for line in f:
                    w = line.strip()
                    if w:
                        words.append(w)
        except Exception:
            pass
    _sensitive_words = words
    return _sensitive_words


def contains_sensitive(text: str) -> bool:
    """是否含敏感词（启用过滤开关时生效）"""
    if not text or not settings.ENABLE_SENSITIVE_FILTER:
        return False
    return any(w in text for w in _get_sensitive_words())


def mask_sensitive(text: str) -> str:
    """把正文里的敏感词替换为***（入库/展示前调用）"""
    if not text or not settings.ENABLE_SENSITIVE_FILTER:
        return text
    for w in _get_sensitive_words():
        if w in text:
            text = text.replace(w, "***")
    return text


def append_disclaimer(text: str, disclaimer: str = None) -> str:
    """在输出末尾强制拼接免责声明（幂等：已带则不重复加）。未启用开关则原样返回。"""
    if not text or not settings.ENABLE_DISCLAIMER:
        return text
    d = disclaimer or DISCLAIMER
    return text if text.endswith(d) else text + d
