RISK_KEYWORDS = [
    "立案调查", "退市", "违约", "财务造假", "破产", "被查",
    "监管处罚", "诉讼", "评级下调", "暴雷", "停牌",
]


def contains_risk_keyword(text: str) -> bool:
    return any(kw in text for kw in RISK_KEYWORDS)


def rule_triggered(sentiment_score: float, text: str, score_threshold: float = -0.5) -> bool:
    """
    规则初筛：情感分数低于阈值 或 命中风险关键词，任一满足即触发初筛。
    注意：这里只是"值得关注"的初筛，不代表一定要推送，最终由 alert_review_node 复核决定。
    """
    return sentiment_score <= score_threshold or contains_risk_keyword(text)
