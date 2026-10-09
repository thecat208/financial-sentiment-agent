"""
轻量模型前置筛选（成本优化，与预警的"规则初筛+LLM复核"同一套设计哲学）。

先用轻量模型判断情感，只有置信度不够的模糊文本才升级到大模型，降低全量调用LLM的成本和延迟。

两个后端，自动降级：
  - transformer：小型中文情感分类模型（HuggingFace pipeline），本地CPU推理、不调外部LLM API；
    需要 `pip install transformers torch`，首次运行联网下载权重；依赖缺失/下载失败自动降级到lexicon。
  - lexicon：金融情感词典+否定词处理的规则打分器，不依赖模型/网络，永远可用，是兜底后端。
置信度：分数绝对值越大、命中情感词越多、正负信号不冲突则越高；低于 settings.LIGHTWEIGHT_CONFIDENCE_THRESHOLD
时上层 chains/analysis_chain.py 会升级到LLM完整分析。
"""
import re
from config.settings import settings
from alert.rules import RISK_KEYWORDS

# ---------------- lexicon 后端：金融情感词典 ----------------
# 权重范围大致对应"这个词单独出现时，情感倾向的强弱"，不追求词典完备，
# 追求覆盖金融舆情里最常见、最无歧义的高频表达。
POSITIVE_LEXICON = {
    "超预期": 0.7, "创新高": 0.6, "扭亏为盈": 0.7, "大幅增长": 0.6, "同比增长": 0.4,
    "净利润增长": 0.5, "营收增长": 0.4, "盈利": 0.4, "上涨": 0.4, "中标": 0.5,
    "签署": 0.3, "分红": 0.3, "回购": 0.3, "获批": 0.3, "突破": 0.4, "增持": 0.4,
    "合作": 0.25, "订单增长": 0.5, "业绩预增": 0.6, "评级上调": 0.6, "利好": 0.5,
    "稳健增长": 0.4, "再创新高": 0.6, "顺利": 0.2, "积极": 0.3,
}
NEGATIVE_LEXICON = {
    "下滑": -0.4, "亏损": -0.6, "违规": -0.6, "处罚": -0.6, "诉讼": -0.4,
    "调查": -0.5, "退市": -0.9, "暴雷": -0.8, "停牌": -0.5, "跌停": -0.7,
    "造假": -0.8, "违约": -0.6, "召回": -0.5, "断供": -0.5, "下降": -0.3,
    "下跌": -0.4, "评级下调": -0.6, "减持": -0.3, "利空": -0.5, "诉讼纠纷": -0.4,
    "业绩预减": -0.6, "警示函": -0.5, "问询函": -0.3, "延期": -0.2, "停产": -0.5,
}
# 风险关键词（复用 alert/rules.py 已有的清单，避免维护两份）视为强负面信号，
# 权重给得比普通负面词更高——这类词在金融舆情里几乎不会是中性/正面语境。
for _kw in RISK_KEYWORDS:
    NEGATIVE_LEXICON.setdefault(_kw, -0.7)

NEGATION_WORDS = ["不", "未", "没有", "无", "非", "并非", "尚未", "并未"]
_NEGATION_WINDOW = 4  # 否定词和情感词之间最多隔几个字符，仍算作否定关系


def _lexicon_score(text: str) -> dict:
    """
    规则打分：命中词典里的词就累加权重；命中词前 _NEGATION_WINDOW 个字符内
    出现否定词，则该词权重取反并打七折（简单否定处理，不追求完美，
    遇到真正复杂的否定/双重否定，置信度会因为"信号冲突"被拉低，交给LLM处理）。
    """
    hits = []  # (词, 原始权重, 是否被否定, 最终权重)
    for lexicon in (POSITIVE_LEXICON, NEGATIVE_LEXICON):
        for term, weight in lexicon.items():
            for m in re.finditer(re.escape(term), text):
                window_start = max(0, m.start() - _NEGATION_WINDOW)
                preceding = text[window_start:m.start()]
                negated = any(neg in preceding for neg in NEGATION_WORDS)
                final_weight = -weight * 0.7 if negated else weight
                hits.append((term, weight, negated, final_weight))

    if not hits:
        return {"label": "中性", "score": 0.0, "confidence": 0.3, "hits": []}

    raw_score = sum(h[3] for h in hits)
    # 归一化：命中词越多，单个词对总分的边际影响应该变小，避免长文本因为词多而分数爆表
    score = max(-1.0, min(1.0, raw_score / max(1, len(hits)) * min(len(hits), 3) ** 0.5 / 1.5))

    pos_hits = [h for h in hits if h[3] > 0]
    neg_hits = [h for h in hits if h[3] < 0]
    # 置信度：|分数|越大越有信心；命中词越多越有信心；正负信号同时出现（矛盾信号）则降低置信度
    magnitude_conf = min(abs(score) / 0.4, 1.0)  # |score|>=0.4 时这一项拉满
    evidence_conf = min(len(hits) / 2, 1.0)       # 至少2个命中词时这一项拉满
    conflict_penalty = 0.35 if (pos_hits and neg_hits) else 0.0
    confidence = max(0.0, min(1.0, 0.5 * magnitude_conf + 0.5 * evidence_conf - conflict_penalty))

    label = "正面" if score > 0.15 else ("负面" if score < -0.15 else "中性")
    return {"label": label, "score": round(score, 3), "confidence": round(confidence, 3),
            "hits": [h[0] for h in hits]}


# ---------------- transformer 后端：小模型（可选，需要联网下载权重）----------------
_transformer_pipeline = None
_transformer_unavailable = False


def _get_transformer_pipeline():
    """懒加载，首次调用才尝试导入transformers+下载模型；失败后不再重试（同一进程内）"""
    global _transformer_pipeline, _transformer_unavailable
    if _transformer_pipeline is not None or _transformer_unavailable:
        return _transformer_pipeline
    try:
        from transformers import pipeline
        _transformer_pipeline = pipeline("sentiment-analysis", model=settings.LIGHTWEIGHT_HF_MODEL)
    except Exception:
        _transformer_unavailable = True
        _transformer_pipeline = None
    return _transformer_pipeline


def _transformer_score(text: str) -> dict | None:
    """返回None代表不可用（未装依赖/下载失败/推理异常），调用方应自动降级到lexicon"""
    pipe = _get_transformer_pipeline()
    if pipe is None:
        return None
    try:
        out = pipe(text[:512])[0]  # 大多数中文情感小模型只做二分类，截断超长文本避免推理变慢
        raw_label = str(out.get("label", "")).lower()
        prob = float(out.get("score", 0.5))
        if "pos" in raw_label or raw_label in ("1", "label_1"):
            label, score = "正面", prob
        elif "neg" in raw_label or raw_label in ("0", "label_0"):
            label, score = "负面", -prob
        else:
            label, score = "中性", 0.0
        return {"label": label, "score": round(score, 3), "confidence": round(prob, 3), "hits": []}
    except Exception:
        return None


def classify(text: str) -> dict:
    """
    对外统一入口。返回 {"label","score","confidence","backend","hits"}：
      label: 正面/中性/负面
      score: -1~1
      confidence: 0~1，越高代表轻量模型越有把握
      backend: "transformer" 或 "lexicon"，标注这次结果是哪个后端给出的，
               便于统计"transformer后端实际命中率"（联网环境下有意义）
    """
    backend_pref = settings.LIGHTWEIGHT_SENTIMENT_BACKEND
    if backend_pref in ("auto", "transformer"):
        result = _transformer_score(text)
        if result is not None:
            result["backend"] = "transformer"
            return result
        if backend_pref == "transformer":
            # 显式要求transformer但不可用：明确返回极低置信度，强制上层升级到LLM，
            # 不悄悄换成lexicon结果去冒充"transformer后端判断"
            return {"label": "中性", "score": 0.0, "confidence": 0.0, "backend": "unavailable", "hits": []}

    result = _lexicon_score(text)
    result["backend"] = "lexicon"
    return result
