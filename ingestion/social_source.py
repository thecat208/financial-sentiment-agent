"""
散户情绪数据采集。
设计取舍：股吧/雪球没有公开官方API能拿到原始帖子文本，直接爬取涉及较高的
ToS/反爬风险（robots.txt和用户协议不允许未授权抓取）。本模块采用合规替代方案：
akshare 的 `stock_comment_em`（"千股千评"）接口——东方财富数据中心公开的官方
聚合数据（关注指数/机构参与度/综合得分/排名变化）。它不是原始社交媒体帖子，
但作为"散户情绪的量化代理指标"合规、稳定、免费；结构化数据转写成自然语言
描述喂给情感分析Prompt，同样能捕捉"市场关注度上升/下降""情绪转暖/转冷"信号。
如有合规渠道（官方开放平台账号或Choice/Wind舆情模块），可在本文件新增
fetch_xxx() 函数，产出同样的 {"raw_text","source","source_type":"社交媒体","date"}
格式，接入方式与 fetch_retail_sentiment() 一致，不需要改下游任何逻辑。
"""
import hashlib
import random
from datetime import datetime

from core.retry import with_retry
from market_data.provider import resolve_stock_code, is_a_share_code

try:
    import akshare as ak
    _AKSHARE_AVAILABLE = True
except ImportError:
    _AKSHARE_AVAILABLE = False


def _mock_retail_sentiment(company: str) -> dict:
    """降级实现：按公司名做种子的确定性伪随机情绪指标，仅用于演示/跑通链路"""
    seed = int(hashlib.sha1(company.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    score = round(rng.uniform(30, 90), 1)
    focus = round(rng.uniform(20, 95), 1)
    rank_change = rng.randint(-50, 50)
    trend = "上升" if rank_change > 0 else ("下降" if rank_change < 0 else "持平")
    text = (
        f"{company}市场热度：综合得分{score}（满分100），用户关注指数{focus}，"
        f"人气排名较此前{trend}{abs(rank_change)}位（演示数据，非真实行情）"
    )
    return {
        "raw_text": text,
        "source": "东方财富-千股千评（演示数据）",
        "source_type": "社交媒体",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "is_mock": True,
    }


@with_retry(max_attempts=2)
def _fetch_comment_em():
    return ak.stock_comment_em()


def fetch_retail_sentiment(company: str) -> dict | None:
    """
    取某公司当前的"千股千评"市场热度数据，转写成自然语言描述。
    找不到该公司数据、akshare未安装、或非A股代码时：
      - akshare不可用/非A股 → 返回mock演示数据（保证功能链路能跑通）
      - akshare可用但查不到这只股票（比如新股还没有评级） → 返回None（真实的"没有数据"，
        不伪造一条假数据出来）
    """
    code = resolve_stock_code(company)
    bare_code = is_a_share_code(code)

    if not (_AKSHARE_AVAILABLE and bare_code):
        return _mock_retail_sentiment(company)

    try:
        df = _fetch_comment_em()
        if df is None or df.empty:
            return _mock_retail_sentiment(company)

        code_col = "代码" if "代码" in df.columns else None
        if not code_col:
            return _mock_retail_sentiment(company)  # 列结构对不上，保守降级

        matched = df[df[code_col].astype(str).str.contains(bare_code, na=False)]
        if matched.empty:
            return None  # 真实查询结果：该股票暂无千股千评数据，不是失败，不用降级

        row = matched.iloc[0]

        def _get(col, default="未知"):
            return row[col] if col in df.columns and row[col] not in (None, "") else default

        text = (
            f"{company}市场热度（东方财富千股千评）：综合得分{_get('综合得分')}，"
            f"用户关注指数{_get('关注指数')}，机构参与度{_get('机构参与度')}，"
            f"目前人气排名第{_get('目前排名')}位，较此前{_get('上升')}"
        )
        return {
            "raw_text": text,
            "source": "东方财富-千股千评",
            "source_type": "社交媒体",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "is_mock": False,
        }
    except Exception:
        return _mock_retail_sentiment(company)


if __name__ == "__main__":
    for name in ["宁德时代", "贵州茅台"]:
        result = fetch_retail_sentiment(name)
        if result:
            tag = "（演示数据）" if result.get("is_mock") else ""
            print(f"{name}{tag}：{result['raw_text']}")
        else:
            print(f"{name}：暂无千股千评数据")
