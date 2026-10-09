"""
行情数据获取。
定位说明：本模块及下游（analysis/backtest.py、Streamlit"舆情-行情联动"页）只是
**研究工具**，不构成任何投资建议，详见 core/compliance.py 的 DISCLAIMER。
数据源：
  - akshare（默认首选）：开源免费，覆盖A股日线行情，内网可直接用，仅支持6位数字代码的A股。
  - mock（降级兜底）：akshare 未安装/网络不通/非A股代码时自动降级，本地按(代码,日期范围)
    确定性生成伪随机价格序列（可复现，方便演示/回测），但**绝不能当成真实行情使用**，
    降级结果带 `is_mock=True` 标记，上层UI必须显著提示。
公司名 -> 股票代码映射复用 knowledge_graph/seed_data.py 的 COMPANIES 表（唯一数据源），
查不到时把输入原样当代码尝试。
"""
import hashlib
import random
from datetime import datetime, timedelta

from core.cache import cached, make_key, get_cache
from core.retry import with_retry
from config.settings import settings
from knowledge_graph.seed_data import COMPANIES

try:
    import akshare as ak
    _AKSHARE_AVAILABLE = True
except ImportError:  # 未安装akshare时整个模块自动降级为mock，不影响其他功能
    _AKSHARE_AVAILABLE = False


def resolve_stock_code(company_or_code: str) -> str:
    """
    公司名 -> 股票代码。复用知识图谱种子库的别名表（"宁王"/"CATL"都能识别到300750）；
    查不到就假设调用方传的已经是代码，原样返回。
    """
    if not company_or_code:
        return company_or_code
    for canonical, info in COMPANIES.items():
        if company_or_code == canonical or company_or_code in info.get("aliases", []):
            return info["stock"]
    return company_or_code


def _is_a_share(code: str) -> str | None:
    """判断是否是A股代码，是的话返回给akshare用的6位纯数字代码，否则返回None"""
    bare = code.split(".")[0]
    if bare.isdigit() and len(bare) == 6:
        return bare
    return None


def is_a_share_code(code: str) -> str | None:
    """_is_a_share的公开别名，供其他模块（如P10的公告/社交采集）复用同一份判断逻辑"""
    return _is_a_share(code)


def _mock_daily_prices(code: str, start_date: str, end_date: str) -> list[dict]:
    """
    降级实现：按 code 做种子的确定性伪随机游走，只用于演示/跑通链路。
    价格从 100 起步，每日涨跌幅在 [-3%, 3%] 之间。
    """
    seed = int(hashlib.sha1(code.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    rows = []
    price = 100.0
    d = start
    while d <= end:
        if d.weekday() < 5:  # 只生成交易日（周一到周五），模拟真实行情节奏
            pct = rng.uniform(-0.03, 0.03)
            open_p = price
            close_p = round(price * (1 + pct), 2)
            high_p = round(max(open_p, close_p) * (1 + rng.uniform(0, 0.01)), 2)
            low_p = round(min(open_p, close_p) * (1 - rng.uniform(0, 0.01)), 2)
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "open": round(open_p, 2),
                "close": close_p,
                "high": high_p,
                "low": low_p,
                "volume": rng.randint(500000, 5000000),
                "pct_change": round(pct * 100, 2),
                "is_mock": True,
            })
            price = close_p
        d += timedelta(days=1)
    return rows


@with_retry(max_attempts=2)
def _fetch_akshare_daily(bare_code: str, start_date: str, end_date: str) -> list[dict]:
    df = ak.stock_zh_a_hist(
        symbol=bare_code,
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust="qfq",
    )
    if df is None or df.empty:
        raise ValueError(f"akshare返回空数据：{bare_code}")

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "date": str(r["日期"]),
            "open": float(r["开盘"]),
            "close": float(r["收盘"]),
            "high": float(r["最高"]),
            "low": float(r["最低"]),
            "volume": int(r["成交量"]),
            "pct_change": float(r["涨跌幅"]),
            "is_mock": False,
        })
    return rows


def _get_daily_prices_uncached(company_or_code: str, start_date: str, end_date: str) -> dict:
    code = resolve_stock_code(company_or_code)
    bare_code = _is_a_share(code)

    if settings.MARKET_DATA_PROVIDER == "akshare" and _AKSHARE_AVAILABLE and bare_code:
        try:
            rows = _fetch_akshare_daily(bare_code, start_date, end_date)
            return {"code": code, "provider": "akshare", "is_mock": False, "prices": rows}
        except Exception:
            pass  # 拿不到真实数据就降级为mock，不让整个功能因为网络问题不可用

    rows = _mock_daily_prices(code, start_date, end_date)
    return {"code": code, "provider": "mock", "is_mock": True, "prices": rows}


@cached(
    "market:daily",
    settings.MARKET_DATA_CACHE_TTL,
    key_parts_fn=lambda company_or_code, start_date, end_date: (company_or_code, start_date, end_date),
)
def get_daily_prices(company_or_code: str, start_date: str, end_date: str) -> dict:
    """
    取某公司/代码在 [start_date, end_date] 的日线行情。

    返回 {"code": 股票代码, "provider": "akshare"/"mock", "is_mock": bool,
          "prices": [{"date","open","close","high","low","volume","pct_change","is_mock"}, ...]}

    结果按 (公司/代码, 起止日期) 缓存 MARKET_DATA_CACHE_TTL 秒，避免重复请求行情接口。
    """
    return _get_daily_prices_uncached(company_or_code, start_date, end_date)


def get_price_on_or_after(company_or_code: str, target_date: str, lookahead_days: int = 10) -> dict | None:
    """
    取 target_date 当天或之后最近一个交易日的行情（预警日/回测起点常常不是交易日）。
    找不到返回 None。
    """
    end = (datetime.strptime(target_date, "%Y-%m-%d") + timedelta(days=lookahead_days)).strftime("%Y-%m-%d")
    result = get_daily_prices(company_or_code, target_date, end)
    prices = result["prices"]
    for row in prices:
        if row["date"] >= target_date:
            return row
    return None


def get_price_n_trading_days_after(company_or_code: str, target_date: str, n: int,
                                    lookahead_days: int = 30) -> dict | None:
    """
    取 target_date（含）之后第 n 个交易日的行情，用于"预警触发后N个交易日"的回测统计。
    n=0 表示预警当天（或之后最近交易日）的行情。
    """
    end = (datetime.strptime(target_date, "%Y-%m-%d") + timedelta(days=lookahead_days)).strftime("%Y-%m-%d")
    result = get_daily_prices(company_or_code, target_date, end)
    prices = [row for row in result["prices"] if row["date"] >= target_date]
    if len(prices) <= n:
        return None
    return prices[n]
