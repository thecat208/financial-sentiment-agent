"""
机构公告采集。
数据源：巨潮资讯网（cninfo.com.cn）—— 沪深京上市公司信息披露的官方权威渠道。
流程：
  1. 拉取指定公司近N天的公告列表（标题/类型/日期/PDF链接）
  2. 默认尝试下载PDF并抽取全文（需要 pdfplumber，ANNOUNCEMENT_FETCH_FULLTEXT 开关
     控制，未安装/下载失败/抽取失败都不阻塞——退化为只用标题，公告标题通常已含
     足够的关键信息）
  3. 输出统一的 {"raw_text","source","date","source_type":"机构公告"} 格式，与 ingestion/rss_source.py 的产出对齐，可直接喂给 async_jobs.submit_ingest_item()
akshare未安装/网络不通/接口异常时自动降级为mock公告（同 provider.py 降级约定），
链路始终能跑通，降级结果里带 is_mock=True 标记。
"""
import hashlib
import random
from datetime import datetime, timedelta

from core.retry import with_retry
from config.settings import settings
from market_data.provider import resolve_stock_code, is_a_share_code

try:
    import akshare as ak
    _AKSHARE_AVAILABLE = True
except ImportError:
    _AKSHARE_AVAILABLE = False

try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    _PDFPLUMBER_AVAILABLE = False

# akshare不同版本/东财数据源列名可能有细微差异，这里列出常见候选，取第一个能匹配上的，
# 避免因为列名对不上直接报错——同样的容错思路也用在下面的URL/日期字段解析上
_TITLE_COLS = ["公告标题", "announcementTitle", "标题"]
_DATE_COLS = ["公告时间", "公告日期", "announcementTime"]
_URL_COLS = ["公告链接", "adjunctUrl", "PDF链接", "url"]

_MOCK_ANNOUNCEMENT_TEMPLATES = [
    "关于近期经营情况的自愿性信息披露公告",
    "关于收到问询函并回复的公告",
    "第十届董事会第X次会议决议公告",
    "关于部分董事、高级管理人员减持股份计划的公告",
    "关于签署重大合同的公告",
]


def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _mock_announcements(company: str, days: int) -> list[dict]:
    """降级实现：按公司名做种子的确定性伪随机公告标题，仅用于演示/跑通链路"""
    seed = int(hashlib.sha1(company.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    n = rng.randint(1, 3)
    items = []
    for i in range(n):
        title = rng.choice(_MOCK_ANNOUNCEMENT_TEMPLATES)
        day_offset = rng.randint(0, max(days - 1, 0))
        date = (datetime.now() - timedelta(days=day_offset)).strftime("%Y-%m-%d")
        items.append({
            "raw_text": f"{company}公告：{title}",
            "source": "巨潮资讯网-公告（演示数据）",
            "source_type": "机构公告",
            "date": date,
            "is_mock": True,
        })
    return items


@with_retry(max_attempts=2)
def _fetch_cninfo_list(bare_code: str, start_date: str, end_date: str):
    return ak.stock_zh_a_disclosure_report_cninfo(
        symbol=bare_code,
        market="沪深京",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
    )


def _extract_pdf_fulltext(pdf_url: str) -> str:
    """尽力下载PDF并抽取文字，任何一步失败都返回空字符串，不抛异常影响主流程"""
    if not _PDFPLUMBER_AVAILABLE or not pdf_url:
        return ""
    try:
        import requests
        import io
        url = pdf_url if pdf_url.startswith("http") else f"https://static.cninfo.com.cn/{pdf_url}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        text_parts = []
        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            for page in pdf.pages[:20]:  # 超长财报只取前20页，避免抽取耗时过长
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)[: settings.ANNOUNCEMENT_FULLTEXT_MAX_CHARS]
    except Exception:
        return ""


def fetch_announcements(company: str, days: int = None) -> list[dict]:
    """
    拉取某公司近N天的公告，返回统一格式的列表：
    [{"raw_text":..., "source":..., "source_type":"机构公告", "date":..., "is_mock":bool}, ...]

    找不到真实数据（akshare未安装/非A股/网络不通/接口异常）时自动降级为mock公告。
    """
    days = days or settings.ANNOUNCEMENT_LOOKBACK_DAYS
    code = resolve_stock_code(company)
    bare_code = is_a_share_code(code)

    if not (_AKSHARE_AVAILABLE and bare_code):
        return _mock_announcements(company, days)

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        df = _fetch_cninfo_list(bare_code, start_date, end_date)
        if df is None or df.empty:
            return []  # 真实查询到"没有公告"是有效结果，不等同于降级，不返回mock数据

        title_col, date_col, url_col = (
            _pick_col(df, _TITLE_COLS), _pick_col(df, _DATE_COLS), _pick_col(df, _URL_COLS)
        )
        if not title_col:
            return _mock_announcements(company, days)  # 列结构对不上，保守降级

        items = []
        for _, row in df.iterrows():
            title = str(row[title_col])
            date = str(row[date_col])[:10] if date_col else end_date
            fulltext = ""
            if settings.ANNOUNCEMENT_FETCH_FULLTEXT and url_col:
                fulltext = _extract_pdf_fulltext(str(row[url_col]))

            raw_text = f"{company}公告：{title}"
            if fulltext:
                raw_text += f"\n\n公告全文摘录：\n{fulltext}"

            items.append({
                "raw_text": raw_text,
                "source": "巨潮资讯网-公告",
                "source_type": "机构公告",
                "date": date,
                "is_mock": False,
            })
        return items
    except Exception:
        return _mock_announcements(company, days)  # 网络/解析问题一律降级，不让采集失败中断流程


if __name__ == "__main__":
    for name in ["宁德时代", "贵州茅台"]:
        result = fetch_announcements(name, days=7)
        print(f"\n{name}：共 {len(result)} 条")
        for it in result[:2]:
            tag = "（演示数据）" if it.get("is_mock") else ""
            print(f"  [{it['date']}]{tag} {it['raw_text'][:60]}")
