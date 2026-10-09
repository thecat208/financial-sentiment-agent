"""
通用RSS采集器。相比逐个网站写定制爬虫，优先用RSS/官方接口更稳定、合规风险更低。
把你要监控的财经站点RSS地址填入 FEEDS 列表即可，无需为每个网站单独写解析逻辑。

如果部分数据源没有RSS，可以后续在 ingestion/ 下新增对应的采集脚本，
只要最终产出 {"raw_text":..., "source":..., "date":...} 格式的字典列表，
就能直接喂给 graphs/pipeline_graph.py 的 process() 处理。
"""
import feedparser
from datetime import datetime

# 示例，请替换为你实际要监控的财经RSS源地址
FEEDS = [
    # "https://example-finance-site.com/rss",
]


def fetch_rss_items(feed_url: str) -> list[dict]:
    parsed = feedparser.parse(feed_url)
    feed_title = parsed.feed.get("title", feed_url)

    items = []
    for entry in parsed.entries:
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        items.append({
            "raw_text": f"{title}。{summary}",
            "source": feed_title,
            "date": entry.get("published", datetime.now().strftime("%Y-%m-%d")),
        })
    return items


def fetch_all_feeds() -> list[dict]:
    all_items = []
    for feed_url in FEEDS:
        try:
            all_items.extend(fetch_rss_items(feed_url))
        except Exception as e:
            print(f"[rss_source] 拉取失败 {feed_url}：{e}")
    return all_items


if __name__ == "__main__":
    if not FEEDS:
        print("FEEDS 列表为空，请先填入至少一个RSS源地址再运行。")
    else:
        items = fetch_all_feeds()
        print(f"共拉取到 {len(items)} 条")
        for it in items[:3]:
            print(it)
