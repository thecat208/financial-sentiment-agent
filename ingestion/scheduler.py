"""
定时任务调度。两个任务：
1. 每N分钟拉取RSS数据，逐条提交到异步队列处理（清洗→入库→分析→预警）
2. 每天固定时间提交一次日报生成任务（复用当天已入库的数据）

采集和日报不在调度线程里同步执行，而是提交到任务队列立即返回，
由后台worker异步消费——调度器不被耗时操作阻塞，多条任务可并发处理，
也为将来拆成独立进程/API服务铺路。另有每日凌晨的向量索引重建任务。

运行方式：python ingestion/scheduler.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apscheduler.schedulers.blocking import BlockingScheduler
from ingestion.rss_source import fetch_all_feeds
from ingestion.async_jobs import (
    ensure_worker, submit_ingest_item, submit_daily_report, submit_vector_rebuild,
)
from core.logger import get_logger
from config.settings import settings

logger = get_logger("scheduler")

REPORT_HOUR = int(os.getenv("REPORT_HOUR", 18))
REPORT_MINUTE = int(os.getenv("REPORT_MINUTE", 0))
VECTOR_REBUILD_HOUR = int(os.getenv("VECTOR_REBUILD_HOUR", settings.VECTOR_REBUILD_HOUR))


def run_ingest_job():
    items = fetch_all_feeds()
    logger.info(f"本次拉取到 {len(items)} 条数据，已提交异步队列")
    for item in items:
        try:
            task_id = submit_ingest_item(item)
            logger.info(f"已提交处理任务：{task_id[:8]}…")
        except Exception as e:
            logger.warning(f"单条提交失败：{e}")


def run_daily_report_job():
    try:
        task_id = submit_daily_report()
        logger.info(f"日报生成任务已提交：{task_id}")
    except Exception as e:
        logger.warning(f"日报任务提交失败：{e}")


def run_vector_rebuild_job():
    """每日凌晨重建向量索引（提交异步队列，旧索引在切换前持续服务）"""
    try:
        task_id = submit_vector_rebuild()
        logger.info(f"向量索引重建任务已提交：{task_id}")
    except Exception as e:
        logger.warning(f"向量重建任务提交失败：{e}")


def start_scheduler(interval_minutes: int = 30):
    ensure_worker()  # 启动后台worker消费队列（幂等）
    scheduler = BlockingScheduler()
    scheduler.add_job(run_ingest_job, "interval", minutes=interval_minutes)
    scheduler.add_job(run_daily_report_job, "cron", hour=REPORT_HOUR, minute=REPORT_MINUTE)
    scheduler.add_job(run_vector_rebuild_job, "cron", hour=VECTOR_REBUILD_HOUR, minute=0)

    logger.info(f"已启动：每 {interval_minutes} 分钟采集一次，"
                 f"每天 {REPORT_HOUR:02d}:{REPORT_MINUTE:02d} 生成日报，"
                 f"每天 {VECTOR_REBUILD_HOUR:02d}:00 重建向量索引（Ctrl+C 停止）")
    run_ingest_job()  # 启动时先跑一次采集，方便立即看到效果
    scheduler.start()


if __name__ == "__main__":
    start_scheduler()
