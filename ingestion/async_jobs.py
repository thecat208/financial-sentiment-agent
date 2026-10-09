"""
异步任务作业定义。

把"处理单条舆情"和"生成日报"包装成可提交到任务队列的作业：
submit_xxx() 立即返回 task_id，后台worker执行，get_task_status() 查询结果。
供 scheduler 与脚本/前端调用，提交方不再阻塞在耗时操作上。
"""
from core.task_queue import get_task_queue
from graphs.pipeline_graph import process
from graphs.report_graph import generate_report
from retrieval.rebuild import rebuild_vector_index
from config.settings import settings


def submit_ingest_item(item: dict, tenant_id: str = "default") -> str:
    """把一条舆情数据提交到异步队列处理（清洗→分析→入库→预警）。
    item支持可选的source_type字段（机构公告/新闻资讯/研报/社交媒体/其他），
    不传则按"新闻资讯"处理，兼容未细分来源的调用方。"""
    return get_task_queue().submit(
        process,
        (item.get("raw_text", ""),),
        {
            "source": item.get("source", "未知来源"),
            "date": item.get("date"),
            "tenant_id": tenant_id,
            "source_type": item.get("source_type"),
        },
    )


def submit_daily_report(date=None, tenant_id: str = "default") -> str:
    """把日报生成提交到异步队列（立即返回task_id，由worker完成统计/LLM小结/落盘/推送）"""
    return get_task_queue().submit(generate_report, (date,), {"tenant_id": tenant_id})


def submit_vector_rebuild(tenant_id: str = None) -> str:
    """把向量索引重建提交到异步队列（每日凌晨由调度器触发，重建期间旧索引继续服务）"""
    return get_task_queue().submit(rebuild_vector_index, (tenant_id,), {})


def ensure_worker(concurrency: int = None):
    """启动后台worker消费队列（幂等，重复调用不会起多个线程）"""
    get_task_queue().start(concurrency or settings.TASK_QUEUE_CONCURRENCY)


def get_task_status(task_id: str) -> dict:
    return get_task_queue().get_status(task_id)
