"""
Locust 压测脚本。

压测对象：FastAPI 服务（api/main.py），重点打"涉及LLM的异步路由"，
观察异步化后事件循环在并发下的吞吐与延迟表现。

前置条件：
    1. 服务已启动：python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
    2. 压测端安装 locust（与项目环境隔离更干净，装在哪个环境都行）：
        pip install locust

启动方式（推荐 headless 命令行模式，结果直接打印+出HTML报告）：
    # 5并发用户、每秒拉起1个、压2分钟
    locust -f scripts/locustfile.py --host http://localhost:8000 \
        --headless -u 5 -r 1 -t 2m --html scripts/eval_results/locust_report.html

    # 更高并发（异步化的主战场：协程不占线程，几十并发事件循环也扛得住，
    # 瓶颈会转移到LLM服务商配额——这正是要观察的点）
    locust -f scripts/locustfile.py --host http://localhost:8000 \
        --headless -u 20 -r 2 -t 3m

API Key：默认从项目根目录 .env 里读 API_KEY（脚本自己 load_dotenv），
也可以用环境变量覆盖：set LOCUST_API_KEY=xxx

结果判读：
    - 看聚类的 p50/p95 延迟和 RPS：LLM 路由的延迟主要取决于 DeepSeek 响应速度；
    - /qa 命中 LLM 缓存时会秒回（cache.py 的 cached 装饰器），这类样本会把
      p50 拉低——问题池随机轮换可以减少缓存命中，但语义相近的问题仍可能命中，
      判读时注意区分"缓存命中"和"真实LLM响应"两类样本；
    - 如果 429 增多，说明触发了限流器（core/rate_limiter.py），调大
      .env 里 RATE_LIMIT_* 或降低并发。
"""
import os
import random

from dotenv import load_dotenv
from locust import HttpUser, between, task

load_dotenv()  # 读项目根目录 .env，拿 API_KEY

API_KEY = os.getenv("LOCUST_API_KEY") or os.getenv("API_KEY") or ""

# 固定问题池：问题固定才能跨轮次比较；随机抽取减少LLM缓存命中
QA_QUESTIONS = [
    "贵州茅台最近的舆情怎么样",
    "宁德时代有什么负面新闻吗",
    "比亚迪最近的舆情怎么样",
    "招商银行最近有什么新闻",
    "中国平安的舆情风险如何",
    "隆基绿能最近表现如何",
    "海天味业有什么舆论动态",
    "东方财富最近的舆情怎么样",
]

ROUTER_QUESTIONS = [
    "帮我生成今天的日报",
    "贵州茅台最近的舆情怎么样",
    "今天有多少条预警",
]


class SentimentApiUser(HttpUser):
    """模拟一个持续调用API的下游系统（等待1~3秒模拟真实调用节奏）"""

    wait_time = between(1, 3)

    def on_start(self):
        """每个虚拟用户启动时设置公共请求头"""
        self.client.headers.update({
            "X-API-Key": API_KEY,
            "Content-Type": "application/json; charset=utf-8",
        })

    @task(3)
    def qa(self):
        """RAG问答（权重3）：检索+LLM生成，异步化后走 ask_async"""
        self.client.post(
            "/api/v1/qa",
            json={"query": random.choice(QA_QUESTIONS)},
            name="POST /api/v1/qa (LLM)",
            timeout=120,
        )

    @task(1)
    def query_router(self):
        """智能入口（权重1）：多一次意图识别LLM调用，再路由到具体能力"""
        self.client.post(
            "/api/v1/query",
            json={"query": random.choice(ROUTER_QUESTIONS)},
            name="POST /api/v1/query (LLM x2)",
            timeout=180,
        )

    @task(1)
    def health(self):
        """健康检查（权重1）：零LLM开销，用来对照'纯事件循环吞吐'与'LLM吞吐'的差距"""
        self.client.get("/health", name="GET /health (baseline)")

    # 日报接口默认不压：每次请求都会真实调LLM生成整份日报并落盘，
    # 高并发下既烧钱又会产生大量报告文件。要压的话取消下面注释（低并发手动跑）：
    # @task(1)
    # def daily_report(self):
    #     self.client.get(
    #         "/api/v1/companies/reports/daily",
    #         name="GET /reports/daily (LLM xN)",
    #         timeout=300,
    #     )
