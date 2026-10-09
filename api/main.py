"""
FastAPI 对外接口层。

定位：这一层不重新实现任何业务逻辑，只是给已有的 Streamlit 页面背后那套
处理链路（graphs/*.py、storage/repository.py、analysis/backtest.py 等）
包一层HTTP外壳，方便其他系统（比如内部工作流、企业IM机器人、定时任务平台）
用REST接口的方式接入，而不是只能通过Streamlit界面手动操作。

启动方式：
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
或者：
    python -m api.main

文档地址：启动后访问 http://localhost:8000/docs （Swagger UI，自动生成）
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from storage.db import init_db
from storage.usage_tracker import record_api_call
from api.routers import query, ingest, companies, alerts, market, industry, usage

app = FastAPI(
    title="金融舆情分析系统 API",
    description=(
        "提供舆情数据提交、智能问答、日报查看、预警统计、舆情-行情关联分析、"
        "多公司对比与行业热度榜等接口。所有涉及投资/行情的接口返回内容均"
        "不构成投资建议，仅供研究参考。"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router)
app.include_router(ingest.router)
app.include_router(companies.router)
app.include_router(alerts.router)
app.include_router(market.router)
app.include_router(industry.router)
app.include_router(usage.router)


@app.middleware("http")
async def _usage_tracking_middleware(request: Request, call_next):
    """
    记录每一次 /api/* 请求（不含/health和/docs这类系统端点，
    那些不是业务用量）。用中间件而不是每个路由里手动调用一次，保证新增路由
    自动被计入，不会有人忘记埋点导致某个端点的用量统计缺失。

    tenant_id从请求头取（和 api/deps.get_tenant_id 逻辑一致，但中间件层
    拿不到FastAPI的依赖注入结果，所以在这里重新读一次请求头，两处逻辑
    需要保持同步——都是"取X-Tenant-Id，没有则用默认租户"这一句，改动风险低）。
    """
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/api/"):
        tenant_id = request.headers.get("X-Tenant-Id") or settings.DEFAULT_TENANT_ID
        try:
            record_api_call(tenant_id, path)
        except Exception:
            pass  # 用量统计失败不应该影响正常响应返回
    return response


@app.on_event("startup")
def _on_startup():
    # 确保表结构存在/迁移已跑过，第一个请求进来之前数据库就是就绪状态，
    # 不用依赖"运气好第一个请求刚好触发了get_conn()里的建表逻辑"。
    init_db()


@app.get("/health", tags=["系统"], summary="健康检查")
def health():
    return {"status": "ok"}


@app.get("/", tags=["系统"], summary="根路径，指向API文档")
def root():
    return {"message": "金融舆情分析系统 API 已启动", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
