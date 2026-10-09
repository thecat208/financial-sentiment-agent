"""
FastAPI 公共依赖项。

设计原则：这一层只做"HTTP相关的横切逻辑"（取租户ID、校验API Key、限流），
不重新实现任何业务逻辑——业务逻辑全部复用 storage/graphs/chains/analysis/market_data
里已经写好、也被 Streamlit 和 scripts/ 用着的函数，保证"同一个功能，Streamlit和API
调用的是完全同一份代码"，不会出现两边逻辑不一致的问题。
"""
from fastapi import Header, HTTPException, status

from config.settings import settings
from core.rate_limiter import allow


def get_tenant_id(x_tenant_id: str = Header(default=None, alias="X-Tenant-Id")) -> str:
    """
    从请求头取租户ID，不传则用默认租户。

    和Streamlit demo里的"租户ID输入框"是同一个概念——生产环境应该从网关/SSO
    传下来的登录态里取，这里简化成请求头，方便调用方直接指定。
    """
    return x_tenant_id or settings.DEFAULT_TENANT_ID


def get_user_id(x_user_id: str = Header(default=None, alias="X-User-Id")) -> str | None:
    """
    从请求头取操作人标识，用于审计日志（storage/audit_log.py）。
    和 tenant_id 目前的处境完全一样——这是调用方自报的标识，不是经过身份验证的
    登录态，本项目还没有真正的SSO/用户认证体系。返回None时下游的
    record_audit()会记为"unknown"，让"这条操作到底是谁做的都不知道"这件事
    在审计记录里显式可见，而不是悄悄跳过不记。
    """
    return x_user_id


def require_api_key(x_api_key: str = Header(default=None, alias="X-API-Key")) -> None:
    """
    API Key校验。settings.API_KEY留空时不做任何校验（本地开发/演示默认），
    设置了值之后所有 /api/* 请求都必须带上匹配的 X-API-Key 头，否则拒绝。

    没有做更复杂的OAuth2/JWT，是因为这是个人/中小团队项目的边界防护，
    不是面向公网的多方SaaS——如果要接企业级账号体系，替换这个函数的实现即可，
    下游路由的依赖签名不用改。
    """
    if not settings.API_KEY:
        return
    if x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少或无效的 X-API-Key",
        )


def rate_limit(tenant_id: str) -> None:
    """
    按租户限流，超限抛429。复用 core/rate_limiter.py 的全局令牌桶+单租户限流，
    和 graphs/qa_graph.ask()、graphs/router_graph.handle_query() 内部用的是同一套限流器
    ——调用方注意：这两个函数自带限流（rate_limit参数），本模块的路由会显式传
    rate_limit=False 给它们，改成统一在路由层限流一次，避免同一个请求被计两次令牌。
    """
    if not allow(tenant_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="请求过于频繁，请稍后再试",
        )
