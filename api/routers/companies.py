"""
公司列表、情感趋势、多维度打分、多公司对比、日报查看。
全部直接复用 storage/repository.py 和 graphs/report_graph.py 里已有的查询/生成函数。
"""
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from api.deps import get_tenant_id, require_api_key, rate_limit
from api.schemas import CompanyCompareRequest
from storage.repository import (
    get_all_companies,
    get_company_daily_trend,
    get_records_by_company,
    get_dimension_avg,
    get_event_type_distribution,
)
from graphs.report_graph import generate_report_async
from export.report_export import export_to_excel, export_to_pdf, load_report_json

router = APIRouter(prefix="/api/v1", tags=["公司与趋势"], dependencies=[Depends(require_api_key)])


@router.get("/companies", summary="已识别公司列表")
def list_companies(tenant_id: str = Depends(get_tenant_id)):
    return {"companies": get_all_companies(tenant_id=tenant_id)}


@router.get("/companies/{company}/trend", summary="某公司近N天情感趋势")
def company_trend(company: str, days: int = 30, tenant_id: str = Depends(get_tenant_id)):
    return {"company": company, "days": days, "trend": get_company_daily_trend(company, days=days, tenant_id=tenant_id)}


@router.get("/companies/{company}/records", summary="某公司近N天舆情原始记录")
def company_records(company: str, days: int = 7, tenant_id: str = Depends(get_tenant_id)):
    return {"company": company, "days": days, "records": get_records_by_company(company, days=days, tenant_id=tenant_id)}


@router.get("/companies/{company}/dimensions", summary="某公司近N天多维度情感均值")
def company_dimensions(company: str, days: int = 30, tenant_id: str = Depends(get_tenant_id)):
    return {"company": company, "days": days, "dimensions": get_dimension_avg(company, days=days, tenant_id=tenant_id)}


@router.get("/companies/{company}/event-types", summary="某公司近N天事件类型分布")
def company_event_types(company: str, days: int = 30, tenant_id: str = Depends(get_tenant_id)):
    return {
        "company": company, "days": days,
        "distribution": get_event_type_distribution(days=days, company=company, tenant_id=tenant_id),
    }


@router.post("/companies/compare", summary="多公司情感趋势对比（2-5家公司）")
def compare_companies(req: CompanyCompareRequest, tenant_id: str = Depends(get_tenant_id)):
    return {
        "days": req.days,
        "series": {
            c: get_company_daily_trend(c, days=req.days, tenant_id=tenant_id)
            for c in req.companies
        },
    }


@router.get("/reports/daily", summary="每日舆情日报（不传date取最新一天）")
async def daily_report(date: str = None, tenant_id: str = Depends(get_tenant_id)):
    # 日报含多家公司的小结LLM调用，改async def走异步图
    try:
        return await generate_report_async(date, tenant_id=tenant_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"日报生成失败（请检查LLM API Key是否配置好）：{e}")


@router.get("/reports/daily/export", summary="导出日报为Excel/PDF（date必填）")
def export_daily_report(date: str, format: str = "xlsx", tenant_id: str = Depends(get_tenant_id)):
    """
    format=xlsx 或 pdf。只能导出已经生成过的日报（先调 /reports/daily 或跑
    scripts/run_daily_report.py），不会现场触发LLM重新生成——导出应该是个轻量、
    免费的动作，不该意外产生LLM调用费用。找不到对应日期的结构化数据时返回404，
    并提示可能是旧版本生成的日报（那种没有存JSON侧车文件）。
    """
    if format not in ("xlsx", "pdf"):
        raise HTTPException(status_code=400, detail="format 只支持 xlsx 或 pdf")

    report_json = load_report_json(date, tenant_id=tenant_id)
    if report_json is None:
        raise HTTPException(
            status_code=404,
            detail=f"找不到 {date} 的日报结构化数据。请先生成该日期的日报"
                   f"（GET /reports/daily?date={date} 或 scripts/run_daily_report.py），"
                   f"如果是旧版本生成的日报则不支持导出，需要重新生成。",
        )

    if format == "xlsx":
        path = export_to_excel(report_json)
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        path = export_to_pdf(report_json)
        media_type = "application/pdf"

    return FileResponse(path, media_type=media_type, filename=os.path.basename(path))
