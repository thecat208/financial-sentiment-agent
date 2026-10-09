"""
日报导出为 Excel / PDF。

输入是 graphs/report_graph.generate_report() 的返回结果（dict，包含
company_stats/company_summaries/report_text/date），不重新查库/不重新调LLM，
纯粹是"把已经生成好的日报内容转换成另一种文件格式"。
Excel用 openpyxl（通过 pandas.DataFrame.to_excel）：两个sheet，"汇总"是可以
拿去做数据分析的结构化表格，"详情"是每家公司的文字小结；PDF用 reportlab：
中文用内置CID字体 STSong-Light（无需额外字体文件），标题/正文分层用
Platypus 的段落样式排版。
"""
import os
import json
from datetime import datetime

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

from core.compliance import DISCLAIMER

EXPORT_DIR = os.getenv("EXPORT_DIR", "./exports")
REPORTS_DIR = os.getenv("REPORTS_DIR", "./reports")

_CJK_FONT = "STSong-Light"
_font_registered = False


def load_report_json(date: str, tenant_id: str = "default") -> dict | None:
    """
    读取 graphs/report_graph.save_report_node() 落盘的结构化JSON日报数据，
    没有的话返回None（比如该日期的日报只有.md没有.json，调用方应该提示用户
    "该日期不支持导出，重新生成日报后即可导出"，而不是现场重新调一次LLM生成
    ——导出应该是个轻量、免费的动作，不该在用户点一下导出按钮时意外触发
    一次LLM调用）。
    """
    path = os.path.join(REPORTS_DIR, tenant_id, f"{date}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ensure_cjk_font():
    """reportlab内置CID字体，首次用时注册一次即可，重复注册会报错所以加个标记位"""
    global _font_registered
    if not _font_registered:
        pdfmetrics.registerFont(UnicodeCIDFont(_CJK_FONT))
        _font_registered = True


def _default_path(report_result: dict, ext: str) -> str:
    tenant_id = report_result.get("tenant_id", "default")
    tenant_dir = os.path.join(EXPORT_DIR, tenant_id)
    os.makedirs(tenant_dir, exist_ok=True)
    return os.path.join(tenant_dir, f"{report_result['date']}.{ext}")


def export_to_excel(report_result: dict, path: str = None) -> str:
    """
    导出日报为Excel，返回文件路径。
    Sheet1「汇总」：公司/条数/平均情感分/负面条数，结构化表格方便二次分析、筛选、透视。
    Sheet2「详情」：公司/LLM生成的文字小结，一行一家公司，方便直接阅读。
    """
    path = path or _default_path(report_result, "xlsx")

    summaries = report_result.get("company_summaries") or []
    summary_rows = [{
        "公司": s["company"],
        "条数": s["stats"]["cnt"],
        "平均情感分": round(s["stats"].get("avg_score") or 0, 3),
        "负面条数": s["stats"]["negative_cnt"],
    } for s in summaries]
    detail_rows = [{"公司": s["company"], "小结": s["summary"]} for s in summaries]

    df_summary = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame(columns=["公司", "条数", "平均情感分", "负面条数"])
    df_detail = pd.DataFrame(detail_rows) if detail_rows else pd.DataFrame(columns=["公司", "小结"])

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="汇总", index=False)
        df_detail.to_excel(writer, sheet_name="详情", index=False)

        # 详情sheet的"小结"列文字较长，加宽列宽+自动换行，不然默认列宽看着像被截断了
        ws = writer.sheets["详情"]
        ws.column_dimensions["A"].width = 16
        ws.column_dimensions["B"].width = 100
        from openpyxl.styles import Alignment
        for row in ws.iter_rows(min_row=2, max_col=2):
            row[1].alignment = Alignment(wrap_text=True, vertical="top")

    return path


def export_to_pdf(report_result: dict, path: str = None) -> str:
    """导出日报为PDF，返回文件路径。中文用reportlab内置CID字体，不需要额外装字体文件。"""
    _ensure_cjk_font()
    path = path or _default_path(report_result, "pdf")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("CJKTitle", parent=styles["Title"], fontName=_CJK_FONT, fontSize=18, leading=22)
    h2_style = ParagraphStyle("CJKHeading2", parent=styles["Heading2"], fontName=_CJK_FONT, fontSize=13, leading=18,
                               spaceBefore=10, spaceAfter=4)
    body_style = ParagraphStyle("CJKBody", parent=styles["BodyText"], fontName=_CJK_FONT, fontSize=10, leading=16)
    meta_style = ParagraphStyle("CJKMeta", parent=styles["BodyText"], fontName=_CJK_FONT, fontSize=9, leading=14,
                                 textColor=colors.grey)

    story = [Paragraph(f"金融舆情日报 {report_result['date']}", title_style), Spacer(1, 8 * mm)]

    summaries = report_result.get("company_summaries") or []
    if not summaries:
        story.append(Paragraph("当日暂无相关舆情数据。", body_style))
    else:
        total = sum(s["stats"]["cnt"] for s in summaries)
        story.append(Paragraph(f"共监测到 {len(summaries)} 家公司、{total} 条相关舆情。", body_style))
        story.append(Spacer(1, 4 * mm))

        table_data = [["公司", "条数", "平均情感分", "负面条数"]] + [
            [s["company"], str(s["stats"]["cnt"]), str(round(s["stats"].get("avg_score") or 0, 3)),
             str(s["stats"]["negative_cnt"])]
            for s in summaries
        ]
        table = Table(table_data, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), _CJK_FONT),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(table)
        story.append(Spacer(1, 6 * mm))

        for s in summaries:
            story.append(Paragraph(s["company"], h2_style))
            # PDF里换行符要转成<br/>，Paragraph不会自动处理裸的\n
            story.append(Paragraph(s["summary"].replace("\n", "<br/>"), body_style))
            story.append(Spacer(1, 3 * mm))

    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(DISCLAIMER.strip("*\n-").replace("\n", " "), meta_style))

    doc = SimpleDocTemplate(path, pagesize=A4, topMargin=18 * mm, bottomMargin=18 * mm)
    doc.build(story)
    return path
