"""
验收测试：多渠道推送与导出能力。

不需要真实的Webhook地址/SMTP账号/API Key——推送渠道测试用mock掉requests.post
的方式验证"HTTP请求构造对不对"（URL、签名、JSON body），而不是真的发出网络请求；
导出测试用真实的pandas/reportlab跑一遍，产出真实文件并验证内容。

用法：
    python scripts/test_p13_push_export.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_push_degrades_when_unconfigured():
    """所有渠道都未配置时，各自应该返回False（降级为打印）而不是抛异常"""
    for k in ["SERVERCHAN_KEY", "DINGTALK_WEBHOOK", "DINGTALK_SECRET", "FEISHU_WEBHOOK",
              "FEISHU_SECRET", "WECOM_WEBHOOK", "SLACK_WEBHOOK", "EMAIL_SMTP_HOST",
              "EMAIL_SMTP_USER", "EMAIL_SMTP_PASSWORD", "EMAIL_TO"]:
        os.environ.pop(k, None)

    import importlib
    import alert.push as push
    importlib.reload(push)

    for fn in [push.push_wechat, push.push_dingtalk, push.push_feishu,
               push.push_wecom, push.push_slack, push.push_email]:
        assert fn("测试标题", "测试内容") is False
    print("✅ test_push_degrades_when_unconfigured 通过")


def test_dingtalk_signature_and_payload():
    """钉钉：验证加签算法产出的URL格式对，且推送内容经过了PII脱敏"""
    os.environ["DINGTALK_WEBHOOK"] = "https://oapi.dingtalk.com/robot/send?access_token=FAKE"
    os.environ["DINGTALK_SECRET"] = "SECfaketest"

    import importlib
    import alert.push as push
    importlib.reload(push)

    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        class R:
            status_code = 200
            def json(self):
                return {"errcode": 0}
        return R()

    push.requests.post = fake_post
    ok = push.push_dingtalk("测试", "手机号13812345678")

    assert ok is True
    assert "timestamp=" in captured["url"] and "sign=" in captured["url"]
    assert "138****5678" in captured["json"]["text"]["content"]  # 已脱敏
    assert "13812345678" not in captured["json"]["text"]["content"]  # 原始号码不应出现
    print("✅ test_dingtalk_signature_and_payload 通过")


def test_feishu_signature_and_payload():
    os.environ["FEISHU_WEBHOOK"] = "https://open.feishu.cn/open-apis/bot/v2/hook/FAKE"
    os.environ["FEISHU_SECRET"] = "fakesecret"

    import importlib
    import alert.push as push
    importlib.reload(push)

    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        class R:
            status_code = 200
            def json(self):
                return {"code": 0}
        return R()

    push.requests.post = fake_post
    ok = push.push_feishu("测试", "内容")

    assert ok is True
    assert "timestamp" in captured["json"] and "sign" in captured["json"]
    assert captured["json"]["msg_type"] == "text"
    print("✅ test_feishu_signature_and_payload 通过")


def test_push_all_fanout_and_unknown_channel():
    """push_all按渠道列表逐个推送，未知渠道名被跳过不报错，返回每个渠道各自的结果"""
    for k in ["SERVERCHAN_KEY", "SLACK_WEBHOOK"]:
        os.environ.pop(k, None)

    import importlib
    import alert.push as push
    importlib.reload(push)

    results = push.push_all("t", "c", channels=["wechat", "slack", "not_a_real_channel"])
    assert results == {"wechat": False, "slack": False}  # 未知渠道被跳过，不出现在结果里
    print("✅ test_push_all_fanout_and_unknown_channel 通过")


def test_push_alert_backward_compatible():
    """push_alert()签名和返回值类型保持不变（graphs/pipeline_graph.py的push_node依赖这一点）"""
    import importlib
    import alert.push as push
    importlib.reload(push)

    result = push.push_alert(company="测试公司", event_type="监管处罚",
                              reason="测试原因", source_text="测试原文")
    assert isinstance(result, bool)
    print("✅ test_push_alert_backward_compatible 通过")


def test_export_to_excel():
    from export.report_export import export_to_excel
    import pandas as pd

    fake_report = {
        "date": "2026-09-10", "tenant_id": "default", "company_stats": [],
        "company_summaries": [
            {"company": "贵州茅台", "summary": "偏正面。",
             "stats": {"cnt": 5, "avg_score": 0.4, "negative_cnt": 1}},
        ],
    }
    path = export_to_excel(fake_report, path="/tmp/p13_test.xlsx")
    assert os.path.exists(path)

    df_summary = pd.read_excel(path, sheet_name="汇总")
    df_detail = pd.read_excel(path, sheet_name="详情")
    assert df_summary.iloc[0]["公司"] == "贵州茅台"
    assert df_summary.iloc[0]["条数"] == 5
    assert df_detail.iloc[0]["小结"] == "偏正面。"
    print("✅ test_export_to_excel 通过")


def test_export_to_pdf():
    from export.report_export import export_to_pdf

    fake_report = {
        "date": "2026-09-10", "tenant_id": "default", "company_stats": [],
        "company_summaries": [
            {"company": "宁德时代", "summary": "订单增长。",
             "stats": {"cnt": 3, "avg_score": 0.5, "negative_cnt": 0}},
        ],
    }
    path = export_to_pdf(fake_report, path="/tmp/p13_test.pdf")
    assert os.path.exists(path)
    with open(path, "rb") as f:
        header = f.read(5)
    assert header == b"%PDF-"
    assert os.path.getsize(path) > 500  # 不是空文件/损坏文件
    print("✅ test_export_to_pdf 通过（PDF文件已生成，%PDF-文件头正确）")


def test_export_empty_report():
    """没有任何公司数据的空日报也应该能正常导出，不报错"""
    from export.report_export import export_to_excel, export_to_pdf

    empty_report = {"date": "2026-09-11", "tenant_id": "default",
                     "company_stats": [], "company_summaries": []}
    xlsx_path = export_to_excel(empty_report, path="/tmp/p13_test_empty.xlsx")
    pdf_path = export_to_pdf(empty_report, path="/tmp/p13_test_empty.pdf")
    assert os.path.exists(xlsx_path) and os.path.exists(pdf_path)
    print("✅ test_export_empty_report 通过")


if __name__ == "__main__":
    test_push_degrades_when_unconfigured()
    test_dingtalk_signature_and_payload()
    test_feishu_signature_and_payload()
    test_push_all_fanout_and_unknown_channel()
    test_push_alert_backward_compatible()
    test_export_to_excel()
    test_export_to_pdf()
    test_export_empty_report()

    print("\n全部验收测试通过 ✅")
