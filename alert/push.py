"""
多渠道消息推送。

降级约定：每个渠道独立配置，没配对应的环境变量就跳过实际发送、只打印日志，
不会因为某个渠道没配置就报错中断流程。
支持渠道：微信（Server酱）、邮件（SMTP，标准库）、钉钉/飞书/企业微信自定义机器人
（Webhook，钉钉/飞书带加签）、Slack Incoming Webhook。
`ALERT_PUSH_CHANNELS` 环境变量控制启用哪些渠道（逗号分隔），默认只有"wechat"，
多个渠道写 "wechat,dingtalk,email" 这样；各渠道凭证未配置时自动跳过、不影响其他渠道。
所有渠道推送前都对内容做PII脱敏，避免舆情原文夹带的隐私信息经推送渠道外泄。
"""
import os
import time
import hmac
import hashlib
import base64
import urllib.parse
import smtplib
from email.mime.text import MIMEText
from email.header import Header

import requests

from core.desensitize import desensitize

SERVERCHAN_KEY = os.getenv("SERVERCHAN_KEY", "")
DINGTALK_WEBHOOK = os.getenv("DINGTALK_WEBHOOK", "")
DINGTALK_SECRET = os.getenv("DINGTALK_SECRET", "")
FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK", "")
FEISHU_SECRET = os.getenv("FEISHU_SECRET", "")
WECOM_WEBHOOK = os.getenv("WECOM_WEBHOOK", "")
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK", "")

EMAIL_SMTP_HOST = os.getenv("EMAIL_SMTP_HOST", "")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", 465))
EMAIL_SMTP_USER = os.getenv("EMAIL_SMTP_USER", "")
EMAIL_SMTP_PASSWORD = os.getenv("EMAIL_SMTP_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", "") or EMAIL_SMTP_USER
EMAIL_TO = [addr.strip() for addr in os.getenv("EMAIL_TO", "").split(",") if addr.strip()]
EMAIL_USE_SSL = os.getenv("EMAIL_USE_SSL", "true").lower() in ("1", "true", "yes")

ALERT_PUSH_CHANNELS = [c.strip() for c in os.getenv("ALERT_PUSH_CHANNELS", "wechat").split(",") if c.strip()]


def push_wechat(title: str, content: str) -> bool:
    """Server酱个人微信推送（原有渠道，未改动行为）"""
    title = desensitize(title)
    content = desensitize(content)
    if not SERVERCHAN_KEY or SERVERCHAN_KEY == "your_serverchan_key":
        print(f"[push_wechat] 未配置 SERVERCHAN_KEY，跳过实际推送，仅打印：\n{title}\n{content}")
        return False

    url = f"https://sctapi.ftqq.com/{SERVERCHAN_KEY}.send"
    try:
        resp = requests.post(url, data={"title": title, "desp": content}, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"[push_wechat] 推送失败：{e}")
        return False


def push_dingtalk(title: str, content: str) -> bool:
    """
    钉钉自定义机器人。DINGTALK_SECRET留空时按"未开启加签"处理，直接用Webhook地址发送
    （钉钉机器人的安全设置也支持"自定义关键词"等不需要加签的方式，此时不用配SECRET）。
    """
    title, content = desensitize(title), desensitize(content)
    if not DINGTALK_WEBHOOK:
        print(f"[push_dingtalk] 未配置 DINGTALK_WEBHOOK，跳过实际推送，仅打印：\n{title}\n{content}")
        return False

    url = DINGTALK_WEBHOOK
    if DINGTALK_SECRET:
        timestamp = str(round(time.time() * 1000))
        string_to_sign = f"{timestamp}\n{DINGTALK_SECRET}"
        hmac_code = hmac.new(
            DINGTALK_SECRET.encode("utf-8"), string_to_sign.encode("utf-8"), digestmod=hashlib.sha256
        ).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        url = f"{DINGTALK_WEBHOOK}&timestamp={timestamp}&sign={sign}"

    payload = {"msgtype": "text", "text": {"content": f"{title}\n{content}"}}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code == 200 and resp.json().get("errcode") == 0
    except Exception as e:
        print(f"[push_dingtalk] 推送失败：{e}")
        return False


def push_feishu(title: str, content: str) -> bool:
    """飞书自定义机器人。FEISHU_SECRET留空时不加签校验（机器人安全设置允许不加签）"""
    title, content = desensitize(title), desensitize(content)
    if not FEISHU_WEBHOOK:
        print(f"[push_feishu] 未配置 FEISHU_WEBHOOK，跳过实际推送，仅打印：\n{title}\n{content}")
        return False

    payload = {"msg_type": "text", "content": {"text": f"{title}\n{content}"}}
    if FEISHU_SECRET:
        timestamp = str(int(time.time()))
        string_to_sign = f"{timestamp}\n{FEISHU_SECRET}"
        hmac_code = hmac.new(
            FEISHU_SECRET.encode("utf-8"), string_to_sign.encode("utf-8"), digestmod=hashlib.sha256
        ).digest()
        payload["timestamp"] = timestamp
        payload["sign"] = base64.b64encode(hmac_code).decode("utf-8")

    try:
        resp = requests.post(FEISHU_WEBHOOK, json=payload, timeout=10)
        return resp.status_code == 200 and resp.json().get("code", resp.json().get("StatusCode", 0)) == 0
    except Exception as e:
        print(f"[push_feishu] 推送失败：{e}")
        return False


def push_wecom(title: str, content: str) -> bool:
    """企业微信群机器人（Webhook地址里已经带了key，不需要额外加签）"""
    title, content = desensitize(title), desensitize(content)
    if not WECOM_WEBHOOK:
        print(f"[push_wecom] 未配置 WECOM_WEBHOOK，跳过实际推送，仅打印：\n{title}\n{content}")
        return False

    payload = {"msgtype": "text", "text": {"content": f"{title}\n{content}"}}
    try:
        resp = requests.post(WECOM_WEBHOOK, json=payload, timeout=10)
        return resp.status_code == 200 and resp.json().get("errcode") == 0
    except Exception as e:
        print(f"[push_wecom] 推送失败：{e}")
        return False


def push_slack(title: str, content: str) -> bool:
    """Slack Incoming Webhook"""
    title, content = desensitize(title), desensitize(content)
    if not SLACK_WEBHOOK:
        print(f"[push_slack] 未配置 SLACK_WEBHOOK，跳过实际推送，仅打印：\n{title}\n{content}")
        return False

    payload = {"text": f"*{title}*\n{content}"}
    try:
        resp = requests.post(SLACK_WEBHOOK, json=payload, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"[push_slack] 推送失败：{e}")
        return False


def push_email(title: str, content: str) -> bool:
    """SMTP邮件推送。EMAIL_TO支持多个收件人（逗号分隔）"""
    title, content = desensitize(title), desensitize(content)
    if not (EMAIL_SMTP_HOST and EMAIL_SMTP_USER and EMAIL_SMTP_PASSWORD and EMAIL_TO):
        print(f"[push_email] 未完整配置SMTP信息，跳过实际推送，仅打印：\n{title}\n{content}")
        return False

    msg = MIMEText(content, "plain", "utf-8")
    msg["Subject"] = Header(title, "utf-8")
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(EMAIL_TO)

    try:
        if EMAIL_USE_SSL:
            server = smtplib.SMTP_SSL(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT, timeout=10)
        else:
            server = smtplib.SMTP(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT, timeout=10)
            server.starttls()
        server.login(EMAIL_SMTP_USER, EMAIL_SMTP_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"[push_email] 推送失败：{e}")
        return False


# 渠道名 -> 推送函数，ALERT_PUSH_CHANNELS 按这个映射查找要调用哪些函数
PUSH_CHANNELS = {
    "wechat": push_wechat,
    "dingtalk": push_dingtalk,
    "feishu": push_feishu,
    "wecom": push_wecom,
    "slack": push_slack,
    "email": push_email,
}


def push_all(title: str, content: str, channels: list = None) -> dict:
    """
    按渠道列表逐个推送（未传则用 ALERT_PUSH_CHANNELS 环境变量配置的渠道），
    返回 {"wechat": True, "dingtalk": False, ...} 每个渠道各自的推送结果，
    方便上层（比如API/Streamlit）知道具体哪个渠道成功/失败，而不是只有一个笼统的bool。
    未知渠道名会被跳过并打印警告，不会让整批推送因为一个拼写错误的渠道名而中断。
    """
    channels = channels if channels is not None else ALERT_PUSH_CHANNELS
    results = {}
    for ch in channels:
        fn = PUSH_CHANNELS.get(ch)
        if fn is None:
            print(f"[push_all] 未知推送渠道 '{ch}'，可选：{list(PUSH_CHANNELS.keys())}，已跳过")
            continue
        results[ch] = fn(title, content)
    return results


def push_alert(company: str, event_type: str, reason: str, source_text: str) -> bool:
    """
    预警推送对外统一入口，graphs/pipeline_graph.py 的 push_node 调用这个。
    签名和返回值都保持不变（返回True代表至少一个渠道推送成功），
    内部实现从"只发微信"改成"按ALERT_PUSH_CHANNELS配置扇出到多个渠道"。
    """
    title = f"【舆情预警】{company} - {event_type}"
    content = f"**判断依据**：{reason}\n\n**原文片段**：{source_text[:200]}..."
    results = push_all(title, content)
    return any(results.values())
