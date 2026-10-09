"""
安全与合规演示。运行方式：python scripts/run_compliance_demo.py

展示四块能力：
1. 敏感词过滤：输入拦截判定 + 正文打码
2. 日志脱敏：手机号/身份证/银行卡/邮箱打码
3. 免责声明：回答/日报末尾强制拼接
4. 带脱敏的 logger：日志自动打码
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.compliance import contains_sensitive, mask_sensitive, append_disclaimer
from core.desensitize import desensitize
from core.logger import get_logger

if __name__ == "__main__":
    # 1) 敏感词过滤
    print("== 1. 敏感词过滤 ==")
    bad_text = "提供线上赌场代收服务，欢迎联系"
    print("含敏感词？", contains_sensitive(bad_text), "（应True）")
    print("打码后：", mask_sensitive(bad_text))
    normal = "宁德时代发布三季度财报，营收同比增长15%。"
    print("正常文本含敏感词？", contains_sensitive(normal), "（应False）")
    print("正常文本打码后不变：", mask_sensitive(normal))

    # 2) 日志脱敏
    print("\n== 2. 日志脱敏 ==")
    raw = "联系我 13812345678，身份证 110101199003071234，银行卡 6222020200098765432，邮箱 zhangsan@example.com"
    print("原文：", raw)
    print("脱敏：", desensitize(raw))

    # 3) 免责声明
    print("\n== 3. 免责声明 ==")
    ans = append_disclaimer("根据公开资料，示例公司近期舆情整体偏正面。")
    print(ans)
    print("\n幂等（重复拼接不叠加）：", append_disclaimer(ans).endswith(append_disclaimer(ans)))

    # 4) 带脱敏的 logger
    print("\n== 4. 带脱敏的logger（自动打码）==")
    logger = get_logger("demo")
    logger.info("用户 13912345678 提交了查询")
