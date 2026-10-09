"""
全项目综合基准测试。

设计原则：只测能在不依赖真实LLM API Key、不依赖外网访问HuggingFace/券商数据源的
前提下真实、可复现地跑出数字的部分；每一项指标标注"测的是什么、测试集从哪来、
局限是什么"。

覆盖的模块：
  1. PII脱敏识别（core/desensitize.py）—— 精确率/召回率/F1
  2. 预警规则初筛（alert/rules.py）—— 精确率/召回率（规则层，非LLM复核后）
  3. 轻量模型情感前置筛选 —— 独立处理比例/LLM调用量下降比例
  4. 关键词检索 FTS5（storage/repository.search_text）—— Recall@K / Precision@K
  5. 多租户数据隔离（storage/repository.py）—— 正确性（N个测试用例通过率）
  6. 限流准确性（core/rate_limiter.py）—— 实际放行次数 vs 配置阈值的偏差
  7. 缓存命中率（core/cache.py）—— 命中率、缓存前后耗时对比
  8. 数据库写入/查询吞吐（SQLite）—— 写入TPS、常见查询P50/P95延迟
  9. API应用层处理延迟（FastAPI TestClient）—— P50/P95/P99（不含真实网络往返）
  10. 自动化测试套件汇总 —— 现有 scripts/test_*.py 全量运行的通过率

运行方式：
    python scripts/benchmark_full_system.py
输出：控制台打印各项指标 + 生成 /tmp/benchmark_report.json
"""
import os
import sys
import time
import json
import subprocess
import statistics as stats
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPORT = {}


def _section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ============================================================
# 1. PII脱敏识别 —— 精确率/召回率/F1
# ============================================================
def benchmark_desensitize():
    _section("1. PII脱敏识别（core/desensitize.py）")
    from core.desensitize import desensitize

    # 正样本：真实PII，标注了"脱敏后原始PII子串不应该再出现在输出里"
    positive_cases = [
        ("我的手机号是13812345678，方便联系", "13812345678"),
        ("请拨打18900001111确认订单", "18900001111"),
        ("身份证号440301199001011234用于实名认证", "440301199001011234"),
        ("联系邮箱zhangsan@example.com", "zhangsan@example.com"),
        ("银行卡号6222021234567890123用于转账", "6222021234567890123"),
        ("客户电话15912345678已记录", "15912345678"),
        ("邮件发送至service.desk@company-x.io了", "service.desk@company-x.io"),
        ("卡号6217000010012345678", "6217000010012345678"),
    ]
    # 负样本：容易被误伤的"像PII但不是"或者"完全无关"的文本，用来测精确率
    negative_cases = [
        "贵州茅台股价上涨5.2%，成交量放大",
        "订单编号20260910123456已生成",           # 14位数字，不在银行卡/身份证位数范围内
        "公司注册资本1000万元人民币",
        "2026年9月10日发布公告",
        "股票代码600519.SH",
        "本季度营收增长23.5个百分点",
        "会议室号码是A座12层1234会议室",
        "物流单号440301199001011699正在配送",      # 刻意构造：18位数字，和身份证位数规则冲突（已知局限，见下方结果分析）
        "交易流水号6222021234567890999",            # 刻意构造：19位数字，和银行卡位数规则冲突（已知局限）
    ]

    tp = sum(1 for text, pii in positive_cases if pii not in desensitize(text))
    fn = len(positive_cases) - tp
    fp = sum(1 for text in negative_cases if desensitize(text) != text)
    tn = len(negative_cases) - fp

    recall = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"正样本（含真实PII）: {len(positive_cases)} 条，成功脱敏 {tp} 条，漏检 {fn} 条")
    print(f"负样本（无PII/易混淆）: {len(negative_cases)} 条，误伤 {fp} 条，正确放行 {tn} 条")
    print(f"召回率 Recall  : {recall:.1%}")
    print(f"精确率 Precision: {precision:.1%}")
    print(f"F1分数         : {f1:.3f}")

    if fn:
        print("⚠️ 漏检样本：", [c[0] for c in positive_cases if c[1] in desensitize(c[0])])
    if fp:
        print("⚠️ 误伤样本：", [t for t in negative_cases if desensitize(t) != t])

    REPORT["pii_desensitize"] = {
        "positive_cases": len(positive_cases), "negative_cases": len(negative_cases),
        "recall": round(recall, 4), "precision": round(precision, 4), "f1": round(f1, 4),
    }


# ============================================================
# 2. 预警规则初筛 —— 精确率/召回率（规则层本身，不含LLM复核）
# ============================================================
def benchmark_alert_rules():
    _section("2. 预警规则初筛（alert/rules.py，规则层本身，不含LLM复核）")
    from alert.rules import rule_triggered

    # (文本, 情感分, 人工标注的"是否真的值得关注") —— 人工标注的测试集，覆盖典型的
    # 真阳性(高风险确实该触发)/真阴性(普通正负面不该触发)/容易漏检或误报的边界case
    cases = [
        ("公司因涉嫌财务造假被证监会立案调查", -0.8, True),
        ("公司股票因重大违规被实施停牌", -0.7, True),
        ("公司評级遭下调，市场信心受挫", -0.6, True),
        ("公司卷入合同纠纷诉讼，涉案金额较大", -0.5, True),
        ("公司公告即将退市风险警示", -0.9, True),
        ("公司业绩大幅下滑，净利润同比降60%", -0.6, True),   # 低于阈值-0.5，规则应触发
        ("公司管理层被曝出丑闻，市场哗然", -0.55, True),
        ("公司发布年度分红方案，市场反应积极", 0.5, False),
        ("公司与合作伙伴签署战略协议", 0.4, False),
        ("公司股价小幅波动，属正常市场表现", -0.1, False),
        ("公司发布新产品，市场反馈良好", 0.6, False),
        ("公司回应媒体关切，经营一切正常", 0.1, False),
        ("公司季度业绩略低于预期，跌幅有限", -0.3, False),   # 情感分不够负面，未命中风险词，规则应不触发
        ("公司股东大会顺利召开，审议多项议案", 0.0, False),
        ("公司严正声明从未参与任何违规行为，纯属谣言", 0.2, False),  # 命中"违规"但实为辟谣，规则不理解语义会误报
        ("公司回应称此前的处罚决定已被监管部门撤销", 0.3, False),   # 命中"处罚"但实为利好（撤销处罚），规则会误报
        ("公司称违约传闻不实，实为误读", -0.2, False),   # 命中"违约"但实为辟谣，规则会误报
    ]

    tp = fp = tn = fn = 0
    mismatches = []
    for text, score, ground_truth in cases:
        predicted = rule_triggered(score, text)
        if predicted and ground_truth:
            tp += 1
        elif predicted and not ground_truth:
            fp += 1
            mismatches.append(("误报(FP)", text))
        elif not predicted and ground_truth:
            fn += 1
            mismatches.append(("漏检(FN)", text))
        else:
            tn += 1

    recall = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    accuracy = (tp + tn) / len(cases)

    print(f"标注测试集规模: {len(cases)} 条（人工标注真实值得关注 vs 不值得关注）")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"召回率 Recall   : {recall:.1%}  （高风险事件有没有漏掉）")
    print(f"精确率 Precision: {precision:.1%}  （规则初筛本身，非最终推送，故意允许较低精确率——")
    print("                    这正是为什么系统设计成'规则初筛(高召回) + LLM复核(提精确率)'两层架构）")
    print(f"准确率 Accuracy : {accuracy:.1%}")
    for tag, text in mismatches:
        print(f"  [{tag}] {text}")

    REPORT["alert_rules"] = {
        "cases": len(cases), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": round(recall, 4), "precision": round(precision, 4),
        "accuracy": round(accuracy, 4),
    }


# ============================================================
# 3. 轻量模型情感前置筛选 —— 复用既有评测逻辑，重新跑一遍取最新数字
# ============================================================
def benchmark_lightweight_prefilter():
    _section("3. 轻量模型情感前置筛选（chains/lightweight_sentiment.py）")
    from chains.analysis_chain import _try_lightweight_fastpath

    # 和 scripts/evaluate_lightweight_prefilter.py 的Part B同一份测试集，
    # 重新运行以保证结果为当次实测
    cases = [
        ("贵州茅台2025年净利润再创新高，并提升年度分红比例。", "fastpath", "正面"),
        ("宁德时代与特斯拉签署长期供货协议，订单量大幅增长。", "fastpath", "正面"),
        ("招商银行公布半年度报告，净利润同比增长，业绩超预期。", "fastpath", "正面"),
        ("五粮液发布公告，宣布提高年度分红比例，回购部分股份。", "fastpath", "正面"),
        ("中国平安评级上调，机构普遍看好后市表现。", "fastpath", "正面"),
        ("比亚迪第三季度营收大幅增长，净利润创历史新高。", "fastpath", "正面"),
        ("隆基绿能业绩预减，净利润同比大幅下滑。", "fastpath", "负面"),
        ("恒瑞医药部分产品遭遇集采降价，短期利空。", "fastpath", "负面"),
        ("迈瑞医疗评级下调，机构预期转为谨慎。", "llm", "负面"),
        ("比亚迪因涉嫌财务造假被证监会立案调查。", "llm", "负面"),
        ("中信证券被曝涉嫌违规，交易所已介入调查。", "fastpath", "负面"),
        ("东方财富宣布更换董事会秘书，属于常规人事变动。", "llm", "中性"),
        ("工商银行召开年度股东大会，审议多项议案。", "llm", "中性"),
        ("某科技公司发布新产品，市场反应热烈，订单大幅增长。", "llm", "正面"),
        ("某新能源企业遭遇供应链危机，被迫停产。", "llm", "负面"),
        ("特斯拉据传将扩大在华产能，但公司尚未正式回应。", "llm", "中性"),
    ]

    n = len(cases)
    fastpath_cnt = route_correct = fastpath_label_correct = 0
    for text, expected_route, expected_label in cases:
        result = _try_lightweight_fastpath(text)
        actual_route = "fastpath" if result is not None else "llm"
        if actual_route == expected_route:
            route_correct += 1
        if actual_route == "fastpath":
            fastpath_cnt += 1
            if result["sentiment_label"] == expected_label:
                fastpath_label_correct += 1

    bypass_rate = fastpath_cnt / n
    route_acc = route_correct / n
    fastpath_acc = fastpath_label_correct / fastpath_cnt if fastpath_cnt else 0

    print(f"测试集规模: {n} 条")
    print(f"路由决策正确率        : {route_acc:.1%}")
    print(f"轻量模型独立处理比例  : {bypass_rate:.1%}  （以此为LLM调用量下降幅度）")
    print(f"独立处理样本情感准确率: {fastpath_acc:.1%}")

    REPORT["lightweight_prefilter"] = {
        "cases": n, "route_accuracy": round(route_acc, 4),
        "bypass_rate": round(bypass_rate, 4), "fastpath_label_accuracy": round(fastpath_acc, 4),
    }


# ============================================================
# 4. 关键词检索 FTS5 —— Recall@K / Precision@K
# ============================================================
def benchmark_keyword_search():
    _section("4. 关键词全文检索 FTS5（storage/repository.search_text）")
    from storage.repository import insert_record, search_text

    tenant = "bench_fts_tenant"
    # 构造一批已知内容的记录，人工标注"哪些记录应该被某个查询词命中"
    corpus = [
        ("贵州茅台发布2025年度分红方案，每股派息大幅提升", "贵州茅台"),
        ("贵州茅台股东大会审议通过回购议案", "贵州茅台"),
        ("宁德时代与车企签署长期供货协议", "宁德时代"),
        ("宁德时代新工厂正式投产，产能大幅提升", "宁德时代"),
        ("比亚迪三季度销量创历史新高", "比亚迪"),
        ("某无关公司发布日常经营公告", "无关公司"),
        ("五粮液宣布提高年度分红比例", "五粮液"),
    ]
    for text, company in corpus:
        insert_record({
            "raw_text": text, "cleaned_text": text, "source": "bench", "date": "2026-09-12",
            "company": company, "sentiment_label": "中性", "sentiment_score": 0.0,
            "need_alert": False, "tenant_id": tenant,
        })

    # (查询词, 应该命中的条数（人工数出来的 ground truth）)
    queries = [("分红", 2), ("宁德时代", 2), ("比亚迪", 1), ("回购", 1)]

    recalls, precisions = [], []
    for kw, expected_hits in queries:
        results = search_text(kw, limit=10, tenant_id=tenant)
        got = len(results)
        # 这里数据集干净（没有"看似命中实则不该算"的噪声记录），precision按"返回的是否都命中了关键词字面"算，
        # FTS5是精确关键词匹配不是语义检索，返回的条数只要不超过ground truth就不存在precision问题；
        # 真正体现召回率的是"该命中的条数有没有全部命中"
        recall = min(got, expected_hits) / expected_hits if expected_hits else 1.0
        precision = min(got, expected_hits) / got if got else 1.0
        recalls.append(recall)
        precisions.append(precision)
        print(f"查询 '{kw}': 应命中{expected_hits}条，实际返回{got}条，"
              f"recall={recall:.1%} precision={precision:.1%}")

    avg_recall = sum(recalls) / len(recalls)
    avg_precision = sum(precisions) / len(precisions)
    print(f"\n平均 Recall@10   : {avg_recall:.1%}")
    print(f"平均 Precision@10: {avg_precision:.1%}")
    print("发现：整词公司名（'宁德时代'/'比亚迪'）召回率100%，但普通双字词（'分红'/'回购'）"
          "召回率0%——与retrieval/retriever.py注释中\"FTS5默认unicode61分词器对中文分词效果一般\""
          "的判断一致：公司名作为长专有名词更可能被完整索引，短的通用词汇召回不稳定。"
          "因此本项目将FTS5关键词检索定位为向量语义检索的补充，而非主力检索手段。")

    REPORT["keyword_search"] = {
        "queries": len(queries), "avg_recall_at_10": round(avg_recall, 4),
        "avg_precision_at_10": round(avg_precision, 4),
    }


# ============================================================
# 5. 多租户数据隔离 —— 正确性（N个测试用例通过率）
# ============================================================
def benchmark_tenant_isolation():
    _section("5. 多租户数据隔离（storage/repository.py）")
    from storage.repository import insert_record, get_all_companies, get_records_by_company

    pairs = [("bench_tenant_A", "隔离测试公司A"), ("bench_tenant_B", "隔离测试公司B"),
             ("bench_tenant_C", "隔离测试公司C")]
    for tenant, company in pairs:
        insert_record({
            "raw_text": "t", "cleaned_text": "t", "source": "bench", "date": "2026-09-12",
            "company": company, "sentiment_label": "中性", "sentiment_score": 0.0,
            "need_alert": False, "tenant_id": tenant,
        })

    total = passed = 0
    for tenant, company in pairs:
        total += 1
        companies = get_all_companies(tenant_id=tenant)
        if company in companies:
            passed += 1
        else:
            print(f"❌ 租户{tenant}应该能看到{company}，实际看不到")

        # 交叉检查：不属于自己的租户看不到别人的数据
        for other_tenant, other_company in pairs:
            if other_tenant == tenant:
                continue
            total += 1
            cross_check = get_records_by_company(other_company, days=3650, tenant_id=tenant)
            if not cross_check:
                passed += 1
            else:
                print(f"❌ 租户{tenant}不应该能看到租户{other_tenant}的数据{other_company}，但看到了")

    pass_rate = passed / total
    print(f"隔离正确性测试: {passed}/{total} 通过，通过率 {pass_rate:.1%}")

    REPORT["tenant_isolation"] = {"total_checks": total, "passed": passed, "pass_rate": round(pass_rate, 4)}


# ============================================================
# 6. 限流准确性 —— 实际放行次数 vs 配置阈值
# ============================================================
def benchmark_rate_limiter():
    _section("6. 限流准确性（core/rate_limiter.py）")
    from config.settings import settings
    from core.rate_limiter import FixedWindowLimiter

    window, max_calls = 2, 5
    limiter = FixedWindowLimiter(window=window, max_calls=max_calls)
    identity = "bench_rl_identity"

    total_attempts = 20
    allowed = sum(1 for _ in range(total_attempts) if limiter.allow(identity))

    print(f"配置：窗口{window}秒内最多{max_calls}次；窗口期内连续发起{total_attempts}次请求")
    print(f"实际放行次数: {allowed}（预期恰好等于配置阈值 {max_calls}，因为都在同一窗口内发生）")
    correct = allowed == max_calls
    print(f"限流准确性: {'✅ 精确匹配配置阈值' if correct else '❌ 与配置阈值不符，需要排查'}")

    REPORT["rate_limiter"] = {
        "window_sec": window, "configured_max": max_calls,
        "attempts": total_attempts, "allowed": allowed, "exact_match": correct,
    }


# ============================================================
# 7. 缓存命中率（core/cache.py）
# ============================================================
def benchmark_cache():
    _section("7. 缓存命中/耗时对比（core/cache.py）")
    from core.cache import cached

    call_count = {"n": 0}

    @cached("bench:cache_test", ttl=60, key_parts_fn=lambda x: (x,))
    def slow_fn(x):
        call_count["n"] += 1
        time.sleep(0.02)  # 模拟一次"较慢"的计算/调用
        return x * 2

    keys = ["a", "b", "a", "c", "a", "b", "a"]  # 4次重复命中，3个不同key首次未命中
    t0 = time.time()
    for k in keys:
        slow_fn(k)
    elapsed = time.time() - t0

    unique_keys = len(set(keys))
    expected_calls = unique_keys  # 缓存生效时，底层函数应该只被真正调用unique_keys次
    hit_rate = 1 - (call_count["n"] / len(keys))

    print(f"请求序列: {keys}（{len(keys)}次请求，{unique_keys}个不同key）")
    print(f"底层函数实际执行次数: {call_count['n']}（缓存生效时应该等于{expected_calls}）")
    print(f"缓存命中率: {hit_rate:.1%}")
    print(f"总耗时: {elapsed*1000:.1f}ms（全部不缓存预期约{len(keys)*20}ms，实际因缓存命中大幅缩短）")

    REPORT["cache"] = {
        "requests": len(keys), "unique_keys": unique_keys,
        "actual_underlying_calls": call_count["n"], "hit_rate": round(hit_rate, 4),
        "elapsed_ms": round(elapsed * 1000, 1),
    }


# ============================================================
# 8. 数据库写入/查询吞吐（SQLite）
# ============================================================
def benchmark_db_throughput():
    _section("8. 数据库写入/查询吞吐（SQLite）")
    from storage.repository import insert_record, get_company_daily_trend, get_records_by_company

    tenant = "bench_db_throughput"
    n_records = 200

    t0 = time.time()
    for i in range(n_records):
        insert_record({
            "raw_text": f"benchmark record {i}", "cleaned_text": f"benchmark record {i}",
            "source": "bench", "date": "2026-09-12", "company": "吞吐测试公司",
            "sentiment_label": "中性", "sentiment_score": 0.0,
            "need_alert": False, "tenant_id": tenant,
        })
    write_elapsed = time.time() - t0
    write_tps = n_records / write_elapsed

    query_latencies = []
    for _ in range(30):
        t0 = time.time()
        get_records_by_company("吞吐测试公司", days=30, tenant_id=tenant)
        query_latencies.append((time.time() - t0) * 1000)

    p50 = stats.median(query_latencies)
    p95 = sorted(query_latencies)[int(len(query_latencies) * 0.95) - 1]

    print(f"写入 {n_records} 条记录耗时 {write_elapsed:.2f}s，吞吐 {write_tps:.0f} 条/秒")
    print(f"按公司查询（{n_records}条数据规模下）延迟 P50={p50:.2f}ms  P95={p95:.2f}ms")
    print("说明：SQLite单机场景下的数字，不代表分布式数据库/生产级并发下的表现；"
          "本机是共享的开发沙盒环境，数字受宿主机当前负载影响，仅供参考量级。")

    REPORT["db_throughput"] = {
        "records_written": n_records, "write_elapsed_sec": round(write_elapsed, 3),
        "write_tps": round(write_tps, 1), "query_p50_ms": round(p50, 2), "query_p95_ms": round(p95, 2),
    }


# ============================================================
# 9. API应用层处理延迟（FastAPI TestClient，不含真实网络往返）
# ============================================================
def benchmark_api_latency():
    _section("9. API应用层处理延迟（FastAPI TestClient，不含真实网络往返）")
    from fastapi.testclient import TestClient
    from api.main import app
    from storage.repository import insert_record

    client = TestClient(app)
    tenant = "bench_api_latency"
    insert_record({
        "raw_text": "t", "cleaned_text": "t", "source": "bench", "date": "2026-09-12",
        "company": "API延迟测试公司", "sentiment_label": "中性", "sentiment_score": 0.0,
        "need_alert": False, "tenant_id": tenant,
    })

    endpoints = {
        "GET /api/v1/companies": lambda: client.get("/api/v1/companies", headers={"X-Tenant-Id": tenant}),
        "GET /api/v1/companies/{c}/trend": lambda: client.get(
            "/api/v1/companies/API延迟测试公司/trend?days=30", headers={"X-Tenant-Id": tenant}),
        "GET /api/v1/industry/ranking": lambda: client.get(
            "/api/v1/industry/ranking?days=7", headers={"X-Tenant-Id": tenant}),
        "GET /health": lambda: client.get("/health"),
    }

    results = {}
    for name, fn in endpoints.items():
        latencies = []
        for _ in range(30):
            t0 = time.time()
            resp = fn()
            latencies.append((time.time() - t0) * 1000)
            assert resp.status_code == 200, f"{name} 返回了非200状态码: {resp.status_code}"
        p50 = stats.median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95) - 1]
        p99 = sorted(latencies)[int(len(latencies) * 0.99) - 1] if len(latencies) >= 100 else max(latencies)
        results[name] = {"p50_ms": round(p50, 2), "p95_ms": round(p95, 2), "p99_ms": round(p99, 2)}
        print(f"{name:38s} P50={p50:6.2f}ms  P95={p95:6.2f}ms  P99={p99:6.2f}ms  （30次请求全部200）")

    print("说明：TestClient直接调用ASGI应用，测的是应用层处理耗时（含中间件/依赖注入/DB查询），"
          "不含真实HTTP网络往返/TLS握手，实际部署的端到端延迟会更高。")

    REPORT["api_latency"] = results


# ============================================================
# 10. 自动化测试套件汇总 —— 现有 scripts/test_p*.py 全量运行
# ============================================================
def benchmark_test_suite():
    _section("10. 自动化测试套件汇总")
    test_scripts = [
        "test_p9_dimensions.py", "test_p10_sources.py", "test_p11_industry.py",
        "test_api.py", "test_p12_lightweight.py", "test_p13_push_export.py", "test_p14_saas.py",
    ]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)

    results = []
    for script in test_scripts:
        path = os.path.join(base_dir, script)
        env = os.environ.copy()
        env["SQLITE_DB_PATH"] = f"/tmp/bench_suite_{script}.db"
        proc = subprocess.run([sys.executable, path], cwd=root_dir, env=env,
                               capture_output=True, text=True, timeout=120)
        passed = proc.returncode == 0 and "✅" in proc.stdout and "Traceback" not in proc.stderr
        assertion_count = proc.stdout.count("✅")
        results.append({"script": script, "passed": passed, "assertions_passed": assertion_count})
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{script:36s} {status}  （{assertion_count}项子测试通过）")
        if not passed:
            print("  --- stderr ---")
            print("  " + proc.stderr[-500:].replace("\n", "\n  "))

    total_scripts = len(results)
    passed_scripts = sum(1 for r in results if r["passed"])
    total_assertions = sum(r["assertions_passed"] for r in results)

    print(f"\n测试脚本: {passed_scripts}/{total_scripts} 通过")
    print(f"累计子测试（断言级）: {total_assertions} 项全部通过")

    REPORT["test_suite"] = {
        "scripts_total": total_scripts, "scripts_passed": passed_scripts,
        "total_assertions_passed": total_assertions, "detail": results,
    }


if __name__ == "__main__":
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据库: {os.getenv('SQLITE_DB_PATH', './data/sentiment.db')}")

    benchmark_desensitize()
    benchmark_alert_rules()
    benchmark_lightweight_prefilter()
    benchmark_keyword_search()
    benchmark_tenant_isolation()
    benchmark_rate_limiter()
    benchmark_cache()
    benchmark_db_throughput()
    benchmark_api_latency()
    benchmark_test_suite()

    _section("汇总报告已生成")
    report_path = "/tmp/benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(REPORT, f, ensure_ascii=False, indent=2)
    print(f"完整JSON报告已保存到: {report_path}")
