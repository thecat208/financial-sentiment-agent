<h1 align="center">金融舆情分析系统</h1>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <a href="https://github.com/langchain-ai/langgraph"><img src="https://img.shields.io/badge/LangGraph-1.x-purple.svg" alt="LangGraph"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.1xx-339933.svg" alt="FastAPI"></a>
  <a href="https://www.sqlite.org/"><img src="https://img.shields.io/badge/SQLite-FTS5-orange.svg" alt="SQLite FTS5"></a>
  <a href="https://www.trychroma.com/"><img src="https://img.shields.io/badge/ChromaDB-1.x-red.svg" alt="ChromaDB"></a>
</p>

<p align="center"><em>LangGraph 编排 · RAG 混合检索 · 幻觉三层校验 · 预警 LLM 复核 · FastAPI 多租户 · Docker 一键部署</em></p>

## 项目概述

一个基于 LangGraph 编排的端到端金融舆情分析系统：从多源渠道（RSS 新闻、机构公告、社交媒体热度代理指标）采集舆情，经 LLM 完成四维度情感分析与标准化事件分类，以"规则初筛 + LLM 复核"两层架构实现低误报率预警，并提供 RAG 混合检索问答、每日简报自动生成、舆情-行情关联分析与回测、多公司横向对比与行业热度榜。

对外通过 FastAPI 暴露 REST 接口，支持多租户隔离、按租户用量统计与操作审计；Streamlit 提供可视化前端。

**定位**：个人开发的全栈工程项目，覆盖 数据采集 → 分析 → 存储 → 检索 → 对外服务 完整链路（已知边界见文末）。

## 核心特性

### 1. RAG 混合检索问答
- **向量 + 关键词双路检索**：ChromaDB（本地 bge 中文嵌入）+ SQLite FTS5，两路各司其职——语义覆盖靠向量，股票代码/专有名词等精确匹配靠关键词兜底
- **中文检索质量实测**：向量 Recall@5 = 0.975、MRR = 0.955；FTS5 经自定义分词修复后 Recall@5 = 0.914、P50 延迟 1.2ms（比向量快约 57 倍）
- **多轮对话记忆**：会话级短期记忆 + 长期摘要，支持连续追问

### 2. 幻觉工程化控制
- **三层程序化校验**（引用编号核查 → 数值/实体忠实性回查 → LLM 蕴含复核），校验失败自动降级放行，不阻断主流程
- **实测**：库外问题拒答率 100%、编造率 0%，库内回答忠实率 100%，引用标注覆盖率 100%（平均 2.9 个引用/回答），校验零误伤

### 3. 两层预警架构
- 规则初筛（关键词 + 情感阈值）→ LLM 复核二次判断 → 人工复核可覆盖并留审计日志
- 多渠道推送：微信 / 钉钉 / 飞书 / 企业微信 / Slack / 邮件，未配置渠道自动降级不中断

### 4. 分层成本优化
- 轻量前置筛选（词典规则 / 小型 transformer 双后端）：明确案例不调 LLM，实测降低 LLM 调用量 56.2%，路由决策正确率 100%
- 与真实 LLM 同源对比：LLM 情感分类准确率 97.0%（Financial PhraseBank，100 条），为"何时用小模型、何时升级 LLM"提供定量依据

### 5. 工程化底座
- **并发正确性**：SQLite 单共享连接 + RLock 串行化 + WAL，实测写入吞吐 3126 写/秒、读 P50 0.03ms；Chroma 多线程初始化双检锁保护
- **异步化**：LLM/API 全链路双模式（同步路径完整保留），多公司日报 `asyncio.gather` 并发；locust 压测 250 请求 5xx 归零，LLM 在途 21s 时健康检查 P50 = 3ms
- **SaaS 基础能力**：多租户数据隔离、按租户 LLM 用量统计（contextvars 自动记账）、PII 脱敏、限流、操作审计

## 架构与调用流向

单向依赖，无循环调用：

```
入口层    app/streamlit_app.py   api/routers/*.py
            │
编排层    graphs/*_graph.py      （每个文件对外一个函数：process / ask / generate_report / handle_query）
            │
业务层    chains/*.py  alert/*.py  analysis/*.py  export/*.py
            │
基础层    storage/repository.py  retrieval/*.py  market_data/  knowledge_graph/
            │
横切层    core/*.py  config/settings.py  storage/db.py
```

舆情处理主链路：

```
采集 ingestion/*_source.py
  → 清洗 processing/cleaner.py（去噪 + PII 脱敏）
  → 情感分析 chains/analysis_chain.py（门控前置筛选 → LLM 四维度打分 + 受控事件分类）
  → 预警 alert/rules.py（规则初筛）→ 命中则 chains/alert_review_chain.py（LLM 复核）
  → 入库 storage/repository.py → 推送 alert/push.py
```

## 项目结构

```
financial-sentiment-agent/
├── app/streamlit_app.py        # Streamlit 前端（8 个标签页）
├── api/                        # FastAPI 接口层：路由/鉴权/限流/Pydantic 模型，20+ 端点
├── graphs/                     # LangGraph 编排：pipeline / qa / report / router 四条业务流程
├── chains/                     # LLM 调用链：情感分析（含门控）/ RAG / 日报 / 意图识别 / 预警复核
│   └── prompts/                #   Prompt 模板 + 受控输出结构
├── ingestion/                  # 采集层：RSS / 机构公告 / 社交热度 / 多模态（OCR + 视频 ASR）
├── alert/                      # 预警规则初筛 + 多渠道推送
├── analysis/backtest.py        # 舆情-行情关联分析 + 预警信号回测
├── market_data/provider.py     # 行情数据（akshare，不可用自动降级 mock）
├── export/report_export.py     # 日报导出 Excel / PDF
├── knowledge_graph/            # 公司实体/行业种子库 + Neo4j/NetworkX 双后端图谱
├── retrieval/                  # Chroma 向量库 + FTS5 混合检索
├── storage/                    # SQLite 连接管理 / repository / 用量统计 / 审计日志
├── core/                       # 缓存 / 重试 / 限流 / 脱敏 / 受控词表 / Token 预算 / 用量记账
├── memory/session_store.py     # 多轮对话记忆
├── config/settings.py          # 全局配置中枢（所有环境变量统一入口）
├── scripts/                    # 评测 / 基准 / 验收测试 / locust 压测脚本
├── Dockerfile                  # 生产镜像（ffmpeg + tesseract 内置）
├── docker-compose.yml          # API + Streamlit 双服务编排
└── 项目测试指标报告.md           # 全部实测指标的测试方法与数据来源
```

## 快速开始

### 1. 安装依赖与配置

```bash
pip install -r requirements.txt
cp .env.example .env    # 填入 OPENAI_API_KEY 或 ANTHROPIC_API_KEY
```

**必需环境变量**：`OPENAI_API_KEY` 或 `ANTHROPIC_API_KEY`（二选一，`config/settings.py` 切换）。
其余配置均有默认值或自动降级逻辑，缺失不影响启动，只降级对应功能（见 `.env.example` 每项注释）。

### 2. 本地运行

```bash
# 跑一轮采集 + 分析
python scripts/run_pipeline.py

# 启动 Streamlit 前端
streamlit run app/streamlit_app.py

# 启动 FastAPI（另开终端）
uvicorn api.main:app --port 8000
# 接口文档：http://localhost:8000/docs
```

### 3. Docker 一键部署

```bash
docker compose up -d --build

docker compose ps        # api 与 streamlit 两个容器 Up
# API:  http://localhost:8000/docs
# 界面: http://localhost:8501
```

镜像基于 python:3.11-slim，内置 ffmpeg 与 tesseract（中英文语言包）；torch 单独走 CPU 源安装控制体积；
SQLite / Chroma / 日报通过 volume 落宿主机，容器可随意销毁重建。

## 测试与评测

```bash
# 自动化测试（不需要 LLM API Key）
python scripts/benchmark_full_system.py      # 全系统基准：PII脱敏/预警规则/检索/并发/缓存
python scripts/test_db_concurrency.py        # 数据库并发验收
python scripts/test_api.py                   # API 全端点测试

# 需要真实 LLM 的评测
python scripts/evaluate_phrasebank.py        # LLM 情感分类评测
python scripts/eval_retrieval.py             # 检索质量评测（40 条标注查询）
python scripts/eval_hallucination.py         # 幻觉控制评测（库内/库外各 15 题）

# 压测
locust -f scripts/locustfile.py --headless -u 10 -r 2 -t 1m
```

各项指标的完整数据、测试方法与数据来源见 [项目测试指标报告.md](./项目测试指标报告.md)。

## 技术栈

| 类别 | 选型 |
| --- | --- |
| 编排 | LangChain、LangGraph |
| LLM | Anthropic / OpenAI 兼容接口（配置切换，已实测 DeepSeek 端点） |
| 结构化存储 | SQLite（FTS5 + jieba 中文分词） |
| 向量存储 | ChromaDB + 本地 bge 中文嵌入（双检锁并发保护） |
| 前端/接口 | Streamlit、FastAPI + Uvicorn |
| 可选组件 | Redis（缓存）、Neo4j（图谱）、akshare（行情）——未安装自动降级 |
| 文档处理 | pdfplumber、openpyxl、reportlab、pytesseract（OCR）、faster-whisper（ASR） |
| 部署 | Docker Compose（API + Streamlit 双服务） |

## 扩展指南

- **新增公司/行业**：只改 `knowledge_graph/seed_data.py`——它是公司名/别名/股票代码/行业的唯一权威数据源，行情解析、采集定位、行业热度榜都从这里读取
- **新增事件类型/情感维度**：只改 `core/taxonomy.py` 受控词表，Prompt 约束与统计查询自动跟随
- **新增推送渠道**：在 `alert/push.py` 实现对应发送函数并注册，未配置凭证时自动降级为日志
- **切换 LLM 供应商**：改 `.env` 的 `OPENAI_BASE_URL` + `OPENAI_API_KEY` 即可，全部走 OpenAI 兼容接口
- **接入云端 ASR**：实现 `video_processor.py` 的 `_transcribe_audio_cloud()` 并在 `_transcribe_audio` 追加分发，上游零改动

## 已知边界

- 多租户隔离已实现，但无真实 SSO/登录鉴权（`tenant_id` 自报），操作审计同样不可抵赖
- SQLite/Chroma 为单机方案，数据量增长后需评估迁移 Postgres / 分布式向量库
- 行情数据未接真实数据源时展示确定性 mock 数据（已明确标注）
- LLM 输出质量依赖服务商稳定性，三层校验只能拦"违背资料"，不能提升生成质量本身
