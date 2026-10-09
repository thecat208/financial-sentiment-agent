# 金融舆情分析系统 —— 项目说明（交接版）

> 本文档为整理后的项目说明，供后续开发者/AI接手时快速理解项目结构与现状，  
> 已删减原README中偏历史流水账的开发日志内容，保留结构性信息。  
> 反映的是"并发/向量检索/幻觉控制"三项问题排查工作**开始之前**的项目状态；  
> 三项问题现状（详见第六节6.1各条目的最新说明）：  
> ①并发问题：storage/db.py已解决（单连接+锁+WAL，8/8验收通过），LLM/API异步化已全部完成（QA/路由/管线/日报双模式+LLM路由async化，真实验证通过）；  
> ②向量检索验证：已解决（2026-10-05，Recall@5=0.975）；  
> ③幻觉工程化控制：已解决（2026-10-05评测，库外拒答率100%）。

---

## 一、项目简介

一个基于 LangGraph 编排的端到端金融舆情分析系统。从多源渠道（RSS新闻、机构公告、  
社交媒体热度代理指标）采集舆情数据，经LLM完成情感分析（含四维度精细化打分）与标准化  
事件分类，结合"规则初筛+LLM复核"两层架构实现低误报率预警，支持多渠道推送  
（微信/钉钉/飞书/企业微信/Slack/邮件）。同时提供RAG混合检索问答、每日简报自动生成  
（可导出Excel/PDF）、舆情-行情关联分析与历史回测、多公司横向对比与行业热度榜。  
对外通过FastAPI暴露REST接口，支持多租户隔离、按租户用量统计与操作审计。

**定位**：个人开发的全栈工程项目，覆盖数据采集→分析→存储→检索→对外服务的完整链路，  
非商业化生产系统（详见第六节已知局限）。

---

## 二、技术栈

| 类别     | 技术选型                                                            |
| ------ | --------------------------------------------------------------- |
| 编排框架   | LangChain、LangGraph                                             |
| LLM    | Claude / GPT（Anthropic / OpenAI API，二选一，`config/settings.py`切换） |
| 结构化存储  | SQLite（含FTS5全文检索虚拟表）                                            |
| 向量存储   | ChromaDB                                                        |
| 缓存     | Redis（可选，未安装自动降级为进程内内存缓存）                                       |
| 知识图谱   | Neo4j（可选，未安装自动降级为NetworkX内存图）                                   |
| Web/接口 | FastAPI + Uvicorn（对外API）、Streamlit（可视化前端）                       |
| 定时任务   | APScheduler                                                     |
| 行情数据   | akshare（可选，未安装/取不到数据自动降级为确定性mock数据）                             |
| 文档处理   | pdfplumber（PDF解析）、openpyxl（Excel）、reportlab（PDF生成）              |
| 其他     | pandas、requests、pytesseract（图片OCR）                              |

---


## 三、项目结构树（带中文注释）

```
financial-sentiment-agent/
├── app/
│   └── streamlit_app.py           # Streamlit前端主入口，8个标签页：智能入口/RAG问答/
│                                   #   历史趋势/日报/预警看板/行情联动/多公司对比/用量审计
│
├── api/                            # FastAPI对外REST接口层，不含业务逻辑，只是HTTP外壳
│   ├── main.py                     #   应用入口，挂载路由+CORS+用量统计中间件
│   ├── deps.py                     #   公共依赖：租户识别/用户识别/API Key鉴权/限流
│   ├── schemas.py                  #   请求/响应的Pydantic模型
│   └── routers/                    #   按功能拆分的路由，每个文件对应一组端点
│       ├── query.py                #     智能问答 /api/v1/query /api/v1/qa
│       ├── ingest.py               #     舆情提交 /api/v1/ingest
│       ├── companies.py            #     公司趋势/对比/日报 /api/v1/companies* /reports*
│       ├── alerts.py               #     预警统计 /api/v1/alerts*
│       ├── market.py               #     行情联动/回测 /api/v1/market*
│       ├── industry.py             #     行业热度榜 /api/v1/industry*
│       └── usage.py                #     用量统计/审计/预警人工复核 /api/v1/usage* /audit-log
│
├── graphs/                         # LangGraph编排层：每个xxx_graph.py是一条完整业务流程，
│                                   #   对外只暴露一个函数（process/ask/generate_report/handle_query）
│   ├── state_schemas.py            #   各流程共用的State类型定义（TypedDict）
│   ├── pipeline_graph.py           #   核心处理流水线：清洗→分析→预警→入库→推送
│   ├── qa_graph.py                 #   RAG问答流程：检索→生成→记忆
│   ├── report_graph.py             #   日报生成流程：按公司聚合→LLM小结→落盘→推送
│   └── router_graph.py             #   智能入口：先做意图识别，再路由到qa/report/alert
│
├── chains/                         # LLM调用链，被graphs/层调用，不直接暴露给外部
│   ├── qa_chain.py                 #   get_llm()：全局LLM客户端单例（Anthropic/OpenAI二选一）
│   ├── analysis_chain.py           #   情感分析主链路；analyze_text_gated()是生产入口，
│   │                                #     内部先过lightweight_sentiment.py的前置筛选门控，
│   │                                #     不满足条件才真正调用LLM（analyze_text()）
│   ├── lightweight_sentiment.py    #   轻量情感分类：词典规则/小型transformer模型双后端，
│   │                                #     供analysis_chain.py的前置筛选调用，不直接对外
│   ├── report_chain.py             #   单公司舆情小结生成（被report_graph.py调用）
│   ├── intent_chain.py             #   意图识别（被router_graph.py调用）
│   ├── alert_review_chain.py       #   预警复核：LLM二次判断规则初筛是否为真实风险
│   └── prompts/                    #   纯Prompt模板，不含调用逻辑
│       ├── analysis_prompt.py      #     情感分析Prompt + 受控输出结构(Pydantic)
│       └── qa_prompt.py            #     RAG问答Prompt，含"不编造/资料不足要如实说明"的约束
│
├── ingestion/                      # 数据采集层，各来源模块产出统一格式后交给graphs/pipeline_graph
│   ├── rss_source.py               #   RSS新闻采集
│   ├── announcement_source.py      #   机构公告采集（akshare/巨潮资讯网，不可用时降级mock）
│   ├── social_source.py            #   社交热度代理指标（东方财富千股千评，非原始帖子）
│   ├── async_jobs.py               #   任务队列封装：submit_ingest_item()异步提交处理任务
│   ├── scheduler.py                #   APScheduler定时任务注册（定时采集+定时日报）
│   └── multimodal/                 #   图片/视频转文字，统一交给pipeline_graph处理
│       ├── normalizer.py           #     统一入口process_input()，识别media_type后分发
│       ├── image_processor.py      #     OCR（pytesseract）
│       └── video_processor.py      #     ASR转写（faster-whisper本地推理，云端API预留接入点；
│
├── alert/
│   ├── rules.py                    #   规则初筛：关键词命中+情感分阈值判断，供pipeline_graph调用
│   └── push.py                     #   多渠道推送：微信/钉钉/飞书/企业微信/Slack/邮件，
│                                   #     未配置的渠道自动降级为打印日志，不报错中断流程
│
├── analysis/
│   └── backtest.py                 #   舆情-行情关联分析+预警信号回测，调用market_data获取行情，
│                                   #     调用storage/repository获取舆情数据，两者对齐后统计
│
├── market_data/
│   └── provider.py                 #   行情数据获取：akshare优先，不可用时降级为确定性mock数据；
│                                   #     公司名→股票代码解析复用knowledge_graph/seed_data.py
│
├── export/
│   └── report_export.py            #   日报导出Excel/PDF，读取report_graph落盘的JSON侧车文件，
│                                   #     不重新调用LLM
│
├── knowledge_graph/                # 公司实体、行业归属、图谱构建
│   ├── seed_data.py                #   种子库：公司名/别名/股票代码/所属行业的权威映射表，
│   │                                #     被market_data/announcement_source/social_source/
│   │                                #     lightweight_sentiment等多处复用，是全项目公司信息的
│   │                                #     唯一数据源，改公司信息只需要改这一个文件
│   ├── data_model.py               #   图节点/边的数据结构定义
│   ├── builder.py                  #   从舆情记录构建"公司-事件类型"等关系图
│   ├── graph_manager.py            #   图存储的统一读写入口，屏蔽底层用Neo4j还是NetworkX
│   ├── networkx_store.py           #   NetworkX内存图实现（Neo4j不可用时的降级后端）
│   └── neo4j_store.py              #   Neo4j图数据库实现（可选，企业部署用）
│
├── retrieval/                      # RAG检索层，被qa_graph.py调用
│   ├── vectorstore.py              #   Chroma向量库读写封装，含Embedding模型加载
│   └── retriever.py                #   混合检索：向量检索+SQLite FTS5关键词检索结果合并
│
├── storage/                        # 数据持久化层，是全项目唯一直接操作数据库的地方
│   ├── db.py                       #   SQLite连接管理、建表、schema迁移（历次功能新增的字段
│   │                                #     都在这里通过_MIGRATIONS列表做兼容旧数据的ALTER TABLE；
│   │                                #     并发补丁：进程内单共享连接+锁串行化+WAL+busy_timeout，
│   │                                #     详见文件内docstring与scripts/test_db_concurrency.py）
│   ├── repository.py               #   所有数据库读写函数的集合（insert_record/get_xxx等），
│   │                                #     graphs/chains/api/app各层都只通过这里访问数据库，
│   │                                #     不直接写SQL
│   ├── usage_tracker.py            #   LLM用量/API调用次数统计的读写
│   └── audit_log.py                #   操作审计日志的读写（如预警人工复核记录）
│
├── core/                           # 工程基础设施，各层都可能依赖，相互之间基本不依赖
│   ├── cache.py                    #   统一缓存装饰器@cached，Redis/内存双后端
│   ├── retry.py                    #   统一重试装饰器@with_retry
│   ├── rate_limiter.py             #   限流器：全局令牌桶+单租户固定窗口
│   ├── task_queue.py               #   简易任务队列（线程池实现，进程内，非持久化）
│   ├── logger.py                   #   统一日志配置，输出自动脱敏
│   ├── desensitize.py              #   PII脱敏：手机号/身份证/银行卡/邮箱正则识别打码
│   ├── compliance.py               #   免责声明文案、敏感词过滤
│   ├── taxonomy.py                 #   受控词表中枢：事件类型/情感维度/来源类型/可信度分级，
│   │                                #     被analysis_chain/lightweight_sentiment/repository等
│   │                                #     多处引用，新增分类只改这一个文件
│   ├── token_budget.py             #   RAG上下文Token预算控制（防止超出模型上下文窗口）
│   ├── usage_context.py            #   contextvars传递"当前租户/场景"，供llm_usage_callback
│   │                                #     记账时读取，避免每个chain函数都要传tenant_id参数
│   └── llm_usage_callback.py       #   LangChain回调：LLM调用结束时自动记录Token消耗
│
├── memory/
│   └── session_store.py            #   多轮对话短期记忆（会话历史），供qa_graph.py使用
│
├── processing/
│   └── cleaner.py                  #   原始文本清洗（去噪+脱敏），pipeline_graph的第一步
│
├── config/
│   └── settings.py                 #   全局配置中枢，所有环境变量在这里统一读取和给默认值，
│                                    #     其他模块一律 from config.settings import settings，
│                                    #     不直接os.getenv()（个别推送渠道凭证例外，见alert/push.py）
│
├── scripts/                        # 独立可执行脚本，不被其他模块import（除测试脚本外）
│   ├── run_pipeline.py             #   手动触发一轮采集+处理
│   ├── run_multi_source_demo.py    #   演示机构公告+社交热度采集
│   ├── run_backtest_demo.py        #   演示舆情-行情回测
│   ├── evaluate_sentiment.py       #   情感分析快速自测（内置10条测试用例）
│   ├── evaluate_phrasebank.py      #   情感分析正式评测（Financial PhraseBank数据集）
│   ├── evaluate_lightweight_prefilter.py  # 轻量情感模型效果评测
│   ├── benchmark_full_system.py    #   全系统基准测试：PII脱敏/预警规则/检索/并发/缓存等
│   ├── test_db_concurrency.py      #   db.py并发补丁验收测试（WAL/迁移兼容/DDL一次/并发正确性/性能对照）
│   ├── test_p9~p14_*.py            #   各功能模块的自动化验收测试（断言式）
│   ├── test_api.py                 #   FastAPI全部端点的自动化测试
│   └── test_p13_push_export.py     #   推送渠道+导出功能测试
│
├── data/
│   └── Sentences_100.txt           #   Financial PhraseBank中文翻译版测试数据
│
├── requirements.txt                 # Python依赖，大量可选依赖（akshare/redis/neo4j/pdfplumber等）
│                                   #   未安装时对应功能自动降级，不影响其他功能
├── .env.example                     # 环境变量模板，每个配置项都有中文注释说明用途
└── README.md                        # 本文件
```

---

## 四、模块关系说明（调用流向）

理解这个项目最关键的是分清"谁调用谁"，遵循单向依赖，没有循环调用：

```
入口层：app/streamlit_app.py 、 api/routers/*.py
              │
              ▼
编排层：graphs/*_graph.py  （每个文件对外一个函数：process() / ask() / generate_report() / handle_query()）
              │
              ▼
业务逻辑层：chains/*.py 、 alert/*.py 、 analysis/*.py 、 export/*.py
              │
              ▼
基础能力层：storage/repository.py 、 retrieval/*.py 、 market_data/*.py 、 knowledge_graph/*.py
              │
              ▼
横切/工具层：core/*.py 、 config/settings.py 、 storage/db.py
```

**几条具体的关键链路**（后续排查问题/加功能时最常用到）：

- **舆情处理主链路**：  
  `ingestion/*_source.py`（采集）→ `ingestion/async_jobs.py`或直接调  
  `graphs/pipeline_graph.process()` → 内部依次调用  
  `processing/cleaner.py`（清洗）→ `chains/analysis_chain.analyze_text_gated()`  
  （情感分析，内部先过`chains/lightweight_sentiment.py`门控）→  
  `alert/rules.py`（规则初筛）→ 命中则调`chains/alert_review_chain.py`（LLM复核）→  
  `storage/repository.insert_record()`（入库）→ 命中预警则调`alert/push.py`（多渠道推送）
- **RAG问答链路**：  
  `graphs/qa_graph.ask()` → `retrieval/retriever.hybrid_retrieve()`  
  （内部分别调`retrieval/vectorstore.py`和`storage/repository.search_text()`，  
  结果合并）→ `chains/prompts/qa_prompt.py`拼装Prompt → `chains/qa_chain.get_llm()`  
  → `memory/session_store.py`（多轮对话记忆读写）
- **日报链路**：  
  `graphs/report_graph.generate_report()` → `storage/repository.py`  
  （按公司聚合当天统计）→ `chains/report_chain.py`（每家公司生成小结）→  
  落盘Markdown + JSON侧车文件 → `alert/push.py`（推送）；  
  `export/report_export.py`独立读取JSON侧车文件生成Excel/PDF，**不经过上面这条链路**
- **公司信息的唯一数据源**：  
  `knowledge_graph/seed_data.py`的`COMPANIES`字典是全项目公司名/别名/股票代码/行业的  
  唯一权威数据，`market_data/provider.py`（解析股票代码）、  
  `ingestion/announcement_source.py`/`social_source.py`（采集时定位公司）、  
  `chains/lightweight_sentiment.py`（前置筛选时识别已知公司）、  
  `storage/repository.get_industry_heat_ranking()`（行业归属）都从这里读取，  
  **新增/修改公司信息只需要改这一个文件**，不用改多处
- **受控词表的唯一数据源**：  
  `core/taxonomy.py`集中定义事件类型、情感维度、来源类型、可信度分级四份受控词表，  
  `chains/prompts/analysis_prompt.py`（Prompt里约束LLM只能从词表里选）、  
  `chains/lightweight_sentiment.py`（前置筛选的粗规则分类）、  
  `storage/repository.py`（统计查询用的GROUP BY类目）都引用这里，**新增分类只改这一个文件**
- **配置的唯一数据源**：  
  `config/settings.py`是全项目环境变量的统一读取入口，绝大多数模块都是  
  `from config.settings import settings`后用`settings.XXX`访问配置，  
  **例外**：`alert/push.py`里各推送渠道的凭证是直接`os.getenv()`读取，没有走  
  `settings`对象（历史遗留，非bug，但排查配置问题时要记得这个例外）

---

## 五、核心功能清单

| 功能            | 关键文件                                                  | 说明                           |
| ------------- | ----------------------------------------------------- | ---------------------------- |
| 舆情采集分析预警主流程   | `graphs/pipeline_graph.py`                            | MVP阶段核心能力                    |
| RAG混合检索问答     | `graphs/qa_graph.py`、`retrieval/*.py`                 | 向量+关键词双检索、多轮记忆               |
| 每日简报          | `graphs/report_graph.py`                              | 定时+手动触发，可推送                  |
| 多维度情感+标准化事件分类 | `chains/analysis_chain.py`、`core/taxonomy.py`         | 四维度打分替代单一情感分                 |
| 舆情-行情关联分析与回测  | `analysis/backtest.py`、`market_data/provider.py`      | 研究工具，非投资建议                   |
| 专业数据源接入       | `ingestion/announcement_source.py`、`social_source.py` | 机构公告+市场热度代理                  |
| 多公司对比/行业热度榜   | `storage/repository.py`（聚合查询函数）                       |                              |
| 轻量模型前置筛选      | `chains/lightweight_sentiment.py`                     | 成本优化，明确案例不调LLM               |
| 多渠道推送与导出      | `alert/push.py`、`export/report_export.py`             | 6个推送渠道+Excel/PDF导出           |
| SaaS基础能力      | `storage/usage_tracker.py`、`storage/audit_log.py`     | 用量统计+操作审计                    |
| REST API      | `api/`                                                | FastAPI，20+端点                |
| 多模态输入         | `ingestion/multimodal/`                               | 图片OCR已实现，视频ASR未完整实现          |
| 多租户隔离         | 各模块的`tenant_id`参数                                     | 数据隔离已实现，无真实SSO鉴权             |
| 知识图谱          | `knowledge_graph/`                                    | 公司-事件-行业关系，Neo4j/NetworkX双后端 |

---

## 六、已知局限（供问题排查/功能扩展参考）

### 6.1 三大问题排查状态（持续更新）

1. **并发处理能力不足 —— 已解决（storage/db.py 2026-10-04 + LLM/API异步化 2026-10-07）**
   - **已解决**：`storage/db.py`原实现每次数据库连接都重跑建表/迁移DDL、未开WAL。  
     已重新实现并发补丁并合入主干：**进程内单共享连接+可重入锁串行化**（所有线程  
     共用一条长连接，with块由RLock串行）+ **WAL模式**（只在初始化时设一次，  
     实测高并发下对活跃库重复执行journal_mode切换会触发readonly错误）+  
     **busy_timeout**。验收测试`scripts/test_db_concurrency.py` 8/8通过，  
     本机对照数据：旧版行为在8写+4读并发下仅10写/秒、读P50达297ms且丢写入；  
     新实现3126写/秒、读P50 0.03ms、200/200零错误。  
     **重要发现**：本机环境（疑似杀毒软件实时扫描干扰SQLite锁语义）下，  
     "多连接并发访问"本身就不稳定，纯标准库即可复现——共享连接+串行化是  
     唯一稳健方案，比原计划的"WAL+多连接"更正确。
   - **LLM/API异步化 —— 全部完成（2026-10-07，阶段1+2+3）**：QA链路（检索→生成→三层校验）
     与意图路由已双模式化——同步路径全保留（Streamlit/评测/worker零变化），异步版
     平行新增：`with_retry_async`/`ainvoke_structured`/`classify_intent_async`/
     `generate_answer_async`/`verify_answer_async`（LLM走`ainvoke`，DB/检索走
     `asyncio.to_thread`，重试退避用`await asyncio.sleep`）；
     `build_qa_graph`/`build_router_graph`/`build_pipeline_graph`/`build_report_graph`
     均支持`async_mode`双模式构图；
     LLM路由全部改为`async def`（`/qa`、`/query`、`/ingest`同步模式、`/reports/daily`）
     ——其余纯DB读路由**故意保持同步**（SQLite微秒级，进事件循环反而阻塞）。
     阶段2将analysis_chain/alert_review_chain/report_chain/pipeline_graph/
     report_graph全部异步化：pipeline的embed/save/push走`to_thread`；
     **日报多公司小结用`asyncio.gather`并发**（N家公司从串行N倍延迟降为≈最慢一家，
     contextvars随gather子任务继承，用量归因P14不受影响）；report落盘逻辑拆出
     共用函数消除两份副本。     改造与压测过程修复三个真bug：
     ①`qa_graph`的`add_edge`误传函数对象（应为节点名字符串，旧版langgraph侥幸兼容）；
     ②`get_vectorstore()`并发初始化竞态——chromadb PersistentClient多线程同时
     首次构建会互相干扰，报"Could not connect to tenant default_tenant"，
     已加RLock双检锁（基准并发首轮5/10线程复现，修复后10线程×3轮全过）；
     ③**条件边路由函数内改state不生效**（langgraph 1.x规定路由函数必须纯净，
     旧版侥幸兼容）——`qa_graph.check_sufficiency`的retry_count自增丢失导致
     检索不足时无限循环（GraphRecursionError，/qa报500，同步版同样潜伏）；
     `pipeline_graph.alert_check_node`的need_alert写入丢失导致落库预警标记
     永远为0（预警统计/回测triggered口径会静默归零）。修复：重试计数移入
     retrieve_node，need_alert移入alert_review_node并由update_alert_review
     写库；主库单条记录复现→修复后异步/同步/桩测三路回归全过。
     基准测试`scripts/bench_concurrency.py`（sync线程池 vs async事件循环，
     同题同库，每请求记录启动偏移/耗时/p50）：并发5共5轮，全部5/5成功，
     **两模式均验证为真并发**（启动偏移≈0s）；吞吐比0.96x~2.13x、均值≈1.3x——
     瓶颈在deepseek-v4-flash服务端的并发排队调度（同一时刻5在途请求全被拖慢），
     不在客户端并发模型；池化单请求p50：async 7.9s vs sync 14.6s（≈1.8x）。
     结论：小并发+服务端限流的场景下，线程池与事件循环吞吐接近（线程等待网络
     I/O时同样释放GIL）；异步化的真实收益在于去掉线程每请求开销（数百并发连接
     不占线程资源）、接入FastAPI async生态，且在服务商配额放开或自建推理时
     直接兑现为吞吐。
     **真实验证已过**：异步`/ingest`提交→LLM分析入库→异步`/reports/daily`
     生成含LLM小结的日报（md+json落盘一致）；同步`run_daily_report.py`回归无变化。
     压测脚本`scripts/locustfile.py`（/qa+/query+baseline健康检查对照，API Key
     自动读.env，支持headless+HTML报告），可复现压测命令见脚本docstring。
   - `core/task_queue.py`是进程内线程池实现，非持久化队列，进程重启会丢失未处理任务
2. **向量检索召回质量未经验证 —— 已解决（2026-10-05）**
   - 向量检索已在真实环境跑通并完成质量评测：本地模型`D:\test_model\bge-large-zh-v1.5`  
     （.env生产配置），语料104篇中文金融新闻、40条人工标注查询（semantic 24 /  
     keyword 8 / multi 8），全部走项目生产链路（repository灌库 → rebuild向量重建  
     → retriever三种检索）。**结果：向量检索 Recall@5=0.975、MRR=0.955、nDCG@5=0.956、  
     Hit@10=100%、P50延迟59.6ms**——质量达标，唯一的实质漏检是1条多文档归集查询  
     （2个答案只捞回1个）；multi类型均分0.875主要是指标口径问题（10个答案的查询  
     Recall@5上限只有0.5），非检索缺陷。报告：`scripts/eval_results/retrieval_eval_report.md`
   - FTS5关键词检索结论被本次评测**大幅修正**：在句子级中文语料上召回率≈0  
     （Recall@5=0.025，40条查询仅"宁王"1条命中），远差于早期小规模测试的"双字词50%"。  
     根因：SQLite默认unicode61分词器按标点切分中文，长句片段成为单个巨型token，  
     句内嵌套的英文词（如IPO）也无法独立成token；含标点的查询（600519.SH）还会触发  
     FTS5查询语法错误、被降级静默吞掉。**修复已实施（2026-10-07）**：索引侧与查询侧  
     同一套分词（`storage/fts_tokenizer.py`，jieba优先/中文二元组降级，永不失败）——  
     入库时预分词写入`sentiment_records.fts_tokens`列（老库自动迁移+存量回填+索引  
     rebuild，见`storage/db.py`），查询侧token双引号包裹根治语法错误；两阶段匹配  
     （AND精确优先→零命中降级OR+bm25排序，解决降级分词的跨界二元组漏召）。  
     离线冒烟测试12/12通过（`scripts/test_fts_fix.py`，含老库迁移回填与并发验收回归）。  
     **复测结果（2026-10-07）**：FTS5 Recall@5 **0.025→0.914**（36.6倍）、MRR=0.786、  
     Hit@10=95%、P50延迟1.2ms（比向量快约57倍）、语法错误查询1→0；向量检索0.975无回退。  
     残余5条漏召全部是词面不同的口语化语义查询（"承销业务最猛"vs"承销规模领先"），  
     属向量检索的领地——两路互补正是混合架构的设计依据。报告含逐条归因  
     （`scripts/eval_results/retrieval_eval_report.md`）
   - 混合检索与纯向量指标逐位相同（40条零差异），但性质已变：FTS5漏召的恰好都是  
     向量满分覆盖的部分，融合无增益属预期；FTS5以1.2ms延迟提供精确匹配兜底  
     （股票代码/专有名词等向量弱项），语义覆盖由向量承担，两路各司其职，  
     混合架构健康度恢复
3. **LLM幻觉与输出质量缺乏工程化控制 —— 已解决（2026-10-05/07）**
   - 实现了三层程序化校验（`core/faithfulness.py`）：①引用编号核查（回答中的\[1\]\[2]标注  
     必须指向真实资料）②数值/实体忠实性回查（回答中的数字/公司名逐个回查检索资料原文，  
     问题原文复述不算编造）③LLM蕴含复核（judge判定回答是否被资料完全支持，可经  
     QA_VERIFY环境变量开关）。拒答不算幻觉；校验自身失败全降级放行，不影响主流程
   - 接入生产链路：`qa_prompt.py`新增引用标注版Prompt（原版保留作baseline）、  
     `qa_graph.py`新增verify_node节点（生成→校验→带具体原因受限重生成1次→  
     仍不过则在回答前加"未经核实"警告标注）
   - 真实LLM评测（deepseek-v4-flash，120次调用，30题=库内15/库外15，两模式对比）：  
     **库外拒答率100%（规则+judge双口径、两模式一致）、编造率0%**；库内忠实率  
     (judge) **86.7%→100%**、引用标注覆盖率100%（平均2.9个引用/回答）、校验零误伤  
     （初版规则的33.3%风险率经逐条归因全部为日期误报，校准后≈0）。校验未引入过度拒答
   - 评测报告：`scripts/eval_results/hallucination_eval_report.md`（含规则校准补记与终版结论）
   - 遗留说明：评测脚本曾发现初版规则的三处误判（含糊回答误判拒答/日期误报/问题原文  
     数字误报编造），均已校准修复并有备份（backups/*.before_p3fix.bak）——校准过程  
     本身可作为"评测驱动迭代"的案例

### 6.2 其他已知的待完善项（优先级供参考，非本次交接重点）

- **企业级SSO鉴权未接入**：当前多租户靠`tenant_id`/`X-Tenant-Id`请求头自报，没有真实身份  
  验证，`storage/audit_log.py`的操作人记录同理不可抵赖——这是"个人项目"和"能上生产"之间  
  最大的单项差距
- **视频ASR转写**：已接入本地faster-whisper（2026-10-08，settings.ASR_PROVIDER=local，懒加载单例+失败降级不阻断主流程；机制冒烟测试`scripts/test_video_asr.py`，转写质量需真实素材人工抽测）；云端ASR API为预留接入点（实现`_transcribe_audio_cloud`并在`_transcribe_audio`追加分发即可，上游零改动）
- **对象存储**：`media_path`目前假设是本地文件路径，未接入S3/OSS等对象存储
- **数据库水平扩展路径**：SQLite/Chroma都是单机方案，数据量增长后需要评估迁移到  
  Postgres/向量数据库集群
- **部署运维**：暂无Docker Compose/K8s部署方案，`Docker集成指南.md`仅为文档指引

---

## 七、环境配置与运行

```bash
pip install -r requirements.txt
cp .env.example .env    # 按需填入ANTHROPIC_API_KEY或OPENAI_API_KEY等

# 跑一轮采集+分析
python scripts/run_pipeline.py

# 启动Streamlit前端
streamlit run app/streamlit_app.py

# 启动FastAPI（另开终端）
uvicorn api.main:app --reload --port 8000
# 访问 http://localhost:8000/docs 看接口文档

# 跑自动化测试（不需要LLM API Key的部分）
python scripts/benchmark_full_system.py
python scripts/test_db_concurrency.py   # db.py并发补丁验收
python scripts/test_p9_dimensions.py    # 以及 test_p10/p11/p12/p13/p14

# 跑需要真实LLM API Key的评测
python scripts/evaluate_phrasebank.py
```

**必需环境变量**：`ANTHROPIC_API_KEY`或`OPENAI_API_KEY`（二选一，见`config/settings.py`  
的`LLM_PROVIDER`开关）。其余所有环境变量都有合理默认值或自动降级逻辑，缺失不会导致  
程序无法启动，只会导致对应功能降级（详见`.env.example`里每一项的注释）。

## 八、Docker 部署

前提：安装 Docker Desktop（Windows 需启用 WSL2）。密钥（`.env`）、数据（`data/`、`reports/`）、
本地模型均不进镜像，运行时挂载。

```bash
# 1. 确认 .env 存在（LLM Key 等由 compose 的 env_file 注入容器）
# 2. 确认 docker-compose.yml 里模型挂载路径正确（默认 D:/test_model/bge-large-zh-v1.5）

# 3. 构建并启动（首次构建较久：CPU版torch+全量依赖）
docker compose up -d --build

# 4. 验证
docker compose ps                     # 两个容器 Up、api 健康
docker compose logs -f api            # 看启动日志
# API:    http://localhost:8000/docs
# 界面:   http://localhost:8501
```

常用运维：

```bash
docker compose down              # 停止（数据保留在宿主机 data/、reports/）
docker compose up -d --build     # 代码更新后重建
docker compose build --no-cache  # 依赖变更后无缓存重建
```

设计说明：镜像基于 python:3.11-slim，内置 ffmpeg 与 tesseract（中英文语言包），
torch 单独走 CPU 源安装以控制镜像体积；API 容器带 `/health` 健康检查；
SQLite/Chroma/日报全部通过 volume 落在宿主机，容器可随意销毁重建。
