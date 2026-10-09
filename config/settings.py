"""
统一配置管理。所有可调参数从 .env 读取，避免代码里硬编码密钥和路径。
使用前请复制 .env.example 为 .env 并填入真实的API Key。
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ---- LLM 配置 ----
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # anthropic / openai

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")
    # 兼容OpenAI协议的第三方服务接入地址（如DeepSeek填 https://api.deepseek.com/v1）。
    # 留空=用官方 https://api.openai.com/v1。填了它，OPENAI_API_KEY 就该是那家服务商的Key，
    # OPENAI_MODEL 也要换成那家的模型名——否则会出现"Key拿对了但发去OpenAI官方端点"的401。
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
    # 结构化输出的绑定方式：auto=先试最严格的json_schema，服务端不支持时自动降级到json_mode
    # （DeepSeek等只支持response_format=json_object的网关必须靠这个自动降级）；
    # 也可以显式写死 json_schema / json_mode / function_calling。实现见 chains/structured_output.py
    LLM_STRUCTURED_METHOD = os.getenv("LLM_STRUCTURED_METHOD", "auto")

    # ---- Embedding 配置 ----
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")  # local / openai
    LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

    # ---- 向量库配置 ----
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "financial_sentiment")

    # ---- 检索参数 ----
    RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", 5))
    RETRIEVE_MIN_DOCS = int(os.getenv("RETRIEVE_MIN_DOCS", 2))
    RETRIEVE_MAX_RETRY = int(os.getenv("RETRIEVE_MAX_RETRY", 2))

    # ---- 企业级多租户 ----
    DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "default")

    # ---- 多模态 ----
    OCR_LANG = os.getenv("OCR_LANG", "chi_sim+eng")
    VIDEO_FRAME_INTERVAL = int(os.getenv("VIDEO_FRAME_INTERVAL", 30))
    # ASR语音转写方案：local=本地faster-whisper（默认，pip install faster-whisper）；
    # cloud=云端ASR API预留接入点（在ingestion/multimodal/video_processor.py的
    # _transcribe_audio docstring里有接入指引，实现_transcribe_audio_cloud后生效）
    ASR_PROVIDER = os.getenv("ASR_PROVIDER", "local")
    # 本地whisper模型规模：tiny/base/small/medium/large-v3——small在纯CPU上速度/准确率较均衡；
    # 首次运行自动从HuggingFace下载权重（国内网络慢可设HF_ENDPOINT=https://hf-mirror.com）
    ASR_WHISPER_MODEL_SIZE = os.getenv("ASR_WHISPER_MODEL_SIZE", "small")
    # 运行设备：auto=有CUDA走GPU否则CPU；也可强制cpu
    ASR_WHISPER_DEVICE = os.getenv("ASR_WHISPER_DEVICE", "auto")
    # 转写语言：zh=中文；auto=自动检测
    ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "zh")

    # ---- 对话记忆（上下文与记忆管理）----
    # 每次问答拼进Prompt的最近对话轮数（短期滑动窗口）
    MEMORY_CONTEXT_RECENT_TURNS = int(os.getenv("MEMORY_CONTEXT_RECENT_TURNS", 4))
    # 累计对话达到该轮数时，触发LLM把历史压缩为长期摘要
    MEMORY_SUMMARY_TRIGGER_TURNS = int(os.getenv("MEMORY_SUMMARY_TRIGGER_TURNS", 16))
    # 压缩完成后短期窗口保留的最近轮数（避免最近上下文断掉）
    MEMORY_TRIM_KEEP_TURNS = int(os.getenv("MEMORY_TRIM_KEEP_TURNS", 2))

    # ---- 缓存与Redis ----
    # Redis地址；不装redis/不启动服务时，缓存与会话自动降级为进程内实现，不影响功能
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    # 检索结果缓存时长（秒），默认10分钟——新舆情入库后最多延迟这个时间可见
    CACHE_TTL_RETRIEVE = int(os.getenv("CACHE_TTL_RETRIEVE", 600))
    # LLM调用结果缓存时长（秒），默认1小时（情感分析/预警复核/意图识别按内容hash缓存）
    CACHE_TTL_LLM = int(os.getenv("CACHE_TTL_LLM", 3600))
    # 热会话记忆TTL（秒），默认30分钟，每次访问滑动续期
    SESSION_TTL_HOT = int(os.getenv("SESSION_TTL_HOT", 1800))

    # ---- Token与成本控制 ----
    # 模型上下文窗口；0=按模型名自动判断（Claude=200K，GPT-4o-mini=128K）
    MODEL_CONTEXT_LIMIT = int(os.getenv("MODEL_CONTEXT_LIMIT", 0))
    # 输入预算占上下文极限的比例，默认80%，预留系统提示词和输出空间
    TOKEN_BUDGET_RATIO = float(os.getenv("TOKEN_BUDGET_RATIO", 0.8))
    # 系统提示词的token预留
    TOKEN_SYSTEM_RESERVED = int(os.getenv("TOKEN_SYSTEM_RESERVED", 800))
    # 可用预算中分给对话历史的比例，其余给RAG检索文档
    TOKEN_CONV_SHARE = float(os.getenv("TOKEN_CONV_SHARE", 0.3))

    # ---- LLM调用超时与重试 ----
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 60))            # 单次LLM调用超时（秒）
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", 3))     # 指数退避最大重试次数
    RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", 1.0))  # 退避基础间隔（秒）

    # ---- 异步任务队列与限流 ----
    TASK_QUEUE_CONCURRENCY = int(os.getenv("TASK_QUEUE_CONCURRENCY", 2))  # 后台worker并发数
    TASK_QUEUE_NAME = os.getenv("TASK_QUEUE_NAME", "tasks:queue")         # Redis队列key
    # 全局令牌桶：每秒补充令牌数（整体QPS上限，对应网关层限流）
    RATE_LIMIT_GLOBAL_QPS = float(os.getenv("RATE_LIMIT_GLOBAL_QPS", 20))
    # 令牌桶容量（允许瞬时突发）
    RATE_LIMIT_GLOBAL_BURST = int(os.getenv("RATE_LIMIT_GLOBAL_BURST", 40))
    # 单用户限流窗口（秒），对应业务层针对单个user_id的限流
    RATE_LIMIT_USER_WINDOW_SEC = int(os.getenv("RATE_LIMIT_USER_WINDOW_SEC", 2))
    # 单用户在窗口内的最大请求数
    RATE_LIMIT_USER_MAX = int(os.getenv("RATE_LIMIT_USER_MAX", 1))

    # ---- 知识图谱 ----
    # auto=优先Neo4j、连不上自动降级NetworkX；neo4j=强制Neo4j；networkx=强制NetworkX
    KG_PROVIDER = os.getenv("KG_PROVIDER", "auto")
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
    # NetworkX降级后端时的图持久化文件
    KG_FILE_PATH = os.getenv("KG_FILE_PATH", "./data/knowledge_graph.json")
    # 构建图时是否加载内置种子金融知识库
    KG_LOAD_SEED = os.getenv("KG_LOAD_SEED", "true").lower() in ("1", "true", "yes")

    # ---- 向量索引重建 ----
    VECTOR_REBUILD_HOUR = int(os.getenv("VECTOR_REBUILD_HOUR", 2))  # 每日重建时间（凌晨2点）
    # 活跃Chroma集合指针文件（重建后切换集合名）
    CHROMA_ACTIVE_POINTER = os.getenv("CHROMA_ACTIVE_POINTER", "./data/chroma_active.json")

    # ---- 安全与合规 ----
    # 敏感词清单文件路径（每行一个词）；留空用内置默认清单
    SENSITIVE_WORDS_FILE = os.getenv("SENSITIVE_WORDS_FILE", "")
    # 是否启用敏感词过滤（输入拦截 + 入库正文打码）
    ENABLE_SENSITIVE_FILTER = os.getenv("ENABLE_SENSITIVE_FILTER", "true").lower() in ("1", "true", "yes")
    # 是否在问答/日报末尾强制拼接免责声明
    ENABLE_DISCLAIMER = os.getenv("ENABLE_DISCLAIMER", "true").lower() in ("1", "true", "yes")

    # ---- 舆情-行情关联分析与回测 ----
    # akshare=优先真实A股行情，未安装/取不到数据/非A股代码时自动降级为mock演示数据；
    # mock=强制使用演示数据（离线开发/演示环境用）
    MARKET_DATA_PROVIDER = os.getenv("MARKET_DATA_PROVIDER", "akshare")
    # 行情数据缓存时长（秒），默认4小时——日线数据不需要太高实时性
    MARKET_DATA_CACHE_TTL = int(os.getenv("MARKET_DATA_CACHE_TTL", 14400))
    # 回测默认持有交易日数（预警触发后持有N个交易日再统计涨跌幅）
    BACKTEST_DEFAULT_HOLDING_DAYS = int(os.getenv("BACKTEST_DEFAULT_HOLDING_DAYS", 5))
    # 样本量低于此值时，回测结果展示"样本过少，仅供参考"提示
    BACKTEST_MIN_SAMPLES = int(os.getenv("BACKTEST_MIN_SAMPLES", 5))

    # ---- 专业数据源与公告全文接入 ----
    # 公告采集默认回溯天数
    ANNOUNCEMENT_LOOKBACK_DAYS = int(os.getenv("ANNOUNCEMENT_LOOKBACK_DAYS", 7))
    # 是否尝试下载公告PDF并抽取全文（需要pdfplumber，未安装/下载失败时自动降级为仅用标题）
    ANNOUNCEMENT_FETCH_FULLTEXT = os.getenv("ANNOUNCEMENT_FETCH_FULLTEXT", "true").lower() in ("1", "true", "yes")
    # 公告PDF全文超过此字数截断（避免超长财报正文把LLM输入撑爆）
    ANNOUNCEMENT_FULLTEXT_MAX_CHARS = int(os.getenv("ANNOUNCEMENT_FULLTEXT_MAX_CHARS", 3000))

    # ---- FastAPI 对外接口 ----
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    # 留空=不校验（本地开发/演示默认），非空时所有 /api/* 请求必须带 X-API-Key 头且值匹配，
    # 生产环境务必设置，否则接口对公网完全开放。
    API_KEY = os.getenv("API_KEY", "")
    # 允许跨域访问的来源，逗号分隔；"*"=允许所有来源（仅本地开发用，生产环境应指定具体域名）
    API_CORS_ORIGINS = [o.strip() for o in os.getenv("API_CORS_ORIGINS", "*").split(",") if o.strip()]

    # ---- 轻量模型前置筛选（成本优化）----
    # 是否启用前置筛选；关闭则每条都直接走LLM（等价于升级前的行为，用于AB对比基线）
    ENABLE_LIGHTWEIGHT_PREFILTER = os.getenv("ENABLE_LIGHTWEIGHT_PREFILTER", "true").lower() in ("1", "true", "yes")
    # auto=优先transformer小模型、不可用自动降级lexicon词典规则；
    # transformer=强制小模型（不可用时判定为低置信度，强制升级LLM，不冒充结果）；lexicon=强制词典规则
    LIGHTWEIGHT_SENTIMENT_BACKEND = os.getenv("LIGHTWEIGHT_SENTIMENT_BACKEND", "auto")
    # transformer后端使用的HuggingFace模型（需要能联网下载权重，仅首次运行需要）
    LIGHTWEIGHT_HF_MODEL = os.getenv("LIGHTWEIGHT_HF_MODEL", "uer/roberta-base-finetuned-jd-binary-chinese")
    # 置信度阈值：轻量模型判断的置信度达到这个值才会被采纳，直接跳过LLM；
    # 没达到、或公司识别不到种子库已知公司，都会升级到LLM完整分析
    LIGHTWEIGHT_CONFIDENCE_THRESHOLD = float(os.getenv("LIGHTWEIGHT_CONFIDENCE_THRESHOLD", 0.6))

    # ---- 多渠道推送与导出 ----
    # 导出文件（Excel/PDF）的输出目录；推送渠道的凭证（DINGTALK_WEBHOOK等）
    # 在 alert/push.py 里直接用 os.getenv 读取，和 SERVERCHAN_KEY 保持同一种写法，不放这里
    EXPORT_DIR = os.getenv("EXPORT_DIR", "./exports")


settings = Settings()
