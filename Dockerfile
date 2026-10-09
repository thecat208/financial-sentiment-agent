# ---------- 金融舆情分析系统 生产镜像 ----------
# 构建：docker compose build   或   docker build -t financial-sentiment-agent .
# 运行：docker compose up -d   （推荐，见 docker-compose.yml）

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_ENDPOINT=https://hf-mirror.com

# 系统依赖：ffmpeg（视频抽帧/音轨）、tesseract（图片OCR，中英文语言包）、curl（健康检查）
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        tesseract-ocr \
        tesseract-ocr-chi-sim \
        tesseract-ocr-eng \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先单独装 CPU 版 torch：PyPI 默认的 Linux torch 自带 CUDA，体积多出数 GB
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 再装项目依赖（先拷 requirements 利用 Docker 层缓存，代码变更不触发重装）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝源码（.dockerignore 已排除 venv/data/.env/数据库文件等）
COPY . .

# 运行时状态目录：生产环境用 volume 挂载持久化，此处仅保证目录存在且可写
RUN mkdir -p /app/data/chroma_db /app/reports \
    && useradd -m appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
