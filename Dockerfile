# Customer Service Assistant 客服 Agent —— API 服务镜像
#
# 构建（在 customer_service_assistant/ 目录下）：
#   docker build -t customer-service-assistant-api:1.0.0 .
# 国内网络拉不到 Docker Hub / PyPI 时可用镜像源：
#   docker build -t customer-service-assistant-api:1.0.0 \
#     --build-arg BASE_IMAGE=docker.m.daocloud.io/library/python:3.13-slim \
#     --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .
ARG BASE_IMAGE=python:3.13-slim
FROM ${BASE_IMAGE}

# 允许覆盖 pypi 源（默认官方源）
ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 依赖单独一层，改代码不用重装依赖
COPY requirements.txt ./
RUN pip install --index-url ${PIP_INDEX_URL} -r requirements.txt

# 只复制运行所需内容（tests/docs 不进镜像）
COPY pyproject.toml README.md ./
COPY customer_service_assistant ./customer_service_assistant
COPY scripts ./scripts
COPY data ./data

# 知识库文件不打包进镜像（它是教程根目录的单一事实来源）：
# 需要 /ingest 时用 volume 挂载到 /knowledge，并设置 CSA_KNOWLEDGE_DIR=/knowledge
ENV CSA_KNOWLEDGE_DIR=/knowledge \
    CSA_LOG_DIR=/app/logs \
    CSA_MILVUS_URI=http://host.docker.internal:19530

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=6s --start-period=25s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8000/api/v1/healthz', timeout=5).status_code == 200 else 1)"

CMD ["python", "-m", "uvicorn", "customer_service_assistant.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
