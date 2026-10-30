"""模型工厂：Chat 模型与 Embedding 模型的唯一创建入口。

统一在这里处理四件事（设计文档附录 C 第 4 条）：
超时、重试、日志、可被测试替换。
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Iterable

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Settings, get_settings


def build_chat_model(settings: Settings | None = None) -> Any:
    """创建对话模型（不带缓存，测试可传入自定义 Settings）。

    踩坑记录：**不要**在这里用 ``model.with_retry(...)`` 包一层。它返回的是 ``RunnableRetry``，
    不暴露 ``bind_tools``，而 ``create_agent`` 需要模型对象直接支持工具绑定，会报
    ``'RunnableRetry' object has no attribute 'bind_tools'``。
    重试策略交给 ``ModelRetryMiddleware``（P3 在 ``agent/middleware.py`` 里配置）。
    """
    settings = settings or get_settings()
    settings.require_llm_credentials()

    model = init_chat_model(
        model=settings.llm_model,
        model_provider=settings.llm_provider,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_retry_attempts,
    )
    logger.debug("chat model ready: {}:{}", settings.llm_provider, settings.llm_model)
    return model


def build_embed_model(settings: Settings | None = None) -> Any:
    """创建嵌入模型（不带缓存）。"""
    settings = settings or get_settings()
    settings.require_embedding_credentials()

    model = init_embeddings(
        model=settings.embed_model,
        api_key=settings.siliconflow_api_key,
        base_url=settings.siliconflow_base_url,
        timeout=settings.embed_timeout,
    )
    logger.debug("embedding model ready: {}", settings.embed_model)
    return model


@lru_cache(maxsize=1)
def get_chat_model() -> Any:
    return build_chat_model()


@lru_cache(maxsize=1)
def get_embed_model() -> Any:
    return build_embed_model()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _embed_batch(model: Any, texts: list[str]) -> list[list[float]]:
    return model.embed_documents(texts)


def embed_documents(texts: Iterable[str], settings: Settings | None = None, model: Any | None = None) -> list[list[float]]:
    """批量嵌入：自动分批 + 重试 + 计时日志。

    分批是必要的：SiliconFlow 的单次请求条数有上限，且批次失败时只需重试该批。
    """
    settings = settings or get_settings()
    model = model or get_embed_model()
    items = [text for text in texts]
    if not items:
        return []

    started = time.perf_counter()
    vectors: list[list[float]] = []
    for start in range(0, len(items), settings.embed_batch):
        batch = items[start : start + settings.embed_batch]
        vectors.extend(_embed_batch(model, batch))
    elapsed = time.perf_counter() - started
    logger.info("embedded {} texts in {:.2f}s ({:.1f} texts/s)", len(items), elapsed, len(items) / elapsed if elapsed else 0)
    return vectors


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _embed_one(model: Any, text: str) -> list[float]:
    return list(model.embed_query(text))


def embed_query(text: str, model: Any | None = None) -> list[float]:
    """单条查询向量化。"""
    return _embed_one(model or get_embed_model(), text)


def setup_tracing(settings: Settings | None = None) -> bool:
    """把配置写入 LangSmith 需要的环境变量（langsmith SDK 直接读环境变量）。

    返回是否启用 tracing。
    """
    import os

    settings = settings or get_settings()
    enabled = settings.tracing_enabled
    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    if enabled:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key or ""
        if settings.langsmith_endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        if settings.langsmith_project:
            os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        logger.info("LangSmith tracing 已启用 -> {}", settings.langsmith_project)
    else:
        logger.info("LangSmith tracing 未启用（CSA_ENABLE_TRACING=false 或未配置 LANGSMITH_API_KEY）")
    return enabled
