"""重排：SiliconFlow ``/v1/rerank``（``BAAI/bge-reranker-v2-m3``）。

为什么需要重排：RRF 融合只用了"排名"，没有看"内容与问题的相关性"。交叉编码器（cross-encoder）
把 query 与候选片段一起编码打分，能显著提升 top-k 的精度，也提供了做拒答判断的**绝对分数**。

设计要点：

1. **按片段打分并缓存**（key = model + query + 片段文本的哈希），重复实验不重复付费；
2. 只对"未命中缓存"的片段发起请求，一次调用批量打分；
3. 失败重试 + 明确异常类型，便于上层降级（P3 会接入 `ModelFallbackMiddleware` 式的降级）。
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Sequence

import httpx
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import Settings, get_settings
from ..models import RetrievedChunk

RERANK_PATH = "/rerank"


class RerankError(RuntimeError):
    """重排服务调用失败。"""


@dataclass(slots=True)
class RerankItem:
    index: int
    score: float


class SiliconFlowReranker:
    """SiliconFlow 重排客户端（带进程内缓存）。"""

    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.require_embedding_credentials()  # 复用 SILICONFLOW_* 凭据
        self._client = client or httpx.Client(timeout=self.settings.rerank_timeout)
        self._cache: OrderedDict[str, float] = OrderedDict()

    # ---------------- 缓存 ----------------
    def _cache_key(self, query: str, document: str) -> str:
        raw = f"{self.settings.rerank_model}|{query}|{document}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> float | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: str, value: float) -> None:
        if self.settings.rerank_cache_size <= 0:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.settings.rerank_cache_size:
            self._cache.popitem(last=False)

    # ---------------- 调用 ----------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((httpx.HTTPError, RerankError)),
        reraise=True,
    )
    def _request(self, query: str, documents: list[str], top_n: int) -> list[RerankItem]:
        url = self.settings.siliconflow_base_url.rstrip("/") + RERANK_PATH
        payload = {
            "model": self.settings.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }
        response = self._client.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {self.settings.siliconflow_api_key}"},
        )
        if response.status_code >= 400:
            raise RerankError(f"rerank 调用失败: HTTP {response.status_code} {response.text[:200]}")
        payload = response.json()
        results = payload.get("results") or []
        if not results:
            raise RerankError(f"rerank 返回为空: {str(payload)[:200]}")
        return [RerankItem(index=int(item["index"]), score=float(item["relevance_score"])) for item in results]

    def rerank(self, query: str, documents: Sequence[str]) -> list[RerankItem]:
        """返回按分数降序的 (原始下标, 分数) 列表。"""
        documents = list(documents)
        if not documents:
            return []

        scores: dict[int, float] = {}
        missing: list[tuple[int, str]] = []
        for index, document in enumerate(documents):
            cached = self._cache_get(self._cache_key(query, document))
            if cached is None:
                missing.append((index, document))
            else:
                scores[index] = cached

        if missing:
            started = time.perf_counter()
            items = self._request(query, [document for _, document in missing], top_n=len(missing))
            elapsed = time.perf_counter() - started
            for item in items:
                if 0 <= item.index < len(missing):
                    original_index = missing[item.index][0]
                    scores[original_index] = item.score
                    self._cache_put(self._cache_key(query, documents[original_index]), item.score)
            logger.debug(
                "rerank {} docs in {:.2f}s（缓存命中 {}）",
                len(missing),
                elapsed,
                len(documents) - len(missing),
            )

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [RerankItem(index=index, score=score) for index, score in ranked]

    def close(self) -> None:
        self._client.close()


def rerank_hits(
    hits: Sequence[RetrievedChunk],
    query: str,
    *,
    settings: Settings | None = None,
    reranker: SiliconFlowReranker | None = None,
    use_parent_text: bool | None = None,
) -> list[RetrievedChunk]:
    """对候选片段重排：写入 ``rerank_score`` 并按分数降序返回。

    默认用 **child 片段文本**打分：它是命中的精确片段，语义集中；
    章节全文（parent）更长、噪声更多会稀释交叉编码器的判断。需要对照时用
    ``CSA_RERANK_ON_PARENT=true`` 打开。
    """
    settings = settings or get_settings()
    hits = list(hits)
    if not hits:
        return []

    use_parent_text = settings.rerank_on_parent if use_parent_text is None else use_parent_text
    documents = [
        (hit.parent_text or hit.text) if use_parent_text else hit.text
        for hit in hits
    ]
    reranker = reranker or SiliconFlowReranker(settings)
    ranked = reranker.rerank(query, documents)

    # 返回副本而不是就地改写：调用方（调试脚本、评测）经常需要保留重排前的候选顺序
    return [
        replace(hits[item.index], rerank_score=item.score, rank=rank)
        for rank, item in enumerate(ranked, start=1)
    ]
