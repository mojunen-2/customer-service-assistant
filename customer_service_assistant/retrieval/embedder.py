"""嵌入层：批量嵌入 + 内容哈希缓存 + 失败重试。

缓存的意义（设计文档 §5.10 成本优化）：知识库文档多数情况下不变，而嵌入是唯一
"每篇文章都要重新付费"的环节。以 ``content_hash`` 为键缓存后，重复 ingest 只对
**新增或改动**的片段调用一次 API。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from loguru import logger

from ..config import Settings, get_settings
from ..llm import embed_documents
from ..ingest.metadata import content_hash


class EmbeddingCache:
    """进程内 + JSON 文件两级缓存（键：文本的 sha1）。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._data: dict[str, list[float]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._data = {key: [float(value) for value in vector] for key, vector in payload.items()}
            logger.debug("嵌入缓存已加载: {} 条 ({})", len(self._data), self.path)
        except Exception as error:
            logger.warning("嵌入缓存读取失败，将忽略: {}", error)
            self._data = {}

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({key: [round(value, 6) for value in vector] for key, vector in self._data.items()}),
            encoding="utf-8",
        )
        self._dirty = False
        logger.debug("嵌入缓存已写入: {} 条", len(self._data))

    def get(self, key: str) -> list[float] | None:
        return self._data.get(key)

    def put(self, key: str, vector: Sequence[float]) -> None:
        self._data[key] = list(vector)
        self._dirty = True


def embed_texts_cached(
    texts: Iterable[str],
    settings: Settings | None = None,
    cache: EmbeddingCache | None = None,
) -> list[list[float]]:
    """带缓存的批量嵌入：命中的直接返回，未命中的批量调用 API 并写回缓存。"""
    settings = settings or get_settings()
    items = list(texts)
    if not items:
        return []

    cache = cache if cache is not None else EmbeddingCache(settings.embed_cache_path)
    keys = [content_hash(text) for text in items]

    vectors: list[list[float] | None] = [cache.get(key) for key in keys]
    missing_indexes = [index for index, vector in enumerate(vectors) if vector is None]

    if missing_indexes:
        fresh = embed_documents([items[index] for index in missing_indexes], settings)
        for index, vector in zip(missing_indexes, fresh):
            vectors[index] = vector
            cache.put(keys[index], vector)
        cache.save()
        logger.info("嵌入缓存: 命中 {} / 新增 {}", len(items) - len(missing_indexes), len(missing_indexes))
    else:
        logger.info("嵌入缓存: 全部命中（{} 条），未调用嵌入 API", len(items))

    return [vector or [] for vector in vectors]
