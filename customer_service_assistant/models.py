"""领域模型：跨层传递的轻量数据结构。

P0 只定义检索相关结构；P3 会增加 Answer / Citation（结构化输出）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ChunkRecord:
    """一个待入库的知识片段。

    P1 起区分 parent（章节全文，用于喂给模型）与 child（章节内的小块，用于检索命中）：
    ``embed_text`` 是实际送进嵌入模型的文本（带章节标题的上下文增强），默认等于 ``text``；
    它不落库（Milvus schema 里没有这个字段）。
    """

    uid: str
    text: str
    seq: int
    doc_id: str
    source: str
    doc_version: str
    content_hash: str
    chunk_type: str = "child"
    parent_id: str = ""
    section: str = ""
    section_no: int = 0
    plan_tier: str = ""
    embed_text: str | None = None

    def to_milvus_row(self, vector: list[float]) -> dict[str, Any]:
        """转换成 Milvus 写入行。

        注意：``sparse`` 字段由服务端 BM25 Function 自动生成，**不能**在写入时提供。
        """
        return {
            "id": self.uid,
            "vector": vector,
            "text": self.text,
            "seq": self.seq,
            "doc_id": self.doc_id,
            "source": self.source,
            "doc_version": self.doc_version,
            "content_hash": self.content_hash,
            "chunk_type": self.chunk_type,
            "parent_id": self.parent_id,
            "section": self.section,
            "section_no": self.section_no,
            "plan_tier": self.plan_tier,
        }

    @property
    def text_for_embedding(self) -> str:
        return self.embed_text or self.text


@dataclass(slots=True)
class RetrievedChunk:
    """一条检索结果。"""

    uid: str
    text: str
    score: float
    rank: int
    seq: int | None = None
    source: str | None = None
    doc_version: str | None = None
    section: str | None = None
    section_no: int | None = None
    plan_tier: str | None = None
    chunk_type: str | None = None
    parent_id: str | None = None
    parent_text: str | None = None
    rerank_score: float | None = None
    # 记录该片段是"稠密 / 稀疏 / 融合"第几路召回的，便于调试（P2）
    dense_rank: int | None = None
    sparse_rank: int | None = None

    @classmethod
    def from_hit(cls, hit: dict[str, Any], rank: int) -> "RetrievedChunk":
        entity = hit.get("entity") or {}
        return cls(
            uid=str(hit.get("id")),
            text=entity.get("text", ""),
            score=float(hit.get("distance", 0.0)),
            rank=rank,
            seq=entity.get("seq"),
            source=entity.get("source"),
            doc_version=entity.get("doc_version"),
            section=entity.get("section"),
            section_no=entity.get("section_no"),
            plan_tier=entity.get("plan_tier"),
            chunk_type=entity.get("chunk_type"),
            parent_id=entity.get("parent_id"),
        )

    @property
    def context_text(self) -> str:
        """喂给模型的文本：优先用回填的父级章节全文，缺失时退回片段自身。"""
        return self.parent_text or self.text

    @property
    def filter_score(self) -> float:
        """用于拒答阈值判断的分数：重排开启时用交叉编码器分数，否则用召回分数。"""
        return self.rerank_score if self.rerank_score is not None else self.score
