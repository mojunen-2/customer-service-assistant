"""P0 baseline 切分器：完全复刻原 notebook 的切分参数。

保留它的目的不是"好用"，而是作为消融实验 A 组的对照组——只有切分参数与 baseline
完全一致，后续 P1 的结构感知切分带来的增益才能被干净地量化。
P1 会用 ``splitter.py`` + ``metadata.py`` 取代本模块。
"""

from __future__ import annotations

import hashlib
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from ..models import ChunkRecord

# 与 05-案例 notebook 完全一致
BASELINE_SEPARATORS = [
    "\n==============================\n",
    "\n\n",
    "\n",
    "。",
    " ",
    "",
]


def make_chunk_uid(doc_id: str, seq: int, content_hash: str) -> str:
    """确定性主键：同一文档、同一切分参数下重复 ingest 必然得到同一个 id。"""
    return hashlib.sha1(f"{doc_id}|{seq}|{content_hash}".encode("utf-8")).hexdigest()[:16]


def build_splitter(chunk_size: int = 200, chunk_overlap: int = 80) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(BASELINE_SEPARATORS),
    )


def split_baseline(
    documents: Iterable[Document],
    *,
    chunk_size: int = 200,
    chunk_overlap: int = 80,
) -> list[ChunkRecord]:
    """按 baseline 参数切分，产出带确定性 uid 的片段列表。"""
    splitter = build_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    records: list[ChunkRecord] = []

    for document in documents:
        metadata = document.metadata
        pieces = splitter.split_text(document.page_content)
        doc_id = str(metadata.get("doc_id", "unknown"))
        source = str(metadata.get("source", "unknown"))
        doc_version = str(metadata.get("doc_version", "unknown"))

        for seq, text in enumerate(pieces):
            content_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
            records.append(
                ChunkRecord(
                    uid=make_chunk_uid(doc_id, seq, content_hash),
                    text=text,
                    seq=seq,
                    doc_id=doc_id,
                    source=source,
                    doc_version=doc_version,
                    content_hash=content_hash,
                )
            )

    uids = [record.uid for record in records]
    if len(set(uids)) != len(uids):
        logger.warning("切分结果存在重复 uid：{} / {}", len(uids), len(set(uids)))
    return records
