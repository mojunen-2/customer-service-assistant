"""离线索引编排：加载 → 结构感知切分 → 向量化 → 幂等写入 → 清理失效文档 → 写清单。"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

from langchain_core.documents import Document
from loguru import logger

from ..config import Settings, get_settings
from ..models import ChunkRecord
from ..retrieval.embedder import embed_texts_cached
from ..retrieval.store import MilvusDocStore
from .loaders import load_documents
from .splitter import Section, build_chunks, parse_sections


@dataclass(slots=True)
class IngestOutcome:
    documents: list[Document] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    records: list[ChunkRecord] = field(default_factory=list)
    doc_infos: list[dict] = field(default_factory=list)
    visible_before: int = 0
    visible_after: int = 0
    removed_doc_ids: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    manifest: dict = field(default_factory=dict)

    @property
    def parents(self) -> list[ChunkRecord]:
        return [record for record in self.records if record.chunk_type == "parent"]

    @property
    def children(self) -> list[ChunkRecord]:
        return [record for record in self.records if record.chunk_type == "child"]


def build_records(documents: list[Document], settings: Settings) -> tuple[list[ChunkRecord], list[Section]]:
    """把文档切分成 parent + child 片段。"""
    sections: list[Section] = []
    for document in documents:
        sections.extend(parse_sections(document))

    records = build_chunks(
        sections,
        child_chunk_size=settings.child_chunk_size,
        child_chunk_overlap=settings.child_chunk_overlap,
        max_text_length=settings.max_text_length,
    )
    return records, sections


def log_statistics(outcome: IngestOutcome, settings: Settings) -> None:
    records = outcome.records
    lengths = [len(record.text) for record in records]
    by_section = Counter(record.section for record in records if record.chunk_type == "child")
    missing_section = [record.uid for record in records if not record.section]

    logger.info(
        "切分完成: 文档 {} 个 / 章节 {} 个 / parent {} / child {} / 平均 {:.1f} 字",
        len(outcome.documents),
        len(outcome.sections),
        len(outcome.parents),
        len(outcome.children),
        sum(lengths) / len(lengths) if lengths else 0,
    )
    logger.info("按章节分布（child 数）:")
    for section_title, count in sorted(by_section.items()):
        logger.info("  - {}: {}", section_title, count)
    if missing_section:
        logger.warning("存在 section 为空的片段 {} 个: {}", len(missing_section), missing_section[:5])
    else:
        logger.info("section 完整性检查: 通过（所有片段都带章节信息）")

    numbered = sorted({section.no for section in outcome.sections if section.no > 0})
    logger.info("识别到的编号章节: {} 个 -> {}", len(numbered), numbered)
    child_lengths = [len(record.text) for record in outcome.children]
    if child_lengths:
        logger.info(
            "child 长度: 平均 {:.1f} / 最短 {} / 最长 {}（child_chunk_size={}）",
            sum(child_lengths) / len(child_lengths),
            min(child_lengths),
            max(child_lengths),
            settings.child_chunk_size,
        )


def write_manifest(settings: Settings, outcome: IngestOutcome) -> str:
    manifest = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "collection": f"{settings.milvus_db}.{settings.milvus_collection}",
        "milvus_uri": settings.milvus_uri,
        "schema_version": "v2",
        "doc_version": settings.doc_version,
        "splitter": "structure_aware(section=parent, paragraph=child)",
        "child_chunk_size": settings.child_chunk_size,
        "child_chunk_overlap": settings.child_chunk_overlap,
        "embed_model": settings.embed_model,
        "embed_dim": settings.embed_dim,
        "documents": outcome.doc_infos,
        "sections": [
            {
                "no": section.no,
                "title": section.title,
                "chars": len(section.text),
                "children": len([r for r in outcome.children if r.section == section.title]),
            }
            for section in outcome.sections
        ],
        "total_records": len(outcome.records),
        "parent_count": len(outcome.parents),
        "child_count": len(outcome.children),
        "visible_before": outcome.visible_before,
        "visible_after": outcome.visible_after,
        "removed_doc_ids": outcome.removed_doc_ids,
        "elapsed_seconds": round(outcome.elapsed, 2),
    }
    settings.index_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    settings.index_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    outcome.manifest = manifest
    return str(settings.index_manifest_path)


def verify_after_ingest(outcome: IngestOutcome) -> bool:
    """幂等性验收：可见记录数应等于"本文档集合的片段数"（清理失效文档后）。"""
    expected = len(outcome.records)
    ok = outcome.visible_after == expected
    if ok:
        logger.info(
            "幂等性检查: 通过（可见记录数 {} == 本次片段数 {}，重复执行不会产生重复数据）",
            outcome.visible_after,
            expected,
        )
    else:
        logger.warning(
            "幂等性检查: 异常（可见记录数 {} != 本次片段数 {}）",
            outcome.visible_after,
            expected,
        )
    return ok


def run_ingest(
    settings: Settings | None = None,
    *,
    dry_run: bool = False,
    compact: bool = True,
) -> IngestOutcome:
    """执行一次完整索引。幂等：同一份文档重复执行不会产生重复数据。"""
    settings = settings or get_settings()
    outcome = IngestOutcome()
    started = time.perf_counter()

    store = MilvusDocStore(settings)
    store.ensure_ready()
    outcome.visible_before = store.count_entities()
    logger.info("写入前可见记录数: {}", outcome.visible_before)

    outcome.documents, outcome.doc_infos = load_documents(settings)
    if not outcome.documents:
        raise RuntimeError("没有加载到任何文档，请检查 CSA_KNOWLEDGE_DIR 与 knowledge.txt")

    outcome.records, outcome.sections = build_records(outcome.documents, settings)
    log_statistics(outcome, settings)

    if dry_run:
        outcome.elapsed = time.perf_counter() - started
        logger.info("--dry-run 已开启，跳过向量化与写入")
        return outcome

    vectors = embed_texts_cached([record.text_for_embedding for record in outcome.records], settings)
    store.upsert_records(outcome.records, vectors)

    # 清理已经不在知识库中的文档（例如从 data/raw/ 删除的文件）
    current_doc_ids = {record.doc_id for record in outcome.records}
    stale_doc_ids = sorted(store.list_doc_ids() - current_doc_ids)
    for doc_id in stale_doc_ids:
        removed = store.delete_by_doc_id(doc_id)
        logger.info("清理失效文档 doc_id={} （删除 {} 条）", doc_id, removed)
    outcome.removed_doc_ids = stale_doc_ids

    store.flush()
    if stale_doc_ids and compact:
        store.compact()

    outcome.visible_after = store.count_entities()
    outcome.elapsed = time.perf_counter() - started
    verify_after_ingest(outcome)

    manifest_path = write_manifest(settings, outcome)
    logger.info("索引清单已写入 {}", manifest_path)
    logger.info("总耗时 {:.2f}s", outcome.elapsed)
    return outcome
