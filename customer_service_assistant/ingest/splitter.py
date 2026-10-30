"""结构感知切分：先切章节（parent），再切章节内小块（child）。

针对 P0 baseline 的两个问题（见 docs/baseline.md）：

1. 固定长度切分把"套餐档位 + 各档额度"这类成组信息切散 → 改为按 ``======`` 章节切 parent；
2. 属性清单型片段语义被摊薄（相似度 0.3527）→ child 携带章节标题做上下文增强嵌入
   （``embed_text = 章节标题 + 段落``），检索命中 child、回填 parent 供模型阅读。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from ..models import ChunkRecord
from .metadata import (
    PREAMBLE_TITLE,
    content_hash,
    detect_plan_tier,
    make_chunk_uid,
    parse_section_no,
)

# knowledge.txt 的章节分隔：30 个 = 单独成行
SEPARATOR_RE = re.compile(r"^={8,}$")

# child 切分优先级：段落 → 换行 → 句号 → 分号 → 空格 → 硬切
CHILD_SEPARATORS = ["\n\n", "\n", "。", "；", " ", ""]


@dataclass(slots=True)
class Section:
    """一个章节（= 一个 parent 片段）。"""

    no: int
    title: str
    text: str
    doc_id: str = ""
    source: str = ""
    doc_version: str = ""

    @property
    def heading_text(self) -> str:
        """带标题的章节全文（parent 落库/喂模型的形态）。"""
        return f"{self.title}\n{self.text}".strip()


def is_separator(line: str) -> bool:
    return bool(SEPARATOR_RE.match(line.strip()))


def parse_sections(document: Document) -> list[Section]:
    """把一篇文档解析成章节列表（首个 ``======`` 之前的内容作为"文档说明"章节）。

    knowledge.txt 的章节形态为三行一组::

        ==============================
        六、退款规则
        ==============================
        <正文>
    """
    lines = document.page_content.splitlines()
    metadata = document.metadata

    blocks: list[tuple[int, str, int]] = []  # (分隔符起始行, 标题, 正文起始行)
    index = 0
    while index + 2 < len(lines):
        if (
            is_separator(lines[index])
            and lines[index + 1].strip()
            and not is_separator(lines[index + 1])
            and is_separator(lines[index + 2])
        ):
            blocks.append((index, lines[index + 1].strip(), index + 3))
            index += 3
        else:
            index += 1

    common = {
        "doc_id": str(metadata.get("doc_id", "unknown")),
        "source": str(metadata.get("source", "unknown")),
        "doc_version": str(metadata.get("doc_version", "unknown")),
    }

    sections: list[Section] = []
    head_end = blocks[0][0] if blocks else len(lines)
    head = "\n".join(lines[:head_end]).strip()
    if head:
        sections.append(Section(no=0, title=PREAMBLE_TITLE, text=head, **common))

    for position, (_, title, content_start) in enumerate(blocks):
        content_end = blocks[position + 1][0] if position + 1 < len(blocks) else len(lines)
        body = "\n".join(lines[content_start:content_end]).strip()
        sections.append(
            Section(no=parse_section_no(title), title=title, text=body, **common)
        )

    return sections


def build_splitter(chunk_size: int = 300, chunk_overlap: int = 60) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(CHILD_SEPARATORS),
    )


def build_chunks(
    sections: Iterable[Section],
    *,
    child_chunk_size: int = 300,
    child_chunk_overlap: int = 60,
    max_text_length: int = 8192,
) -> list[ChunkRecord]:
    """把章节切成 parent + child 片段列表。"""
    splitter = build_splitter(child_chunk_size, child_chunk_overlap)
    records: list[ChunkRecord] = []

    for section in sections:
        body = section.text.strip()
        parent_text = section.heading_text[:max_text_length]
        parent_hash = content_hash(parent_text)
        parent_id = make_chunk_uid(section.doc_id, "parent", section.no, 0, parent_hash)

        records.append(
            ChunkRecord(
                uid=parent_id,
                text=parent_text,
                seq=0,
                doc_id=section.doc_id,
                source=section.source,
                doc_version=section.doc_version,
                content_hash=parent_hash,
                chunk_type="parent",
                parent_id="",
                section=section.title,
                section_no=section.no,
                plan_tier=detect_plan_tier(body),
                embed_text=parent_text,
            )
        )

        for seq, piece in enumerate(splitter.split_text(body), start=1):
            piece = piece.strip()
            if not piece:
                continue
            piece_hash = content_hash(piece)
            records.append(
                ChunkRecord(
                    uid=make_chunk_uid(section.doc_id, "child", section.no, seq, piece_hash),
                    text=piece,
                    seq=seq,
                    doc_id=section.doc_id,
                    source=section.source,
                    doc_version=section.doc_version,
                    content_hash=piece_hash,
                    chunk_type="child",
                    parent_id=parent_id,
                    section=section.title,
                    section_no=section.no,
                    plan_tier=detect_plan_tier(piece),
                    # 上下文增强嵌入：把章节标题拼进待嵌入文本，但不落库
                    embed_text=f"{section.title}\n{piece}",
                )
            )

    children = [record for record in records if record.chunk_type == "child"]
    if not children:
        logger.warning("切分结果没有任何 child 片段，请检查章节分隔符是否与文档一致")
    return records
