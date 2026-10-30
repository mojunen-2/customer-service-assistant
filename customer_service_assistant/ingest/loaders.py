"""文档加载：发现知识库文件并计算内容指纹。

P0 只加载 ``knowledge.txt``（与 baseline 保持一致，保证对照公平）；``data/raw/`` 下的
额外文档会自动纳入，用于 P1 之后扩展多文档场景。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from loguru import logger

from ..config import Settings


def sha1_of_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha1_of_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_files(settings: Settings) -> list[Path]:
    """返回待索引文件列表（主知识库 + data/raw 下的额外文档）。"""
    files: list[Path] = []

    primary = settings.knowledge_file_path
    if primary.exists():
        files.append(primary)
    else:
        logger.warning("主知识库文件不存在：{}", primary)

    if settings.extra_docs_dir.exists():
        for pattern in settings.doc_globs:
            files.extend(sorted(settings.extra_docs_dir.glob(pattern)))

    # 去重并保持稳定顺序（保证 uid 稳定 → 幂等）
    unique: dict[str, Path] = {}
    for path in files:
        unique[str(path.resolve())] = path
    return [unique[key] for key in sorted(unique)]


def load_documents(settings: Settings) -> tuple[list[Document], list[dict]]:
    """加载文档，返回 (documents, 加载信息列表)。

    每个 Document 的 metadata 至少包含：
    source / file_path / doc_id / file_hash / mtime / doc_version
    """
    documents: list[Document] = []
    infos: list[dict] = []

    for path in discover_files(settings):
        raw_documents = TextLoader(str(path), encoding="utf-8").load()
        file_hash = sha1_of_file(path)
        doc_id = file_hash[:12]
        try:
            source = str(path.relative_to(settings.knowledge_dir))
        except ValueError:
            source = str(path)

        for document in raw_documents:
            document.metadata.update(
                {
                    "source": source,
                    "file_path": str(path),
                    "doc_id": doc_id,
                    "file_hash": file_hash,
                    "doc_version": settings.doc_version,
                }
            )
            documents.append(document)

        infos.append(
            {
                "path": str(path),
                "source": source,
                "doc_id": doc_id,
                "file_hash": file_hash,
                "chars": sum(len(document.page_content) for document in raw_documents),
                "doc_version": settings.doc_version,
            }
        )
        logger.info("loaded {} ({} chars, doc_id={})", path.name, infos[-1]["chars"], doc_id)

    return documents, infos
