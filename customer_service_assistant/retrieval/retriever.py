"""检索入口：全项目唯一的对外检索函数（P2 版）。

管线（设计文档 §5.2）::

    查询改写 → ① dense 召回（HNSW/COSINE）        top 20
              → ② sparse 召回（BM25，关键词扩展）  top 20
              → ③ RRF 融合（Milvus 原生 RRFRanker） top 20
              → ④ 重排（bge-reranker-v2-m3）        top 5
              → ⑤ 阈值过滤 → 拒答判定
              → ⑥ small-to-big 回填章节全文

为什么是 RRF 而不是加权求和：dense 是余弦相似度（0~1），BM25 是词频打分（0~几十），
两者尺度不可比，加权求和需要先归一化且对分布敏感；RRF 只用排名，天然稳健。

接口签名保持稳定：``retrieve()`` / ``retrieve_with_report()`` / ``build_context()``
与 P0、P1 一致，调用方（脚本、评测、后续的 Agent 工具）无需改动。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from loguru import logger

from ..config import Settings, get_settings
from ..llm import embed_query
from ..models import RetrievedChunk
from .query_rewrite import RewriteResult, rewrite_for_retrieval
from .rerank import SiliconFlowReranker, rerank_hits
from .store import MilvusDocStore

CHILD_FILTER = 'chunk_type == "child"'
VALID_MODES = ("dense", "sparse", "hybrid")
# 命中的 child 要按 parent 去重，需多取候选，避免 top5 全落进同一章节
OVER_FETCH_FACTOR = 3
MAX_FETCH_LIMIT = 50


@dataclass(slots=True)
class RetrievalReport:
    """一次检索的完整过程信息，便于调试、打点与报告。"""

    query: str
    mode: str
    limit: int
    elapsed: float
    hits: list[RetrievedChunk] = field(default_factory=list)
    keyword_query: str = ""
    expansions: list[str] = field(default_factory=list)
    condensed: bool = False
    fetch_limit: int = 0
    candidates: int = 0
    reranked: int = 0
    filtered_out: int = 0
    threshold: float | None = None
    refusal_threshold: float | None = None
    refused: bool = False

    @property
    def top_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0

    @property
    def top_rerank_score(self) -> float | None:
        return self.hits[0].rerank_score if self.hits else None


def _fetch_candidates(
    store: MilvusDocStore,
    mode: str,
    vector: Sequence[float],
    keyword_query: str,
    fetch_limit: int,
    child_filter: str | None,
    settings: Settings,
) -> list[RetrievedChunk]:
    if mode == "dense":
        return store.search(vector, limit=fetch_limit, filter=child_filter)
    if mode == "sparse":
        return store.search_sparse(keyword_query, limit=fetch_limit, filter=child_filter)
    if mode == "hybrid":
        return store.hybrid_search(
            vector,
            keyword_query,
            limit=fetch_limit,
            filter=child_filter,
            rrf_k=settings.rrf_k,
            fusion=settings.fusion,
            dense_weight=settings.dense_weight,
            sparse_weight=settings.sparse_weight,
        )
    raise ValueError(f"不支持的检索模式：{mode}（可选 {VALID_MODES}）")


def _collapse_by_parent(hits: Sequence[RetrievedChunk], limit: int) -> list[RetrievedChunk]:
    """同一章节的多个 child 合并为一个上下文（保留分数最高的那个片段）。"""
    collapsed: dict[str, RetrievedChunk] = {}
    for hit in hits:
        collapsed.setdefault(hit.parent_id or hit.uid, hit)
    selected = list(collapsed.values())[:limit]
    for rank, hit in enumerate(selected, start=1):
        hit.rank = rank
    return selected


def _attach_parents(store: MilvusDocStore, hits: Sequence[RetrievedChunk]) -> int:
    """回填父级章节全文，返回成功回填的数量（small-to-big）。"""
    parent_ids = [hit.parent_id for hit in hits if hit.parent_id]
    if not parent_ids:
        return 0
    parents = store.get_by_ids(sorted(set(parent_ids)), output_fields=["text", "section", "section_no"])
    attached = 0
    for hit in hits:
        parent = parents.get(hit.parent_id or "")
        if parent:
            hit.parent_text = parent.get("text") or hit.text
            hit.section = hit.section or parent.get("section")
            attached += 1
    return attached


def retrieve(
    query: str,
    *,
    limit: int | None = None,
    store: MilvusDocStore | None = None,
    settings: Settings | None = None,
    expand_parent: bool = True,
    history: str | None = None,
) -> list[RetrievedChunk]:
    """对用户问题做检索，返回按相关性降序、已按章节去重的片段列表。"""
    report = retrieve_with_report(
        query,
        limit=limit,
        store=store,
        settings=settings,
        expand_parent=expand_parent,
        history=history,
    )
    return report.hits


def retrieve_with_report(
    query: str,
    *,
    limit: int | None = None,
    store: MilvusDocStore | None = None,
    settings: Settings | None = None,
    query_vector: Sequence[float] | None = None,
    mode: str | None = None,
    expand_parent: bool = True,
    history: str | None = None,
    reranker: SiliconFlowReranker | None = None,
    rerank: bool | None = None,
    apply_threshold: bool = True,
    threshold: float | None = None,
    refusal_threshold: float | None = None,
) -> RetrievalReport:
    """带过程信息的检索。

    * ``query_vector`` 传入时跳过向量化（对照实验用）；
    * ``mode`` 覆盖配置里的检索模式（dense / sparse / hybrid）；
    * ``rerank`` 为 ``None`` 时按配置决定（hybrid + ``CSA_RERANK_ENABLED``）；
    * 阈值分两级（P2 实测后的设计修正）：
      ``threshold`` 只用来剔除噪声片段（宽松，默认 0.02），
      ``refusal_threshold`` 用"最高分"决定是否拒答（严格，默认 0.23）。
      两者合一会导致"语义弱但相关的片段被误滤"——q-08 的文档说明片段只有 0.11 分。
    """
    started = time.perf_counter()
    settings = settings or get_settings()
    store = store or MilvusDocStore(settings)
    store.ensure_ready()

    mode = mode or settings.retrieval_mode
    if mode not in VALID_MODES:
        raise ValueError(f"不支持的检索模式：{mode}（可选 {VALID_MODES}）")
    limit = limit or settings.rerank_top_n

    rewrite: RewriteResult = rewrite_for_retrieval(query, history, settings)
    vector = list(query_vector) if query_vector is not None else embed_query(rewrite.query)

    # 只有 v2 schema 才有 chunk_type 字段；P0 的 collection 不做 parent / child 区分
    use_parents = expand_parent and store.has_field("chunk_type")
    child_filter = CHILD_FILTER if use_parents else None
    use_rerank = settings.rerank_enabled if rerank is None else rerank
    # 重排只在混合检索之上有意义：单路 dense 的排序问题不是重排要解决的（对照实验里可显式打开）
    use_rerank = bool(use_rerank and mode == "hybrid")

    fetch_limit = (
        settings.hybrid_fetch_limit
        if mode == "hybrid"
        else (min(limit * OVER_FETCH_FACTOR, MAX_FETCH_LIMIT) if use_parents else limit)
    )
    candidates = _fetch_candidates(store, mode, vector, rewrite.keyword_query, fetch_limit, child_filter, settings)
    fetched_candidates = len(candidates)

    reranked_count = 0
    filtered_out = 0
    applied_threshold: float | None = None
    applied_refusal: float | None = None

    if use_rerank and candidates:
        candidates = rerank_hits(
            candidates,
            rewrite.query,
            settings=settings,
            reranker=reranker,
        )
        reranked_count = len(candidates)
        applied_threshold = settings.rerank_threshold if threshold is None else threshold
        applied_refusal = settings.refusal_threshold if refusal_threshold is None else refusal_threshold
        if apply_threshold:
            # ① 先做拒答判定：最高分都不到线，说明知识库里没有可用证据
            top_score = max((hit.rerank_score or 0.0) for hit in candidates)
            if applied_refusal is not None and top_score < applied_refusal:
                filtered_out = len(candidates)
                candidates = []
                logger.debug("拒答：最高重排分 {:.4f} < 拒答阈值 {}", top_score, applied_refusal)
            else:
                # ② 再做片段过滤：只剔除噪声，保留"弱相关但可能有用"的片段
                kept = [hit for hit in candidates if (hit.rerank_score or 0.0) >= applied_threshold]
                filtered_out = len(candidates) - len(kept)
                candidates = kept

    if use_parents:
        hits = _collapse_by_parent(candidates, limit)
        expanded = _attach_parents(store, hits)
    else:
        hits = list(candidates[:limit])
        for rank, hit in enumerate(hits, start=1):
            hit.rank = rank
        expanded = 0

    report = RetrievalReport(
        query=query,
        mode=f"{mode}+rerank" if (use_rerank and reranked_count) else mode,
        limit=limit,
        elapsed=time.perf_counter() - started,
        hits=hits,
        keyword_query=rewrite.keyword_query,
        expansions=rewrite.expansions,
        condensed=rewrite.condensed,
        fetch_limit=fetch_limit,
        candidates=fetched_candidates,
        reranked=reranked_count,
        filtered_out=filtered_out,
        threshold=applied_threshold,
        refusal_threshold=applied_refusal,
        refused=not hits,
    )
    logger.debug(
        "retrieved {} hits (mode={}, candidates={}, filtered={}, parents={}) in {:.3f}s",
        len(hits),
        report.mode,
        report.candidates,
        filtered_out,
        expanded,
        report.elapsed,
    )
    return report


def build_context(hits: Sequence[RetrievedChunk]) -> str:
    """把检索结果拼成带编号与来源信息的上下文块。

    命中已回填 parent 时使用章节全文（small-to-big），否则退回片段自身；
    头部字段按有无 section 自适应，保证 P0/P1 的历史报告仍可复现。
    """
    blocks = []
    for index, hit in enumerate(hits, start=1):
        parts = [f"片段{index}", f"chunk_id={hit.seq}", f"source={hit.source}"]
        if hit.section:
            parts.append(f"section={hit.section}")
        blocks.append(f"[{' | '.join(parts)}]\n{hit.context_text}")
    return "\n\n".join(blocks)
