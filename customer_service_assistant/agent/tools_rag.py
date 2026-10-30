"""检索工具：把 P2 打磨好的检索层封装成 Agent 可调用的工具。

要点：工具只是**薄封装**——内部仍走 ``retrieve_with_report()``（全项目唯一检索入口），
因此评测脚本测的就是 Agent 实际用的那条链路。
"""

from __future__ import annotations

from typing import Any

from langchain.tools import tool
from loguru import logger

from ..retrieval.retriever import retrieve_with_report
from .schemas import RAG_TOOL


def _serialize_hit(hit: Any) -> dict:
    return {
        "chunk_id": hit.seq,
        "section": hit.section,
        "score": round(hit.score, 4),
        "rerank_score": round(hit.rerank_score, 4) if hit.rerank_score is not None else None,
        # text 是命中的原文片段（用于引用与逐字校验），context 是所属章节全文（用于完整作答）
        "text": hit.text,
        "context": hit.context_text,
    }


@tool(RAG_TOOL)
def search_knowledge_base(query: str, top_k: int = 5, section: str | None = None) -> dict:
    """检索 客服助手客服知识库，返回相关章节。

    query：用用户的原话或其中的关键词检索，不要改写成很长的句子。
    top_k：返回的章节数量，默认 5。
    section：可选，限定某个章节（例如「六、退款规则」）；不确定就不要填。

    返回 hits：每条的 text 是命中的原文片段、context 是所属章节全文、section 是章节名。
    refused=true 表示知识库没有可用依据（证据分低于拒答阈值），此时**不要编造答案**，
    应告知用户无法确认并建议转人工。
    """
    try:
        report = retrieve_with_report(query, limit=top_k)
    except Exception as error:  # 工具失败要让模型知道并降级，而不是让整轮崩掉
        logger.exception("search_knowledge_base 失败")
        return {"error": f"检索失败：{error}", "refused": True, "hits": []}

    hits = [_serialize_hit(hit) for hit in report.hits]
    if section:
        filtered = [hit for hit in hits if section in (hit.get("section") or "")]
        if filtered:
            hits = filtered
    return {
        "refused": report.refused,
        "query": query,
        "hits": hits,
        "note": (
            "知识库没有可用依据（证据分低于拒答阈值）：不要编造答案，"
            "请如实告知用户当前无法确认，并把 need_human 设为 true 建议转人工"
            if report.refused
            else "请仅依据这些片段作答，引用时必须逐字抄写 text/context 里的原句"
        ),
    }


RAG_TOOLS = [search_knowledge_base]
