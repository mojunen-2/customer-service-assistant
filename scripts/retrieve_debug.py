"""检索管线逐阶段调试：纯向量 / BM25 / 混合(RRF) / 混合+重排 的命中对比。

用法::

    python scripts/retrieve_debug.py --id q-02              # 用探针集里的题（自带 gold 串）
    python scripts/retrieve_debug.py --question "..." --gold "..."
    python scripts/retrieve_debug.py --all                  # 对全部探针题输出汇总表

这是 P2 DoD 要求的工具：必须能直观看出"纯向量 vs 混合 vs 混合+重排"的差异。
"""

import _bootstrap  # noqa: F401  必须最先导入

import argparse
import json
from pathlib import Path

from customer_service_assistant.config import Settings, get_settings
from customer_service_assistant.llm import embed_query, setup_tracing
from customer_service_assistant.logging_setup import setup_logging
from customer_service_assistant.models import RetrievedChunk
from customer_service_assistant.retrieval.query_rewrite import rewrite_for_retrieval
from customer_service_assistant.retrieval.rerank import rerank_hits
from customer_service_assistant.retrieval.retriever import CHILD_FILTER, retrieve_with_report
from customer_service_assistant.retrieval.store import MilvusDocStore

PROBE_SET_PATH = Path("data/eval/probe_questions.json")
PREVIEW_CHARS = 46


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检索管线逐阶段调试")
    parser.add_argument("--id", default=None, help="探针集里的题号（如 q-02）")
    parser.add_argument("--question", default=None, help="直接指定问题")
    parser.add_argument("--gold", default="", help="正确片段中应出现的关键串（可选）")
    parser.add_argument("--all", action="store_true", help="对全部探针题输出汇总表")
    parser.add_argument("--top", type=int, default=60, help="每阶段打印前 N 条（默认全部）")
    parser.add_argument("--limit", type=int, default=5, help="最终上下文条数")
    parser.add_argument("--threshold", type=float, default=None, help="覆盖拒答阈值")
    parser.add_argument("--probe-set", default=str(PROBE_SET_PATH), help="探针集路径")
    return parser.parse_args()


def load_probe_set(settings: Settings, path: str) -> list[dict]:
    probe_path = Path(path)
    if not probe_path.is_absolute():
        probe_path = settings.project_root / probe_path
    return json.loads(probe_path.read_text(encoding="utf-8"))["questions"]


def preview(text: str) -> str:
    return text[:PREVIEW_CHARS].replace("\n", " ")


def print_stage(title: str, hits: list[RetrievedChunk], gold: str, top: int, score_label: str = "score") -> None:
    gold_rank = next((hit.rank for hit in hits if gold and gold in hit.text), None)
    mark = f"，gold 排名={gold_rank}" if gold else ""
    print(f"\n--- {title}（命中 {len(hits)} 条{mark}）---")
    for position, hit in enumerate(hits[:top], start=1):
        score = hit.rerank_score if hit.rerank_score is not None else hit.score
        flag = " ★gold" if gold and gold in hit.text else ""
        print(
            f"  {position:>2}. {score_label}={score:.4f} "
            f"section={hit.section or '-'} :: {preview(hit.text)}{flag}"
        )


def inspect_one(settings: Settings, item: dict, args: argparse.Namespace, store: MilvusDocStore) -> dict:
    question = item["question"]
    gold = item.get("gold", "")
    print("=" * 100)
    print(f"[{item.get('id', '-')}] {question}" + (f"    gold={gold!r}" if gold else ""))

    rewrite = rewrite_for_retrieval(question, settings=settings)
    if rewrite.expansions or rewrite.condensed:
        print(f"查询改写: dense={rewrite.query!r}")
        print(f"          bm25 ={rewrite.keyword_query!r}（扩展词 {rewrite.expansions}）")

    vector = embed_query(rewrite.query)
    child_filter = CHILD_FILTER if store.has_field("chunk_type") else None
    total = store.count_entities(filter=child_filter)
    fetch = settings.hybrid_fetch_limit

    dense_hits = store.search(vector, limit=total, filter=child_filter)
    sparse_hits = store.search_sparse(rewrite.keyword_query, limit=total, filter=child_filter)
    hybrid_hits = store.hybrid_search(
        vector, rewrite.keyword_query, limit=total, filter=child_filter, rrf_k=settings.rrf_k
    )
    rerank_candidates = rerank_hits(hybrid_hits[:fetch], rewrite.query, settings=settings)

    print_stage(f"① 纯向量 dense（全量 {total} 条）", dense_hits, gold, args.top)
    print_stage(f"② BM25 全文（全量 {total} 条）", sparse_hits, gold, args.top, score_label="bm25")
    print_stage(f"③ 混合检索 RRF（全量 {total} 条）", hybrid_hits, gold, args.top, score_label="rrf")
    print_stage(f"④ 混合 + 重排（候选 {len(rerank_candidates)} 条）", rerank_candidates, gold, args.top, score_label="rerank")

    threshold = settings.rerank_threshold if args.threshold is None else args.threshold
    report = retrieve_with_report(
        question,
        limit=args.limit,
        settings=settings,
        store=store,
        threshold=threshold,
    )
    print(
        f"\n--- ⑤ 最终管线（{report.mode} / 阈值 {report.threshold}）---\n"
        f"  候选 {report.candidates} 条 → 重排 {report.reranked} 条 → 阈值过滤掉 {report.filtered_out} 条 "
        f"→ 上下文 {len(report.hits)} 段（{report.elapsed:.2f}s）"
    )
    for hit in report.hits:
        flag = " ★含 gold" if gold and gold in hit.context_text else ""
        print(
            f"  {hit.rank}. rerank={hit.rerank_score:.4f} section={hit.section} "
            f"parent={'已回填' if hit.parent_text else '无'}{flag}"
        )
    print(f"  拒答判定: {'是（上下文为空）' if report.refused else '否'}")

    def rank_of(hits: list[RetrievedChunk]) -> int | None:
        return next((hit.rank for hit in hits if gold and gold in hit.text), None)

    return {
        "id": item.get("id", "-"),
        "question": question,
        "dense": rank_of(dense_hits),
        "bm25": rank_of(sparse_hits),
        "hybrid": rank_of(hybrid_hits),
        "rerank_top5": rank_of(rerank_candidates[: args.limit]),
        "context_hit": bool(gold) and any(gold in hit.context_text for hit in report.hits),
        "refused": report.refused,
        "top_rerank": rerank_candidates[0].rerank_score if rerank_candidates else None,
        "threshold": threshold,
    }


def main() -> int:
    args = parse_args()
    settings = get_settings()
    setup_logging("WARNING", None)
    setup_tracing(settings)

    store = MilvusDocStore(settings)
    store.ensure_ready()
    print(f"collection={settings.milvus_collection}（{store.count_entities()} 条可见记录）")

    if args.question:
        items = [{"id": args.id or "-", "question": args.question, "gold": args.gold}]
    else:
        items = load_probe_set(settings, args.probe_set)
        if args.id:
            items = [item for item in items if item["id"] == args.id]
            if not items:
                print(f"探针集里没有题号 {args.id}")
                return 1
        if not args.all and not args.id:
            print("请指定 --id 或 --question，或使用 --all 输出全部题的汇总")
            return 1

    summaries = [inspect_one(settings, item, args, store) for item in items]

    if len(summaries) > 1 or args.all:
        print("\n" + "=" * 100)
        print("汇总：gold 片段在各阶段的排名（- 表示未命中）")
        print(f"{'题号':<6}{'dense':<8}{'bm25':<8}{'hybrid':<8}{'rerank top5':<13}{'上下文命中':<11}{'拒答'}")
        for row in summaries:
            print(
                f"{row['id']:<6}{str(row['dense']):<8}{str(row['bm25']):<8}{str(row['hybrid']):<8}"
                f"{str(row['rerank_top5']):<13}{'是' if row['context_hit'] else '否':<11}"
                f"{'是' if row['refused'] else '否'}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
