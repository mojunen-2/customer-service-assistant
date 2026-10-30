"""检索质量评估：正确片段排名、Recall@5、nDCG@5、Section@5。

P1 建立的验收工具，P2 扩展了模式 / 重排 / 阈值选项，用于产出消融对照数据。

用法::

    # 当前配置（hybrid + 重排）
    python scripts/rank_eval.py --label "P2 hybrid+rerank" --out docs/rank_eval_p2.md

    # 消融：纯向量 / BM25 / 混合（不重排）
    python scripts/rank_eval.py --mode dense  --no-rerank --label "dense"  --out docs/rank_eval_dense.md
    python scripts/rank_eval.py --mode sparse --no-rerank --label "bm25"   --out docs/rank_eval_bm25.md
    python scripts/rank_eval.py --mode hybrid --no-rerank --label "hybrid" --out docs/rank_eval_hybrid.md

    # 与 P0 / P1 对齐的核心 8 题
    python scripts/rank_eval.py --ids q-01,q-02,q-03,q-04,q-05,q-06,q-07,q-08 --label "核心 8 题" \
        --collection customer_service_docs --mode dense --no-rerank --no-expand-parent --out docs/rank_eval_p0.md

指标定义：

* ``gold 片段排名``：全量 child 排序中首个包含 gold 串的名次（不重排，反映召回能力）；
* ``Recall@5(child)``：top5 命中片段里出现 gold 串；
* ``Recall@5(context)``：top5 送给模型的上下文块里出现 gold 串（含 small-to-big 回填）；
* ``nDCG@5``：以上下文是否含 gold 为相关性（二值），衡量正确章节排得够不够高；
* ``Section@5``：top5 所属章节是否包含期望章节。
"""

import _bootstrap  # noqa: F401  必须最先导入

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from customer_service_assistant.config import Settings, get_settings
from customer_service_assistant.llm import embed_query, setup_tracing
from customer_service_assistant.logging_setup import setup_logging
from customer_service_assistant.retrieval.query_rewrite import rewrite_for_retrieval
from customer_service_assistant.retrieval.retriever import CHILD_FILTER, retrieve_with_report
from customer_service_assistant.retrieval.store import MilvusDocStore

PROBE_SET_PATH = Path("data/eval/probe_questions.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检索质量评估（探针集）")
    parser.add_argument("--probe-set", default=str(PROBE_SET_PATH), help="探针集 JSON 路径")
    parser.add_argument("--limit", type=int, default=5, help="评估的 top-k")
    parser.add_argument("--collection", default=None, help="目标 collection（默认取配置值）")
    parser.add_argument("--mode", choices=["dense", "sparse", "hybrid"], default=None, help="检索模式")
    parser.add_argument("--rerank", dest="rerank", action="store_true", default=None, help="强制开启重排")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false", help="关闭重排")
    parser.add_argument("--threshold", type=float, default=None, help="拒答阈值（默认取配置）")
    parser.add_argument("--refusal-threshold", type=float, default=None, help="拒答阈值（基于最高分）")
    parser.add_argument("--no-threshold", action="store_true", help="关闭阈值过滤（评估召回上限）")
    parser.add_argument("--ids", default=None, help="只评估指定题号，逗号分隔（如 q-01,q-02）")
    parser.add_argument("--no-expand-parent", action="store_true", help="关闭 small-to-big 回填")
    parser.add_argument("--label", default="current", help="报告标题里的配置标签")
    parser.add_argument("--out", default=None, help="输出 markdown 报告路径（相对项目根目录）")
    return parser.parse_args()


def load_probe_set(settings: Settings, path: str) -> list[dict]:
    probe_path = Path(path)
    if not probe_path.is_absolute():
        probe_path = settings.project_root / probe_path
    return json.loads(probe_path.read_text(encoding="utf-8"))["questions"]


def ndcg_at_k(relevances: list[int], k: int) -> float:
    """二值相关性的 nDCG@k。理想排序 = 所有相关项排在最前。"""
    dcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances[:k]))
    ideal_count = min(sum(relevances), k)
    if ideal_count == 0:
        return 0.0
    ideal = sum(1 / math.log2(index + 2) for index in range(ideal_count))
    return dcg / ideal


def evaluate(
    settings: Settings,
    questions: list[dict],
    *,
    limit: int,
    expand_parent: bool,
    mode: str | None,
    rerank: bool | None,
    threshold: float | None,
    refusal_threshold: float | None,
    apply_threshold: bool,
) -> dict:
    store = MilvusDocStore(settings)
    store.ensure_ready()

    has_chunk_type = store.has_field("chunk_type")
    child_filter = CHILD_FILTER if has_chunk_type else None
    total = store.count_entities(filter=child_filter)
    effective_mode = mode or settings.retrieval_mode
    logger.info(
        "collection={} schema={} 可检索片段={} mode={} rerank={} expand_parent={}",
        settings.milvus_collection,
        "v2" if has_chunk_type else "p0",
        total,
        effective_mode,
        settings.rerank_enabled if rerank is None else rerank,
        expand_parent,
    )

    rows: list[dict] = []
    for item in questions:
        rewrite = rewrite_for_retrieval(item["question"], settings=settings)
        vector = embed_query(rewrite.query)

        if effective_mode == "hybrid":
            full_hits = store.hybrid_search(
                vector,
                rewrite.keyword_query,
                limit=total,
                filter=child_filter,
                rrf_k=settings.rrf_k,
                fusion=settings.fusion,
                dense_weight=settings.dense_weight,
                sparse_weight=settings.sparse_weight,
            )
        elif effective_mode == "sparse":
            full_hits = store.search_sparse(rewrite.keyword_query, limit=total, filter=child_filter)
        else:
            full_hits = store.search(vector, limit=total, filter=child_filter)
        gold_child_rank = next((hit.rank for hit in full_hits if item["gold"] in hit.text), None)

        report = retrieve_with_report(
            item["question"],
            limit=limit,
            settings=settings,
            store=store,
            query_vector=vector,
            mode=effective_mode,
            expand_parent=expand_parent,
            rerank=rerank,
            apply_threshold=apply_threshold,
            threshold=threshold,
            refusal_threshold=refusal_threshold,
        )
        hit5_child = any(item["gold"] in hit.text for hit in report.hits)
        hit5_context = any(item["gold"] in hit.context_text for hit in report.hits)
        relevances = [1 if item["gold"] in hit.context_text else 0 for hit in report.hits]
        sections = {hit.section for hit in report.hits if hit.section}
        section_hit = item.get("expected_section") in sections if item.get("expected_section") else None

        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "gold": item["gold"],
                "expected_section": item.get("expected_section"),
                "gold_child_rank": gold_child_rank,
                "top_rerank_score": round(report.hits[0].rerank_score, 4) if report.hits and report.hits[0].rerank_score else None,
                "top1_score": round(report.top_score, 4),
                "hit5_child": hit5_child,
                "hit5_context": hit5_context,
                "ndcg5": round(ndcg_at_k(relevances, limit), 4),
                "section_hit": section_hit,
                "retrieved_sections": sorted(sections),
                "refused": report.refused,
                "elapsed": round(report.elapsed, 3),
            }
        )
        logger.info(
            "{} gold_rank={} top5_child={} top5_context={} ndcg@{}={} 拒答={}",
            item["id"],
            gold_child_rank,
            hit5_child,
            hit5_context,
            limit,
            rows[-1]["ndcg5"],
            report.refused,
        )

    count = len(rows)
    ranks = [row["gold_child_rank"] for row in rows if row["gold_child_rank"]]
    summary = {
        "collection": settings.milvus_collection,
        "mode": effective_mode,
        "rerank": settings.rerank_enabled if rerank is None else rerank,
        "expand_parent": expand_parent,
        "threshold": threshold if apply_threshold else None,
        "refusal_threshold": refusal_threshold if apply_threshold else None,
        "top_k": limit,
        "questions": count,
        "child_recall": sum(row["hit5_child"] for row in rows),
        "context_recall": sum(row["hit5_context"] for row in rows),
        "section_hits": sum(1 for row in rows if row["section_hit"]),
        "section_evaluable": sum(1 for row in rows if row["section_hit"] is not None),
        "mean_ndcg": round(sum(row["ndcg5"] for row in rows) / count, 4) if count else 0.0,
        "mean_gold_rank": round(sum(ranks) / len(ranks), 1) if ranks else None,
        "refused": sum(row["refused"] for row in rows),
    }
    return {"summary": summary, "rows": rows}


def render_markdown(result: dict, label: str) -> str:
    summary = result["summary"]
    questions = summary["questions"] or 1
    lines = [
        f"# 检索质量评估（{label}）",
        "",
        f"- 生成时间：{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        f"- collection：`{summary['collection']}` ｜ 模式：{summary['mode']} ｜ 重排：{summary['rerank']} "
        f"｜ small-to-big：{summary['expand_parent']} ｜ 片段阈值：{summary['threshold']} "
        f"｜ 拒答阈值：{summary['refusal_threshold']} ｜ top-k：{summary['top_k']}",
        "",
        "## 汇总",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| Recall@5（命中片段） | {summary['child_recall']}/{questions} = {summary['child_recall'] / questions:.1%} |",
        f"| Recall@5（上下文） | {summary['context_recall']}/{questions} = {summary['context_recall'] / questions:.1%} |",
        f"| nDCG@5 均值 | {summary['mean_ndcg']} |",
        f"| Section@5（命中期望章节） | {summary['section_hits']}/{summary['section_evaluable'] or questions} |",
        f"| gold 片段平均排名 | {summary['mean_gold_rank']} / 全量 |",
        f"| 最终拒答题数 | {summary['refused']}/{questions} |",
        "",
        "## 逐题明细",
        "",
        "| 题号 | 类别 | gold 排名 | 重排分数 | 命中片段@5 | 命中上下文@5 | nDCG@5 | 命中章节 | 检索到的章节 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['id']} | {row['category']} | {row['gold_child_rank']} | {row['top_rerank_score']} | "
            f"{'是' if row['hit5_child'] else '否'} | {'是' if row['hit5_context'] else '否'} | {row['ndcg5']} | "
            f"{'是' if row['section_hit'] else ('否' if row['section_hit'] is False else '-')} | "
            f"{' / '.join(row['retrieved_sections']) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    settings = get_settings()
    if args.collection:
        settings = settings.model_copy(update={"milvus_collection": args.collection})
    setup_logging(settings.log_level, settings.log_dir)
    setup_tracing(settings)

    questions = load_probe_set(settings, args.probe_set)
    if args.ids:
        wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
        questions = [item for item in questions if item["id"] in wanted]
        if not questions:
            logger.error("--ids 没有匹配到任何题目：{}", args.ids)
            return 1
    logger.info("探针集: {} 题", len(questions))

    result = evaluate(
        settings,
        questions,
        limit=args.limit,
        expand_parent=not args.no_expand_parent,
        mode=args.mode,
        rerank=args.rerank,
        threshold=args.threshold,
        refusal_threshold=args.refusal_threshold,
        apply_threshold=not args.no_threshold,
    )
    summary = result["summary"]
    print(
        f"\n[{args.label}] mode={summary['mode']} rerank={summary['rerank']} "
        f"Recall@5(child)={summary['child_recall']}/{summary['questions']} "
        f"Recall@5(context)={summary['context_recall']}/{summary['questions']} "
        f"nDCG@5={summary['mean_ndcg']} Section@5={summary['section_hits']}/{summary['section_evaluable'] or summary['questions']} "
        f"平均 gold 排名={summary['mean_gold_rank']} 拒答={summary['refused']}"
    )

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = settings.project_root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render_markdown(result, args.label), encoding="utf-8")
        logger.info("报告已写入 {}", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
