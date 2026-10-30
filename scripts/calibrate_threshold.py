"""拒答阈值标定：扫描重排分数阈值，产出曲线与推荐值。

思路：拒答本质上是一个二分类决策——"检索到的证据够不够回答"。

* **正样本**：知识库能答的检索类问题（P2 用 16 题探针集，P5 用 100 题冻结集里的 fact / boundary）；
* **负样本**：超范围问题（P2 用 8 题，P5 扩充到 15 题，含"靠近业务但未覆盖"的难负样本）；
* 对每题取"混合检索 + 重排后的最高分"，扫描阈值 → 统计误拒 / 漏拒 → 选最优点。

**为什么正样本排除 calc 与 multi_turn**：这两类能否回答不取决于检索分数
（calc 靠计算工具，multi_turn 靠对话历史），把它们的检索分当"应答题证据分"会让曲线失真。

用法::

    # P2 口径（探针集 16 题 + 8/15 题负样本）
    python scripts/calibrate_threshold.py --out docs/threshold_calibration.md

    # P5 口径（冻结集 60 题 fact/boundary + 15 题拒答）
    python scripts/calibrate_threshold.py --probe-set data/eval/golden_set.jsonl \
        --refuse-set data/eval/golden_set.jsonl --out docs/threshold_calibration_p5.md
"""

import _bootstrap  # noqa: F401  必须最先导入

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from customer_service_assistant.config import Settings, get_settings
from customer_service_assistant.llm import setup_tracing
from customer_service_assistant.logging_setup import setup_logging
from customer_service_assistant.retrieval.retriever import retrieve_with_report

PROBE_SET_PATH = Path("data/eval/probe_questions.json")
REFUSE_SET_PATH = Path("data/eval/refuse_questions.json")

# 分数是交叉编码器输出的绝对分（bge-reranker-v2-m3，sigmoid 后 0~1），
# 相关片段约 0.1~0.3、无关片段接近 0，因此网格在低分区加密。
DEFAULT_GRID = [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="拒答阈值标定")
    parser.add_argument("--probe-set", default=str(PROBE_SET_PATH), help="正样本集")
    parser.add_argument("--refuse-set", default=str(REFUSE_SET_PATH), help="负样本集")
    parser.add_argument(
        "--positive-categories",
        default="fact,boundary",
        help="JSONL 正样本集里纳入标定的类别（逗号分隔；JSON 探针集不受影响）",
    )
    parser.add_argument("--label", default=None, help="报告标题里的样本口径标签")
    parser.add_argument("--out", default="docs/threshold_calibration.md", help="输出报告路径")
    parser.add_argument("--collection", default=None, help="目标 collection")
    return parser.parse_args()


def resolve(settings: Settings, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else settings.project_root / candidate


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_questions(path: Path, *, exclude_categories: set[str] | None = None,
                   include_categories: set[str] | None = None) -> list[dict]:
    """同时支持 JSON（``{"questions": [...]}``）与 JSONL（P5 冻结集）两种格式。"""
    if path.suffix.lower() == ".jsonl":
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        items = load_json(path)["questions"]
    if exclude_categories:
        items = [item for item in items if item.get("category") not in exclude_categories]
    if include_categories:
        items = [item for item in items if item.get("category") in include_categories]
    return items


def collect_scores(settings: Settings, samples: list[dict]) -> list[dict]:
    """对每个样本取"混合检索 + 重排后的最高分"（不做阈值过滤）。"""
    results: list[dict] = []
    for sample in samples:
        report = retrieve_with_report(
            sample["question"],
            limit=1,
            settings=settings,
            mode="hybrid",
            rerank=True,
            apply_threshold=False,
        )
        score = report.hits[0].rerank_score if report.hits and report.hits[0].rerank_score is not None else 0.0
        results.append(
            {
                "id": sample["id"],
                "question": sample["question"],
                # 探针集用 gold 字段判断"应回答"；冻结集由调用方显式标注（_should_answer）
                "should_answer": bool(sample.get("_should_answer", bool(sample.get("gold")))),
                "section": report.hits[0].section if report.hits else None,
                "section_no": report.hits[0].section_no if report.hits else None,
                "score": round(score, 6),
            }
        )
        logger.info(
            "{} should_answer={} score={:.6f} top_section={}",
            sample["id"],
            results[-1]["should_answer"],
            results[-1]["score"],
            results[-1]["section"],
        )
    return results


def sweep(samples: list[dict], grid: list[float]) -> list[dict]:
    table: list[dict] = []
    positives = [item for item in samples if item["should_answer"]]
    negatives = [item for item in samples if not item["should_answer"]]

    for threshold in grid:
        tp = sum(1 for item in positives if item["score"] >= threshold)
        fn = len(positives) - tp
        tn = sum(1 for item in negatives if item["score"] < threshold)
        fp = len(negatives) - tn

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        refusal_rate = tn / len(negatives) if negatives else 0.0
        false_refusal = fn / len(positives) if positives else 0.0
        balanced = (recall + refusal_rate) / 2

        table.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "answer_recall": round(recall, 4),
                "false_refusal": round(false_refusal, 4),
                "refusal_rate": round(refusal_rate, 4),
                "precision": round(precision, 4),
                "f1": round(f1, 4),
                "balanced_accuracy": round(balanced, 4),
            }
        )
    return table


def pick_best(table: list[dict]) -> dict:
    """先看平衡准确率，再看 F1；并列时取更小的阈值（宁可多答，避免误拒）。"""
    return max(table, key=lambda row: (row["balanced_accuracy"], row["f1"], -row["threshold"]))


def candidate_thresholds(samples: list[dict], grid: list[float]) -> list[float]:
    """在固定网格之外补上"样本分数之间的中点"。

    固定网格可能整段跨过最优切分点（例如负样本最高 0.2097、正样本最低 0.2430，
    而网格只有 0.2 / 0.3），因此用相邻分数中点枚举所有可能改变分类结果的阈值。
    """
    scores = sorted({item["score"] for item in samples})
    midpoints = [(low + high) / 2 for low, high in zip(scores, scores[1:])]
    return sorted({round(value, 6) for value in [*grid, *midpoints]})


def separation(samples: list[dict]) -> dict:
    """正负样本的分数重叠情况。"""
    positives = [item["score"] for item in samples if item["should_answer"]]
    negatives = [item["score"] for item in samples if not item["should_answer"]]
    max_negative = max(negatives) if negatives else 0.0
    min_positive = min(positives) if positives else 0.0
    gap = min_positive - max_negative
    return {
        "max_negative": round(max_negative, 6),
        "min_positive": round(min_positive, 6),
        "gap": round(gap, 6),
        "separable": gap > 0,
        "suggested": round((max_negative + min_positive) / 2, 4) if gap > 0 else None,
    }


def render(samples: list[dict], table: list[dict], best: dict, label: str, gap_info: dict | None = None) -> str:
    lines = [
        f"# 拒答阈值标定报告（{label}）",
        "",
        f"- 生成时间：{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        f"- 样本：可答 {sum(1 for s in samples if s['should_answer'])} 题 + 超范围 "
        f"{sum(1 for s in samples if not s['should_answer'])} 题，共 {len(samples)} 题",
        "- 判据：混合检索 + 重排后的最高分是否 ≥ 阈值；低于阈值即返回空上下文（拒答）",
        "",
        "## 每题最高重排分",
        "",
        "| 题号 | 是否应回答 | 问题 | 最高分 | 命中章节 |",
        "|---|---|---|---|---|",
    ]
    for item in samples:
        lines.append(
            f"| {item['id']} | {'是' if item['should_answer'] else '否'} | {item['question']} | "
            f"{item['score']:.6f} | {item['section'] or '-'} |"
        )

    lines += [
        "",
        "## 阈值扫描",
        "",
        "| 阈值 | TP | FP | TN | FN | 应答题通过率 | 误拒率 | 超范围拒答率 | 精确率 | F1 | 平衡准确率 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in table:
        lines.append(
            f"| {row['threshold']} | {row['tp']} | {row['fp']} | {row['tn']} | {row['fn']} | "
            f"{row['answer_recall']:.1%} | {row['false_refusal']:.1%} | {row['refusal_rate']:.1%} | "
            f"{row['precision']:.1%} | {row['f1']:.3f} | {row['balanced_accuracy']:.3f} |"
        )

    lines += [
        "",
        "## 推荐阈值",
        "",
        f"**`CSA_REFUSAL_THRESHOLD={best['threshold']}`**",
        "",
        "> 注意：这里标定的是**拒答阈值**（基于最高重排分做二分类），与**片段过滤阈值**"
        "`CSA_RERANK_THRESHOLD`（默认 0.02，只用于剔除噪声片段）是两件事。"
        "P2 实测：两者合一会导致「语义弱但相关」的片段被误滤（如 q-08 的文档说明片段仅 0.11 分）。",
        "",
        f"- 应答题全部通过（误拒率 {best['false_refusal']:.1%}）",
        f"- 超范围问题拒答率 {best['refusal_rate']:.1%}（{best['tn']}/{best['tn'] + best['fp']}）",
        f"- 平衡准确率 {best['balanced_accuracy']:.3f}，F1 {best['f1']:.3f}",
    ]
    if gap_info:
        lines += [
            "",
            "### 分数间隔分析",
            "",
            f"- 最高负样本分：{gap_info['max_negative']}；最低正样本分：{gap_info['min_positive']}",
            f"- 间隔：{gap_info['gap']}"
            + (
                f"（**正负样本线性可分**，任一阈值落在 ({gap_info['max_negative']}, {gap_info['min_positive']}] 区间内都能实现 100% 拒答 + 0% 误拒；建议取中点 {gap_info['suggested']} 以留出波动余量）"
                if gap_info["separable"]
                else "（**存在重叠**，无法同时做到 0 误拒与 100% 拒答，需要按业务偏好取舍）"
            ),
        ]
    lines += [
        "",
        "> 说明：阈值越低越保守（宁可多答），越高越激进（宁可拒答）。客服场景下 **误答的代价高于拒答**，"
        "因此若两档平衡准确率接近，应选更高的一档；反之若业务更怕\"把能答的问题拒掉\"，则选更低的一档。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    settings = get_settings()
    if args.collection:
        settings = settings.model_copy(update={"milvus_collection": args.collection})
    setup_logging(settings.log_level, settings.log_dir)
    setup_tracing(settings)

    positive_categories = {part.strip() for part in args.positive_categories.split(",") if part.strip()}
    probe_path = resolve(settings, args.probe_set)
    positives = load_questions(
        probe_path,
        include_categories=positive_categories if probe_path.suffix.lower() == ".jsonl" else None,
    )
    refuse_path = resolve(settings, args.refuse_set)
    if refuse_path.suffix.lower() == ".jsonl":
        negatives = load_questions(refuse_path, include_categories={"refuse"})
    else:
        negatives = load_questions(refuse_path)
    # 显式标注正负，避免不同来源（gold 字段 / category）的判据混用
    for item in positives:
        item["_should_answer"] = True
    for item in negatives:
        item["_should_answer"] = False
    logger.info("正样本 {} 题 / 负样本 {} 题", len(positives), len(negatives))

    samples = collect_scores(settings, positives) + collect_scores(settings, negatives)
    gap_info = separation(samples)
    table = sweep(samples, DEFAULT_GRID)  # 固定网格：用于画曲线
    fine_table = sweep(samples, candidate_thresholds(samples, DEFAULT_GRID))  # 含样本中点：用于选点
    best = pick_best(fine_table)

    print("\n阈值扫描结果（摘要）:")
    for row in table:
        print(
            f"  t={row['threshold']:<6} 应答题通过={row['answer_recall']:>6.1%} "
            f"误拒={row['false_refusal']:>6.1%} 超范围拒答={row['refusal_rate']:>6.1%} "
            f"平衡准确率={row['balanced_accuracy']:.3f} F1={row['f1']:.3f}"
        )
    print(
        f"\n推荐阈值: {best['threshold']}（平衡准确率 {best['balanced_accuracy']:.3f}，"
        f"误拒 {best['false_refusal']:.1%}，超范围拒答率 {best['refusal_rate']:.1%}）"
    )
    print(
        f"分数间隔: 最高负样本 {gap_info['max_negative']} / 最低正样本 {gap_info['min_positive']} "
        f"/ 间隔 {gap_info['gap']} ｜ 线性可分={gap_info['separable']} ｜ 中点建议={gap_info['suggested']}"
    )

    out_path = resolve(settings, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render(
            samples,
            table,
            best,
            args.label
            or f"{probe_path.name}（{len(positives)} 正） + {refuse_path.name}（{len(negatives)} 负）",
            gap_info,
        ),
        encoding="utf-8",
    )
    logger.info("标定报告已写入 {}", out_path)

    # 同时落一份原始分数，便于复算与对比
    raw_path = out_path.with_name(out_path.stem + "_raw.json")
    raw_path.write_text(
        json.dumps(
            {
                "samples": samples,
                "sweep": table,
                "fine_sweep": fine_table,
                "best": best,
                "gap": gap_info,
                "collection": settings.milvus_collection,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("原始分数已写入 {}", raw_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
