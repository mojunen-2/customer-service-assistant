"""消融实验运行器（设计文档 §6.3 / §6.4）。

用法::

    # 跑一组（A~E），落盘逐题记录 + 汇总
    python -m customer_service_assistant.eval.runner --config configs/E_production.json --judge \
        --out data/eval/results/E_production.jsonl

    # 把 data/eval/results/*.summary.json 渲染成 docs/ablation.md
    python -m customer_service_assistant.eval.runner --render docs/ablation.md

设计要点：

1. **同一评测集、同一模型、同一提示词**，只有检索/后处理配置不同 —— 增益才能归因；
2. 逐题原始记录（JSONL）与汇总（JSON）都落盘，报告里的每个数字都能回溯到单题；
3. judge 只在显式 ``--judge`` 时开启（软性指标，费 token）；
4. ``--render`` 只生成**数据驱动**的段落，人工分析段落用标记包起来，重新渲染时保留。
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from loguru import logger

from ..config import Settings, get_settings
from ..llm import get_chat_model, setup_tracing
from ..logging_setup import setup_logging
from .evaluators import ItemResult, aggregate, evaluate_rules, format_metrics_line
from .judge_prompts import JUDGE_PROMPT_VERSION, judge_prompt_hash, judge_prompt_hashes
from .pipelines import GroupConfig, build_runner, derive_refused, load_config, run_item

EVAL_DIR = Path("data/eval")
RESULTS_DIR = EVAL_DIR / "results"
GOLDEN_PATH = EVAL_DIR / "golden_set.jsonl"
GROUP_ORDER = ["A_baseline", "B_structure", "C_hybrid", "D_rerank", "E_production"]
MANUAL_BEGIN = "<!-- MANUAL-ANALYSIS-BEGIN -->"
MANUAL_END = "<!-- MANUAL-ANALYSIS-END -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="消融实验运行器")
    parser.add_argument("--config", default=None, help="消融配置 YAML（A~E 之一）")
    parser.add_argument("--dataset", default=str(GOLDEN_PATH), help="评测集 JSONL")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0 = 全量）")
    parser.add_argument("--ids", default=None, help="只跑指定题号（逗号分隔）")
    parser.add_argument("--categories", default=None, help="只跑指定类别（逗号分隔）")
    parser.add_argument("--judge", action="store_true", help="开启 LLM judge（答案正确性 + faithfulness）")
    parser.add_argument("--concurrency", type=int, default=1, help="并发题数（默认串行，便于定位问题）")
    parser.add_argument("--out", default=None, help="逐题记录输出路径（默认 data/eval/results/<name>.jsonl）")
    parser.add_argument("--price-input", type=float, default=None, help="输入 token 单价（元/百万 token，可选）")
    parser.add_argument("--price-output", type=float, default=None, help="输出 token 单价（元/百万 token，可选）")
    parser.add_argument(
        "--recheck",
        action="store_true",
        help="判据更新后**离线重算规则判据**（复用已落盘的回答，不重跑模型/不重新检索）",
    )
    parser.add_argument(
        "--rejudge",
        choices=["none", "correctness", "faithfulness", "both"],
        default="none",
        help="只重跑指定 judge（复用已落盘的回答），用于判据提示词迭代后保持一致口径",
    )
    parser.add_argument("--render", default=None, help="只渲染报告：把 results/*.summary.json 写成 markdown")
    return parser.parse_args()


def resolve(settings: Settings, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(settings.project_root) / candidate


def load_dataset(settings: Settings, path: str, args: argparse.Namespace) -> list[dict]:
    target = resolve(settings, path)
    items = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.ids:
        wanted = {part.strip() for part in args.ids.split(",") if part.strip()}
        items = [item for item in items if item["id"] in wanted]
    if args.categories:
        wanted_categories = {part.strip() for part in args.categories.split(",") if part.strip()}
        items = [item for item in items if item["category"] in wanted_categories]
    if args.limit:
        items = items[: args.limit]
    return items


def load_golden(settings: Settings, path: str) -> list[dict]:
    """读取完整冻结集（不做 --ids/--limit 过滤）。"""
    target = resolve(settings, path)
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]


def records_path(settings: Settings, args: argparse.Namespace, name: str) -> Path:
    return resolve(settings, args.out or (RESULTS_DIR / f"{name}.jsonl"))


def _write_results(path: Path, records: Sequence[ItemResult], summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record.to_dict(), ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _record_from_dict(payload: dict) -> ItemResult:
    return ItemResult(**{key: value for key, value in payload.items() if key in ItemResult.__dataclass_fields__})


def recompute(args: argparse.Namespace, settings: Settings) -> int:
    """离线重算：判据（规则 / judge）迭代后，复用已落盘的回答重新打分。

    这是评测链路的可维护性设计——报告里的"回答"始终是模型真实产出的，
    只有判据可以独立迭代，改判据不必重跑 500 次模型调用。
    """
    from .evaluators import judge_correctness, judge_faithfulness

    config = load_config(args.config, settings)
    path = records_path(settings, args, config.name)
    if not path.exists():
        raise SystemExit(f"找不到逐题记录：{path}")
    payloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    items = {item["id"]: item for item in load_golden(settings, args.dataset)}
    records = [_record_from_dict(payload) for payload in payloads]

    if args.recheck:
        for record in records:
            item = items.get(record.id)
            if item is None:
                logger.warning("{} 不在冻结集里，跳过规则重算", record.id)
                continue
            if record.has_tools:
                # 工具轨迹是落盘的原始证据，判据迭代后同样可以离线重算
                record.refused = derive_refused(record.tool_results)
            record.checks = evaluate_rules(item, record)
        logger.info("规则判据已重算：{} 题", len(records))

    if args.rejudge != "none":
        model = get_chat_model()
        wanted = {"correctness", "faithfulness"} if args.rejudge == "both" else {args.rejudge}

        def rejudge_one(record: ItemResult) -> ItemResult:
            item = items.get(record.id)
            if item is None:
                return record
            if "correctness" in wanted:
                record.judge["correctness"] = judge_correctness(item, record, model)
            if "faithfulness" in wanted:
                record.judge["faithfulness"] = judge_faithfulness(item, record, model)
            return record

        if args.concurrency and args.concurrency > 1:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                records = list(pool.map(rejudge_one, records))
        else:
            records = [rejudge_one(record) for record in records]
        logger.info("judge 已重跑：{}", sorted(wanted))

    summary = aggregate(records)
    previous = path.with_suffix(".summary.json")
    if previous.exists():
        old = json.loads(previous.read_text(encoding="utf-8"))
        summary.update(
            {
                "config": old.get("config", config.as_dict()),
                "dataset": old.get("dataset"),
                "judge_enabled": old.get("judge_enabled"),
                "generated_at": old.get("generated_at"),
                "wall_time": old.get("wall_time"),
                "price": old.get("price"),
                "cost_cny": old.get("cost_cny"),
            }
        )
    summary["recomputed"] = {
        "at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "rules": bool(args.recheck),
        "judges": [] if args.rejudge == "none" else sorted(wanted),
    }
    _write_results(path, records, summary)
    print(f"\n[{config.name}·重算] {format_metrics_line(summary)}")
    return 0


def run_group(args: argparse.Namespace, settings: Settings) -> int:
    config: GroupConfig = load_config(args.config, settings)
    items = load_dataset(settings, args.dataset, args)
    if not items:
        logger.error("评测集为空，检查 --dataset / --limit / --ids")
        return 1

    model = get_chat_model()
    runner = build_runner(config, settings, model=model)
    judge_models = {"correctness": model} if args.judge else None

    logger.info(
        "开始评测 | 组={} | 题数={} | judge={} | 并发={} | 配置={}",
        config.name,
        len(items),
        args.judge,
        args.concurrency,
        json.dumps(config.as_dict(), ensure_ascii=False),
    )
    started = time.perf_counter()
    if args.concurrency and args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            records = list(pool.map(lambda item: run_item(runner, item, judge_models), items))
    else:
        records = [run_item(runner, item, judge_models) for item in items]
    elapsed = time.perf_counter() - started

    summary = aggregate(records)
    summary.update(
        {
            "config": config.as_dict(),
            "dataset": str(Path(args.dataset).name),
            "judge_enabled": bool(args.judge),
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "wall_time": round(elapsed, 1),
            "price": {"input_per_mtok": args.price_input, "output_per_mtok": args.price_output},
        }
    )
    if summary.get("tokens") and args.price_input is not None and args.price_output is not None:
        tokens = summary["tokens"]
        summary["cost_cny"] = {
            "per_query_mean": round(
                ((tokens["total_input"] or 0) / 1e6) * args.price_input + ((tokens["total_output"] or 0) / 1e6) * args.price_output,
                4,
            )
            / max(summary["n"], 1),
            "total": round(
                ((tokens["total_input"] or 0) / 1e6) * args.price_input + ((tokens["total_output"] or 0) / 1e6) * args.price_output,
                4,
            ),
        }

    out_path = resolve(settings, args.out or (RESULTS_DIR / f"{config.name}.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(record.to_dict(), ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[{config.name}] {format_metrics_line(summary)}")
    print(f"  逐题记录：{out_path}")
    print(f"  汇总：{summary_path}")
    return 0


# --------------------------------------------------------------------------- #
# 报告渲染
# --------------------------------------------------------------------------- #
def _load_summaries(settings: Settings) -> list[dict]:
    directory = resolve(settings, RESULTS_DIR)
    if not directory.exists():
        return []
    summaries: list[dict] = []
    for path in sorted(directory.glob("*.summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = path.name
        summaries.append(payload)
    summaries.sort(key=lambda item: GROUP_ORDER.index(item["config"]["name"]) if item["config"]["name"] in GROUP_ORDER else 99)
    return summaries


def _load_records(settings: Settings, config_name: str) -> list[dict]:
    path = resolve(settings, RESULTS_DIR / f"{config_name}.jsonl")
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fmt(entry: dict | None, *, percent: bool = True) -> str:
    if not entry or not entry.get("n"):
        return "—"
    value = entry.get("value")
    if value is None:
        return "—"
    return f"{entry['passed']}/{entry['n']} = {value:.1%}" if percent else f"{entry['passed']}/{entry['n']} = {value}"


def render_table(summaries: Sequence[dict]) -> str:
    rows = [
        "| 组 | 配置 | 规则正确率 | judge 正确率（严格） | 数值精确率 | 工具数值一致 | 路由准确率 | 拒答准确率 | 误拒率 | 引用支撑 | 忠实率 | 期望章节命中 | p50 / p95 延迟 | 平均 token（入/出） | judge 判题数 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for summary in summaries:
        config = summary["config"]
        latency = summary.get("latency") or {}
        tokens = summary.get("tokens") or {}
        label = f"{config['name']}｜{config['mode']}{'+rerank' if config['rerank'] else ''}{'+阈值' if config['apply_threshold'] else ''}"
        rows.append(
            f"| {config['name'].split('_')[0]} | {label} | "
            f"{_fmt(summary.get('rule_answer_accuracy'))} | {_fmt(summary.get('judge_correctness_strict'))} | "
            f"{_fmt(summary.get('numeric_exactness'))} | {_fmt(summary.get('tool_value_match'))} | "
            f"{_fmt(summary.get('routing_accuracy'))} | "
            f"{_fmt(summary.get('refusal_accuracy'))} | {_fmt(summary.get('over_refusal_rate'))} | "
            f"{_fmt(summary.get('citation_precision'))} | {_fmt(summary.get('faithfulness'))} | "
            f"{_fmt(summary.get('section_recall'))} | "
            f"{latency.get('p50', '—')} / {latency.get('p95', '—')} s | "
            f"{tokens.get('mean_input', '—')} / {tokens.get('mean_output', '—')} | {summary.get('judged_items', 0)} |"
        )
    return "\n".join(rows) + "\n"


def render_latency_cost(summaries: Sequence[dict]) -> str:
    rows = [
        "| 组 | 检索均值(s) | 生成均值(s) | p50(s) | p95(s) | 输入 token/题 | 输出 token/题 | 单题成本(元) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for summary in summaries:
        latency = summary.get("latency") or {}
        tokens = summary.get("tokens") or {}
        cost = summary.get("cost_cny")
        rows.append(
            f"| {summary['config']['name']} | {latency.get('mean_retrieval') or '—'} | "
            f"{latency.get('mean_generation') or '—'} | {latency.get('p50') or '—'} | {latency.get('p95') or '—'} | "
            f"{tokens.get('mean_input') or '—'} | {tokens.get('mean_output') or '—'} | "
            f"{(cost or {}).get('per_query_mean', '未提供单价')} |"
        )
    return "\n".join(rows) + "\n"


def render_threshold_curve(settings: Settings) -> str:
    """把阈值标定的原始数据渲染成曲线表（设计文档 §6.5 第②条）。"""
    raw_path = resolve(settings, "docs/threshold_calibration_p5_raw.json")
    if not raw_path.exists():
        return "（未找到 docs/threshold_calibration_p5_raw.json，先跑 scripts/calibrate_threshold.py）\n"
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    gap = payload.get("gap") or {}
    sweep = payload.get("sweep") or []
    lines = [
        "| 拒绝阈值 | 应答题通过率 | 误拒率 | 超范围拒答率 | 平衡准确率 | F1 |",
        "|---|---|---|---|---|---|",
    ]
    for row in sweep:
        lines.append(
            f"| {row['threshold']} | {row['answer_recall']:.1%} | {row['false_refusal']:.1%} | "
            f"{row['refusal_rate']:.1%} | {row['balanced_accuracy']:.3f} | {row['f1']:.3f} |"
        )
    lines += [
        "",
        f"- 样本：{len(payload.get('samples', []))} 题（"
        f"可答 {sum(1 for s in payload.get('samples', []) if s['should_answer'])} + "
        f"超范围 {sum(1 for s in payload.get('samples', []) if not s['should_answer'])}）",
        f"- 最高负样本分：{gap.get('max_negative')}；最低正样本分：{gap.get('min_positive')}；"
        f"间隔：{gap.get('gap')}（线性可分：{gap.get('separable')}）",
        f"- 相邻样本中点建议：{gap.get('suggested') or payload.get('best', {}).get('threshold')}"
        f"；生产配置取 `CSA_REFUSAL_THRESHOLD=0.22`",
        "",
        "> 读法：阈值越高越激进（宁可拒答）。P5 口径下正负样本**有重叠**——存在 1 条低分正样本"
        "（`g-b05` 第 8 天才申请退款，0.1451，但命中的是正确章节）和 1 条高分负样本"
        "（`g-r08` 专业版可以接入企业微信吗，0.2115）；取中点 0.22 时误拒 1/60、超范围拒答 15/15。",
    ]
    return "\n".join(lines) + "\n"


def pick_cases(records: Sequence[dict]) -> dict[str, tuple[str, dict | None]]:
    """挑三个典型案例：可溯源的成功、正确拒答、失败（含原因）。

    失败案例优先取"规则判据未通过 / judge 判错"的题；若整组都通过，
    则退化为**忠实度**上被抓到无依据主张的题（P5 的生产配置就是这种情况）。
    """
    traced = [
        record
        for record in records
        if record.get("citations")
        and (record.get("judge", {}).get("correctness", {}) or {}).get("verdict") == "correct"
    ]
    traced.sort(key=lambda record: -len(record["citations"]))
    refused = [
        record
        for record in records
        if record["category"] == "refuse" and record.get("checks", {}).get("refusal_ok")
    ]
    failures = [
        record
        for record in records
        if record.get("checks", {}).get("rule_answer_ok") is False
        or (record.get("judge", {}).get("correctness", {}) or {}).get("verdict") == "incorrect"
    ]
    kind = "规则/judge 未通过"
    if not failures:
        failures = [
            record
            for record in records
            if (record.get("judge", {}).get("faithfulness", {}) or {}).get("unsupported")
        ]
        kind = "忠实度未通过（回答里有无依据的主张）"
    return {
        "traced": ("成功案例", traced[0] if traced else None),
        "refused": ("正确拒答", refused[0] if refused else None),
        "failure": (kind, failures[0] if failures else None),
    }


def render_cases(records: Sequence[dict]) -> str:
    cases = pick_cases(records)
    lines: list[str] = []
    order = [
        ("traced", "案例 1：答案可溯源（引用支撑 + 结构化输出）"),
        ("refused", "案例 2：正确拒答（不编造）"),
        ("failure", "案例 3：失败案例与原因"),
    ]
    for key, title in order:
        kind, record = cases.get(key, ("", None))
        suffix = f"（{kind}）" if key == "failure" and record else ""
        lines += [f"### {title}{suffix}", ""]
        if not record:
            lines += ["（本次结果中没有符合条件的样本）", ""]
            continue
        checks = record.get("checks") or {}
        judgement = (record.get("judge") or {}).get("correctness") or {}
        lines += [
            f"- 题号：`{record['id']}`（{record['category']}）",
            f"- 问题：{record['question']}",
            f"- 工具调用：{' → '.join(record.get('tool_calls') or []) or '（无）'}",
            f"- 规则判定：{json.dumps({k: v for k, v in checks.items() if v is not None}, ensure_ascii=False)}",
            f"- judge：{judgement.get('verdict', '未开启')}｜{judgement.get('reason', '')}",
            "",
            "```text",
            (record.get("answer") or "（无回答）")[:1200],
            "```",
            "",
        ]
        citations = record.get("citations") or []
        if citations:
            lines.append("引用片段：")
            lines.append("")
            for citation in citations[:3]:
                lines.append(f"- `{citation.get('section', '')}`：{citation.get('quote', '')[:120]}")
            lines.append("")
    return "\n".join(lines)


def render_report(settings: Settings, out_path: Path) -> str:
    summaries = _load_summaries(settings)
    if not summaries:
        raise SystemExit("没有找到 results/*.summary.json，先跑评测")
    production_name = next(
        (summary["config"]["name"] for summary in summaries if summary["config"]["answer_mode"] == "agent"),
        summaries[-1]["config"]["name"],
    )
    records = _load_records(settings, production_name)
    generated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    manual = ""
    if out_path.exists():
        existing = out_path.read_text(encoding="utf-8")
        match = re.search(rf"{re.escape(MANUAL_BEGIN)}(.*?){re.escape(MANUAL_END)}", existing, flags=re.S)
        if match:
            manual = match.group(1).strip()
    if not manual:
        manual = "（待人工填写：逐组增益来源、为什么保留某些模块、与 P0~P4 结论的差异）"

    parts = [
        "# 消融实验报告（P5）",
        "",
        f"- 生成时间：{generated}",
        f"- 评测集：`data/eval/golden_set.jsonl`（100 题：40 事实 / 20 计算 / 20 边界 / 15 拒答 / 5 多轮）",
        "- judge（只用于软性指标，版本化）："
        f"正确性 `{judge_prompt_hashes()['correctness']}` ｜ 忠实度 `{judge_prompt_hashes()['faithfulness']}`",
        "- 逐题原始记录：`data/eval/results/<组>.jsonl`，汇总：`data/eval/results/<组>.summary.json`",
        "",
        "## 1. A~E 消融表",
        "",
        render_table(summaries),
        "",
        "指标定义：",
        "",
        "- **规则正确率**：该题全部规则判据通过（must_include / must_not_include / 数值 / 拒答 / 不误拒）；",
        "- **judge 正确率（严格）**：LLM judge 判定 correct 的比例（软性指标，仅作参考）；",
        "- **数值精确率**：计算题的期望数值是否出现在回答中（预期值人工手算，不由被测工具生成）；",
        "- **路由准确率**：期望工具是否被调用；**拒答准确率**：应拒答题是否真的拒答且没有编造；",
        "- **误拒率**：可答题被判为「答不上来」的比例；**引用支撑**：引用 quote 是否逐字出现在检索片段中；",
        "- **忠实率**：judge 未发现「上下文不支持」的结论；**期望章节命中**：命中题目声明的期望章节。",
        "",
        "## 2. 延迟与成本拆解",
        "",
        render_latency_cost(summaries),
        "",
        "> 说明：token 为**实测**值（LangChain 返回的 usage_metadata 累加）。金额一栏只在显式传入",
        "> `--price-input/--price-output`（元/百万 token）时计算——供应商价目会变，报告不内置会过期的单价。",
        "> A~D 组的检索/生成耗时分开统计；E 组是完整 Agent 多轮工具调用，只记录端到端耗时。",
        "",
        "## 3. 拒答阈值曲线（P5 口径：60 题可答 + 15 题超范围）",
        "",
        render_threshold_curve(settings),
        "",
        "## 4. 典型案例（来自生产配置的逐题记录）",
        "",
        render_cases(records),
        "",
        "## 5. 结论与归因（人工分析）",
        "",
        MANUAL_BEGIN,
        manual,
        MANUAL_END,
        "",
    ]
    return "\n".join(parts)


def main() -> int:
    args = parse_args()
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_dir)
    setup_tracing(settings)

    if args.render:
        out_path = resolve(settings, args.render)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render_report(settings, out_path), encoding="utf-8")
        logger.info("报告已写入 {}", out_path)
        return 0

    if not args.config:
        raise SystemExit("需要 --config（跑评测 / 重算）或 --render（渲染报告）")
    if args.recheck or args.rejudge != "none":
        return recompute(args, settings)
    return run_group(args, settings)


if __name__ == "__main__":
    # 从项目根目录执行：python -m customer_service_assistant.eval.runner ...
    raise SystemExit(main())
