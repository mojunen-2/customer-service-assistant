"""计算类评测：20 道手算题，验证 Agent 是否走工具且数值正确。

判据（两级，避免自证循环）：

1. **主判据**：Agent 必须调用对应的计算工具，且工具返回值与**人工手算**的 expected 完全一致；
2. **辅助判据**：最终回答里要出现关键数字（例如超额费用 64 元），防止"算对了但没告诉用户"。

用法::

    python scripts/eval_calc.py                 # 全量 20 题
    python scripts/eval_calc.py --limit 2       # 先跑 2 题验证评测链路
    python scripts/eval_calc.py --out docs/calc_eval.md
"""

import _bootstrap  # noqa: F401  必须最先导入

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from customer_service_assistant.agent.builder import build_agent, run_agent
from customer_service_assistant.agent.schemas import collect_tool_results
from customer_service_assistant.config import Settings, get_settings
from customer_service_assistant.llm import setup_tracing
from customer_service_assistant.logging_setup import setup_logging

QUESTIONS_FILE = Path("data/eval/calc_questions.json")

# 题型 -> (负责计算的工具, 参与比对的字段)
CASE_CHECKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "api_overage": ("calc_api_overage_tool", ("total_fee", "overage")),
    "ai_quota": ("calc_ai_quota_status_tool", ("remaining", "exceeded")),
    "refund": ("calc_refund_tool", ("refund_ratio", "eligible")),
    "data_retention": ("check_data_retention_tool", ("state", "days_left")),
}
# 回答里应当出现的关键数字（按题型取 expected 里的哪些字段）
ANSWER_NUMBER_FIELDS: dict[str, tuple[str, ...]] = {
    "api_overage": ("total_fee",),
    "ai_quota": ("remaining",),
    "refund": (),
    "data_retention": ("days_left",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="计算类评测（20 题）")
    parser.add_argument("--questions", default=str(QUESTIONS_FILE), help="题库路径")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0 表示全量）")
    parser.add_argument("--ids", default=None, help="只跑指定题号，逗号分隔")
    parser.add_argument("--out", default="docs/calc_eval.md", help="输出报告路径")
    return parser.parse_args()


def resolve(settings: Settings, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(settings.project_root) / candidate


def number_variants(value: object) -> list[str]:
    """把数值转成"回答里可能出现的写法"，用于辅助判据。"""
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, float):
        variants = [f"{value:g}", f"{value:.2f}".rstrip("0").rstrip(".")]
        return list(dict.fromkeys(variants))
    return [str(value)]


def check_case(case: dict, run) -> dict:
    tool_name, _ = CASE_CHECKS[case["type"]]
    expected: dict = case["expected"]
    payloads = collect_tool_results(run.messages).get(tool_name, [])

    matched = [
        payload
        for payload in payloads
        if all(payload.get(key) == value for key, value in expected.items())
    ]
    answer_text = run.answer.answer if run.answer else ""
    key_blob = " ".join(run.answer.key_facts) if run.answer else ""
    haystack = f"{answer_text} {key_blob}"

    missing_numbers: list[str] = []
    for field in ANSWER_NUMBER_FIELDS[case["type"]]:
        value = expected.get(field)
        if value is None:
            continue
        if not any(variant in haystack for variant in number_variants(value)):
            missing_numbers.append(f"{field}={value}")

    return {
        "id": case["id"],
        "type": case["type"],
        "question": case["question"],
        "expected": expected,
        "tool_called": bool(payloads),
        "tool_values": payloads[-1] if payloads else None,
        "value_match": bool(matched),
        "answer_numbers_ok": not missing_numbers,
        "missing_numbers": missing_numbers,
        "answer": answer_text,
        "confidence": run.answer.confidence if run.answer else None,
        "need_human": run.answer.need_human if run.answer else None,
        "tools": run.tool_calls,
        "structured_ok": run.answer is not None,
        "elapsed": round(run.elapsed, 2),
        "passed": bool(matched) and not missing_numbers and run.answer is not None,
    }


def render(rows: list[dict], label: str) -> str:
    passed = sum(row["passed"] for row in rows)
    lines = [
        f"# 计算类评测报告（{label}）",
        "",
        f"- 生成时间：{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        f"- 样本：{len(rows)} 题（expected 为人工手算值，非工具生成）",
        f"- **通过 {passed}/{len(rows)} = {passed / len(rows):.1%}**",
        "",
        "判据：① 必须调用对应计算工具且返回值与手算值一致；② 回答中必须出现关键数字。",
        "",
        "| 题号 | 题型 | 期望值 | 调用工具 | 工具值一致 | 回答含关键数字 | 置信度 | 需转人工 | 耗时(s) | 结论 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        numbers_cell = "是" if row["answer_numbers_ok"] else "否（缺 " + ",".join(row["missing_numbers"]) + "）"
        lines.append(
            f"| {row['id']} | {row['type']} | {json.dumps(row['expected'], ensure_ascii=False)} | "
            f"{'是' if row['tool_called'] else '否'} | {'是' if row['value_match'] else '否'} | "
            f"{numbers_cell} | {row['confidence']} | {'是' if row['need_human'] else '否'} | "
            f"{row['elapsed']} | {'通过' if row['passed'] else '未通过'} |"
        )
    lines += ["", "## 逐题回答", ""]
    for row in rows:
        lines.append(f"### {row['id']}（{row['type']}）{row['question']}")
        lines.append("")
        lines.append(f"- 工具调用：{' → '.join(row['tools']) or '（无）'}")
        lines.append(f"- 期望值：{json.dumps(row['expected'], ensure_ascii=False)}")
        tool_values = json.dumps(row["tool_values"], ensure_ascii=False) if row["tool_values"] else "（未调用）"
        lines.append(f"- 工具实际返回：{tool_values}")
        lines.append("")
        lines.append("```text")
        lines.append(row["answer"] or "（无结构化答案）")
        lines.append("```")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_dir)
    setup_tracing(settings)

    payload = json.loads(resolve(settings, args.questions).read_text(encoding="utf-8"))
    cases: list[dict] = payload["questions"]
    if args.ids:
        wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
        cases = [case for case in cases if case["id"] in wanted]
    if args.limit:
        cases = cases[: args.limit]
    logger.info("计算类评测：{} 题", len(cases))

    bundle = build_agent(settings)
    rows = []
    for case in cases:
        run = run_agent(bundle, case["question"], thread_id=f"calc-{case['id']}")
        row = check_case(case, run)
        rows.append(row)
        logger.info(
            "{} {} | 工具={} 值一致={} 数字={} | {:.1f}s",
            case["id"],
            "通过" if row["passed"] else "未通过",
            row["tool_called"],
            row["value_match"],
            row["answer_numbers_ok"],
            row["elapsed"],
        )

    passed = sum(row["passed"] for row in rows)
    print(f"\n计算类评测：{passed}/{len(rows)} 通过（{passed / len(rows):.1%}）")
    for row in rows:
        if not row["passed"]:
            print(
                f"  未通过 {row['id']}: 工具调用={row['tool_called']} 值一致={row['value_match']} "
                f"缺数字={row['missing_numbers']} 结构化={row['structured_ok']}"
            )

    out_path = resolve(settings, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(rows, "P3 工具化 Agent"), encoding="utf-8")
    logger.info("报告已写入 {}", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
