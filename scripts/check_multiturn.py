"""多轮对话验收：验证同一 thread_id 下 Agent 能正确解析指代追问。

判据不看回答文本，而看**工具入参**：第二轮调用计算工具时，参数里必须带上第一轮提到的信息
（例如"那如果是专业版呢？"要变成 plan=pro + used=4500）。这比看文本更硬——文本可以蒙，参数蒙不了。

用法::

    python scripts/check_multiturn.py
    python scripts/check_multiturn.py --case refund
"""

import _bootstrap  # noqa: F401  必须最先导入

import argparse

from loguru import logger

from customer_service_assistant.agent.builder import build_agent, run_agent
from customer_service_assistant.agent.schemas import iter_tool_calls
from customer_service_assistant.config import get_settings
from customer_service_assistant.llm import setup_tracing
from customer_service_assistant.logging_setup import setup_logging

# 每个用例：两轮对话 + 第二轮必须出现的工具入参（用于判定指代是否解析正确）
CASES: list[dict] = [
    {
        "name": "quota",
        "turns": [
            "基础版用户这个月已经用了 4500 次 AI 问答，还剩多少次额度？",
            "那如果是专业版呢？",
        ],
        "expect_tool": "calc_ai_quota_status_tool",
        "expect_args": {"plan": "pro", "used": 4500},
        "expect_numbers": ["25500"],
    },
    {
        "name": "overage",
        "turns": [
            "基础版用户某月总共调用 18000 次 API，超额部分要付多少钱？",
            "那如果是 60000 次呢？",
        ],
        "expect_tool": "calc_api_overage_tool",
        "expect_args": {"plan": "basic", "total_calls": 60000},
        "expect_numbers": ["320"],
    },
    {
        "name": "refund",
        "turns": [
            "基础版首购第 3 天，AI 问答用量占额度 5%，能全额退款吗？",
            "那如果用量到了 60% 呢？",
        ],
        "expect_tool": "calc_refund_tool",
        "expect_args": {"plan": "basic", "days_since_payment": 3},
        "expect_numbers": ["不能", "不支持"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多轮指代验收")
    parser.add_argument("--case", default=None, help="只跑指定用例（quota / overage / refund）")
    return parser.parse_args()


def run_case(bundle, case: dict) -> dict:
    thread_id = f"multiturn-{case['name']}"
    run = run_agent(bundle, case["turns"][0], thread_id=thread_id)
    first_answer = run.answer.answer if run.answer else run.problem
    calls_before = len(iter_tool_calls(run.messages))

    run2 = run_agent(bundle, case["turns"][1], thread_id=thread_id)
    second_calls = iter_tool_calls(run2.messages)[calls_before:]
    second_answer = run2.answer.answer if run2.answer else (run2.problem or "")
    key_facts = " ".join(run2.answer.key_facts) if run2.answer else ""
    haystack = f"{second_answer} {key_facts}"

    matched_call = next(
        (args for name, args in second_calls if name == case["expect_tool"] and all(args.get(k) == v for k, v in case["expect_args"].items())),
        None,
    )
    numbers_ok = any(
        any(needle in haystack for needle in [str(value), str(value) + " 元"])
        for value in case["expect_numbers"]
    ) or all(needle in haystack for needle in case["expect_numbers"])

    return {
        "name": case["name"],
        "first_question": case["turns"][0],
        "first_answer": first_answer,
        "second_question": case["turns"][1],
        "second_answer": second_answer,
        "second_turn_tools": [name for name, _ in second_calls],
        "expect_tool": case["expect_tool"],
        "expect_args": case["expect_args"],
        "matched_args": matched_call,
        "numbers_ok": numbers_ok,
        "keep_alive": len(run2.messages) > len(run.messages),
        "passed": matched_call is not None,
    }


def main() -> int:
    args = parse_args()
    settings = get_settings()
    setup_logging("WARNING", None)
    setup_tracing(settings)

    cases = [case for case in CASES if args.case is None or case["name"] == args.case]
    if not cases:
        print(f"没有匹配的用例：{args.case}")
        return 1

    bundle = build_agent(settings)
    results = []
    for case in cases:
        result = run_case(bundle, case)
        results.append(result)
        print("=" * 90)
        print(f"[{result['name']}] 指代解析：{'通过' if result['passed'] else '未通过'}")
        print(f"  第一轮：{result['first_question']}")
        print(f"    -> {result['first_answer'][:120]}")
        print(f"  第二轮：{result['second_question']}")
        print(f"    第二轮工具调用：{result['second_turn_tools']}")
        print(f"    期望入参：{result['expect_args']}")
        print(f"    实际命中：{result['matched_args']}")
        print(f"    回答含关键结论：{result['numbers_ok']}")

    passed = sum(result["passed"] for result in results)
    print(f"\n多轮指代验收：{passed}/{len(results)} 通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
