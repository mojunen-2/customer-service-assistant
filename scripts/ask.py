"""命令行问答（P3 工具化 Agent）。

用法::

    python scripts/ask.py "为什么我在 7 天内申请退款，还是被拒了？"
    python scripts/ask.py "那如果用量到 60% 呢？" --thread-id demo-1   # 同一 thread 多轮（指代追问）
    python scripts/ask.py "帮我查订单 ORD-2026-0102 能退多少" --show-trace
    python scripts/ask.py "我要投诉" --manual-approval                 # 转人工时手动确认
    python scripts/ask.py "专业版多少钱" --baseline                     # 用 P0 baseline Agent 对比
"""

import _bootstrap  # noqa: F401  必须最先导入

import argparse

from customer_service_assistant.agent.schemas import SUBMIT_ANSWER_TOOL

from customer_service_assistant.agent.builder import build_agent, run_agent
from customer_service_assistant.agent.prompts import build_user_prompt
from customer_service_assistant.config import get_settings
from customer_service_assistant.llm import setup_tracing
from customer_service_assistant.logging_setup import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Customer Service Assistant 命令行问答")
    parser.add_argument("query", help="用户问题")
    parser.add_argument("--thread-id", default="cli", help="会话 ID（同一 ID 共享多轮记忆）")
    parser.add_argument("--show-trace", action="store_true", help="打印工具调用轨迹")
    parser.add_argument("--show-citations", action="store_true", help="打印引用原文")
    parser.add_argument("--manual-approval", action="store_true", help="转人工时手动确认（默认自动批准）")
    parser.add_argument("--baseline", action="store_true", help="用 P0 baseline Agent（无工具，纯检索问答）")
    return parser.parse_args()


def run_baseline(settings, query: str) -> None:
    """P0 baseline 对照：无工具、单轮、自由文本。"""
    from customer_service_assistant.agent.baseline_agent import build_baseline_agent, extract_final_text
    from customer_service_assistant.retrieval.retriever import build_context, retrieve_with_report

    report = retrieve_with_report(query, limit=5, settings=settings)
    print(f"\n=== 检索（{report.mode} / 候选 {report.candidates} → 上下文 {len(report.hits)} 段）===")
    for hit in report.hits:
        print(f"  [{hit.rank}] rerank={hit.rerank_score} section={hit.section}")
    result = build_baseline_agent().invoke(
        {"messages": [{"role": "user", "content": build_user_prompt(query, build_context(report.hits))}]}
    )
    print(f"\n=== baseline 回答 ===\n{extract_final_text(result)}\n")


def main() -> int:
    args = parse_args()
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_dir)
    setup_tracing(settings)

    if args.baseline:
        run_baseline(settings, args.query)
        return 0

    bundle = build_agent(settings)
    run = run_agent(
        bundle,
        args.query,
        thread_id=args.thread_id,
        auto_approve=not args.manual_approval,
    )

    if args.show_trace:
        print(f"\n=== 工具调用轨迹（{len(run.tool_calls)} 次）===")
        for index, name in enumerate(run.tool_calls, start=1):
            print(f"  {index}. {name}")
        rejections = [
            getattr(message, "content", "")
            for message in run.messages
            if getattr(message, "name", None) == SUBMIT_ANSWER_TOOL
            and "提交被拒绝" in str(getattr(message, "content", ""))
        ]
        for index, reason in enumerate(rejections, start=1):
            print(f"\n  引用校验拒绝了第 {index} 次提交，模型据此重试：")
            for line in str(reason).splitlines():
                print(f"    {line}")

    print(f"\n=== 回答（{run.elapsed:.2f}s / thread={run.thread_id}）===")
    if run.answer is None:
        print(f"[未产出结构化答案] {run.problem}")
        last = run.messages[-1] if run.messages else None
        if last is not None:
            print(f"最后一条消息：{getattr(last, 'content', '')}")
        return 1

    answer = run.answer
    print(answer.answer)
    if answer.key_facts:
        print("\n关键数字：")
        for fact in answer.key_facts:
            print(f"  - {fact}")
    print(f"\n置信度：{answer.confidence}" + ("  ⚠️ 需要转人工：" + str(answer.human_reason) if answer.need_human else ""))
    if answer.citations:
        print(f"\n引用（{len(answer.citations)} 条）：")
        for index, citation in enumerate(answer.citations, start=1):
            prefix = f"  {index}. [{citation.section}]" if args.show_citations else f"  {index}. {citation.section}"
            print(prefix)
            if args.show_citations:
                print(f"     原文：{citation.quote}")
    if answer.suggested_followups:
        print("\n可以接着问：")
        for question in answer.suggested_followups:
            print(f"  - {question}")
    if settings.tracing_enabled:
        print(f"\n[LangSmith] project={settings.langsmith_project}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
