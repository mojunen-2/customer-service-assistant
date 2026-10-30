"""Agent 组装与调用：全项目唯一出口。

``build_agent()`` 负责把模型、工具、提示词、中间件、记忆装配起来；
``run_agent()`` 负责调用（含 HITL 中断-恢复循环）并把结果整理成 ``AgentRun``。
脚本、评测、后续的 FastAPI 服务都复用这两个函数，保证"评测的就是线上跑的"。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from langchain.agents import create_agent
from loguru import logger

from ..config import Settings, get_settings
from ..llm import get_chat_model
from .middleware import build_middleware
from .prompts import get_system_prompt
from .schemas import (
    SUBMIT_ANSWER_TOOL,
    Answer,
    extract_answer,
    iter_tool_calls,
    submit_answer,
)
from .tools_business import BUSINESS_TOOLS
from .tools_calc import CALC_TOOLS
from .tools_rag import RAG_TOOLS


@dataclass(slots=True)
class AgentRun:
    """一次 Agent 调用的结果与轨迹。"""

    message: str
    answer: Answer | None
    problem: str | None
    tool_calls: list[str]
    messages: list[Any]
    interrupts: int = 0
    elapsed: float = 0.0
    thread_id: str = "default"

    @property
    def ok(self) -> bool:
        return self.answer is not None

    def summary(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "message": self.message,
            "tools": self.tool_calls,
            "answer": self.answer.answer if self.answer else None,
            "citations": [c.model_dump() for c in self.answer.citations] if self.answer else [],
            "key_facts": self.answer.key_facts if self.answer else [],
            "confidence": self.answer.confidence if self.answer else None,
            "need_human": self.answer.need_human if self.answer else None,
            "problem": self.problem,
            "interrupts": self.interrupts,
            "elapsed": round(self.elapsed, 2),
        }


@dataclass(slots=True)
class AgentBundle:
    agent: Any
    settings: Settings
    prompt_version: str
    tools: list[Any] = field(default_factory=list)

    def tool_names(self) -> list[str]:
        return [getattr(tool, "name", str(tool)) for tool in self.tools]


def build_tools(settings: Settings | None = None) -> list[Any]:
    """组装工具集：检索 + 计算 + 业务 + 提交答案。"""
    settings = settings or get_settings()
    return [*RAG_TOOLS, *CALC_TOOLS, *BUSINESS_TOOLS, submit_answer]


def build_agent(
    settings: Settings | None = None,
    *,
    model: Any | None = None,
    checkpointer: Any | None = None,
    enable_hitl: bool | None = None,
    prompt_version: str | None = None,
) -> AgentBundle:
    """构造工具化客服 Agent（P3 的生产形态）。"""
    settings = settings or get_settings()
    prompt_version = prompt_version or settings.prompt_version
    model = model or get_chat_model()

    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()

    tools = build_tools(settings)
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=get_system_prompt(prompt_version),
        middleware=build_middleware(settings, model=model, enable_hitl=enable_hitl),
        checkpointer=checkpointer,
    )
    logger.info(
        "agent ready | prompt={} | tools={} | hitl={}",
        prompt_version,
        [getattr(tool, "name", "?") for tool in tools],
        settings.enable_hitl if enable_hitl is None else enable_hitl,
    )
    return AgentBundle(agent=agent, settings=settings, prompt_version=prompt_version, tools=tools)


def _default_decision_provider(action_requests: Sequence[dict], auto_approve: bool) -> list[dict]:
    """默认的 HITL 决策：auto_approve 时直接批准，否则在命令行询问。"""
    if auto_approve:
        return [{"type": "approve"} for _ in action_requests]

    decisions: list[dict] = []
    for request in action_requests:
        # 注意：中断载荷里工具名的键是 name（action_name 在 review_configs 里）
        print(f"\n[需人工确认] 工具：{request.get('name')}")
        print(f"  参数：{request.get('args')}")
        while True:
            choice = input("  请选择 approve / reject：").strip().lower()
            if choice in {"approve", "reject"}:
                decisions.append({"type": choice})
                break
    return decisions


def run_agent(
    bundle: AgentBundle,
    message: str,
    *,
    thread_id: str = "default",
    auto_approve: bool = True,
    decision_provider: Callable[[Sequence[dict]], list[dict]] | None = None,
) -> AgentRun:
    """调用 Agent（自动处理转人工的人工确认中断）。"""
    from langgraph.types import Command

    config = {"configurable": {"thread_id": thread_id}}
    started = time.perf_counter()
    result = bundle.agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)

    interrupt_count = 0
    interrupts = result.get("__interrupt__") or []
    while interrupts:
        interrupt_count += 1
        action_requests = []
        for item in interrupts:
            value = getattr(item, "value", item)
            action_requests.extend(value.get("action_requests", []) if isinstance(value, dict) else [])
        logger.info("HITL 中断：待确认动作 {}", [req.get("name") for req in action_requests])
        decisions = (
            decision_provider(action_requests)
            if decision_provider is not None
            else _default_decision_provider(action_requests, auto_approve)
        )
        result = bundle.agent.invoke(Command(resume={"decisions": decisions}), config=config)
        interrupts = result.get("__interrupt__") or []

    messages = result.get("messages", [])
    answer, problem = extract_answer(messages)
    if problem:
        logger.warning("未能取得结构化答案：{}", problem)

    return AgentRun(
        message=message,
        answer=answer,
        problem=problem,
        tool_calls=[name for name, _ in iter_tool_calls(messages)],
        messages=messages,
        interrupts=interrupt_count,
        elapsed=time.perf_counter() - started,
        thread_id=thread_id,
    )
