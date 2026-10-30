"""P0 baseline Agent：无工具、单轮、自由文本输出。

刻意保持"弱"：它对应设计文档 §6.3 的 A 组对照组。P3 的 ``builder.py`` 会引入工具、
结构化输出与中间件，届时用评测集量化两者差距。
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent

from ..llm import get_chat_model
from .prompts import BASELINE_SYSTEM_PROMPT


def build_baseline_agent(model: Any | None = None):
    return create_agent(
        model=model or get_chat_model(),
        tools=[],
        system_prompt=BASELINE_SYSTEM_PROMPT,
    )


def extract_final_text(result: dict) -> str:
    """从 create_agent 的返回结果中取出最后一条 AI 消息文本。"""
    messages = result.get("messages", [])
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if content and getattr(message, "type", None) == "ai":
            if isinstance(content, str):
                return content
            if isinstance(content, list):  # 部分模型返回 content block 列表
                return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return ""
