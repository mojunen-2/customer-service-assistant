"""中间件装配：重试、限流、PII、长对话摘要、转人工中断。

对应设计文档 §5.5 / §5.8 / §5.9。装配顺序有意如此：

1. ``ModelRetryMiddleware`` / ``ToolRetryMiddleware``：网络与工具抖动重试
   （注意：模型层的重试**不能**用 ``model.with_retry()``，那会返回 ``RunnableRetry``，
    ``create_agent`` 拿不到 ``bind_tools`` 会直接报错——已在 llm.py 里记录）；
2. 调用次数上限：防止 Agent 在"检索不到→换个说法再检索"里空转烧钱；
3. PII 脱敏：客服会话里常见手机号/邮箱，进模型前先 mask；
4. 长对话摘要：多轮对话超过阈值后压缩历史，避免 token 爆炸；
5. 转人工中断：``escalate_to_human`` 必须先经人工确认（HITL）。
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)

from ..config import Settings, get_settings

# 中国大陆手机号（用于 PII 脱敏；PIIMiddleware 内置类型里没有）
CN_PHONE_PATTERN = r"1[3-9]\d{9}"
HITL_TOOLS = ("escalate_to_human",)


def build_middleware(
    settings: Settings | None = None,
    *,
    model: Any | None = None,
    enable_hitl: bool | None = None,
) -> list[Any]:
    """构造中间件列表（顺序即执行顺序）。"""
    settings = settings or get_settings()
    enable_hitl = settings.enable_hitl if enable_hitl is None else enable_hitl

    middleware: list[Any] = [
        ModelRetryMiddleware(max_retries=max(settings.llm_retry_attempts - 1, 1)),
        ToolRetryMiddleware(max_retries=2),
        ModelCallLimitMiddleware(run_limit=settings.max_model_calls),
        ToolCallLimitMiddleware(run_limit=settings.max_tool_calls),
        PIIMiddleware("email", strategy="mask", apply_to_input=True),
        PIIMiddleware("cn_phone", strategy="mask", apply_to_input=True, detector=CN_PHONE_PATTERN),
    ]

    if model is not None:
        middleware.append(
            SummarizationMiddleware(model=model, trigger=("tokens", settings.summary_trigger))
        )

    if enable_hitl:
        middleware.append(
            HumanInTheLoopMiddleware(
                interrupt_on={name: {"allowed_decisions": ["approve", "reject"]} for name in HITL_TOOLS}
            )
        )
    return middleware
