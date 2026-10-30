"""服务层请求 / 响应模型（与设计文档 §5.11 的接口契约一致）。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..agent.schemas import Answer


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000, description="用户问题")
    thread_id: str | None = Field(default=None, description="会话 ID；不传则服务端生成")
    user_id: str | None = Field(default=None, description="用户标识（预留：用于越权校验）")


class ChatResponse(BaseModel):
    thread_id: str
    answer: Answer | None = None
    problem: str | None = None
    tool_calls: list[str] = Field(default_factory=list)
    interrupts: int = 0
    elapsed: float = 0.0


class HitlDecision(BaseModel):
    type: Literal["approve", "reject"]


class HitlRequest(BaseModel):
    decisions: list[HitlDecision] = Field(min_length=1, description="与待确认动作一一对应")


class IngestResponse(BaseModel):
    collection: str
    total_records: int
    visible_before: int
    visible_after: int
    removed_doc_ids: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    manifest_path: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    milvus: dict[str, Any]
    model: dict[str, Any]
    prompt_version: str
    tools: list[str]


class StatsResponse(BaseModel):
    collection: str
    visible_count: int
    schema_name: str
    dimension: int
    manifest: dict[str, Any] | None = None
