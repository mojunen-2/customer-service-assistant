"""FastAPI 服务层：SSE 流式问答 + 同步接口 + 健康检查 + 索引管理 + 人工审核。

启动（项目根目录）::

    python -m uvicorn customer_service_assistant.service.api:app --host 0.0.0.0 --port 8000

设计要点：

* Agent 只在首次请求时装配（懒加载单例），因此 Milvus 不可用时服务仍能启动，
  由 ``/healthz`` 报 degraded，问答时检索工具自行降级；
* SSE 事件协议见 design doc §5.11；**答案会以 ``answer_token`` 逐字推送**——
  因为结构化输出走工具契约，答案正文在 ``submit_answer`` 的参数流里（见 streaming.py 的说明）；
* ``/ingest`` 用 ``X-Admin-Token`` 保护（对应 ``CSA_ADMIN_TOKEN``），未配置则直接禁用该接口。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from ..agent.builder import AgentBundle, build_agent, build_tools
from ..agent.schemas import SUBMIT_ANSWER_TOOL, extract_answer, iter_tool_calls
from ..config import get_settings
from ..ingest.pipeline import run_ingest
from ..retrieval.store import MilvusDocStore
from .schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HitlRequest,
    IngestResponse,
    StatsResponse,
)
from .streaming import (
    EVENT_ANSWER_TOKEN,
    EVENT_ANSWER_RESET,
    EVENT_CITATION,
    EVENT_ERROR,
    EVENT_FINAL,
    EVENT_INTERRUPT,
    EVENT_META,
    EVENT_TOKEN,
    EVENT_TOOL_CALL,
    ToolCallAccumulator,
    extract_partial_string,
    interrupt_requests,
    sse_event,
    text_of,
    upstream_citations,
)

app = FastAPI(title="Customer Service Assistant API", version="1.0.0")

_bundle: AgentBundle | None = None


def get_bundle() -> AgentBundle:
    """懒加载 Agent 单例（测试可注入假实现，见 tests/test_api.py）。"""
    global _bundle
    if _bundle is None:
        _bundle = build_agent(get_settings())
    return _bundle


def set_bundle(bundle: AgentBundle | None) -> None:
    global _bundle
    _bundle = bundle


# --------------------------------------------------------------------------- #
# 健康检查与索引
# --------------------------------------------------------------------------- #
@app.get("/api/v1/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    settings = get_settings()
    milvus: dict[str, Any] = {"reachable": False, "error": None}
    try:
        store = MilvusDocStore(settings)
        store.ensure_ready()
        stats = store.collection_stats()
        milvus.update(
            reachable=True,
            collection=stats["collection"],
            visible_count=stats["visible_count"],
            schema=stats["schema"],
        )
    except Exception as error:  # Milvus 挂了也要能回结果，方便前端与运维定位
        milvus["error"] = f"{type(error).__name__}: {error}"[:200]

    return HealthResponse(
        status="ok" if milvus["reachable"] else "degraded",
        milvus=milvus,
        model={
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "credentials": bool(settings.deepseek_api_key),
            "embed_model": settings.embed_model,
            "rerank_enabled": settings.rerank_enabled,
        },
        prompt_version=settings.prompt_version,
        tools=[getattr(tool, "name", "?") for tool in build_tools(settings)],
    )


@app.get("/api/v1/collection/stats", response_model=StatsResponse)
def collection_stats() -> StatsResponse:
    settings = get_settings()
    store = MilvusDocStore(settings)
    store.ensure_ready()
    stats = store.collection_stats()
    manifest = None
    manifest_path = Path(settings.index_manifest_path)
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:  # pragma: no cover - 清单损坏时不影响主流程
            manifest = None
    return StatsResponse(
        collection=stats["collection"],
        visible_count=stats["visible_count"],
        schema_name=stats["schema"],
        dimension=stats["dimension"],
        manifest=manifest,
    )


def require_admin_token(x_admin_token: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if not settings.admin_token:
        raise HTTPException(status_code=503, detail="未配置 CSA_ADMIN_TOKEN，索引接口已禁用")
    if x_admin_token != settings.admin_token:
        raise HTTPException(status_code=401, detail="X-Admin-Token 无效")


@app.post("/api/v1/ingest", response_model=IngestResponse, dependencies=[Depends(require_admin_token)])
async def ingest() -> IngestResponse:
    settings = get_settings()
    outcome = await asyncio.to_thread(run_ingest, settings)
    return IngestResponse(
        collection=f"{settings.milvus_db}.{settings.milvus_collection}",
        total_records=len(outcome.records),
        visible_before=outcome.visible_before,
        visible_after=outcome.visible_after,
        removed_doc_ids=outcome.removed_doc_ids,
        elapsed_seconds=round(outcome.elapsed, 2),
        manifest_path=str(settings.index_manifest_path),
    )


# --------------------------------------------------------------------------- #
# 问答：同步 + SSE 流式
# --------------------------------------------------------------------------- #
@app.post("/api/v1/chat/sync", response_model=ChatResponse)
async def chat_sync(request: ChatRequest) -> ChatResponse:
    from ..agent.builder import run_agent

    bundle = get_bundle()
    thread_id = request.thread_id or f"web-{uuid4().hex[:8]}"
    run = await asyncio.to_thread(run_agent, bundle, request.message, thread_id=thread_id, auto_approve=True)
    return ChatResponse(
        thread_id=thread_id,
        answer=run.answer,
        problem=run.problem,
        tool_calls=run.tool_calls,
        interrupts=run.interrupts,
        elapsed=round(run.elapsed, 2),
    )


async def chat_stream(request: ChatRequest) -> AsyncIterator[dict[str, str]]:
    """把 Agent 的流式输出翻译成 SSE 事件序列。"""
    bundle = get_bundle()
    thread_id = request.thread_id or f"web-{uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    accumulator = ToolCallAccumulator()
    active_answer_slot = None
    emitted_answer_chars = 0
    started = time.perf_counter()

    yield sse_event(
        EVENT_META,
        {"thread_id": thread_id, "prompt_version": bundle.prompt_version, "tools": bundle.tool_names()},
    )

    interrupted = False
    try:
        async for mode, payload in bundle.agent.astream(
            {"messages": [{"role": "user", "content": request.message}]},
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                chunk, metadata = payload
                if metadata.get("langgraph_node") != "model":
                    continue
                accumulator.add_chunks(getattr(chunk, "tool_call_chunks", None) or [])

                # 新的 submit_answer 调用开始（例如引用校验拒绝后的重试）→ 让前端清空上一次的预览。
                # 只比较"最新的那个槽位"，否则每来一批增量都会把历史槽位误判成新调用。
                submit_slots = accumulator.slots_named(SUBMIT_ANSWER_TOOL)
                if submit_slots and submit_slots[-1] is not active_answer_slot:
                    if active_answer_slot is not None:
                        # 只有"重试"才需要清空预览；首次提交时预览本来就是空的
                        yield sse_event(EVENT_ANSWER_RESET, {"reason": "new_submission"})
                    active_answer_slot = submit_slots[-1]
                    emitted_answer_chars = 0

                text = text_of(chunk)  # 已丢弃 reasoning_content
                if text:
                    yield sse_event(EVENT_TOKEN, {"text": text})

                # 结构化答案正文：从 submit_answer 的参数增量里实时提取
                if active_answer_slot is not None:
                    snapshot = extract_partial_string(active_answer_slot.args_blob)
                    if snapshot and len(snapshot) > emitted_answer_chars:
                        delta = snapshot[emitted_answer_chars :]
                        emitted_answer_chars = len(snapshot)
                        yield sse_event(EVENT_ANSWER_TOKEN, {"text": delta})
                continue

            for node, update in (payload or {}).items():
                if node == "__interrupt__":
                    interrupted = True
                    yield sse_event(
                        EVENT_INTERRUPT,
                        {"thread_id": thread_id, "action_requests": interrupt_requests(update)},
                    )
                    continue
                for message in (update or {}).get("messages", []) or []:
                    if getattr(message, "type", None) == "tool":
                        yield sse_event(EVENT_TOOL_CALL, {"name": getattr(message, "name", "?"), "status": "completed"})
                        for citation in upstream_citations(message):
                            yield sse_event(EVENT_CITATION, citation)
                    elif getattr(message, "type", None) == "ai":
                        for call in getattr(message, "tool_calls", None) or []:
                            yield sse_event(
                                EVENT_TOOL_CALL,
                                {"name": call.get("name"), "status": "requested", "args": call.get("args")},
                            )
    except Exception as error:  # 流中间出错也要让前端收到明确信号
        logger.exception("SSE 流式失败")
        yield sse_event(EVENT_ERROR, {"message": f"{type(error).__name__}: {error}"})
        return

    state = await bundle.agent.aget_state(config)
    values = getattr(state, "values", {}) or {}
    messages = values.get("messages", [])
    answer, problem = extract_answer(messages)
    yield sse_event(
        EVENT_FINAL,
        {
            "thread_id": thread_id,
            "answer": answer.model_dump() if answer else None,
            "problem": problem,
            "tool_calls": [name for name, _ in iter_tool_calls(messages)],
            "interrupted": interrupted,
            "elapsed": round(time.perf_counter() - started, 2),
        },
    )


@app.post("/api/v1/chat")
async def chat(request: ChatRequest) -> EventSourceResponse:
    return EventSourceResponse(chat_stream(request))


@app.post("/api/v1/hitl/{thread_id}", response_model=ChatResponse)
async def submit_hitl(thread_id: str, request: HitlRequest) -> ChatResponse:
    """提交人工审核决定并继续执行被中断的那一轮。"""
    from langgraph.types import Command

    bundle = get_bundle()
    config = {"configurable": {"thread_id": thread_id}}
    decisions = [decision.model_dump() for decision in request.decisions]
    started = time.perf_counter()
    state = await bundle.agent.ainvoke(Command(resume={"decisions": decisions}), config=config)
    messages = state.get("messages", [])
    answer, problem = extract_answer(messages)
    return ChatResponse(
        thread_id=thread_id,
        answer=answer,
        problem=problem,
        tool_calls=[name for name, _ in iter_tool_calls(messages)],
        elapsed=round(time.perf_counter() - started, 2),
    )


# --------------------------------------------------------------------------- #
# 演示页面（无前端依赖，直接消费 SSE）
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
