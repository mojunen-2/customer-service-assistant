"""把 LangGraph 的流式输出翻译成前端事件协议。

踩坑记录（实测得出，直接决定这里的实现）：

1. **思考模式模型会同时输出正文与 ``reasoning_content``**：一次问答里正文 212 段、推理 206 段。
   推理内容必须丢弃，否则等于把"内心独白"推给用户。
2. **结构化输出走工具契约，所以答案正文在工具参数里而不是 message content 里**：
   ``submit_answer`` 的 args 是增量 JSON 片段（实测 869 段），需要做「局部 JSON 字符串提取」
   才能把答案逐字推给前端。
3. **``tool_call_chunks`` 里只有首块带工具名**，其余块 ``name`` 为空。
4. **``index`` 不是全局唯一的，而是"每轮模型输出内"的序号**：实测第一轮的
   ``search_knowledge_base`` 与第二轮的 ``submit_answer`` 都是 ``index=0``。
   因此不能只用 index 累积（第一版就踩了：两次调用的参数被拼进同一个桶，答案被推了两遍）。
   正确做法是"**出现 name 即表示一次新调用开始**"——续传块的 name 必为空。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# SSE 事件类型（与设计文档 §5.11 的协议一致）
EVENT_META = "meta"
EVENT_TOKEN = "token"
EVENT_ANSWER_TOKEN = "answer_token"
EVENT_ANSWER_RESET = "answer_reset"
EVENT_TOOL_CALL = "tool_call"
EVENT_CITATION = "citation"
EVENT_INTERRUPT = "interrupt"
EVENT_FINAL = "final"
EVENT_ERROR = "error"

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f"}


def sse_event(event: str, data: Any) -> dict[str, str]:
    """sse-starlette 期望的事件字典。"""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return {"event": event, "data": payload}


def extract_partial_string(blob: str, key: str = "answer") -> str | None:
    """从**可能未闭合**的 JSON 片段里取出某个字符串字段的当前值。

    用于把 ``submit_answer`` 的参数流实时转成给用户看的文本。片段不完整时返回已解析出的部分，
    遇到转义序列或 ``\\uXXXX`` 没收完就停在最后一个完整字符。
    """
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"', blob)
    if not match:
        return None

    out: list[str] = []
    index, length = match.end(), len(blob)
    while index < length:
        char = blob[index]
        if char == "\\":
            if index + 1 >= length:
                break
            escaped = blob[index + 1]
            if escaped == "u":
                if index + 6 > length:
                    break
                try:
                    out.append(chr(int(blob[index + 2 : index + 6], 16)))
                except ValueError:
                    break
                index += 6
                continue
            out.append(_ESCAPES.get(escaped, escaped))
            index += 2
            continue
        if char == '"':
            break
        out.append(char)
        index += 1
    return "".join(out)


def _field(chunk: Any, key: str) -> Any:
    return chunk.get(key) if isinstance(chunk, dict) else getattr(chunk, key, None)


@dataclass
class ToolCallSlot:
    """一次工具调用的流式参数缓冲（对应"某轮输出里的第 index 个调用"）。"""

    index: int
    name: str | None = None
    fragments: list[str] = field(default_factory=list)

    @property
    def args_blob(self) -> str:
        return "".join(self.fragments)

    @property
    def key(self) -> str:
        return f"{self.index}:{self.name}"


@dataclass
class ToolCallAccumulator:
    """按"槽位"累积流式工具调用参数。"""

    slots: dict[int, list[ToolCallSlot]] = field(default_factory=dict)
    sequence: list[ToolCallSlot] = field(default_factory=list)

    def add_chunks(self, chunks: Iterable[Any]) -> list[ToolCallSlot]:
        """累积一批增量，返回本次被写入的槽位（含新建的，便于调用方识别"新调用开始"）。"""
        touched: list[ToolCallSlot] = []
        for chunk in chunks or []:
            index = _field(chunk, "index")
            if index is None:
                continue
            name = _field(chunk, "name")
            fragment = _field(chunk, "args")

            bucket = self.slots.setdefault(index, [])
            if (name and bucket and bucket[-1].name is not None) or not bucket:
                slot = ToolCallSlot(index=index)
                bucket.append(slot)
                self.sequence.append(slot)
            slot = bucket[-1]
            if name:
                slot.name = name
            if fragment:
                slot.fragments.append(fragment)
            touched.append(slot)
        return touched

    def slots_named(self, name: str) -> list[ToolCallSlot]:
        return [slot for slot in self.sequence if slot.name == name]

    def all_names(self) -> list[str]:
        return [slot.name or "?" for slot in self.sequence]


def text_of(chunk: Any) -> str:
    """取正文内容；思考模式的 ``reasoning_content`` 一律丢弃。"""
    content = getattr(chunk, "content", "") or ""
    if isinstance(content, list):  # content block 形态
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content if isinstance(content, str) else ""


def reasoning_of(chunk: Any) -> str:
    return (getattr(chunk, "additional_kwargs", {}) or {}).get("reasoning_content") or ""


def citations_from_tool_payload(payload: dict) -> list[dict]:
    """把 ``search_knowledge_base`` 的返回整理成 citation 事件的数据。"""
    citations = []
    for hit in payload.get("hits", []) or []:
        if not isinstance(hit, dict):
            continue
        context = hit.get("context") or ""
        citations.append(
            {
                "chunk_id": hit.get("chunk_id"),
                "section": hit.get("section"),
                "score": hit.get("rerank_score") if hit.get("rerank_score") is not None else hit.get("score"),
                "quote": (hit.get("text") or "")[:120],
                "context_preview": context[:160],
            }
        )
    return citations


def upstream_citations(message: Any) -> list[dict]:
    """从一条 ToolMessage 里解析 citation 事件（非检索工具返回空列表）。"""
    if getattr(message, "name", None) != "search_knowledge_base":
        return []
    content = getattr(message, "content", None)
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return []
    return citations_from_tool_payload(content) if isinstance(content, dict) else []


def interrupt_requests(interrupts: Sequence[Any]) -> list[dict]:
    """整理 HITL 中断：工具名在 ``action_requests[].name``（不是 action_name）。"""
    requests: list[dict] = []
    for item in interrupts or []:
        value = getattr(item, "value", item)
        if not isinstance(value, dict):
            continue
        for request in value.get("action_requests", []) or []:
            requests.append({"name": request.get("name"), "args": request.get("args")})
    return requests
