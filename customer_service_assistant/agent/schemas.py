"""结构化输出契约 + 引用校验。

**为什么不用 ``ToolStrategy``**（踩坑记录）：本项目的 DeepSeek 模型是思考模式模型，实测
三种原生结构化输出策略全部失败——``ToolStrategy`` / ``AutoStrategy`` 报
``400 Thinking mode does not support this tool_choice``，``ProviderStrategy`` 报
``400 This response_format type is unavailable now``；而不加 ``response_format`` 时**工具调用完全正常**。

因此改用**工具调用契约**：注册一个 ``submit_answer`` 工具，模型像调用别的工具一样提交结构化答案，
再从消息历史里取回参数做 Pydantic 校验。好处是 ``Answer`` schema、引用校验、后续评测都不受影响，
而且校验失败可以直接作为工具返回值回灌给模型，让它在同一轮里自我纠正（框架原生的重试语义）。

引用校验放在这个工具内部（设计文档写的是"after_model 中间件"，这里是有意偏离，理由见
``docs/architecture.md``：工具层拦截能让 Agent 自然重试，中间件里注入反馈消息会打乱图的终止条件）。
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Sequence

from langchain.tools import tool
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel, Field

SUBMIT_ANSWER_TOOL = "submit_answer"
RAG_TOOL = "search_knowledge_base"
CALC_TOOL_PREFIX = "calc_"
CALC_TOOL_EXTRA = ("check_data_retention_tool",)
BUSINESS_TOOL_NAMES = ("check_order_context", "escalate_to_human")

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


class Citation(BaseModel):
    """一条引用：必须能在检索片段里逐字找到。"""

    quote: str = Field(description="支撑结论的原文片段，必须逐字来自检索结果，不要改写")
    section: str = Field(default="", description="该片段所属章节，例如「六、退款规则」")
    chunk_id: int | None = Field(default=None, description="片段编号，可选")


class Answer(BaseModel):
    """客服 Agent 的最终输出契约。"""

    answer: str = Field(description="给用户看的自然语言回答")
    citations: list[Citation] = Field(default_factory=list, description="支撑答案的原文引用")
    key_facts: list[str] = Field(default_factory=list, description="结论涉及的关键数字与条件")
    confidence: Literal["high", "medium", "low"] = Field(default="medium", description="证据充分程度")
    need_human: bool = Field(default=False, description="是否需要转人工")
    human_reason: str | None = Field(default=None, description="转人工的原因")
    suggested_followups: list[str] = Field(default_factory=list, description="可推荐的追问")


# --------------------------------------------------------------------------- #
# 文本归一化与解析工具
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    """去掉空白、统一全半角与常见标点，用于"逐字匹配"的宽松比较。"""
    if not text:
        return ""
    table = str.maketrans(
        {"（": "(", "）": ")", "，": ",", "。": ".", "：": ":", "；": ";", "“": '"', "”": '"', "％": "%", "　": ""}
    )
    return re.sub(r"\s+", "", text.translate(table))


def _loads(content: Any) -> Any:
    """工具返回的 dict 会被序列化成 JSON 字符串，这里统一还原。"""
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None
    return content


def iter_tool_calls(messages: Sequence[Any]) -> list[tuple[str, dict]]:
    """按顺序取出所有工具调用：(工具名, 参数)。"""
    calls: list[tuple[str, dict]] = []
    for message in messages:
        if getattr(message, "type", None) == "ai":
            for call in getattr(message, "tool_calls", None) or []:
                calls.append((call.get("name", ""), call.get("args", {}) or {}))
    return calls


def collect_retrieved(messages: Sequence[Any]) -> list[dict]:
    """收集会话中所有检索到的片段（child 文本 + 回填的章节全文）。"""
    retrieved: list[dict] = []
    for message in messages:
        if getattr(message, "type", None) != "tool" or getattr(message, "name", None) != RAG_TOOL:
            continue
        payload = _loads(message.content)
        if not isinstance(payload, dict):
            continue
        for hit in payload.get("hits", []) or []:
            if isinstance(hit, dict):
                retrieved.append(hit)
    return retrieved


def collect_computed(messages: Sequence[Any]) -> list[dict]:
    """收集会话中所有计算结果（用于校验"数字是否来自工具而不是模型心算"）。"""
    computed: list[dict] = []
    for message in messages:
        name = getattr(message, "name", None) or ""
        if getattr(message, "type", None) != "tool":
            continue
        if not (name.startswith(CALC_TOOL_PREFIX) or name in CALC_TOOL_EXTRA):
            continue
        payload = _loads(message.content)
        if isinstance(payload, dict):
            computed.append(payload)
    return computed


def collect_tool_results(messages: Sequence[Any]) -> dict[str, list[dict]]:
    """按工具名收集所有工具的返回结果（评测与调试用）。"""
    results: dict[str, list[dict]] = {}
    for message in messages:
        name = getattr(message, "name", None)
        if getattr(message, "type", None) != "tool" or not name:
            continue
        payload = _loads(message.content)
        if isinstance(payload, dict):
            results.setdefault(name, []).append(payload)
    return results


def collect_provenance(messages: Sequence[Any]) -> list[dict]:
    """收集"数字来源"：除 submit_answer 外的所有工具返回（检索片段、计算结果、订单信息）。

    校验"数字是否心算"时不能只认计算工具——订单号里的 2026/0102、订单金额 199、
    支付日期、usage_ratio 都来自 check_order_context，漏掉它就会把正常回答判成心算。
    """
    provenance: list[dict] = []
    for name, payloads in collect_tool_results(messages).items():
        if name == SUBMIT_ANSWER_TOOL:
            continue
        provenance.extend(payloads)
    return provenance


def allowed_numbers(text: str) -> set[str]:
    """从文本里抽出所有数字，并补上等价的百分数/小数写法。

    工具返回 ``usage_ratio: 0.6667``，模型回答里写"66.7%"是等价的，不能判成心算。
    因此对每个数字补上"原值 / 百分数"两种形态，各取 0、1、2、4 位小数的写法。
    """

    def strip_trailing(value: str) -> str:
        return value.rstrip("0").rstrip(".") if "." in value else value

    allowed: set[str] = set()
    for token in NUMBER_RE.findall(text):
        allowed.add(token)
        try:
            value = float(token)
        except ValueError:  # pragma: no cover - 正则保证是数字
            continue
        for decimals in (0, 1, 2, 4):
            allowed.add(strip_trailing(f"{value:.{decimals}f}"))
            allowed.add(strip_trailing(f"{value * 100:.{decimals}f}"))
    return allowed


def extract_answer(messages: Sequence[Any]) -> tuple[Answer | None, str | None]:
    """从消息历史里取最后一次 ``submit_answer`` 调用并校验成 ``Answer``。

    返回 ``(answer, problem)``：取到合法答案时 problem 为 ``None``。
    """
    for name, args in reversed(iter_tool_calls(messages)):
        if name != SUBMIT_ANSWER_TOOL:
            continue
        try:
            return Answer(**args), None
        except Exception as error:  # 校验失败：把原因返回给调用方
            return None, f"submit_answer 参数校验失败：{error}"
    return None, "模型没有调用 submit_answer，未产出结构化答案"


# --------------------------------------------------------------------------- #
# 引用校验
# --------------------------------------------------------------------------- #
def verify_answer(answer: Answer, retrieved: Sequence[dict], provenance: Sequence[dict] = ()) -> list[str]:
    """返回校验问题列表（空列表表示通过）。"""
    problems: list[str] = []

    corpus = [normalize_text(hit.get("text", "")) + normalize_text(hit.get("context", "")) for hit in retrieved]
    corpus = [text for text in corpus if text]

    for index, citation in enumerate(answer.citations, start=1):
        quote = normalize_text(citation.quote)
        if not quote:
            problems.append(f"引用{index}的 quote 为空")
            continue
        if not any(quote in text for text in corpus):
            problems.append(
                f"引用{index}的 quote 在检索结果里找不到（可能被改写或来自模型自身知识）：{citation.quote[:40]!r}"
            )

    if not answer.citations and answer.confidence != "low" and not answer.need_human:
        problems.append("没有任何引用，但 confidence 不是 low 且 need_human=false：要么补引用，要么降低置信度/转人工")

    # 数字溯源：答案里的数字必须能在「引用原文」或「任意工具的返回值」里找到
    if answer.citations or answer.key_facts:
        cited_text = " ".join(citation.quote for citation in answer.citations)
        provenance_text = json.dumps(provenance, ensure_ascii=False) if provenance else ""
        trusted = allowed_numbers(cited_text) | allowed_numbers(provenance_text)
        for fact in answer.key_facts:
            for number in NUMBER_RE.findall(fact):
                if number not in trusted:
                    problems.append(
                        f"关键数字 {number}（来自 {fact[:60]!r}）既不在引用原文中，也不在任何工具返回值里，疑似心算"
                    )
    return problems


# --------------------------------------------------------------------------- #
# submit_answer 工具（工具调用契约的实现）
# --------------------------------------------------------------------------- #
@tool(SUBMIT_ANSWER_TOOL)
def submit_answer(
    answer: str,
    citations: list[Citation],
    key_facts: list[str],
    confidence: Literal["high", "medium", "low"],
    runtime: ToolRuntime,
    need_human: bool = False,
    human_reason: str | None = None,
    suggested_followups: list[str] | None = None,
) -> str:
    """提交最终答案（**必须调用**，否则用户看不到回答）。

    answer：给用户看的完整回答。
    citations：支撑回答的原文引用，每条的 quote 必须**逐字**来自 search_knowledge_base 返回的片段；
               没有引用就必须把 confidence 设为 low 或 need_human=true。
    key_facts：结论涉及的关键数字与条件，例如「超出 8000 次」「每 1000 次 8 元」。
    confidence：证据充分程度，high / medium / low。
    need_human：是否需要转人工（超出知识库范围、涉及合同定制、投诉等）。
    human_reason：需要转人工时说明原因。
    suggested_followups：可推荐的下一步问题。
    """
    messages = runtime.state.get("messages", []) if runtime and runtime.state else []
    candidate = Answer(
        answer=answer,
        citations=list(citations or []),
        key_facts=list(key_facts or []),
        confidence=confidence,
        need_human=need_human,
        human_reason=human_reason,
        suggested_followups=list(suggested_followups or []),
    )
    problems = verify_answer(candidate, collect_retrieved(messages), collect_provenance(messages))
    if problems:
        # 返回错误说明 → 模型在下一轮看到后自我修正（框架原生的"工具报错重试"语义）
        return "提交被拒绝：引用校验未通过，请修正后重新调用 submit_answer。问题如下：\n- " + "\n- ".join(problems)
    return "答案已提交并通过引用校验"
