"""评测器：规则判定（可判定指标）+ LLM judge（软性指标）。

设计文档 §6.2 的指标分两类，本项目刻意分开实现，避免"用 LLM 判 LLM"的自证循环：

* **规则评测器**（零随机性、零成本）：数值精确率、must_include / must_not_include、
  拒答准确率、路由准确率、引用支撑（逐字子串校验）、期望章节命中；
* **LLM judge**（版本化提示词，见 ``judge_prompts.py``）：答案正确性、faithfulness。

每条记录的判定结果都带 ``None`` 语义：**不适用的判据返回 None 而不是 False**，
例如 A 组（无工具、无引用）没有路由与引用指标，它们不参与聚合。
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from loguru import logger

from ..agent.schemas import normalize_text
from ..config import Settings, get_settings
from .judge_prompts import (
    CORRECTNESS_PROMPT,
    CORRECTNESS_PROMPT_VERSION,
    FAITHFULNESS_PROMPT,
    FAITHFULNESS_PROMPT_VERSION,
    JUDGE_PROMPT_VERSION,
    judge_prompt_hash,
    judge_prompt_hashes,
)

# 结构化回答（E 组）的"答不上来"判据：只认**第一人称/明确的无法回答**表述。
# P5 实测两次踩坑：词表太宽会把"抱歉""建议联系人工"这类正常服务用语算成误拒；
# 而"知识库里没有可对外报出的标价"这种**解释政策**的句子也不该算（它不是"答不上来"）。
STRUCTURED_REFUSAL_MARKERS: tuple[str, ...] = (
    "无法确认",
    "不能确认",
    "无法回答",
    "无法提供",
    "无法给出",
    "不能给出",
    "无法确定",
    "回答不了",
    "帮不上忙",
    "我不知道",
    "我查不到",
    "没有相关依据",
    "没有相关信息",
    "未覆盖",
    "不在知识库",
    "超出知识库",
    "不在我能查到",
)
# 宽松词表：只用于 A~D 组的自由文本回答（没有结构化字段可依赖）
LOOSE_REFUSAL_MARKERS: tuple[str, ...] = (
    *STRUCTURED_REFUSAL_MARKERS,
    "没有相关",
    "没有相关信息",
    "抱歉",
    "建议联系人工",
    "建议转人工",
    "请联系人工客服",
)

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# 否定/疑问前缀：命中这些词说明该短语出现在"否定句"或"用户的疑问复述"里，不算编造
NEGATION_PREFIXES: tuple[str, ...] = ("不", "未", "没", "无法", "不能", "非", "否", "未支持", "无需")


@dataclass(slots=True)
class ItemResult:
    """一道题的原始执行记录（评测的"事实层"，markdown 报告由它渲染）。"""

    id: str
    category: str
    question: str
    config: str = ""
    has_tools: bool = False
    answer: str = ""
    structured: bool = True
    citations: list[dict[str, Any]] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    confidence: str | None = None
    need_human: bool | None = None
    tool_calls: list[str] = field(default_factory=list)
    tool_results: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    retrieved_sections: list[str] = field(default_factory=list)
    refused: bool = False
    latency: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    checks: dict[str, Any] = field(default_factory=dict)
    judge: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "config": self.config,
            "question": self.question,
            "has_tools": self.has_tools,
            "answer": self.answer,
            "structured": self.structured,
            "citations": self.citations,
            "key_facts": self.key_facts,
            "confidence": self.confidence,
            "need_human": self.need_human,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "retrieved_sections": self.retrieved_sections,
            "retrieved": self.retrieved,
            "refused": self.refused,
            "latency": self.latency,
            "tokens": self.tokens,
            "error": self.error,
            "checks": self.checks,
            "judge": self.judge,
        }


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _norm(text: str | None) -> str:
    return normalize_text(text or "")


def number_variants(value: float) -> list[str]:
    """数值的常见写法（整数、2 位小数、百分数），避免因格式差异误判。"""

    def trim(text: str) -> str:
        return text.rstrip("0").rstrip(".") if "." in text else text

    variants = {f"{value:g}", f"{value:.2f}", f"{value * 100:g}%", f"{value * 100:.1f}%"}
    return [trim(item) for item in variants if item]


# 只用于工具值比对的字段：``overage`` 是"题面总调用量 - 额度"的差，
# 容易与题面数字互为子串（"18000" 里就含 "8000"），放在回答文本判据里会误判为通过。
WEAK_NUMBER_KEYS = {"overage"}


def missing_numbers(expected: dict[str, Any], haystack: str) -> list[str]:
    """返回 expected 里"没有出现在回答中"的数值字段。

    跳过三类：bool/str 字段（由 must_include 判据覆盖）、``WEAK_NUMBER_KEYS``、
    以及 0（任何含 0 的文本都能匹配，判据没有区分度，交给 must_include_any）。
    """
    text = _norm(haystack)
    missing: list[str] = []
    for key, value in (expected or {}).items():
        if isinstance(value, bool) or isinstance(value, str) or value is None or key in WEAK_NUMBER_KEYS:
            continue
        if float(value) == 0.0:
            continue
        if not any(_norm(variant) in text for variant in number_variants(float(value))):
            missing.append(f"{key}={value}")
    return missing


def tool_values_match(expected_tool: str | None, tool_results: dict[str, list[dict]], expected: dict) -> bool | None:
    """工具返回值是否与人工手算的 expected 一致（不适用时返回 None）。"""
    if not expected_tool or not expected:
        return None
    payloads = tool_results.get(expected_tool, [])
    if not payloads:
        return False
    for payload in payloads:
        if all(payload.get(key) == value for key, value in expected.items()):
            return True
    return False


def refusal_signal(result: ItemResult) -> bool:
    """一次回答是否表现为"答不上来"（不包含"转人工建单"这种积极处理）。"""
    if result.refused:
        return True
    text = f"{result.answer} {' '.join(result.key_facts)}"
    if result.structured:
        # 结构化答案只有两种"答不上来"：检索层面直接拒答（refused），
        # 或者回答里明确写出"无法确认/知识库没有相关依据"这类断言。
        # 注意：不能因为 need_human=true 就判定误拒——客服把复杂问题转人工是正常动作。
        return any(marker in text for marker in STRUCTURED_REFUSAL_MARKERS)
    return any(marker in text for marker in LOOSE_REFUSAL_MARKERS) or not text.strip()


def contains_claim(haystack: str, phrase: str) -> bool:
    """判断 haystack 里是否存在 phrase 的**肯定式**出现。

    `must_not_include` 要抓的是"编造了某项能力"，而下面两种情况不该判为编造：

    1. 否定语境：`不支持 OCR 图片解析` 里含有子串 `支持 OCR 图片解析`；
    2. 复述用户的疑问：`关于「基础版是否支持绑定自定义域名」这个问题…`。

    因此命中位置前 3 个字符里若出现否定/疑问词，就跳过这次命中。
    """
    target = _norm(phrase)
    if not target:
        return False
    text = _norm(haystack)
    index = text.find(target)
    while index != -1:
        prefix = text[max(0, index - 3) : index]
        if not any(negation in prefix for negation in NEGATION_PREFIXES):
            return True
        index = text.find(target, index + 1)
    return False


# --------------------------------------------------------------------------- #
# 规则评测
# --------------------------------------------------------------------------- #
def evaluate_rules(item: dict, result: ItemResult) -> dict[str, Any]:
    """对一道题做全部规则判定，返回判定字典（None = 该判据不适用）。"""
    answer_text = f"{result.answer} {' '.join(result.key_facts)}"
    normalized_answer = _norm(answer_text)
    corpus = [_norm(entry.get("text", "")) + _norm(entry.get("context", "")) for entry in result.retrieved]
    corpus = [text for text in corpus if text]

    must_include = item.get("must_include") or []
    must_include_any = item.get("must_include_any") or []
    must_not_include = item.get("must_not_include") or []

    checks: dict[str, Any] = {}
    checks["must_include_ok"] = all(_norm(part) in normalized_answer for part in must_include) if must_include else None
    checks["must_include_any_ok"] = (
        any(_norm(part) in normalized_answer for part in must_include_any) if must_include_any else None
    )
    checks["must_not_include_ok"] = (
        not any(contains_claim(answer_text, part) for part in must_not_include) if must_not_include else None
    )

    expected_values = item.get("expected_values") or {}
    missing = missing_numbers(expected_values, answer_text)
    checks["numbers_ok"] = (not missing) if expected_values else None
    checks["missing_numbers"] = missing
    checks["tool_values_ok"] = tool_values_match(item.get("expected_tool"), result.tool_results, expected_values)
    if not result.has_tools:
        # 无工具的组（A~D）：路由与"工具返回值"不适用，返回 None 而不是 False
        checks["tool_values_ok"] = None
        checks["routing_ok"] = None
    else:
        checks["routing_ok"] = (item["expected_tool"] in result.tool_calls) if item.get("expected_tool") else None

    if item.get("category") == "calc" and item.get("expected_values"):
        # 数值精确率：有工具的组要求"工具返回值与手算一致"，无工具的组只看回答里有没有正确数值
        checks["numeric_exactness"] = (
            bool(checks["tool_values_ok"] and checks["numbers_ok"])
            if checks["tool_values_ok"] is not None
            else checks["numbers_ok"]
        )
    else:
        checks["numeric_exactness"] = None

    refused_here = refusal_signal(result)
    if item.get("refuse_expected"):
        checks["refusal_ok"] = bool(refused_here and checks["must_not_include_ok"] is not False)
        checks["over_refusal"] = None
    else:
        checks["refusal_ok"] = None
        checks["over_refusal"] = bool(refused_here)

    if result.structured and result.citations:
        checks["citation_rule_ok"] = all(
            any(_norm(citation.get("quote", "")) in text for text in corpus) for citation in result.citations
        )
    elif not result.structured:
        checks["citation_rule_ok"] = None
    else:
        checks["citation_rule_ok"] = None  # 无引用的结构化回答：由 judge 与 faithfulness 兜底

    expected_sections = item.get("expected_sections") or []
    retrieved_sections = set(result.retrieved_sections) | {
        entry.get("section") for entry in result.retrieved if entry.get("section")
    }
    checks["section_hit"] = bool(set(expected_sections) & retrieved_sections) if expected_sections else None

    flags = [
        checks["must_include_ok"],
        checks["must_include_any_ok"],
        checks["must_not_include_ok"],
        checks["numbers_ok"],
        checks["tool_values_ok"],
        checks["refusal_ok"],
    ]
    applicable = [flag for flag in flags if flag is not None]
    if item.get("refuse_expected"):
        ok = all(applicable)
    else:
        ok = all(applicable) and not checks["over_refusal"] and bool(answer_text.strip())
    checks["rule_answer_ok"] = bool(ok)
    return checks


# --------------------------------------------------------------------------- #
# LLM judge
# --------------------------------------------------------------------------- #
def parse_judge_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _judge(model: Any, prompt: str) -> dict[str, Any]:
    response = model.invoke([{"role": "user", "content": prompt}])
    content = response.content if isinstance(response.content, str) else str(response.content)
    payload = parse_judge_json(content)
    if not payload:
        payload = {"verdict": "unknown", "reason": f"judge 输出无法解析：{content[:120]}"}
    return payload


def judge_correctness(
    item: dict,
    result: ItemResult,
    model: Any,
    *,
    max_answer_chars: int = 4000,
) -> dict[str, Any]:
    """LLM judge：答案与标准答案的一致性（correct / partial / incorrect）。"""
    prompt = CORRECTNESS_PROMPT.format(
        question=item["question"],
        reference=item.get("ground_truth", ""),
        answer=(result.answer or "（没有返回答案）")[:max_answer_chars],
    )
    payload = _judge(model, prompt)
    verdict = str(payload.get("verdict", "unknown")).strip().lower()
    if verdict not in {"correct", "partial", "incorrect"}:
        verdict = "unknown"
    return {
        "version": CORRECTNESS_PROMPT_VERSION,
        "judge": "correctness",
        "verdict": verdict,
        "reason": payload.get("reason", ""),
    }


def judge_faithfulness(
    item: dict,
    result: ItemResult,
    model: Any,
    *,
    max_context_chars: int = 6000,
    max_answer_chars: int = 4000,
) -> dict[str, Any]:
    """LLM judge：回答里的结论是否都能被检索上下文支撑（幻觉率）。"""
    if result.refused:
        return {
            "version": FAITHFULNESS_PROMPT_VERSION,
            "judge": "faithfulness",
            "unsupported": [],
            "reason": "拒答，无编造",
        }
    if not result.retrieved:
        return {
            "version": FAITHFULNESS_PROMPT_VERSION,
            "judge": "faithfulness",
            "unsupported": [] if refusal_signal(result) else ["没有检索上下文，回答无法被支撑"],
            "reason": "无检索上下文",
        }
    context = "\n\n".join(
        (entry.get("context") or entry.get("text") or "") for entry in result.retrieved
    )[:max_context_chars]
    prompt = FAITHFULNESS_PROMPT.format(
        context=context,
        answer=(result.answer or "（没有返回答案）")[:max_answer_chars],
    )
    payload = _judge(model, prompt)
    unsupported = payload.get("unsupported")
    if not isinstance(unsupported, list):
        unsupported = []
    return {
        "version": FAITHFULNESS_PROMPT_VERSION,
        "judge": "faithfulness",
        "unsupported": [str(item) for item in unsupported],
        "reason": payload.get("reason", ""),
    }


# --------------------------------------------------------------------------- #
# 聚合
# --------------------------------------------------------------------------- #
def _rate(values: Iterable[bool | None]) -> dict[str, Any]:
    items = [bool(value) for value in values if value is not None]
    if not items:
        return {"value": None, "n": 0, "passed": 0}
    return {"value": round(sum(items) / len(items), 4), "n": len(items), "passed": sum(items)}


def _percentile(values: Sequence[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(int(round(ratio * (len(ordered) - 1))), len(ordered) - 1)
    return round(ordered[index], 3)


def aggregate(records: Sequence[ItemResult]) -> dict[str, Any]:
    """把逐题记录聚合成消融表需要的指标。"""
    if not records:
        return {"n": 0}

    def checks(item_id: str, key: str) -> bool | None:
        for record in records:
            if record.id == item_id:
                return record.checks.get(key)
        return None

    def values(key: str) -> list[bool | None]:
        return [record.checks.get(key) for record in records]

    judge_verdicts = [record.judge.get("correctness", {}).get("verdict") for record in records]
    judge_applicable = [verdict for verdict in judge_verdicts if verdict in {"correct", "partial", "incorrect"}]
    latency = [record.latency.get("total", 0.0) for record in records if record.latency.get("total")]
    retrieval_latency = [record.latency.get("retrieval", 0.0) for record in records if record.latency.get("retrieval")]
    generation_latency = [record.latency.get("generation", 0.0) for record in records if record.latency.get("generation")]
    inputs = [record.tokens.get("input", 0) for record in records if record.tokens]
    outputs = [record.tokens.get("output", 0) for record in records if record.tokens]

    faithfulness_values: list[bool | None] = []
    for record in records:
        judgement = record.judge.get("faithfulness")
        if judgement is None:
            faithfulness_values.append(None)
        else:
            faithfulness_values.append(not judgement.get("unsupported"))

    calc_records = [record for record in records if record.category == "calc"]
    refuse_records = [record for record in records if record.category == "refuse"]
    answerable_records = [record for record in records if record.category != "refuse"]

    def sub_rate(subset: Sequence[ItemResult], key: str) -> dict[str, Any]:
        return _rate(record.checks.get(key) for record in subset)

    metrics = {
        "n": len(records),
        "categories": {category: sum(1 for r in records if r.category == category) for category in sorted({r.category for r in records})},
        "rule_answer_accuracy": _rate(values("rule_answer_ok")),
        "must_include_pass": _rate(values("must_include_ok")),
        "must_include_any_pass": _rate(values("must_include_any_ok")),
        "must_not_include_pass": _rate(values("must_not_include_ok")),
        "numeric_exactness": sub_rate(calc_records, "numeric_exactness"),
        "numbers_present": sub_rate(calc_records, "numbers_ok"),
        "tool_value_match": sub_rate(calc_records, "tool_values_ok"),
        "routing_accuracy": _rate(values("routing_ok")),
        "refusal_accuracy": sub_rate(refuse_records, "refusal_ok"),
        "over_refusal_rate": _rate(values("over_refusal")),
        "citation_precision": _rate(values("citation_rule_ok")),
        "citation_coverage": _rate(
            bool(record.citations) if record.structured and record.category != "refuse" else None
            for record in answerable_records
        ),
        "section_recall": _rate(values("section_hit")),
        "judge_correctness_strict": _rate(verdict == "correct" for verdict in judge_verdicts),
        "judge_correctness_lenient": (
            {
                "value": round(
                    sum({"correct": 1.0, "partial": 0.5}.get(verdict, 0.0) for verdict in judge_applicable)
                    / len(judge_applicable),
                    4,
                ),
                "n": len(judge_applicable),
                "passed": sum(1 for verdict in judge_applicable if verdict == "correct"),
            }
            if judge_applicable
            else {"value": None, "n": 0, "passed": 0}
        ),
        "faithfulness": _rate(faithfulness_values),
        "judged_items": len(judge_applicable),
        "structured_rate": _rate(record.structured for record in records),
        "errors": sum(1 for record in records if record.error),
        "latency": {
            "p50": _percentile(latency, 0.5),
            "p95": _percentile(latency, 0.95),
            "mean_retrieval": round(statistics.fmean(retrieval_latency), 3) if retrieval_latency else None,
            "mean_generation": round(statistics.fmean(generation_latency), 3) if generation_latency else None,
        },
        "tokens": {
            "mean_input": round(statistics.fmean(inputs), 1) if inputs else None,
            "mean_output": round(statistics.fmean(outputs), 1) if outputs else None,
            "total_input": sum(inputs),
            "total_output": sum(outputs),
        },
        "judge": {
            "prompt_version": JUDGE_PROMPT_VERSION,
            "prompt_hash": judge_prompt_hash(),
            "prompts": judge_prompt_hashes(),
            "correctness_version": CORRECTNESS_PROMPT_VERSION,
            "faithfulness_version": FAITHFULNESS_PROMPT_VERSION,
        },
    }
    logger.debug("聚合完成：{} 条记录", len(records))
    return metrics


def format_metrics_line(metrics: dict[str, Any]) -> str:
    """命令行一行摘要。"""
    def show(key: str) -> str:
        entry = metrics.get(key) or {}
        if not entry.get("n"):
            return "n/a"
        return f"{entry['passed']}/{entry['n']}={entry['value']:.1%}"

    return (
        f"n={metrics['n']} 规则正确率={show('rule_answer_accuracy')} judge={show('judge_correctness_strict')} "
        f"数值={show('numeric_exactness')} 拒答={show('refusal_accuracy')} 误拒={show('over_refusal_rate')} "
        f"引用={show('citation_precision')} 忠实={show('faithfulness')} "
        f"p95={(metrics['latency'] or {}).get('p95')}s"
    )
