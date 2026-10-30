"""消融实验的执行管线（设计文档 §6.3 的 A~E 组）。

五组配置只差"开了哪些模块"，其余全部相同（同一模型、同一提示词、同一评测集），
这样每一组的增益才能归因到具体模块：

| 组 | 检索 | 上下文 | 输出 |
|---|---|---|---|
| A | 旧 collection（P0 quick-create，45 段）+ 单路 dense top5 | 片段本身 | 自由文本 |
| B | v2 collection + 单路 dense top5 | small-to-big 回填章节 | 自由文本 |
| C | B + BM25 混合（weighted 融合） | 同上 | 自由文本 |
| D | C + bge-reranker 重排 | 同上 | 自由文本 |
| E | D + 两级阈值拒答 | 同上 | 工具化 Agent：结构化答案 + 引用校验 |

A~D 用与 notebook 一致的 baseline 提示词（``BASELINE_SYSTEM_PROMPT``），
E 用 P3 的工具化 Agent（``build_agent``）。多轮题在**所有组**里都用同样的方式注入历史
（拼进用户消息），保证可比；P3 真正的 checkpointer 多轮验收见 ``scripts/check_multiturn.py``。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from ..agent.baseline_agent import build_baseline_agent, extract_final_text
from ..agent.builder import build_agent, run_agent
from ..agent.prompts import build_user_prompt
from ..agent.schemas import collect_tool_results
from ..config import Settings, get_settings
from ..retrieval.retriever import build_context, retrieve_with_report
from .evaluators import ItemResult, evaluate_rules


@dataclass(slots=True)
class GroupConfig:
    """一组消融配置。"""

    name: str
    description: str = ""
    answer_mode: str = "plain"  # plain（A~D）| agent（E）
    collection: str = "customer_service_docs_v2"
    mode: str = "dense"
    rerank: bool = False
    top_n: int = 5
    expand_parent: bool = False
    apply_threshold: bool = False
    structured: bool = False
    calibration: dict[str, Any] = field(default_factory=dict)

    def settings_for(self, settings: Settings) -> Settings:
        """把该组的检索参数覆盖到全局配置上（返回副本，不改全局单例）。"""
        return settings.model_copy(
            update={
                "milvus_collection": self.collection,
                "retrieval_mode": self.mode,
                "rerank_enabled": self.rerank,
                "rerank_top_n": self.top_n,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "answer_mode": self.answer_mode,
            "collection": self.collection,
            "mode": self.mode,
            "rerank": self.rerank,
            "top_n": self.top_n,
            "expand_parent": self.expand_parent,
            "apply_threshold": self.apply_threshold,
            "structured": self.structured,
        }


def load_config(path: str | Path, settings: Settings | None = None) -> GroupConfig:
    """从 YAML 读取一组配置。"""
    settings = settings or get_settings()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(settings.project_root) / candidate
    payload = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    retrieval = payload.pop("retrieval", {}) or {}
    payload.setdefault("name", candidate.stem)
    merged = {**payload, **retrieval}
    known = {key: value for key, value in merged.items() if key in GroupConfig.__dataclass_fields__}
    unknown = set(merged) - set(known)
    if unknown:
        logger.warning("配置 {} 里有未识别字段（已忽略）：{}", candidate.name, sorted(unknown))
    return GroupConfig(**known)


def build_message(item: dict) -> str:
    """把多轮历史拼进用户消息（A~E 统一做法，保证口径一致）。"""
    history = item.get("history") or []
    if not history:
        return item["question"]
    lines = []
    for turn in history:
        role = "用户" if turn.get("role") == "user" else "客服"
        lines.append(f"{role}：{turn.get('content', '')}")
    lines.append(f"用户：{item['question']}")
    return "对话历史：\n" + "\n".join(lines)


def history_text(item: dict) -> str | None:
    """多轮题的历史文本，交给检索层做指代消解（``CSA_CONDENSE_WITH_HISTORY``）。"""
    history = item.get("history") or []
    if not history:
        return None
    lines = [
        f"{'用户' if turn.get('role') == 'user' else '客服'}：{turn.get('content', '')}" for turn in history
    ]
    return "\n".join(lines)


def _sum_usage(messages: Any) -> dict[str, int]:
    prompt = completion = 0
    for message in messages or []:
        usage = getattr(message, "usage_metadata", None) or {}
        prompt += int(usage.get("input_tokens") or 0)
        completion += int(usage.get("output_tokens") or 0)
    return {"input": prompt, "output": completion, "total": prompt + completion}


def derive_refused(tool_results: dict[str, list[dict]]) -> bool:
    """从工具轨迹里推导"这次回答有没有依据"。

    只有**所有** `search_knowledge_base` 调用都返回 refused=true 才算没有依据；
    中间某次查询被拒、随后换词命中，不影响最终证据的存在
    （P5 实测：按"任一次拒答"标记会把正确回答误判成误拒）。
    """
    searches = tool_results.get("search_knowledge_base", [])
    return bool(searches) and all(payload.get("refused") for payload in searches)


class PlainRunner:
    """A~D 组：检索 → 上下文 → baseline 提示词 → 自由文本回答。"""

    def __init__(self, config: GroupConfig, settings: Settings, model: Any | None = None) -> None:
        self.config = config
        self.settings = settings
        self.agent = build_baseline_agent(model)

    def run(self, item: dict) -> ItemResult:
        config = self.config
        group_settings = config.settings_for(self.settings)
        message = build_message(item)
        started = time.perf_counter()
        report = retrieve_with_report(
            item["question"],
            history=history_text(item),
            limit=config.top_n,
            settings=group_settings,
            mode=config.mode,
            rerank=config.rerank,
            expand_parent=config.expand_parent,
            apply_threshold=config.apply_threshold,
        )
        retrieval_elapsed = time.perf_counter() - started
        context = build_context(report.hits)
        generation_started = time.perf_counter()
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": build_user_prompt(item["question"], context)}]}
        )
        generation_elapsed = time.perf_counter() - generation_started
        messages = result.get("messages", [])
        answer = extract_final_text(result)
        record = ItemResult(
            id=item["id"],
            category=item["category"],
            question=item["question"],
            config=config.name,
            answer=answer,
            has_tools=False,
            structured=False,
            citations=[],
            key_facts=[],
            confidence=None,
            need_human=None,
            tool_calls=[],
            tool_results={},
            retrieved=[
                {"text": hit.text, "context": hit.context_text, "section": hit.section} for hit in report.hits
            ],
            retrieved_sections=sorted({hit.section for hit in report.hits if hit.section}),
            refused=report.refused,
            latency={
                "retrieval": round(retrieval_elapsed, 3),
                "generation": round(generation_elapsed, 3),
                "total": round(retrieval_elapsed + generation_elapsed, 3),
            },
            tokens=_sum_usage(messages),
        )
        record.checks = evaluate_rules(item, record)
        return record


class AgentRunner:
    """E 组：P3 工具化 Agent（结构化答案 + 引用校验 + 拒答 + 计算工具）。"""

    def __init__(self, config: GroupConfig, settings: Settings, model: Any | None = None) -> None:
        self.config = config
        self.settings = config.settings_for(settings)
        self.bundle = build_agent(self.settings, model=model)

    def run(self, item: dict) -> ItemResult:
        thread_id = f"eval-{self.config.name}-{item['id']}"
        run = run_agent(self.bundle, build_message(item), thread_id=thread_id)
        answer = run.answer
        tool_results = collect_tool_results(run.messages)
        refused = derive_refused(tool_results)
        record = ItemResult(
            id=item["id"],
            category=item["category"],
            question=item["question"],
            config=self.config.name,
            answer=(answer.answer if answer else (run.problem or "")) or "",
            has_tools=True,
            structured=answer is not None,
            citations=[citation.model_dump() for citation in answer.citations] if answer else [],
            key_facts=list(answer.key_facts) if answer else [],
            confidence=answer.confidence if answer else None,
            need_human=answer.need_human if answer else None,
            tool_calls=list(run.tool_calls),
            tool_results=tool_results,
            retrieved=[],
            retrieved_sections=[],
            refused=refused,
            latency={"total": round(run.elapsed, 3)},
            tokens=_sum_usage(run.messages),
            error=run.problem,
        )
        # E 组的检索片段藏在工具返回里：抽出来用于引用校验与 faithfulness 判据
        for payloads in record.tool_results.get("search_knowledge_base", []):
            for hit in payloads.get("hits", []) or []:
                record.retrieved.append(hit)
                if hit.get("section"):
                    record.retrieved_sections.append(hit["section"])
        record.retrieved_sections = sorted(set(record.retrieved_sections))
        record.checks = evaluate_rules(item, record)
        return record


def build_runner(config: GroupConfig, settings: Settings, model: Any | None = None):
    if config.answer_mode == "agent":
        return AgentRunner(config, settings, model=model)
    return PlainRunner(config, settings, model=model)


def run_item(runner: Any, item: dict, judge_models: dict[str, Any] | None = None) -> ItemResult:
    """跑一道题并（可选）接上 judge。任何异常都记进 ``error``，不让整批评测崩掉。"""
    from .evaluators import judge_correctness, judge_faithfulness

    try:
        record = runner.run(item)
    except Exception as error:  # 单题失败不应中断整批评测
        logger.exception("题目 {} 执行失败", item.get("id"))
        record = ItemResult(
            id=item["id"],
            category=item["category"],
            question=item["question"],
            config=getattr(runner, "config", GroupConfig(name="?")).name,
            structured=False,
            error=f"{type(error).__name__}: {error}",
        )
        record.checks = evaluate_rules(item, record)
        return record

    if judge_models:
        correctness_model = judge_models.get("correctness")
        faithfulness_model = judge_models.get("faithfulness") or correctness_model
        if correctness_model is not None:
            try:
                record.judge["correctness"] = judge_correctness(item, record, correctness_model)
            except Exception as error:  # judge 失败不影响规则判据
                logger.warning("judge(correctness) 失败 {}：{}", item.get("id"), error)
                record.judge["correctness"] = {"version": "unknown", "verdict": "unknown", "reason": str(error)}
        if faithfulness_model is not None:
            try:
                record.judge["faithfulness"] = judge_faithfulness(item, record, faithfulness_model)
            except Exception as error:
                logger.warning("judge(faithfulness) 失败 {}：{}", item.get("id"), error)
    return record
