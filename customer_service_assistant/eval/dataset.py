"""LangSmith dataset + experiment（设计文档 §6.4）。

本地评测（``eval/runner.py``）负责产出可复现的 markdown 报告；
LangSmith 负责"在 UI 里按 prompt_version / retriever_config 对比不同实验"。
两者用**同一份**冻结集与**同一套**评测器，避免出现两套口径。

用法::

    python -m customer_service_assistant.eval.dataset --upload                  # 上传/复用 dataset
    python -m customer_service_assistant.eval.dataset --upload --recreate       # 重建（改了冻结集时用）
    python -m customer_service_assistant.eval.dataset --experiment --config configs/E_production.yaml --limit 20

需要教程根目录 `.env` 里的 ``LANGSMITH_API_KEY``（``LANGSMITH_ENDPOINT`` 可选，国内可用自建网关）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from loguru import logger

from ..config import Settings, get_settings
from ..llm import get_chat_model, setup_tracing
from ..logging_setup import setup_logging
from .evaluators import ItemResult, evaluate_rules, judge_correctness
from .pipelines import build_runner, load_config, run_item

DEFAULT_DATASET_NAME = "customer-service-assistant-golden-set-v1"
GOLDEN_PATH = Path("data/eval/golden_set.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangSmith dataset / experiment")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset", default=str(GOLDEN_PATH), help="本地冻结集 JSONL")
    parser.add_argument("--upload", action="store_true", help="上传（或复用）dataset")
    parser.add_argument("--recreate", action="store_true", help="先删除同名 dataset 再重建")
    parser.add_argument("--experiment", action="store_true", help="跑一次 experiment")
    parser.add_argument("--config", default="configs/E_production.yaml", help="experiment 用的消融配置")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0 = 全部）")
    parser.add_argument("--prefix", default=None, help="experiment 前缀（默认取配置名）")
    parser.add_argument("--concurrency", type=int, default=2, help="并发")
    return parser.parse_args()


def resolve(settings: Settings, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(settings.project_root) / candidate


def require_client(settings: Settings):
    if not settings.langsmith_api_key:
        raise SystemExit("缺少 LANGSMITH_API_KEY（教程根目录 .env），无法使用 LangSmith")
    from langsmith import Client

    return Client(api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint)


def load_items(settings: Settings, path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in resolve(settings, path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def upload(client: Any, settings: Settings, args: argparse.Namespace) -> Any:
    items = load_items(settings, args.dataset)
    if args.recreate and client.has_dataset(dataset_name=args.dataset_name):
        dataset = client.read_dataset(dataset_name=args.dataset_name)
        client.delete_dataset(dataset_id=dataset.id)
        logger.info("已删除同名 dataset {}", args.dataset_name)

    if client.has_dataset(dataset_name=args.dataset_name):
        dataset = client.read_dataset(dataset_name=args.dataset_name)
        existing = len(list(client.list_examples(dataset_id=dataset.id)))
        logger.info("dataset 已存在：{}（{} 条示例），本次不重复上传", args.dataset_name, existing)
        if existing != len(items):
            logger.warning("本地冻结集 {} 条 ≠ 远端 {} 条，建议加 --recreate 重建", len(items), existing)
        return dataset

    dataset = client.create_dataset(
        dataset_name=args.dataset_name,
        description="Customer Service Assistant 客服知识库评测集（100 题：40 事实 / 20 计算 / 20 边界 / 15 拒答 / 5 多轮）",
    )
    examples = [
        {
            "inputs": {"id": item["id"], "question": item["question"], "item": item},
            "outputs": {
                "ground_truth": item["ground_truth"],
                "expected_sections": item["expected_sections"],
                "expected_tool": item["expected_tool"],
                "must_include": item["must_include"],
                "refuse_expected": item["refuse_expected"],
            },
            "metadata": {"category": item["category"]},
        }
        for item in items
    ]
    client.create_examples(dataset_id=dataset.id, examples=examples)
    logger.info("已上传 {} 条示例到 dataset {}", len(examples), args.dataset_name)
    return dataset


def make_target(settings: Settings, args: argparse.Namespace):
    """experiment 的执行函数：本地同一套管线（保证"评测的就是线上跑的"）。"""
    config = load_config(args.config, settings)
    model = get_chat_model()
    runner = build_runner(config, settings, model=model)

    def target(inputs: dict) -> dict:
        record = run_item(runner, inputs["item"])
        return record.to_dict()

    return target


def make_evaluators(settings: Settings):
    model = get_chat_model()

    def rule_evaluator(run: Any, example: Any) -> dict:
        item = example.inputs["item"]
        outputs = run.outputs or {}
        record = ItemResult(**{key: value for key, value in outputs.items() if key in ItemResult.__dataclass_fields__})
        checks = evaluate_rules(item, record)
        record.checks = checks
        return {
            "key": "rule_answer_ok",
            "score": 1.0 if checks.get("rule_answer_ok") else 0.0,
            "comment": json.dumps({key: value for key, value in checks.items() if value is not None}, ensure_ascii=False),
        }

    def judge_evaluator(run: Any, example: Any) -> dict:
        item = example.inputs["item"]
        outputs = run.outputs or {}
        record = ItemResult(**{key: value for key, value in outputs.items() if key in ItemResult.__dataclass_fields__})
        judgement = judge_correctness(item, record, model)
        score = {"correct": 1.0, "partial": 0.5}.get(judgement["verdict"], 0.0)
        return {"key": "judge_correctness", "score": score, "comment": judgement.get("reason", "")}

    return [rule_evaluator, judge_evaluator]


def run_experiment(client: Any, settings: Settings, args: argparse.Namespace) -> int:
    from langsmith.evaluation import evaluate

    if not client.has_dataset(dataset_name=args.dataset_name):
        logger.error("dataset {} 不存在，先跑 --upload", args.dataset_name)
        return 1
    config = load_config(args.config, settings)
    data: Any = args.dataset_name
    if args.limit:
        dataset = client.read_dataset(dataset_name=args.dataset_name)
        data = list(client.list_examples(dataset_id=dataset.id, limit=args.limit))

    results = evaluate(
        make_target(settings, args),
        data=data,
        evaluators=make_evaluators(settings),
        experiment_prefix=args.prefix or f"p5-{config.name}",
        metadata={"config": config.as_dict(), "prompt_version": settings.prompt_version},
        max_concurrency=args.concurrency,
    )
    logger.info("experiment 完成：{}", results.experiment_name)
    print(f"\nexperiment: {results.experiment_name}")
    print(f"查看：{settings.langsmith_endpoint}/o/default/projects/p/{settings.langsmith_project}")
    return 0


def main() -> int:
    args = parse_args()
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_dir)
    setup_tracing(settings)
    if not (args.upload or args.experiment):
        raise SystemExit("需要 --upload 或 --experiment")

    client = require_client(settings)
    if args.upload:
        upload(client, settings, args)
    if args.experiment:
        return run_experiment(client, settings, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
