"""离线索引 CLI：结构感知切分 → 向量化 → 幂等写入 Milvus。

用法::

    python scripts/ingest.py                 # 正式写入（幂等）
    python scripts/ingest.py --dry-run       # 只切分、打印章节与片段统计
    python scripts/ingest.py --preview 2     # 额外打印前 2 个 child 片段
    python scripts/ingest.py --no-compact    # 有文档被清理时不触发 compaction
"""

import _bootstrap  # noqa: F401  必须最先导入

import argparse

from loguru import logger

from customer_service_assistant.config import get_settings
from customer_service_assistant.ingest.pipeline import run_ingest
from customer_service_assistant.llm import setup_tracing
from customer_service_assistant.logging_setup import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Customer Service Assistant 知识库索引")
    parser.add_argument("--dry-run", action="store_true", help="只切分并打印统计，不向量化、不入库")
    parser.add_argument("--preview", type=int, default=0, help="打印前 N 个 child 片段")
    parser.add_argument("--no-compact", action="store_true", help="清理失效文档后不触发 compaction")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_dir)
    setup_tracing(settings)
    logger.info("配置快照: {}", settings.safe_summary())

    try:
        outcome = run_ingest(settings, dry_run=args.dry_run, compact=not args.no_compact)
    except RuntimeError as error:
        logger.error("{}", error)
        return 1

    for index, record in enumerate(outcome.children[: args.preview], start=1):
        logger.info(
            "child#{} [{}] section={} tier={}\n{}",
            index,
            record.uid,
            record.section,
            record.plan_tier or "-",
            record.text,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
