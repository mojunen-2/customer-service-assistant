"""日志初始化：loguru 控制台 + 文件双输出。

Windows 控制台默认不是 UTF-8，中文日志会出现乱码，因此这里统一把
stdout/stderr 重配置为 UTF-8（失败时静默跳过，不影响业务）。
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
    "<cyan>{name}:{function}</cyan> - <level>{message}</level>"
)
_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {name}:{function}:{line} - {message}"


def force_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # pragma: no cover - 非标准流时忽略
            pass


def setup_logging(
    level: str = "INFO",
    log_dir: Path | str | None = None,
    file_name: str = "customer_service_assistant.log",
    *,
    force_utf8: bool = True,
):
    """初始化日志，返回 loguru logger。

    重复调用是安全的（会先清空已有 sink），便于脚本与测试各自初始化。
    """
    if force_utf8:
        force_utf8_console()

    logger.remove()
    logger.add(sys.stderr, level=level, format=_CONSOLE_FORMAT, colorize=True, backtrace=False, diagnose=False)

    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(directory / file_name),
            level=level,
            format=_FILE_FORMAT,
            rotation="10 MB",
            retention=5,
            encoding="utf-8",
            enqueue=True,
        )
    return logger
