"""脚本引导：把项目根目录加入 sys.path。

用法（必须在其它 customer_service_assistant 导入之前）::

    import _bootstrap  # noqa: F401
    from customer_service_assistant.config import get_settings

这样 ``python scripts/ingest.py`` 无论从哪个工作目录执行都能正常导入包。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
