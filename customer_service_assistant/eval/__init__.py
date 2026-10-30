"""评测闭环（P5）：规则评测器 + LLM judge + 消融运行器 + LangSmith dataset。

设计依据：设计文档 §6。模块分工：

* :mod:`customer_service_assistant.eval.evaluators` —— 规则判定 + judge（判据与提示词版本化）；
* :mod:`customer_service_assistant.eval.pipelines` —— A~E 五组配置对应的执行管线；
* :mod:`customer_service_assistant.eval.runner` —— 跑评测、落盘逐题记录、渲染 `docs/ablation.md`；
* :mod:`customer_service_assistant.eval.dataset` —— 上传 LangSmith dataset 并跑 experiment（可选）。
"""

from .evaluators import ItemResult, evaluate_rules, judge_correctness, judge_faithfulness
from .judge_prompts import JUDGE_PROMPT_VERSION, judge_prompt_hash
from .pipelines import GroupConfig, load_config, run_item

__all__ = [
    "ItemResult",
    "evaluate_rules",
    "judge_correctness",
    "judge_faithfulness",
    "JUDGE_PROMPT_VERSION",
    "judge_prompt_hash",
    "GroupConfig",
    "load_config",
    "run_item",
]
