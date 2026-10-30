"""业务工具（mock）：订单上下文查询与转人工建单。

真实系统里这两个工具会对接订单中心与工单系统；这里用 JSON 数据模拟，
接口形态（入参、返回字段）保持一致，便于后续替换实现而不改 Agent。

``escalate_to_human`` 被 ``HumanInTheLoopMiddleware`` 拦截：调用前必须经过人工确认。
"""

from __future__ import annotations

import json
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from langchain.tools import tool
from loguru import logger

from ..config import get_settings
from .tools_calc import AI_QUOTA

ORDERS_FILE = "data/mock/orders.json"


@lru_cache(maxsize=1)
def load_orders() -> dict[str, dict]:
    path = Path(get_settings().project_root) / ORDERS_FILE
    if not path.exists():
        logger.warning("mock 订单文件不存在: {}", path)
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["order_id"]: item for item in payload["orders"]}


def reset_orders_cache() -> None:
    """测试用：清掉 lru_cache。"""
    load_orders.cache_clear()


def _days_since(first_purchase_date: str, today: date | None = None) -> int:
    purchase = datetime.strptime(first_purchase_date, "%Y-%m-%d").date()
    return max(((today or date.today()) - purchase).days, 0)


@tool
def check_order_context(order_id: str) -> dict[str, Any]:
    """按订单号查询订单上下文（套餐、订单金额、首购时间、用量、发票类型）。

    order_id：用户提供的订单号，例如 ORD-2026-0101。用户没有给订单号时**不要猜**，
    应先向用户询问订单号。

    返回 usage_ratio（AI 问答使用量占额度比例）等参数，可直接传给 calc_* 工具计算。
    """
    orders = load_orders()
    order = orders.get(order_id.strip().upper())
    if not order:
        return {
            "found": False,
            "order_id": order_id,
            "note": "未查询到该订单号，请与用户核对（不要臆测订单信息）",
        }

    plan = order["plan"]
    quota = AI_QUOTA.get(plan, 0)
    used = int(order.get("ai_qa_used", 0))
    return {
        "found": True,
        "order_id": order["order_id"],
        "plan": plan,
        "order_amount": order.get("amount"),
        "first_purchase_date": order.get("first_purchase_date"),
        "days_since_payment": _days_since(order["first_purchase_date"]),
        "invoice_type": order.get("invoice_type", "none"),
        "ai_qa_quota": quota,
        "ai_qa_used": used,
        "usage_ratio": round(used / quota, 4) if quota else None,
        "api_calls_this_month": order.get("api_calls_this_month"),
        "has_upgraded": order.get("has_upgraded", False),
        "owner": order.get("owner"),
        "note": "usage_ratio 已按套餐额度算好，可直接传给 calc_refund_tool / calc_ai_quota_status_tool",
    }


@tool
def escalate_to_human(
    reason: str,
    summary: str,
    priority: Literal["low", "normal", "high"] = "normal",
) -> dict[str, Any]:
    """创建人工工单并转人工客服（**调用前会经过人工确认**）。

    reason：转人工的原因，例如「企业版合同定制条款」「用户投诉」「知识库无依据」。
    summary：给人工客服看的简要说明，包含用户诉求与已确认的信息。
    priority：优先级，low / normal / high。
    """
    ticket_id = f"TICKET-{datetime.now():%Y%m%d}-{uuid4().hex[:6].upper()}"
    logger.info("创建工单 {}（priority={}）: {}", ticket_id, priority, reason)
    return {
        "ticket_id": ticket_id,
        "status": "created",
        "priority": priority,
        "reason": reason,
        "summary": summary,
        "sla": {"low": "3 个工作日", "normal": "1 个工作日", "high": "4 小时"}[priority],
        "note": "请把工单号告知用户，并说明人工客服会在 SLA 内联系",
    }


BUSINESS_TOOLS = [check_order_context, escalate_to_human]
