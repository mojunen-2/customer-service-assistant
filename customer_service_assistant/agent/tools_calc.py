"""确定性计算工具：把知识库里的数值规则写成纯函数。

设计原则（设计文档 §5.4）：**涉及数字的结论必须由 Python 算出**，LLM 只负责抽取参数与解释结论。
每个返回值都带 ``policy_ref``（条款出处），让最终答案能标注依据。

规则出处：``knowledge.txt``

* §三.1 AI 问答额度：试用 200 / 基础 5000 / 专业 30000（自然月重置，不结转）
* §三.2 API 超额阶梯：超出套餐内额度的部分按 0~10000 次 8 元/千次、10001~50000 次 6 元/千次、
  50001 次以上 4 元/千次计费
* §三.4 超额停用：AI 问答额度用尽即停服务，直到下月重置或升级；API 超额不停用，计入账单
* §五.1 数据保留：试用冻结 7 天 + 归档 23 天（共 30 天）；基础/专业 15 天宽限 + 只读 30/60 天；
  企业默认 30 天宽限 + 90 天只读（以合同为准）
* §六.2 退款：首次购买 7 个自然日内；用量 ≤10% 全退、>10% 且 ≤50% 退 50%、>50% 不退；
  升级部分不适用；已开专票须先走红字发票流程；企业版默认不支持线上退款
"""

from __future__ import annotations

from typing import Any, Literal

from langchain.tools import tool

Plan = Literal["trial", "basic", "pro", "enterprise"]
PaidPlan = Literal["basic", "pro"]

API_INCLUDED_CALLS: dict[str, int] = {"basic": 10000, "pro": 80000}
AI_QUOTA: dict[str, int] = {"trial": 200, "basic": 5000, "pro": 30000}
# 超额阶梯：(该档可容纳的调用次数, 每 1000 次单价/元)
API_OVERAGE_BRACKETS: tuple[tuple[int, float], ...] = ((10000, 8.0), (40000, 6.0), (10**9, 4.0))
POLICY_REF_OVERAGE = "knowledge.txt §三.2"
POLICY_REF_QUOTA = "knowledge.txt §三.1/§三.4"
POLICY_REF_RETENTION = "knowledge.txt §五.1"
POLICY_REF_REFUND = "knowledge.txt §六.2"


def _fee(value: float) -> float:
    """计费金额：保留 4 位小数。

    知识库只规定"每 1000 次收费 N 元"，没有规定取整规则，因此**不自行发明四舍五入**——
    保留足够精度（例如超出 1 次 = 0.008 元），由上层决定如何展示。
    """
    return round(value + 0.0, 4)


def _money(value: float) -> float:
    """订单相关金额：保留 2 位小数。"""
    return round(value + 0.0, 2)


# --------------------------------------------------------------------------- #
# 纯函数（可单测、无网络、无副作用）
# --------------------------------------------------------------------------- #
def calc_api_overage(plan: PaidPlan, total_calls: int) -> dict[str, Any]:
    """计算当月 API 调用超额费用（只对超出套餐额度的部分计费）。"""
    if plan not in API_INCLUDED_CALLS:
        raise ValueError(f"套餐 {plan!r} 没有 API 调用额度（仅基础版/专业版支持）")
    if total_calls < 0:
        raise ValueError("total_calls 不能为负")

    included = API_INCLUDED_CALLS[plan]
    overage = max(total_calls - included, 0)
    remaining = overage
    brackets: list[dict[str, Any]] = []
    total_fee = 0.0
    lower = 0

    for capacity, unit_price in API_OVERAGE_BRACKETS:
        take = min(remaining, capacity)
        if take > 0:
            fee = take / 1000 * unit_price
            brackets.append(
                {
                    "range": f"{lower + 1}~{lower + take}" if capacity < 10**9 else f"{lower + 1} 以上",
                    "count": take,
                    "unit_price": unit_price,
                    "fee": _fee(fee),
                }
            )
            total_fee += fee
            remaining -= take
            lower += take
        else:
            lower += capacity
        if remaining <= 0:
            break

    return {
        "plan": plan,
        "total_calls": total_calls,
        "included": included,
        "overage": overage,
        "brackets": brackets,
        "total_fee": _fee(total_fee),
        "currency": "CNY",
        "policy_ref": POLICY_REF_OVERAGE,
    }


def calc_ai_quota_status(plan: Plan, used: int) -> dict[str, Any]:
    """查询 AI 问答额度使用情况（用尽后的行为按 §三.4）。"""
    if plan not in AI_QUOTA:
        raise ValueError(f"未知套餐：{plan!r}（可选 trial / basic / pro）")
    if used < 0:
        raise ValueError("used 不能为负")

    quota = AI_QUOTA[plan]
    remaining = max(quota - used, 0)
    exceeded = used >= quota
    return {
        "plan": plan,
        "quota": quota,
        "used": used,
        "remaining": remaining,
        "usage_ratio": round(used / quota, 4),
        "exceeded": exceeded,
        "behavior": (
            "额度已用尽：系统停止提供 AI 问答服务，直到下月 1 日额度重置或用户主动升级套餐"
            if exceeded
            else "额度充足，按自然月统计，未使用额度不结转到下月"
        ),
        "policy_ref": POLICY_REF_QUOTA,
    }


def calc_refund(
    plan: Plan,
    days_since_payment: int,
    usage_ratio: float,
    invoice_type: Literal["none", "normal", "special_vat"] = "none",
    order_amount: float | None = None,
) -> dict[str, Any]:
    """判断退款资格与退款比例。

    参数口径：

    * ``days_since_payment``：从支付成功到申请退款的自然日数（第 7 日 23:59:59 截止，故取 ≤7）
    * ``usage_ratio``：AI 问答实际使用量占套餐月度额度的比例（0~1）
    * ``invoice_type``：``special_vat``（专票）需先完成红字发票流程
    * ``order_amount``：订单金额（可选）；知识库只给比例，金额需由订单数据提供
    """
    if days_since_payment < 0:
        raise ValueError("days_since_payment 不能为负")
    if not 0 <= usage_ratio <= 1:
        raise ValueError("usage_ratio 必须在 0~1 之间")

    detail: dict[str, Any] = {
        "plan": plan,
        "days_since_payment": days_since_payment,
        "usage_ratio": usage_ratio,
        "invoice_type": invoice_type,
        "policy_ref": POLICY_REF_REFUND,
    }

    if plan == "trial":
        detail.update(
            eligible=False, refund_ratio=0.0, refund_amount=0.0,
            reason="试用版为免费版本，不涉及退款（§六.1）",
        )
        return detail

    if plan == "enterprise":
        detail.update(
            eligible=False, refund_ratio=0.0, refund_amount=0.0,
            reason="企业版默认不支持线上直接退款；重复付款、合同签署失败或服务未开通等特殊原因需销售、法务、财务联合审批（§六.3）",
            need_manual_review=True,
        )
        return detail

    if days_since_payment > 7:
        detail.update(
            eligible=False, refund_ratio=0.0, refund_amount=0.0,
            reason="已超过首次购买后的 7 个自然日（从支付成功时间起算，第 7 日 23:59:59 截止），不符合无理由退款条件（§六.2）",
        )
        return detail

    if usage_ratio > 0.5:
        detail.update(
            eligible=False, refund_ratio=0.0, refund_amount=0.0,
            reason="AI 问答实际使用量超过套餐月度额度的 50%，不支持退款（§六.2）",
        )
        return detail

    if usage_ratio > 0.1:
        ratio, reason = 0.5, "AI 问答实际使用量超过月度额度的 10% 但未超过 50%，退还 50%（§六.2）"
    else:
        ratio, reason = 1.0, "首购 7 日内且使用量未超过月度额度的 10%，可全额退款（§六.2）"

    blocked = invoice_type == "special_vat"
    detail.update(
        eligible=True,
        refund_ratio=ratio,
        refund_amount=_money(order_amount * ratio) if order_amount is not None else None,
        reason=reason,
        blocked_by_invoice=blocked,
        invoice_note=(
            "已开具增值税专用发票：需先完成红字发票流程后才能退款（§六.2 补充说明）" if blocked else None
        ),
        upgrade_note="若发生过套餐升级，升级部分金额不适用首购 7 日无理由退款（§六.2 补充说明）",
    )
    return detail


def check_data_retention(plan: Plan, days_since_expiry: int) -> dict[str, Any]:
    """判断订阅到期后第 N 天，数据处于什么状态。"""
    if days_since_expiry < 0:
        raise ValueError("days_since_expiry 不能为负")

    if plan == "trial":
        freeze_days, read_only_days = 7, 23
        note = "试用版到期后先冻结 7 天（可查看、不可上传与问答），再进入 23 天只读归档期，共 30 天"
    elif plan == "basic":
        freeze_days, read_only_days = 15, 30
        note = "基础版到期后 15 天宽限期内可登录但上传/编辑/新增问答受限，之后进入 30 天只读保留期"
    elif plan == "pro":
        freeze_days, read_only_days = 15, 60
        note = "专业版到期后 15 天宽限期内可登录但上传/编辑/新增问答受限，之后进入 60 天只读保留期"
    else:
        freeze_days, read_only_days = 30, 90
        note = "企业版按合同执行；合同未约定时默认 30 天宽限期 + 90 天只读保留期"

    total = freeze_days + read_only_days
    if days_since_expiry < freeze_days:
        state, days_left = "grace_period", freeze_days - days_since_expiry
    elif days_since_expiry < total:
        state, days_left = "read_only_archive", total - days_since_expiry
    else:
        state, days_left = "deleted", 0

    return {
        "plan": plan,
        "days_since_expiry": days_since_expiry,
        "state": state,
        "days_left": days_left,
        "retention_days": total,
        "recoverable": state != "deleted",
        "note": note,
        "policy_ref": POLICY_REF_RETENTION,
    }


# --------------------------------------------------------------------------- #
# 给 Agent 用的工具包装（docstring 即工具描述，LLM 依赖它抽参数）
# --------------------------------------------------------------------------- #
@tool
def calc_api_overage_tool(plan: PaidPlan, total_calls: int) -> dict:
    """计算某套餐当月 API 调用的超额费用。

    plan：套餐档位，只能填 basic 或 pro。
    total_calls：该月 API 总调用次数（不是超额次数）。
    返回每档明细与应付金额（元）。
    """
    return calc_api_overage(plan, total_calls)


@tool
def calc_ai_quota_status_tool(plan: Plan, used: int) -> dict:
    """查询 AI 问答额度使用情况。

    plan：套餐档位，取 trial / basic / pro。
    used：当月已使用的 AI 问答次数。
    返回总额度、剩余额度与是否超限后的服务行为。
    """
    return calc_ai_quota_status(plan, used)


@tool
def calc_refund_tool(
    plan: Plan,
    days_since_payment: int,
    usage_ratio: float,
    invoice_type: Literal["none", "normal", "special_vat"] = "none",
    order_amount: float | None = None,
) -> dict:
    """判断能否退款、能退多少。

    plan：套餐档位，取 trial / basic / pro / enterprise。
    days_since_payment：从支付成功到申请退款经过的自然日数（第 7 日 23:59:59 截止）。
    usage_ratio：AI 问答使用量占套餐月度额度的比例，用 0~1 的小数表示（例如 0.6 表示 60%）。
    invoice_type：发票类型，none=未开票 / normal=电子普通发票 / special_vat=增值税专用发票。
    order_amount：订单金额（元），仅在用户提供订单信息时填写；不填则只返回比例。
    """
    return calc_refund(plan, days_since_payment, usage_ratio, invoice_type, order_amount)


@tool
def check_data_retention_tool(plan: Plan, days_since_expiry: int) -> dict:
    """查询订阅到期后数据的保留状态。

    plan：套餐档位，取 trial / basic / pro / enterprise。
    days_since_expiry：从订阅到期算起经过的天数。
    返回数据当前状态（宽限期/只读归档/已删除）与剩余天数。
    """
    return check_data_retention(plan, days_since_expiry)


CALC_TOOLS = [
    calc_api_overage_tool,
    calc_ai_quota_status_tool,
    calc_refund_tool,
    check_data_retention_tool,
]
