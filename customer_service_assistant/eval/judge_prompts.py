"""LLM judge 提示词（版本化 + 可校验哈希）。

设计文档 §6.2 要求 judge 只用于**软性指标**（答案正确性、faithfulness），
计算/拒答/路由这类可判定的指标一律用规则评测器。§8 风险表要求
"judge prompt 固定并记录版本"——这里的 :data:`JUDGE_PROMPT_VERSION` 与
:func:`judge_prompt_hash` 会写进评测报告，提示词一改，版本与哈希就变，
历史报告的数据可追溯到具体判据。
"""

from __future__ import annotations

import hashlib

CORRECTNESS_PROMPT_VERSION = "correctness-v2"
FAITHFULNESS_PROMPT_VERSION = "faithfulness-v3"
# 兼容旧调用：整体版本号（报告里分别展示两个 judge 的版本）
JUDGE_PROMPT_VERSION = f"{CORRECTNESS_PROMPT_VERSION}+{FAITHFULNESS_PROMPT_VERSION}"

# 判据：与标准答案的语义一致性 + 关键数字/条件完整性
# v1 → v2 的修订原因（实测）：v1 把"回答比标准答案更详细、补充了知识库里确实存在的其它条款"
# 判成了 incorrect（人工抽检 3 条即发现），导致正确率被系统性低估。
CORRECTNESS_PROMPT = """你是严格的客服评测裁判。请判断【回答】是否与【标准答案】一致。

判定档位（只能选一个）：
- correct：标准答案里的关键结论、数字、条件都出现在回答里且正确；
- partial：关键结论出现，但缺少标准答案中的某个关键条件/数字，或表述含糊难以确认；
- incorrect：关键结论或数字错误、答非所问、缺少关键条件导致结论改变，或出现知识库不支持的内容。

**不算错（必须判 correct）**：
1. 回答的措辞、顺序、格式与标准答案不同，但语义一致；
2. 回答比标准答案更详细，补充了知识库里确实存在的其它条款/数字，且**不与标准答案矛盾**——
   评测对象是客服助手，多给正确信息不是错误；请只依据【标准答案】和【回答】判断，
   **不要因为"回答里出现了标准答案没有提到的数字"就判错**；
3. 计算题给出了计算过程。

**算错**：
1. 标准答案里的关键条件/数字在回答中缺失（例如标准答案强调"7 个自然日内"，回答完全没提时间限制）；
2. 关键数字算错或写错；
3. 结论与标准答案矛盾（标准答案说不能退，回答却说可以退）；
4. 回答表示"无法确认 / 建议联系人工"，而标准答案是可回答的具体政策。

只输出 JSON，不要输出其它内容：{{"verdict": "correct|partial|incorrect", "reason": "一句话理由"}}

【问题】
{question}

【标准答案】
{reference}

【回答】
{answer}
"""

# 判据：回答中的结论是否都能被检索上下文支撑（幻觉率）
# v2 → v3 的修订原因（P5 实测）：v2 会把三类**不该算无依据**的内容记成幻觉——
#   ① 复述用户问题里的前提（"你在首购 7 个自然日内""已开专票"）；
#   ② 工具产生的客观记录（工单号、订单号、计算结果）与"可提供订单号帮你查"这类服务性表述；
#   ③ 由上下文规则直接推算出的数字。
# E 组（工具化 Agent）的回答更长、信息更多，被误伤尤其明显，故补上明确的豁免规则。
FAITHFULNESS_PROMPT = """你是事实性审核员。请检查【回答】里的结论与数字是否都能被【知识库上下文】支撑。

规则：
1. 只找"上下文里没有依据"的结论或数字（包括把上下文的规则错误外推、编造额度/价格/期限/流程）；
2. 合理改写、同义表述、从上下文数字直接换算得到的结果，不算无依据；
3. 回答里表示"知识库无法确认、建议转人工"的表述，不算无依据；
4. **以下情况一律不算无依据**（不要写进 unsupported）：
   - 复述问题中已给出的前提（例如"第 25 天""已开专票""用量 30%"这类用户自带条件）；
   - 工具执行产生的客观记录与服务性表述（工单号、订单号、计算结果、"可提供订单号帮你查询"、
     "建议联系人工客服"、客服话术里的礼貌用语）；
   - 提示词允许的档位对照与引用条款原文。
5. 只标注**产品政策 / 数字 / 流程**上的主张，且该主张无法由上下文（含由上下文规则直接推算）支撑；
6. 如果全部有支撑，unsupported 返回空数组。

只输出 JSON，不要输出其它内容：
{{"unsupported": ["无依据的结论（逐条，尽量短）"], "reason": "一句话说明"}}

【知识库上下文】
{context}

【回答】
{answer}
"""


def judge_prompt_hash() -> str:
    """judge 提示词的稳定哈希（写进报告，便于追溯判据版本）。"""
    return judge_prompt_hashes()["all"]


def judge_prompt_hashes() -> dict[str, str]:
    """每个 judge 各自的版本与哈希（两个判据独立迭代，报告里分别展示）。"""

    def digest(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

    return {
        "correctness": f"{CORRECTNESS_PROMPT_VERSION}:{digest(CORRECTNESS_PROMPT)}",
        "faithfulness": f"{FAITHFULNESS_PROMPT_VERSION}:{digest(FAITHFULNESS_PROMPT)}",
        "all": digest(f"{CORRECTNESS_PROMPT}|{FAITHFULNESS_PROMPT}"),
    }
