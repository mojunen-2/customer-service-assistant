"""评测集流水线（设计文档 §6.1）：LLM 生成候选 → 人工审核 → 冻结 → 校验。

三个子命令对应流程的三个阶段，**审核那一环由人完成**（本脚本不假装自动审核）：

1. ``--generate`` 按章节用 LLM 批量生成候选题，落盘到 ``data/eval/golden_candidates.jsonl``，
   每条带 ``reviewed: false``（原始素材，不参与评测）；
2. 人工审核：逐条核对候选提问与知识库原文，把可用的题改写后写入
   ``data/eval/golden_set.jsonl``（冻结资产，``review: accepted``）；
   ``--audit`` 输出"候选 ↔ 冻结"的覆盖报告，让审核过程可追溯；
3. ``--verify`` 对冻结集做机器校验（must_include 必须逐字出现在 knowledge.txt、
   章节名必须真实存在、类别分布符合设计文档、字段完整）并生成 ``docs/eval_set.md``。

用法::

    python scripts/make_eval_set.py --generate            # 生成候选（调 LLM）
    python scripts/make_eval_set.py --audit --verify      # 审核覆盖报告 + 冻结集校验
    python scripts/make_eval_set.py --verify --out docs/eval_set.md
"""

import _bootstrap  # noqa: F401  必须最先导入

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from customer_service_assistant.config import Settings, get_settings
from customer_service_assistant.ingest.loaders import discover_files, load_documents
from customer_service_assistant.ingest.splitter import parse_sections

EVAL_DIR = Path("data/eval")
GOLDEN_PATH = EVAL_DIR / "golden_set.jsonl"
CANDIDATES_PATH = EVAL_DIR / "golden_candidates.jsonl"

# 设计文档 §6.1 的目标分布
CATEGORY_TARGET: dict[str, int] = {
    "fact": 40,
    "calc": 20,
    "boundary": 20,
    "refuse": 15,
    "multi_turn": 5,
}

GENERATE_SYSTEM_PROMPT = """你是客服知识库的评测集作者。请依据给定的知识库章节原文，生成候选题。

要求：
1. 每道题给出 question（用户口吻，允许口语化）、ground_truth（标准答案要点）、
   must_include（必须在标准答案里出现的原文要点，逐字取自章节原文）、anchor（支撑该题的原文字符串）。
2. 题目必须**能只靠该章节原文回答**，不得引入原文之外的信息。
3. 不要编造数字：所有数字必须出现在原文中。
4. 只输出 JSON 数组，不要输出解释文字。格式：
   [{"question": "...", "ground_truth": "...", "must_include": ["..."], "anchor": "..."}]
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测集生成 / 审核审计 / 校验")
    parser.add_argument("--generate", action="store_true", help="用 LLM 按章节生成候选题")
    parser.add_argument("--per-section", type=int, default=8, help="每个章节生成多少道候选题")
    parser.add_argument("--sections", default=None, help="只处理指定章节，逗号分隔（如 六、退款规则）")
    parser.add_argument("--audit", action="store_true", help="输出候选集与冻结集的覆盖报告")
    parser.add_argument("--verify", action="store_true", help="校验 golden_set.jsonl 并生成报告")
    parser.add_argument("--out", default="docs/eval_set.md", help="校验报告输出路径")
    return parser.parse_args()


def resolve(settings: Settings, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(settings.project_root) / candidate


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items: list[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path} 第 {line_no} 行不是合法 JSON：{error}") from error
    return items


def normalize(text: str) -> str:
    """去掉空白差异后比较（知识库原文里的空格数量不影响"逐字"判定）。"""
    return re.sub(r"\s+", "", text)


# --------------------------------------------------------------------------- #
# 1) 生成候选
# --------------------------------------------------------------------------- #
def extract_json_array(text: str) -> list[dict]:
    match = re.search(r"\[.*\]", text, flags=re.S)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [item for item in payload if isinstance(item, dict) and item.get("question")]


def generate(settings: Settings, per_section: int, only: set[str] | None) -> int:
    from customer_service_assistant.llm import get_chat_model

    documents, _ = load_documents(settings)
    model = get_chat_model()
    rows: list[dict] = []
    for document in documents:
        for section in parse_sections(document):
            if only and section.title not in only:
                continue
            body = section.text.strip()
            if len(body) < 80:  # 太短的章节（如"产品简介"过短时）跳过，避免凑题
                logger.info("跳过章节 {}（正文过短）", section.title)
                continue
            prompt = (
                f"章节标题：{section.title}\n"
                f"请生成 {per_section} 道候选题。\n\n章节原文：\n{body}"
            )
            response = model.invoke(
                [{"role": "system", "content": GENERATE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            )
            content = response.content if isinstance(response.content, str) else str(response.content)
            candidates = extract_json_array(content)
            for index, item in enumerate(candidates, start=1):
                rows.append(
                    {
                        "candidate_id": f"{section.title}-{index:02d}",
                        "section": section.title,
                        "question": str(item.get("question", "")).strip(),
                        "ground_truth": str(item.get("ground_truth", "")).strip(),
                        "must_include": item.get("must_include") or [],
                        "anchor": str(item.get("anchor", "")).strip(),
                        "reviewed": False,
                    }
                )
            logger.info("{} → 生成 {} 条候选", section.title, len(candidates))

    out_path = settings.project_root / CANDIDATES_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    logger.info("候选集写入 {}（{} 条，全部 reviewed=false，需人工审核）", out_path, len(rows))
    return 0


# --------------------------------------------------------------------------- #
# 2) 审核审计
# --------------------------------------------------------------------------- #
def audit(settings: Settings) -> str:
    golden = load_jsonl(settings.project_root / GOLDEN_PATH)
    candidates = load_jsonl(settings.project_root / CANDIDATES_PATH)
    if not candidates:
        return "（未找到候选集，无法审计；先跑 `--generate`）\n"

    golden_questions = {normalize(item["question"]) for item in golden}
    # 候选的 anchor 是知识库原句，冻结集的 kb_anchor 也是知识库原句 —— 用它们判断"要点是否被采纳"
    golden_anchors = [normalize(anchor) for item in golden for anchor in item.get("kb_anchor", [])]
    golden_keys = [
        normalize(key)
        for item in golden
        for key in [*item.get("must_include", []), *item.get("must_include_any", [])]
    ]

    def judged(candidate: dict) -> str:
        if normalize(candidate["question"]) in golden_questions:
            return "exact"
        anchor = normalize(candidate.get("anchor") or "")
        if anchor and any(anchor in kb or kb in anchor for kb in golden_anchors):
            return "rewritten"
        for key in candidate.get("must_include", []):
            normalized = normalize(key)
            if len(normalized) >= 4 and any(normalized in golden or golden in normalized for golden in golden_keys):
                return "rewritten"
        return "dropped"

    lines = [
        f"- 候选集：{len(candidates)} 条（LLM 生成，`reviewed=false`）",
        f"- 冻结集：{len(golden)} 条（人工审核，`review: accepted`）",
        "",
        "| 审核结果 | 数量 | 说明 |",
        "|---|---|---|",
    ]
    verdicts = [judged(candidate) for candidate in candidates]
    exact = verdicts.count("exact")
    rewritten = verdicts.count("rewritten")
    dropped = verdicts.count("dropped")
    lines.append(f"| 原样采纳（问题完全一致） | {exact} | LLM 候选直接可用 |")
    lines.append(f"| 改写后采纳（要点被引用） | {rewritten} | 提问被改成更口语化 / 更可判定的表述 |")
    lines.append(f"| 未采纳 | {dropped} | 与原文对不上、重复或不可判定，审核时剔除 |")
    lines.append("")
    lines.append(
        "> 审核动作：逐条核对候选提问与 `knowledge.txt` 原文；提问改写为真实用户口吻（含口语化表达），"
        "并把要点拆成两层——`kb_anchor` 逐字复制原文（证明题目有据可依），"
        "`must_include` / `must_include_any` 写成答案里必须出现的**短关键短语**（规则评测用）。"
        "冻结集里另有部分题目由评测需求直接编写（覆盖边界/多轮/拒答与计算题），不与候选集一一对应。"
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 3) 冻结集校验
# --------------------------------------------------------------------------- #
def verify(settings: Settings) -> tuple[bool, str]:
    golden = load_jsonl(settings.project_root / GOLDEN_PATH)
    problems: list[str] = []

    documents, _ = load_documents(settings)
    corpus = normalize("\n".join(document.page_content for document in documents))
    section_titles = {section.title for document in documents for section in parse_sections(document)}

    seen: set[str] = set()
    for item in golden:
        item_id = item.get("id", "?")
        if item_id in seen:
            problems.append(f"{item_id}: id 重复")
        seen.add(item_id)
        if item.get("category") not in CATEGORY_TARGET:
            problems.append(f"{item_id}: 未知类别 {item.get('category')!r}")
        for field in ("question", "ground_truth"):
            if not str(item.get(field, "")).strip():
                problems.append(f"{item_id}: {field} 为空")
        # kb_anchor：题目必须有知识库原文依据（逐字校验）
        for anchor in item.get("kb_anchor", []):
            if normalize(anchor) not in corpus:
                problems.append(f"{item_id}: kb_anchor 未逐字出现在知识库：{anchor!r}")
        # must_include / must_include_any：答案级判据，必须是短关键短语（长句会误判）
        for key in item.get("must_include", []):
            if len(key) > 24:
                problems.append(f"{item_id}: must_include 过长（{len(key)} 字符），应是答案里的短关键短语：{key!r}")
        if item.get("category") not in {"refuse", "calc"}:
            if not item.get("kb_anchor"):
                problems.append(f"{item_id}: 缺少 kb_anchor（无法证明题目有知识库依据）")
            if not item.get("must_include") and not item.get("must_include_any"):
                problems.append(f"{item_id}: must_include 与 must_include_any 不能同时为空（规则判据缺失）")
        for section in item.get("expected_sections", []):
            if section not in section_titles:
                problems.append(f"{item_id}: expected_sections 不存在：{section!r}")
        for forbidden in item.get("must_not_include", []):
            if normalize(forbidden) in corpus:
                problems.append(f"{item_id}: must_not_include 竟然出现在知识库（判据失效）：{forbidden!r}")
        if item.get("category") == "refuse":
            if not item.get("refuse_expected"):
                problems.append(f"{item_id}: refuse 题目必须 refuse_expected=true")
            if item.get("expected_sections"):
                problems.append(f"{item_id}: refuse 题目不应声明 expected_sections")
        if item.get("category") == "calc":
            if not item.get("expected_tool"):
                problems.append(f"{item_id}: calc 题目缺少 expected_tool")
            if not item.get("expected_values"):
                problems.append(f"{item_id}: calc 题目缺少 expected_values")
        if item.get("category") == "multi_turn" and not item.get("history"):
            problems.append(f"{item_id}: multi_turn 题目缺少 history")

    counts = Counter(item["category"] for item in golden)
    report = [
        "# 评测集（golden_set.jsonl）校验报告",
        "",
        f"- 生成时间：{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        f"- 题量：{len(golden)} 条 ｜ 校验结果：{'通过' if not problems else f'{len(problems)} 个问题'}",
        "",
        "## 类别分布（对照设计文档 §6.1）",
        "",
        "| 类别 | 目标 | 实际 | 说明 |",
        "|---|---|---|---|",
    ]
    notes = {
        "fact": "事实检索（含口语化改写）",
        "calc": "数值计算（expected_values 为人工手算）",
        "boundary": "条件/多跳边界",
        "refuse": "应拒答（超范围）",
        "multi_turn": "多轮指代追问",
    }
    for category, target in CATEGORY_TARGET.items():
        actual = counts.get(category, 0)
        mark = "" if actual == target else " ⚠️"
        report.append(f"| {category} | {target} | {actual}{mark} | {notes[category]} |")

    report += [
        "",
        "## 机器校验项",
        "",
        "1. `id` 唯一；`question` / `ground_truth` 非空；",
        "2. 每条 `kb_anchor` **逐字**出现在 `knowledge.txt`（去空白比较）——保证题目有据可依；",
        "3. `expected_sections` 必须是切分后真实存在的章节标题；",
        "4. `must_not_include` 不得出现在知识库中（否则判据自相矛盾）；",
        "5. `refuse` 题必须 `refuse_expected=true` 且不声明期望章节；",
        "6. `calc` 题必须有 `expected_tool` 与 `expected_values`；`multi_turn` 必须有 `history`；",
        "7. `must_include` 是**答案级**判据，只写短关键短语（≤24 字符）——"
        "把整条知识库条款当判据会误判（实测：模型不会逐字复述整句；判据说明见 docs/evaluation.md）。",
        "",
    ]
    if problems:
        report += ["## 发现的问题", ""] + [f"- {problem}" for problem in problems] + [""]
    report.append("## 审核覆盖（候选 → 冻结）")
    report.append("")
    report.append(audit(settings).rstrip())
    report.append("")
    return not problems, "\n".join(report) + "\n"


def main() -> int:
    args = parse_args()
    settings = get_settings()
    if args.generate:
        only = {item.strip() for item in args.sections.split(",")} if args.sections else None
        return generate(settings, args.per_section, only)

    if args.audit and not args.verify:
        print(audit(settings))
        return 0

    if args.verify:
        ok, report = verify(settings)
        if args.audit:
            pass  # verify 内部已包含审核覆盖
        out_path = resolve(settings, args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        logger.info("校验报告已写入 {}（{}）", out_path, "通过" if ok else "存在问题")
        print(report)
        return 0 if ok else 1

    logger.info("未指定动作：用 --generate / --audit / --verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
