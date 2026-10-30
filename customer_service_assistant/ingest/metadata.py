"""元数据抽取：章节号、章节标题、套餐档位、片段主键。

设计文档 §5.1 要求每个片段自带 ``section`` / ``section_no`` / ``plan_tier`` /
``doc_version`` / ``content_hash``，这里集中实现抽取规则，避免散落在切分逻辑里。
"""

from __future__ import annotations

import hashlib
import re

CN_NUMERALS: dict[str, int] = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

SECTION_TITLE_RE = re.compile(r"^(?P<num>[一二三四五六七八九十]+|\d+)\s*[、.．]\s*(?P<title>.+?)\s*$")

# 计划档位关键词：只有片段里能唯一确定档位时才写入，混档（如"套餐说明"整节）留空，
# 目的是让 plan_tier 作为过滤条件时"宁可不用，也不出错"。
PLAN_TIER_KEYWORDS: dict[str, tuple[str, ...]] = {
    "trial": ("试用版",),
    "basic": ("基础版",),
    "pro": ("专业版",),
    "enterprise": ("企业版",),
}

PREAMBLE_TITLE = "文档说明"


def parse_section_no(title: str) -> int:
    """把章节标题解析成序号：``六、退款规则`` -> 6；无法解析返回 0（如"文档说明"）。"""
    match = SECTION_TITLE_RE.match(title.strip())
    if not match:
        return 0
    raw = match.group("num")
    if raw.isdigit():
        return int(raw)
    return CN_NUMERALS.get(raw, 0)


def section_body_title(title: str) -> str:
    """去掉序号后的章节名：``六、退款规则`` -> ``退款规则``。"""
    match = SECTION_TITLE_RE.match(title.strip())
    return match.group("title").strip() if match else title.strip()


def detect_plan_tier(text: str) -> str:
    """识别片段所属套餐档位；命中多个档位时返回空串（表示"不唯一"）。"""
    matched = [tier for tier, keywords in PLAN_TIER_KEYWORDS.items() if any(kw in text for kw in keywords)]
    return matched[0] if len(matched) == 1 else ""


def make_chunk_uid(doc_id: str, chunk_type: str, section_no: int, seq: int, content_hash: str) -> str:
    """确定性片段主键（幂等的关键）。

    与 P0 的 ``baseline_splitter.make_chunk_uid`` 不同：这里把 ``chunk_type`` 与
    ``section_no`` 也纳入哈希，避免 parent / child 或不同章节的片段撞 id。
    """
    raw = f"{doc_id}|{chunk_type}|{section_no}|{seq}|{content_hash}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
