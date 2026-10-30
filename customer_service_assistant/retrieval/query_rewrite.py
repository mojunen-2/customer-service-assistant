"""查询理解：关键词扩展（BM25 路）+ 多轮指代消解（LLM）。

设计原则（设计文档 §5.3）：**只在有证据的地方使用技巧**，且每条技巧都只作用于它擅长的那一路：

* 关键词扩展只加进 **BM25 查询文本**——补上"退款→退货/退费"这类同义词能提升精确串匹配，
  而把它塞进稠密查询会稀释语义（用户原话的语义才是最准的）；
* 指代消解只在**存在多轮历史**时触发，且改写结果用于两路查询（否则"那专业版呢"这类追问
  在两路上都检索不到东西）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from ..config import Settings, get_settings
from ..llm import get_chat_model

# 同义词表：只在 BM25 查询里追加（左=用户可能说的词，右=知识库里实际使用的词）
KEYWORD_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "退款": ("退货", "退费", "退款规则"),
    "退钱": ("退款", "退费"),
    "发票": ("开票", "普票", "专票", "增值税"),
    "专票": ("增值税专用发票", "红字发票"),
    "额度": ("配额", "限额", "用量"),
    "超额": ("超出", "超量", "阶梯计费"),
    "API": ("接口", "调用", "开放接口"),
    "成员": ("人数", "成员数"),
    "外部协作者": ("外部成员", "协作账号"),
    "数据保留": ("保留期", "归档", "数据删除"),
    "套餐": ("版本", "档位", "订阅"),
    "OCR": ("图片解析", "识别"),
    "升级": ("变更套餐", "续费"),
}

CONDENSE_PROMPT = """把用户的追问改写成一句不依赖上下文、可以直接检索的问题。
要求：保留原问题的全部关键信息（套餐档位、数字、时间等）；不要回答问题；只输出改写后的那一句话。

对话历史：
{history}

用户的追问：{question}

改写后的问题："""


@dataclass(slots=True)
class RewriteResult:
    query: str
    keyword_query: str
    expansions: list[str] = field(default_factory=list)
    condensed: bool = False


def expand_keywords(query: str) -> list[str]:
    """按同义词表找出可追加的关键词（保留顺序、去重）。"""
    matched: list[str] = []
    lowered = query.lower()
    for trigger, synonyms in KEYWORD_EXPANSIONS.items():
        if trigger.lower() in lowered:
            for synonym in synonyms:
                if synonym not in matched:
                    matched.append(synonym)
    return matched


def build_keyword_query(query: str, settings: Settings | None = None) -> tuple[str, list[str]]:
    """构造 BM25 路的查询文本（原问题 + 同义词）。"""
    settings = settings or get_settings()
    if not settings.keyword_expansion:
        return query, []
    expansions = expand_keywords(query)
    if not expansions:
        return query, []
    return f"{query} {' '.join(expansions)}", expansions


def condense_question(question: str, history: str, settings: Settings | None = None) -> str | None:
    """用 LLM 把追问改写成自包含问题；失败时返回 None（由调用方回退到原问题）。"""
    settings = settings or get_settings()
    if not settings.condense_with_history or not history.strip():
        return None
    try:
        model = get_chat_model()
        response = model.invoke(CONDENSE_PROMPT.format(history=history.strip(), question=question))
        content = getattr(response, "content", "")
        rewritten = content.strip() if isinstance(content, str) else ""
        if not rewritten:
            return None
        logger.debug("指代消解: {!r} -> {!r}", question, rewritten)
        return rewritten
    except Exception as error:  # 改写失败不能影响主链路
        logger.warning("指代消解失败，回退原问题: {}", error)
        return None


def rewrite_for_retrieval(
    query: str,
    history: str | None = None,
    settings: Settings | None = None,
) -> RewriteResult:
    """给检索层用的统一改写入口。"""
    settings = settings or get_settings()
    condensed_query = None
    if history:
        condensed_query = condense_question(query, history, settings)

    base_query = condensed_query or query
    keyword_query, expansions = build_keyword_query(base_query, settings)
    return RewriteResult(
        query=base_query,
        keyword_query=keyword_query,
        expansions=expansions,
        condensed=condensed_query is not None,
    )
