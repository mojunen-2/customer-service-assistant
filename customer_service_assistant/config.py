"""全局配置：项目唯一的配置入口。

设计要点（对应设计文档附录 A）：

1. 本项目自有配置统一使用 ``CSA_`` 前缀，未设置时使用代码内默认值；
2. ``DEEPSEEK_*`` / ``SILICONFLOW_*`` / ``LANGSMITH_*`` 从项目根目录的 ``.env`` 读取，
   通过 ``validation_alias`` 绕过 ``CSA_`` 前缀读取，避免密钥重复维护；
3. 相对路径统一按项目根目录解析，保证"在哪个目录执行脚本"不影响结果；
4. 启动时用 :meth:`Settings.safe_summary` 打印配置快照，密钥一律脱敏。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# <repo>/customer_service_assistant（Python 包目录）
PACKAGE_DIR: Path = Path(__file__).resolve().parent
# <repo>（包含 pyproject.toml / data / .env）
PROJECT_ROOT: Path = PACKAGE_DIR.parent
# 只读项目根目录的 .env（公开仓库不依赖任何外部目录层级）
DEFAULT_ENV_FILES: tuple[Path, ...] = (PROJECT_ROOT / ".env",)


class Settings(BaseSettings):
    """项目配置。所有字段都可通过 ``CSA_<字段名>`` 环境变量覆盖。"""

    model_config = SettingsConfigDict(
        env_prefix="CSA_",
        env_file=DEFAULT_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # 允许用字段名直接构造（测试与脚本里更直观）；环境变量仍按 CSA_ 前缀 / alias 读取
        populate_by_name=True,
    )

    # ---------------- 路径 ----------------
    project_root: Path = PROJECT_ROOT
    knowledge_dir: Path = PROJECT_ROOT / "data" / "knowledge"
    extra_docs_dir: Path = PROJECT_ROOT / "data" / "raw"
    log_dir: Path = PROJECT_ROOT / "logs"
    index_manifest_path: Path = PROJECT_ROOT / "data" / "index_manifest.json"
    embed_cache_path: Path = PROJECT_ROOT / "data" / "cache" / "embeddings.json"

    # ---------------- Milvus ----------------
    milvus_uri: str = "http://localhost:19530"
    milvus_db: str = "customer_service_assistant"
    # P1 起使用显式 schema 的 collection（含 BM25 sparse 字段与完整元数据）；
    # 旧的 customer_service_docs（P0 quick-create schema）保留不动，作为 baseline 复现用
    milvus_collection: str = "customer_service_docs_v2"
    baseline_collection: str = "customer_service_docs"
    milvus_timeout: float = 30.0

    # ---------------- 知识库 ----------------
    knowledge_file: str = "knowledge.txt"
    doc_glob: str = "*.txt,*.md"
    doc_version: str = "2026Q1"
    # P0 baseline 切分参数（仅 baseline_splitter 使用，用于复现对照组）
    chunk_size: int = Field(default=200, ge=50)
    chunk_overlap: int = Field(default=80, ge=0)
    # P1 结构感知切分的 child 参数（章节作为 parent，章节内切 child）
    child_chunk_size: int = Field(default=300, ge=50)
    child_chunk_overlap: int = Field(default=60, ge=0)
    max_text_length: int = Field(default=8192, ge=256)

    # ---------------- 模型 ----------------
    llm_model: str = "deepseek-flash"
    llm_provider: str = "deepseek"
    llm_temperature: float = 0.0
    llm_timeout: float = 60.0
    llm_retry_attempts: int = Field(default=3, ge=1)

    embed_model: str = "openai:BAAI/bge-m3"
    embed_dim: int = Field(default=1024, ge=1)
    embed_batch: int = Field(default=32, ge=1)
    embed_timeout: float = 60.0

    # ---------------- 检索（P2：混合检索 + 重排 + 拒答阈值）----------------
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    dense_top_k: int = Field(default=20, ge=1)
    sparse_top_k: int = Field(default=20, ge=1)
    hybrid_fetch_limit: int = Field(default=20, ge=1)
    # 融合策略：rrf（只看排名，稳健）或 weighted（分数归一化后加权，实测效果更好）
    fusion: Literal["rrf", "weighted"] = "weighted"
    rrf_k: int = Field(default=60, ge=1)
    dense_weight: float = 0.2
    sparse_weight: float = 0.8
    # 关键词扩展：在 8 题 / 16 题探针集上实测"略有损害"（nDCG 与平均排名都变差），
    # 因此默认关闭，保留实现与开关，等 P5 的 100 题评测集（含改写型提问）再验证。
    keyword_expansion: bool = False
    condense_with_history: bool = True

    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_n: int = Field(default=5, ge=1)
    # 上下文片段过滤阈值：只用来剔除明确的噪声片段，应保持宽松（否则会误滤掉
    # "语义弱但确实相关"的片段——实测 q-08 的文档说明片段只有 0.11）。
    rerank_threshold: float = 0.02
    # 拒答阈值：用"最高重排分"做二分类决策，由 scripts/calibrate_threshold.py 标定。
    # P2 口径（16 题探针 + 8 题负样本）：最高负样本 0.2106 < 最低正样本 0.2426，取中点 0.23。
    # P5 口径（冻结集 60 题 fact/boundary + 15 题负样本，见 docs/threshold_calibration_p5.md）：
    # 新增口语化提问把正样本最低分拉到 0.2286（8 题负样本口径未见此样本），最高负样本 0.2115，
    # 相邻样本中点 0.2201 同时做到"误拒 1/60 = 1.7%、超范围拒答 15/15 = 100%"，故下调到 0.22。
    refusal_threshold: float = 0.22
    rerank_on_parent: bool = False
    rerank_timeout: float = 30.0
    rerank_retry_attempts: int = Field(default=3, ge=1)
    rerank_cache_size: int = Field(default=4096, ge=0)

    # ---------------- 运行 ----------------
    prompt_version: str = "v1"
    enable_tracing: bool = True
    log_level: str = "INFO"
    max_tool_calls: int = Field(default=8, ge=1)
    max_model_calls: int = Field(default=12, ge=1)
    summary_trigger: int = Field(default=8000, ge=100)
    enable_hitl: bool = True
    admin_token: str | None = None

    # ---------------- 复用教程 .env 的共享凭据（绕过 CSA_ 前缀）----------------
    deepseek_api_key: str | None = Field(default=None, validation_alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str | None = Field(default=None, validation_alias="DEEPSEEK_BASE_URL")
    siliconflow_api_key: str | None = Field(default=None, validation_alias="SILICONFLOW_API_KEY")
    siliconflow_base_url: str | None = Field(default=None, validation_alias="SILICONFLOW_BASE_URL")
    langsmith_api_key: str | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_endpoint: str | None = Field(default=None, validation_alias="LANGSMITH_ENDPOINT")
    langsmith_project: str | None = Field(default=None, validation_alias="LANGSMITH_PROJECT")

    # ---------------- 校验 ----------------
    @field_validator("langsmith_endpoint")
    @classmethod
    def _normalize_endpoint(cls, value: str | None) -> str | None:
        """统一小写：教程 .env 中写成 https://api.smith.LangChain.com（域名大小写不影响解析，但保持规范）。"""
        if not value:
            return value
        return value.strip().rstrip("/").lower()

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        return value.strip().upper()

    @model_validator(mode="after")
    def _resolve_relative_paths(self) -> "Settings":
        for name in ("knowledge_dir", "extra_docs_dir", "log_dir", "index_manifest_path", "embed_cache_path"):
            value: Path = getattr(self, name)
            if not value.is_absolute():
                setattr(self, name, (self.project_root / value).resolve())
        return self

    # ---------------- 派生属性 ----------------
    @property
    def doc_globs(self) -> list[str]:
        return [item.strip() for item in self.doc_glob.split(",") if item.strip()]

    @property
    def knowledge_file_path(self) -> Path:
        return self.knowledge_dir / self.knowledge_file

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.enable_tracing and self.langsmith_api_key)

    @property
    def effective_mode(self) -> str:
        """实际生效的检索模式（重排开启时记为 hybrid+rerank）。"""
        base = self.retrieval_mode
        if base == "hybrid" and self.rerank_enabled:
            return "hybrid+rerank"
        return base

    # ---------------- 断言式校验 ----------------
    def require_llm_credentials(self) -> None:
        if not self.deepseek_api_key:
            raise RuntimeError(
                "缺少 DEEPSEEK_API_KEY：请在教程根目录 .env 中配置，或设置环境变量 CSA_LLM_MODEL 指向可用模型。"
            )

    def require_embedding_credentials(self) -> None:
        missing = [
            name
            for name, value in (
                ("SILICONFLOW_API_KEY", self.siliconflow_api_key),
                ("SILICONFLOW_BASE_URL", self.siliconflow_base_url),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(f"缺少嵌入模型凭据：{', '.join(missing)}（请检查教程根目录 .env）")

    # ---------------- 日志用配置快照（脱敏）----------------
    def safe_summary(self) -> dict[str, Any]:
        def mask(value: str | None) -> str:
            if not value:
                return "<未配置>"
            return f"{value[:4]}***{value[-2:]}" if len(value) > 8 else "***"

        return {
            "project_root": str(self.project_root),
            "milvus": f"{self.milvus_uri} / {self.milvus_db} / {self.milvus_collection}",
            "knowledge_dir": str(self.knowledge_dir),
            "chunk": f"size={self.chunk_size}, overlap={self.chunk_overlap}",
            "llm": f"{self.llm_provider}:{self.llm_model}",
            "embed": f"{self.embed_model} (dim={self.embed_dim})",
            "retrieval": (
                f"dense_top_k={self.dense_top_k}, sparse_top_k={self.sparse_top_k}, "
                f"fusion={self.fusion}({self.dense_weight}/{self.sparse_weight}), rrf_k={self.rrf_k}, "
                f"rerank={self.rerank_enabled}@{self.rerank_top_n}, threshold={self.rerank_threshold}"
            ),
            "prompt_version": self.prompt_version,
            "tracing": f"{self.tracing_enabled} -> {self.langsmith_project}",
            "deepseek_api_key": mask(self.deepseek_api_key),
            "siliconflow_api_key": mask(self.siliconflow_api_key),
            "langsmith_api_key": mask(self.langsmith_api_key),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例配置（避免重复解析 .env）。"""
    return Settings()
