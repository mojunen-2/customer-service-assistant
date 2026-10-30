"""Milvus 访问层：显式 schema（dense + BM25 sparse + 元数据）、写入、检索、维护。

P1 起使用显式 schema（schema_version=v2）：

* **dense**：``BAAI/bge-m3`` 1024 维，HNSW / COSINE
* **sparse**：由服务端 BM25 Function 从 ``text`` 自动生成，SPARSE_INVERTED_INDEX，中文分析器
* **标量字段**：chunk_type / parent_id / section / section_no / plan_tier / doc_id / seq / ...

P0 的 ``customer_service_docs``（quick-create schema）保持不动，用于复现 baseline 对照；
本模块对两种 schema 都做了兼容（``has_field`` 判断），P0 的对照脚本仍可运行。

踩过的坑（见 README）：

1. ``get_collection_stats().row_count`` 含墓碑行 → 幂等判断必须用 ``count(*)``；
2. ``MilvusClient`` 默认连 ``default`` 库 → 建连时即 ``use_database``；
3. 同时存在 dense 与 sparse 两个向量字段时必须显式指定 ``anns_field``。
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Sequence

from loguru import logger
from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker, WeightedRanker

from ..config import Settings, get_settings
from ..models import ChunkRecord, RetrievedChunk

DENSE_FIELD = "vector"
SPARSE_FIELD = "sparse"
BM25_FUNCTION_NAME = "bm25"
ALL_ID_FILTER = 'id != ""'

# v2 schema 的标量字段（顺序即建表顺序）
V2_SCALAR_FIELDS: tuple[tuple[str, int], ...] = (
    ("chunk_type", 8),
    ("parent_id", 64),
    ("section", 128),
    ("plan_tier", 16),
    ("doc_id", 64),
    ("doc_version", 16),
    ("source", 512),
    ("content_hash", 64),
)
V2_INT_FIELDS: tuple[str, ...] = ("seq", "section_no")

V2_OUTPUT_FIELDS: tuple[str, ...] = (
    "text", "seq", "section", "section_no", "plan_tier", "chunk_type", "parent_id", "source", "doc_version",
)

# P0 quick-create collection 的字段（没有 section / chunk_type 等）
P0_OUTPUT_FIELDS: tuple[str, ...] = ("text", "seq", "doc_id", "source", "doc_version")


class MilvusDocStore:
    """知识库片段的 Milvus 存储。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: MilvusClient | None = None

    # ---------------- 连接 ----------------
    @property
    def client(self) -> MilvusClient:
        """懒加载客户端（建连时即切换到目标数据库，避免落到 default 库）。"""
        if self._client is None:
            self._client = MilvusClient(uri=self.settings.milvus_uri, timeout=self.settings.milvus_timeout)
            if self.settings.milvus_db in self._client.list_databases():
                self._client.use_database(db_name=self.settings.milvus_db)
        return self._client

    @cached_property
    def field_names(self) -> set[str]:
        try:
            described = self.client.describe_collection(collection_name=self.settings.milvus_collection)
        except Exception as error:  # collection 不存在时保持空集合，交由 ensure_ready 处理
            logger.debug("describe_collection 失败（可能尚未建表）: {}", error)
            return set()
        return {field["name"] for field in described.get("fields", [])}

    def has_field(self, name: str) -> bool:
        return name in self.field_names

    @property
    def is_v2(self) -> bool:
        """是否为本项目 P1 的显式 schema（含 chunk_type / sparse）。"""
        return "chunk_type" in self.field_names

    def ensure_ready(self) -> None:
        """确保数据库与 collection 存在（不存在才创建，绝不删除已有数据）。"""
        databases = self.client.list_databases()
        if self.settings.milvus_db not in databases:
            self.client.create_database(db_name=self.settings.milvus_db)
            logger.info("created database {}", self.settings.milvus_db)
        self.client.use_database(db_name=self.settings.milvus_db)

        if not self.client.has_collection(collection_name=self.settings.milvus_collection):
            self._create_v2_collection()
        self.client.load_collection(collection_name=self.settings.milvus_collection)

    def _create_v2_collection(self) -> None:
        """建 v2 显式 schema：dense + BM25 sparse + 元数据字段。"""
        client = self.client
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, max_length=64, is_primary=True)
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=self.settings.max_text_length,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        )
        schema.add_field(DENSE_FIELD, DataType.FLOAT_VECTOR, dim=self.settings.embed_dim)
        schema.add_field(SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
        for name, max_length in V2_SCALAR_FIELDS:
            schema.add_field(name, DataType.VARCHAR, max_length=max_length)
        for name in V2_INT_FIELDS:
            schema.add_field(name, DataType.INT16)
        schema.add_function(
            Function(
                name=BM25_FUNCTION_NAME,
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=[SPARSE_FIELD],
            )
        )

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name=DENSE_FIELD,
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        index_params.add_index(
            field_name=SPARSE_FIELD,
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"drop_ratio_build": 0.2},
        )
        client.create_collection(
            collection_name=self.settings.milvus_collection,
            schema=schema,
            index_params=index_params,
        )
        logger.info(
            "created collection {} (schema=v2, dim={}, dense=HNSW/COSINE, sparse=BM25)",
            self.settings.milvus_collection,
            self.settings.embed_dim,
        )

    # ---------------- 写入 ----------------
    def upsert_records(self, records: Sequence[ChunkRecord], vectors: Sequence[Sequence[float]]) -> int:
        if len(records) != len(vectors):
            raise ValueError(f"records 与 vectors 数量不一致：{len(records)} != {len(vectors)}")
        if not records:
            return 0
        rows = [record.to_milvus_row(list(vector)) for record, vector in zip(records, vectors)]
        result = self.client.upsert(collection_name=self.settings.milvus_collection, data=rows)
        inserted = int(result.get("upsert_count", len(rows))) if isinstance(result, dict) else len(rows)
        logger.info("upsert {} records into {}", inserted, self.settings.milvus_collection)
        return inserted

    def flush(self) -> None:
        self.client.flush(collection_name=self.settings.milvus_collection)

    def compact(self) -> None:
        """触发 compaction，物理清理 upsert / delete 产生的墓碑行（异步执行）。"""
        self.client.compact(collection_name=self.settings.milvus_collection)
        logger.info("compaction 已提交: {}", self.settings.milvus_collection)

    # ---------------- 读取 ----------------
    def count_entities(self, filter: str | None = None) -> int:
        """可见实体数（**不含**逻辑删除的墓碑行）。

        坑点记录：``get_collection_stats`` 的 ``row_count`` 会统计墓碑行，
        实测同一批数据 upsert 两次时它从 45 变成 90，而 query 只能看到 45 条。
        """
        rows = self.client.query(
            collection_name=self.settings.milvus_collection,
            filter=filter or ALL_ID_FILTER,
            output_fields=["count(*)"],
        )
        if not rows:
            return 0
        return int(rows[0].get("count(*)", 0))

    def list_doc_ids(self) -> set[str]:
        rows = self.client.query(
            collection_name=self.settings.milvus_collection,
            filter=ALL_ID_FILTER,
            output_fields=["doc_id"],
            limit=16384,
        )
        doc_ids = {str(row["doc_id"]) for row in rows if row.get("doc_id")}
        if len(rows) >= 16384:
            logger.warning("list_doc_ids 达到单次查询上限 16384，结果可能不完整")
        return doc_ids

    def delete_by_doc_id(self, doc_id: str) -> int:
        """按文档删除全部片段（用于清理已从知识库移除的文件）。返回匹配到的条数。"""
        expression = f'doc_id == "{doc_id}"'
        matched = self.count_entities(filter=expression)
        if matched:
            self.client.delete(collection_name=self.settings.milvus_collection, filter=expression)
        return matched

    def default_output_fields(self) -> list[str]:
        return list(V2_OUTPUT_FIELDS if self.is_v2 else P0_OUTPUT_FIELDS)

    def search(
        self,
        vector: Sequence[float],
        limit: int = 5,
        *,
        filter: str | None = None,
        output_fields: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """dense 向量检索（必须显式指定 anns_field：本 schema 有两个向量字段）。"""
        results = self.client.search(
            collection_name=self.settings.milvus_collection,
            data=[list(vector)],
            anns_field=DENSE_FIELD,
            search_params={"metric_type": "COSINE"},
            limit=limit,
            filter=filter,
            output_fields=list(output_fields or self.default_output_fields()),
        )
        hits = results[0] if results else []
        return [RetrievedChunk.from_hit(hit, rank=index) for index, hit in enumerate(hits, start=1)]

    def search_sparse(
        self,
        query_text: str,
        limit: int = 10,
        *,
        filter: str | None = None,
        output_fields: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """BM25 全文检索（服务端用中文分析器解析查询文本）。

        P1 只准备好能力并纳入集成测试；P2 的混合检索会把它与 dense 结果做 RRF 融合。
        """
        if not self.has_field(SPARSE_FIELD):
            raise RuntimeError(f"collection {self.settings.milvus_collection} 没有 sparse 字段，无法做全文检索")
        results = self.client.search(
            collection_name=self.settings.milvus_collection,
            data=[query_text],
            anns_field=SPARSE_FIELD,
            search_params={"metric_type": "BM25"},
            limit=limit,
            filter=filter,
            output_fields=list(output_fields or self.default_output_fields()),
        )
        hits = results[0] if results else []
        return [RetrievedChunk.from_hit(hit, rank=index) for index, hit in enumerate(hits, start=1)]

    def hybrid_search(
        self,
        query_vector: Sequence[float],
        query_text: str,
        limit: int = 20,
        *,
        filter: str | None = None,
        dense_limit: int | None = None,
        sparse_limit: int | None = None,
        rrf_k: int = 60,
        fusion: str = "rrf",
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        output_fields: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """dense + BM25 双路召回，按 ``fusion`` 策略融合（P2 的混合检索）。

        * ``rrf``：只使用排名（``1/(k+rank)`` 求和），不依赖两路分数尺度，最稳健；
        * ``weighted``：分数归一化后加权求和（``norm_score=True``）。实测在本项目语料上
          **明显优于 RRF**——RRF 只看排名，会把稠密路的高排名噪声一起抬进 top5，
          而 BM25 在数字/专有名词上的信号被稀释；加权融合保留了 BM25 的强信号，
          又让稠密路作为补充（见 docs/retrieval_eval.md 的消融表）。
        """
        if not self.has_field(SPARSE_FIELD):
            raise RuntimeError(f"collection {self.settings.milvus_collection} 没有 sparse 字段，无法做混合检索")
        fields = list(output_fields or self.default_output_fields())
        dense_limit = dense_limit or self.settings.dense_top_k
        sparse_limit = sparse_limit or self.settings.sparse_top_k

        requests = [
            AnnSearchRequest(
                data=[list(query_vector)],
                anns_field=DENSE_FIELD,
                param={"metric_type": "COSINE"},
                limit=dense_limit,
                expr=filter,
            ),
            AnnSearchRequest(
                data=[query_text],
                anns_field=SPARSE_FIELD,
                param={"metric_type": "BM25"},
                limit=sparse_limit,
                expr=filter,
            ),
        ]
        ranker = (
            WeightedRanker(dense_weight, sparse_weight, norm_score=True)
            if fusion == "weighted"
            else RRFRanker(k=rrf_k)
        )
        results = self.client.hybrid_search(
            collection_name=self.settings.milvus_collection,
            reqs=requests,
            ranker=ranker,
            limit=limit,
            output_fields=fields,
        )
        hits = results[0] if results else []
        return [RetrievedChunk.from_hit(hit, rank=index) for index, hit in enumerate(hits, start=1)]

    def get_by_ids(self, ids: Sequence[str], output_fields: Sequence[str] | None = None) -> dict[str, dict]:
        """按主键批量取回记录（small-to-big 回填父级章节用）。"""
        if not ids:
            return {}
        rows = self.client.get(
            collection_name=self.settings.milvus_collection,
            ids=list(ids),
            output_fields=list(output_fields or self.default_output_fields()),
        )
        return {str(row["id"]): dict(row) for row in rows}

    def collection_stats(self) -> dict[str, Any]:
        """统计信息：``visible_count`` 为准，``stats_row_count`` 仅供参考（含墓碑）。"""
        stats = self.client.get_collection_stats(collection_name=self.settings.milvus_collection)
        return {
            "uri": self.settings.milvus_uri,
            "database": self.settings.milvus_db,
            "collection": self.settings.milvus_collection,
            "dimension": self.settings.embed_dim,
            "schema": "v2" if self.is_v2 else "p0-quick-create",
            "visible_count": self.count_entities(),
            "stats_row_count": int(stats.get("row_count", 0)),
            "note": "stats_row_count 含 upsert / delete 墓碑行，请以 visible_count 为准",
        }
