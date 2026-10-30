"""客服知识库 Agent。

分层：config / llm → ingest → retrieval → agent → service（见设计文档 §4）。
P0 阶段只有 ingest（baseline 切分）、retrieval（单路向量检索）、agent（baseline 提示词）。
"""

__version__ = "0.1.0"
