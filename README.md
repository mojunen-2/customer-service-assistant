# Customer Service Assistant · 客服知识库 Agent

一个面向 SaaS 客服场景的知识库 Agent：**检索负责事实、Python 工具负责算数、Agent 负责路由与组织答案、评测负责证明它变好了**。

项目从"能跑通的 RAG Demo"出发，补齐了结构感知切分、混合检索、重排、拒答、引用校验、
工具化计算与自动评测，并服务化为 FastAPI + SSE，附带一个零依赖的网页演示。

## 架构

```text
接入层   FastAPI + SSE（/chat、/chat/sync、/healthz、/collection/stats、/ingest、/hitl）
   │
Agent 层 create_agent（对话模型 + 8 个工具 + 中间件 + checkpointer）
   │      submit_answer：结构化答案 + 引用逐字校验 + 关键数字溯源
   │      工具：知识库检索 / 额度与退款计算 / 订单上下文 / 转人工
   │
检索层   retrieve_with_report()（唯一入口）
   │      指代消解 → 向量召回 ∥ BM25 召回 → 加权融合 → 重排
   │      → 两级阈值（片段过滤 / 拒答判定）→ small-to-big 回填章节
   │
数据层   文档加载 → 结构感知切分（章节 / 段落）→ 元数据 → 嵌入缓存
   │      → Milvus（dense HNSW + BM25 sparse + 标量字段，确定性主键幂等写入）
   │
评测层   规则评测（数值 / 关键词 / 拒答 / 路由 / 引用逐字）
           + LLM judge（正确性 / 忠实度）→ 100 题冻结集 + A~E 消融
```

更详细的时序与取舍见 [docs/architecture.md](docs/architecture.md)。

## 核心特性

- **结构感知切分 + small-to-big**：`======` 章节为父块、段落为子块；命中子块后回填整章，让"条件 + 结论"同时出现在上下文里。
- **混合检索**：BGE-M3 向量 + Milvus 服务端 BM25，加权融合后交给 `bge-reranker-v2-m3` 精排。
- **两级阈值拒答**：片段过滤只剔噪声，拒答判定看最高分；无依据时不编造并转人工。
- **工具化计算**：额度、超额阶梯、退款分档、数据保留写成纯函数；模型只抽参数，数字必须能追溯到工具返回值。
- **引用可溯源**：`submit_answer` 的每条引用必须在检索片段中逐字出现；校验失败会把原因回灌给模型自我修正。
- **可评测**：规则/LLM judge 双轨 + A~E 五组消融（评测集私有，结果见 [docs/evaluation.md](docs/evaluation.md)）。
- **可服务化**：SSE 流式输出、六个 HTTP 接口、Docker Compose、零依赖演示页。

## 快速开始

### 0. 前置条件

- Python 3.11+（开发环境使用 3.13）
- Docker（用于 Milvus；如已有 Milvus 实例可跳过本地 Milvus）
- 模型凭据：DeepSeek（对话）与 SiliconFlow（BGE-M3 嵌入 + 重排）

### 1. 安装与配置

```bash
git clone <your-repo-url> customer-service-assistant
cd customer-service-assistant

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # Windows: Copy-Item .env.example .env
# 编辑 .env，至少填写 DEEPSEEK_API_KEY 与 SILICONFLOW_API_KEY
```

### 2. 启动 Milvus（如已有实例可跳过）

```bash
# 方式一：用 compose 自带一套本地 Milvus（同时启动 API 容器）
docker compose --profile local-milvus up -d

# 方式二：使用已有 Milvus，把 .env 里的 CSA_MILVUS_URI 指向它
```

### 3. 构建索引

```bash
# 先把自有知识库放到 data/knowledge/knowledge.txt（该文件不随仓库发布）
python scripts/ingest.py              # 幂等，可重复执行；首次会调用嵌入接口
```

> `ingest` 默认读取 `data/knowledge/knowledge.txt`。项目**不公开示例知识库**，
> 请按 [`data/knowledge/README.md`](data/knowledge/README.md) 的格式放入自己的内容（`======` + 中文序号章节）。

### 4. 提问

```bash
# 命令行（支持 --show-citations / --show-trace / --thread-id 多轮）
python scripts/ask.py "专业版成员人数上限是多少？" --show-citations
python scripts/ask.py "那基础版呢？" --thread-id demo-1

# 起 Web 服务（浏览器打开 http://127.0.0.1:8000/）
python -m uvicorn customer_service_assistant.service.api:app --port 8000

# 同步接口
curl -s -X POST http://127.0.0.1:8000/api/v1/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"message":"基础版调用 18000 次 API 要付多少钱？"}'
```

网页演示会实时展示工具调用轨迹、逐字答案与引用原文；复杂问题需要 10~20 秒（多轮工具调用），页面会显示进度。

## 配置速查

所有项目配置使用 `CSA_` 前缀，详见 [`.env.example`](.env.example)。最常改的几项：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CSA_MILVUS_URI` | `http://localhost:19530` | Milvus 地址 |
| `CSA_MILVUS_DB` | `customer_service_assistant` | 数据库（不存在时自动创建） |
| `CSA_MILVUS_COLLECTION` | `customer_service_docs_v2` | 索引 collection |
| `CSA_KNOWLEDGE_DIR` | `./data/knowledge` | 知识库目录 |
| `CSA_RETRIEVAL_MODE` | `hybrid` | `dense` / `sparse` / `hybrid` |
| `CSA_RERANK_ENABLED` | `true` | 是否启用重排 |
| `CSA_RERANK_THRESHOLD` | `0.02` | 片段过滤阈值 |
| `CSA_REFUSAL_THRESHOLD` | `0.22` | 拒答阈值（基于最高重排分） |
| `CSA_ADMIN_TOKEN` | 空 | `/api/v1/ingest` 的 `X-Admin-Token`；留空则禁用该接口 |

## 评测

> 评测集与测试用例暂不随仓库发布。下面的命令需要你自备私有评测集（`data/eval/`，同 `.gitignore` 忽略）。

```bash
python scripts/eval_calc.py                 # 20 道手算计算题
python scripts/check_multiturn.py           # 多轮指代验收
python scripts/make_eval_set.py --verify    # 校验 100 题评测集

# A~E 消融（judge 会调用模型，较慢）
python -m customer_service_assistant.eval.runner --config configs/E_production.yaml --judge --concurrency 3
python -m customer_service_assistant.eval.runner --render docs/ablation.md
```

实测结论：规则正确率 **23% → 73% → 95% → 96% → 100%**（结构切分 → 混合检索 → 重排 → 工具化）。
完整指标、阈值曲线与局限见 [docs/evaluation.md](docs/evaluation.md)。

## 项目结构

```text
.
├─ customer_service_assistant/
│  ├─ config.py llm.py logging_setup.py models.py
│  ├─ ingest/              # 加载、结构感知切分、元数据、幂等入库
│  ├─ retrieval/           # Milvus 存储、混合检索、重排、嵌入缓存、统一检索入口
│  ├─ agent/               # 工具、结构化输出与引用校验、提示词、中间件、Agent 组装
│  ├─ eval/                # 规则评测 + LLM judge + A~E 运行器 + LangSmith 数据上传
│  └─ service/             # FastAPI + SSE + 演示页
├─ scripts/                # ingest / ask / 评测 / 调试脚本
├─ configs/                # A~E 消融配置
├─ data/
│  ├─ knowledge/           # 知识库位置（内容不公开，仅保留格式说明）
│  ├─ mock/                # 合成订单数据
│  └─ eval/                # 私有评测集位置（不随仓库发布）
└─ docs/                   # 架构说明、评测报告
```

测试用例（`tests/`）与评测集（`data/eval/`）当前不公开，已在 `.gitignore` 中排除；
本地开发时把私有测试与评测数据放回对应目录即可运行 `pytest`。

## 设计取舍

1. **直连 Milvus 客户端**：需要完全控制 dense + BM25 sparse 的 schema、幂等写入与融合检索。
2. **检索只有一个入口**：评测脚本、Agent 工具、HTTP 接口共用同一函数，避免"评测的和线上跑的不是一条链路"。
3. **结构化输出用工具契约**：思考型模型不支持原生 `response_format` / 强制 `tool_choice`，用 `submit_answer` 工具提交 Pydantic 结构最稳。
4. **引用校验放在工具内部**：失败原因作为工具返回值回灌，模型可自我修正，且不打断图终止条件。
5. **拒绝与片段过滤拆成两个阈值**：合成一个会误滤"语义弱但确实相关"的证据。

## 已知限制

- 订单与计费是合成 mock 数据，不接真实业务系统；没有鉴权/多租户/限流。
- 生产配置 p95 延迟约 14s，未达到 6s 目标（多轮工具调用 + 长上下文）。
- 评测集为自建（LLM 候选 + 人工审核），不是对抗集；每组只跑一遍，无方差估计，且不随仓库发布。
- 知识库为示例数据，请替换为你自己的内容；如涉及版权请自行处理后再公开。

## License

本项目暂未附带开源许可证。公开发布前请根据你的用途补充（例如 MIT / Apache-2.0），
否则默认保留所有权利。
