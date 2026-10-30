# 架构说明（Customer Service Assistant 客服 Agent）

> 本文件是 `README.md` 架构图的长版说明。

## 1. 分层结构

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ 接入层  service/  :  FastAPI + SSE（/chat、/chat/sync、/healthz、/ingest …）│
│                    ：零依赖演示页 service/static/index.html                │
├──────────────────────────────────────────────────────────────────────────┤
│ Agent 层 agent/  :  create_agent（DeepSeek + 8 个工具 + 中间件 + checkpointer）│
│                    ：submit_answer 契约（结构化答案 + 引用逐字校验）        │
│                    ：工具：search_knowledge_base / calc_* / check_order_*  │
│                    ：      / escalate_to_human                            │
├──────────────────────────────────────────────────────────────────────────┤
│ 检索层 retrieval/:  retrieve_with_report() 唯一入口                       │
│                    ：查询改写 → dense 召回 ∥ BM25 召回 → weighted 融合     │
│                    ：→ bge-reranker 精排 → 两级阈值（剔噪 / 拒答）         │
│                    ：→ small-to-big 回填章节全文                           │
├──────────────────────────────────────────────────────────────────────────┤
│ 数据层 ingest/  :  loaders（文件指纹）→ splitter（章节/片段）→ metadata    │
│                  ：→ embedder（content_hash 缓存）→ Milvus v2（dense+BM25+元数据）│
├──────────────────────────────────────────────────────────────────────────┤
│ 评测层 eval/    :  规则评测器（数值/关键词/拒答/路由/引用逐字）            │
│                  ：+ LLM judge（正确性、faithfulness，提示词版本化）        │
│                  ：runner（A~E 消融）｜ dataset（LangSmith 实验）           │
└──────────────────────────────────────────────────────────────────────────┘
```

## 2. 一次问答的时序（工具化 Agent）

```text
用户问题
  │
  ├─(多轮) checkpointer 载入同一 thread_id 的历史
  │
  ├─ 模型决策 → search_knowledge_base(query)
  │        └─ retrieve_with_report()
  │              ├─ 指代消解（有历史时）
  │              ├─ dense 召回 20（BGE-M3 向量）
  │              ├─ BM25 召回 20（Milvus 服务端 sparse 字段）
  │              ├─ weighted 融合（dense 0.2 / BM25 0.8）
  │              ├─ bge-reranker-v2-m3 精排 → top 5
  │              ├─ 最高分 < 0.22 → refused=true（拒答，不编造）
  │              └─ 命中 child → 回填所属章节全文
  │
  ├─ 模型决策 → calc_* / check_order_context（数字一律由 Python 算）
  │
  ├─ 模型决策 → submit_answer(answer, citations, key_facts, confidence, need_human)
  │        └─ 引用校验：quote 必须能在本次检索结果里逐字找到；关键数字必须来自工具返回值
  │              ├─ 通过 → 结束
  │              └─ 不通过 → 把问题清单作为工具返回值回灌，模型自我修正后重提
  │
  └─ need_human=true → escalate_to_human（HITL 中断，人工确认后建单）
```

## 3. 为什么这样分层

| 决策 | 理由 |
|---|---|
| 直连 `MilvusClient`，不套现成 VectorStore | 需要完全控制 schema（dense + BM25 sparse + 12 个标量字段）、幂等 upsert 与 `hybrid_search` |
| 检索只有一个入口 `retrieve_with_report()` | 评测脚本、Agent 工具、服务接口走同一条链路，"评测的就是线上跑的" |
| 结构化输出用 `submit_answer` 工具契约 | DeepSeek 思考模式不支持原生 `response_format` / `tool_choice`（实测 400），工具契约是唯一可行解 |
| 引用校验放在工具里，而非 `after_model` 中间件 | 校验失败能作为工具返回值回灌，走框架原生的重试语义；中间件注入会打乱图的终止条件 |
| 计算一律走 Python 工具 | 数值类问题零容错，LLM 只抽参数，不参与计算 |
| 拒答拆成两级阈值 | 片段过滤（0.02）只剔噪声；拒答判定（0.22）基于最高分——合一会让"语义弱但相关"的证据被误滤 |
| 拼装上下文用章节全文（small-to-big） | 属性清单型片段语义被摊薄，child 命中后补全章节，条件和结论才能同现 |

## 4. 评测闭环

```text
knowledge.txt ──► make_eval_set.py --generate ──► 候选集（LLM 生成，人工审核）
                                                        │ 审核（kb_anchor 逐字校验）
                                                        ▼
                          冻结评测集（100 题，不随仓库发布）
                                                        │
        configs/A~E.yaml ──► eval/runner.py ────────────►├─► results/<组>.jsonl（逐题原始记录）
                                                        │       └─► summary.json ──► markdown 报告
                                                        └─► eval/dataset.py ──► LangSmith experiment
```

评测集分五类（事实 40 / 计算 20 / 边界 20 / 拒答 15 / 多轮 5），每题都带
`kb_anchor`（知识库原文锚点）、答案级判据与期望章节。指标定义与实测结果见
[`evaluation.md`](evaluation.md)。
