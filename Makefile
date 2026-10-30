# Customer Service Assistant —— 常用命令
# Windows 下如未安装 make，可直接执行 README 中的等价 python 命令
PY ?= python

.PHONY: help test ingest ask rank api up down eval eval-calc multiturn clean eval-set eval-ablation ablation

help:
	@echo "make test    运行单测（需私有 tests/）"
	@echo "make ingest  重建/增量写入索引（幂等）"
	@echo "make ask Q='专业版多少钱？'  命令行提问"
	@echo "make rank    检索质量评估（探针集）"
	@echo "make api     本地启动 FastAPI（uvicorn）"
	@echo "make up      用 docker compose 起服务（Milvus 复用宿主机）"
	@echo "make eval-calc   计算类评测（20 题手算）"
	@echo "make multiturn   多轮指代验收"
	@echo "make eval-set        校验/审计评测集（需私有 data/eval/）"
	@echo "make eval-ablation   跑 A~E 五组消融（需私有评测集，较慢）"
	@echo "make ablation        把 results/*.summary.json 渲染成 docs/ablation.md"

test:
	$(PY) -m pytest

ingest:
	$(PY) scripts/ingest.py

ask:
	$(PY) scripts/ask.py "$(Q)"

rank:
	$(PY) scripts/rank_eval.py --label "make rank" --out docs/rank_eval_current.md

api:
	$(PY) -m uvicorn customer_service_assistant.service.api:app --host 127.0.0.1 --port 8000

up:
	docker compose up -d api

down:
	docker compose down

eval-calc:
	$(PY) scripts/eval_calc.py

multiturn:
	$(PY) scripts/check_multiturn.py

eval:
	$(PY) scripts/rank_eval.py --label "make eval" --out docs/rank_eval_p2.md

# ---------------- P5 评测闭环 ----------------
eval-set:
	$(PY) scripts/make_eval_set.py --verify --audit

eval-ablation:
	$(PY) -m customer_service_assistant.eval.runner --config configs/A_baseline.yaml --judge --concurrency 3
	$(PY) -m customer_service_assistant.eval.runner --config configs/B_structure.yaml --judge --concurrency 3
	$(PY) -m customer_service_assistant.eval.runner --config configs/C_hybrid.yaml --judge --concurrency 3
	$(PY) -m customer_service_assistant.eval.runner --config configs/D_rerank.yaml --judge --concurrency 3
	$(PY) -m customer_service_assistant.eval.runner --config configs/E_production.yaml --judge --concurrency 3

ablation:
	$(PY) -m customer_service_assistant.eval.runner --render docs/ablation.md

clean:
	$(PY) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['logs', '.pytest_cache'] if pathlib.Path(p).exists()]"
