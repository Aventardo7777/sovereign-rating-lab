# ===========================================================================
# sovereign-rating-lab — 常用任务快捷方式
#
#   make help        查看全部可用命令
#   make setup       创建虚拟环境并安装依赖
#   make check       提交前完整检查（lint + test）
#   make pipeline    运行端到端演示流水线
#   make dashboard   启动 Streamlit 仪表盘
# ===========================================================================

PYTHON ?= python
VENV   ?= .venv
ifeq ($(OS),Windows_NT)
	BIN := $(VENV)/Scripts
else
	BIN := $(VENV)/bin
endif
PY := $(BIN)/python
PIP := $(PY) -m pip

.DEFAULT_GOAL := help
.PHONY: help setup setup-conda venv install install-dev install-hooks \
        format lint lint-fix typecheck test test-fast test-cov \
        sample pipeline notebooks dashboard clean clean-data clean-all \
        build docker-build docker-run check precommit

# ---------------------------------------------------------------------------
help:  ## 显示可用命令
	@printf "sovereign-rating-lab — 可用命令\n\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@printf "\n变量覆盖示例： make test PYTHON=python3.12\n"

# ---------------------------------------------------------------------------
# 环境
# ---------------------------------------------------------------------------
venv:  ## 创建虚拟环境（.venv）
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel

setup: venv install-dev install-hooks  ## 一键初始化开发环境
	@printf "\n✅ 开发环境就绪。下一步：make sample && make pipeline\n"

setup-conda:  ## 用 conda 创建环境（需已安装 conda）
	conda create -n sovereign-rating-lab python=3.11 -y
	@printf "请运行： conda activate sovereign-rating-lab && make install-dev\n"

install:  ## 安装运行时依赖
	$(PIP) install -r requirements.txt

install-dev:  ## 安装运行时 + 开发依赖
	$(PIP) install -r requirements-dev.txt

install-hooks:  ## 安装 pre-commit 钩子
	$(BIN)/pre-commit install

# ---------------------------------------------------------------------------
# 代码质量
# ---------------------------------------------------------------------------
format:  ## 自动格式化（black + ruff --fix）
	$(BIN)/black .
	$(BIN)/ruff check . --fix

lint:  ## 静态检查：ruff + black --check（CI 使用同一命令）
	$(BIN)/ruff check .
	$(BIN)/black --check .

lint-fix:  ## 同 format，保留以兼容习惯用法
	$(MAKE) format

typecheck:  ## 可选：mypy 静态类型检查
	$(BIN)/mypy src 2>/dev/null || printf "未安装 mypy，跳过（pip install mypy）\n"

precommit:  ## 对全部文件运行 pre-commit
	$(BIN)/pre-commit run --all-files

# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------
test:  ## 运行全部测试
	$(PY) -m pytest

test-fast:  ## 运行测试，跳过标记为 slow 的用例
	$(PY) -m pytest -m "not slow"

test-cov:  ## 运行测试并输出覆盖率报告
	$(PY) -m pytest --cov=src --cov-report=term-missing --cov-report=html

check: lint test  ## 提交前完整检查

# ---------------------------------------------------------------------------
# 数据与流水线
# ---------------------------------------------------------------------------
sample:  ## 重新生成合成示例数据
	$(PY) scripts/make_sample_data.py

pipeline:  ## 运行端到端演示流水线（合成数据）
	$(PY) -m src.pipeline

pipeline-fast:  ## 流水线但跳过建模步骤
	$(PY) -m src.pipeline --no-models

real-data:  ## 用自有数据运行流水线，需提供 RATINGS= 与 MACRO=
	@if [ -z "$(RATINGS)" ]; then \
		printf "用法： make real-data RATINGS=data/raw/ratings.csv MACRO=data/raw/macro.csv\n"; \
		exit 1; \
	fi
	$(PY) -m src.pipeline --ratings $(RATINGS) $(if $(MACRO),--macro $(MACRO),)

notebooks:  ## 启动 Jupyter Lab
	$(BIN)/jupyter lab notebooks

dashboard:  ## 启动 Streamlit 仪表盘
	$(BIN)/streamlit run app/streamlit_app.py

# ---------------------------------------------------------------------------
# 构建与容器
# ---------------------------------------------------------------------------
build:  ## 构建源码分发包与 wheel
	$(PY) -m build

docker-build:  ## 构建 Docker 镜像
	docker build -t sovereign-rating-lab:0.1.0 .

docker-run:  ## 运行 Streamlit 容器（映射 8501 端口）
	docker run --rm -p 8501:8501 sovereign-rating-lab:0.1.0

# ---------------------------------------------------------------------------
# 清理
# ---------------------------------------------------------------------------
clean:  ## 清理 Python 缓存与构建产物（不动数据）
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.py[co]" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist htmlcov .coverage 2>/dev/null || true
	@printf "✅ 已清理缓存与构建产物\n"

clean-data:  ## 清理可由流水线重新生成的数据（保留 sample 与模板）
	rm -rf data/interim/* data/processed/* data/raw/cache 2>/dev/null || true
	rm -rf reports/figures/* reports/tables/* reports/logs 2>/dev/null || true
	@printf "✅ 已清理派生数据与报告输出\n"

clean-all: clean clean-data  ## 清理一切可再生成的产物
	@printf "✅ 完成（.venv 与 data/sample 未删除）\n"
