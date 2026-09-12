# ===========================================================================
# sovereign-rating-lab — 研究环境镜像
#
# 构建：
#   docker build -t sovereign-rating-lab:0.1.0 .
#
# 运行 Streamlit 仪表盘：
#   docker run --rm -p 8501:8501 sovereign-rating-lab:0.1.0
#
# 运行测试或流水线：
#   docker run --rm sovereign-rating-lab:0.1.0 pytest -q
#   docker run --rm sovereign-rating-lab:0.1.0 python -m src.pipeline
# ===========================================================================
FROM python:3.11-slim AS base

# --- 基础环境 --------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

# 中文字体（图表标注需要）+ 编译 xgboost/shap 轮子可能用到的工具链
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      fonts-noto-cjk \
      fontconfig \
      libgomp1 \
      git \
 && fc-cache -f \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- 依赖层（单独缓存，源码改动不会导致重装依赖） --------------------------
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# --- 源码层 ----------------------------------------------------------------
COPY . .

# 以非 root 用户运行，降低容器风险
RUN useradd --create-home --shell /bin/bash researcher \
 && chown -R researcher:researcher /app
USER researcher

# 预创建输出目录（卷挂载点的权限由宿主机决定，这里只保证目录存在）
RUN mkdir -p data/raw data/interim data/processed reports/figures reports/tables reports/logs

# --- 健康检查 --------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import src, pandas, sklearn; print('ok')" || exit 1

# --- 默认入口 --------------------------------------------------------------
EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
