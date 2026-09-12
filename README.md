# sovereign-rating-lab · 主权信用评级迁移与预测研究

[![CI](https://github.com/Aventardo7777/sovereign-rating-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Aventardo7777/sovereign-rating-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pytest](https://img.shields.io/badge/tests-pytest-0A9EDC.svg)](https://docs.pytest.org/)
[![GitHub stars](https://img.shields.io/github/stars/Aventardo7777/sovereign-rating-lab?style=social)](https://github.com/Aventardo7777/sovereign-rating-lab/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Aventardo7777/sovereign-rating-lab?style=social)](https://github.com/Aventardo7777/sovereign-rating-lab/network/members)
[![GitHub issues](https://img.shields.io/github/issues/Aventardo7777/sovereign-rating-lab)](https://github.com/Aventardo7777/sovereign-rating-lab/issues)
[![Last commit](https://img.shields.io/github/last-commit/Aventardo7777/sovereign-rating-lab)](https://github.com/Aventardo7777/sovereign-rating-lab/commits/main)
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> **An open, reproducible research toolkit for sovereign credit rating migration and prediction.**
> Built a country-year-agency rating panel on a unified 1-21 scale, estimate annual
> transition matrices, study macroeconomic drivers with panel fixed effects, forecast
> ratings with ordinal Logit/Probit, Random Forest and XGBoost under strict
> time-series cross-validation, explain them with SHAP, and run event studies —
> all with tests, CI, and a Streamlit dashboard.

---

## 目录

- [项目简介](#项目简介)
- [研究问题](#研究问题)
- [数据来源与版权](#数据来源与版权)
- [方法概览](#方法概览)
- [项目结构](#项目结构)
- [安装](#安装)
- [快速开始](#快速开始)
- [如何导入评级数据](#如何导入评级数据)
- [如何运行 Notebook](#如何运行-notebook)
- [如何启动 Streamlit 仪表盘](#如何启动-streamlit-仪表盘)
- [如何部署](#如何部署)
- [复现步骤](#复现步骤)
- [开发与贡献](#开发与贡献)
- [引用](#引用)
- [License](#license)
- [免责声明](#免责声明)

---

## 项目简介

主权信用评级是国际资本市场的核心定价输入。评级变动（尤其下调）会传导至
主权融资成本、银行抵押品价值与企业跨境融资条件。

**sovereign-rating-lab** 提供一个完整、可复现、可扩展的研究工具箱，把
「原始评级记录 + 公开宏观数据」变成可检验的实证结果：

- 📐 **统一评级刻度** —— S&P / Moody's / Fitch 映射到 1-21 有序刻度
- 🔄 **迁移分析** —— 年度迁移矩阵、上下调概率、评级周期、机构分歧
- 📊 **驱动因素** —— 面板双向固定效应 + 聚类稳健标准误
- 🤖 **预测建模** —— 有序 Logit/Probit、随机森林、XGBoost + 时间序列 CV
- 🔍 **可解释性** —— SHAP 全局与局部解释
- ⚡ **事件研究** —— 评级行动前后 `[-20, +20]` 交易日的异常变化
- 📈 **交互仪表盘** —— Streamlit 一键浏览与导出
- ✅ **工程规范** —— 测试、CI、Docker、pre-commit、完整开源规范

### 设计原则

| 原则 | 具体做法 |
| --- | --- |
| **不伪造数据** | 不打包任何第三方版权评级数据；数据不可得时返回明确状态而非编造 |
| **可复现** | 固定随机种子；运行清单记录数据指纹；原始响应缓存并登记来源 |
| **诚实标注局限** | 每个方法在 `docs/methodology.md` 中显式列出假设与失效条件 |
| **合规优先** | 抓取强制校验 `robots.txt`；限速；原始 HTML 缓存并记录来源 URL |

---

## 研究问题

| 编号 | 问题 | 实现 |
| --- | --- | --- |
| **RQ1** | 主权评级在年度尺度上的迁移结构是什么？哪些档位更「粘」？ | `build_migration_matrix`、`rating_spell_stats` |
| **RQ2** | 上调与下调概率是否不对称？如何随时间、机构、评级档变化？ | `upgrade_downgrade_probabilities`、`downgrade_probability` |
| **RQ3** | 哪些宏观基本面因素与评级变动系统性相关？ | `fit_fixed_effects`、`run_driver_regressions` |
| **RQ4** | 能否用公开宏观数据预测评级水平与下调风险？ | `OrdinalProbabilityModel`、`build_xgboost`、`cross_validate_model` |
| **RQ5** | 评级行动是否伴随市场异常反应？反应在事件前还是事件后？ | `event_study` |

详细的研究设计、待检验假设与边界条件见 [`docs/research_design.md`](docs/research_design.md)。

---

## 数据来源与版权

| 来源 | 内容 | 密钥 | 状态 |
| --- | --- | :---: | --- |
| [World Bank WDI](https://data.worldbank.org) | 增长、通胀、债务、储备、经常账户、汇率 | 否 | ✅ 已接入 |
| [World Bank WGI](https://info.worldbank.org/governance/wgi/) | 六项治理指标 | 否 | ✅ 已接入 |
| [IMF WEO](https://www.imf.org/en/Publications/WEO) | 增长、通胀、政府债务、经常账户 | 否 | ✅ 已接入 |
| [BIS Statistics](https://stats.bis.org) | 政策利率、有效汇率、跨境信贷 | 否 | ⚙️ 默认禁用 |
| [FRED](https://fred.stlouisfed.org) | 利差、汇率、股指 | **是** | ⚙️ 需自配密钥 |
| 主权评级历史 | 评级、展望、行动日期 | — | 📥 用户导入 |

### ⚠️ 评级数据版权声明（必读）

S&P Global Ratings、Moody's、Fitch Ratings 的评级历史数据库均为**商业授权产品**。
本项目：

1. **不提供、不打包、不伪造**任何完整的真实评级历史数据；
2. 提供 [`data/raw/ratings_template.csv`](data/raw/ratings_template.csv) 作为**导入模板**；
3. 提供 [`data/sample/ratings_sample.csv`](data/sample/ratings_sample.csv) 作为
   **合成演示数据**，每行带 `provenance = synthetic_demo_only` 标记列，
   **明确不是真实评级数据**；
4. 提供**合规的公开页面解析脚本**（`fetch_public_ratings_page`），
   抓取前校验 `robots.txt`、限速、缓存原始 HTML 并登记来源 URL 与下载时间。

> **用户责任**：使用任何数据前，请自行确认你拥有合法使用权限。
> 本项目不对用户导入或抓取的数据的合法性负责。

### 🔑 FRED API Key 配置方式

FRED 密钥**绝不硬编码**，通过环境变量读取：

```bash
# Windows PowerShell（当前会话）
$env:FRED_API_KEY = "your_key_here"

# Windows（永久，需重开终端）
setx FRED_API_KEY "your_key_here"

# macOS / Linux
export FRED_API_KEY="your_key_here"
```

申请地址：<https://fred.stlouisfed.org/docs/api/api_key.html>

未配置密钥时，FRED 相关函数**不会报错中断**，而是返回空表并打印配置指引，
流水线的其余部分照常运行。

---

## 方法概览

```text
                    ┌──────────────────────────────────────────┐
   评级数据          │  评级映射  →  1-21 统一有序刻度            │
   (用户导入)        │  outlook / watch / 修饰符 / 违约 处理      │
                    └──────────────────┬───────────────────────┘
                                       │
   宏观数据           ┌───────────────▼──────────────────────────┐
   WDI/WGI/WEO       │  面板构建  国家-年份-机构                   │
   BIS/FRED          │  rating_score        （当年实际评级）        │
                     │  score_in_effect     （当年生效评级）        │
                     │  rating_change       （评级迁移）            │
                     └───────────────┬──────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
   ┌────▼─────┐              ┌───────▼────────┐          ┌────────▼────────┐
   │ 迁移分析  │              │  驱动因素回归   │          │   预测建模       │
   │ 迁移矩阵  │              │  双向固定效应   │          │  有序Logit/Probit│
   │ 上下调概率│              │  聚类稳健标准误 │          │  RF / XGBoost    │
   │ 评级周期  │              │  LPM + Logit   │          │  时间序列CV       │
   │ 机构分歧  │              └───────┬────────┘          │  + SHAP          │
   └────┬─────┘                      │                   └────────┬────────┘
        │                            │                            │
        └────────────────────────────┼────────────────────────────┘
                                     │
   ┌─────────────────────────────────▼─────────────────────────────────┐
   │  事件研究 · 报告 · 图表 · Streamlit 仪表盘                          │
   └───────────────────────────────────────────────────────────────────┘
```

### 关键方法选择

| 决策 | 选择 | 理由 |
| --- | --- | --- |
| 评级刻度 | 1-21 有序数值（越高越好） | 尊重评级的**次序**性质；投资级门槛为整数 12 |
| 面板聚合 | 按年取年末状态 | 与年度宏观数据对齐；迁移 horizon 明确为 1 年 |
| 迁移统计 | 只计**相邻年份** | 避免把数据缺口误当作迁移 |
| 标准误 | 按国家聚类 | 同一国家跨年扰动项高度相关 |
| 预测特征 | **滞后一期** | 避免用同期信息预测（伪回归） |
| 交叉验证 | 按年**前向链式** | 随机 K 折会「看到未来」，产生乐观偏差 |
| 缺失值 | 只填内部缺口（`forward`） | 序列开头缺失属结构性缺失，回填等于造数 |
| 评估重点 | 相邻档位准确率 + 方向 AUC | 评级分布极不平衡，精确命中不是实务目标 |

每一项方法的完整假设与局限见 [`docs/methodology.md`](docs/methodology.md)。

---

## 项目结构

```text
sovereign-rating-lab/
├─ README.md                    本文件
├─ LICENSE                      MIT
├─ CITATION.cff                 引用格式（Citation File Format 1.2.0）
├─ CHANGELOG.md                 Keep a Changelog 规范
├─ CONTRIBUTING.md              开发环境、代码规范、PR 流程
├─ CODE_OF_CONDUCT.md           Contributor Covenant 2.1
├─ SECURITY.md                  漏洞报告方式
├─ .editorconfig                统一编码风格
├─ .gitignore                    覆盖 Python / IDE / 数据 / 虚拟环境 / Jupyter / Streamlit
├─ .pre-commit-config.yaml      ruff + black + 文件卫生 + 密钥检查
├─ .github/
│  ├─ workflows/ci.yml          CI：ruff + black --check + pytest + 流水线冒烟测试
│  ├─ ISSUE_TEMPLATE/
│  │  ├─ bug_report.md
│  │  └─ feature_request.md
│  └─ PULL_REQUEST_TEMPLATE.md
├─ pyproject.toml               打包、依赖、black / ruff / pytest / coverage 配置
├─ requirements.txt             运行时依赖
├─ requirements-dev.txt          开发依赖
├─ Makefile                     常用任务快捷方式（make help）
├─ Dockerfile                   研究环境镜像（含中文字体）
├─ config.yaml                  **全局配置**（路径、样本、模型、数据源）
│
├─ data/
│  ├─ raw/
│  │  ├─ ratings_template.csv   **评级数据导入模板**（仅表头）
│  │  └─ README.md              版权说明与导入指南
│  ├─ sample/
│  │  ├─ ratings_sample.csv     **合成演示数据**（synthetic_demo_only）
│  │  ├─ macro_sample.csv       合成宏观面板
│  │  └─ README.md              合成数据说明与设计意图
│  ├─ interim/                  清洗中间产物（不提交）
│  └─ processed/                建模用面板（不提交，流水线生成）
│
├─ docs/
│  ├─ research_design.md        研究问题、假设、边界与局限
│  ├─ data_dictionary.md        **每一列数据的定义**
│  └─ methodology.md            **每一项方法的模型、假设与失效条件**
│
├─ notebooks/
│  ├─ 01_data_collection.ipynb  数据抓取与面板构建
│  ├─ 02_rating_migration.ipynb 迁移矩阵、上下调概率、评级周期、机构分歧
│  ├─ 03_macro_drivers.ipynb    面板固定效应回归与稳健性检验
│  ├─ 04_prediction_models.ipynb 有序模型、RF、XGBoost、时间序列 CV、SHAP
│  └─ 05_event_study.ipynb      事件研究框架
│
├─ src/
│  ├─ __init__.py
│  ├─ config.py                 配置加载与路径解析（唯一读 config.yaml 的地方）
│  ├─ pipeline.py               **端到端流水线**（python -m src.pipeline）
│  ├─ ingest/                   数据抓取
│  │  ├─ worldbank.py           WDI / WGI
│  │  ├─ imf.py                 IMF WEO
│  │  ├─ bis.py                 BIS（SDMX-JSON 通用解析器）
│  │  ├─ fred.py                FRED（环境变量密钥）
│  │  └─ ratings.py             评级数据载入、HTML 解析、导入校验
│  ├─ clean/
│  │  ├─ panel.py               面板构建与校验
│  │  └─ missing.py             缺失值诊断与三类处理策略
│  ├─ features/
│  │  ├─ rating_scale.py        **评级映射核心**（1-21 刻度）
│  │  └─ macro_features.py      滞后、差分、标准化、缩尾、特征矩阵
│  ├─ analysis/
│  │  ├─ migration.py           **迁移矩阵、上下调概率、评级周期、机构分歧**
│  │  ├─ panel_regression.py    固定效应 + 聚类稳健标准误
│  │  └─ event_study.py         事件研究框架
│  ├─ models/
│  │  ├─ ordinal.py             有序 Logit / Probit（+ sklearn 兼容封装）
│  │  ├─ tree_models.py         随机森林、XGBoost
│  │  ├─ timeseries_cv.py       前向链式交叉验证
│  │  ├─ evaluate.py            评估指标、混淆矩阵、特征重要性
│  │  └─ explain.py             SHAP
│  ├─ visualization/
│  │  ├─ plots.py               matplotlib / seaborn 静态图
│  │  └─ interactive.py         plotly 交互图
│  └─ utils/
│     ├─ http.py                合规 HTTP 客户端（robots.txt + 限速 + 缓存 + 溯源）
│     ├─ io.py                  数据读写与指纹
│     └─ logging_utils.py       统一日志
│
├─ app/
│  └─ streamlit_app.py          **交互式仪表盘**
│
├─ scripts/
│  ├─ make_sample_data.py       重新生成合成示例数据（确定性）
│  ├─ check_no_secrets.py       pre-commit：禁止硬编码密钥
│  └─ check_no_raw_data.py      pre-commit：禁止提交原始数据
│
├─ tests/
│  ├─ conftest.py               公共夹具
│  ├─ test_rating_mapping.py    评级映射（含 100+ 断言）
│  ├─ test_migration_matrix.py  迁移矩阵
│  ├─ test_missing_values.py    缺失值处理
│  └─ test_panel_and_models.py  面板构建与回归/建模集成
│
└─ reports/
   ├─ figures/                  图表输出（流水线生成）
   └─ tables/                   表格输出（含 run_manifest.json）
```

---

## 安装

### 前置要求

* **Python 3.11 或更高**（3.11 / 3.12 / 3.13 均已测试）
* Git
* （可选）Docker

### 方式 A：标准 venv（推荐）

```bash
git clone https://github.com/Aventardo7777/sovereign-rating-lab.git
cd sovereign-rating-lab

python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install -r requirements-dev.txt
```

### 方式 B：Makefile 一键

```bash
make setup      # 创建 .venv + 安装依赖 + 安装 pre-commit 钩子
make help       # 查看全部可用命令
```

### 方式 C：conda

```bash
conda create -n sovereign-rating-lab python=3.11 -y
conda activate sovereign-rating-lab
pip install -r requirements-dev.txt
```

### 方式 D：Docker

```bash
docker build -t sovereign-rating-lab:0.1.0 .
docker run --rm -p 8501:8501 sovereign-rating-lab:0.1.0          # 启动仪表盘
docker run --rm sovereign-rating-lab:0.1.0 python -m src.pipeline  # 跑流水线
docker run --rm sovereign-rating-lab:0.1.0 pytest -q               # 跑测试
```

### 安装为可编辑包（可选）

```bash
pip install -e .
```

这样可以在任意工作目录 `import src`，也便于在其他项目中复用工具函数。

---

## 快速开始

### 30 秒跑通全流程

```bash
# 1. 重新生成合成示例数据（固定种子，结果确定）
python scripts/make_sample_data.py

# 2. 运行端到端流水线（数据 → 面板 → 迁移 → 回归 → 建模 → 图表 → 报告）
python -m src.pipeline

# 3. 查看产出
ls reports/tables/      # 表格与运行清单
ls reports/figures/     # 图表

# 4. 启动交互仪表盘
streamlit run app/streamlit_app.py
```

> ⚠️ 上述流程使用的是**合成演示数据**。结果用于验证代码路径，
> **不构成任何实证结论**。

### 用你自己的数据

```bash
python -m src.pipeline --ratings data/raw/my_ratings.csv --macro data/raw/my_macro.csv
```

### 在 Python 中调用

```python
from src.ingest.ratings import load_ratings_csv
from src.clean.panel import build_country_year_panel
from src.analysis.migration import build_migration_matrix, migration_summary
from src.features.rating_scale import map_rating

# 评级映射
map_rating("AA+", "S&P")        # 20.0
map_rating("Baa3", "Moody's")   # 12.0
map_rating("NR", "S&P")         # nan

# 面板与迁移矩阵
ratings = load_ratings_csv("data/raw/my_ratings.csv")
panel = build_country_year_panel(ratings, macro=None)
matrix = build_migration_matrix(panel, score_is_effective=True)
print(migration_summary(matrix))
# {'n_transitions': 1234.0, 'p_stable': 0.93, 'p_upgrade': 0.03, 'p_downgrade': 0.04, ...}
```

---

## 如何导入评级数据

### 第 1 步：准备 CSV

参照 [`data/raw/ratings_template.csv`](data/raw/ratings_template.csv) 的列结构：

```csv
country_iso3,country_name,year,agency,rating,outlook,action,action_date
BRA,Brazil,2015,S&P,BB,Stable,downgrade,2015-09-09
BRA,Brazil,2016,S&P,BB-,Negative,downgrade,2016-02-17
CHN,China,2017,Moody's,Aa3,Stable,downgrade,2017-05-24
```

| 列 | 必需 | 说明 |
| --- | :---: | --- |
| `country_iso3` | ✅ | ISO 3166-1 alpha-3 代码 |
| `year` | ✅ | 年份 |
| `agency` | ✅ | `S&P` / `Moody's` / `Fitch`（支持 `SP`、`Moodys` 等别名） |
| `rating` | ✅ | 评级符号，支持 `AA+u`、`(P)AAA`、`BBB-*` 等写法 |
| `outlook` | ⬜ | 展望，支持 `CW-Negative`、`Positive Watch`、`+`/`-` 等 |
| `action_date` | ⬜ | `YYYY-MM-DD`（事件研究必需） |

**`rating_score` 无需提供**——程序会自动计算并覆盖，确保刻度一致。

完整的列说明与常见输入写法见 [`data/raw/README.md`](data/raw/README.md) 与
[`docs/data_dictionary.md`](docs/data_dictionary.md)。

### 第 2 步：校验

```python
from src.ingest.ratings import load_ratings_csv, validate_imported_ratings

ratings = load_ratings_csv("data/raw/my_ratings.csv")
print(validate_imported_ratings(ratings))
```

输出示例：

```text
                        check  n_violations  passed                                    detail
                  必需列齐全             0    True
             评级分值映射完成             3   False  未映射占比 0.42%（阈值 5%）
            机构名称可识别             0    True
               年份可解析             0    True
        国家代码为 3 位 ISO3            0    True
           年份连续（无缺口）            12   False  缺口年份会削弱迁移矩阵的有效样本量
 主键 (国家, 机构, 年份) 唯一            0    True
```

**未映射的记录不会丢失**——它们保留在数据中，`rating_map_reason` 列记录原因
（`no_rating` / `short_term` / `unmapped` / `missing`）。

### 第 3 步：运行

```bash
python -m src.pipeline --ratings data/raw/my_ratings.csv
```

所有分析、图表与报告会自动改用你的数据，并在
`reports/tables/run_manifest.json` 中记录数据指纹（便于日后核对版本）。

### 关于合规抓取

```python
from src.ingest.ratings import fetch_public_ratings_page

df = fetch_public_ratings_page(
    "https://example.com/sovereign-ratings",
    table_selector="table.ratings",
    column_map={"Country": "country_name", "Rating": "rating", "Outlook": "outlook"},
    agency="S&P",
)
```

该函数会自动：

1. 校验目标路径的 `robots.txt`（被禁止时抛 `RobotsDisallowedError`）
2. 请求间隔 ≥ 2 秒
3. 原始 HTML 缓存到 `data/raw/cache/html/`
4. 把 URL、抓取时间、状态码写入 `data/raw/cache/provenance.json`

> ⚠️ 使用前请确认目标站点的服务条款允许自动化抓取与再分发。

---

## 如何运行 Notebook

```bash
# 启动 Jupyter Lab
jupyter lab notebooks
# 或
make notebooks
```

按顺序运行：

| Notebook | 内容 | 前置条件 |
| --- | --- | --- |
| `01_data_collection.ipynb` | 数据抓取、映射、面板构建 | 无（离线可用示例数据） |
| `02_rating_migration.ipynb` | 迁移矩阵、上下调概率、评级周期、机构分歧 | 建议先跑 01 |
| `03_macro_drivers.ipynb` | 固定效应回归、稳健性检验、分组异质性 | 建议先跑 01 |
| `04_prediction_models.ipynb` | 有序模型、RF、XGBoost、时间序列 CV、SHAP | 建议先跑 01 |
| `05_event_study.ipynb` | 事件研究（含数据不足时的降级演示） | 建议先跑 01 |

**每个 notebook 都会自动检测 `data/processed/panel_country_year.csv`**：
存在则直接读取，不存在则从示例数据即时构建。因此可以独立运行。

> 💡 运行 `python -m src.pipeline` 后再打开 notebook，可以跳过重复的数据构建。

---

## 如何启动 Streamlit 仪表盘

```bash
streamlit run app/streamlit_app.py
# 或
make dashboard
```

浏览器会自动打开 <http://localhost:8501>。

### 仪表盘功能

| 板块 | 功能 |
| --- | --- |
| **1 · 概览** | 观测数、经济体数、评级行动次数、平均评级 |
| **2 · 评级轨迹** | 多机构评级轨迹对比，标注投资级门槛 |
| **3 · 迁移矩阵** | 交互热力图（概率 / 频数切换）+ 总体指标 + 按机构概率 |
| **4 · 机构分歧** | split rating 比例的时间趋势与分歧最大的国家-年份 |
| **5 · 宏观指标** | 选定国家的宏观指标与评级对照 |
| **6 · 预测概率** | 有序模型样本外的下调 / 稳定 / 上调概率 |
| **7 · 数据下载** | 导出筛选面板、完整面板、年度迁移概率 |

侧边栏可筛选**国家 / 机构 / 年份区间**。若 `data/processed/` 面板不存在，
应用会自动从示例数据构建，并在页面顶部显示合成数据警示。

### 自定义端口

```bash
streamlit run app/streamlit_app.py --server.port 8600
```

---

## 如何部署

### Streamlit Community Cloud（免费，最简）

1. Fork 本仓库到你的 GitHub 账号；
2. 访问 <https://share.streamlit.io/> → **New app**；
3. 选择你的 fork，Main file path 填 `app/streamlit_app.py`；
4. （可选）在 **Advanced settings → Secrets** 中配置 `FRED_API_KEY`；
5. 点击 **Deploy**。

> ⚠️ 部署到公网前请注意：如果你接入的是有版权限制的数据，
> 公网分享可能构成再分发。**建议只在公网部署合成示例数据的版本。**

### Docker

```bash
docker build -t sovereign-rating-lab:0.1.0 .
docker run -d --name srl -p 8501:8501 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/reports:/app/reports" \
  sovereign-rating-lab:0.1.0
```

### 反向代理（Nginx 示例）

```nginx
location / {
    proxy_pass http://127.0.0.1:8501;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_read_timeout 86400;
}
```

### 生成静态报告（不走 Web）

```bash
python -m src.pipeline       # 产出 reports/figures/ 与 reports/tables/
```

`reports/tables/*.csv` 可直接导入 LaTeX、Excel 或 Word；
`reports/figures/*.png` 为 150 DPI 图片，可直接用于论文。

### 定时更新（GitHub Actions 示例）

在本仓库的 `.github/workflows/` 下新建一个定时工作流：

```yaml
on:
  schedule:
    - cron: "0 6 * * 1"   # 每周一 06:00 UTC
jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: python -m src.pipeline --no-models
```

---

## 复现步骤

### 完整复现（从零到结果）

```bash
# 1. 克隆
git clone https://github.com/Aventardo7777/sovereign-rating-lab.git
cd sovereign-rating-lab

# 2. 环境
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# 3. 验证代码库健康
make check                         # ruff + black --check + pytest

# 4. 复现示例结果
python scripts/make_sample_data.py # 重新生成合成数据（确定性）
python -m src.pipeline             # 跑通全流程

# 5. 核对运行清单
cat reports/tables/run_manifest.json
```

`run_manifest.json` 中的 `dataframe_fingerprint` 可用于核对是否使用了同一份数据。

### 用真实数据复现

```bash
python -m src.pipeline \
  --ratings data/raw/ratings.csv \
  --macro   data/raw/macro.csv
```

### 论文写作建议

在方法部分明确说明：

1. 使用的数据源与获取日期（WEO 会修订历史值，需注明版本如「WEO 2024/04」）
2. 面板构建的关键选择（年末状态、相邻年份限制）
3. 交叉验证方案（前向链式，训练集上界）
4. `docs/methodology.md` 中列出的**哪些局限适用于你的数据**

---

## 开发与贡献

欢迎贡献！请先阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md) 与
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)。

### 开发环境

```bash
make setup       # 创建 .venv + 安装依赖 + pre-commit 钩子
make format      # 自动格式化（black + ruff --fix）
make lint        # ruff check + black --check
make test        # pytest
make check       # lint + test（提交前必跑）
```

### 提交规范

采用 [Conventional Commits](https://www.conventionalcommits.org/)：

```text
feat(models): add discrete-time hazard model for rating transitions
fix(migration): exclude non-consecutive years from transition counts
docs(readme): clarify rating data licensing constraints
```

### 三条红线

提交的代码**不得**触犯以下任何一条：

1. ❌ 不打包任何第三方评级机构的版权数据
2. ❌ 不硬编码 API 密钥（pre-commit 与 CI 都会检查）
3. ❌ 不伪造数据（数据不可得时返回空结果 + 说明）

### 报告问题

* Bug → [Bug 报告模板](.github/ISSUE_TEMPLATE/bug_report.md)
* 功能建议 → [功能建议模板](.github/ISSUE_TEMPLATE/feature_request.md)
* 安全漏洞 → 见 [`SECURITY.md`](SECURITY.md)（**不要**开公开 Issue）

---

## 引用

如果你在学术工作中使用本项目，请按 [`CITATION.cff`](CITATION.cff) 引用：

```bibtex
@software{liu2026sovereign,
  title        = {sovereign-rating-lab: Sovereign Credit Rating Migration and Prediction Lab},
  author       = {Liu, Jietao and {sovereign-rating-lab contributors}},
  year         = {2026},
  version      = {0.1.0},
  url          = {https://github.com/Aventardo7777/sovereign-rating-lab},
  license      = {MIT}
}
```

GitHub 会在仓库右侧显示 **"Cite this repository"** 按钮（基于 `CITATION.cff` 自动生成）。

> **另请引用你实际使用的数据源**，例如 World Bank WDI/WGI、IMF WEO，
> 以及你所使用的方法文献。

---

## License

本项目采用 **MIT License** —— 详见 [`LICENSE`](LICENSE)。

```text
MIT License · Copyright (c) 2026 Jietao Liu
```

> **注意**：MIT 许可**仅覆盖本项目的代码与文档**，不覆盖任何第三方数据。
> 评级数据的权利归各评级机构所有。

---

## 免责声明

1. **本项目仅用于研究与教学目的。** 所有输出均为统计结果，**不构成投资建议**、
   不构成任何形式的金融意见，不得用于信贷决策、交易执行或监管报送。

2. **本项目不提供评级数据。** S&P Global Ratings、Moody's、Fitch Ratings 的
   评级历史均为商业授权产品。使用者需**自行确认数据来源的合法性与使用权限**，
   并对使用行为及其后果承担全部责任。

3. **合成示例数据不是真实数据。** `data/sample/` 下的所有 CSV 由
   `scripts/make_sample_data.py` 程序生成，带有 `provenance = synthetic_demo_only`
   标记，不代表任何真实国家、机构或评级的实际状况，**不得用于任何实证结论**。

4. **模型结果存在不确定性。** 评级变动受大量不可观测因素影响；本项目文档中
   显式列出的假设与局限（见 `docs/methodology.md`）必须在解读结果时予以考虑。
   本项目不做因果推断。

5. **数据源可用性不受本项目控制。** World Bank、IMF、BIS、FRED 的接口可能
   变更或中断。本项目会给出提示并优雅降级，但不保证任何数据源的持续可用性。

6. **软件按「原样」提供。** 作者不对使用本软件造成的任何损失承担责任。

---

<p align="center">
  <sub>Made for reproducible sovereign credit research · MIT License · 2026</sub>
</p>
