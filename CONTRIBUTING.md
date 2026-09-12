# Contributing to sovereign-rating-lab

感谢你愿意为本项目贡献代码、数据或研究方法。本文档说明开发环境、代码规范与提交流程。

> **English summary** — Contributions are welcome. Please read
> [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) first, follow the code style enforced by
> `ruff` and `black` (line length 100), add tests for behaviour changes, and open a
> Pull Request using the provided template. For research-method changes, please also
> update the relevant document under `docs/`.

---

## 1. 可以贡献什么

| 类型 | 说明 | 从哪里开始 |
| --- | --- | --- |
| 🐛 Bug 修复 | 计算错误、边界情形、文档与代码不一致 | [Bug 报告模板](.github/ISSUE_TEMPLATE/bug_report.md) |
| ✨ 新功能 | 新模型、新指标、新数据源 | [功能建议模板](.github/ISSUE_TEMPLATE/feature_request.md) |
| 📈 数据接入 | 更多公开数据源（IMF IFS、各国央行等） | `src/ingest/` |
| 📊 可视化 | 更清晰的图表与仪表盘 | `src/visualization/`、`app/` |
| 📝 文档 | 方法说明、教程、翻译 | `docs/`、`README.md` |
| 🧪 测试 | 提升覆盖率、补充边界情形 | `tests/` |

### 不能接受的内容

* ❌ 任何第三方评级机构（S&P / Moody's / Fitch 等）的**版权数据**，即使是以示例形式。
* ❌ 伪造的「真实」数据。所有非公开数据必须明确标注来源与获取方式。
* ❌ 硬编码的 API 密钥、令牌或个人凭证。
* ❌ 绕过目标站点 `robots.txt`、服务条款或访问限制的抓取代码。
* ❌ 把模型输出包装成投资建议的功能。

---

## 2. 开发环境

### 2.1 获取代码

```bash
git clone https://github.com/Aventardo7777/sovereign-rating-lab.git
cd sovereign-rating-lab
```

### 2.2 创建虚拟环境

```bash
# 方式 A：标准 venv（推荐）
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install -r requirements-dev.txt
```

```bash
# 方式 B：conda
conda create -n sovereign-rating-lab python=3.11 -y
conda activate sovereign-rating-lab
pip install -r requirements-dev.txt
```

支持 **Python 3.11 / 3.12 / 3.13**。3.11 为最低支持版本，也是 CI 的基准版本。

### 2.3 安装 pre-commit 钩子

```bash
pre-commit install
```

钩子会在每次 `git commit` 前自动运行 `ruff`、`black` 与通用文件检查，
自动修复可修复的问题。首次运行会下载钩子环境，需要网络。

### 2.4 可选：安装为可编辑包

```bash
pip install -e .
```

这样可以 `import src` 而不依赖工作目录，也便于在项目外部复用工具函数。

### 2.5 配置 API 密钥（可选）

只有 FRED 数据源需要密钥，**绝不写入代码或配置文件**：

```bash
# Windows PowerShell（当前会话）
$env:FRED_API_KEY = "your_key_here"

# macOS / Linux
export FRED_API_KEY="your_key_here"
```

申请地址：<https://fred.stlouisfed.org/docs/api/api_key.html>

---

## 3. 代码规范

### 3.1 自动格式化与检查

```bash
make format     # black + ruff --fix
make lint       # ruff check + black --check
make test       # pytest
make check      # lint + test（提交前的完整检查）
```

等价的手工命令：

```bash
ruff check . --fix
ruff format .          # 可选，本项目以 black 为准
black .
black --check .
pytest -q
```

### 3.2 硬性要求

| 项目 | 要求 |
| --- | --- |
| 行宽 | 100 字符（`black` / `ruff` 共同约束） |
| 导入顺序 | `ruff` 的 isort 规则（标准库 → 第三方 → 本项目 `src`） |
| 类型注解 | 所有公开函数必须标注参数与返回值类型 |
| 文档字符串 | 所有公开模块、类、函数必须有 docstring，中英文均可，**优先中文** |
| 文件编码 | UTF-8，LF 换行（见 `.editorconfig`） |
| 命名 | `snake_case`（函数/变量）、`PascalCase`（类）、模块内常量 `UPPER_SNAKE` |

### 3.3 代码设计约定

1. **不要在库代码里调用 `print`** —— 使用 `src.utils.logging_utils.get_logger(__name__)`。
2. **不要在库代码里 `plt.show()`** —— 返回 `Figure` 对象，由调用方决定保存或展示。
3. **不要在源码或配置中硬编码路径** —— 通过 `src.config.get_path()` / `resolve_path()`。
4. **不要在源码或配置中硬编码密钥** —— 通过 `src.config.get_env()` 读取环境变量。
5. **缺失数据不要静默填充** —— 使用 `src.clean.missing` 中的函数，并保留 `*_imputed` 标记。
6. **不要伪造数据** —— 数据不可得时应返回空结果 + 明确说明，而不是编造。
7. **随机性必须可复现** —— 所有随机过程显式传入 `random_state` / `np.random.default_rng(seed)`。

### 3.4 文档字符串风格

```python
def build_migration_matrix(panel: pd.DataFrame, *, normalize: bool = False) -> pd.DataFrame:
    """构建年度评级迁移矩阵。

    Parameters
    ----------
    panel:
        长表面板，至少含 ``country_iso3`` / ``agency`` / ``year`` / ``rating_score``。
    normalize:
        ``False`` 返回频数矩阵；``True`` 返回按行归一化的概率矩阵。

    Returns
    -------
    pandas.DataFrame
        索引与列均为 1-21 的方阵。

    Examples
    --------
    >>> matrix = build_migration_matrix(panel)
    >>> matrix.shape
    (21, 21)
    """
```

---

## 4. 测试

### 4.1 运行

```bash
pytest                                  # 全部测试
pytest tests/test_rating_mapping.py -v  # 单个文件
pytest -k "migration" -v                # 按关键字
pytest --cov=src --cov-report=term-missing   # 覆盖率
pytest -m "not slow"                    # 跳过慢测试
```

### 4.2 测试要求

* **新增或修改功能必须附带测试**。仅重构的改动不应改变测试结果。
* 测试必须**完全确定**：不依赖网络、不依赖当前日期、不依赖本地时区。
* 涉及网络的测试必须标记 `@pytest.mark.network`，默认在 CI 中跳过。
* 涉及真实数据文件的测试请使用 `tmp_path` 夹具，不要污染 `data/`。
* 优先使用 `tests/conftest.py` 中的小型夹具，而不是加载完整数据集。

### 4.3 数据相关的改动

如果你修改了 `scripts/make_sample_data.py` 或 `data/sample/` 下的文件，
请在 PR 描述中说明改动的原因，并确认示例数据仍然带有
`provenance = synthetic_demo_only` 标记。

---

## 5. 提交与 Pull Request 流程

### 5.1 分支命名

| 前缀 | 用途 | 示例 |
| --- | --- | --- |
| `feat/` | 新功能 | `feat/survival-model` |
| `fix/` | Bug 修复 | `fix/migration-gap-handling` |
| `docs/` | 文档 | `docs/data-dictionary-typo` |
| `refactor/` | 重构 | `refactor/split-ingest-module` |
| `test/` | 测试 | `test/add-missing-value-cases` |
| `chore/` | 构建、依赖、CI | `chore/bump-xgboost` |

### 5.2 提交信息

采用 [Conventional Commits](https://www.conventionalcommits.org/)：

```text
<type>(<scope>): <简短描述>

<可选：为什么这样改、影响了什么>
```

示例：

```text
feat(models): add discrete-time hazard model for rating transitions

fix(migration): exclude non-consecutive years from transition counts

Non-adjacent years were being counted as transitions, inflating the
diagonal share. Closes #42.

docs(readme): clarify rating data licensing constraints
```

### 5.3 完整流程

```bash
# 1. 从最新的 main 创建分支
git checkout main
git pull origin main
git checkout -b feat/your-feature

# 2. 开发并本地验证
make check          # lint + test
make pipeline       # 确认端到端流水线仍然可运行

# 3. 提交
git add .
git commit -m "feat(scope): your change"

# 4. 推送并开 PR
git push -u origin feat/your-feature
```

然后在 GitHub 上发起 Pull Request，使用仓库提供的
[PR 模板](.github/PULL_REQUEST_TEMPLATE.md)。

### 5.4 评审标准

PR 会被从以下角度评审：

- [ ] 逻辑正确性，特别是**边界情形**（缺失值、缺口年份、单机构国家、违约状态）
- [ ] 是否引入了前视偏差（look-ahead bias）—— 这是本类研究最常见的错误
- [ ] 测试是否覆盖新行为
- [ ] 文档（含 `docs/`、docstring、CHANGELOG）是否同步更新
- [ ] 是否违反「不伪造数据 / 不硬编码密钥 / 不打包版权数据」三条红线

### 5.5 CHANGELOG 维护

任何**面向使用者**的改动都必须在 `CHANGELOG.md` 的 `[Unreleased]` 段落中添加条目，
分类为 `Added` / `Changed` / `Deprecated` / `Removed` / `Fixed` / `Security`。

---

## 6. 研究方法相关的贡献

如果你的改动涉及**方法论**（新模型、新指标、新的识别策略），
除了代码与测试，请一并更新：

* `docs/methodology.md` —— 方法的假设、实现细节与局限
* `docs/research_design.md` —— 研究设计层面的定位（若涉及）

并在 PR 描述中说明新方法相对于现有方法的优势与代价。

---

## 7. 问题与讨论

* 使用 Issue 模板提问，避免直接开 PR 讨论设计。
* 报告 Bug 时请提供：Python 版本、完整依赖版本（`pip freeze`）、
  最小可复现代码、完整报错栈。
* 安全相关问题请走 [`SECURITY.md`](SECURITY.md) 中的私密渠道，不要开公开 Issue。

再次感谢你的贡献！🌊
