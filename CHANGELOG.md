# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Survival analysis (Cox / discrete-time hazard) for rating transition timing.
- Panel VAR / local projections for dynamic macro-rating responses.
- Bootstrap confidence intervals for transition matrices.
- Survival-adjusted (right-censoring aware) rating duration statistics.
- Additional data connectors (IMF IFS, national central banks).

## [0.1.0] - 2026-09-12

### Added

- **统一评级刻度**：S&P / Moody's / Fitch 长期主权评级映射到 1-21 有序数值刻度，
  处理展望、信用观察、未邀约（`u`）、临时（`p`）、结构化融资（`sf`）等修饰符，
  以及违约类评级（`D`/`SD`/`RD`）与短期评级（`A-1+`/`P-1`/`F1`）的区分。
- **面板构建**：把事件式评级记录转换为国家-年份-机构面板，区分
  `rating_score`（当年实际评级）与 `score_in_effect`（当年生效评级），
  并导出 `rating_change` / `is_downgrade` / `is_upgrade` 等迁移变量。
- **迁移分析**：年度迁移矩阵（频数 / 概率）、上调下调概率（按年 / 机构 / 评级档分组）、
  多期下调概率、评级周期与停留片段统计、迁移熵、机构间分歧（split rating）度量。
- **宏观驱动回归**：面板双向固定效应 + 国家聚类稳健标准误，
  二元因变量的线性概率模型与 Logit 稳健性检验，模型对比表。
- **预测模型**：有序 Logit / Probit（含 scikit-learn 兼容封装）、
  随机森林、XGBoost；按年前向链式（expanding/rolling）时间序列交叉验证；
  准确率、平衡准确率、宏平均 F1、有序 MAE、相邻档位准确率、二次加权 kappa、
  多分类 AUC、对数损失等指标；混淆矩阵与特征重要性输出。
- **可解释性**：SHAP（TreeExplainer / KernelExplainer 自动选择，失败时优雅降级）
  全局重要性表与逐样本贡献矩阵。
- **事件研究**：`[-20, +20]` 交易日事件窗口，市场模型与均值调整模型，
  AAR / CAAR 汇总，数据不足时返回明确状态而不伪造数据。
- **数据接入**：World Bank WDI / WGI、IMF WEO、BIS（SDMX-JSON）、FRED（环境变量密钥）、
  合规的评级公开页面解析（robots.txt 校验 + 限速 + 原始 HTML 缓存 + 来源登记）。
- **可视化**：matplotlib / seaborn 静态图（迁移热力图、评级轨迹、系数森林图、
  混淆矩阵、特征重要性、概率演化）与 plotly 交互图。
- **Streamlit 仪表盘**：国家 / 机构 / 年份筛选、评级轨迹、迁移热力图、
  机构分歧、宏观指标、下调预警概率、数据导出。
- **示例数据与导入模板**：`data/sample/ratings_sample.csv`（合成演示数据，
  带 `provenance` 标记列）与 `data/raw/ratings_template.csv`（导入模板）。
- **复现流水线**：`python -m src.pipeline` 一键跑通全链路并输出
  图表、表格与运行清单（含数据指纹与全部警告）。
- **工程规范**：`pyproject.toml`、`requirements.txt`、`Makefile`、`Dockerfile`、
  GitHub Actions CI（ruff + black + pytest on Python 3.11/3.12/3.13）、
  pre-commit 钩子、Issue / PR 模板、Contributor Covenant 行为准则、
  SECURITY 政策与 CITATION.cff。
- **测试**：评级映射、迁移矩阵、缺失值处理三大模块的单元测试，
  以及面板构建与回归的集成测试。

### Notes

- 本项目**不打包**任何第三方评级机构的版权数据。示例数据为人工生成的合成数据，
  明确标注 `provenance = synthetic_demo_only`，不得用于实证结论。

[Unreleased]: https://github.com/Aventardo7777/sovereign-rating-lab/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Aventardo7777/sovereign-rating-lab/releases/tag/v0.1.0
