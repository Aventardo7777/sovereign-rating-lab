# 数据字典

> 本文档定义项目中的**每一列数据**：含义、类型、取值范围、缺失规则与来源。
> 任何新增字段都必须在此登记。

---

## 目录

1. [评级刻度（1-21）](#1-评级刻度1-21)
2. [原始评级数据](#2-原始评级数据data-raw--data-sample)
3. [面板数据](#3-面板数据data-processed)
4. [宏观指标](#4-宏观指标)
5. [治理指标（WGI）](#5-治理指标wgi)
6. [建模矩阵](#6-建模矩阵)
7. [输出表格](#7-输出表格reports-tables)
8. [缺失值约定](#8-缺失值约定)

---

## 1. 评级刻度（1-21）

**分值越高 = 信用质量越好。** 分值 1 为违约档。

| score | S&P | Fitch | Moody's | 投资级 | 大档 |
| ---: | --- | --- | --- | :---: | --- |
| 21 | AAA | AAA | Aaa | ✅ | AAA |
| 20 | AA+ | AA+ | Aa1 | ✅ | AA |
| 19 | AA | AA | Aa2 | ✅ | AA |
| 18 | AA- | AA- | Aa3 | ✅ | AA |
| 17 | A+ | A+ | A1 | ✅ | A |
| 16 | A | A | A2 | ✅ | A |
| 15 | A- | A- | A3 | ✅ | A |
| 14 | BBB+ | BBB+ | Baa1 | ✅ | BBB |
| 13 | BBB | BBB | Baa2 | ✅ | BBB |
| 12 | BBB- | BBB- | Baa3 | ✅ | BBB |
| 11 | BB+ | BB+ | Ba1 | ❌ | BB |
| 10 | BB | BB | Ba2 | ❌ | BB |
| 9 | BB- | BB- | Ba3 | ❌ | BB |
| 8 | B+ | B+ | B1 | ❌ | B |
| 7 | B | B | B2 | ❌ | B |
| 6 | B- | B- | B3 | ❌ | B |
| 5 | CCC+ | CCC+ | Caa1 | ❌ | CCC及以下 |
| 4 | CCC | CCC | Caa2 | ❌ | CCC及以下 |
| 3 | CCC- | CCC- | Caa3 | ❌ | CCC及以下 |
| 2 | CC | CC | Ca | ❌ | CCC及以下 |
| 1 | C / D / SD / RD | C / D | C | ❌ | CCC及以下 |

### 刻度设计说明

* **投资级门槛为 12**：对应 BBB-（S&P/Fitch）与 Baa3（Moody's），
  与市场惯例一致，且为整数便于分组。
* **S&P 的违约类评级合并到 1 分**：`D`（default）、`SD`（selective default）、
  `RD`（restricted default）、`DD`、`DDD` 与 `C` 统一映射到 1。
  这样把 S&P 的 22 个符号压缩到 21 档，与 Moody's 的 21 档一一对应。
  **代价**：无法区分「选择性违约」与「完全违约」。若研究需要该区分，
  请在使用原始符号列 `rating` 而非 `rating_score`。
* **短期评级不在刻度范围内**：`A-1+`、`P-1`、`F1` 等映射结果为缺失值，
  `rating_map_reason = "short_term"`。
* **大档（bucket）**：AAA / AA / A / BBB / BB / B / CCC及以下，共 7 档，
  用于降低迁移矩阵的视觉复杂度与稀疏性。

### 修饰符处理

| 原始写法 | 处理 | 说明 |
| --- | --- | --- |
| `AA+u` | → `AA+` | `u` = unsolicited（未邀约评级） |
| `(P)AAA` | → `AAA` | 括号 = provisional（临时评级） |
| `BBB+*` | → `BBB+` | 上标符号 |
| `AA- (sf)` | → `AA-` | `sf` = structured finance |
| `A  p` | → `A` | `p` = provisional |
| `Baa1 (hyb)` | → `Baa1` | 括号注释 |
| `C/D` | → 1 分 | 复合符号，属违约类 |

### 展望与信用观察

| 标准取值 | 常见输入写法 |
| --- | --- |
| `Positive` | `positive`、`pos`、`+`、`正面`、`积极` |
| `Negative` | `negative`、`neg`、`-`、`负面`、`消极` |
| `Stable` | `stable`、`sta`、`稳定` |
| `Developing` | `developing`、`evolving`、`uncertain`、`发展中` |
| `Watch-Positive` | `CW-Positive`、`Positive Watch`、`CreditWatch Positive`、`观察正面` |
| `Watch-Negative` | `CW-Negative`、`Negative Watch`、`CreditWatch Negative`、`观察负面` |
| `Watch-Developing` | `CW-Developing`、`Developing Watch`、`观察发展中` |
| 缺失（`None`） | `n/a`、`N/A`、`NR`、`none`、空字符串 |

---

## 2. 原始评级数据（`data/raw/` 与 `data/sample/`）

### `ratings_template.csv`（导入模板）

| 列名 | 类型 | 必需 | 说明 |
| --- | --- | :---: | --- |
| `country_iso3` | str(3) | ✅ | ISO 3166-1 alpha-3 代码 |
| `country_name` | str | ⬜ | 国家名称 |
| `year` | int | ✅ | 年份 |
| `agency` | str | ✅ | 评级机构（接受别名） |
| `rating` | str | ✅ | 评级符号 |
| `outlook` | str | ⬜ | 展望 |
| `action` | str | ⬜ | `assign` / `upgrade` / `downgrade` / `affirm` |
| `action_date` | date | ⬜ | `YYYY-MM-DD` |

### 列名别名（导入时自动转换）

| 输入列名 | 标准化为 |
| --- | --- |
| `iso3`、`iso`、`country_code`、`ccode` | `country_iso3` |
| `country`、`economy`、`国别` | `country_name` |
| `年份` | `year` |
| `机构` | `agency` |
| `评级` | `rating` |
| `展望` | `outlook` |
| `日期`、`date`、`event_date`、`rating_date` | `action_date` |
| `行动`、`action_type`、`rating_action` | `action` |
| `rating_agency` | `agency` |
| `credit_rating` | `rating` |

### 程序新增列（`load_ratings_csv` 后）

| 列名 | 类型 | 说明 |
| --- | --- | --- |
| `rating_score` | float64 | 1-21 分值；无法映射时为 `NaN` |
| `rating_map_reason` | str | `ok` / `missing` / `no_rating` / `short_term` / `unmapped` / `numeric_out_of_range` |

### `ratings_sample.csv`（合成演示数据）

列结构同模板，另加：

| 列名 | 说明 |
| --- | --- |
| `provenance` | 固定为 `synthetic_demo_only` |
| `data_note` | 数据声明文本 |

> ⚠️ 这是**合成数据**，不是真实评级。详见 `data/sample/README.md`。

---

## 3. 面板数据（`data/processed/`）

### `panel_country_year.csv`

主键：`(country_iso3, agency, year)`

| 列名 | 类型 | 说明 |
| --- | --- | --- |
| `country_iso3` | str | ISO3 国家代码 |
| `country_name` | str | 国家名称 |
| `agency` | str | `S&P` / `Moody's` / `Fitch` |
| `year` | Int64 | 年份 |
| `rating` | str | 该年**实际发生**的评级（无行动年份为空） |
| `rating_score` | float64 | `rating` 对应的 1-21 分值（无行动年份为空） |
| `outlook` | str | 该年行动时的展望 |
| `action` | str | 行动类型 |
| `action_date` | datetime | 行动日期 |
| **`has_action`** | bool | 该年是否有评级行动 |
| **`rating_in_effect`** | str | 该年**生效**的评级（无行动年份沿用上一次） |
| **`score_in_effect`** | float64 | 生效评级的 1-21 分值 |
| **`rating_change`** | float64 | `score_in_effect` 相对上一年的变化；首年为 `NaN` |
| **`action_type`** | str | `upgrade` / `downgrade` / `affirm` / `None` |
| **`is_downgrade`** | int | `rating_change < 0` |
| **`is_upgrade`** | int | `rating_change > 0` |
| **`is_stable`** | int | `rating_change == 0` |

**加粗列**为程序派生列，不要在导入文件中提供。

### 关键区分：`rating_score` vs `score_in_effect`

```text
年份      2019  2020  2021  2022  2023
评级行动   AA    —     AA-   —     A+
rating_score    14    空    13    空    17
score_in_effect 14    14    13    13    17
rating_change   空    0     -1    0     +4
```

* **迁移分析**与**预测模型**使用 `score_in_effect` 与 `rating_change`
* **事件研究**使用 `rating_score`（真实行动）与 `action_date`

### 宏观指标列

面板会左连接 `macro_sample.csv` 或用户提供的宏观面板的所有列（见下节）。

### 派生特征列（`add_derived_features`）

| 列名 | 计算方式 | 说明 |
| --- | --- | --- |
| `gdp_per_capita_log` | `ln(gdp_per_capita)` | 人均收入的右偏修正 |
| `fx_depreciation` | `fx_official_rate` 的组内百分比变化 | 正值为本币贬值 |

---

## 4. 宏观指标

### World Bank WDI（`config.yaml` → `data_sources.worldbank.indicators`）

| 内部键 | WDI 指标代码 | 含义 | 单位 |
| --- | --- | --- | --- |
| `gdp_growth` | `NY.GDP.MKTP.KD.ZG` | GDP 增长率 | % |
| `inflation` | `FP.CPI.TOTL.ZG` | CPI 通胀率 | % |
| `central_gov_debt_gdp` | `GC.DOD.TOTL.GD.ZS` | 中央政府债务 / GDP | % |
| `gross_debt_gdp` | `GC.DOD.TOTL.GD.ZS` | 一般政府总债务 / GDP | % |
| `external_debt_gni` | `DT.DOD.DECT.GN.ZS` | 外债存量 / GNI | % |
| `reserves_months_imports` | `FI.RES.TOTL.MO` | 外汇储备 / 进口月数 | 月 |
| `reserves_usd` | `FI.RES.TOTL.CD` | 外汇储备总额 | 现价美元 |
| `current_account_gdp` | `BN.CAB.XOKA.GD.ZS` | 经常账户余额 / GDP | % |
| `exports_gdp` | `NE.EXP.GNFS.ZS` | 出口 / GDP | % |
| `gdp_per_capita` | `NY.GDP.PCAP.CD` | 人均 GDP | 现价美元 |
| `fx_official_rate` | `PA.NUS.FCRF` | 官方汇率（本币/美元，期间均值） | 本币/美元 |
| `real_interest_rate` | `FR.INR.RINR` | 实际利率 | % |

### IMF WEO（`data_sources.imf.indicators`）

| 内部键 | WEO 代码 | 含义 |
| --- | --- | --- |
| `ngdp_rpch` | `NGDP_RPCH` | 实际 GDP 增长 % |
| `pcpipch` | `PCPIPCH` | 通胀 % |
| `ggxwdg_ngdp` | `GGXWDG_NGDP` | 一般政府总债务 / GDP |
| `bca_ngdp` | `BCA_NGDPD` | 经常账户余额 / GDP |
| `lp` | `LP` | 人口 |

> ⚠️ WEO 历史值会被修订。若需严格复现，请在论文中注明 WEO 版本
> （如「WEO 2024/04」）或在本地缓存中固定下载快照。

### 非官方来源指标（需用户自行接入）

| 列名 | 说明 | 常见来源 |
| --- | --- | --- |
| `spread_bps` | 主权债券利差（基点） | EMBI（付费）、FRED（部分国家） |
| `fiscal_balance_gdp` | 财政余额 / GDP | IMF WEO、各国财政部 |

### FRED 序列（`data_sources.fred.series`，需 `FRED_API_KEY`）

| 序列 ID | 含义 |
| --- | --- |
| `DGS10` | 美国 10 年期国债收益率（日频） |
| `VIXCLS` | VIX 波动率指数（日频） |
| `DTWEXBGS` | 美元贸易加权指数（日频） |

---

## 5. 治理指标（WGI）

Worldwide Governance Indicators，估计值约在 **-2.5 至 2.5** 之间（越高越好）。

| 内部键 | WGI 代码 | 含义 |
| --- | --- | --- |
| `voice_accountability` | `VA.EST` | 话语权与问责制 |
| `political_stability` | `PV.EST` | 政治稳定与无暴力 |
| `government_effectiveness` | `GE.EST` | 政府效能 |
| `regulatory_quality` | `RQ.EST` | 监管质量 |
| `rule_of_law` | `RL.EST` | 法治 |
| `control_corruption` | `CC.EST` | 腐败控制 |

> ⚠️ **结构性缺失**：WGI 自 1996 年起发布（1996-2002 为双年发布），
> 早期年份与部分小国缺失严重。这属于「指标尚未统计」而非「数值未知」，
> **不应插值填补**。默认插值方向为 `forward`，不填补序列开头的缺失。

---

## 6. 建模矩阵

### `data/processed/model_matrix.csv`

| 列名 | 说明 |
| --- | --- |
| `<feature>_lag1` | t-1 期的宏观特征（每个宏观变量一个滞后列） |
| `<feature>_was_missing` | 原始值缺失的标记（0/1）；缺失按列中位数填补后生成的审计列 |
| `score_in_effect` | **目标变量**：t 期生效评级分值 |
| `prior_score` | t-1 期生效评级（用于拆分下调/上调概率） |
| `country_iso3` / `agency` / `year` | 识别列 |

### 默认特征清单（`src/features/macro_features.py` → `DEFAULT_FEATURES`）

| 组 | 特征 |
| --- | --- |
| 增长 | `gdp_growth`、`gdp_per_capita_log` |
| 物价 | `inflation` |
| 财政 | `gov_debt_gdp` |
| 外部 | `external_debt_gni`、`reserves_months_imports`、`current_account_gdp`、`exports_gdp` |
| 金融 | `fx_depreciation`、`real_interest_rate`、`spread_bps` |
| 治理 | 六项 WGI 指标 |

共 17 个特征。缺失值按列中位数填补，并新增 `<feature>_was_missing` 布尔列，
使模型能够区分「真实值」与「填补值」。

---

## 7. 输出表格（`reports/tables/`）

| 文件 | 内容 | 关键列 |
| --- | --- | --- |
| `panel_validation.csv` | 面板完整性检查 | `check` / `n_violations` / `passed` |
| `panel_coverage.csv` | 按机构的面板覆盖汇总 | `n_obs` / `n_countries` / `year_min` / `year_max` |
| `rating_scale_1_to_21.csv` | 完整映射表 | `rating_score` / `sp` / `fitch` / `moodys` |
| `missingness_report_raw.csv` | 缺失诊断 | `column` / `n_missing` / `pct_missing` |
| `missingness_fill_report.csv` | 填补前后对比 | `n_missing_before` / `n_missing_after` / `n_filled` |
| `migration_matrix_counts.csv` | 迁移频数矩阵 | 21×21，索引 `from_score`，列 `to_score` |
| `migration_matrix_probabilities.csv` | 迁移概率矩阵 | 同上，行归一化 |
| `migration_summary.json` | 总体迁移指标 | `p_stable` / `p_upgrade` / `p_downgrade` / `mean_delta` |
| `probability_by_year.csv` | 年度迁移概率 | `year` / `p_upgrade` / `p_downgrade` / `p_stable` / `p_*_given_change` |
| `probability_by_agency.csv` | 机构间差异 | `agency` / 同上 |
| `probability_by_rating_bucket.csv` | 按评级档 | `bucket_from` / 同上 |
| `rating_cycle_stats.csv` | 评级周期 | `mean_spell_years` / `max_spell_years` / `mean_abs_change` |
| `agency_dispersion.csv` | 机构分歧 | `n_agencies` / `range_score` / `is_split` |
| `regression_<target>.csv` | 回归系数表 | `coef` / `std_err` / `t` / `p_value` / `signif` / `ci_lower` / `ci_upper` |
| `regression_comparison.csv` | 多模型系数对比 | 列为模型名 |
| `cv_model_comparison.csv` | 交叉验证对比 | `model` / `metric` / `mean` / `std` |
| `oof_predictions_<model>.csv` | 样本外预测 | `prior_score` / `y_true` / `y_pred` / `p_downgrade` / `p_stable` / `p_upgrade` / 各档概率 |
| `rf_feature_importance.csv` | 特征重要性 | `feature` / `importance` / `importance_pct` |
| `shap_summary.csv` | SHAP 全局重要性 | `feature` / `mean_abs_shap` / `mean_shap` / `direction` |
| `ordinal_logit_coefficients.csv` | 有序 Logit 系数 | `coef` / `std_err` / `z` / `p_value` |
| `rating_events.csv` | 评级事件表 | `country_iso3` / `event_date` / `event_type` / `rating_change` |
| `run_manifest.json` | **运行清单** | 数据指纹、各步骤结果、全部警告 |

### 各指标定义

| 指标 | 定义 | 方向 |
| --- | --- | --- |
| `accuracy` | 精确命中的比例 | ↑ 越好 |
| `balanced_accuracy` | 各类召回率的算术平均 | ↑ 越好 |
| `f1_macro` | 宏平均 F1 | ↑ 越好 |
| `mae_ordinal` | 预测档位与真实档位的平均绝对差 | ↓ 越好 |
| `adjacent_acc_1` | 预测误差 ≤ 1 档的比例 | ↑ 越好 |
| `adjacent_acc_2` | 预测误差 ≤ 2 档的比例 | ↑ 越好 |
| `quadratic_kappa` | 二次加权 kappa（考虑档位距离的一致性） | ↑ 越好 |
| `roc_auc_ovr_macro` | 多分类 one-vs-rest 宏平均 AUC | ↑ 越好 |
| `log_loss` | 概率预测的对数损失 | ↓ 越好 |
| `p_downgrade` | 预测分布中**低于**预测前评级的概率之和 | — |
| `p_stable` | 预测分布中**等于**预测前评级的概率 | — |
| `p_upgrade` | 预测分布中**高于**预测前评级的概率之和 | — |

---

## 8. 缺失值约定

### 三类缺失，三种处理

| 类型 | 例子 | 处理 | 理由 |
| --- | --- | --- | --- |
| **未知**（可插值） | GDP 增长率某年缺失 | 组内线性插值，`limit=2` | 数值客观存在，只是未采集 |
| **结构性缺失**（不插值） | WGI 在 2001 年前缺失 | 保留 `NaN`，不放宽 `limit_direction` | 指标当时尚未统计，「不存在」而非「未知」 |
| **评级缺失**（前向填充） | 某年无评级行动 | `score_in_effect` 前向填充；`rating_score` 保留 `NaN` | 评级具有粘性；但「迁移」不能用填充值计算 |

### 缺失值的可追溯性

任何填补操作都会：

1. 新增 `<column>_imputed` 布尔列标记被填补的位置
2. 在 `missingness_fill_report.csv` 中记录填补前后的缺失数与填补数量
3. 在 `run_manifest.json` 中汇总

### 符号约定

| 符号 | 含义 |
| --- | --- |
| `NaN` / 空 | 缺失 |
| `n/a`、`NR`、`WR`、`WD` | 无评级（解析后转为缺失） |
| `0` | 真实的零值（例如评级变化为 0），**不是**缺失 |

> ⚠️ 本项目**绝不用 0 表示缺失**。评级的「0 分」不存在，
> 评级变化为 0 有明确含义（评级未变）。

---

## 附录：数据来源与许可

| 数据源 | 许可 | 再分发限制 |
| --- | --- | --- |
| World Bank WDI / WGI | CC BY 4.0 | 需署名 |
| IMF WEO | IMF 版权，研究用途 | 需署名 IMF |
| BIS 统计 | BIS 版权 | 需确认条款 |
| FRED | 各序列许可不同 | 需逐序列确认 |
| S&P / Moody's / Fitch 评级 | **商业授权产品** | ❌ 禁止再分发 |

本项目**仅打包**自己生成的合成示例数据。
