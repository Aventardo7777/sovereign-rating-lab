# `data/sample/` — 合成演示数据

## ⚠️ 警告：这些不是真实数据

本目录下的 CSV 文件由 [`scripts/make_sample_data.py`](../../scripts/make_sample_data.py)
**程序生成**，用于演示与测试分析流程。

> **它们不代表任何真实国家、真实机构或真实评级的实际状况。**
> **不得**用于任何实证结论、政策分析、投资决策或报道引用。

每一行数据都带有溯源标记列：

```text
provenance = synthetic_demo_only
```

程序在载入示例数据时会检查该标记并打印警告。若你看到
`provenance` 列出现其他取值，说明数据已被替换——请核对来源。

---

## 文件说明

### `ratings_sample.csv`

事件式主权评级记录。列结构：

| 列名 | 说明 |
| --- | --- |
| `country_iso3` | ISO3 国家代码（真实代码，仅为方便与 WDI/WGI 等公开数据连接） |
| `country_name` | 国家名称 |
| `year` | 年份（2000-2023） |
| `agency` | `S&P` / `Moody's` / `Fitch` |
| `rating` | 评级符号（含刻意注入的 `u` 后缀、`C/D` 复合符号等「脏值」） |
| `outlook` | 展望（含 `CW-Negative`、`Positive Watch`、缺失、`n/a`） |
| `action` | `assign` / `upgrade` / `downgrade` |
| `action_date` | 行动日期（部分为缺失，模拟历史数据不完整） |
| `rating_score` | 1-21 统一分值 |
| `provenance` | 固定为 `synthetic_demo_only` |
| `data_note` | 数据声明文本 |

### `macro_sample.csv`

与国家-年份对齐的宏观面板。列包括：

`gdp_growth`、`inflation`、`gov_debt_gdp`、`external_debt_gni`、
`reserves_months_imports`、`current_account_gdp`、`exports_gdp`、`gdp_per_capita`、
`fx_official_rate`、`real_interest_rate`、`spread_bps`，
以及 WGI 六项治理指标（`voice_accountability`、`political_stability`、
`government_effectiveness`、`regulatory_quality`、`rule_of_law`、`control_corruption`）。

其中刻意注入了：

* **结构性缺失**：WGI 在 2001 年及以前随机缺失 35%（模拟治理指标早期覆盖不全）
* **随机间断**：少数宏观指标随机缺失约 3.5%

这些缺失用于测试 `src/clean/missing.py` 的处理逻辑。

---

## 合成数据的设计意图

生成器不是随机噪声，而是刻意模拟真实数据的若干特征，使演示流水线能跑出
**非平凡且有经济含义**的结果：

| 设计 | 对应真实特征 | 验证的代码路径 |
| --- | --- | --- |
| 评级行动仅在变化时记录（事件式） | 评级具有粘性，不逐年更新 | `collapse_to_year_end`、迁移矩阵 |
| 三家机构有系统性偏置（S&P 略高、Moody's 略低） | 机构评级标准存在差异 | `agency_dispersion`、机构差异分析 |
| 危机年份对国家施加持续性冲击 | 危机后评级需多年修复 | 迁移的时间聚类、事件研究 |
| 宏观指标围绕评级水平生成 | 债务/GDP、利差与评级强相关 | 回归能识别出方向一致的系数 |
| 展望反映评级与基本面的偏离 | 展望是评级变动的前瞻信号 | 展望标准化、有序模型 |
| 注入脏值与缺失 | 真实导出数据从不干净 | 清洗、映射、缺失值处理 |

有了这些结构，`python -m src.pipeline` 才能验证「流水线是通的」，
而不只是「没报错」。

---

## 重新生成

```bash
python scripts/make_sample_data.py
# 或
make sample
```

使用固定随机种子 `20240625`，同一版本代码生成的数据**逐字节一致**，保证可复现。

---

## 想要真实数据？

请看 [`data/raw/README.md`](../raw/README.md) 的版权说明与导入模板。
简要版：

1. 把自有评级数据按 `data/raw/ratings_template.csv` 的列结构整理好；
2. 运行 `python -m src.pipeline --ratings data/raw/your_ratings.csv`；
3. 所有分析与图表会自动改用你的数据，并在 `reports/tables/run_manifest.json`
   中记录数据指纹（便于日后核对是否用了同一份数据）。
