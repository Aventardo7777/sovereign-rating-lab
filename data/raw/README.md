# `data/raw/` — 原始数据与导入模板

本目录用于存放**你自己导入的原始数据**。目录内的文件**不会被提交到 Git**
（见 `.gitignore`），避免误传版权数据。

## ⚠️ 评级数据版权声明

S&P Global Ratings、Moody's、Fitch Ratings 的主权评级历史数据库均为**商业授权产品**。
本项目：

* **不提供**、**不打包**、**不伪造**任何完整的真实评级历史数据；
* 只提供**导入模板**与**解析脚本**；
* 使用任何数据前，**你需自行确认拥有合法使用权限**，并对使用结果负责。

合法获取评级历史的常见途径：

1. 机构官方订阅产品（S&P Capital IQ、Moody's Analytics、Fitch Connect）
2. 所在高校/机构订阅的数据库（多数高校图书馆有 Bloomberg、WRDS、Eikon 终端）
3. 公开可得的评级公告（需注意服务条款是否允许批量抓取与再分发）
4. 学术数据库（部分论文附录提供整理后的历史评级，需遵守其许可）

---

## 1. 导入模板

[`ratings_template.csv`](ratings_template.csv) 是本项目认可的**最小列结构**：

```csv
country_iso3,country_name,year,agency,rating,outlook,action,action_date
```

### 列说明

| 列名 | 必需 | 类型 | 说明 |
| --- | --- | --- | --- |
| `country_iso3` | ✅ | 字符串 | ISO 3166-1 alpha-3 国家代码，如 `CHN`、`BRA` |
| `country_name` | ⬜ | 字符串 | 国家名称（仅用于展示） |
| `year` | ✅ | 整数 | 评级行动发生的年份 |
| `agency` | ✅ | 字符串 | `S&P` / `Moody's` / `Fitch`，也接受 `SP`、`Moodys`、`Fitch Ratings` 等别名 |
| `rating` | ✅ | 字符串 | 评级符号，如 `AA+`、`Baa1`、`BBB-`；支持 `AA+u`、`(P)AAA`、`BBB+*` 等带修饰符写法 |
| `outlook` | ⬜ | 字符串 | 展望，如 `Stable`、`Positive`、`Negative`、`Developing`；也接受 `CW-Negative`、`Positive Watch` 等 |
| `action` | ⬜ | 字符串 | 行动类型，如 `upgrade`、`downgrade`、`assign`、`affirm`（可由程序重新推导） |
| `action_date` | ⬜ | 日期 | 行动日期，格式 `YYYY-MM-DD`（推荐提供，事件研究必需） |

### 可选列

程序会自动计算 `rating_score`（1-21 分值），无需在导入文件中提供。
你提供的话也会被**重新计算并覆盖**，以确保刻度一致。

| 列名 | 说明 |
| --- | --- |
| `rating_score` | 程序自动生成，无需提供 |
| `source_url` | 数据来源 URL，用于溯源（推荐填写） |
| `provenance` | 溯源标记，例如 `bloomberg_export_2026` |

### 允许一行一条评级行动

评级数据天然是**事件式**的：只在评级发生变化时记录一行。例如：

```csv
country_iso3,country_name,year,agency,rating,outlook,action,action_date
BRA,Brazil,2015,S&P,BB,Stable,downgrade,2015-09-09
BRA,Brazil,2016,S&P,BB-,Negative,downgrade,2016-02-17
BRA,Brazil,2019,Moody's,Ba2,Stable,upgrade,2019-06-11
```

也允许同一年同一机构有多行（例如 3 月上调、11 月下调），
流水线会按 `action_date` 取**年末状态**作为该年的年度观测。

---

## 2. 使用方式

### 方式 A：放进本目录，改一行代码

```bash
cp /your/export/ratings.csv data/raw/ratings.csv
```

```python
from src.ingest.ratings import load_ratings_csv, validate_imported_ratings

ratings = load_ratings_csv("data/raw/ratings.csv")
print(validate_imported_ratings(ratings))
```

### 方式 B：直接传给流水线

```bash
python -m src.pipeline --ratings data/raw/ratings.csv --macro data/raw/macro.csv
```

### 方式 C：在 notebook 里用

```python
RATINGS_PATH = "data/raw/ratings.csv"
ratings = load_ratings_csv(RATINGS_PATH)
```

---

## 3. 图表字体与中文显示

本项目的图表默认使用 `Microsoft YaHei` → `SimHei` → `Noto Sans CJK SC` 作为
中文字体回退链。如果图表中的中文显示为方块：

```bash
# Linux
sudo apt-get install -y fonts-noto-cjk
# macOS：系统自带中文字体，通常无需处理
```

或在 Python 中显式指定：

```python
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
```

---

## 4. 抓取到的数据放在哪里？

合规抓取的原始响应缓存在 `data/raw/cache/`，抓取登记表为
`data/raw/cache/provenance.json`（记录 URL、抓取时间、HTTP 状态码、robots 判定结果）。

**这些目录同样不会被提交到 Git**：它们可能包含目标站点的原始内容，
再分发可能构成侵权。

如需在论文附录中说明数据来源，可运行：

```python
from src.ingest.ratings import write_provenance_manifest
write_provenance_manifest()   # 输出 data/raw/provenance_manifest.json
```
