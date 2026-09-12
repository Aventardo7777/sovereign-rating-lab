"""面板构建与校验。

评级数据天然是「事件式」的（每一次评级行动一行），而计量分析需要
「国家-年份」的平衡/非平衡面板。本模块负责两者之间的转换，
并显式区分两个容易混淆的概念：

* ``rating`` —— 该年**实际发生**的评级（仅在有评级行动的年份非空）
* ``rating_in_effect`` —— 该年**生效**的评级（在无行动年份沿用上一次评级）
* ``rating_change`` —— 生效评级相对上一年的变化（即评级迁移的度量）
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.rating_scale import rating_action_type
from src.utils.logging_utils import get_logger

__all__ = [
    "build_country_year_panel",
    "collapse_to_year_end",
    "standardize_columns",
    "validate_panel",
]

logger = get_logger(__name__)

_COLUMN_ALIASES: dict[str, str] = {
    "iso3": "country_iso3",
    "iso": "country_iso3",
    "country_code": "country_iso3",
    "countrycode": "country_iso3",
    "ccode": "country_iso3",
    "country": "country_name",
    "economy": "country_name",
    "国别": "country_name",
    "年份": "year",
    "机构": "agency",
    "评级": "rating",
    "展望": "outlook",
    "日期": "action_date",
    "行动": "action",
    "event_date": "action_date",
    "date": "action_date",
    "rating_date": "action_date",
    "rating_agency": "agency",
    "credit_rating": "rating",
    "outlook_watch": "outlook",
    "watch": "outlook",
    "action_type": "action",
    "rating_action": "action",
    "rating_score": "rating_score",
    "score": "rating_score",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名：去空白、转小写、下划线化，并映射常见别名。"""
    out = df.copy()
    renamed: dict[str, str] = {}
    for column in out.columns:
        key = str(column).strip()
        normalized = key.lower().replace(" ", "_").replace("-", "_")
        normalized = normalized.replace("__", "_").strip("_")
        renamed[column] = _COLUMN_ALIASES.get(normalized, _COLUMN_ALIASES.get(key, normalized))
    out = out.rename(columns=renamed)
    if out.columns.duplicated().any():
        duplicated = out.columns[out.columns.duplicated()].tolist()
        logger.warning("列名标准化后出现重复列 %s，保留首个", sorted(set(duplicated)))
        out = out.loc[:, ~out.columns.duplicated()]
    return out


def collapse_to_year_end(
    ratings: pd.DataFrame,
    *,
    country_col: str = "country_iso3",
    year_col: str = "year",
    agency_col: str = "agency",
    date_col: str = "action_date",
    keep_last: bool = True,
) -> pd.DataFrame:
    """把「国家-机构-年份内的多次评级行动」压缩为一条年度观测。

    默认保留该年**最后一次**评级行动（年末状态），这是构造年度迁移矩阵的
    标准做法；``keep_last=False`` 时保留第一次。
    """
    out = standardize_columns(ratings)
    for required in (country_col, year_col, agency_col):
        if required not in out.columns:
            raise KeyError(f"缺少必需列: {required}")

    out[year_col] = pd.to_numeric(out[year_col], errors="coerce").astype("Int64")
    out = out.dropna(subset=[country_col, year_col, agency_col])

    sort_cols = [country_col, agency_col, year_col]
    if date_col in out.columns:
        out["_action_dt"] = pd.to_datetime(out[date_col], errors="coerce")
        sort_cols.append("_action_dt")
    else:
        out["_action_dt"] = pd.NaT

    out = out.sort_values(sort_cols, na_position="first")
    grouped = out.groupby([country_col, agency_col, year_col], dropna=False, sort=True)
    collapsed = grouped.tail(1).copy() if keep_last else grouped.head(1).copy()
    collapsed = collapsed.drop(columns=["_action_dt"])
    collapsed = collapsed.sort_values([country_col, agency_col, year_col], ignore_index=True)
    return collapsed


def build_country_year_panel(
    ratings: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    *,
    country_col: str = "country_iso3",
    year_col: str = "year",
    agency_col: str = "agency",
    score_col: str = "rating_score",
    start_year: int | None = None,
    end_year: int | None = None,
    carry_forward: bool = True,
    carry_limit: int | None = None,
) -> pd.DataFrame:
    """构建国家-年份-机构面板。

    Parameters
    ----------
    ratings:
        评级数据（事件式或年度式均可，内部会先做 :func:`collapse_to_year_end`）。
    macro:
        宏观面板，须含 ``country_iso3`` 与 ``year``；按这两列左连接。
    start_year, end_year:
        面板年份范围。缺省时取评级数据中的最小/最大年份。
    carry_forward:
        是否生成 ``rating_in_effect``（在无行动年份沿用上一次评级）。
        这是计算「评级迁移」的前提，默认开启。
    carry_limit:
        前向填充的最大年数。``None`` 表示不限制（评级一直沿用至下次行动）。

    Returns
    -------
    pandas.DataFrame
        含以下关键列：``country_iso3``、``agency``、``year``、``rating``、
        ``rating_score``、``outlook``、``has_action``、``rating_in_effect``、
        ``score_in_effect``、``rating_change``、``action_type``、``is_downgrade``、
        ``is_upgrade``、``is_stable``。
    """
    ratings = standardize_columns(ratings)
    annual = collapse_to_year_end(
        ratings, country_col=country_col, year_col=year_col, agency_col=agency_col
    )

    years = pd.to_numeric(annual[year_col], errors="coerce").dropna()
    low = int(start_year if start_year is not None else years.min())
    high = int(end_year if end_year is not None else years.max())
    if low > high:
        raise ValueError(f"起始年份 {low} 大于结束年份 {high}")

    entities = annual[[country_col, agency_col]].drop_duplicates()
    if "country_name" in annual.columns:
        names = annual[[country_col, "country_name"]].drop_duplicates(subset=[country_col])
    else:
        names = None

    grid = entities.merge(pd.DataFrame({year_col: range(low, high + 1)}), how="cross")
    panel = grid.merge(annual, on=[country_col, agency_col, year_col], how="left")

    if names is not None:
        panel = panel.merge(names, on=country_col, how="left")

    panel["has_action"] = panel[score_col].notna()
    panel = panel.sort_values([country_col, agency_col, year_col], ignore_index=True)

    if carry_forward:
        grouped = panel.groupby([country_col, agency_col], dropna=False)
        if "rating" in panel.columns:
            panel["rating_in_effect"] = grouped["rating"].ffill(limit=carry_limit)
        panel["score_in_effect"] = grouped[score_col].ffill(limit=carry_limit)
        previous = panel.groupby([country_col, agency_col], dropna=False)["score_in_effect"].shift(
            1
        )
        panel["rating_change"] = panel["score_in_effect"] - previous
        panel["action_type"] = panel["rating_change"].map(rating_action_type)
        panel["is_downgrade"] = (panel["rating_change"] < 0).astype(int)
        panel["is_upgrade"] = (panel["rating_change"] > 0).astype(int)
        panel["is_stable"] = (panel["rating_change"] == 0).astype(int)
        panel.loc[panel["rating_change"].isna(), ["is_downgrade", "is_upgrade", "is_stable"]] = 0
        panel["rating_change"] = panel["rating_change"].where(previous.notna())
        panel.loc[panel["rating_change"].isna(), "action_type"] = None
    else:
        panel["score_in_effect"] = panel[score_col]
        panel["rating_change"] = np.nan
        panel["is_downgrade"] = 0
        panel["is_upgrade"] = 0
        panel["is_stable"] = 0

    if macro is not None and not macro.empty:
        macro_std = standardize_columns(macro)
        if country_col not in macro_std.columns or year_col not in macro_std.columns:
            raise KeyError("macro 面板必须包含 country_iso3 与 year 列")
        macro_std[year_col] = pd.to_numeric(macro_std[year_col], errors="coerce").astype("Int64")
        overlap = [
            c for c in macro_std.columns if c in panel.columns and c not in {country_col, year_col}
        ]
        if overlap:
            logger.debug("宏观面板与评级面板共有列 %s，将保留评级面板版本", overlap)
            macro_std = macro_std.drop(columns=overlap)
        panel = panel.merge(macro_std, on=[country_col, year_col], how="left")

    panel = panel.drop(columns=[c for c in ("rating_map_reason",) if c in panel.columns])
    return panel


def validate_panel(
    panel: pd.DataFrame,
    *,
    country_col: str = "country_iso3",
    year_col: str = "year",
    agency_col: str = "agency",
    score_col: str = "rating_score",
) -> pd.DataFrame:
    """对面板运行一组完整性检查。

    Returns
    -------
    pandas.DataFrame
        每行一项检查，列 ``check`` / ``n_violations`` / ``passed`` / ``detail``。
    """
    checks: list[dict[str, Any]] = []

    def add(name: str, n: int, detail: str = "") -> None:
        checks.append(
            {"check": name, "n_violations": int(n), "passed": int(n) == 0, "detail": detail}
        )

    key_cols = [c for c in (country_col, agency_col, year_col) if c in panel.columns]
    if len(key_cols) == len([country_col, agency_col, year_col]):
        n_dup = int(panel.duplicated(subset=key_cols).sum())
        add("主键唯一 (country, agency, year)", n_dup, "重复的主键组合数")

    if score_col in panel.columns:
        scores = pd.to_numeric(panel[score_col], errors="coerce")
        out_of_range = int(((scores < 1) | (scores > 21)).sum())
        add("评级分值落在 [1, 21]", out_of_range, "越界观测数")

    if "score_in_effect" in panel.columns:
        eff = pd.to_numeric(panel["score_in_effect"], errors="coerce")
        add("生效分值落在 [1, 21]", int(((eff < 1) | (eff > 21)).sum()), "越界观测数")

    if year_col in panel.columns:
        years = pd.to_numeric(panel[year_col], errors="coerce")
        add("年份无缺失", int(years.isna().sum()), "缺失年份的观测数")
        add(
            "年份在合理区间 [1900, 2100]",
            int(((years < 1900) | (years > 2100)).sum()),
            "越界年份数",
        )

    if {"rating_change", "score_in_effect"}.issubset(panel.columns):
        change = pd.to_numeric(panel["rating_change"], errors="coerce")
        add(
            "评级变化绝对值不超过 20",
            int((change.abs() > 20).sum()),
            "可能由数据错位造成",
        )

    if "agency" in panel.columns:
        add(
            "机构取值非空",
            int(panel["agency"].isna().sum()),
            "机构缺失的观测数",
        )

    return pd.DataFrame(checks, columns=["check", "n_violations", "passed", "detail"])


def summarize_panel(panel: pd.DataFrame, group_col: str = "agency") -> pd.DataFrame:
    """按机构（或任意分组）汇总面板覆盖情况。"""
    if group_col not in panel.columns:
        raise KeyError(f"缺少分组列: {group_col}")
    grouped = panel.groupby(group_col, dropna=False)
    summary = pd.DataFrame(
        {
            "n_obs": grouped.size(),
            "n_countries": grouped["country_iso3"].nunique(),
            "year_min": grouped["year"].min(),
            "year_max": grouped["year"].max(),
            "n_actions": grouped["has_action"].sum(),
            "mean_score_in_effect": grouped["score_in_effect"].mean().round(3),
        }
    )
    return summary.reset_index()
