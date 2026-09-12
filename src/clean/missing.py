"""缺失值诊断与处理。

面板数据中缺失值有三种性质截然不同的来源，处理策略必须区分：

1. **评级缺失**：某年某机构未发布评级。对「评级水平」而言应做**前向填充**
   （评级具有粘性，未更新时沿用上一次有效评级），并保留 ``is_stale_rating`` 标记；
   对「评级迁移」而言则**不能填充**，否则会人为制造 0 迁移观测。
2. **宏观指标缺失**：多为统计口径年度间断。可在国家内做线性插值。
3. **结构性缺失**：2015 年后才有数据的指标（如某些治理指标）在早期年份缺失，
   属于「不存在」而非「未知」，应标注后剔除，避免插值造数。

本模块提供诊断报告与三类可复现的处理函数，全部返回结果并保留处理痕迹
（``*_imputed`` 标记列 + 处理报告），以便审稿人核查。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

__all__ = [
    "handle_missing",
    "impute_panel",
    "interpolate_within_group",
    "missingness_report",
    "stale_rating_flags",
]

logger = get_logger(__name__)


def missingness_report(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """逐列缺失诊断表。

    Returns
    -------
    pandas.DataFrame
        列为 ``column`` / ``dtype`` / ``n_obs`` / ``n_missing`` / ``pct_missing``
        / ``n_unique`` / ``n_groups_with_missing``（当提供 ``group_cols`` 时）。
        按缺失比例降序排列。
    """
    rows: list[dict[str, Any]] = []
    n_rows = len(df)
    for column in df.columns:
        series = df[column]
        n_missing = int(series.isna().sum())
        row: dict[str, Any] = {
            "column": column,
            "dtype": str(series.dtype),
            "n_obs": int(n_rows),
            "n_missing": n_missing,
            "pct_missing": round(100.0 * n_missing / n_rows, 4) if n_rows else np.nan,
            "n_unique": int(series.nunique(dropna=True)),
        }
        if group_cols:
            present = [c for c in group_cols if c in df.columns]
            if present:
                grouped = series.isna().groupby([df[c] for c in present], dropna=False)
                row["n_groups_with_missing"] = int((grouped.sum() > 0).sum())
                row["n_groups_total"] = int(grouped.ngroups)
        rows.append(row)
    report = pd.DataFrame(rows)
    if report.empty:
        return report
    return report.sort_values(["pct_missing", "column"], ascending=[False, True], ignore_index=True)


def interpolate_within_group(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    *,
    group_cols: Sequence[str] = ("country_iso3",),
    time_col: str = "year",
    method: str = "linear",
    limit: int | None = 2,
    limit_direction: str = "forward",
    add_flags: bool = True,
) -> pd.DataFrame:
    """在实体内部做时间插值。

    * 插值**不会跨实体**发生——这是面板数据最基本的纪律。
    * ``limit`` 限制连续插值的最大长度，默认 2：只填补 1-2 年的短缺口，
      避免用一段稀疏观测「补」出冗长的人造序列。
    * ``limit_direction`` 默认为 ``"forward"``：**不填补序列开头的缺失**。
      早期年份缺失通常属于「该指标尚未开始统计」的结构性缺失，
      用后一年份回填等于凭空造数，会污染早期样本。
      如需回填请显式传入 ``"both"`` 并在论文中说明。
    * ``add_flags=True`` 时新增 ``<col>_imputed`` 布尔列标记被填补的位置。
    """
    if time_col not in df.columns:
        raise KeyError(f"缺少时间列: {time_col}")
    present_groups = [c for c in group_cols if c in df.columns]
    out = df.sort_values([*present_groups, time_col]).copy()
    for column in value_cols:
        if column not in out.columns:
            raise KeyError(f"缺少列: {column}")

    for column in value_cols:
        numeric = pd.to_numeric(out[column], errors="coerce")
        filled = numeric.groupby(
            [out[c] for c in present_groups] if present_groups else np.zeros(len(out)),
            dropna=False,
        ).transform(
            lambda s: s.interpolate(
                method=method,
                limit=limit,
                limit_direction=limit_direction,
            )
        )
        if add_flags:
            out[f"{column}_imputed"] = numeric.isna() & filled.notna()
        # ``transform`` 在空组等边界情形可能返回 NaN，需按索引对齐
        out[column] = pd.Series(filled, index=out.index).where(
            pd.Series(filled, index=out.index).notna(), numeric
        )
    return out


def impute_panel(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    *,
    group_cols: Sequence[str] = ("country_iso3",),
    time_col: str = "year",
    method: str = "interpolate",
    limit: int | None = 2,
    add_flags: bool = True,
) -> pd.DataFrame:
    """面板缺失值填补的统一入口。

    Parameters
    ----------
    method:
        * ``"interpolate"`` —— 组内线性插值（默认，仅填补内部缺口）
        * ``"ffill"`` —— 组内前向填充（适用于评级等具有粘性的变量）
        * ``"group_median"`` —— 组内中位数
        * ``"median"`` —— 全样本中位数
        * ``"mean"`` —— 全样本均值
        * ``"drop"`` —— 不填补，直接丢弃含缺失的行
    """
    valid = {"interpolate", "ffill", "group_median", "median", "mean", "drop"}
    if method not in valid:
        raise ValueError(f"不支持的 method={method!r}，可选: {sorted(valid)}")

    present_groups = [c for c in group_cols if c in df.columns]
    if method == "drop":
        return df.dropna(subset=[c for c in value_cols if c in df.columns]).copy()

    if method == "interpolate":
        return interpolate_within_group(
            df,
            value_cols,
            group_cols=group_cols,
            time_col=time_col,
            limit=limit,
            add_flags=add_flags,
        )

    out = df.sort_values([*present_groups, time_col]).copy() if present_groups else df.copy()
    for column in value_cols:
        if column not in out.columns:
            raise KeyError(f"缺少列: {column}")
        numeric = pd.to_numeric(out[column], errors="coerce")
        if method == "ffill":
            filled = numeric.groupby(
                [out[c] for c in present_groups] if present_groups else np.zeros(len(out)),
                dropna=False,
            ).transform(lambda s: s.ffill(limit=limit))
        else:
            if method in {"group_median"} and present_groups:
                filled = numeric.groupby([out[c] for c in present_groups]).transform("median")
            elif method == "group_median" or method == "median":
                filled = numeric.fillna(numeric.median())
            else:  # mean
                filled = numeric.fillna(numeric.mean())
        if add_flags:
            out[f"{column}_imputed"] = numeric.isna() & filled.notna()
        out[column] = filled.combine_first(numeric)
    return out


def stale_rating_flags(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("country_iso3", "agency"),
    time_col: str = "year",
    rating_col: str = "rating",
    out_col: str = "is_stale_rating",
) -> pd.DataFrame:
    """标记「沿用上一年评级」的观测（评级未发生变化的年份）。

    研究评级迁移时，重复观测必须保留（它们构成「稳定」状态），
    但在描述性统计中应能区分「新评级行动」与「沿用」。
    """
    present_groups = [c for c in group_cols if c in df.columns]
    out = df.sort_values([*present_groups, time_col]).copy()
    grouped = out.groupby(present_groups, dropna=False) if present_groups else None
    previous = grouped[rating_col].shift(1) if grouped is not None else out[rating_col].shift(1)
    out[out_col] = (previous.notna()) & (out[rating_col].astype(str) == previous.astype(str))
    return out


def handle_missing(
    df: pd.DataFrame,
    value_cols: Iterable[str] | None = None,
    *,
    strategy: str = "interpolate",
    group_cols: Sequence[str] = ("country_iso3",),
    time_col: str = "year",
    limit: int | None = 2,
    subset_for_drop: Sequence[str] | None = None,
    add_flags: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """一站式缺失值处理，返回 ``(处理后的数据, 处理前后对比报告)``。

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"country_iso3": ["AAA"] * 3, "year": [2000, 2001, 2002],
    ...                       "gdp_growth": [1.0, None, 3.0]})
    >>> cleaned, report = handle_missing(frame, ["gdp_growth"], strategy="interpolate")
    >>> float(cleaned.loc[1, "gdp_growth"])
    2.0
    """
    columns = list(value_cols) if value_cols is not None else list(df.columns)
    before = missingness_report(df, group_cols=group_cols)
    before = before[before["column"].isin(columns)].copy()

    if strategy == "drop":
        cleaned = impute_panel(
            df,
            columns,
            group_cols=group_cols,
            time_col=time_col,
            method="drop",
        )
        if subset_for_drop:
            cleaned = cleaned.dropna(subset=[c for c in subset_for_drop if c in cleaned.columns])
    else:
        cleaned = impute_panel(
            df,
            columns,
            group_cols=group_cols,
            time_col=time_col,
            method=strategy,
            limit=limit,
            add_flags=add_flags,
        )

    after = missingness_report(cleaned, group_cols=group_cols)
    after = after[after["column"].isin(columns)].copy()

    report = before.rename(
        columns={"n_missing": "n_missing_before", "pct_missing": "pct_missing_before"}
    )[["column", "n_missing_before", "pct_missing_before"]].merge(
        after.rename(columns={"n_missing": "n_missing_after", "pct_missing": "pct_missing_after"})[
            ["column", "n_missing_after", "pct_missing_after"]
        ],
        on="column",
        how="outer",
    )
    report["n_filled"] = report["n_missing_before"] - report["n_missing_after"]
    report["strategy"] = strategy
    logger.info(
        "缺失值处理完成：strategy=%s，填补单元格 %s 个", strategy, int(report["n_filled"].sum())
    )
    return cleaned, report
