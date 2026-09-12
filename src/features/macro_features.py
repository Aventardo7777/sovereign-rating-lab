"""宏观特征工程：滞后、差分、标准化、缩尾与特征矩阵组装。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_FEATURES",
    "FEATURE_GROUPS",
    "add_changes",
    "add_derived_features",
    "add_lags",
    "build_feature_matrix",
    "describe_features",
    "winsorize",
    "zscore_within",
]

#: 宏观解释变量分组（同时用于回归与机器学习）。
FEATURE_GROUPS: dict[str, list[str]] = {
    "增长": ["gdp_growth", "gdp_per_capita_log"],
    "物价": ["inflation"],
    "财政": ["gov_debt_gdp"],
    "外部": [
        "external_debt_gni",
        "reserves_months_imports",
        "current_account_gdp",
        "exports_gdp",
    ],
    "金融": ["fx_depreciation", "real_interest_rate", "spread_bps"],
    "治理": [
        "voice_accountability",
        "political_stability",
        "government_effectiveness",
        "regulatory_quality",
        "rule_of_law",
        "control_corruption",
    ],
}

#: 由原始指标派生而来的特征，:func:`build_feature_matrix` 会在缺失时自动构造。
DERIVED_FEATURES: tuple[str, ...] = ("gdp_per_capita_log", "fx_depreciation")


#: 默认进入模型的解释变量（在 :data:`FEATURE_GROUPS` 基础上展开并去重）。
DEFAULT_FEATURES: list[str] = [name for names in FEATURE_GROUPS.values() for name in names]


def add_derived_features(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("country_iso3",),
    time_col: str = "year",
) -> pd.DataFrame:
    """构造派生特征（已存在则保留原值，不覆盖）。

    * ``gdp_per_capita_log`` = ``ln(人均 GDP)``——人均收入取对数后与评级高度相关，
      且能缓解右偏。
    * ``fx_depreciation`` = 官方汇率的百分比变化（正值为本币贬值）。
    """
    out = df.copy()
    present_groups = [c for c in group_cols if c in out.columns]

    if "gdp_per_capita" in out.columns and "gdp_per_capita_log" not in out.columns:
        level = pd.to_numeric(out["gdp_per_capita"], errors="coerce")
        out["gdp_per_capita_log"] = np.log(level.where(level > 0))

    if "fx_official_rate" in out.columns and "fx_depreciation" not in out.columns:
        numeric = pd.to_numeric(out["fx_official_rate"], errors="coerce")
        if present_groups:
            out["fx_depreciation"] = (
                numeric.groupby([out[c] for c in present_groups], dropna=False).pct_change() * 100.0
            )
        else:
            out["fx_depreciation"] = numeric.pct_change() * 100.0

    return out


def add_lags(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    group_cols: Sequence[str] = ("country_iso3",),
    time_col: str = "year",
    lags: Iterable[int] = (1, 2),
    drop_original: bool = False,
) -> pd.DataFrame:
    """按实体分组添加滞后项。

    评级机构通常使用**上一期**宏观数据做决策，因此解释变量普遍需要滞后 1-2 期。
    """
    out = df.sort_values([*group_cols, time_col]).copy()
    grouped = out.groupby(list(group_cols), dropna=False)
    for column in columns:
        if column not in out.columns:
            raise KeyError(f"缺少列: {column}")
        for lag in lags:
            out[f"{column}_lag{lag}"] = grouped[column].shift(lag)
    if drop_original:
        out = out.drop(columns=[c for c in columns if c in out.columns])
    return out


def add_changes(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    group_cols: Sequence[str] = ("country_iso3",),
    time_col: str = "year",
    periods: int = 1,
    suffix: str = "_chg",
) -> pd.DataFrame:
    """按实体分组添加一阶差分（变化率）。"""
    out = df.sort_values([*group_cols, time_col]).copy()
    grouped = out.groupby(list(group_cols), dropna=False)
    for column in columns:
        if column not in out.columns:
            raise KeyError(f"缺少列: {column}")
        out[f"{column}{suffix}{periods}"] = grouped[column].diff(periods)
    return out


def zscore_within(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """标准化。

    ``group_cols`` 为 ``None`` 时做全样本 z-score；否则在每个组内标准化
    （例如按年份标准化以吸收全球共同冲击）。
    """
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            raise KeyError(f"缺少列: {column}")
        series = pd.to_numeric(out[column], errors="coerce")
        if group_cols:
            grouped = series.groupby([out[c] for c in group_cols], dropna=False)
            mean = grouped.transform("mean")
            std = grouped.transform("std")
        else:
            mean = series.mean()
            std = series.std()
        std = std.replace(0, np.nan)
        out[f"{column}_z"] = (series - mean) / std
    return out


def winsorize(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """按分位数缩尾，抑制极端值对线性模型的杠杆影响。"""
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            raise KeyError(f"缺少列: {column}")
        series = pd.to_numeric(out[column], errors="coerce")
        lo, hi = series.quantile(lower), series.quantile(upper)
        out[column] = series.clip(lo, hi)
    return out


def build_feature_matrix(
    panel: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    *,
    target: str = "rating_score",
    entity_col: str = "country_iso3",
    time_col: str = "year",
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """组装 ``(X, y, meta)`` 三元组，供建模与交叉验证使用。

    ``meta`` 保留实体、时间等标识列，方便时间序列切分与结果解读。

    Returns
    -------
    X : pandas.DataFrame
        设计矩阵（数值型，缺失值按列中位数填补并新增 ``<col>_was_missing`` 标记）。
    y : pandas.Series
        目标变量。
    meta : pandas.DataFrame
        标识列（实体、时间、机构、评级符号等）。
    """
    features = list(feature_cols) if feature_cols is not None else list(DEFAULT_FEATURES)
    if set(features) & set(DERIVED_FEATURES):
        panel = add_derived_features(panel, group_cols=(entity_col,), time_col=time_col)
    missing = [c for c in features if c not in panel.columns]
    if missing:
        raise KeyError(
            f"以下特征列不存在于面板中: {missing}。"
            f"可用 add_derived_features() 构造派生特征，或从 FEATURE_GROUPS 中挑选可用变量。"
        )
    if target not in panel.columns:
        raise KeyError(f"目标列不存在: {target}")

    meta_cols = [
        c
        for c in (entity_col, time_col, "agency", "rating", "outlook", "rating_bucket")
        if c in panel.columns
    ]
    meta = panel[meta_cols].copy()

    X = panel[features].apply(pd.to_numeric, errors="coerce").copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    missing_flags = X.isna()
    if missing_flags.to_numpy().any():
        medians = X.median(numeric_only=True)
        X = X.fillna(medians)
        X = X.fillna(0.0)
        flagged = missing_flags.columns[missing_flags.any()].tolist()
        for column in flagged:
            X[f"{column}_was_missing"] = missing_flags[column].astype(int)

    y = pd.to_numeric(panel[target], errors="coerce")

    if dropna:
        keep = y.notna() & X.notna().all(axis=1)
        X, y, meta = X.loc[keep], y.loc[keep], meta.loc[keep]

    X = X.astype("float64")
    y = y.astype("float64")
    return X, y, meta


def describe_features(X: pd.DataFrame) -> pd.DataFrame:
    """生成特征描述统计表（用于附录）。"""
    stats = X.describe().T
    stats["missing"] = X.isna().sum()
    stats["missing_pct"] = (X.isna().mean() * 100).round(3)
    return stats
