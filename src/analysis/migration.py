"""评级迁移分析。

核心概念
--------
* **迁移矩阵（migration / transition matrix）**：行 = t 期评级，列 = t+1 期评级，
  元素为观测频数或迁移概率。只统计**连续两年**都存在的观测，避免把「数据缺口」
  误当作「未迁移」。
* **上调 / 下调概率**：矩阵的非对角块；本模块同时给出「无条件概率」（占全部
  观测）与「条件概率」（占发生变化的观测），二者常被混用而导致结论差异。
* **评级周期（rating cycle）**：某个评级从进入（评级行动）到离开（下一次行动）
  的持续时间，以及评级行动之间的间隔。
* **机构差异**：同一国家-年份下不同机构的分值离散度（split rating）。

所有函数都接受长表（long format）面板，至少包含
``country_iso3``、``agency``、``year``、``rating_score`` 四列。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.features.rating_scale import MAX_SCORE, MIN_SCORE, rating_bucket

__all__ = [
    "agency_dispersion",
    "build_migration_matrix",
    "downgrade_probability",
    "migration_entropy",
    "migration_summary",
    "rating_cycle_stats",
    "rating_spell_stats",
    "transition_probability_matrix",
    "upgrade_downgrade_probabilities",
]

_ENTITY_COLS = ("country_iso3", "agency")


def _prepare_transitions(
    panel: pd.DataFrame,
    *,
    entity_cols: Sequence[str] = _ENTITY_COLS,
    time_col: str = "year",
    score_col: str = "rating_score",
    score_is_effective: bool = False,
) -> pd.DataFrame:
    """构造「t -> t+1」的迁移明细表。

    只有满足 ``t+1 == t + 1``（真正相邻的年份）且两期分值均非空的观测才保留。
    """
    present = [c for c in entity_cols if c in panel.columns]
    if not present:
        raise KeyError(f"面板中缺少实体识别列，至少需要 {list(_ENTITY_COLS)} 之一")
    if time_col not in panel.columns:
        raise KeyError(f"缺少时间列: {time_col}")

    use_col = score_col
    if score_is_effective and "score_in_effect" in panel.columns:
        use_col = "score_in_effect"
    if use_col not in panel.columns:
        raise KeyError(f"缺少评级分值列: {use_col}")

    frame = panel[[*present, time_col, use_col]].copy()
    frame[time_col] = pd.to_numeric(frame[time_col], errors="coerce")
    frame[use_col] = pd.to_numeric(frame[use_col], errors="coerce")
    frame = frame.dropna(subset=[*present, time_col, use_col])
    frame = frame.sort_values([*present, time_col])

    grouped = frame.groupby(present, dropna=False)
    frame["to_score"] = grouped[use_col].shift(-1)
    frame["to_year"] = grouped[time_col].shift(-1)
    transitions = frame[
        (frame["to_year"] == frame[time_col] + 1) & frame["to_score"].notna()
    ].copy()
    transitions = transitions.rename(columns={use_col: "from_score"})
    transitions["from_score"] = transitions["from_score"].astype(int)
    transitions["to_score"] = transitions["to_score"].astype(int)
    transitions["delta"] = transitions["to_score"] - transitions["from_score"]
    transitions["direction"] = np.select(
        [transitions["delta"] > 0, transitions["delta"] < 0],
        ["upgrade", "downgrade"],
        default="stable",
    )
    return transitions.reset_index(drop=True)


def build_migration_matrix(
    panel: pd.DataFrame,
    *,
    entity_cols: Sequence[str] = _ENTITY_COLS,
    time_col: str = "year",
    score_col: str = "rating_score",
    from_year: int | None = None,
    to_year: int | None = None,
    normalize: bool = False,
    score_is_effective: bool = False,
) -> pd.DataFrame:
    """构建年度评级迁移矩阵。

    Parameters
    ----------
    from_year, to_year:
        限定起始年份区间（闭区间）。例如 ``from_year=2008, to_year=2012`` 只统计
        2008-2012 年间发生的迁移。
    normalize:
        ``False`` 返回频数矩阵；``True`` 返回按行归一化的迁移概率矩阵。

    Returns
    -------
    pandas.DataFrame
        索引为 ``from_score``（1-21），列为 ``to_score``（1-21），
        行/列名分别为 ``from_score`` / ``to_score``。
    """
    transitions = _prepare_transitions(
        panel,
        entity_cols=entity_cols,
        time_col=time_col,
        score_col=score_col,
        score_is_effective=score_is_effective,
    )
    if from_year is not None:
        transitions = transitions[transitions[time_col] >= from_year]
    if to_year is not None:
        transitions = transitions[transitions[time_col] <= to_year]

    ticks = list(range(MIN_SCORE, MAX_SCORE + 1))
    if transitions.empty:
        matrix = pd.DataFrame(0, index=ticks, columns=ticks, dtype="float64")
    else:
        matrix = pd.crosstab(transitions["from_score"], transitions["to_score"])
        matrix = matrix.reindex(index=ticks, columns=ticks, fill_value=0).astype("float64")

    matrix.index.name = "from_score"
    matrix.columns.name = "to_score"

    if normalize:
        return transition_probability_matrix(matrix)
    return matrix


def transition_probability_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """按行归一化得到迁移概率矩阵。

    行为 0 的评级档（样本中未出现）保持为 0，而不是 NaN，以便热力图直接绘制。
    """
    values = matrix.to_numpy(dtype="float64")
    row_sums = values.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probabilities = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > 0)
    return pd.DataFrame(probabilities, index=matrix.index, columns=matrix.columns, dtype="float64")


def migration_summary(matrix: pd.DataFrame) -> dict[str, float]:
    """从频数矩阵提取总体迁移指标。"""
    values = matrix.to_numpy(dtype="float64")
    ticks = np.asarray(matrix.index, dtype=int)
    total = values.sum()
    if total == 0:
        return {
            "n_transitions": 0.0,
            "p_stable": np.nan,
            "p_upgrade": np.nan,
            "p_downgrade": np.nan,
            "mean_delta": np.nan,
            "diagonal_share": np.nan,
        }
    idx_from, idx_to = np.meshgrid(ticks, ticks, indexing="ij")
    delta = idx_to - idx_from
    p_up = values[delta > 0].sum() / total
    p_down = values[delta < 0].sum() / total
    mean_delta = float((values * delta).sum() / total)
    return {
        "n_transitions": float(total),
        "p_stable": float(values[delta == 0].sum() / total),
        "p_upgrade": float(p_up),
        "p_downgrade": float(p_down),
        "mean_delta": mean_delta,
        "diagonal_share": float(np.trace(values) / total),
    }


def upgrade_downgrade_probabilities(
    panel: pd.DataFrame,
    *,
    group_cols: Sequence[str] | None = None,
    entity_cols: Sequence[str] = _ENTITY_COLS,
    time_col: str = "year",
    score_col: str = "rating_score",
    score_is_effective: bool = False,
) -> pd.DataFrame:
    """按分组计算上调 / 下调 / 稳定概率。

    Parameters
    ----------
    group_cols:
        分组维度，可为 ``["year"]``、``["country_iso3"]``、``["agency"]``、
        ``["rating_bucket"]`` 或 ``["income_group"]``（若面板中存在该列）等。
        同时会统计「以评级档为条件」的迁移概率，便于观察评级粘性。

    Returns
    -------
    pandas.DataFrame
        列：分组列 + ``n_transitions`` / ``n_upgrade`` / ``n_downgrade`` /
        ``n_stable`` / ``p_upgrade`` / ``p_downgrade`` / ``p_stable`` /
        ``p_change`` / ``mean_delta`` / ``p_downgrade_given_change`` /
        ``p_upgrade_given_change``。
    """
    transitions = _prepare_transitions(
        panel,
        entity_cols=entity_cols,
        time_col=time_col,
        score_col=score_col,
        score_is_effective=score_is_effective,
    )
    if transitions.empty:
        columns = [
            *(group_cols or []),
            "n_transitions",
            "n_upgrade",
            "n_downgrade",
            "n_stable",
            "p_upgrade",
            "p_downgrade",
            "p_stable",
            "p_change",
            "mean_delta",
            "p_downgrade_given_change",
            "p_upgrade_given_change",
        ]
        return pd.DataFrame(columns=columns)

    frame = transitions.copy()
    frame["is_upgrade"] = (frame["delta"] > 0).astype(int)
    frame["is_downgrade"] = (frame["delta"] < 0).astype(int)
    frame["is_stable"] = (frame["delta"] == 0).astype(int)
    frame["bucket_from"] = frame["from_score"].map(rating_bucket)

    keys: list[str] = list(group_cols) if group_cols else []
    keys = [k for k in keys if k in frame.columns]
    # 无分组维度时退化为「单个伪分组」，保持下游循环逻辑一致
    grouped: Any = frame.groupby(keys, dropna=False) if keys else [(None, frame)]

    rows: list[dict[str, Any]] = []
    for key, chunk in grouped:
        n = len(chunk)
        n_up = int(chunk["is_upgrade"].sum())
        n_down = int(chunk["is_downgrade"].sum())
        n_stable = int(chunk["is_stable"].sum())
        n_change = n_up + n_down
        record: dict[str, Any] = {}
        if keys:
            key_tuple = key if isinstance(key, tuple) else (key,)
            record.update(dict(zip(keys, key_tuple, strict=False)))
        record.update(
            {
                "n_transitions": n,
                "n_upgrade": n_up,
                "n_downgrade": n_down,
                "n_stable": n_stable,
                "p_upgrade": n_up / n if n else np.nan,
                "p_downgrade": n_down / n if n else np.nan,
                "p_stable": n_stable / n if n else np.nan,
                "p_change": n_change / n if n else np.nan,
                "mean_delta": float(chunk["delta"].mean()) if n else np.nan,
                "p_downgrade_given_change": n_down / n_change if n_change else np.nan,
                "p_upgrade_given_change": n_up / n_change if n_change else np.nan,
            }
        )
        rows.append(record)

    result = pd.DataFrame(rows)
    if keys and not result.empty:
        result = result.sort_values(keys, ignore_index=True)
    return result


def downgrade_probability(
    panel: pd.DataFrame,
    *,
    horizon: int = 1,
    score_col: str = "rating_score",
    entity_cols: Sequence[str] = _ENTITY_COLS,
    time_col: str = "year",
    by_rating: bool = True,
) -> pd.DataFrame:
    """计算未来 ``horizon`` 年内发生下调的概率（可按当前评级档分解）。

    与 :func:`upgrade_downgrade_probabilities` 不同，这里允许跨越不连续的年份，
    更适合回答「当前评级为 BB 的国家，未来 3 年内被下调的概率是多少」。
    """
    present = [c for c in entity_cols if c in panel.columns]
    use_col = "score_in_effect" if "score_in_effect" in panel.columns else score_col
    frame = panel[[*present, time_col, use_col]].copy()
    frame[time_col] = pd.to_numeric(frame[time_col], errors="coerce")
    frame[use_col] = pd.to_numeric(frame[use_col], errors="coerce")
    frame = frame.dropna(subset=[*present, time_col, use_col]).sort_values([*present, time_col])

    grouped = frame.groupby(present, dropna=False)
    frame["future_score"] = grouped[use_col].shift(-horizon)
    frame["future_year"] = grouped[time_col].shift(-horizon)
    valid = frame[frame["future_year"] == frame[time_col] + horizon].copy()
    valid["future_delta"] = valid["future_score"] - valid[use_col]
    valid["downgraded"] = (valid["future_delta"] < 0).astype(int)
    valid["upgraded"] = (valid["future_delta"] > 0).astype(int)
    if by_rating:
        valid["rating_bucket"] = valid[use_col].map(rating_bucket)
        keys = ["rating_bucket"]
    else:
        keys = []

    if not keys:
        return pd.DataFrame(
            {
                "horizon": [horizon],
                "n_obs": [len(valid)],
                "p_downgrade": [valid["downgraded"].mean() if len(valid) else np.nan],
                "p_upgrade": [valid["upgraded"].mean() if len(valid) else np.nan],
            }
        )

    summary = (
        valid.groupby(keys, dropna=False)
        .agg(
            n_obs=("downgraded", "size"),
            p_downgrade=("downgraded", "mean"),
            p_upgrade=("upgraded", "mean"),
        )
        .reset_index()
    )
    summary["horizon"] = horizon
    return summary


def rating_spell_stats(
    panel: pd.DataFrame,
    *,
    entity_cols: Sequence[str] = _ENTITY_COLS,
    time_col: str = "year",
    score_col: str = "rating_score",
) -> pd.DataFrame:
    """计算每个实体在各个评级档上的「停留片段」（spell）。

    Returns
    -------
    pandas.DataFrame
        每个国家-机构-评级片段一行，含 ``start_year`` / ``end_year`` /
        ``duration``（年）/ ``ongoing``（片段是否在样本期末仍持续，属于右删失）。
    """
    present = [c for c in entity_cols if c in panel.columns]
    frame = panel[[*present, time_col, score_col]].copy()
    frame[time_col] = pd.to_numeric(frame[time_col], errors="coerce")
    frame[score_col] = pd.to_numeric(frame[score_col], errors="coerce")
    frame = frame.dropna(subset=[*present, time_col, score_col])
    frame = frame.sort_values([*present, time_col]).reset_index(drop=True)

    records: list[dict[str, Any]] = []
    for key, chunk in frame.groupby(present, dropna=False, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        base = dict(zip(present, key_tuple, strict=False))
        scores = chunk[score_col].to_numpy()
        years = chunk[time_col].to_numpy()
        if len(chunk) == 0:
            continue
        start = 0
        for i in range(1, len(chunk) + 1):
            if i == len(chunk) or scores[i] != scores[start]:
                records.append(
                    {
                        **base,
                        "rating_score": int(scores[start]),
                        "start_year": int(years[start]),
                        "end_year": int(years[i - 1]),
                        "duration": int(years[i - 1] - years[start] + 1),
                        "ongoing": i == len(chunk),
                    }
                )
                start = i
    return pd.DataFrame(records)


def rating_cycle_stats(
    panel: pd.DataFrame,
    *,
    group_cols: Sequence[str] | None = ("agency",),
    entity_cols: Sequence[str] = _ENTITY_COLS,
    time_col: str = "year",
    score_col: str = "rating_score",
) -> pd.DataFrame:
    """评级周期统计：平均停留时长、行动频率、迁移幅度。

    ``group_cols=None`` 时返回全样本一行。
    """
    spells = rating_spell_stats(
        panel, entity_cols=entity_cols, time_col=time_col, score_col=score_col
    )
    if spells.empty:
        return pd.DataFrame(
            columns=[
                "n_spells",
                "mean_spell_years",
                "median_spell_years",
                "max_spell_years",
                "mean_abs_change",
                "share_ongoing",
            ]
        )

    changes = _prepare_transitions(
        panel, entity_cols=entity_cols, time_col=time_col, score_col=score_col
    )

    keys = [c for c in (group_cols or []) if c in spells.columns]
    grouped: Any = spells.groupby(keys, dropna=False) if keys else [(None, spells)]
    rows: list[dict[str, Any]] = []
    for key, chunk in grouped:
        record: dict[str, Any] = {}
        if keys:
            key_tuple = key if isinstance(key, tuple) else (key,)
            record.update(dict(zip(keys, key_tuple, strict=False)))
        record.update(
            {
                "n_spells": len(chunk),
                "mean_spell_years": float(chunk["duration"].mean()),
                "median_spell_years": float(chunk["duration"].median()),
                "max_spell_years": int(chunk["duration"].max()),
                "mean_abs_change": np.nan,
                "share_ongoing": float(chunk["ongoing"].mean()),
            }
        )
        if not changes.empty and keys:
            subset = changes
            for column in keys:
                if column in subset.columns and column in record:
                    subset = subset[subset[column] == record[column]]
            if len(subset):
                record["mean_abs_change"] = float(subset["delta"].abs().mean())
        elif not changes.empty:
            record["mean_abs_change"] = float(changes["delta"].abs().mean())
        rows.append(record)

    return pd.DataFrame(rows)


def agency_dispersion(
    panel: pd.DataFrame,
    *,
    country_col: str = "country_iso3",
    time_col: str = "year",
    agency_col: str = "agency",
    score_col: str = "rating_score",
) -> pd.DataFrame:
    """计算同一国家-年份下不同机构之间的评级分歧。

    ``range_score > 0`` 即所谓的 **split rating**（机构间评级不一致）。
    分歧本身是重要的风险信号，也是本研究关注「机构差异」的核心度量。
    """
    for required in (country_col, time_col, agency_col):
        if required not in panel.columns:
            raise KeyError(f"缺少必需列: {required}")
    use_col = "score_in_effect" if "score_in_effect" in panel.columns else score_col
    frame = panel[[country_col, time_col, agency_col, use_col]].copy()
    frame[use_col] = pd.to_numeric(frame[use_col], errors="coerce")
    frame = frame.dropna(subset=[country_col, time_col, use_col])

    result = (
        frame.groupby([country_col, time_col], dropna=False)
        .agg(
            n_agencies=(use_col, "nunique"),
            mean_score=(use_col, "mean"),
            min_score=(use_col, "min"),
            max_score=(use_col, "max"),
            std_score=(use_col, "std"),
        )
        .reset_index()
    )
    result["range_score"] = result["max_score"] - result["min_score"]
    result["is_split"] = (result["range_score"] > 0).astype(int)
    result["std_score"] = result["std_score"].fillna(0.0)
    return result


def migration_entropy(matrix: pd.DataFrame, *, per_row: bool = False) -> float | pd.Series:
    """迁移矩阵的信息熵（迁移不确定性的度量）。

    ``per_row=False`` 返回全矩阵归一化后的香农熵；``per_row=True`` 返回每一行的熵
    （评级档越高，通常熵越小 = 越稳定）。
    """
    probabilities = transition_probability_matrix(matrix)
    values = probabilities.to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(values > 0, np.log(values), 0.0)
        entropies = -(values * log_p).sum(axis=1)
    if per_row:
        return pd.Series(entropies, index=matrix.index, name="entropy")
    # 用总体迁移概率加权
    row_mass = matrix.to_numpy(dtype="float64").sum(axis=1)
    weights = row_mass / row_mass.sum() if row_mass.sum() > 0 else np.zeros_like(row_mass)
    return float((entropies * weights).sum())
