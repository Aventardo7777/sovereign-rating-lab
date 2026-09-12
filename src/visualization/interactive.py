"""交互式图表（plotly）。

返回 plotly ``Figure`` 对象，可直接在 Jupyter / Streamlit 中渲染。
配色同样遵循中国金融惯例：上调=红，下调=绿。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.features.rating_scale import MAX_SCORE, MIN_SCORE, score_to_grade
from src.utils.logging_utils import get_logger

__all__ = [
    "migration_heatmap_plotly",
    "rating_trajectory_plotly",
    "shap_bar_plotly",
]

logger = get_logger(__name__)

_UPGRADE_COLOR = "#C0392B"
_DOWNGRADE_COLOR = "#1E8449"
_NEUTRAL_COLOR = "#7F8C8D"


def migration_heatmap_plotly(
    matrix: pd.DataFrame,
    *,
    normalize: bool = True,
    title: str = "主权评级迁移矩阵",
) -> go.Figure:
    """交互式迁移矩阵热力图。"""
    values = matrix.to_numpy(dtype="float64").copy()
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > 0)
    ticks = list(matrix.index)
    labels = [score_to_grade(t, "S&P") or str(t) for t in ticks]
    text = np.where(values > 0, np.round(values, 3).astype(str), "")

    figure = go.Figure(
        data=go.Heatmap(
            z=values,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=text,
            texttemplate="%{text}",
            hovertemplate="t 期 %{y} → t+1 期 %{x}<br>概率 %{z:.3f}<extra></extra>",
            colorbar={"title": "概率" if normalize else "观测数"},
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="t+1 期评级",
        yaxis_title="t 期评级",
        height=680,
        template="plotly_white",
    )
    return figure


def rating_trajectory_plotly(
    panel: pd.DataFrame,
    *,
    country: str,
    score_col: str = "score_in_effect",
    agencies: Sequence[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """交互式评级轨迹图（含投资级门槛参考线）。"""
    subset = panel[panel["country_iso3"] == country].copy()
    if agencies:
        subset = subset[subset["agency"].isin(list(agencies))]
    figure = go.Figure()
    if subset.empty:
        figure.update_layout(
            title=f"{country}：无可用观测",
            template="plotly_white",
            height=480,
        )
        return figure

    for agency, chunk in subset.groupby("agency"):
        chunk = chunk.sort_values("year")
        figure.add_trace(
            go.Scatter(
                x=chunk["year"],
                y=chunk[score_col],
                mode="lines+markers",
                name=str(agency),
                customdata=[
                    score_to_grade(v, "S&P") if pd.notna(v) else None for v in chunk[score_col]
                ],
                hovertemplate="%{x} 年<br>分值 %{y}<br>评级 %{customdata}<extra>%{fullData.name}</extra>",
            )
        )
    tick_values = list(range(MIN_SCORE, MAX_SCORE + 1, 2))
    figure.update_layout(
        title=title or f"{country} 主权评级轨迹",
        xaxis_title="年份",
        yaxis_title="评级（1-21 统一刻度）",
        yaxis={
            "tickmode": "array",
            "tickvals": tick_values,
            "ticktext": [score_to_grade(v, "S&P") or str(v) for v in tick_values],
        },
        template="plotly_white",
        height=520,
        legend_title="评级机构",
    )
    figure.add_hline(
        y=12,
        line_dash="dash",
        line_color=_NEUTRAL_COLOR,
        annotation_text="投资级门槛",
        annotation_position="top left",
    )
    return figure


def shap_bar_plotly(
    summary: pd.DataFrame,
    *,
    top_n: int = 15,
    title: str = "SHAP 特征重要性",
    value_col: str | None = None,
) -> go.Figure:
    """SHAP 全局重要性条形图，按方向着色。"""
    if value_col is None:
        value_col = "mean_abs_shap" if "mean_abs_shap" in summary.columns else "importance"
    subset = summary.head(top_n).iloc[::-1]
    colors = [
        _UPGRADE_COLOR if v >= 0 else _DOWNGRADE_COLOR
        for v in subset.get("mean_shap", pd.Series(np.zeros(len(subset))))
    ]
    figure = go.Figure(
        go.Bar(
            x=subset[value_col],
            y=subset["feature"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>贡献 %{x:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title=value_col,
        template="plotly_white",
        height=max(360, 26 * len(subset) + 120),
    )
    return figure
