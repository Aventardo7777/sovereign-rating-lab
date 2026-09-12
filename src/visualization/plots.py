"""静态图表（matplotlib + seaborn）。

配色约定（遵循中国金融惯例，与欧美相反）：
**信用质量改善 / 评级上调 = 红色，信用质量恶化 / 评级下调 = 绿色。**
可在 ``config.yaml`` 的 ``visualization`` 段中调整。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # 无显示环境下也能出图（CI / 服务器）

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import get_config, resolve_path
from src.features.rating_scale import MAX_SCORE, MIN_SCORE, score_to_grade
from src.utils.logging_utils import get_logger

__all__ = [
    "apply_theme",
    "plot_coefficient_table",
    "plot_confusion_matrix",
    "plot_importance",
    "plot_migration_heatmap",
    "plot_probability_series",
    "plot_rating_trajectory",
    "save_figure",
]

logger = get_logger(__name__)

_THEME_APPLIED = False


def apply_theme(config: dict | None = None) -> dict[str, str]:
    """应用统一绘图主题（幂等），返回配色字典。"""
    global _THEME_APPLIED
    cfg = (config or get_config()).get("visualization", {})
    colors = {
        "upgrade": cfg.get("color_upgrade", "#C0392B"),
        "downgrade": cfg.get("color_downgrade", "#1E8449"),
        "neutral": cfg.get("color_neutral", "#7F8C8D"),
        "palette": cfg.get("palette", "RdYlGn_r"),
    }
    if _THEME_APPLIED:
        return colors

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": int(cfg.get("dpi", 150)),
            "figure.figsize": tuple(cfg.get("figsize", [10, 6])),
            "axes.unicode_minus": False,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "DejaVu Sans",
                "Arial",
            ],
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )
    _THEME_APPLIED = True
    return colors


def save_figure(
    fig: Any,
    filename: str,
    *,
    subdir: str | None = None,
    close: bool = True,
    formats: Sequence[str] = ("png",),
) -> list[Path]:
    """把图形保存到 ``reports/figures``（或指定子目录），返回写入的文件路径。"""
    cfg = get_config().get("visualization", {})
    base = resolve_path(get_config()["paths"]["figures"])
    if subdir:
        base = base / subdir
    base.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem
    written: list[Path] = []
    for fmt in formats:
        path = base / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", format=fmt)
        written.append(path)
        logger.info("已保存图: %s", path)
    if close:
        plt.close(fig)
    _ = cfg
    return written


def _grade_ticks(step: int = 1) -> tuple[list[int], list[str]]:
    ticks = list(range(MIN_SCORE, MAX_SCORE + 1, step))
    labels = [score_to_grade(t, "S&P") or str(t) for t in ticks]
    return ticks, labels


def plot_migration_heatmap(
    matrix: pd.DataFrame,
    *,
    normalize: bool = False,
    title: str = "主权评级年度迁移矩阵",
    annot: bool = True,
    figsize: tuple[float, float] = (11, 9),
) -> Any:
    """绘制迁移矩阵热力图。

    对角线（评级未变）用同一色系，非对角线保留原始数值以便观察迁移强度。
    """
    apply_theme()
    values = matrix.to_numpy(dtype="float64").copy()
    if normalize or (values.max() <= 1.0000001 and values.sum() > 0):
        row_sums = values.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > 0)
        fmt = ".2f"
        vmax = 1.0
    else:
        fmt = ".0f"
        vmax = None

    fig, ax = plt.subplots(figsize=figsize)
    ticks, labels = _grade_ticks()
    sns.heatmap(
        values,
        ax=ax,
        cmap="Blues",
        vmin=0.0,
        vmax=vmax,
        square=True,
        linewidths=0.4,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        annot=annot,
        fmt=fmt,
        annot_kws={"size": 6},
        cbar_kws={"label": "迁移概率" if normalize else "观测数"},
    )
    ax.set_xlabel("t+1 期评级")
    ax.set_ylabel("t 期评级")
    ax.set_title(title, pad=14)
    _ = ticks
    fig.tight_layout()
    return fig


def plot_rating_trajectory(
    panel: pd.DataFrame,
    *,
    country: str,
    agencies: Sequence[str] | None = None,
    score_col: str = "score_in_effect",
    title: str | None = None,
) -> Any:
    """绘制某国的评级轨迹（多机构对比）。"""
    colors = apply_theme()
    subset = panel[panel["country_iso3"] == country].copy()
    if agencies:
        subset = subset[subset["agency"].isin(list(agencies))]
    if subset.empty:
        raise ValueError(f"面板中没有国家 {country!r} 的观测")

    fig, ax = plt.subplots(figsize=(11, 6))
    palette = sns.color_palette("deep", max(subset["agency"].nunique(), 3))
    for color, (agency, chunk) in zip(palette, subset.groupby("agency"), strict=False):
        chunk = chunk.sort_values("year")
        ax.plot(
            chunk["year"],
            chunk[score_col],
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=color,
            label=str(agency),
        )
    ticks, labels = _grade_ticks(2)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axhline(12, color=colors["neutral"], linestyle="--", linewidth=1)
    ax.text(
        subset["year"].min(),
        12.25,
        "投资级门槛（BBB-/Baa3）",
        fontsize=8,
        color=colors["neutral"],
    )
    ax.set_xlabel("年份")
    ax.set_ylabel("评级（1-21 统一刻度）")
    ax.set_title(title or f"{country} 主权评级轨迹")
    ax.legend(title="评级机构", loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_probability_series(
    prob_df: pd.DataFrame,
    *,
    title: str = "评级迁移概率",
    xlabel: str = "年份",
    ylabel: str = "概率",
    highlight: str | None = None,
) -> Any:
    """绘制概率随时间的演化（上调/下调/稳定概率等）。"""
    colors = apply_theme()
    fig, ax = plt.subplots(figsize=(10, 5))
    color_map = {
        "p_downgrade": colors["downgrade"],
        "p_upgrade": colors["upgrade"],
        "p_stable": colors["neutral"],
    }
    x = prob_df.index
    for column in prob_df.columns:
        style = {"linewidth": 2.2 if column == highlight else 1.6}
        if column in color_map:
            style["color"] = color_map[column]
        ax.plot(x, prob_df[column], marker="o", markersize=3, label=str(column), **style)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_importance(
    importance: pd.DataFrame,
    *,
    top_n: int = 15,
    title: str = "特征重要性",
    value_col: str = "importance",
    label_col: str = "feature",
) -> Any:
    """水平条形图展示特征重要性（自动识别 SHAP 或内置重要性表）。"""
    apply_theme()
    value_col = "mean_abs_shap" if "mean_abs_shap" in importance.columns else value_col
    subset = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.42 * len(subset))))
    ax.barh(subset[label_col], subset[value_col], color="#2E86C1")
    ax.set_xlabel(value_col)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    table: pd.DataFrame,
    *,
    normalize: bool = False,
    title: str = "混淆矩阵",
) -> Any:
    """绘制混淆矩阵（行列标注为评级符号）。"""
    apply_theme()
    values = table.to_numpy(dtype="float64")
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        values,
        ax=ax,
        cmap="Blues",
        square=True,
        annot=values.shape[0] <= 12,
        fmt=".2f" if normalize else ".0f",
        annot_kws={"size": 7},
        xticklabels=list(table.columns),
        yticklabels=list(table.index),
        cbar_kws={"label": "比例" if normalize else "观测数"},
    )
    ax.set_xlabel("预测")
    ax.set_ylabel("实际")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_coefficient_table(
    coefficient_table: pd.DataFrame,
    *,
    title: str = "系数与 95% 置信区间",
    top_n: int = 20,
) -> Any:
    """森林图（coefplot）：系数点估计 + 置信区间。"""
    colors = apply_theme()
    subset = coefficient_table.head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.4 * len(subset))))
    y_positions = np.arange(len(subset))
    ax.errorbar(
        subset["coef"],
        y_positions,
        xerr=[
            subset["coef"] - subset.get("ci_lower", subset["coef"] - 1.96 * subset["std_err"]),
            subset.get("ci_upper", subset["coef"] + 1.96 * subset["std_err"]) - subset["coef"],
        ],
        fmt="o",
        color="#2C3E50",
        ecolor="#95A5A6",
        capsize=3,
        markersize=5,
    )
    ax.axvline(0, color=colors["neutral"], linestyle="--", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(i) for i in subset.index])
    ax.invert_yaxis()
    ax.set_xlabel("系数")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig
