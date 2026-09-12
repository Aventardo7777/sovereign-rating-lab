"""可视化模块：静态图（matplotlib/seaborn）与交互图（plotly）。"""

from __future__ import annotations

from src.visualization.interactive import (
    migration_heatmap_plotly,
    rating_trajectory_plotly,
    shap_bar_plotly,
)
from src.visualization.plots import (
    apply_theme,
    plot_coefficient_table,
    plot_confusion_matrix,
    plot_importance,
    plot_migration_heatmap,
    plot_probability_series,
    plot_rating_trajectory,
    save_figure,
)

__all__ = [
    "apply_theme",
    "migration_heatmap_plotly",
    "plot_coefficient_table",
    "plot_confusion_matrix",
    "plot_importance",
    "plot_migration_heatmap",
    "plot_probability_series",
    "plot_rating_trajectory",
    "rating_trajectory_plotly",
    "save_figure",
    "shap_bar_plotly",
]
