"""分析模块：评级迁移、面板回归、事件研究。"""

from __future__ import annotations

from src.analysis.event_study import (
    abnormal_changes,
    event_study,
    event_study_summary,
    prepare_event_windows,
)
from src.analysis.migration import (
    agency_dispersion,
    build_migration_matrix,
    downgrade_probability,
    migration_entropy,
    migration_summary,
    rating_cycle_stats,
    rating_spell_stats,
    transition_probability_matrix,
    upgrade_downgrade_probabilities,
)
from src.analysis.panel_regression import (
    RegressionResult,
    fit_fixed_effects,
    fit_pooled_ols,
    run_driver_regressions,
)

__all__ = [
    "RegressionResult",
    "abnormal_changes",
    "agency_dispersion",
    "build_migration_matrix",
    "downgrade_probability",
    "event_study",
    "event_study_summary",
    "fit_fixed_effects",
    "fit_pooled_ols",
    "migration_entropy",
    "migration_summary",
    "prepare_event_windows",
    "rating_cycle_stats",
    "rating_spell_stats",
    "run_driver_regressions",
    "transition_probability_matrix",
    "upgrade_downgrade_probabilities",
]
