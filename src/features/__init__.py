"""特征工程模块。"""

from __future__ import annotations

from src.features.macro_features import (
    DEFAULT_FEATURES,
    DERIVED_FEATURES,
    FEATURE_GROUPS,
    add_changes,
    add_derived_features,
    add_lags,
    build_feature_matrix,
    describe_features,
    winsorize,
    zscore_within,
)
from src.features.rating_scale import (
    INVESTMENT_GRADE_THRESHOLD,
    MAX_SCORE,
    MIN_SCORE,
    apply_rating_mapping,
    is_investment_grade,
    map_rating,
    map_rating_series,
    normalize_agency,
    normalize_outlook,
    rating_action_type,
    rating_scale_table,
    score_to_grade,
)

__all__ = [
    "DEFAULT_FEATURES",
    "DERIVED_FEATURES",
    "FEATURE_GROUPS",
    "INVESTMENT_GRADE_THRESHOLD",
    "MAX_SCORE",
    "MIN_SCORE",
    "add_changes",
    "add_derived_features",
    "add_lags",
    "apply_rating_mapping",
    "build_feature_matrix",
    "describe_features",
    "is_investment_grade",
    "map_rating",
    "map_rating_series",
    "normalize_agency",
    "normalize_outlook",
    "rating_action_type",
    "rating_scale_table",
    "score_to_grade",
    "winsorize",
    "zscore_within",
]
