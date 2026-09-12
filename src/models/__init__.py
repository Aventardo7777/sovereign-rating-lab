"""预测模型：有序概率模型、树模型、时间序列交叉验证、可解释性与评估。"""

from __future__ import annotations

from src.models.evaluate import (
    classification_metrics,
    confusion_table,
    feature_importance_table,
    roc_curve_points,
)
from src.models.explain import (
    explain_with_shap,
    shap_available,
    shap_summary_table,
)
from src.models.ordinal import (
    OrdinalModelResult,
    fit_ordinal_logit,
    fit_ordinal_probit,
    predict_ordinal_probabilities,
)
from src.models.timeseries_cv import (
    TimeSeriesFold,
    cross_validate_ordinal,
    expanding_year_folds,
    make_forward_chaining_splits,
)
from src.models.tree_models import (
    OrdinalTreeClassifier,
    build_random_forest,
    build_xgboost,
    fit_tree_model,
    tree_predict_proba,
)

__all__ = [
    "OrdinalModelResult",
    "OrdinalTreeClassifier",
    "TimeSeriesFold",
    "build_random_forest",
    "build_xgboost",
    "classification_metrics",
    "confusion_table",
    "cross_validate_model",
    "cross_validate_ordinal",
    "expanding_year_folds",
    "explain_with_shap",
    "feature_importance_table",
    "fit_ordinal_logit",
    "fit_ordinal_probit",
    "fit_tree_model",
    "make_forward_chaining_splits",
    "predict_ordinal_probabilities",
    "roc_curve_points",
    "shap_available",
    "shap_summary_table",
    "tree_predict_proba",
]
