"""树模型：随机森林与 XGBoost。

定位
----
有序概率模型给出可解释的参数与方向，但假设了「平行回归线」（proportional odds），
且难以捕捉非线性与交互效应。随机森林与 XGBoost 作为**预测性能基准**，
允许任意非线性关系；配合 SHAP 仍然可以解释特征贡献。

注意：树模型对评级这类高度粘性的目标容易「过度保守」（倾向于预测众数档），
因此除了准确率外，必须报告**相邻档位准确率**与**下调方向的召回率**——
实务中「预警下调」比「精确命中某一档」更有价值。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.config import get_config
from src.utils.logging_utils import get_logger

__all__ = [
    "OrdinalTreeClassifier",
    "adjacent_accuracy",
    "build_random_forest",
    "build_xgboost",
    "fit_tree_model",
    "tree_predict_proba",
    "tune_ordinal_weights",
]

logger = get_logger(__name__)

try:  # pragma: no cover - 环境相关
    from xgboost import XGBClassifier as _XGBClassifier

    _HAS_XGBOOST = True
except Exception:
    _XGBClassifier = None  # type: ignore[assignment]
    _HAS_XGBOOST = False


def build_random_forest(
    n_classes: int = 21,
    *,
    random_state: int | None = None,
    **overrides: Any,
) -> RandomForestClassifier:
    """构造随机森林分类器（默认参数读取 ``config.yaml``）。"""
    cfg = get_config().get("models", {}).get("random_forest", {})
    params: dict[str, Any] = {
        "n_estimators": int(cfg.get("n_estimators", 400)),
        "max_depth": cfg.get("max_depth", None),
        "min_samples_leaf": int(cfg.get("min_samples_leaf", 3)),
        "n_jobs": int(cfg.get("n_jobs", -1)),
        "random_state": random_state
        if random_state is not None
        else int(get_config().get("models", {}).get("random_state", 42)),
        "class_weight": "balanced_subsample",
    }
    params.update(overrides)
    logger.debug("随机森林参数: %s (n_classes=%s)", params, n_classes)
    return RandomForestClassifier(**params)


def build_xgboost(
    n_classes: int = 21,
    *,
    objective: str | None = None,
    random_state: int | None = None,
    **overrides: Any,
) -> Any:
    """构造 XGBoost 分类器。

    多分类时使用 ``multi:softprob``（输出每档概率），二分类时使用
    ``binary:logistic``。若环境未安装 xgboost，会抛出带说明的 ``ImportError``。
    """
    if not _HAS_XGBOOST:  # pragma: no cover
        raise ImportError(
            "未检测到 xgboost。请安装：pip install xgboost>=2.0，"
            "或改用 build_random_forest() 作为替代。"
        )
    cfg = get_config().get("models", {}).get("xgboost", {})
    params: dict[str, Any] = {
        "n_estimators": int(cfg.get("n_estimators", 400)),
        "learning_rate": float(cfg.get("learning_rate", 0.05)),
        "max_depth": int(cfg.get("max_depth", 4)),
        "subsample": float(cfg.get("subsample", 0.85)),
        "colsample_bytree": float(cfg.get("colsample_bytree", 0.85)),
        "reg_lambda": float(cfg.get("reg_lambda", 1.0)),
        "random_state": random_state
        if random_state is not None
        else int(get_config().get("models", {}).get("random_state", 42)),
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if objective is None:
        objective = "binary:logistic" if n_classes <= 2 else "multi:softprob"
    params["objective"] = objective
    params["eval_metric"] = "logloss" if n_classes <= 2 else "mlogloss"
    # 不显式设置 num_class：XGBoost 的 sklearn 接口会从训练标签自动推断，
    # 手动设置反而会与实际类别数冲突（各折的类别数可能少于全局类别数）。
    params.update(overrides)
    logger.debug("XGBoost 参数: %s", params)
    return _XGBClassifier(**params)


def fit_tree_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: Sequence[Any],
    *,
    X_valid: pd.DataFrame | None = None,
    y_valid: Sequence[Any] | None = None,
    balance_classes: bool = True,
) -> Any:
    """拟合树模型，并统一 sklearn / XGBoost 的接口差异。

    关键处理：XGBoost 要求类别标签为从 0 开始的连续整数，而评级分值是 1-21
    且样本中可能缺少某些档位（例如没有处于 CCC- 的国家）。本函数在内部完成
    ``原分值 <-> 内部编码`` 的映射，并把原始类别顺序挂到
    ``model._sovereign_classes``，供 :func:`tree_predict_proba` 还原标签。

    ``balance_classes=True`` 时按类别频次倒数加权——评级样本绝大多数集中在少数
    档位，不加权会使模型退化为「永远预测众数」。
    """
    y_array = np.asarray(y_train)
    classes = np.unique(y_array)

    weights = None
    if balance_classes:
        try:
            weights = compute_sample_weight(class_weight="balanced", y=y_array)
        except Exception as exc:
            logger.warning("类别权重计算失败（%s），改回不加权拟合", exc)
            weights = None

    is_xgb = _HAS_XGBOOST and model.__class__.__module__.startswith("xgboost")

    if is_xgb:
        code_map = {value: index for index, value in enumerate(classes)}
        y_codes = np.asarray([code_map[v] for v in y_array], dtype="int32")
        fit_kwargs: dict[str, Any] = {"verbose": False}
        if weights is not None:
            fit_kwargs["sample_weight"] = weights
        if X_valid is not None and y_valid is not None:
            y_valid_codes = np.asarray([code_map.get(v, -1) for v in np.asarray(y_valid)])
            keep = y_valid_codes >= 0
            if keep.any():
                fit_kwargs["eval_set"] = [(X_valid.loc[keep], y_valid_codes[keep])]
        model.fit(X_train, y_codes, **fit_kwargs)
    else:
        fit_kwargs = {}
        if weights is not None:
            fit_kwargs["sample_weight"] = weights
        model.fit(X_train, y_array, **fit_kwargs)

    model._sovereign_classes = classes  # type: ignore[attr-defined]
    return model


def tree_predict_proba(model: Any, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """返回 ``(概率矩阵, 类别数组)``，统一 XGBoost 与 sklearn 的差异。"""
    proba = np.asarray(model.predict_proba(X), dtype="float64")
    classes = getattr(model, "_sovereign_classes", None)
    if classes is None:
        classes = getattr(model, "classes_", None)
    if classes is None:  # pragma: no cover
        classes = np.arange(proba.shape[1])
    return proba, np.asarray(classes)


def tune_ordinal_weights(y: Sequence[Any], *, decay: float = 0.6) -> np.ndarray:
    """为「相邻档位误差」构造样本权重。

    评级预测中，把 AAA 预测成 AA+ 与预测成 CCC 的代价天差地别。本函数给每个
    观测按其**实际档位频次**与**档位间距**加权：低频档位权重更高，用于缓解
    类别极度不平衡。
    """
    values = np.asarray(y, dtype="float64")
    counts = pd.Series(values).value_counts()
    inv = values.copy()
    for value, count in counts.items():
        inv[values == value] = 1.0 / max(count, 1)
    weights = np.power(inv, decay)
    return weights / weights.mean()


def adjacent_accuracy(y_true: Sequence[Any], y_pred: Sequence[Any], tolerance: int = 1) -> float:
    """相邻档位准确率：预测值与真实值相差不超过 ``tolerance`` 档的占比。

    这是评级预测中比「精确命中」更有意义的指标。
    """
    truth = np.asarray(y_true, dtype="float64")
    pred = np.asarray(y_pred, dtype="float64")
    if truth.size == 0:
        return float("nan")
    return float((np.abs(truth - pred) <= tolerance).mean())


class OrdinalTreeClassifier(BaseEstimator, ClassifierMixin):
    """scikit-learn 兼容的树分类器封装：内部完成「任意标签 -> 0..K-1」的编码与还原。

    为什么需要它
    ------------
    评级分值是 1-21 的整数，且不同折（时间窗口）中的类别集合可能不同。
    XGBoost 的 sklearn 接口要求标签为从 0 开始的连续整数，直接把这套标签喂给它
    会报 ``Invalid classes inferred from unique values of y``。
    本封装在 ``fit`` 时把标签重编码为 ``0..K-1``，在 ``predict`` 时还原，
    并把类别顺序暴露为 ``classes_`` —— 与 scikit-learn 的约定一致，
    因此可以直接用于 :func:`src.models.timeseries_cv.cross_validate_model`。

    Parameters
    ----------
    backend:
        ``"xgboost"`` 或 ``"random_forest"``。
    random_state:
        随机种子；``None`` 时读取 ``config.yaml``。
    **params:
        覆盖 ``config.yaml`` 中的模型超参数（如 ``n_estimators``、``max_depth``）。
    """

    def __init__(self, backend: str = "xgboost", random_state: int | None = None, **params: Any):
        self.backend = backend
        self.random_state = random_state
        self.params = params

    def _make_estimator(self, n_classes: int) -> Any:
        if self.backend == "xgboost":
            return build_xgboost(
                n_classes=n_classes, random_state=self.random_state, **self.params
            )
        if self.backend == "random_forest":
            return build_random_forest(
                n_classes=n_classes, random_state=self.random_state, **self.params
            )
        raise ValueError(f"不支持的 backend={self.backend!r}，可选: 'xgboost' / 'random_forest'")

    def fit(self, X: pd.DataFrame, y: Sequence[Any]) -> OrdinalTreeClassifier:
        y_array = np.asarray(y)
        self.classes_, codes = np.unique(y_array, return_inverse=True)
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else len(X[0])
        self.estimator_ = self._make_estimator(len(self.classes_))

        weights = None
        try:
            weights = compute_sample_weight(class_weight="balanced", y=y_array)
        except Exception as exc:
            logger.warning("类别权重计算失败（%s），改回不加权拟合", exc)

        if self.backend == "xgboost":
            fit_kwargs: dict[str, Any] = {"verbose": False}
            if weights is not None:
                fit_kwargs["sample_weight"] = weights
            self.estimator_.fit(X, codes, **fit_kwargs)
        else:
            fit_kwargs = {}
            if weights is not None:
                fit_kwargs["sample_weight"] = weights
            self.estimator_.fit(X, codes, **fit_kwargs)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("模型尚未拟合，请先调用 fit")
        return np.asarray(self.estimator_.predict_proba(X), dtype="float64")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @property
    def feature_importances_(self) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("模型尚未拟合，请先调用 fit")
        return np.asarray(self.estimator_.feature_importances_, dtype="float64")

    @property
    def estimator(self) -> Any:
        """底层的 sklearn / XGBoost 估计器（用于 SHAP 等需要原生对象的场景）。"""
        if not hasattr(self, "estimator_"):
            raise RuntimeError("模型尚未拟合，请先调用 fit")
        return self.estimator_
