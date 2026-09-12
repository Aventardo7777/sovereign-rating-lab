"""有序 Logit / Probit 模型（含 scikit-learn 兼容封装）。

计量背景
--------
评级是**有序**离散变量（1-21），用普通多分类模型会忽略档位间的次序信息。
有序概率模型设为：

.. math::

    P(y_i \\le j \\mid x_i) = F(\\theta_j - x_i'\\beta)

其中 :math:`F` 为 logistic（Logit）或标准正态（Probit）分布函数，
:math:`\\theta_j` 为待估阈值。

**重要实现细节**：阈值 :math:`\\theta_j` 会吸收线性指数中的常数项，
因此在设计矩阵中加入常数会导致不可识别（Hessian 奇异、不收敛）。
本模块会自动剔除名为 ``const`` / ``intercept`` 的列并给出警告。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from src.utils.logging_utils import get_logger

__all__ = [
    "OrdinalModelResult",
    "OrdinalProbabilityModel",
    "fit_ordinal_logit",
    "fit_ordinal_probit",
    "predict_ordinal_probabilities",
]

logger = get_logger(__name__)

_CONSTANT_NAMES = {"const", "intercept", "constant"}

try:  # pragma: no cover - 依赖 statsmodels 版本
    from statsmodels.miscmodels.ordinal_model import OrderedModel as _OrderedModel
except Exception:
    _OrderedModel = None  # type: ignore[assignment]


def _as_design(X: pd.DataFrame | np.ndarray, feature_names: Sequence[str] | None = None):
    """把设计矩阵转为 statsmodels 友好的形式，并剔除常数项。"""
    if isinstance(X, pd.DataFrame):
        kept = [c for c in X.columns if str(c).lower() not in _CONSTANT_NAMES]
        dropped = [c for c in X.columns if c not in kept]
        if dropped:
            logger.warning(
                "已从设计矩阵中剔除常数列 %s：有序模型的阈值参数会吸收常数项，"
                "保留会导致不可识别。",
                dropped,
            )
        frame = X[kept].astype("float64")
        return frame, list(frame.columns)
    array = np.asarray(X, dtype="float64")
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    names = list(feature_names) if feature_names is not None else [
        f"x{i}" for i in range(array.shape[1])
    ]
    return pd.DataFrame(array, columns=names), names


@dataclass
class OrdinalModelResult:
    """有序模型的拟合结果封装。"""

    name: str
    distribution: str
    params: pd.Series
    std_errors: pd.Series
    pvalues: pd.Series
    zvalues: pd.Series
    llf: float
    nobs: int
    categories: np.ndarray
    feature_names: list[str]
    threshold_names: list[str]
    raw_result: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def prsquared(self) -> float:
        """McFadden 伪 R²。"""
        return float(1.0 - self.llf / self.extra["ll_null"]) if self.extra.get("ll_null") else np.nan

    def to_frame(self, *, thresholds: bool = False) -> pd.DataFrame:
        """系数表（默认只返回解释变量，``thresholds=True`` 时含阈值）。"""
        frame = pd.DataFrame(
            {
                "coef": self.params,
                "std_err": self.std_errors,
                "z": self.zvalues,
                "p_value": self.pvalues,
            }
        )
        frame["signif"] = np.where(
            frame["p_value"] < 0.01,
            "***",
            np.where(frame["p_value"] < 0.05, "**", np.where(frame["p_value"] < 0.1, "*", "")),
        )
        if not thresholds:
            frame = frame.loc[[n for n in self.feature_names if n in frame.index]]
        frame.attrs["name"] = self.name
        frame.attrs["nobs"] = self.nobs
        frame.attrs["pseudo_r2"] = self.prsquared
        return frame

    def summary_text(self) -> str:
        lines = [
            f"有序{self.distribution.title()} 模型: {self.name}",
            f"观测数: {self.nobs} | 类别数: {len(self.categories)} | "
            f"Log-Likelihood: {self.llf:.2f} | 伪 R²: {self.prsquared:.4f}",
            "-" * 68,
        ]
        for index, row in self.to_frame().iterrows():
            lines.append(
                f"{index!s:<28} {row['coef']:>10.4f} {row['std_err']:>10.4f} "
                f"{row['z']:>8.2f} {row['p_value']:>8.4f} {row['signif']}"
            )
        return "\n".join(lines)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        return predict_ordinal_probabilities(self, X)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.categories[np.argmax(proba.to_numpy(), axis=1)]

    def __repr__(self) -> str:  # pragma: no cover
        return f"<OrdinalModelResult {self.name!r} k={len(self.categories)} nobs={self.nobs}>"


def _fit_ordered(
    X: pd.DataFrame | np.ndarray,
    y: Sequence[Any],
    *,
    distribution: str,
    name: str | None = None,
    method: str = "bfgs",
    maxiter: int = 500,
) -> OrdinalModelResult:
    if _OrderedModel is None:  # pragma: no cover
        raise ImportError("需要 statsmodels >= 0.14 才能使用 OrderedModel")

    design, feature_names = _as_design(X)
    target = pd.Series(np.asarray(y)).reset_index(drop=True)
    finite = target.notna().to_numpy() & design.notna().all(axis=1).to_numpy()
    design, target = design.loc[finite].reset_index(drop=True), target.loc[finite].reset_index(drop=True)
    if design.empty:
        raise ValueError("清洗后无可用观测")

    codes, uniques = pd.factorize(target, sort=True)
    model = _OrderedModel(codes, design, distr=distribution)
    result = model.fit(method=method, disp=False, maxiter=maxiter)

    param_names = list(result.model.exog_names) if hasattr(result.model, "exog_names") else list(
        design.columns
    )
    # statsmodels 可能自动命名参数，这里以设计矩阵列名 + 阈值名为准
    n_features = design.shape[1]
    if len(param_names) != len(result.params):
        param_names = [*design.columns, *[f"threshold_{i}" for i in range(len(result.params) - n_features)]]
    else:
        param_names = [str(n) for n in param_names]
        param_names[:n_features] = list(design.columns)

    params = pd.Series(np.asarray(result.params, dtype="float64"), index=param_names)
    bse = pd.Series(np.asarray(result.bse, dtype="float64"), index=param_names)
    pvalues = pd.Series(np.asarray(result.pvalues, dtype="float64"), index=param_names)
    zvalues = pd.Series(np.asarray(result.tvalues, dtype="float64"), index=param_names)

    ll_null = float(np.log(1.0 / len(uniques)) * len(target))
    return OrdinalModelResult(
        name=name or f"有序{distribution.title()}：评分 ~ {' + '.join(feature_names)}",
        distribution=distribution,
        params=params,
        std_errors=bse,
        pvalues=pvalues,
        zvalues=zvalues,
        llf=float(result.llf),
        nobs=int(result.nobs),
        categories=np.asarray(uniques),
        feature_names=feature_names,
        threshold_names=list(param_names[n_features:]),
        raw_result=result,
        extra={"ll_null": ll_null, "method": method},
    )


def fit_ordinal_logit(
    X: pd.DataFrame | np.ndarray,
    y: Sequence[Any],
    *,
    name: str | None = None,
    method: str = "bfgs",
    maxiter: int = 500,
) -> OrdinalModelResult:
    """拟合有序 Logit 模型。

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame({"gdp": rng.normal(size=200), "debt": rng.normal(size=200)})
    >>> y = pd.cut(X["gdp"] - X["debt"] + rng.normal(size=200), bins=[-9, -0.5, 0.5, 9],
    ...            labels=[1, 2, 3]).astype(int)
    >>> res = fit_ordinal_logit(X, y)
    >>> res.nobs
    200
    """
    return _fit_ordered(X, y, distribution="logit", name=name, method=method, maxiter=maxiter)


def fit_ordinal_probit(
    X: pd.DataFrame | np.ndarray,
    y: Sequence[Any],
    *,
    name: str | None = None,
    method: str = "bfgs",
    maxiter: int = 500,
) -> OrdinalModelResult:
    """拟合有序 Probit 模型。"""
    return _fit_ordered(X, y, distribution="probit", name=name, method=method, maxiter=maxiter)


def predict_ordinal_probabilities(
    result: OrdinalModelResult,
    X: pd.DataFrame | np.ndarray,
) -> pd.DataFrame:
    """预测各类别的概率，返回 ``DataFrame``（列 = 类别，行 = 观测）。"""
    design, _ = _as_design(X)
    design = design.reindex(columns=result.feature_names, fill_value=0.0)
    raw = result.raw_result.model.predict(
        result.raw_result.params, exog=design.astype("float64"), which="prob"
    )
    proba = np.asarray(raw, dtype="float64")
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    if proba.shape[1] != len(result.categories) and proba.shape[0] == len(result.categories):
        proba = proba.T
    index = design.index if hasattr(design, "index") else None
    return pd.DataFrame(proba, columns=list(result.categories), index=index)


class OrdinalProbabilityModel(BaseEstimator, ClassifierMixin):
    """scikit-learn 兼容的有序概率分类器。

    用于在 :mod:`src.models.timeseries_cv` 的时间序列交叉验证中直接复用
    sklearn 的流程（``fit`` / ``predict`` / ``predict_proba`` / ``classes_``）。

    Parameters
    ----------
    distribution:
        ``"logit"`` 或 ``"probit"``。
    method, maxiter:
        传给 statsmodels 的优化设置。
    """

    def __init__(self, distribution: str = "logit", method: str = "bfgs", maxiter: int = 500) -> None:
        self.distribution = distribution
        self.method = method
        self.maxiter = maxiter
        self._result: OrdinalModelResult | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: Sequence[Any]) -> OrdinalProbabilityModel:
        self._result = _fit_ordered(
            X,
            y,
            distribution=self.distribution,
            method=self.method,
            maxiter=self.maxiter,
        )
        self.classes_ = np.asarray(self._result.categories)
        self.n_features_in_ = len(self._result.feature_names)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit")
        return self._result.predict_proba(X).to_numpy()

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit")
        return self._result.predict(X)

    @property
    def result(self) -> OrdinalModelResult:
        if self._result is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit")
        return self._result

    def summary_text(self) -> str:
        return self.result.summary_text()
