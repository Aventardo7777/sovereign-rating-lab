"""SHAP 可解释性分析。

SHAP（SHapley Additive exPlanations）把模型预测分解为各特征的加性贡献，
是树模型与线性模型都可用的统一解释框架。在本研究中它回答两个问题：

1. **哪些宏观变量最重要？**（全局重要性）
2. **对某个具体国家-年份，为什么模型给出较高的下调概率？**（局部解释）

实现说明
--------
* SHAP 对多分类模型返回的是「类别 × 样本 × 特征」的三维数组；本模块统一聚合为
  「样本 × 特征」的平均绝对贡献，便于绘图与制表。
* SHAP 是**可选依赖**。若环境未安装或计算失败（例如模型类型不支持、样本过大），
  本模块返回 ``None`` 并给出说明，而不是中断整个流水线。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_config
from src.utils.logging_utils import get_logger

__all__ = [
    "ShapResult",
    "explain_with_shap",
    "shap_available",
    "shap_summary_table",
]

logger = get_logger(__name__)

try:  # pragma: no cover - 依赖 shap 版本
    import shap as _shap

    _HAS_SHAP = True
except Exception:
    _shap = None  # type: ignore[assignment]
    _HAS_SHAP = False


@dataclass
class ShapResult:
    """SHAP 分析结果。"""

    values: np.ndarray  # shape = (n_samples, n_features)，平均绝对贡献
    feature_names: list[str]
    base_value: float | None = None
    explainer: str = ""
    raw: Any = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.values.size > 0

    def to_frames(self) -> dict[str, pd.DataFrame]:
        """返回 ``{"summary": ..., "per_sample": ...}`` 两个表。"""
        summary = shap_summary_table(self.values, self.feature_names)
        per_sample = pd.DataFrame(self.values, columns=self.feature_names)
        return {"summary": summary, "per_sample": per_sample}


def shap_available() -> bool:
    """当前环境是否可用 SHAP。"""
    return _HAS_SHAP


def _normalize_shap_values(raw: Any, n_samples: int, n_features: int) -> np.ndarray:
    """把 SHAP 的各种返回形状统一为 ``(n_samples, n_features)`` 的平均绝对贡献。"""
    if isinstance(raw, list):
        stacked = np.stack([np.asarray(item, dtype="float64") for item in raw], axis=0)
        stacked = np.abs(stacked)
        return stacked.mean(axis=0).reshape(n_samples, n_features)
    array = np.asarray(raw, dtype="float64")
    if array.ndim == 3:
        # (n_samples, n_features, n_classes)
        if array.shape[0] == n_samples and array.shape[1] == n_features:
            return np.abs(array).mean(axis=2)
        # (n_classes, n_samples, n_features)
        if array.shape[1] == n_samples and array.shape[2] == n_features:
            return np.abs(array).mean(axis=0)
    if array.ndim == 2:
        return np.abs(array)
    raise ValueError(f"无法识别的 SHAP 输出形状: {array.shape}")


def explain_with_shap(
    model: Any,
    X: pd.DataFrame,
    *,
    max_samples: int | None = None,
    background: pd.DataFrame | None = None,
    feature_names: Sequence[str] | None = None,
) -> ShapResult:
    """计算 SHAP 值。

    Parameters
    ----------
    model:
        已拟合的模型（树模型或线性模型）。
    X:
        需要解释的样本（通常为测试集）。
    max_samples:
        最大解释样本数（默认读取 ``config.yaml`` 的 ``models.shap.max_samples``）。
        大数据集上 SHAP 计算很慢，采样是必要的工程折中。
    background:
        背景数据集，仅对 ``KernelExplainer`` / ``LinearExplainer`` 有意义。

    Returns
    -------
    ShapResult
        失败时 ``ok`` 为 ``False``，``message`` 说明原因。
    """
    if not _HAS_SHAP:
        return ShapResult(
            values=np.empty((0, 0)),
            feature_names=list(feature_names or X.columns),
            message="未安装 shap。请运行 pip install shap>=0.44，或跳过 SHAP 分析。",
        )

    cfg = get_config().get("models", {}).get("shap", {})
    resolved_max = int(max_samples if max_samples is not None else cfg.get("max_samples", 500))

    frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    if len(frame) > resolved_max:
        frame = frame.sample(n=resolved_max, random_state=42).reset_index(drop=True)
        logger.info("SHAP：样本量超过 %d，已随机抽样", resolved_max)

    names = list(feature_names) if feature_names is not None else list(frame.columns)

    try:
        explainer_name = ""
        raw_values: Any = None
        base_value: float | None = None

        tree_explainer = getattr(_shap, "TreeExplainer", None)
        is_tree = hasattr(model, "feature_importances_") or any(
            marker in type(model).__module__ for marker in ("xgboost", "sklearn.ensemble")
        )
        if tree_explainer is not None and is_tree:
            try:
                explainer = tree_explainer(model)
                raw_values = explainer.shap_values(frame)
                base_value = float(np.asarray(explainer.expected_value).ravel()[0])
                explainer_name = "TreeExplainer"
            except Exception as exc:
                logger.info("TreeExplainer 不适用（%s），改用 KernelExplainer", exc)
                raw_values = None

        if raw_values is None:
            if background is None:
                background = _shap.sample(frame, min(len(frame), 100), random_state=42)
            predict_fn = (
                model.predict_proba if hasattr(model, "predict_proba") else model.predict
            )
            explainer = _shap.KernelExplainer(predict_fn, background)
            raw_values = explainer.shap_values(frame, nsamples=min(200, 2 * len(names) + 100))
            expected = np.asarray(explainer.expected_value, dtype="float64").ravel()
            base_value = float(expected[0]) if expected.size else None
            explainer_name = "KernelExplainer"

        values = _normalize_shap_values(raw_values, len(frame), len(names))
        return ShapResult(
            values=values,
            feature_names=names,
            base_value=base_value,
            explainer=explainer_name,
            raw=raw_values,
        )
    except Exception as exc:
        logger.warning("SHAP 计算失败：%s", exc)
        return ShapResult(
            values=np.empty((0, 0)),
            feature_names=names,
            message=f"SHAP 计算失败：{exc}。可改用内置特征重要性（feature_importance_table）。",
        )


def shap_summary_table(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
    *,
    top_n: int | None = None,
) -> pd.DataFrame:
    """把 SHAP 值汇总为全局重要性表。"""
    values = np.asarray(shap_values, dtype="float64")
    if values.ndim != 2:
        raise ValueError("shap_values 须为二维数组 (n_samples, n_features)")
    names = list(feature_names)
    if values.shape[1] != len(names):
        raise ValueError(
            f"特征数不匹配：SHAP 有 {values.shape[1]} 列，特征名有 {len(names)} 个"
        )
    table = pd.DataFrame(
        {
            "feature": names,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
            "std_shap": values.std(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    total = table["mean_abs_shap"].sum()
    table["share_pct"] = 100 * table["mean_abs_shap"] / total if total else np.nan
    table["direction"] = np.where(
        table["mean_shap"] > 0, "正向（提升分值）", "负向（压低分值）"
    )
    if top_n:
        table = table.head(top_n)
    return table
