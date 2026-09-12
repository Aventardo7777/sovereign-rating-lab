"""模型评估指标与表格。

评级预测的评估必须超越「准确率」：

* **准确率**在极度不平衡的评级分布上具有误导性（全部猜 BBB 也能得到不低的分数）；
* **平衡准确率 / 宏平均 F1** 对少数档位给予同等权重；
* **相邻档位准确率**（±1、±2 档）反映「预警方向是否正确」，实务价值最高；
* **有序 MAE / 二次加权 kappa** 直接度量档位距离，符合评级的有序性质；
* **AUC** 用于二分类化后的「是否下调」判别能力。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)

from src.features.rating_scale import score_to_grade
from src.utils.logging_utils import get_logger

__all__ = [
    "adjacent_accuracy",
    "classification_metrics",
    "confusion_table",
    "feature_importance_table",
    "roc_curve_points",
]

logger = get_logger(__name__)


def adjacent_accuracy(
    y_true: Sequence[Any], y_pred: Sequence[Any], tolerance: int = 1
) -> float:
    """预测值与真实值相差不超过 ``tolerance`` 档的比例。"""
    truth = np.asarray(y_true, dtype="float64")
    pred = np.asarray(y_pred, dtype="float64")
    if truth.size == 0:
        return float("nan")
    return float((np.abs(truth - pred) <= tolerance).mean())


def _safe_auc(y_true: np.ndarray, y_proba: np.ndarray, labels: np.ndarray) -> float:
    """计算多分类 one-vs-rest 宏平均 AUC，自动适配折内类别缺失。

    时间序列交叉验证的每一折里，测试集通常**只包含全部档位中的一部分**
    （例如某年没有国家处于 CCC 档）。而 ``roc_auc_score`` 在收到 ``labels`` 中
    存在但 ``y_true`` 中没有的类别时会报错。因此这里先求出实际出现的类别，
    再把概率矩阵的对应列抽出来计算——这比直接返回 ``NaN`` 有用得多。
    """
    try:
        present = np.asarray([label for label in labels if label in set(np.unique(y_true))])
        if present.size < 2:
            return float("nan")
        lookup = {value: index for index, value in enumerate(labels)}
        columns = [lookup[value] for value in present]
        block = y_proba[:, columns]
        if present.size == 2:
            positive = block[:, 1]
            if np.unique(y_true).size < 2:
                return float("nan")
            return float(roc_auc_score(y_true, positive))
        return float(
            roc_auc_score(y_true, block, multi_class="ovr", average="macro", labels=present)
        )
    except Exception as exc:
        logger.debug("AUC 计算失败：%s", exc)
        return float("nan")


def classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_proba: np.ndarray | pd.DataFrame | None = None,
    *,
    labels: Sequence[Any] | None = None,
    binary_positive: Any | None = None,
) -> dict[str, float]:
    """计算一整套分类指标。

    Parameters
    ----------
    y_proba:
        概率矩阵（行 = 观测，列顺序须与 ``labels`` 一致）。多分类时用于计算
        宏平均 AUC 与对数损失。
    binary_positive:
        若提供，则把问题二元化为「是否等于该类别」（例如「是否下调」），
        并额外给出该方向的 AUC 与平均精度。
    """
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    result: dict[str, float] = {}

    if truth.size == 0:
        return {
            "n_obs": 0.0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "f1_macro": np.nan,
            "mae_ordinal": np.nan,
            "adjacent_acc_1": np.nan,
            "adjacent_acc_2": np.nan,
        }

    result["n_obs"] = float(truth.size)
    result["accuracy"] = float(accuracy_score(truth, pred))
    result["balanced_accuracy"] = float(balanced_accuracy_score(truth, pred))
    result["f1_macro"] = float(f1_score(truth, pred, average="macro", zero_division=0))
    result["f1_weighted"] = float(f1_score(truth, pred, average="weighted", zero_division=0))
    result["mae_ordinal"] = float(mean_absolute_error(truth.astype(float), pred.astype(float)))
    result["adjacent_acc_1"] = adjacent_accuracy(truth, pred, 1)
    result["adjacent_acc_2"] = adjacent_accuracy(truth, pred, 2)
    try:
        result["quadratic_kappa"] = float(cohen_kappa_score(truth, pred, weights="quadratic"))
    except Exception:
        result["quadratic_kappa"] = float("nan")

    if y_proba is not None:
        proba = np.asarray(y_proba, dtype="float64")
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        resolved_labels = np.asarray(labels) if labels is not None else np.unique(truth)
        if proba.shape[1] == len(resolved_labels):
            result["roc_auc_ovr_macro"] = _safe_auc(truth, proba, resolved_labels)
            try:
                result["log_loss"] = float(log_loss(truth, proba, labels=resolved_labels))
            except Exception:
                result["log_loss"] = float("nan")

        if binary_positive is not None:
            positive_index = int(np.where(resolved_labels == binary_positive)[0][0]) if (
                binary_positive in resolved_labels
            ) else None
            if positive_index is not None:
                binary_truth = (truth == binary_positive).astype(int)
                scores = proba[:, positive_index]
                try:
                    result["roc_auc_binary"] = float(roc_auc_score(binary_truth, scores))
                    result["average_precision_binary"] = float(
                        average_precision_score(binary_truth, scores)
                    )
                except Exception:
                    result["roc_auc_binary"] = float("nan")
                    result["average_precision_binary"] = float("nan")
    return result


def confusion_table(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
    use_grade_labels: bool = True,
    normalize: bool = False,
) -> pd.DataFrame:
    """混淆矩阵，可读性优先（默认用评级符号而非数字标注行列）。"""
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    resolved = list(labels) if labels is not None else sorted(set(truth) | set(pred))
    matrix = confusion_matrix(truth, pred, labels=resolved)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            matrix = np.divide(
                matrix, row_sums, out=np.zeros_like(matrix, dtype="float64"), where=row_sums > 0
            )
    if use_grade_labels:
        names = [score_to_grade(v, "S&P") or str(v) for v in resolved]
    else:
        names = [str(v) for v in resolved]
    table = pd.DataFrame(matrix, index=names, columns=names)
    table.index.name = "actual \\ predicted"
    return table


def feature_importance_table(
    model: Any,
    feature_names: Sequence[str] | None = None,
    *,
    top_n: int | None = None,
) -> pd.DataFrame:
    """提取树模型的特征重要性（内置重要性，非 SHAP）。"""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise AttributeError("该模型没有 feature_importances_ 属性（请使用树模型）")
    names = list(feature_names) if feature_names is not None else [
        f"f{i}" for i in range(len(importances))
    ]
    table = pd.DataFrame({"feature": names, "importance": np.asarray(importances, dtype="float64")})
    table = table.sort_values("importance", ascending=False, ignore_index=True)
    table["importance_pct"] = 100 * table["importance"] / table["importance"].sum()
    if top_n:
        table = table.head(top_n)
    return table


def roc_curve_points(y_true: Sequence[Any], y_score: Sequence[float]) -> pd.DataFrame:
    """二分类 ROC 曲线坐标点（用于绘图）。"""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(np.asarray(y_true), np.asarray(y_score))
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
