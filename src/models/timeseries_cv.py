"""时间序列交叉验证（前向链式 / 扩张窗口）。

为什么不能用随机 K 折
---------------------
评级数据同时具有**时间序列**与**面板**结构。随机切分会让模型在训练集中「看到未来」，
产生严重乐观偏差：模型可以记住某国在 2015 年的评级，然后在测试集里"预测"同一国家
2013 年的评级。正确做法是**按时间前向切分**：训练集只包含测试年份之前的数据。

两种方案
--------
* ``expanding``（默认）：训练窗口随时间扩张，模拟「用全部历史预测下一年」。
* ``rolling``：训练窗口固定长度，模拟「只用近 N 年数据」，对结构变化更敏感。

面板结构处理
------------
默认按「年份」整体切分（所有国家同年一起进入测试集），这避免了同一横截面内部
的信息泄漏；如需按「国家×年份」交错切分，可传入 ``group_by``。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.models.evaluate import classification_metrics
from src.utils.logging_utils import get_logger

__all__ = [
    "TimeSeriesFold",
    "compare_cv_results",
    "cross_validate_model",
    "cross_validate_ordinal",
    "expanding_year_folds",
    "make_forward_chaining_splits",
]

logger = get_logger(__name__)


@dataclass
class TimeSeriesFold:
    """一个时间切分折。"""

    fold: int
    train_years: list[int]
    test_years: list[int]
    train_index: np.ndarray
    test_index: np.ndarray

    @property
    def name(self) -> str:
        return f"fold{self.fold}: train<={max(self.train_years)} test={self.test_years}"

    def __len__(self) -> int:  # pragma: no cover - 便捷
        return len(self.test_index)


def expanding_year_folds(
    years: Sequence[int] | pd.Series,
    *,
    n_splits: int = 5,
    min_train_years: int = 8,
    scheme: str = "expanding",
    test_years_per_fold: int = 1,
    rolling_window: int | None = None,
) -> list[TimeSeriesFold]:
    """构造按年份前向切分的折列表。

    Parameters
    ----------
    years:
        与样本等长的年份序列（或索引）。
    n_splits:
        折数。测试集取样本末尾的 ``n_splits`` 个年份块。
    min_train_years:
        训练集至少包含的**不同年份数**；不满足的折会被跳过（记录警告）。
    scheme:
        ``"expanding"`` 或 ``"rolling"``。
    test_years_per_fold:
        每折测试集包含的年份数。
    rolling_window:
        ``scheme="rolling"`` 时训练窗口的年份数，默认与 ``min_train_years`` 相同。

    Returns
    -------
    list[TimeSeriesFold]
    """
    if scheme not in {"expanding", "rolling"}:
        raise ValueError(f"scheme 只能为 'expanding' 或 'rolling'，收到 {scheme!r}")

    series = pd.Series(np.asarray(years))
    unique_years = sorted(pd.unique(series.dropna().astype(int)))
    if len(unique_years) < min_train_years + 1:
        raise ValueError(
            f"年份数量不足：仅 {len(unique_years)} 个年份，"
            f"至少需要 {min_train_years + 1} 个才能构造前向切分"
        )

    blocks: list[list[int]] = []
    cursor = len(unique_years)
    for _ in range(n_splits):
        start = cursor - test_years_per_fold
        if start < min_train_years:
            break
        blocks.append(unique_years[start:cursor])
        cursor = start
    blocks.reverse()

    folds: list[TimeSeriesFold] = []
    for fold_id, test_block in enumerate(blocks, start=1):
        test_start = min(test_block)
        if scheme == "expanding":
            train_years = [y for y in unique_years if y < test_start]
        else:
            window = rolling_window or min_train_years
            train_years = [y for y in unique_years if y < test_start][-window:]
        if len(train_years) < min_train_years:
            logger.warning("跳过 fold%d：训练年份仅 %d 个", fold_id, len(train_years))
            continue
        train_index = series[series.isin(train_years)].index.to_numpy()
        test_index = series[series.isin(test_block)].index.to_numpy()
        if len(test_index) == 0 or len(train_index) == 0:
            continue
        folds.append(
            TimeSeriesFold(
                fold=len(folds) + 1,
                train_years=list(train_years),
                test_years=list(test_block),
                train_index=train_index,
                test_index=test_index,
            )
        )
    return folds


def make_forward_chaining_splits(
    years: Sequence[int] | pd.Series, **kwargs: Any
) -> list[tuple[np.ndarray, np.ndarray]]:
    """返回 ``(train_index, test_index)`` 元组列表，便于直接传入既有流程。"""
    return [(f.train_index, f.test_index) for f in expanding_year_folds(years, **kwargs)]


def _align_probabilities(
    proba: np.ndarray, model_classes: np.ndarray, global_classes: np.ndarray
) -> np.ndarray:
    """把折内概率矩阵对齐到全局类别集合（缺失类别填 0）。"""
    aligned = np.zeros((proba.shape[0], len(global_classes)), dtype="float64")
    lookup = {value: i for i, value in enumerate(global_classes)}
    for source, value in enumerate(model_classes):
        target = lookup.get(value)
        if target is not None:
            aligned[:, target] = proba[:, source]
    row_sums = aligned.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        aligned = np.divide(
            aligned, row_sums, out=np.zeros_like(aligned), where=row_sums > 0
        )
    return aligned


def cross_validate_model(
    model_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: Sequence[Any],
    years: Sequence[int] | pd.Series,
    *,
    folds: list[TimeSeriesFold] | None = None,
    n_splits: int = 5,
    min_train_years: int = 8,
    scheme: str = "expanding",
    needs_proba: bool = True,
    model_name: str = "model",
) -> dict[str, Any]:
    """对任意 scikit-learn 兼容模型执行时间序列交叉验证。

    Returns
    -------
    dict
        键 ``model`` / ``fold_metrics``（每折一行的 ``DataFrame``）/
        ``summary``（各指标均值标准差）/ ``oof_pred`` / ``oof_proba`` /
        ``oof_index`` / ``folds``。
    """
    # 统一使用位置索引，避免调用方传入的索引出现重复标签导致 .loc 返回多行
    X = X.reset_index(drop=True)
    y_series = pd.Series(np.asarray(y)).reset_index(drop=True)
    years_series = pd.Series(np.asarray(years)).reset_index(drop=True)

    if folds is None:
        folds = expanding_year_folds(
            years_series, n_splits=n_splits, min_train_years=min_train_years, scheme=scheme
        )
    if not folds:
        raise ValueError("未能构造任何交叉验证折，请放宽 min_train_years 或增加年份")

    global_classes = np.sort(pd.unique(y_series.dropna()))
    oof_pred = pd.Series(np.nan, index=X.index, dtype="float64", name="oof_pred")
    oof_proba = pd.DataFrame(
        0.0, index=X.index, columns=[str(c) for c in global_classes], dtype="float64"
    )
    records: list[dict[str, Any]] = []

    for fold in folds:
        y_train = y_series.loc[fold.train_index].to_numpy()
        y_test = y_series.loc[fold.test_index].to_numpy()
        X_train = X.loc[fold.train_index]
        X_test = X.loc[fold.test_index]

        model = model_factory()
        model.fit(X_train, y_train)
        pred = np.asarray(model.predict(X_test))
        proba = None
        if needs_proba and hasattr(model, "predict_proba"):
            raw = np.asarray(model.predict_proba(X_test), dtype="float64")
            classes = np.asarray(getattr(model, "classes_", global_classes))
            proba = _align_probabilities(raw, classes, global_classes)

        metrics = classification_metrics(y_test, pred, proba, labels=global_classes)
        metrics.update(
            {
                "fold": fold.fold,
                "train_n": len(fold.train_index),
                "test_n": len(fold.test_index),
                "train_years": f"{min(fold.train_years)}-{max(fold.train_years)}",
                "test_years": ",".join(str(t) for t in fold.test_years),
            }
        )
        records.append(metrics)
        oof_pred.loc[fold.test_index] = pred
        if proba is not None:
            oof_proba.loc[fold.test_index] = proba
        logger.info(
            "fold%d 完成：accuracy=%.4f | adjacent±1=%.4f",
            fold.fold,
            metrics.get("accuracy", float("nan")),
            metrics.get("adjacent_acc_1", float("nan")),
        )

    fold_metrics = pd.DataFrame(records)
    metric_cols = [
        c
        for c in (
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "mae_ordinal",
            "adjacent_acc_1",
            "adjacent_acc_2",
            "quadratic_kappa",
            "roc_auc_ovr_macro",
            "log_loss",
        )
        if c in fold_metrics.columns
    ]
    summary = pd.DataFrame(
        {
            "mean": fold_metrics[metric_cols].mean(),
            "std": fold_metrics[metric_cols].std(),
            "min": fold_metrics[metric_cols].min(),
            "max": fold_metrics[metric_cols].max(),
        }
    )
    summary.attrs["model"] = model_name
    return {
        "model": model_name,
        "fold_metrics": fold_metrics,
        "summary": summary,
        "oof_pred": oof_pred,
        "oof_proba": oof_proba,
        "oof_index": X.index,
        "classes": global_classes,
        "folds": folds,
    }


def cross_validate_ordinal(
    X: pd.DataFrame,
    y: Sequence[Any],
    years: Sequence[int] | pd.Series,
    *,
    distribution: str = "logit",
    n_splits: int = 5,
    min_train_years: int = 8,
    scheme: str = "expanding",
) -> dict[str, Any]:
    """有序 Logit / Probit 的时间序列交叉验证（便捷封装）。"""
    from src.models.ordinal import OrdinalProbabilityModel

    def factory() -> Any:
        return OrdinalProbabilityModel(distribution=distribution)

    return cross_validate_model(
        factory,
        X,
        y,
        years,
        n_splits=n_splits,
        min_train_years=min_train_years,
        scheme=scheme,
        model_name=f"ordinal_{distribution}",
    )


@dataclass
class ComparisonResult:
    """多模型交叉验证的对比容器（保留原始结果以便进一步分析）。"""

    table: pd.DataFrame
    details: dict[str, Any] = field(default_factory=dict)

    def best_by(self, metric: str = "adjacent_acc_1") -> str:
        """返回在指定指标上均值最高的模型名。"""
        subset = self.table[self.table["metric"] == metric]
        if subset.empty:
            raise KeyError(f"对比表中没有指标 {metric!r}")
        return str(subset.loc[subset["mean"].idxmax(), "model"])


def compare_cv_results(results: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """把多个模型的 CV 摘要合并为长表（``model`` / ``metric`` / ``mean`` / ``std``）。"""
    frames: list[pd.DataFrame] = []
    for result in results:
        summary = result["summary"].reset_index()
        summary = summary.rename(columns={summary.columns[0]: "metric"})
        frames.append(
            pd.DataFrame(
                {
                    "model": result["model"],
                    "metric": summary["metric"],
                    "mean": summary["mean"],
                    "std": summary["std"],
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["model", "metric", "mean", "std"])
    return pd.concat(frames, ignore_index=True)
