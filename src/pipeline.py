"""端到端演示流水线。

用途
----
用**合成示例数据**跑通「数据 → 面板 → 迁移分析 → 驱动因素回归 → 预测模型 →
可解释性 → 事件研究 → 报告」的完整链路，并产出 ``reports/`` 下的图表与表格。

.. warning::
   本流水线默认使用 ``data/sample/*.csv``（**合成演示数据**）。
   产生的任何数值都不构成实证结论。接入真实数据后请改用
   ``--ratings`` / ``--macro`` 参数指向自有数据。

方法要点
--------
* **预测设定**：用 ``t-1`` 期的宏观特征预测 ``t`` 期的**生效评级**
  （``score_in_effect``，即在无评级行动年份沿用上一次评级的序列）。
  使用滞后特征而非同期特征，避免「用同年后验信息预测同年评级」的伪回归。
* **迁移概率**：把有序模型的概率分布按「预测前评级」拆分为
  下调 / 稳定 / 上调三部分，得到可直接解读的方向性概率。
* **交叉验证**：按年前向链式（expanding window），训练集严格早于测试年份。

用法
----
.. code-block:: bash

    python -m src.pipeline              # 使用示例数据
    python -m src.pipeline --no-models  # 跳过较慢的建模步骤
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.event_study import event_study, ratings_to_events
from src.analysis.migration import (
    agency_dispersion,
    build_migration_matrix,
    migration_summary,
    rating_cycle_stats,
    transition_probability_matrix,
    upgrade_downgrade_probabilities,
)
from src.analysis.panel_regression import compare_results, run_driver_regressions
from src.clean.missing import handle_missing, missingness_report
from src.clean.panel import build_country_year_panel, summarize_panel, validate_panel
from src.config import ensure_directories, get_config, get_path, resolve_path
from src.features.macro_features import (
    DEFAULT_FEATURES,
    add_derived_features,
    add_lags,
    build_feature_matrix,
)
from src.features.rating_scale import rating_scale_table
from src.ingest.ratings import load_ratings_csv, load_sample_ratings
from src.models.evaluate import classification_metrics, confusion_table, feature_importance_table
from src.utils.io import dataframe_fingerprint, write_csv, write_json
from src.utils.logging_utils import get_logger
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_importance,
    plot_migration_heatmap,
    plot_probability_series,
    plot_rating_trajectory,
    save_figure,
)

logger = get_logger("pipeline")

__all__ = ["attach_migration_probabilities", "main", "run_demo_pipeline"]

ENTITY_GROUP: tuple[str, ...] = ("country_iso3", "agency")


def _write_table(frame: pd.DataFrame, name: str) -> Path:
    path = get_path("tables") / f"{name}.csv"
    write_csv(frame, path)
    return path


def _section(title: str) -> None:
    logger.info("=" * 78)
    logger.info(title)
    logger.info("=" * 78)


def attach_migration_probabilities(
    oof: pd.DataFrame,
    prob_columns: list[str],
    prior_scores: np.ndarray,
) -> pd.DataFrame:
    """把有序模型的概率分布按「预测前评级」拆分为下调 / 稳定 / 上调概率。

    Parameters
    ----------
    oof:
        含概率列的样本外表。
    prob_columns:
        概率列名（即类别标签的数字字符串，如 ``"7"``、``"12"``）。
    prior_scores:
        预测前的评级分值（即 t-1 期分值），形状与 ``oof`` 行数一致。

    Returns
    -------
    pandas.DataFrame
        在 ``oof`` 上追加 ``p_downgrade`` / ``p_stable`` / ``p_upgrade`` 三列。
    """
    matrix = oof[prob_columns].to_numpy(dtype="float64")
    labels = np.array([float(c) for c in prob_columns])
    prior = np.asarray(prior_scores, dtype="float64")
    valid = ~np.isnan(prior)

    p_down = np.full(len(oof), np.nan)
    p_up = np.full(len(oof), np.nan)
    p_stable = np.full(len(oof), np.nan)
    if valid.any():
        block = matrix[valid]
        reference = prior[valid][:, None]
        p_down[valid] = (block * (labels[None, :] < reference)).sum(axis=1)
        p_up[valid] = (block * (labels[None, :] > reference)).sum(axis=1)
        p_stable[valid] = (block * (labels[None, :] == reference)).sum(axis=1)

    result = oof.copy()
    result["p_downgrade"] = p_down
    result["p_stable"] = p_stable
    result["p_upgrade"] = p_up
    return result


def run_demo_pipeline(
    *,
    ratings_path: str | Path | None = None,
    macro_path: str | Path | None = None,
    run_models: bool = True,
) -> dict[str, Any]:
    """执行完整流水线，返回一份「运行清单」（用于复现记录）。"""
    ensure_directories()
    cfg = get_config()
    manifest: dict[str, Any] = {
        "project": cfg.get("project", {}).get("name"),
        "version": cfg.get("project", {}).get("version"),
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "data_source": "user_provided" if ratings_path else "synthetic_sample",
        "warnings": [],
        "steps": {},
    }
    if ratings_path is None:
        manifest["warnings"].append(
            "本次运行使用合成演示数据（data/sample/ratings_sample.csv）。"
            "所有数值均不构成实证结论，请接入真实评级数据后重新运行。"
        )

    # ---------------------------------------------------------------- 1. 数据
    _section("步骤 1/8 · 载入数据并构建面板")
    ratings = load_ratings_csv(ratings_path) if ratings_path is not None else load_sample_ratings()
    if macro_path is not None:
        macro = pd.read_csv(resolve_path(macro_path), encoding="utf-8-sig")
    else:
        macro = pd.read_csv(get_path("sample") / "macro_sample.csv", encoding="utf-8-sig")

    sample_cfg = cfg.get("sample", {})
    panel = build_country_year_panel(
        ratings,
        macro,
        start_year=int(sample_cfg.get("start_year", 1990)),
        end_year=int(sample_cfg.get("end_year", 2023)),
    )
    panel = add_derived_features(panel, group_cols=ENTITY_GROUP, time_col="year")

    validation = validate_panel(panel)
    _write_table(validation, "panel_validation")
    _write_table(summarize_panel(panel), "panel_coverage")
    _write_table(rating_scale_table(), "rating_scale_1_to_21")
    write_csv(panel, get_path("processed") / "panel_country_year.csv")

    manifest["steps"]["panel"] = {
        "n_rows": len(panel),
        "n_countries": int(panel["country_iso3"].nunique()),
        "n_agencies": int(panel["agency"].nunique()),
        "year_range": [int(panel["year"].min()), int(panel["year"].max())],
        "checks_passed": int(validation["passed"].sum()),
        "checks_total": len(validation),
        "fingerprint": dataframe_fingerprint(panel),
    }
    logger.info(
        "面板构建完成：%d 行，%d 国，%d 机构，校验通过 %d/%d",
        len(panel),
        panel["country_iso3"].nunique(),
        panel["agency"].nunique(),
        int(validation["passed"].sum()),
        len(validation),
    )

    # -------------------------------------------------------- 2. 缺失值诊断
    _section("步骤 2/8 · 缺失值诊断与处理")
    feature_cols = [c for c in DEFAULT_FEATURES if c in panel.columns]
    _write_table(missingness_report(panel), "missingness_report_raw")
    panel, fill_report = handle_missing(panel, feature_cols, strategy="interpolate")
    _write_table(fill_report, "missingness_fill_report")
    manifest["steps"]["missing"] = {
        "columns_treated": len(feature_cols),
        "cells_filled": int(fill_report["n_filled"].sum()),
    }

    # ------------------------------------------------------- 3. 迁移矩阵
    _section("步骤 3/8 · 评级迁移分析")
    matrix = build_migration_matrix(panel, score_is_effective=True)
    _write_table(matrix, "migration_matrix_counts")
    probabilities = transition_probability_matrix(matrix)
    _write_table(probabilities, "migration_matrix_probabilities")
    summary = migration_summary(matrix)
    write_json(summary, get_path("tables") / "migration_summary.json")

    by_year = upgrade_downgrade_probabilities(panel, group_cols=["year"])
    _write_table(by_year, "probability_by_year")
    by_agency = upgrade_downgrade_probabilities(panel, group_cols=["agency"])
    _write_table(by_agency, "probability_by_agency")
    by_bucket = upgrade_downgrade_probabilities(panel, group_cols=["bucket_from"])
    _write_table(by_bucket, "probability_by_rating_bucket")
    _write_table(rating_cycle_stats(panel, group_cols=["agency"]), "rating_cycle_stats")

    dispersion = agency_dispersion(panel)
    _write_table(dispersion, "agency_dispersion")
    _write_table(
        pd.DataFrame(
            {
                "metric": ["n_country_year", "share_split", "mean_range", "max_range"],
                "value": [
                    float(len(dispersion)),
                    float(dispersion["is_split"].mean()) if len(dispersion) else np.nan,
                    float(dispersion["range_score"].mean()) if len(dispersion) else np.nan,
                    float(dispersion["range_score"].max()) if len(dispersion) else np.nan,
                ],
            }
        ),
        "agency_dispersion_summary",
    )

    save_figure(plot_migration_heatmap(probabilities, normalize=True), "migration_matrix_heatmap")
    save_figure(
        plot_probability_series(
            by_year.set_index("year")[["p_upgrade", "p_downgrade", "p_stable"]],
            title="年度评级迁移概率（全样本）",
        ),
        "probability_by_year",
    )
    manifest["steps"]["migration"] = {
        **summary,
        "share_split_ratings": float(dispersion["is_split"].mean()) if len(dispersion) else None,
    }
    logger.info(
        "迁移分析完成：稳定 %.3f / 上调 %.3f / 下调 %.3f",
        summary.get("p_stable", float("nan")),
        summary.get("p_upgrade", float("nan")),
        summary.get("p_downgrade", float("nan")),
    )

    # --------------------------------------------------- 4. 驱动因素回归
    _section("步骤 4/8 · 宏观驱动因素面板回归")
    regressions = run_driver_regressions(panel, feature_cols)
    regression_frames: list[pd.DataFrame] = []
    for target, result in regressions.items():
        table = result.to_frame()
        _write_table(table, f"regression_{target}")
        regression_frames.append(
            table.reset_index().rename(columns={"index": "variable"}).assign(target=target)
        )
    if regression_frames:
        _write_table(pd.concat(regression_frames, ignore_index=True), "regression_all_targets")
        _write_table(compare_results(regressions), "regression_comparison")
    manifest["steps"]["panel_regression"] = {
        target: {
            "nobs": result.nobs,
            "r2": result.r2,
            "r2_within": result.r2_within,
            "method": result.method,
            "top_terms": result.to_frame()
            .assign(abs_coef=lambda d: d["coef"].abs())
            .sort_values("abs_coef", ascending=False)
            .head(5)
            .index.tolist(),
        }
        for target, result in regressions.items()
    }
    logger.info("面板回归完成：%d 个因变量", len(regressions))

    # ------------------------------------------------------- 5. 预测模型
    _section("步骤 5/8 · 预测模型与时间序列交叉验证")
    if not run_models:
        logger.info("已按参数跳过建模步骤")
        manifest["steps"]["models"] = {"skipped": True}
    else:
        model_step = _run_models(panel, feature_cols, manifest)
        manifest["steps"]["models"] = model_step

    # ------------------------------------------------------- 6. 事件研究
    _section("步骤 6/8 · 事件研究框架")
    events = ratings_to_events(panel)
    _write_table(events, "rating_events")
    es_result = event_study(None, events)
    manifest["steps"]["event_study"] = es_result.to_dict()
    manifest["warnings"].append(
        "事件研究使用框架占位：项目未打包高频利差 / 汇率 / 股指数据（多受版权限制）。"
        "请按 docs/methodology.md 接入合法高频数据源后重新运行。"
    )
    logger.info("事件研究：%s（%s）", es_result.status, es_result.message)

    # ------------------------------------------------------- 7. 图表
    _section("步骤 7/8 · 生成图表")
    action_counts = panel.groupby("country_iso3")["has_action"].sum()
    if not action_counts.empty and action_counts.max() > 0:
        top_country = action_counts.sort_values(ascending=False).index[0]
        try:
            save_figure(
                plot_rating_trajectory(panel, country=top_country),
                f"rating_trajectory_{top_country}",
            )
            manifest["steps"]["figures"] = {"trajectory_country": str(top_country)}
        except Exception as exc:
            logger.warning("评级轨迹图生成失败：%s", exc)

    # ------------------------------------------------------- 8. 清单
    _section("步骤 8/8 · 写出运行清单")
    write_json(manifest, get_path("tables") / "run_manifest.json")
    logger.info("运行清单已写出：reports/tables/run_manifest.json")
    return manifest


def _run_models(
    panel: pd.DataFrame,
    feature_cols: list[str],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """建模与可解释性子步骤，独立成函数便于阅读与单元测试。"""
    lagged = add_lags(panel, feature_cols, group_cols=ENTITY_GROUP, time_col="year", lags=(1,))
    lagged = lagged.sort_values([*ENTITY_GROUP, "year"])
    lagged["prior_score"] = lagged.groupby(list(ENTITY_GROUP), dropna=False)[
        "score_in_effect"
    ].shift(1)
    lag_cols = [f"{c}_lag1" for c in feature_cols]

    X, y, meta = build_feature_matrix(lagged, lag_cols, target="score_in_effect")
    years = meta["year"].to_numpy()
    prior_scores = lagged.loc[X.index, "prior_score"].to_numpy(dtype="float64")
    logger.info("建模矩阵：%d 行 × %d 特征（特征为 t-1 期，目标为 t 期生效评级）", *X.shape)

    train_frame = X.copy()
    train_frame["score_in_effect"] = y.to_numpy()
    train_frame["prior_score"] = prior_scores
    train_frame["country_iso3"] = meta["country_iso3"].to_numpy()
    train_frame["agency"] = meta["agency"].to_numpy()
    train_frame["year"] = years
    write_csv(train_frame, get_path("processed") / "model_matrix.csv")

    from src.models.ordinal import OrdinalProbabilityModel, fit_ordinal_logit
    from src.models.timeseries_cv import compare_cv_results, cross_validate_model
    from src.models.tree_models import (
        OrdinalTreeClassifier,
        build_random_forest,
        fit_tree_model,
    )

    cv_kwargs: dict[str, Any] = {"n_splits": 5, "min_train_years": 10, "scheme": "expanding"}
    cv_outputs: list[dict[str, Any]] = [
        cross_validate_model(
            lambda: OrdinalProbabilityModel(distribution="logit"),
            X,
            y,
            years,
            model_name="ordinal_logit",
            **cv_kwargs,
        ),
        cross_validate_model(
            lambda: OrdinalProbabilityModel(distribution="probit"),
            X,
            y,
            years,
            model_name="ordinal_probit",
            **cv_kwargs,
        ),
        cross_validate_model(
            lambda: OrdinalTreeClassifier(backend="random_forest"),
            X,
            y,
            years,
            model_name="random_forest",
            **cv_kwargs,
        ),
    ]
    try:
        cv_outputs.append(
            cross_validate_model(
                lambda: OrdinalTreeClassifier(backend="xgboost"),
                X,
                y,
                years,
                model_name="xgboost",
                **cv_kwargs,
            )
        )
    except Exception as exc:
        logger.warning("XGBoost 交叉验证跳过：%s", exc)
        manifest["warnings"].append(f"XGBoost 未运行：{exc}")

    _write_table(compare_cv_results(cv_outputs), "cv_model_comparison")
    _write_table(cv_outputs[0]["fold_metrics"], "cv_fold_metrics_ordinal_logit")

    prob_columns = [str(c) for c in cv_outputs[0]["classes"]]
    for output in cv_outputs:
        oof = pd.DataFrame(
            {
                "country_iso3": meta["country_iso3"].to_numpy(),
                "agency": meta["agency"].to_numpy(),
                "year": years,
                "prior_score": prior_scores,
                "y_true": y.to_numpy(),
                "y_pred": output["oof_pred"].to_numpy(),
            }
        )
        proba = output["oof_proba"].copy()
        proba.columns = [str(c) for c in output["classes"]]
        aligned = [c for c in prob_columns if c in proba.columns]
        oof = pd.concat([oof, proba[aligned]], axis=1)
        oof = attach_migration_probabilities(oof, aligned, prior_scores)
        _write_table(oof, f"oof_predictions_{output['model']}")

    models_summary: dict[str, Any] = {}
    for output in cv_outputs:
        summary = output["summary"]
        models_summary[output["model"]] = {
            metric: float(summary.loc[metric, "mean"])
            for metric in ("accuracy", "adjacent_acc_1", "mae_ordinal", "p_upgrade", "p_downgrade")
            if metric in summary.index
        }
        if "roc_auc_ovr_macro" in summary.index:
            models_summary[output["model"]]["roc_auc_ovr_macro"] = float(
                summary.loc["roc_auc_ovr_macro", "mean"]
            )

    # ---- 全样本拟合：系数、重要性、SHAP、混淆矩阵 ------------------------
    try:
        ordinal = fit_ordinal_logit(X, y)
        _write_table(ordinal.to_frame(), "ordinal_logit_coefficients")
        _write_table(
            ordinal.to_frame(thresholds=True), "ordinal_logit_coefficients_with_thresholds"
        )
        models_summary["ordinal_logit"]["pseudo_r2"] = float(ordinal.prsquared)
    except Exception as exc:
        logger.warning("有序 Logit 全样本拟合失败：%s", exc)
        manifest["warnings"].append(f"有序 Logit 全样本拟合未完成：{exc}")

    tree_model = None
    try:
        tree_model = fit_tree_model(build_random_forest(), X, y)
        predictions = np.asarray(tree_model.predict(X))
        importance = feature_importance_table(tree_model, list(X.columns), top_n=20)
        _write_table(importance, "rf_feature_importance")
        save_figure(
            plot_importance(importance, title="随机森林特征重要性"), "rf_feature_importance"
        )
        confusion = confusion_table(y, predictions, labels=sorted(pd.unique(y)))
        _write_table(confusion, "rf_confusion_matrix_train")
        save_figure(
            plot_confusion_matrix(confusion, title="随机森林混淆矩阵（训练集）"),
            "rf_confusion_matrix",
        )
        write_json(
            classification_metrics(y, predictions, labels=sorted(pd.unique(y))),
            get_path("tables") / "rf_train_metrics.json",
        )
    except Exception as exc:
        logger.warning("随机森林全样本拟合失败：%s", exc)
        manifest["warnings"].append(f"随机森林全样本拟合未完成：{exc}")

    if tree_model is not None:
        try:
            from src.models.explain import explain_with_shap

            shap_result = explain_with_shap(tree_model, X)
            frames = shap_result.to_frames()
            _write_table(frames["summary"], "shap_summary")
            if shap_result.ok:
                models_summary["shap"] = {
                    "explainer": shap_result.explainer,
                    "top_features": frames["summary"]["feature"].head(8).tolist(),
                }
            else:
                logger.info("SHAP 不可用：%s", shap_result.message)
                manifest["warnings"].append(f"SHAP 未运行：{shap_result.message}")
        except Exception as exc:
            logger.warning("SHAP 分析失败：%s", exc)
            manifest["warnings"].append(f"SHAP 未运行：{exc}")

    return models_summary


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="sovereign-rating-lab 端到端演示流水线",
    )
    parser.add_argument(
        "--ratings",
        type=str,
        default=None,
        help="真实评级数据 CSV 路径（缺省则使用 data/sample/ratings_sample.csv）",
    )
    parser.add_argument(
        "--macro",
        type=str,
        default=None,
        help="宏观面板 CSV 路径（缺省则使用 data/sample/macro_sample.csv）",
    )
    parser.add_argument("--no-models", action="store_true", help="跳过耗时较长的建模与可解释性步骤")
    args = parser.parse_args(argv)

    manifest = run_demo_pipeline(
        ratings_path=args.ratings,
        macro_path=args.macro,
        run_models=not args.no_models,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
