"""面板构建、固定效应回归、预测模型与事件研究的集成测试。

这些测试验证「模块之间能否正确协作」，与三个核心模块的单元测试互补。
所有夹具都在测试内部即时构造，保证完全确定、不依赖网络。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analysis.event_study import (
    abnormal_changes,
    event_study,
    event_study_summary,
    prepare_event_windows,
    ratings_to_events,
)
from src.analysis.panel_regression import (
    fit_fixed_effects,
    fit_logit,
    fit_pooled_ols,
    run_driver_regressions,
)
from src.clean.panel import (
    build_country_year_panel,
    collapse_to_year_end,
    standardize_columns,
    summarize_panel,
    validate_panel,
)
from src.models.evaluate import (
    adjacent_accuracy,
    classification_metrics,
    confusion_table,
    feature_importance_table,
)
from src.models.ordinal import OrdinalProbabilityModel, fit_ordinal_logit, fit_ordinal_probit
from src.models.timeseries_cv import (
    cross_validate_model,
    expanding_year_folds,
    make_forward_chaining_splits,
)
from src.models.tree_models import OrdinalTreeClassifier


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------
@pytest.fixture()
def event_style_ratings() -> pd.DataFrame:
    """事件式评级记录：2001 与 2002 年缺失，用于验证前向填充。"""
    return pd.DataFrame(
        {
            "country_iso3": ["AAA", "AAA", "AAA"],
            "year": [2000, 2003, 2004],
            "agency": ["S&P", "S&P", "S&P"],
            "rating_score": [10.0, 12.0, 10.0],
            "rating": ["BB", "BBB", "BB"],
            "outlook": ["Stable", "Positive", "Negative"],
        }
    )


@pytest.fixture()
def regression_panel() -> pd.DataFrame:
    """带已知数据生成过程的平衡面板，用于回归测试。"""
    rng = np.random.default_rng(2024)
    n_countries, n_years = 14, 12
    records = []
    for country in range(n_countries):
        country_effect = rng.normal(0, 1.0)
        for year in range(n_years):
            x1 = rng.normal(0, 1)
            x2 = rng.normal(0, 1)
            x3 = rng.normal(0, 1)
            noise = rng.normal(0, 0.5)
            y_continuous = country_effect + 0.3 * year + 0.8 * x1 - 0.5 * x2 + noise
            records.append(
                {
                    "country_iso3": f"C{country:02d}",
                    "year": 2000 + year,
                    "x1": x1,
                    "x2": x2,
                    "x3": x3,
                    "y": y_continuous,
                    "y_binary": int(y_continuous < 0),
                }
            )
    return pd.DataFrame.from_records(records)


@pytest.fixture()
def ordinal_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """生成有序多分类数据：类别由潜在变量分箱得到，保证有序性。"""
    rng = np.random.default_rng(7)
    n = 320
    X = pd.DataFrame(
        {
            "macro_a": rng.normal(size=n),
            "macro_b": rng.normal(size=n),
            "macro_c": rng.normal(size=n),
        }
    )
    latent = 1.1 * X["macro_a"] - 0.7 * X["macro_b"] + 0.4 * X["macro_c"] + rng.normal(size=n)
    y = pd.cut(latent, bins=[-10, -0.6, 0.6, 10], labels=[8, 12, 16]).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# 面板构建
# ---------------------------------------------------------------------------
class TestPanelConstruction:
    def test_forward_fill_and_change(self, event_style_ratings: pd.DataFrame) -> None:
        panel = build_country_year_panel(event_style_ratings, None, start_year=2000, end_year=2004)
        assert len(panel) == 5
        assert panel["year"].tolist() == [2000, 2001, 2002, 2003, 2004]

        scores = panel["score_in_effect"].tolist()
        assert scores == [10.0, 10.0, 10.0, 12.0, 10.0]

        has_action = panel["has_action"].tolist()
        assert has_action == [True, False, False, True, True]

        changes = panel["rating_change"].tolist()
        assert math.isnan(changes[0])
        assert changes[1:] == pytest.approx([0.0, 0.0, 2.0, -2.0])

    def test_action_flags(self, event_style_ratings: pd.DataFrame) -> None:
        panel = build_country_year_panel(event_style_ratings, None, start_year=2000, end_year=2004)
        assert panel["is_upgrade"].tolist() == [0, 0, 0, 1, 0]
        assert panel["is_downgrade"].tolist() == [0, 0, 0, 0, 1]
        assert panel["is_stable"].tolist() == [0, 1, 1, 0, 0]
        assert panel["action_type"].tolist()[3] == "upgrade"
        assert panel["action_type"].tolist()[4] == "downgrade"

    def test_carry_forward_disabled(self, event_style_ratings: pd.DataFrame) -> None:
        panel = build_country_year_panel(
            event_style_ratings, None, start_year=2000, end_year=2004, carry_forward=False
        )
        assert panel["score_in_effect"].isna().sum() == 2
        assert panel["rating_change"].isna().all()

    def test_year_grid_covers_full_range(self, event_style_ratings: pd.DataFrame) -> None:
        panel = build_country_year_panel(event_style_ratings, None, start_year=1998, end_year=2006)
        assert panel["year"].min() == 1998
        assert panel["year"].max() == 2006
        assert len(panel) == 9

    def test_invalid_year_range_raises(self, event_style_ratings: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="起始年份"):
            build_country_year_panel(event_style_ratings, None, start_year=2020, end_year=2010)

    def test_collapse_keeps_year_end_action(self) -> None:
        frame = pd.DataFrame(
            {
                "country_iso3": ["AAA", "AAA", "AAA"],
                "agency": ["S&P"] * 3,
                "year": [2005, 2005, 2006],
                "rating_score": [10.0, 12.0, 13.0],
                "action_date": ["2005-03-01", "2005-11-20", "2006-04-01"],
            }
        )
        collapsed = collapse_to_year_end(frame)
        assert len(collapsed) == 2
        assert collapsed.loc[collapsed["year"] == 2005, "rating_score"].iloc[0] == 12.0

    def test_standardize_columns_aliases(self) -> None:
        frame = pd.DataFrame({"ISO3": ["AAA"], "年份": [2000], "机构": ["S&P"], "评级": ["AA"]})
        standardized = standardize_columns(frame)
        assert {"country_iso3", "year", "agency", "rating"}.issubset(standardized.columns)

    def test_validate_panel_detects_duplicates(self, event_style_ratings: pd.DataFrame) -> None:
        panel = build_country_year_panel(event_style_ratings, None, start_year=2000, end_year=2004)
        duplicated = pd.concat([panel, panel.head(1)], ignore_index=True)
        report = validate_panel(duplicated).set_index("check")
        assert report.loc["主键唯一 (country, agency, year)", "passed"] == 0
        assert report.loc["评级分值落在 [1, 21]", "passed"] == 1

    def test_summarize_panel(self, event_style_ratings: pd.DataFrame) -> None:
        panel = build_country_year_panel(event_style_ratings, None, start_year=2000, end_year=2004)
        summary = summarize_panel(panel)
        assert len(summary) == 1
        assert summary["n_obs"].iloc[0] == 5
        assert summary["n_countries"].iloc[0] == 1
        assert summary["n_actions"].iloc[0] == 3  # 2000 / 2003 / 2004 三年有评级行动


# ---------------------------------------------------------------------------
# 回归
# ---------------------------------------------------------------------------
class TestPanelRegression:
    def test_fixed_effects_recovers_signal(self, regression_panel: pd.DataFrame) -> None:
        result = fit_fixed_effects(regression_panel, "y", ["x1", "x2", "x3"], time_effects=True)
        assert result.nobs == len(regression_panel)
        assert {"x1", "x2", "x3"}.issubset(set(result.params.index))
        # 数据生成过程设定 x1 系数 +0.8、x2 系数 -0.5
        assert result.params["x1"] == pytest.approx(0.8, abs=0.25)
        assert result.params["x2"] == pytest.approx(-0.5, abs=0.25)
        assert result.params["x1"] > 0 > result.params["x2"]
        assert 0.0 <= result.r2 <= 1.0
        assert result.cluster == "country_iso3"

    def test_result_export(self, regression_panel: pd.DataFrame) -> None:
        result = fit_fixed_effects(regression_panel, "y", ["x1", "x2"])
        frame = result.to_frame()
        assert {"coef", "std_err", "t", "p_value", "signif", "ci_lower", "ci_upper"}.issubset(
            frame.columns
        )
        assert frame.loc["x1", "ci_lower"] < frame.loc["x1", "coef"] < frame.loc["x1", "ci_upper"]
        assert "观测数" in result.summary_text()

    def test_pooled_ols(self, regression_panel: pd.DataFrame) -> None:
        result = fit_pooled_ols(regression_panel, "y", ["x1", "x2"])
        assert "const" in result.params.index
        assert result.nobs == len(regression_panel)

    def test_cluster_by_alternative_dimension(self, regression_panel: pd.DataFrame) -> None:
        result = fit_fixed_effects(
            regression_panel, "y", ["x1", "x2"], cluster_col="year", time_effects=False
        )
        assert result.cluster == "year"
        assert result.nobs == len(regression_panel)

    def test_run_driver_regressions_multi_target(self, regression_panel: pd.DataFrame) -> None:
        panel = regression_panel.rename(columns={"y": "rating_change", "y_binary": "is_downgrade"})
        panel["rating_score"] = 10.0
        results = run_driver_regressions(
            panel, ["x1", "x2"], targets=("rating_change", "is_downgrade")
        )
        assert set(results) == {"rating_change", "is_downgrade"}

    def test_logit_on_binary(self, regression_panel: pd.DataFrame) -> None:
        result = fit_logit(regression_panel, "y_binary", ["x1", "x2"])
        assert result.nobs > 0
        assert "x1" in result.params.index

    def test_missing_columns_raise(self, regression_panel: pd.DataFrame) -> None:
        with pytest.raises(KeyError):
            fit_fixed_effects(regression_panel, "y", ["not_a_column"])

    def test_empty_after_cleaning_raises(self) -> None:
        frame = pd.DataFrame(
            {
                "country_iso3": ["A", "B"],
                "year": [2000, 2001],
                "y": [np.nan, np.nan],
                "x": [1.0, 2.0],
            }
        )
        with pytest.raises(ValueError, match="无可用观测"):
            fit_fixed_effects(frame, "y", ["x"])


# ---------------------------------------------------------------------------
# 有序模型与交叉验证
# ---------------------------------------------------------------------------
class TestOrdinalModels:
    def test_logit_fit_and_predict(self, ordinal_dataset) -> None:
        X, y = ordinal_dataset
        result = fit_ordinal_logit(X, y)
        assert result.nobs == len(X)
        assert len(result.categories) == 3
        assert result.prsquared > 0

        proba = result.predict_proba(X)
        assert proba.shape == (len(X), 3)
        assert np.allclose(proba.to_numpy().sum(axis=1), 1.0, atol=1e-6)
        assert list(proba.columns) == list(result.categories)

        predictions = result.predict(X)
        assert set(np.unique(predictions)).issubset(set(result.categories))

    def test_probit_runs(self, ordinal_dataset) -> None:
        X, y = ordinal_dataset
        result = fit_ordinal_probit(X, y)
        assert result.nobs == len(X)
        assert result.distribution == "probit"

    def test_coefficient_table(self, ordinal_dataset) -> None:
        X, y = ordinal_dataset
        result = fit_ordinal_logit(X, y)
        table = result.to_frame()
        assert set(table.index) == set(X.columns)
        with_thresholds = result.to_frame(thresholds=True)
        assert len(with_thresholds) >= len(table)

    def test_constant_column_is_dropped(self, ordinal_dataset) -> None:
        X, y = ordinal_dataset
        with_const = X.assign(const=1.0)
        result = fit_ordinal_logit(with_const, y)
        assert "const" not in result.feature_names
        assert result.nobs == len(X)

    def test_sklearn_compatible_wrapper(self, ordinal_dataset) -> None:
        X, y = ordinal_dataset
        clf = OrdinalProbabilityModel(distribution="logit").fit(X, y)
        assert list(clf.classes_) == sorted(y.unique())
        assert clf.predict_proba(X).shape == (len(X), len(clf.classes_))
        assert len(clf.predict(X)) == len(X)
        assert "观测数" in clf.summary_text()

    def test_wrapper_requires_fit(self, ordinal_dataset) -> None:
        X, _ = ordinal_dataset
        with pytest.raises(RuntimeError):
            OrdinalProbabilityModel().predict_proba(X)


class TestTimeSeriesCrossValidation:
    def test_folds_never_use_future(self) -> None:
        years = pd.Series(sorted(list(range(2000, 2015)) * 3))
        folds = expanding_year_folds(years, n_splits=4, min_train_years=6)
        assert len(folds) == 4
        for fold in folds:
            assert max(fold.train_years) < min(fold.test_years)

    def test_rolling_scheme_caps_window(self) -> None:
        years = pd.Series(sorted(list(range(2000, 2015)) * 3))
        folds = expanding_year_folds(
            years, n_splits=3, min_train_years=5, scheme="rolling", rolling_window=5
        )
        for fold in folds:
            assert len(fold.train_years) <= 5

    def test_insufficient_years_raises(self) -> None:
        with pytest.raises(ValueError, match="年份数量不足"):
            expanding_year_folds(pd.Series([2000, 2001, 2002]), n_splits=3, min_train_years=5)

    def test_forward_chaining_helper(self) -> None:
        years = pd.Series(sorted(list(range(2000, 2016)) * 2))
        splits = make_forward_chaining_splits(years, n_splits=3, min_train_years=6)
        assert all(
            isinstance(train, np.ndarray) and isinstance(test, np.ndarray) for train, test in splits
        )

    def test_cross_validate_random_forest(self, regression_panel: pd.DataFrame) -> None:
        from sklearn.ensemble import RandomForestClassifier

        # 用回归面板构造一个有序分类目标，保证有足够年份与样本
        panel = regression_panel.copy()
        panel["target"] = pd.cut(panel["y"], bins=[-10, -0.5, 0.5, 10], labels=[1, 2, 3]).astype(
            int
        )
        X = panel[["x1", "x2", "x3"]]
        y = panel["target"]
        years = panel["year"]

        output = cross_validate_model(
            lambda: RandomForestClassifier(n_estimators=15, random_state=0),
            X,
            y,
            years,
            n_splits=3,
            min_train_years=6,
            model_name="rf_smoke",
        )
        assert output["model"] == "rf_smoke"
        assert len(output["fold_metrics"]) == 3
        assert {"accuracy", "adjacent_acc_1", "mae_ordinal"}.issubset(
            output["fold_metrics"].columns
        )
        assert 0.0 <= output["summary"].loc["accuracy", "mean"] <= 1.0
        # 样本外预测应覆盖测试集的行，训练期行保持缺失
        assert output["oof_pred"].notna().sum() > 0
        assert output["oof_proba"].shape[1] == len(output["classes"])

    def test_cross_validate_ordinal_smoke(self, ordinal_dataset) -> None:
        X, y = ordinal_dataset
        years = pd.Series(np.repeat(np.arange(2000, 2016), 20))
        output = cross_validate_model(
            lambda: OrdinalProbabilityModel("logit"),
            X,
            y,
            years,
            n_splits=2,
            min_train_years=8,
            model_name="ordinal_smoke",
        )
        assert len(output["fold_metrics"]) == 2
        assert output["summary"].loc["adjacent_acc_1", "mean"] >= 0.0


# ---------------------------------------------------------------------------
# 评估指标
# ---------------------------------------------------------------------------
class TestEvaluation:
    def test_perfect_prediction(self) -> None:
        y = np.array([1, 2, 3, 4])
        metrics = classification_metrics(y, y)
        assert metrics["accuracy"] == 1.0
        assert metrics["mae_ordinal"] == 0.0
        assert metrics["adjacent_acc_1"] == 1.0

    def test_adjacent_accuracy_tolerance(self) -> None:
        y_true = np.array([10, 10, 10, 10])
        y_pred = np.array([10, 11, 13, 5])
        assert adjacent_accuracy(y_true, y_pred, 0) == pytest.approx(0.25)
        assert adjacent_accuracy(y_true, y_pred, 1) == pytest.approx(0.5)
        assert adjacent_accuracy(y_true, y_pred, 3) == pytest.approx(0.75)

    def test_metrics_with_probabilities(self) -> None:
        rng = np.random.default_rng(3)
        y = rng.integers(1, 4, size=200)
        proba = np.full((200, 3), 1 / 3)
        metrics = classification_metrics(y, y, proba, labels=[1, 2, 3])
        assert metrics["log_loss"] == pytest.approx(math.log(3), abs=1e-6)
        assert "roc_auc_ovr_macro" in metrics

    def test_empty_input(self) -> None:
        metrics = classification_metrics([], [])
        assert metrics["n_obs"] == 0
        assert math.isnan(metrics["accuracy"])

    def test_confusion_table_labels(self) -> None:
        table = confusion_table([12, 12, 13], [12, 13, 13])
        assert "BBB-" in table.index
        assert "BBB" in table.columns
        assert table.loc["BBB-", "BBB-"] == 1

    def test_confusion_table_normalized(self) -> None:
        table = confusion_table([12, 12], [12, 13], normalize=True)
        assert table.to_numpy().sum() == pytest.approx(1.0)

    def test_feature_importance_table(self) -> None:
        class Dummy:
            feature_importances_ = np.array([0.1, 0.6, 0.3])

        table = feature_importance_table(Dummy(), ["a", "b", "c"])
        assert table["feature"].tolist() == ["b", "c", "a"]
        assert table["importance_pct"].sum() == pytest.approx(100.0)

    def test_feature_importance_requires_tree(self) -> None:
        with pytest.raises(AttributeError):
            feature_importance_table(object(), ["a"])

    def test_auc_tolerates_classes_missing_from_y_true(self) -> None:
        """时间序列交叉验证的折内往往缺少部分评级档，AUC 仍应可算。"""
        y_true = np.array([12, 12, 13, 13])
        proba = np.array(
            [
                [0.6, 0.3, 0.1],
                [0.5, 0.4, 0.1],
                [0.2, 0.5, 0.3],
                [0.1, 0.6, 0.3],
            ]
        )
        metrics = classification_metrics(y_true, y_true, proba, labels=[11, 12, 13])
        assert not math.isnan(metrics["roc_auc_ovr_macro"])

    def test_auc_allowed_when_single_class_present(self) -> None:
        y_true = np.array([12, 12, 12])
        proba = np.full((3, 3), 1 / 3)
        metrics = classification_metrics(y_true, y_true, proba, labels=[11, 12, 13])
        assert math.isnan(metrics["roc_auc_ovr_macro"])
        assert metrics["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# 树模型封装（标签编码）
# ---------------------------------------------------------------------------
class TestOrdinalTreeClassifier:
    """验证 ``OrdinalTreeClassifier`` 能把 1-21 的非零起始标签正确喂给 XGBoost。

    这是流水线早期出现过的真实缺陷：绕过封装直接把 1-21 的标签交给 XGBoost 会
    抛出 ``Invalid classes inferred from unique values of y``。
    """

    def test_random_forest_backend(self, ordinal_dataset) -> None:
        X, y = ordinal_dataset
        clf = OrdinalTreeClassifier(backend="random_forest", n_estimators=15, random_state=0).fit(
            X, y
        )
        assert list(clf.classes_) == sorted(y.unique())
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), len(clf.classes_))
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert set(clf.predict(X)).issubset(set(y.unique()))
        assert clf.feature_importances_.shape == (X.shape[1],)

    def test_xgboost_backend(self, ordinal_dataset) -> None:
        pytest.importorskip("xgboost")
        X, y = ordinal_dataset
        clf = OrdinalTreeClassifier(
            backend="xgboost", n_estimators=20, max_depth=3, random_state=0
        ).fit(X, y)
        assert list(clf.classes_) == sorted(y.unique())
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), len(clf.classes_))
        assert set(clf.predict(X)).issubset(set(y.unique()))
        assert clf.estimator is not None

    def test_labels_may_start_above_zero(self) -> None:
        """标签为 [8, 12, 16]（非 0 起始且不连续）时也必须正常工作。"""
        rng = np.random.default_rng(1)
        X = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120)})
        y = pd.Series(np.tile([8, 12, 16], 40))
        clf = OrdinalTreeClassifier(backend="random_forest", n_estimators=10).fit(X, y)
        assert list(clf.classes_) == [8, 12, 16]
        assert set(clf.predict(X)).issubset({8, 12, 16})

    def test_invalid_backend_raises(self, ordinal_dataset) -> None:
        X, y = ordinal_dataset
        with pytest.raises(ValueError, match="不支持的 backend"):
            OrdinalTreeClassifier(backend="knn").fit(X, y)

    def test_predict_before_fit_raises(self, ordinal_dataset) -> None:
        X, _ = ordinal_dataset
        with pytest.raises(RuntimeError):
            OrdinalTreeClassifier().predict_proba(X)


# ---------------------------------------------------------------------------
# 事件研究
# ---------------------------------------------------------------------------
class TestEventStudy:
    @pytest.fixture()
    def synthetic_prices(self) -> pd.DataFrame:
        rng = np.random.default_rng(11)
        dates = pd.bdate_range("2017-01-02", "2020-12-31")
        market = rng.normal(0.0002, 0.010, len(dates))
        returns = 0.7 * market + rng.normal(0.0, 0.008, len(dates))
        return pd.DataFrame(
            {
                "country_iso3": "AAA",
                "date": dates,
                "value": 100 * np.exp(np.cumsum(returns)),
                "market": 100 * np.exp(np.cumsum(market)),
            }
        )

    @pytest.fixture()
    def single_event(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "country_iso3": ["AAA"],
                "event_date": pd.to_datetime(["2019-06-28"]),
                "event_type": ["downgrade"],
            }
        )

    def test_missing_prices_degrades_gracefully(self, single_event: pd.DataFrame) -> None:
        result = event_study(None, single_event)
        assert result.status == "insufficient_data"
        assert not result.ok
        assert "利差" in result.message or "行情" in result.message
        assert result.abnormal.empty

    def test_missing_events_degrades_gracefully(self, synthetic_prices: pd.DataFrame) -> None:
        result = event_study(synthetic_prices, pd.DataFrame())
        assert result.status == "insufficient_data"

    def test_unmatched_data_degrades_gracefully(self, synthetic_prices: pd.DataFrame) -> None:
        foreign = pd.DataFrame(
            {
                "country_iso3": ["ZZZ"],
                "event_date": pd.to_datetime(["2019-06-28"]),
                "event_type": ["downgrade"],
            }
        )
        result = event_study(synthetic_prices, foreign)
        assert result.status == "insufficient_data"

    def test_windows_are_built(self, synthetic_prices, single_event) -> None:
        windows = prepare_event_windows(synthetic_prices, single_event, window=(-20, 20))
        assert not windows.empty
        # 默认把估计窗口 [-120, -21] 一并抽取出来，否则无法估计市场模型
        assert windows["relative_day"].min() == -120
        assert windows["relative_day"].max() == 20
        event_only = windows[windows["relative_day"].between(-20, 20)]
        assert len(event_only) == 41

    def test_windows_without_estimation_window(self, synthetic_prices, single_event) -> None:
        windows = prepare_event_windows(
            synthetic_prices, single_event, window=(-20, 20), estimation_window=None
        )
        assert len(windows) == 41
        assert windows["relative_day"].min() == -20

    def test_full_pipeline(self, synthetic_prices, single_event) -> None:
        result = event_study(synthetic_prices, single_event, window=(-20, 20))
        assert result.ok
        assert set(result.abnormal["model"].unique()).issubset({"market_model", "mean_adjusted"})
        assert "car" in result.abnormal.columns
        summary = event_study_summary(result.abnormal)
        assert len(summary) == 41
        assert summary["caar"].iloc[-1] == pytest.approx(summary["mean_abnormal"].sum())

    def test_abnormal_changes_column_contract(self, synthetic_prices, single_event) -> None:
        windows = prepare_event_windows(synthetic_prices, single_event)
        abnormal = abnormal_changes(windows)
        assert {
            "event_id",
            "relative_day",
            "actual",
            "expected",
            "abnormal",
            "car",
            "model",
        }.issubset(abnormal.columns)
        # 切片的第一日缺前一日价格，差分应为缺失
        first_day = abnormal["relative_day"].min()
        assert pd.isna(abnormal.loc[abnormal["relative_day"] == first_day, "actual"]).all()
        # 事件窗口内的观测应有非缺失的异常收益
        event_window = abnormal[abnormal["relative_day"].between(-20, 20)]
        assert event_window["abnormal"].notna().sum() > 0

    def test_ratings_to_events(self) -> None:
        panel = pd.DataFrame(
            {
                "country_iso3": ["AAA", "AAA", "BBB"],
                "agency": ["S&P"] * 3,
                "year": [2010, 2011, 2010],
                "action_date": pd.to_datetime(["2010-04-01", "2011-06-01", "2010-09-01"]),
                "rating_change": [0.0, -2.0, 1.0],
            }
        )
        events = ratings_to_events(panel)
        assert len(events) == 2
        assert set(events["event_type"]) == {"upgrade", "downgrade"}
        assert "event_date" in events.columns

    def test_ratings_to_events_requires_change_column(self) -> None:
        with pytest.raises(KeyError):
            ratings_to_events(pd.DataFrame({"country_iso3": ["AAA"]}))
