"""缺失值处理测试：诊断报告、组内插值边界、各策略行为与处理痕迹。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clean.missing import (
    handle_missing,
    impute_panel,
    interpolate_within_group,
    missingness_report,
    stale_rating_flags,
)


@pytest.fixture()
def panel_with_na() -> pd.DataFrame:
    """两组各 5 年的面板：Alpha 有内部缺口，Beta 有开头与内部缺口。"""
    return pd.DataFrame(
        {
            "country_iso3": ["Alpha"] * 5 + ["Beta"] * 5,
            "year": [2000, 2001, 2002, 2003, 2004] * 2,
            "gdp_growth": [1.0, np.nan, 3.0, np.nan, 5.0, np.nan, np.nan, 4.0, np.nan, 6.0],
            "inflation": [2.0, 2.1, 2.2, 2.3, 2.4, 3.0, 3.1, np.nan, 3.3, 3.4],
        }
    )


class TestMissingnessReport:
    def test_counts_and_percentages(self, panel_with_na: pd.DataFrame) -> None:
        report = missingness_report(panel_with_na)
        by_column = report.set_index("column")
        assert by_column.loc["gdp_growth", "n_missing"] == 5
        assert by_column.loc["gdp_growth", "pct_missing"] == pytest.approx(50.0)
        assert by_column.loc["year", "n_missing"] == 0
        assert by_column.loc["year", "pct_missing"] == pytest.approx(0.0)

    def test_sorted_by_missing_share_descending(self, panel_with_na: pd.DataFrame) -> None:
        report = missingness_report(panel_with_na)
        assert report["pct_missing"].is_monotonic_decreasing

    def test_group_level_counts(self, panel_with_na: pd.DataFrame) -> None:
        report = missingness_report(panel_with_na, group_cols=["country_iso3"])
        row = report.set_index("column").loc["gdp_growth"]
        assert row["n_groups_total"] == 2
        assert row["n_groups_with_missing"] == 2

    def test_empty_frame(self) -> None:
        report = missingness_report(pd.DataFrame())
        assert report.empty

    def test_no_missing_data(self) -> None:
        frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        report = missingness_report(frame)
        assert report["n_missing"].sum() == 0
        assert bool(report["pct_missing"].eq(0.0).all())


class TestInterpolateWithinGroup:
    def test_interior_gap_is_filled(self, panel_with_na: pd.DataFrame) -> None:
        filled = interpolate_within_group(
            panel_with_na, ["gdp_growth"], group_cols=["country_iso3"], time_col="year"
        )
        alpha = filled[filled["country_iso3"] == "Alpha"].reset_index(drop=True)
        assert alpha.loc[1, "gdp_growth"] == pytest.approx(2.0)
        assert alpha.loc[3, "gdp_growth"] == pytest.approx(4.0)

    def test_leading_missing_is_not_backfilled(self, panel_with_na: pd.DataFrame) -> None:
        """序列开头的缺失属于结构性缺失，默认不回填。"""
        filled = interpolate_within_group(
            panel_with_na, ["gdp_growth"], group_cols=["country_iso3"], time_col="year"
        )
        beta = filled[filled["country_iso3"] == "Beta"].reset_index(drop=True)
        assert np.isnan(beta.loc[0, "gdp_growth"])
        assert np.isnan(beta.loc[1, "gdp_growth"])

    def test_interpolation_does_not_cross_groups(self) -> None:
        """A 组的数值绝不能用于填补 B 组。"""
        frame = pd.DataFrame(
            {
                "country_iso3": ["A", "A", "A", "B", "B", "B"],
                "year": [2000, 2001, 2002] * 2,
                "value": [10.0, np.nan, 30.0, np.nan, np.nan, 300.0],
            }
        )
        filled = interpolate_within_group(frame, ["value"], group_cols=["country_iso3"])
        group_a = filled[filled["country_iso3"] == "A"].reset_index(drop=True)
        assert group_a.loc[1, "value"] == pytest.approx(20.0)
        group_b = filled[filled["country_iso3"] == "B"].reset_index(drop=True)
        assert np.isnan(group_b.loc[0, "value"])
        assert np.isnan(group_b.loc[1, "value"])
        assert group_b.loc[2, "value"] == pytest.approx(300.0)

    def test_limit_caps_consecutive_fills(self) -> None:
        frame = pd.DataFrame(
            {
                "country_iso3": ["A"] * 6,
                "year": [2000, 2001, 2002, 2003, 2004, 2005],
                "value": [0.0, np.nan, np.nan, np.nan, np.nan, 50.0],
            }
        )
        filled_limit2 = interpolate_within_group(frame, ["value"], limit=2)
        assert filled_limit2["value"].isna().sum() == 2

        filled_unlimited = interpolate_within_group(frame, ["value"], limit=None)
        assert filled_unlimited["value"].isna().sum() == 0

    def test_imputed_flags(self, panel_with_na: pd.DataFrame) -> None:
        filled = interpolate_within_group(panel_with_na, ["gdp_growth"])
        assert "gdp_growth_imputed" in filled.columns
        # Alpha 的 2 个内部缺口 + Beta 的 1 个内部缺口 = 3
        assert int(filled["gdp_growth_imputed"].sum()) == 3
        # 未被填补的位置不应被误标
        assert not filled.loc[filled["gdp_growth"].isna(), "gdp_growth_imputed"].any()

    def test_imputed_flag_can_be_disabled(self, panel_with_na: pd.DataFrame) -> None:
        filled = interpolate_within_group(panel_with_na, ["gdp_growth"], add_flags=False)
        assert "gdp_growth_imputed" not in filled.columns

    def test_row_count_and_columns_preserved(self, panel_with_na: pd.DataFrame) -> None:
        filled = interpolate_within_group(panel_with_na, ["gdp_growth", "inflation"])
        assert len(filled) == len(panel_with_na)
        assert set(panel_with_na.columns).issubset(set(filled.columns))

    def test_missing_column_raises(self, panel_with_na: pd.DataFrame) -> None:
        with pytest.raises(KeyError):
            interpolate_within_group(panel_with_na, ["not_a_column"])

    def test_missing_time_column_raises(self, panel_with_na: pd.DataFrame) -> None:
        with pytest.raises(KeyError):
            interpolate_within_group(panel_with_na, ["gdp_growth"], time_col="not_a_column")


class TestImputePanelStrategies:
    def test_ffill_respects_groups(self) -> None:
        frame = pd.DataFrame(
            {
                "country_iso3": ["A", "A", "B", "B"],
                "year": [2000, 2001, 2000, 2001],
                "value": [1.0, np.nan, np.nan, 9.0],
            }
        )
        filled = impute_panel(frame, ["value"], method="ffill")
        assert filled.loc[1, "value"] == pytest.approx(1.0)
        assert np.isnan(filled.loc[2, "value"])

    def test_group_median(self) -> None:
        frame = pd.DataFrame(
            {
                "country_iso3": ["A", "A", "A", "B", "B", "B"],
                "year": [2000, 2001, 2002] * 2,
                "value": [1.0, 2.0, np.nan, 100.0, 200.0, np.nan],
            }
        )
        filled = impute_panel(frame, ["value"], method="group_median")
        assert filled.loc[2, "value"] == pytest.approx(1.5)  # A 组中位数
        assert filled.loc[5, "value"] == pytest.approx(150.0)  # B 组中位数

    def test_global_median(self, panel_with_na: pd.DataFrame) -> None:
        filled = impute_panel(panel_with_na, ["gdp_growth"], method="median")
        expected = float(np.nanmedian(panel_with_na["gdp_growth"]))
        assert filled["gdp_growth"].isna().sum() == 0
        assert filled.loc[filled["gdp_growth"].notna(), "gdp_growth"].notna().all()
        assert expected in filled["gdp_growth"].to_numpy()

    def test_mean_strategy(self, panel_with_na: pd.DataFrame) -> None:
        filled = impute_panel(panel_with_na, ["inflation"], method="mean")
        assert filled["inflation"].isna().sum() == 0

    def test_drop_strategy_removes_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "country_iso3": ["A", "A", "A"],
                "year": [2000, 2001, 2002],
                "value": [1.0, np.nan, 3.0],
            }
        )
        dropped = impute_panel(frame, ["value"], method="drop")
        assert len(dropped) == 2
        assert dropped["value"].notna().all()

    def test_invalid_method_raises(self, panel_with_na: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="不支持的 method"):
            impute_panel(panel_with_na, ["gdp_growth"], method="magic")


class TestHandleMissing:
    def test_returns_tuple_with_report(self, panel_with_na: pd.DataFrame) -> None:
        cleaned, report = handle_missing(panel_with_na, ["gdp_growth"], strategy="interpolate")
        assert isinstance(cleaned, pd.DataFrame)
        assert isinstance(report, pd.DataFrame)
        assert {"column", "n_missing_before", "n_missing_after", "n_filled"}.issubset(
            report.columns
        )

    def test_report_accounts_for_filled_cells(self, panel_with_na: pd.DataFrame) -> None:
        _, report = handle_missing(panel_with_na, ["gdp_growth"], strategy="interpolate")
        row = report.set_index("column").loc["gdp_growth"]
        # 原始 5 个缺失：Alpha 的内部缺口 2 个 + Beta 的开头 2 个 + Beta 的内部缺口 1 个
        assert row["n_missing_before"] == 5
        # 仅内部缺口被填补（Alpha 2 个 + Beta 1 个），Beta 开头的 2 个结构性缺失保留
        assert row["n_missing_after"] == 2
        assert row["n_filled"] == 3
        assert row["strategy"] == "interpolate"

    def test_drop_strategy_with_subset(self, panel_with_na: pd.DataFrame) -> None:
        cleaned, report = handle_missing(
            panel_with_na,
            ["gdp_growth", "inflation"],
            strategy="drop",
            subset_for_drop=["gdp_growth"],
        )
        assert cleaned["gdp_growth"].notna().all()
        assert report["strategy"].iloc[0] == "drop"

    def test_no_regression_when_no_missing(self) -> None:
        frame = pd.DataFrame(
            {"country_iso3": ["A", "A"], "year": [2000, 2001], "value": [1.0, 2.0]}
        )
        cleaned, report = handle_missing(frame, ["value"], strategy="interpolate")
        pd.testing.assert_series_equal(cleaned["value"], frame["value"], check_names=False)
        assert report["n_filled"].sum() == 0

    def test_documented_doctest_behaviour(self) -> None:
        frame = pd.DataFrame(
            {
                "country_iso3": ["AAA"] * 3,
                "year": [2000, 2001, 2002],
                "gdp_growth": [1.0, None, 3.0],
            }
        )
        cleaned, _ = handle_missing(frame, ["gdp_growth"], strategy="interpolate")
        assert float(cleaned.loc[1, "gdp_growth"]) == pytest.approx(2.0)


class TestStaleRatingFlags:
    def test_flags_repeated_ratings(self) -> None:
        frame = pd.DataFrame(
            {
                "country_iso3": ["A"] * 4,
                "agency": ["S&P"] * 4,
                "year": [2000, 2001, 2002, 2003],
                "rating": ["AA", "AA", "AA-", "AA-"],
            }
        )
        flagged = stale_rating_flags(frame)
        assert flagged["is_stale_rating"].tolist() == [False, True, False, True]

    def test_first_observation_is_not_stale(self) -> None:
        frame = pd.DataFrame(
            {
                "country_iso3": ["A", "A"],
                "agency": ["S&P", "S&P"],
                "year": [2000, 2001],
                "rating": ["AA", "A+"],
            }
        )
        flagged = stale_rating_flags(frame)
        assert bool(flagged.loc[0, "is_stale_rating"]) is False
