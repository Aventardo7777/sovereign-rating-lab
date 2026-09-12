"""评级迁移矩阵测试：频数/概率矩阵、上调度、机构分歧、评级周期。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.migration import (
    agency_dispersion,
    build_migration_matrix,
    downgrade_probability,
    migration_entropy,
    migration_summary,
    rating_cycle_stats,
    rating_spell_stats,
    transition_probability_matrix,
    upgrade_downgrade_probabilities,
)


class TestMigrationMatrix:
    def test_matrix_shape_is_full_scale(self, simple_panel: pd.DataFrame) -> None:
        matrix = build_migration_matrix(simple_panel)
        assert matrix.shape == (21, 21)
        assert matrix.index.tolist() == list(range(1, 22))
        assert matrix.columns.tolist() == list(range(1, 22))
        assert matrix.index.name == "from_score"
        assert matrix.columns.name == "to_score"

    def test_matrix_counts(self, simple_panel: pd.DataFrame) -> None:
        matrix = build_migration_matrix(simple_panel)
        assert matrix.loc[10, 10] == 1  # AAA: 2000 -> 2001 未变
        assert matrix.loc[10, 9] == 1  # AAA: 2001 -> 2002 下调 1 档
        assert matrix.loc[15, 15] == 2  # BBB: 两年均未变
        assert matrix.to_numpy().sum() == 4

    def test_probability_matrix_rows_sum_to_one(self, simple_panel: pd.DataFrame) -> None:
        matrix = build_migration_matrix(simple_panel)
        probabilities = transition_probability_matrix(matrix)
        row_10 = probabilities.loc[10].sum()
        row_15 = probabilities.loc[15].sum()
        assert row_10 == pytest.approx(1.0)
        assert row_15 == pytest.approx(1.0)
        assert probabilities.loc[10, 9] == pytest.approx(0.5)
        assert probabilities.loc[10, 10] == pytest.approx(0.5)

    def test_empty_row_stays_zero_not_nan(self, simple_panel: pd.DataFrame) -> None:
        """样本中未出现的评级档不应产生 NaN，否则热力图会出现空洞。"""
        probabilities = transition_probability_matrix(build_migration_matrix(simple_panel))
        assert not probabilities.isna().to_numpy().any()
        assert probabilities.loc[1].sum() == 0.0

    def test_normalize_flag_matches_explicit_function(self, simple_panel: pd.DataFrame) -> None:
        matrix = build_migration_matrix(simple_panel)
        via_flag = build_migration_matrix(simple_panel, normalize=True)
        pd.testing.assert_frame_equal(via_flag, transition_probability_matrix(matrix))

    def test_non_consecutive_years_are_excluded(self, panel_with_gaps: pd.DataFrame) -> None:
        """2002 -> 2004 存在年份缺口，不能计为一次迁移。"""
        matrix = build_migration_matrix(panel_with_gaps)
        assert matrix.to_numpy().sum() == 3  # 2000-01, 2001-02, 2004-05
        assert matrix.loc[12, 12] == 2
        assert matrix.loc[12, 8] == 0  # 缺口造成的"迁移"被正确排除
        assert matrix.loc[8, 8] == 1

    def test_year_filtering(self, simple_panel: pd.DataFrame) -> None:
        only_2001 = build_migration_matrix(simple_panel, from_year=2001, to_year=2001)
        assert only_2001.to_numpy().sum() == 2
        assert only_2001.loc[10, 9] == 1

    def test_score_is_effective_option(self, simple_panel: pd.DataFrame) -> None:
        enriched = simple_panel.assign(score_in_effect=simple_panel["rating_score"])
        via_effective = build_migration_matrix(simple_panel, score_is_effective=True)
        via_default = build_migration_matrix(enriched, score_is_effective=True)
        pd.testing.assert_frame_equal(via_effective, via_default)

    def test_empty_panel_yields_zero_matrix(self) -> None:
        empty = pd.DataFrame(
            {
                "country_iso3": pd.Series(dtype="object"),
                "agency": pd.Series(dtype="object"),
                "year": pd.Series(dtype="int64"),
                "rating_score": pd.Series(dtype="float64"),
            }
        )
        matrix = build_migration_matrix(empty)
        assert matrix.shape == (21, 21)
        assert matrix.to_numpy().sum() == 0

    def test_missing_entity_columns_raises(self) -> None:
        frame = pd.DataFrame({"year": [2000], "rating_score": [10.0]})
        with pytest.raises(KeyError):
            build_migration_matrix(frame)


class TestMigrationSummary:
    def test_summary_probabilities_sum_to_one(self, simple_panel: pd.DataFrame) -> None:
        summary = migration_summary(build_migration_matrix(simple_panel))
        assert summary["n_transitions"] == 4
        assert summary["p_stable"] == pytest.approx(0.75)
        assert summary["p_downgrade"] == pytest.approx(0.25)
        assert summary["p_upgrade"] == pytest.approx(0.0)
        total = summary["p_stable"] + summary["p_downgrade"] + summary["p_upgrade"]
        assert total == pytest.approx(1.0)

    def test_summary_of_empty_matrix(self) -> None:
        empty = pd.DataFrame(0.0, index=range(1, 22), columns=range(1, 22))
        summary = migration_summary(empty)
        assert summary["n_transitions"] == 0
        assert np.isnan(summary["p_stable"])


class TestUpgradeDowngradeProbabilities:
    def test_counts_and_probabilities(self, simple_panel: pd.DataFrame) -> None:
        table = upgrade_downgrade_probabilities(simple_panel, group_cols=["year"])
        assert len(table) == 2
        row_2000 = table[table["year"] == 2000].iloc[0]
        assert row_2000["n_transitions"] == 2
        assert row_2000["p_stable"] == pytest.approx(1.0)
        assert row_2000["p_change"] == pytest.approx(0.0)

        row_2001 = table[table["year"] == 2001].iloc[0]
        assert row_2001["n_transitions"] == 2
        assert row_2001["n_downgrade"] == 1
        assert row_2001["p_downgrade"] == pytest.approx(0.5)
        assert row_2001["p_downgrade_given_change"] == pytest.approx(1.0)
        # 该年唯一的一次变动就是下调，因此「变动条件下上调」的概率为 0
        assert row_2001["p_upgrade_given_change"] == pytest.approx(0.0)

    def test_group_by_agency(self, simple_panel: pd.DataFrame) -> None:
        table = upgrade_downgrade_probabilities(simple_panel, group_cols=["agency"])
        assert table["agency"].tolist() == ["S&P"]
        assert table["n_transitions"].iloc[0] == 4

    def test_group_by_rating_bucket(self, simple_panel: pd.DataFrame) -> None:
        # 分值 10 属于 BB 档（9-11），分值 15 属于 A 档（15-17）
        table = upgrade_downgrade_probabilities(simple_panel, group_cols=["bucket_from"])
        assert set(table["bucket_from"]) == {"BB", "A"}

    def test_no_group_returns_single_row(self, simple_panel: pd.DataFrame) -> None:
        table = upgrade_downgrade_probabilities(simple_panel, group_cols=None)
        assert len(table) == 1
        assert table["n_transitions"].iloc[0] == 4

    def test_empty_input_returns_empty_table(self) -> None:
        empty = pd.DataFrame(
            {
                "country_iso3": pd.Series(dtype="object"),
                "agency": pd.Series(dtype="object"),
                "year": pd.Series(dtype="int64"),
                "rating_score": pd.Series(dtype="float64"),
            }
        )
        table = upgrade_downgrade_probabilities(empty, group_cols=["year"])
        assert table.empty
        assert "p_downgrade" in table.columns


class TestMultiHorizonDowngradeProbability:
    def test_horizon_one_matches_single_step(self, simple_panel: pd.DataFrame) -> None:
        table = downgrade_probability(simple_panel, horizon=1, by_rating=False)
        assert table["n_obs"].iloc[0] == 4
        assert table["p_downgrade"].iloc[0] == pytest.approx(0.25)

    def test_horizon_two_requires_consecutive_years(self, simple_panel: pd.DataFrame) -> None:
        table = downgrade_probability(simple_panel, horizon=2, by_rating=False)
        # 2000 -> 2002 是唯一满足"相隔恰好 2 年"的组合
        assert table["n_obs"].iloc[0] == 2
        # AAA 从 10 降到 9；BBB 保持 15
        assert table["p_downgrade"].iloc[0] == pytest.approx(0.5)

    def test_by_rating_bucket(self, simple_panel: pd.DataFrame) -> None:
        table = downgrade_probability(simple_panel, horizon=1, by_rating=True)
        assert set(table["rating_bucket"]) == {"BB", "A"}
        assert table["horizon"].unique().tolist() == [1]


class TestRatingCycle:
    def test_spell_stats(self, simple_panel: pd.DataFrame) -> None:
        spells = rating_spell_stats(simple_panel)
        assert len(spells) == 3
        aaa = spells[(spells["country_iso3"] == "AAA") & (spells["rating_score"] == 10)].iloc[0]
        assert aaa["start_year"] == 2000
        assert aaa["end_year"] == 2001
        assert aaa["duration"] == 2
        assert bool(aaa["ongoing"]) is False

        ongoing = spells[spells["ongoing"]]
        assert set(ongoing["rating_score"]) == {9, 15}

    def test_spell_duration_counts_years_not_observations(self) -> None:
        frame = pd.DataFrame(
            {
                "country_iso3": ["X", "X", "X"],
                "agency": ["S&P"] * 3,
                "year": [2000, 2003, 2005],
                "rating_score": [10.0, 10.0, 10.0],
            }
        )
        spells = rating_spell_stats(frame)
        assert len(spells) == 1
        assert spells["duration"].iloc[0] == 6  # 2000..2005 共 6 个年份

    def test_cycle_stats_by_agency(self, simple_panel: pd.DataFrame) -> None:
        stats = rating_cycle_stats(simple_panel, group_cols=["agency"])
        assert len(stats) == 1
        assert stats["n_spells"].iloc[0] == 3
        # 最长片段：BBB 的 15 分从 2000 持续到 2002 年，共 3 年
        assert stats["max_spell_years"].iloc[0] == 3
        assert stats["mean_spell_years"].iloc[0] == pytest.approx(2.0)
        # 四次迁移的 |Δ| 为 0, 1, 0, 0
        assert stats["mean_abs_change"].iloc[0] == pytest.approx(0.25)
        assert stats["share_ongoing"].iloc[0] == pytest.approx(2 / 3)

    def test_cycle_stats_overall(self, simple_panel: pd.DataFrame) -> None:
        stats = rating_cycle_stats(simple_panel, group_cols=None)
        assert len(stats) == 1
        assert stats["n_spells"].iloc[0] == 3


class TestAgencyDispersion:
    @pytest.fixture()
    def multi_agency_panel(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "country_iso3": ["AAA"] * 4 + ["BBB"] * 2,
                "year": [2000, 2000, 2000, 2001, 2000, 2000],
                "agency": ["S&P", "Moody's", "Fitch", "S&P", "S&P", "Fitch"],
                "rating_score": [14.0, 12.0, 13.0, 14.0, 10.0, 10.0],
            }
        )

    def test_split_rating_detected(self, multi_agency_panel: pd.DataFrame) -> None:
        dispersion = agency_dispersion(multi_agency_panel)
        row = dispersion[(dispersion["country_iso3"] == "AAA") & (dispersion["year"] == 2000)].iloc[
            0
        ]
        assert row["n_agencies"] == 3
        assert row["range_score"] == 2
        assert row["is_split"] == 1
        assert row["mean_score"] == pytest.approx(13.0)

    def test_no_split_when_agencies_agree(self, multi_agency_panel: pd.DataFrame) -> None:
        dispersion = agency_dispersion(multi_agency_panel)
        row = dispersion[(dispersion["country_iso3"] == "BBB") & (dispersion["year"] == 2000)].iloc[
            0
        ]
        assert row["range_score"] == 0
        assert row["is_split"] == 0
        assert row["std_score"] == 0.0

    def test_missing_columns_raise(self) -> None:
        with pytest.raises(KeyError):
            agency_dispersion(pd.DataFrame({"year": [2000], "rating_score": [10.0]}))


class TestMigrationEntropy:
    def test_perfectly_stable_matrix_has_zero_entropy(self) -> None:
        matrix = pd.DataFrame(0.0, index=range(1, 22), columns=range(1, 22))
        for score in range(1, 22):
            matrix.loc[score, score] = 5.0
        assert migration_entropy(matrix) == pytest.approx(0.0)

    def test_uniform_dispersion_has_positive_entropy(self) -> None:
        matrix = pd.DataFrame(0.0, index=range(1, 22), columns=range(1, 22))
        for score in range(1, 22):
            matrix.loc[score, :] = 1.0
        entropy = migration_entropy(matrix)
        assert entropy == pytest.approx(np.log(21), rel=1e-6)

    def test_per_row_returns_series(self) -> None:
        matrix = pd.DataFrame(0.0, index=range(1, 22), columns=range(1, 22))
        matrix.loc[10, 10] = 3.0
        matrix.loc[10, 9] = 1.0
        per_row = migration_entropy(matrix, per_row=True)
        assert isinstance(per_row, pd.Series)
        assert len(per_row) == 21
        assert per_row.loc[10] > 0
        assert per_row.loc[1] == 0.0
