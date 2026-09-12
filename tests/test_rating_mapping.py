"""评级映射测试：1-21 统一刻度、机构一致性、展望标准化、异常输入处理。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.features.rating_scale import (
    INVESTMENT_GRADE_THRESHOLD,
    MAX_SCORE,
    MIN_SCORE,
    MOODYS_TO_SCORE,
    SCORE_TO_GRADE,
    SP_FITCH_TO_SCORE,
    apply_rating_mapping,
    clean_rating_text,
    is_investment_grade,
    map_rating,
    map_rating_series,
    map_rating_with_reason,
    normalize_agency,
    normalize_outlook,
    rating_action_type,
    rating_bucket,
    rating_scale_table,
    score_to_grade,
)

SP_EXPECTED = {
    "AAA": 21,
    "AA+": 20,
    "AA": 19,
    "AA-": 18,
    "A+": 17,
    "A": 16,
    "A-": 15,
    "BBB+": 14,
    "BBB": 13,
    "BBB-": 12,
    "BB+": 11,
    "BB": 10,
    "BB-": 9,
    "B+": 8,
    "B": 7,
    "B-": 6,
    "CCC+": 5,
    "CCC": 4,
    "CCC-": 3,
    "CC": 2,
    "C": 1,
}

MOODYS_EXPECTED = {
    "Aaa": 21,
    "Aa1": 20,
    "Aa2": 19,
    "Aa3": 18,
    "A1": 17,
    "A2": 16,
    "A3": 15,
    "Baa1": 14,
    "Baa2": 13,
    "Baa3": 12,
    "Ba1": 11,
    "Ba2": 10,
    "Ba3": 9,
    "B1": 8,
    "B2": 7,
    "B3": 6,
    "Caa1": 5,
    "Caa2": 4,
    "Caa3": 3,
    "Ca": 2,
    "C": 1,
}


class TestScaleDefinition:
    def test_scale_boundaries(self) -> None:
        assert (MIN_SCORE, MAX_SCORE) == (1, 21)

    def test_scale_table_has_21_rows(self) -> None:
        table = rating_scale_table()
        assert len(table) == 21
        assert table["rating_score"].tolist() == list(range(21, 0, -1))

    def test_scale_table_investment_grade_boundary(self) -> None:
        table = rating_scale_table().set_index("rating_score")
        assert bool(table.loc[12, "is_investment_grade"]) is True
        assert bool(table.loc[11, "is_investment_grade"]) is False
        assert INVESTMENT_GRADE_THRESHOLD == 12

    def test_sp_and_moodys_are_grade_aligned(self) -> None:
        """两个体系必须在同一分值上对应同一信用档位。"""
        for score in range(1, 22):
            assert SCORE_TO_GRADE[score]["S&P"] in SP_FITCH_TO_SCORE
            assert SCORE_TO_GRADE[score]["Moody's"] in MOODYS_TO_SCORE
            assert SP_FITCH_TO_SCORE[SCORE_TO_GRADE[score]["S&P"]] == score
            assert MOODYS_TO_SCORE[SCORE_TO_GRADE[score]["Moody's"]] == score

    def test_fitch_shares_sp_mapping(self) -> None:
        for rating in SP_EXPECTED:
            assert map_rating(rating, "Fitch") == map_rating(rating, "S&P")


class TestRatingMapping:
    @pytest.mark.parametrize(("rating", "expected"), sorted(SP_EXPECTED.items()))
    def test_sp_ratings(self, rating: str, expected: int) -> None:
        assert map_rating(rating, "S&P") == float(expected)

    @pytest.mark.parametrize(("rating", "expected"), sorted(MOODYS_EXPECTED.items()))
    def test_moodys_ratings(self, rating: str, expected: int) -> None:
        assert map_rating(rating, "Moody's") == float(expected)

    @pytest.mark.parametrize("rating", ["D", "SD", "RD"])
    def test_default_ratings_map_to_floor(self, rating: str) -> None:
        assert map_rating(rating, "S&P") == 1.0
        assert map_rating(rating, "Fitch") == 1.0

    def test_case_insensitive_lookup(self) -> None:
        assert map_rating("aaa", "S&P") == 21.0
        assert map_rating("aa+", "S&P") == 20.0
        assert map_rating("bAa3", "Moody's") == 12.0

    def test_unknown_agency_falls_back_to_cross_agency_search(self) -> None:
        assert map_rating("Aaa", None) == 21.0
        assert map_rating("BBB", None) == 13.0

    def test_numeric_input_passthrough(self) -> None:
        assert map_rating(12, "S&P") == 12.0
        assert map_rating(12.0, "S&P") == 12.0

    def test_numeric_out_of_range(self) -> None:
        score, reason = map_rating_with_reason(99, "S&P")
        assert math.isnan(score)
        assert reason == "numeric_out_of_range"

    @pytest.mark.parametrize(
        "value",
        [None, np.nan, "", "NR", "N/R", "n/a", "WD", "WR", "Not Rated"],
    )
    def test_non_rated_values(self, value: object) -> None:
        score, reason = map_rating_with_reason(value, "S&P")
        assert math.isnan(score)
        assert reason in {"missing", "no_rating"}

    @pytest.mark.parametrize("value", ["A-1+", "P-1", "F1", "A-2", "F2"])
    def test_short_term_ratings_are_out_of_scope(self, value: str) -> None:
        score, reason = map_rating_with_reason(value, None)
        assert math.isnan(score)
        assert reason == "short_term"

    def test_bare_b_is_not_short_term(self) -> None:
        """S&P 的长期评级 B 必须被正确映射，而不是当作短期评级丢弃。"""
        assert map_rating("B", "S&P") == 7.0
        assert map_rating_with_reason("B", "S&P")[1] == "ok"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("AA+u", "AA+"),
            ("(P)AAA", "AAA"),
            ("Baa1 (hyb)", "Baa1"),
            ("AA- (sf)", "AA-"),
            ("BBB+*", "BBB+"),
            ("A  p", "A"),
            # 回归测试：不带 +/- 的评级 + 未邀约后缀，主体必须非贪婪匹配
            ("AAAu", "AAA"),
            ("Au", "A"),
            ("Bu", "B"),
            ("CCCu", "CCC"),
            ("CCC-u", "CCC-"),
            ("C/Du", "C/D"),
        ],
    )
    def test_clean_rating_text(self, raw: str, expected: str) -> None:
        assert clean_rating_text(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        ["AA+u", "(P)AAA", "BBB+*", "A- (sf)", "Baa3 (hyb)"],
    )
    def test_dirty_ratings_still_map(self, raw: str) -> None:
        assert not math.isnan(map_rating(raw, None))

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("AAAu", 21),
            ("AAu", 19),
            ("Au", 16),
            ("BBB+u", 14),
            ("BB-u", 9),
            ("Bu", 7),
            ("CCCu", 4),
            ("CCC-u", 3),
            ("C/Du", 1),
        ],
    )
    def test_bare_unsolicited_suffix_maps_correctly(self, raw: str, expected: int) -> None:
        """回归测试：``AA+u`` 这类带 ``+/-`` 的符号一直能解析，但 **不带** ``+/-`` 的
        ``AAAu`` / ``Au`` / ``CCCu`` 早期会被整体当作评级主体而映射失败
        （示例数据中约 1.6% 的记录受影响）。本用例锁定该行为。
        """
        assert map_rating(raw, "S&P") == float(expected)
        assert map_rating_with_reason(raw, "S&P")[1] == "ok"

    def test_composite_default_symbol_is_in_scale_table(self) -> None:
        """分值 1 的 S&P 符号为复合形式 ``C/D``，必须在反向映射表中可查。"""
        assert SP_FITCH_TO_SCORE[SCORE_TO_GRADE[1]["S&P"]] == 1

    def test_mapping_reason_ok(self) -> None:
        assert map_rating_with_reason("BBB-", "S&P")[1] == "ok"


class TestAgencyNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("S&P", "S&P"),
            ("SP", "S&P"),
            ("Standard & Poor's", "S&P"),
            ("standard and poors", "S&P"),
            ("Moody's", "Moody's"),
            ("MOODYS", "Moody's"),
            ("moody", "Moody's"),
            ("Fitch Ratings", "Fitch"),
            ("fitch", "Fitch"),
        ],
    )
    def test_aliases(self, raw: str, expected: str) -> None:
        assert normalize_agency(raw) == expected

    @pytest.mark.parametrize("raw", [None, np.nan, "", "Unknown Agency"])
    def test_unrecognized(self, raw: object) -> None:
        assert normalize_agency(raw) is None


class TestOutlookNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("stable", "Stable"),
            ("Stable Outlook", "Stable"),
            ("positive", "Positive"),
            ("+", "Positive"),
            ("negative", "Negative"),
            ("Negative", "Negative"),
            ("-", "Negative"),
            ("developing", "Developing"),
            ("CW-Negative", "Watch-Negative"),
            ("Negative Watch", "Watch-Negative"),
            ("CreditWatch Positive", "Watch-Positive"),
            ("观察负面", "Watch-Negative"),
            ("稳定", "Stable"),
        ],
    )
    def test_normalization(self, raw: str, expected: str) -> None:
        assert normalize_outlook(raw) == expected

    @pytest.mark.parametrize("raw", [None, np.nan, "", "n/a", "N/A", "none"])
    def test_missing_outlook(self, raw: object) -> None:
        assert normalize_outlook(raw) is None


class TestDerivedHelpers:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [(21, "AAA"), (19, "AA"), (12, "BBB-"), (1, "C/D"), (2, "CC")],
    )
    def test_score_to_grade_sp(self, score: int, expected: str) -> None:
        assert score_to_grade(score, "S&P") == expected

    def test_score_to_grade_moodys(self) -> None:
        assert score_to_grade(12, "Moody's") == "Baa3"
        assert score_to_grade(21, "Moody's") == "Aaa"

    def test_score_to_grade_out_of_range(self) -> None:
        assert score_to_grade(0, "S&P") is None
        assert score_to_grade(22, "S&P") is None
        assert score_to_grade(None, "S&P") is None

    def test_investment_grade(self) -> None:
        assert is_investment_grade(12) is True
        assert is_investment_grade(11) is False
        assert is_investment_grade(np.nan) is False

    @pytest.mark.parametrize(
        ("score", "expected"),
        [(21, "AAA"), (19, "AA"), (16, "A"), (13, "BBB"), (10, "BB"), (7, "B"), (3, "CCC及以下")],
    )
    def test_rating_bucket(self, score: int, expected: str) -> None:
        assert rating_bucket(score) == expected

    @pytest.mark.parametrize(
        ("delta", "expected"),
        [(1, "upgrade"), (-1, "downgrade"), (0, "affirm"), (np.nan, None)],
    )
    def test_rating_action_type(self, delta: float, expected: str | None) -> None:
        assert rating_action_type(delta) == expected


class TestSeriesAndFrameMapping:
    def test_map_rating_series_with_scalar_agency(self) -> None:
        series = pd.Series(["AAA", "BBB-", "NR", "CC"])
        mapped = map_rating_series(series, "S&P")
        assert mapped.iloc[0] == 21.0
        assert mapped.iloc[1] == 12.0
        assert math.isnan(mapped.iloc[2])
        assert mapped.iloc[3] == 2.0
        assert mapped.dtype == "float64"

    def test_map_rating_series_with_agency_column(self) -> None:
        ratings = pd.Series(["Aaa", "AA+", "Baa3"])
        agencies = pd.Series(["Moody's", "S&P", "Moody's"])
        mapped = map_rating_series(ratings, agencies)
        assert mapped.tolist() == [21.0, 20.0, 12.0]

    def test_apply_rating_mapping(self, raw_ratings: pd.DataFrame) -> None:
        result = apply_rating_mapping(raw_ratings, "rating", "agency")
        assert "rating_score" in result.columns
        assert "rating_map_reason" in result.columns
        assert result.loc[result["country_iso3"] == "AAA", "rating_score"].tolist() == [20.0, 19.0]
        assert result.loc[result["agency"] == "Moody's", "rating_score"].tolist() == [14.0, 12.0]
        assert result.loc[result["agency"] == "Fitch", "rating_score"].tolist() == [6.0, 5.0]
        # "NR" 无法映射
        assert math.isnan(result.loc[result["rating"] == "NR", "rating_score"].iloc[0])
        assert result.loc[result["rating"] == "NR", "rating_map_reason"].iloc[0] == "no_rating"
        # "BB+u" 带修饰符，应能映射
        assert result.loc[result["rating"] == "BB+u", "rating_score"].iloc[0] == 11.0

    def test_apply_rating_mapping_does_not_mutate_by_default(
        self, raw_ratings: pd.DataFrame
    ) -> None:
        original_columns = list(raw_ratings.columns)
        apply_rating_mapping(raw_ratings, "rating", "agency")
        assert list(raw_ratings.columns) == original_columns

    def test_apply_rating_mapping_inplace(self, raw_ratings: pd.DataFrame) -> None:
        apply_rating_mapping(raw_ratings, "rating", "agency", inplace=True)
        assert "rating_score" in raw_ratings.columns

    def test_missing_rating_column_raises(self, raw_ratings: pd.DataFrame) -> None:
        with pytest.raises(KeyError):
            apply_rating_mapping(raw_ratings, "not_a_column", "agency")

    def test_score_range_is_respected(self, raw_ratings: pd.DataFrame) -> None:
        result = apply_rating_mapping(raw_ratings, "rating", "agency")
        scores = result["rating_score"].dropna()
        assert scores.between(MIN_SCORE, MAX_SCORE).all()
