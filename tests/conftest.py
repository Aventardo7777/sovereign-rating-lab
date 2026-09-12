"""pytest 公共夹具。

所有夹具都使用**人工构造的小样本**，保证测试结果完全确定、不依赖网络，
也不依赖项目中的合成示例数据（示例数据可能随版本更新而变化）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 让测试在未执行 `pip install -e .` 时也能导入 src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture()
def raw_ratings() -> pd.DataFrame:
    """含正确评级、展望、缺失与无法识别值的原始评级数据。"""
    return pd.DataFrame(
        {
            "country_iso3": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC", "DDD", "DDD"],
            "country_name": ["Alpha"] * 2 + ["Beta"] * 2 + ["Gamma"] * 2 + ["Delta"] * 2,
            "year": [2000, 2001, 2000, 2001, 2000, 2001, 2000, 2001],
            "agency": ["S&P", "S&P", "Moody's", "Moody's", "Fitch", "Fitch", "S&P", "S&P"],
            "rating": ["AA+", "AA", "Baa1", "Baa3", "B-", "CCC+", "NR", "BB+u"],
            "outlook": ["Stable", "Negative", "CW-Negative", "n/a", "Positive", "+", None, "稳定"],
            "action_date": pd.to_datetime(
                [
                    "2000-06-01",
                    "2001-05-01",
                    "2000-07-01",
                    "2001-07-15",
                    "2000-03-01",
                    "2001-09-01",
                    "2000-01-01",
                    "2001-11-01",
                ]
            ),
        }
    )


@pytest.fixture()
def simple_panel() -> pd.DataFrame:
    """含已知迁移路径的小面板（用于迁移矩阵的确定性断言）。"""
    rows = [
        ("AAA", "S&P", 2000, 10.0),
        ("AAA", "S&P", 2001, 10.0),
        ("AAA", "S&P", 2002, 9.0),
        ("BBB", "S&P", 2000, 15.0),
        ("BBB", "S&P", 2001, 15.0),
        ("BBB", "S&P", 2002, 15.0),
    ]
    frame = pd.DataFrame(rows, columns=["country_iso3", "agency", "year", "rating_score"])
    frame["year"] = frame["year"].astype("Int64")
    return frame


@pytest.fixture()
def panel_with_gaps() -> pd.DataFrame:
    """含非连续年份的面板：2002 → 2004 的跳跃不应被计为一次迁移。"""
    rows = [
        ("AAA", "S&P", 2000, 12.0),
        ("AAA", "S&P", 2001, 12.0),
        ("AAA", "S&P", 2002, 12.0),
        ("AAA", "S&P", 2004, 8.0),
        ("AAA", "S&P", 2005, 8.0),
    ]
    frame = pd.DataFrame(rows, columns=["country_iso3", "agency", "year", "rating_score"])
    frame["year"] = frame["year"].astype("Int64")
    return frame


@pytest.fixture()
def macro_panel() -> pd.DataFrame:
    """与国家-年份网格对齐的宏观指标（含刻意制造的缺失值）。"""
    countries = ["AAA", "BBB", "CCC", "DDD"]
    years = list(range(2000, 2010))
    records = []
    rng = np.random.default_rng(20240625)
    for country in countries:
        for year in years:
            records.append(
                {
                    "country_iso3": country,
                    "year": year,
                    "gdp_growth": float(rng.normal(2.5, 1.5)),
                    "inflation": float(abs(rng.normal(3.0, 2.0))),
                    "gov_debt_gdp": float(rng.uniform(20, 120)),
                    "external_debt_gni": float(rng.uniform(10, 90)),
                    "reserves_months_imports": float(rng.uniform(1, 12)),
                    "current_account_gdp": float(rng.normal(-2, 4)),
                    "exports_gdp": float(rng.uniform(10, 60)),
                    "fx_depreciation": float(rng.normal(2, 8)),
                    "real_interest_rate": float(rng.normal(3, 5)),
                    "spread_bps": float(rng.uniform(50, 900)),
                    "voice_accountability": float(rng.normal(0, 1)),
                    "political_stability": float(rng.normal(0, 1)),
                    "government_effectiveness": float(rng.normal(0, 1)),
                    "regulatory_quality": float(rng.normal(0, 1)),
                    "rule_of_law": float(rng.normal(0, 1)),
                    "control_corruption": float(rng.normal(0, 1)),
                }
            )
    frame = pd.DataFrame.from_records(records)
    frame.loc[frame.index[3:6], "gdp_growth"] = np.nan
    frame.loc[frame.index[10:11], "inflation"] = np.nan
    return frame


@pytest.fixture()
def feature_panel(simple_panel: pd.DataFrame, macro_panel: pd.DataFrame) -> pd.DataFrame:
    """合并后的建模用面板（评级 + 宏观），供回归与模型测试复用。"""
    from src.clean.panel import build_country_year_panel

    return build_country_year_panel(
        simple_panel,
        macro_panel,
        start_year=2000,
        end_year=2009,
    )
