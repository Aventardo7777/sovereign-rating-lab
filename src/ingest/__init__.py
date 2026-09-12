"""数据抓取模块。

统一约定
--------
* 所有抓取函数返回**整洁长表**（tidy long format），列名与项目其余部分一致；
* 所有原始响应缓存在 ``data/raw/cache``，并记录来源 URL 与下载时间；
* 每个函数都支持 ``offline=True``：只读缓存，不发网络请求；
* 缺少凭证 / 接口变更 / 站点拒绝时，返回空表并给出**可操作的提示**，
  而不是静默失败或伪造数据。
"""

from __future__ import annotations

from src.ingest.bis import fetch_bis_dataset
from src.ingest.fred import fetch_fred_series, fred_available
from src.ingest.imf import fetch_weo_indicators
from src.ingest.ratings import (
    REQUIRED_RATING_COLUMNS,
    fetch_public_ratings_page,
    load_ratings_csv,
    load_sample_ratings,
    parse_ratings_html,
    validate_imported_ratings,
)
from src.ingest.worldbank import fetch_wdi_indicators, fetch_wgi_indicators

__all__ = [
    "REQUIRED_RATING_COLUMNS",
    "fetch_bis_dataset",
    "fetch_fred_series",
    "fetch_public_ratings_page",
    "fetch_wdi_indicators",
    "fetch_weo_indicators",
    "fetch_wgi_indicators",
    "fred_available",
    "load_ratings_csv",
    "load_sample_ratings",
    "parse_ratings_html",
    "validate_imported_ratings",
]
