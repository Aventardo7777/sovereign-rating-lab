"""World Bank API 抓取：WDI（世界发展指标）与 WGI（全球治理指标）。

API 说明
--------
World Bank v2 API 返回 ``[metadata, rows]`` 两元素数组：

.. code-block:: json

    [{"page": 1, "pages": 3, "per_page": 20000, "total": 60000},
     [{"indicator": {"id": "...", "value": "..."},
       "country": {"id": "CN", "value": "China"},
       "countryiso3code": "CHN",
       "date": "2020",
       "value": 2.24,
       "decimal": 1}, ...]]

本模块处理分页、缺失值与宽表/长表转换。

WGI 说明
--------
全球治理指标（WGI）以 ``*.EST`` 代码提供**治理估计值**（约 -2.5 至 2.5）。
WGI 自 1996 年起每两年发布一次（2002 年起改为年度），因此早期年份缺失属于
**结构性缺失**，不应插值填补——请参见 :mod:`src.clean.missing` 的说明。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from src.config import get_config
from src.utils.http import HttpClient
from src.utils.logging_utils import get_logger

__all__ = ["fetch_wdi_indicators", "fetch_wgi_indicators", "fetch_worldbank_indicator"]

logger = get_logger(__name__)


def _parse_worldbank_payload(payload: Any, indicator_key: str) -> list[dict[str, Any]]:
    """把 World Bank 的 JSON 响应解析为记录列表。"""
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        return []
    rows = payload[1]
    if not isinstance(rows, list):
        return []
    records: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        country_iso3 = item.get("countryiso3code") or (item.get("country") or {}).get("id")
        if not country_iso3:
            continue
        records.append(
            {
                "country_iso3": str(country_iso3),
                "country_name": (item.get("country") or {}).get("value"),
                "year": item.get("date"),
                "indicator_key": indicator_key,
                "indicator_code": (item.get("indicator") or {}).get("id"),
                "indicator_name": (item.get("indicator") or {}).get("value"),
                "value": value,
            }
        )
    return records


def fetch_worldbank_indicator(
    indicator_code: str,
    indicator_key: str,
    *,
    countries: Sequence[str] | str = "all",
    start_year: int | None = None,
    end_year: int | None = None,
    client: HttpClient | None = None,
    offline: bool | None = None,
    base_url_template: str | None = None,
) -> pd.DataFrame:
    """抓取单个 World Bank 指标（自动分页）。

    Parameters
    ----------
    indicator_code:
        官方指标代码，如 ``"NY.GDP.MKTP.KD.ZG"``。
    indicator_key:
        本项目内部使用的短键名，如 ``"gdp_growth"``。
    countries:
        ISO3 代码集合（分号分隔或序列），``"all"`` 表示全部经济体。
    start_year, end_year:
        年份区间（闭区间）。
    """
    cfg = get_config()
    source_cfg = cfg.get("data_sources", {}).get("worldbank", {})
    client = client or HttpClient(cfg)
    if offline is not None:
        client.offline = offline

    if isinstance(countries, (list, tuple, set)):
        country_param = ";".join(str(c) for c in countries)
    else:
        country_param = str(countries)

    template = base_url_template or source_cfg.get(
        "wdi_base_url", "https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"
    )
    url = template.format(countries=country_param, indicator=indicator_code)
    per_page = int(source_cfg.get("per_page", 20000))

    params: dict[str, Any] = {"format": "json", "per_page": per_page}
    if start_year is not None or end_year is not None:
        params["date"] = f"{start_year or 1960}:{end_year or 2030}"
    else:
        params["date"] = "1960:2030"

    records: list[dict[str, Any]] = []
    page = 1
    while True:
        current_params = {**params, "page": page}
        try:
            payload = client.get_json(url, params=current_params)
        except Exception as exc:
            logger.warning("World Bank 指标 %s 抓取失败: %s", indicator_code, exc)
            break
        records.extend(_parse_worldbank_payload(payload, indicator_key))
        meta = payload[0] if isinstance(payload, list) and payload else {}
        pages = int(meta.get("pages", 1) or 1)
        if page >= pages:
            break
        page += 1

    if not records:
        logger.warning(
            "指标 %s（%s）未取得任何数据。请检查网络、指标代码，或设置 "
            "data_sources.offline = true 后使用本地缓存。",
            indicator_key,
            indicator_code,
        )
        return pd.DataFrame(
            columns=[
                "country_iso3",
                "country_name",
                "year",
                "indicator_key",
                "indicator_code",
                "indicator_name",
                "value",
            ]
        )

    frame = pd.DataFrame.from_records(records)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["year"]).sort_values(["country_iso3", "year"], ignore_index=True)
    logger.info("World Bank %s：取得 %d 条观测", indicator_key, len(frame))
    return frame


def fetch_wdi_indicators(
    indicators: dict[str, str] | None = None,
    *,
    countries: Sequence[str] | str = "all",
    start_year: int | None = None,
    end_year: int | None = None,
    client: HttpClient | None = None,
) -> pd.DataFrame:
    """批量抓取 WDI 指标并合并为一张长表。

    ``indicators`` 形如 ``{"gdp_growth": "NY.GDP.MKTP.KD.ZG", ...}``；
    默认读取 ``config.yaml`` 的 ``data_sources.worldbank.indicators``。
    """
    cfg = get_config()
    mapping = dict(
        indicators or cfg.get("data_sources", {}).get("worldbank", {}).get("indicators", {})
    )
    client = client or HttpClient(cfg)

    frames = []
    for key, code in mapping.items():
        frame = fetch_worldbank_indicator(
            code,
            key,
            countries=countries,
            start_year=start_year,
            end_year=end_year,
            client=client,
            base_url_template=cfg.get("data_sources", {}).get("worldbank", {}).get("wdi_base_url"),
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["country_iso3", "year", "indicator_key"], keep="last"
    )
    return combined


def fetch_wgi_indicators(
    indicators: dict[str, str] | None = None,
    *,
    countries: Sequence[str] | str = "all",
    start_year: int | None = None,
    end_year: int | None = None,
    client: HttpClient | None = None,
) -> pd.DataFrame:
    """批量抓取 WGI 六项治理指标。

    WGI 与 WDI 使用同一套 v2 API，仅指标代码不同（``*.EST`` 后缀）。
    """
    cfg = get_config()
    wb_cfg = cfg.get("data_sources", {}).get("worldbank", {})
    mapping = indicators or wb_cfg.get("wgi_indicators", {})
    client = client or HttpClient(cfg)

    frames = []
    for key, code in mapping.items():
        frame = fetch_worldbank_indicator(
            code,
            key,
            countries=countries,
            start_year=start_year,
            end_year=end_year,
            client=client,
            base_url_template=wb_cfg.get("wgi_base_url"),
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["country_iso3", "year", "indicator_key"], keep="last")


def to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """把指标长表透视为「国家-年份」宽表（每列一个指标）。"""
    if long_df.empty:
        return pd.DataFrame()
    wide = long_df.pivot_table(
        index=["country_iso3", "year"],
        columns="indicator_key",
        values="value",
        aggfunc="last",
    )
    return wide.reset_index()
