"""IMF WEO（世界经济展望）数据抓取。

接口说明
--------
IMF 提供公开的 DataMapper API：

.. code-block:: text

    https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH/PCPIPCH

返回结构为：

.. code-block:: json

    {"values": {"NGDP_RPCH": {"CHN": {"2020": 2.2, "2021": 8.4}, ...}, ...}}

WEO 每年 4 月与 10 月更新，**历史值会被修订**。因此本项目在缓存文件中记录下载
时间；若需要严格复现，请固定下载日期或在论文中注明 WEO 版本（如 WEO 2024/04）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from src.config import get_config
from src.utils.http import HttpClient
from src.utils.logging_utils import get_logger

__all__ = ["fetch_weo_indicators"]

logger = get_logger(__name__)


def fetch_weo_indicators(
    indicators: dict[str, str] | Sequence[str] | None = None,
    *,
    countries: Sequence[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    client: HttpClient | None = None,
) -> pd.DataFrame:
    """抓取 IMF WEO 指标。

    Parameters
    ----------
    indicators:
        指标映射（``{"gdp_growth": "NGDP_RPCH"}``）或代码序列。
        默认读取 ``config.yaml`` 的 ``data_sources.imf.indicators``。
    countries:
        ISO3 代码过滤；``None`` 表示全部经济体。

    Returns
    -------
    pandas.DataFrame
        列：``country_iso3`` / ``year`` / ``indicator_key`` / ``value``。
    """
    cfg = get_config()
    imf_cfg = cfg.get("data_sources", {}).get("imf", {})
    if not imf_cfg.get("enabled", True):
        logger.info("IMF 数据源在 config.yaml 中被禁用，跳过")
        return pd.DataFrame(columns=["country_iso3", "year", "indicator_key", "value"])

    mapping = indicators if indicators is not None else imf_cfg.get("indicators", {})
    if isinstance(mapping, (list, tuple)):
        mapping = {str(code): str(code) for code in mapping}
    mapping = dict(mapping)
    if not mapping:
        return pd.DataFrame(columns=["country_iso3", "year", "indicator_key", "value"])

    client = client or HttpClient(cfg)
    base_url = str(imf_cfg.get("weo_base_url", "https://www.imf.org/external/datamapper/api/v1"))
    url = f"{base_url.rstrip('/')}/{'/'.join(mapping.values())}"

    try:
        payload: Any = client.get_json(url)
    except Exception as exc:
        logger.warning(
            "IMF WEO 抓取失败：%s。可改用 World Bank 数据或在论文中说明该数据源不可得。",
            exc,
        )
        return pd.DataFrame(columns=["country_iso3", "year", "indicator_key", "value"])

    values = (payload or {}).get("values", {}) if isinstance(payload, dict) else {}
    if not values:
        logger.warning("IMF WEO 返回体中没有 values 字段，接口结构可能已变更")
        return pd.DataFrame(columns=["country_iso3", "year", "indicator_key", "value"])

    code_to_key = {str(code): key for key, code in mapping.items()}
    allowed = set(countries) if countries else None
    records: list[dict[str, Any]] = []
    for code, by_country in values.items():
        if not isinstance(by_country, dict):
            continue
        key = code_to_key.get(str(code), str(code))
        for country_iso3, by_year in by_country.items():
            if allowed is not None and country_iso3 not in allowed:
                continue
            if not isinstance(by_year, dict):
                continue
            for year, value in by_year.items():
                try:
                    numeric_year = int(year)
                except (TypeError, ValueError):
                    continue
                if start_year is not None and numeric_year < start_year:
                    continue
                if end_year is not None and numeric_year > end_year:
                    continue
                if value is None:
                    continue
                records.append(
                    {
                        "country_iso3": str(country_iso3),
                        "year": numeric_year,
                        "indicator_key": key,
                        "value": value,
                    }
                )

    if not records:
        logger.warning("IMF WEO 未解析出任何观测（可能指标代码错误或年份过滤过严）")
        return pd.DataFrame(columns=["country_iso3", "year", "indicator_key", "value"])

    frame = pd.DataFrame.from_records(records)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["year"] = frame["year"].astype("Int64")
    frame = frame.dropna(subset=["value"]).sort_values(["country_iso3", "year"], ignore_index=True)
    logger.info("IMF WEO：取得 %d 条观测（%d 个指标）", len(frame), len(mapping))
    return frame
