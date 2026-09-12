"""BIS（国际清算银行）统计数据抓取。

状态说明
--------
BIS 统计 API 采用 SDMX-JSON 结构，且不同数据集（dataflow）的维度顺序差异较大，
解析逻辑高度依赖具体数据集版本。因此本项目：

* **默认禁用**该数据源（``config.yaml`` 的 ``data_sources.bis.enabled: false``）；
* 提供一个**通用的 SDMX-JSON 解析器**，用户确认数据集结构后可启用；
* 抓取失败时返回空表并给出提示，不影响其余流程。

若研究需要政策利率、实际有效汇率、跨境信贷等指标，建议优先使用：

* IMF IFS / World Bank WDI（口径清晰，接口稳定）
* 各国央行公开数据（更权威，但需要逐国接入）
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import get_config
from src.utils.http import HttpClient
from src.utils.logging_utils import get_logger

__all__ = ["fetch_bis_dataset", "parse_sdmx_json"]

logger = get_logger(__name__)

_EMPTY_COLUMNS = ["country_iso3", "year", "indicator_key", "value", "period"]


def parse_sdmx_json(payload: Any, *, indicator_key: str) -> pd.DataFrame:
    """通用 SDMX-JSON 解析器（BIS / OECD / ECB 等使用同一规范）。

    SDMX-JSON 的核心结构是：``structure.dimensions`` 定义各维度的取值顺序，
    ``dataSets[0].series`` 以**维度索引元组**为键、``observations`` 以**时间索引**
    为键存放数值。本函数把这一层层索引还原为可读的长表。
    """
    if not isinstance(payload, dict):
        return pd.DataFrame(columns=_EMPTY_COLUMNS)
    data = payload.get("data", payload)
    structure = (data or {}).get("structure", {})
    dimensions = structure.get("dimensions", {})
    series_dims = dimensions.get("series", []) or []
    obs_dims = dimensions.get("observation", []) or []
    datasets = data.get("dataSets", []) or []
    if not datasets or not series_dims:
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    series_dim_values = [
        [str(v.get("id", "")) for v in (dim.get("values") or [])] for dim in series_dims
    ]
    series_dim_names = [str(dim.get("id", f"dim{i}")) for i, dim in enumerate(series_dims)]
    time_values = (
        [str(v.get("id", "")) for v in (obs_dims[0].get("values") or [])] if obs_dims else []
    )

    records: list[dict[str, Any]] = []
    for series_key, series in (datasets[0].get("series") or {}).items():
        indices = [int(part) for part in str(series_key).split(":")]
        labels: dict[str, str] = {}
        for name, values, index in zip(series_dim_names, series_dim_values, indices, strict=False):
            labels[name] = values[index] if 0 <= index < len(values) else ""
        for obs_key, obs in (series.get("observations") or {}).items():
            time_index = int(obs_key)
            value = obs[0] if isinstance(obs, (list, tuple)) and obs else None
            if value is None:
                continue
            period = (
                time_values[time_index] if 0 <= time_index < len(time_values) else str(time_index)
            )
            record = {
                "country_iso3": labels.get("REF_AREA")
                or labels.get("COUNTRY")
                or labels.get("BORROWERS_CTY", ""),
                "year": _extract_year(period),
                "period": period,
                "indicator_key": indicator_key,
                "value": value,
            }
            record.update({f"dim_{k}": v for k, v in labels.items()})
            records.append(record)

    if not records:
        return pd.DataFrame(columns=_EMPTY_COLUMNS)
    frame = pd.DataFrame.from_records(records)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["value"])


def _extract_year(period: str) -> Any:
    digits = "".join(ch for ch in str(period) if ch.isdigit())
    if len(digits) >= 4:
        try:
            return int(digits[:4])
        except ValueError:  # pragma: no cover
            return pd.NA
    return pd.NA


def fetch_bis_dataset(
    dataset: str,
    *,
    indicator_key: str | None = None,
    client: HttpClient | None = None,
) -> pd.DataFrame:
    """抓取一个 BIS 数据集（SDMX-JSON）。

    Parameters
    ----------
    dataset:
        BIS 数据集的路径片段，例如 ``"WS_CBPOL_D/1.0/D.."``。
        完整 URL = ``{base_url}/{dataset}?format=jsondata``。
    indicator_key:
        输出表中的指标名；缺省时使用数据集名称。
    """
    cfg = get_config()
    bis_cfg = cfg.get("data_sources", {}).get("bis", {})
    if not bis_cfg.get("enabled", False):
        logger.info(
            "BIS 数据源默认禁用（config.yaml: data_sources.bis.enabled = false）。"
            "确认数据集结构后再启用。"
        )
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    client = client or HttpClient(cfg)
    base_url = str(bis_cfg.get("base_url", "https://stats.bis.org/api/v2/data/dataflow/BIS"))
    url = f"{base_url.rstrip('/')}/{dataset.lstrip('/')}"
    try:
        payload = client.get_json(url, params={"format": "jsondata"})
    except Exception as exc:
        logger.warning("BIS 数据集 %s 抓取失败：%s", dataset, exc)
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    frame = parse_sdmx_json(payload, indicator_key=indicator_key or dataset.split("/")[0])
    logger.info("BIS %s：取得 %d 条观测", dataset, len(frame))
    return frame
