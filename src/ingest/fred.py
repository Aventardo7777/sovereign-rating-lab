"""FRED（圣路易斯联储经济数据）抓取。

**API Key 配置（必读）**
------------------------
FRED 要求注册后使用个人 API Key，本项目**绝不硬编码**任何密钥。请设置环境变量：

.. code-block:: bash

    # Windows PowerShell（当前会话）
    $env:FRED_API_KEY = "your_key_here"
    # Windows（永久，需重启终端）
    setx FRED_API_KEY "your_key_here"
    # macOS / Linux
    export FRED_API_KEY="your_key_here"

免费申请地址：https://fred.stlouisfed.org/docs/api/api_key.html

缺少 Key 时本模块**不会报错中断**，而是返回空表并打印配置指引，
以便整个研究流水线在无凭证环境下依然可运行。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from src.config import get_config, get_env
from src.utils.http import HttpClient
from src.utils.logging_utils import get_logger

__all__ = ["fetch_fred_series", "fred_available"]

logger = get_logger(__name__)

_EMPTY_COLUMNS = ["series_id", "date", "year", "value"]

_SETUP_HINT = (
    "未检测到 FRED API Key。请在环境变量 FRED_API_KEY 中配置（不要写入代码或配置文件）：\n"
    '  PowerShell : $env:FRED_API_KEY = "<your_key>"\n'
    '  bash/zsh   : export FRED_API_KEY="<your_key>"\n'
    "申请地址：https://fred.stlouisfed.org/docs/api/api_key.html"
)


def fred_available() -> bool:
    """环境变量中是否已配置 FRED API Key。"""
    cfg = get_config().get("data_sources", {}).get("fred", {})
    env_name = str(cfg.get("api_key_env", "FRED_API_KEY"))
    return bool(get_env(env_name))


def fetch_fred_series(
    series_ids: str | Sequence[str] | None = None,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    client: HttpClient | None = None,
) -> pd.DataFrame:
    """抓取一个或多个 FRED 时间序列。

    Returns
    -------
    pandas.DataFrame
        列：``series_id`` / ``date`` / ``year`` / ``value``。
        凭据缺失或抓取失败时返回空表（保留列结构）。
    """
    cfg = get_config()
    fred_cfg = cfg.get("data_sources", {}).get("fred", {})
    env_name = str(fred_cfg.get("api_key_env", "FRED_API_KEY"))
    api_key = get_env(env_name)

    if series_ids is None:
        series_ids = list(fred_cfg.get("series", {}).keys()) or ["DGS10"]
    ids = [series_ids] if isinstance(series_ids, str) else [str(s) for s in series_ids]
    if not api_key:
        logger.warning(_SETUP_HINT)
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    client = client or HttpClient(cfg)
    base_url = str(fred_cfg.get("base_url", "https://api.stlouisfed.org/fred/series/observations"))

    frames: list[pd.DataFrame] = []
    for series_id in ids:
        params: dict[str, Any] = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
        }
        if start_date:
            params["observation_start"] = start_date
        if end_date:
            params["observation_end"] = end_date
        try:
            payload = client.get_json(base_url, params=params, check_robots=False)
        except Exception as exc:
            logger.warning("FRED 序列 %s 抓取失败：%s", series_id, exc)
            continue

        observations = (payload or {}).get("observations", []) if isinstance(payload, dict) else []
        if not observations:
            logger.warning("FRED 序列 %s 无观测返回", series_id)
            continue
        frame = pd.DataFrame.from_records(observations)
        frame = frame.assign(series_id=series_id)
        frame["value"] = pd.to_numeric(frame.get("value"), errors="coerce")
        frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce")
        frame["year"] = frame["date"].dt.year
        frames.append(frame[["series_id", "date", "year", "value"]])

    if not frames:
        return pd.DataFrame(columns=_EMPTY_COLUMNS)
    combined = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    logger.info("FRED：取得 %d 条观测（%d 个序列）", len(combined), len(ids))
    return combined
