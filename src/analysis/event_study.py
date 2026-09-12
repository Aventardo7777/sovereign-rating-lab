"""评级行动的事件研究框架。

方法
----
对每一次评级行动（下调 / 上调 / 展望调整），考察事件日前后
``[-20, +20]`` 个交易日内目标变量的异常变化：

.. math::

    AR_{i,t} = R_{i,t} - E[R_{i,t}], \\qquad
    CAR_i = \\sum_{t=-20}^{+20} AR_{i,t}

其中期望收益 :math:`E[R_{i,t}]` 采用**市场模型**（估计窗口
``[-120, -21]``，用当地股指或全球因子做解释变量）估计；若估计窗口数据不足，
自动退化为**均值调整模型**（用估计窗口的均值作为期望值），并在结果中记录
``model`` 字段说明使用了哪一类模型。

数据可得性说明
--------------
主权利差（如 EMBI 全球利差）与高频汇率、股指数据通常受版权限制或需要付费终端。
本项目提供完整的框架与样本数据演示；当真实高频序列不可得时，
:func:`event_study` 会返回带 ``status="insufficient_data"`` 的结果对象，
明确标注限制，而**不会**用伪造数据填充。这符合本项目「不伪造数据」的基本约定。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

__all__ = [
    "EventStudyResult",
    "abnormal_changes",
    "cumulative_abnormal_changes",
    "event_study",
    "event_study_summary",
    "prepare_event_windows",
    "ratings_to_events",
]

logger = get_logger(__name__)

_EVENT_WINDOW_DEFAULT = (-20, 20)
_ESTIMATION_WINDOW_DEFAULT = (-120, -21)


@dataclass
class EventStudyResult:
    """事件研究的容器对象。"""

    abnormal: pd.DataFrame
    car: pd.DataFrame
    events: pd.DataFrame
    status: str
    message: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "meta": self.meta,
            "n_events": len(self.events),
            "n_abnormal_rows": len(self.abnormal),
        }


def prepare_event_windows(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    *,
    date_col: str = "date",
    entity_col: str = "country_iso3",
    value_col: str = "value",
    window: tuple[int, int] = _EVENT_WINDOW_DEFAULT,
    estimation_window: tuple[int, int] | None = _ESTIMATION_WINDOW_DEFAULT,
) -> pd.DataFrame:
    """把「宽表行情 + 事件表」展开为事件窗口长表。

    Parameters
    ----------
    prices:
        长表行情数据：``country_iso3`` / ``date`` / ``value``（+ 可选的 ``market`` 因子列）。
    events:
        事件表：``country_iso3`` / ``event_date``（+ 可选 ``event_type``）。
    window:
        事件窗口（相对交易日），默认 ``(-20, 20)``。
    estimation_window:
        正常收益的**估计窗口**，默认 ``(-120, -21)``。由于估计窗口不包含在事件
        窗口内，函数会把两者的并集一并抽取出来（相对日范围约为 ``-120`` 至 ``+20``），
        否则 :func:`abnormal_changes` 将无法估计市场模型。
        传入 ``None`` 表示只抽取事件窗口本身。

    Returns
    -------
    pandas.DataFrame
        列为 ``event_id`` / ``country_iso3`` / ``event_date`` / ``relative_day`` /
        ``date`` / ``value``（+ ``market``）。
    """
    for required in (date_col, entity_col, value_col):
        if required not in prices.columns:
            raise KeyError(f"prices 缺少列: {required}")
    if entity_col not in events.columns:
        raise KeyError(f"events 缺少列: {entity_col}")
    date_field = "event_date" if "event_date" in events.columns else date_col
    if date_field not in events.columns:
        raise KeyError(f"events 缺少事件日期列（{date_col} 或 event_date）")

    lower_rel = window[0] if estimation_window is None else min(window[0], estimation_window[0])
    upper_rel = window[1]

    frame = prices.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col, entity_col, value_col]).sort_values(
        [entity_col, date_col]
    )

    event_table = events.copy()
    event_table[date_field] = pd.to_datetime(event_table[date_field], errors="coerce")
    event_table = event_table.dropna(subset=[date_field, entity_col]).reset_index(drop=True)
    event_table["event_id"] = np.arange(len(event_table))

    extra_cols = [
        c for c in ("event_type", "agency", "rating", "from_score", "to_score") if c in event_table
    ]
    payload = frame[[entity_col, date_col, value_col, *(["market"] if "market" in frame else [])]]

    chunks: list[pd.DataFrame] = []
    for row in event_table.itertuples(index=False):
        entity = getattr(row, entity_col)
        event_date = getattr(row, date_field)
        event_id = row.event_id
        subset = payload[payload[entity_col] == entity].reset_index(drop=True)
        if subset.empty:
            continue
        # 以「事件日或之前最近的交易日」为第 0 日
        positions = subset.index[subset[date_col] <= event_date]
        if len(positions) == 0:
            continue
        anchor = int(positions[-1])
        lo = anchor + lower_rel
        hi = anchor + upper_rel
        if lo < 0 or hi >= len(subset):
            # 窗口不完整仍然保留，但只截取可得部分
            lo, hi = max(lo, 0), min(hi, len(subset) - 1)
            if hi - lo < 5:
                continue
        window_slice = subset.iloc[lo : hi + 1].copy()
        window_slice["relative_day"] = np.arange(lo - anchor, hi - anchor + 1)
        window_slice["event_id"] = event_id
        window_slice["event_date"] = event_date
        for column in extra_cols:
            window_slice[column] = getattr(row, column)
        chunks.append(window_slice)

    columns = [
        "event_id",
        entity_col,
        "event_date",
        "relative_day",
        date_col,
        value_col,
        *extra_cols,
    ]
    if not chunks:
        return pd.DataFrame(columns=columns)
    result = pd.concat(chunks, ignore_index=True)
    result = result.rename(columns={date_col: "date", value_col: "value"})
    ordered = [c for c in columns if c in result.columns]
    remaining = [c for c in result.columns if c not in ordered]
    return result[ordered + remaining]


def abnormal_changes(
    windows: pd.DataFrame,
    *,
    estimation_window: tuple[int, int] = _ESTIMATION_WINDOW_DEFAULT,
    log_returns: bool = True,
    market_col: str | None = "market",
) -> pd.DataFrame:
    """在事件窗口内计算异常收益 / 异常变化。

    对每个事件：

    1. 在估计窗口内用市场模型 ``R = a + b * R_m + e`` 拟合（无市场因子时退化为均值模型）；
    2. 用估计系数在事件窗口内预测正常变化；
    3. ``abnormal = actual - predicted``，并累积得到 ``car``。
    """
    if windows.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "relative_day",
                "date",
                "actual",
                "expected",
                "abnormal",
                "car",
                "model",
            ]
        )

    results: list[pd.DataFrame] = []
    has_market = market_col is not None and market_col in windows.columns

    for _event_id, chunk in windows.groupby("event_id", sort=True):
        chunk = chunk.sort_values("relative_day").copy()
        values = pd.to_numeric(chunk["value"], errors="coerce").astype("float64")
        changes = np.log(values).diff() if log_returns and (values > 0).all() else values.diff()
        chunk["_ret"] = changes

        est_mask = chunk["relative_day"].between(*estimation_window)
        est = chunk[est_mask]
        evt = chunk[~chunk["_ret"].isna()]
        if len(est) < 10 or evt.empty:
            model = "insufficient_data"
            chunk["expected"] = np.nan
            chunk["abnormal"] = np.nan
            chunk["model"] = model
            chunk["actual"] = changes.to_numpy()
            chunk["car"] = np.nan
            results.append(chunk)
            continue

        y = est["_ret"].astype("float64")
        if has_market:
            x = pd.to_numeric(est[market_col], errors="coerce").astype("float64")
            valid = x.notna() & y.notna()
            if valid.sum() >= 10 and x[valid].std() > 0:
                beta, alpha = np.polyfit(x[valid], y[valid], 1)
                model = "market_model"
            else:
                alpha, beta = float(y.mean()), 0.0
                model = "mean_adjusted"
        else:
            alpha, beta = float(y.mean()), 0.0
            model = "mean_adjusted"

        if has_market and model == "market_model":
            x_evt = pd.to_numeric(chunk[market_col], errors="coerce").astype("float64")
            expected = alpha + beta * x_evt
        else:
            expected = pd.Series(alpha, index=chunk.index, dtype="float64")

        chunk["actual"] = changes.to_numpy()
        chunk["expected"] = expected
        chunk["abnormal"] = (changes - expected).where(~changes.isna())
        chunk["car"] = chunk["abnormal"].fillna(0.0).cumsum()
        chunk["model"] = model
        results.append(chunk)

    combined = pd.concat(results, ignore_index=True)
    keep = [
        c
        for c in (
            "event_id",
            "country_iso3",
            "event_date",
            "event_type",
            "relative_day",
            "date",
            "value",
            "actual",
            "expected",
            "abnormal",
            "car",
            "model",
        )
        if c in combined.columns
    ]
    return combined[keep]


def cumulative_abnormal_changes(
    abnormal: pd.DataFrame,
    *,
    start: int = -20,
    end: int = 20,
) -> pd.DataFrame:
    """按事件汇总窗口 ``[start, end]`` 内的累积异常变化（CAR）。"""
    if abnormal.empty:
        return pd.DataFrame(
            columns=["event_id", "country_iso3", "event_type", "car", "n_days", "model"]
        )
    subset = abnormal[abnormal["relative_day"].between(start, end)]
    group_cols = [c for c in ("event_id", "country_iso3", "event_type", "model") if c in subset]
    car = (
        subset.groupby(group_cols, dropna=False)
        .agg(car=("abnormal", "sum"), n_days=("abnormal", "count"))
        .reset_index()
    )
    return car


def event_study_summary(
    abnormal: pd.DataFrame,
    *,
    window: tuple[int, int] = _EVENT_WINDOW_DEFAULT,
) -> pd.DataFrame:
    """按相对日汇总平均异常变化（AAR）与累积平均异常变化（CAAR）。

    Returns
    -------
    pandas.DataFrame
        列 ``relative_day`` / ``mean_abnormal`` / ``std_abnormal`` / ``n`` /
        ``t_stat`` / ``caar``。
    """
    if abnormal.empty:
        return pd.DataFrame(
            columns=["relative_day", "mean_abnormal", "std_abnormal", "n", "t_stat", "caar"]
        )
    subset = abnormal[abnormal["relative_day"].between(*window)].dropna(subset=["abnormal"])
    if subset.empty:
        return pd.DataFrame(
            columns=["relative_day", "mean_abnormal", "std_abnormal", "n", "t_stat", "caar"]
        )
    grouped = subset.groupby("relative_day")["abnormal"]
    summary = pd.DataFrame(
        {
            "mean_abnormal": grouped.mean(),
            "std_abnormal": grouped.std(),
            "n": grouped.count(),
        }
    ).reset_index()
    summary["t_stat"] = summary["mean_abnormal"] / (
        summary["std_abnormal"] / np.sqrt(summary["n"].clip(lower=1))
    )
    summary["caar"] = summary["mean_abnormal"].cumsum()
    return summary


def event_study(
    prices: pd.DataFrame | None,
    events: pd.DataFrame | None,
    *,
    window: tuple[int, int] = _EVENT_WINDOW_DEFAULT,
    estimation_window: tuple[int, int] = _ESTIMATION_WINDOW_DEFAULT,
    log_returns: bool = True,
) -> EventStudyResult:
    """事件研究的端到端入口。

    当 ``prices`` 缺失或数据不足以覆盖事件窗口时，返回
    ``status="insufficient_data"`` 的结果对象并说明限制，而不是抛出异常——
    这样研究流水线可以在缺少付费高频数据的环境中继续运行其余部分。
    """
    empty_abnormal = pd.DataFrame(
        columns=[
            "event_id",
            "relative_day",
            "date",
            "actual",
            "expected",
            "abnormal",
            "car",
            "model",
        ]
    )
    empty_car = pd.DataFrame(columns=["event_id", "car", "n_days"])
    empty_events = events if events is not None else pd.DataFrame()

    if prices is None or (isinstance(prices, pd.DataFrame) and prices.empty):
        return EventStudyResult(
            abnormal=empty_abnormal,
            car=empty_car,
            events=empty_events,
            status="insufficient_data",
            message=(
                "未提供高频行情数据（利差 / 汇率 / 股指）。事件研究框架已就绪，"
                "但主权利差与高频汇率多受版权限制，需要用户自行接入合法数据源。"
            ),
            meta={"window": window, "estimation_window": estimation_window},
        )
    if events is None or events.empty:
        return EventStudyResult(
            abnormal=empty_abnormal,
            car=empty_car,
            events=empty_events,
            status="insufficient_data",
            message="未提供评级行动事件表。请先由评级面板构造事件表（见 README）。",
            meta={"window": window, "estimation_window": estimation_window},
        )

    windows = prepare_event_windows(
        prices, events, window=window, estimation_window=estimation_window
    )
    if windows.empty:
        return EventStudyResult(
            abnormal=empty_abnormal,
            car=empty_car,
            events=empty_events,
            status="insufficient_data",
            message=(
                "行情数据与事件日期无法匹配（常见原因：行情未覆盖事件年份，"
                "或国家代码不一致）。请核对数据覆盖范围。"
            ),
            meta={"window": window, "estimation_window": estimation_window},
        )

    abnormal = abnormal_changes(
        windows, estimation_window=estimation_window, log_returns=log_returns
    )
    car = cumulative_abnormal_changes(abnormal, start=window[0], end=window[1])
    valid_models = abnormal["model"].isin(["market_model", "mean_adjusted"])
    share = float(valid_models.mean()) if len(abnormal) else 0.0
    status = "ok" if share >= 0.5 else "partial"
    message = ""
    if status == "partial":
        message = (
            "超过一半的事件窗口数据不足，无法估计正常收益；结果仅供参考，"
            "请检查行情数据的交易频率与覆盖区间。"
        )
    logger.info("事件研究完成：status=%s，事件数=%d", status, len(events))
    return EventStudyResult(
        abnormal=abnormal,
        car=car,
        events=empty_events,
        status=status,
        message=message,
        meta={
            "window": window,
            "estimation_window": estimation_window,
            "n_events": len(events),
            "n_valid_windows": int(valid_models.sum()),
        },
    )


def ratings_to_events(
    panel: pd.DataFrame,
    *,
    event_types: Sequence[str] = ("downgrade", "upgrade"),
    date_col: str = "action_date",
    entity_col: str = "country_iso3",
) -> pd.DataFrame:
    """从评级面板构造事件表，供 :func:`event_study` 使用。

    仅保留实际发生评级行动的观测（``rating_change`` 非空且方向匹配 ``event_types``）。
    """
    if "rating_change" not in panel.columns:
        raise KeyError("面板缺少 rating_change 列，请先用 build_country_year_panel 构造")
    frame = panel.copy()
    frame["event_type"] = np.select(
        [frame["rating_change"] > 0, frame["rating_change"] < 0],
        ["upgrade", "downgrade"],
        default="affirm",
    )
    frame = frame[frame["event_type"].isin(list(event_types))]
    if date_col in frame.columns:
        frame = frame[frame[date_col].notna()]
    keep = [
        c
        for c in (entity_col, "agency", date_col, "year", "event_type", "rating_change")
        if c in frame.columns
    ]
    frame = frame[keep].copy()
    if date_col in frame.columns:
        frame = frame.rename(columns={date_col: "event_date"})
    elif "year" in frame.columns:
        frame["event_date"] = pd.to_datetime(
            frame["year"].astype("Int64").astype(str) + "-12-31", errors="coerce"
        )
    return frame.reset_index(drop=True)
