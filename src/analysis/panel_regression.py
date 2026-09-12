"""宏观驱动因素的面板回归。

方法
----
1. **双向固定效应（two-way fixed effects）**：吸收不随时间变化的国家特征
   （地理、制度传统、文化）与全球共同冲击（金融危机、疫情）。
2. **聚类稳健标准误**：同一国家跨年的扰动项高度相关，标准误按国家聚类；
   若检验对象是机构层面行为，也可按国家×机构聚类。
3. **线性概率模型（LPM）** 作为二元因变量（是否下调 / 是否上调）的基准，
   并提供 Logit 作为稳健性检验；系数解释为边际效应的近似。

设计取舍
--------
* 优先使用 ``linearmodels.PanelOLS``（对固定效应与聚类方差的处理更规范）；
  若环境中缺少该依赖，则自动退化为「虚拟变量 + 聚类稳健 OLS」。二者系数完全
  一致，方差估计在小样本下略有差别——返回结果中的 ``method`` 字段会明确标注。
* 面板回归不追求预测精度，只用于识别相关性方向与量级；因果解释需谨慎。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.utils.logging_utils import get_logger

__all__ = [
    "RegressionResult",
    "compare_results",
    "fit_fixed_effects",
    "fit_logit",
    "fit_pooled_ols",
    "run_driver_regressions",
]

logger = get_logger(__name__)

try:  # pragma: no cover - 取决于环境
    from linearmodels.panel import PanelOLS as _PanelOLS

    _HAS_LINEARMODELS = True
except Exception:
    _PanelOLS = None  # type: ignore[assignment]
    _HAS_LINEARMODELS = False


@dataclass
class RegressionResult:
    """对 statsmodels / linearmodels 结果的轻量统一封装。"""

    name: str
    params: pd.Series
    std_errors: pd.Series
    tstats: pd.Series
    pvalues: pd.Series
    nobs: int
    r2: float
    method: str
    entity_effects: bool = False
    time_effects: bool = False
    cluster: str | None = None
    r2_within: float | None = None
    r2_between: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def conf_int(self) -> pd.DataFrame:
        z = 1.959963984540054
        return pd.DataFrame(
            {
                "lower": self.params - z * self.std_errors,
                "upper": self.params + z * self.std_errors,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        """把结果整理为可导出的系数表。"""
        frame = pd.DataFrame(
            {
                "coef": self.params,
                "std_err": self.std_errors,
                "t": self.tstats,
                "p_value": self.pvalues,
            }
        )
        frame["signif"] = np.where(
            frame["p_value"] < 0.01,
            "***",
            np.where(frame["p_value"] < 0.05, "**", np.where(frame["p_value"] < 0.1, "*", "")),
        )
        ci = self.conf_int
        frame["ci_lower"] = ci["lower"]
        frame["ci_upper"] = ci["upper"]
        frame.attrs["name"] = self.name
        frame.attrs["nobs"] = self.nobs
        frame.attrs["r2"] = self.r2
        frame.attrs["method"] = self.method
        return frame

    def summary_text(self) -> str:
        """生成人类可读的摘要（用于 notebook 与报告）。"""
        lines = [
            f"模型: {self.name}",
            f"方法: {self.method}"
            f"{' | 国家固定效应' if self.entity_effects else ''}"
            f"{' | 时间固定效应' if self.time_effects else ''}"
            f"{f' | 聚类: {self.cluster}' if self.cluster else ''}",
            f"观测数: {self.nobs} | R²: {self.r2:.4f}"
            + (f" | within R²: {self.r2_within:.4f}" if self.r2_within is not None else ""),
            "-" * 72,
        ]
        frame = self.to_frame()
        for index, row in frame.iterrows():
            lines.append(
                f"{index!s:<28} {row['coef']:>10.4f} {row['std_err']:>10.4f} "
                f"{row['t']:>8.2f} {row['p_value']:>8.4f} {row['signif']}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - 展示用
        return f"<RegressionResult {self.name!r} nobs={self.nobs} r2={self.r2:.3f}>"


def _clean_inputs(
    df: pd.DataFrame,
    y_col: str,
    x_cols: Sequence[str],
    entity_col: str | None,
    time_col: str | None,
    cluster_col: str | None,
) -> pd.DataFrame:
    needed = [y_col, *x_cols]
    for column in (entity_col, time_col, cluster_col):
        if column and column not in needed:
            needed.append(column)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"回归所需列不存在: {missing}")

    frame = df[needed].copy()
    for column in [y_col, *x_cols]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna()
    if frame.empty:
        raise ValueError("清洗后无可用观测，请检查缺失值或列名")
    return frame


def _build_design(
    frame: pd.DataFrame,
    x_cols: Sequence[str],
    entity_col: str | None,
    time_col: str | None,
    entity_effects: bool,
    time_effects: bool,
) -> pd.DataFrame:
    design = frame[list(x_cols)].astype("float64").copy()
    extras: list[pd.DataFrame] = []
    if entity_effects and entity_col:
        extras.append(
            pd.get_dummies(
                frame[entity_col].astype("category"), prefix="fe", drop_first=True, dtype=float
            )
        )
    if time_effects and time_col:
        extras.append(
            pd.get_dummies(
                frame[time_col].astype("category"), prefix="tte", drop_first=True, dtype=float
            )
        )
    if extras:
        design = pd.concat([design, *extras], axis=1)
    return sm.add_constant(design, has_constant="add")


def _within_r2(
    frame: pd.DataFrame, y_col: str, x_cols: Sequence[str], entity_col: str | None
) -> float | None:
    if not entity_col or entity_col not in frame.columns:
        return None
    y = frame[y_col].astype("float64")
    X = frame[list(x_cols)].astype("float64")
    groups = frame[entity_col]
    y_dm = y - y.groupby(groups).transform("mean")
    X_dm = X - X.groupby(groups).transform("mean")
    try:
        result = sm.OLS(y_dm, X_dm).fit()
        return float(result.rsquared)
    except Exception:
        return None


def fit_fixed_effects(
    df: pd.DataFrame,
    y_col: str,
    x_cols: Sequence[str],
    *,
    entity_col: str = "country_iso3",
    time_col: str = "year",
    cluster_col: str | None = None,
    entity_effects: bool = True,
    time_effects: bool = False,
    name: str | None = None,
) -> RegressionResult:
    """估计面板固定效应模型（默认世界银行标准做法：国家固定效应 + 国家聚类）。

    Parameters
    ----------
    df:
        面板数据（长表）。
    y_col:
        因变量列名，例如 ``rating_change`` / ``is_downgrade`` / ``is_upgrade``。
    x_cols:
        解释变量列名列表。
    entity_col, time_col:
        实体与时间标识列。
    cluster_col:
        聚类维度。``None`` 时默认按 ``entity_col`` 聚类。
    entity_effects, time_effects:
        是否纳入国家 / 时间固定效应。

    Returns
    -------
    RegressionResult
    """
    resolved_cluster = cluster_col or entity_col
    frame = _clean_inputs(df, y_col, x_cols, entity_col, time_col, resolved_cluster)
    model_name = name or f"{y_col} ~ {' + '.join(x_cols)}"

    if _HAS_LINEARMODELS and entity_col and time_col:
        try:
            panel = frame.set_index([entity_col, time_col])
            panel = panel[~panel.index.duplicated(keep="first")]
            y = panel[y_col].astype("float64")
            X = panel[list(x_cols)].astype("float64")
            mod = _PanelOLS(y, X, entity_effects=entity_effects, time_effects=time_effects)
            clusters = panel[[resolved_cluster]] if resolved_cluster != entity_col else None
            if clusters is not None and not clusters.empty:
                fitted = mod.fit(cov_type="clustered", clusters=clusters)
            else:
                fitted = mod.fit(cov_type="clustered", cluster_entity=True)
            params = fitted.params.astype("float64")
            return RegressionResult(
                name=model_name,
                params=params,
                std_errors=fitted.std_errors.astype("float64"),
                tstats=fitted.tstats.astype("float64"),
                pvalues=fitted.pvalues.astype("float64"),
                nobs=int(fitted.nobs),
                r2=float(fitted.rsquared),
                method="linearmodels.PanelOLS (clustered)",
                entity_effects=entity_effects,
                time_effects=time_effects,
                cluster=str(resolved_cluster),
                r2_within=(
                    float(getattr(fitted, "rsquared_within", np.nan))
                    if getattr(fitted, "rsquared_within", None) is not None
                    else _within_r2(frame, y_col, x_cols, entity_col)
                ),
                extra={
                    "n_entities": int(fitted.entity_info.total),
                    "cov_type": "clustered",
                },
            )
        except Exception as exc:
            logger.warning("PanelOLS 估计失败（%s），改用虚拟变量 + 聚类稳健 OLS", exc)

    design = _build_design(frame, x_cols, entity_col, time_col, entity_effects, time_effects)
    y = frame[y_col].astype("float64")
    kwargs: dict[str, Any] = {
        "cov_type": "cluster",
        "cov_kwds": {"groups": frame[resolved_cluster]},
    }
    fitted = sm.OLS(y, design).fit(**kwargs)
    params = fitted.params

    core = [c for c in params.index if not str(c).startswith(("fe_", "tte_"))]
    return RegressionResult(
        name=model_name,
        params=params.loc[core],
        std_errors=fitted.bse.loc[core],
        tstats=fitted.tvalues.loc[core],
        pvalues=fitted.pvalues.loc[core],
        nobs=int(fitted.nobs),
        r2=float(fitted.rsquared),
        method="statsmodels.OLS (dummies + clustered SE)",
        entity_effects=entity_effects,
        time_effects=time_effects,
        cluster=str(resolved_cluster),
        r2_within=_within_r2(frame, y_col, x_cols, entity_col),
        extra={"n_entities": int(frame[entity_col].nunique()) if entity_col else None},
    )


def fit_pooled_ols(
    df: pd.DataFrame,
    y_col: str,
    x_cols: Sequence[str],
    *,
    cluster_col: str | None = "country_iso3",
    name: str | None = None,
) -> RegressionResult:
    """混合 OLS（无固定效应），作为固定效应模型的对照基准。"""
    frame = _clean_inputs(df, y_col, x_cols, None, None, cluster_col)
    design = sm.add_constant(frame[list(x_cols)].astype("float64"), has_constant="add")
    y = frame[y_col].astype("float64")
    if cluster_col and cluster_col in frame.columns:
        fitted = sm.OLS(y, design).fit(cov_type="cluster", cov_kwds={"groups": frame[cluster_col]})
        method = "statsmodels.OLS (pooled, clustered SE)"
    else:
        fitted = sm.OLS(y, design).fit(cov_type="HC1")
        method = "statsmodels.OLS (pooled, HC1)"

    return RegressionResult(
        name=name or f"pooled: {y_col} ~ {' + '.join(x_cols)}",
        params=fitted.params.astype("float64"),
        std_errors=fitted.bse.astype("float64"),
        tstats=fitted.tvalues.astype("float64"),
        pvalues=fitted.pvalues.astype("float64"),
        nobs=int(fitted.nobs),
        r2=float(fitted.rsquared),
        method=method,
        cluster=cluster_col,
    )


def fit_logit(
    df: pd.DataFrame,
    y_col: str,
    x_cols: Sequence[str],
    *,
    cluster_col: str = "country_iso3",
    name: str | None = None,
    max_iter: int = 200,
) -> RegressionResult:
    """Logit 模型（因变量须为 0/1），聚类稳健标准误。

    用于二元因变量的稳健性检验。若完全分离导致不收敛，会抛出异常供调用方处理。
    """
    frame = _clean_inputs(df, y_col, x_cols, None, None, cluster_col)
    y = frame[y_col].astype("float64")
    if not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError(f"Logit 因变量必须为 0/1，实际取值: {sorted(np.unique(y))[:5]}")
    design = sm.add_constant(frame[list(x_cols)].astype("float64"), has_constant="add")
    model = sm.Logit(y, design)
    fitted = model.fit(
        disp=False,
        maxiter=max_iter,
        cov_type="cluster",
        cov_kwds={"groups": frame[cluster_col]},
    )
    params = fitted.params
    return RegressionResult(
        name=name or f"logit: {y_col} ~ {' + '.join(x_cols)}",
        params=params.astype("float64"),
        std_errors=fitted.bse.astype("float64"),
        tstats=fitted.tvalues.astype("float64"),
        pvalues=fitted.pvalues.astype("float64"),
        nobs=int(fitted.nobs),
        r2=float(getattr(fitted, "prsquared", np.nan)),
        method="statsmodels.Logit (clustered SE)",
        cluster=cluster_col,
    )


def run_driver_regressions(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    targets: Sequence[str] = ("rating_change", "is_downgrade", "is_upgrade"),
    entity_col: str = "country_iso3",
    time_col: str = "year",
    cluster_col: str | None = None,
    time_effects: bool = True,
) -> dict[str, RegressionResult]:
    """对多个因变量批量运行固定效应回归，返回 ``{因变量: 结果}``。"""
    results: dict[str, RegressionResult] = {}
    for target in targets:
        if target not in panel.columns:
            logger.warning("跳过因变量 %s：面板中不存在该列", target)
            continue
        try:
            results[target] = fit_fixed_effects(
                panel,
                target,
                feature_cols,
                entity_col=entity_col,
                time_col=time_col,
                cluster_col=cluster_col,
                entity_effects=True,
                time_effects=time_effects,
                name=f"{target} ~ 宏观驱动因素（双向固定效应）",
            )
        except Exception as exc:
            logger.warning("因变量 %s 的回归失败: %s", target, exc)
    return results


def compare_results(
    results: dict[str, RegressionResult] | Sequence[RegressionResult],
    *,
    only_significant: bool = False,
) -> pd.DataFrame:
    """把多个模型的系数并排展示（星号标注显著性）。"""
    items = list(results.values()) if isinstance(results, dict) else list(results)
    if not items:
        return pd.DataFrame()
    columns: dict[str, pd.Series] = {}
    for result in items:
        frame = result.to_frame()
        if only_significant:
            frame = frame[frame["p_value"] < 0.1]
        columns[result.name] = frame.apply(lambda row: f"{row['coef']:.3f}{row['signif']}", axis=1)
    table = pd.DataFrame(columns)
    table.index.name = "variable"
    return table
