"""主权信用评级研究仪表盘（Streamlit）。

启动
----
.. code-block:: bash

    streamlit run app/streamlit_app.py

功能
----
* 选择国家 / 机构 / 年份区间，查看评级轨迹与投资级门槛；
* 迁移矩阵热力图（频数 / 概率可切换）；
* 宏观驱动指标的时间序列与评级对照；
* 预测概率（若已运行流水线生成 OOF 预测，则直接读取；否则提示先运行流水线）；
* 下载处理后的面板数据与关键表格。

数据说明：默认读取 ``data/processed/panel_country_year.csv``。若该文件不存在，
应用会自动从示例数据（**合成演示数据**）构建，并在页面顶部给出警示。
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# --- 让脚本在任意工作目录下都能导入 src -------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.migration import (  # noqa: E402
    agency_dispersion,
    build_migration_matrix,
    migration_summary,
    upgrade_downgrade_probabilities,
)
from src.config import get_path  # noqa: E402
from src.features.rating_scale import score_to_grade  # noqa: E402
from src.visualization.interactive import (  # noqa: E402
    migration_heatmap_plotly,
    rating_trajectory_plotly,
)

st.set_page_config(
    page_title="主权信用评级研究 | sovereign-rating-lab",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_WARNING = (
    "⚠️ **当前展示的是合成演示数据**（`data/sample/*.csv`，provenance = "
    "`synthetic_demo_only`）。数据由 `scripts/make_sample_data.py` 生成，"
    "**不是真实主权评级**，仅用于验证分析流程，不得用于任何实证结论。"
)


# ---------------------------------------------------------------------------
# 数据载入
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def available_assets() -> list[str]:
    """列出可用的数据资产。"""
    assets = []
    if (get_path("processed") / "panel_country_year.csv").exists():
        assets.append("processed")
    if (get_path("sample") / "ratings_sample.csv").exists():
        assets.append("sample")
    return assets


@st.cache_data(show_spinner="正在载入数据…")
def load_panel() -> tuple[pd.DataFrame, str]:
    """载入面板数据，返回 ``(面板, 数据来源说明)``。"""
    processed = get_path("processed") / "panel_country_year.csv"
    if processed.exists():
        frame = pd.read_csv(processed, encoding="utf-8-sig")
        for column in ("rating", "agency", "country_name", "outlook"):
            if column in frame.columns:
                frame[column] = frame[column].astype("string")
        return frame, "processed"

    # 回退：从示例数据即时构建
    from src.clean.panel import build_country_year_panel
    from src.ingest.ratings import load_sample_ratings

    ratings = load_sample_ratings()
    macro_path = get_path("sample") / "macro_sample.csv"
    macro = pd.read_csv(macro_path, encoding="utf-8-sig") if macro_path.exists() else None
    frame = build_country_year_panel(ratings, macro, start_year=2000, end_year=2023)
    return frame, "sample"


@st.cache_data(show_spinner=False)
def load_oof_predictions() -> pd.DataFrame | None:
    """载入有序 Logit 的样本外预测概率（若已运行流水线）。"""
    path = get_path("tables") / "oof_predictions_ordinal_logit.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, encoding="utf-8-sig")
    return frame


def _download_button(frame: pd.DataFrame, label: str, filename: str, key: str) -> None:
    buffer = io.BytesIO()
    frame.to_csv(buffer, index=False, encoding="utf-8-sig")
    st.download_button(
        label=label,
        data=buffer.getvalue(),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


# ---------------------------------------------------------------------------
# 主界面
# ---------------------------------------------------------------------------
def main() -> None:
    panel, source = load_panel()

    st.title("🌐 主权信用评级迁移与预测研究")
    st.caption(
        "sovereign-rating-lab · 统一 1-21 评级刻度 · 迁移矩阵 · 宏观驱动 · "
        "有序概率模型 · SHAP · 事件研究"
    )

    if source == "sample" or (
        "provenance" in panel.columns
        and (panel["provenance"].astype(str) == "synthetic_demo_only").any()
    ):
        st.warning(SAMPLE_WARNING)
        st.caption(
            "要使用真实数据：把评级 CSV 放到 `data/raw/`，按 `README.md` "
            "「如何导入评级数据」章节导入后重新运行 `python -m src.pipeline`。"
        )

    required = {"country_iso3", "agency", "year"}
    missing = required - set(panel.columns)
    if missing:
        st.error(
            f"面板缺少必需列 {sorted(missing)}，无法构建界面。"
            "请先运行 `python -m src.pipeline` 生成标准面板。"
        )
        st.stop()

    # ------------------------------------------------------------ 侧边栏
    st.sidebar.header("筛选条件")
    countries = sorted(panel["country_iso3"].dropna().unique().tolist())
    name_lookup = (
        panel.dropna(subset=["country_iso3"])
        .drop_duplicates("country_iso3")
        .set_index("country_iso3")["country_name"]
        .to_dict()
        if "country_name" in panel.columns
        else {}
    )
    labels = {code: f"{code} — {name_lookup.get(code, code)}" for code in countries}

    selected_country = st.sidebar.selectbox(
        "国家 / 地区",
        options=countries,
        index=0,
        format_func=lambda c: labels.get(c, c),
        help="选择要查看的主权发行体",
    )
    all_agencies = sorted(panel["agency"].dropna().unique().tolist())
    selected_agencies = st.sidebar.multiselect(
        "评级机构", options=all_agencies, default=all_agencies
    )
    year_min = int(panel["year"].min())
    year_max = int(panel["year"].max())
    year_range = st.sidebar.slider(
        "年份区间", min_value=year_min, max_value=year_max, value=(year_min, year_max)
    )
    view_mode = st.sidebar.radio("迁移矩阵视图", ["概率", "频数"], horizontal=True)

    filtered = panel[(panel["year"].between(*year_range))]
    if selected_agencies:
        filtered = filtered[filtered["agency"].isin(selected_agencies)]

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**面板规模**：{len(panel):,} 行 · {panel['country_iso3'].nunique()} 个经济体 · "
        f"{panel['agency'].nunique()} 家机构 · {year_min}-{year_max} 年"
    )
    st.sidebar.markdown(
        "**免责声明**：本工具仅用于研究与教学。评级数据的所有权利归各评级机构所有，"
        "使用者需自行确认数据使用权限。"
    )

    # ------------------------------------------------------------ 概览
    st.subheader("1 · 概览")
    metric_cols = st.columns(4)
    metric_cols[0].metric("观测数（筛选后）", f"{len(filtered):,}")
    metric_cols[1].metric("经济体", f"{filtered['country_iso3'].nunique()}")
    metric_cols[2].metric(
        "评级行动次数", f"{int(filtered['has_action'].sum()) if 'has_action' in filtered else 0:,}"
    )
    score_col = "score_in_effect" if "score_in_effect" in filtered.columns else "rating_score"
    mean_score = filtered[score_col].mean()
    metric_cols[3].metric(
        "平均评级", f"{score_to_grade(mean_score, 'S&P') or '—'}" if pd.notna(mean_score) else "—"
    )

    # ------------------------------------------------------------ 轨迹
    st.subheader("2 · 评级轨迹")
    trajectory = rating_trajectory_plotly(
        filtered, country=selected_country, agencies=selected_agencies or None
    )
    st.plotly_chart(trajectory, use_container_width=True)

    with st.expander("查看该国的评级明细"):
        detail = filtered[filtered["country_iso3"] == selected_country].copy()
        detail = (
            detail[detail.get("has_action", True).fillna(False)]
            if "has_action" in detail
            else detail
        )
        keep = [
            c
            for c in (
                "year",
                "agency",
                "rating",
                "outlook",
                "action",
                "action_date",
                "rating_change",
            )
            if c in detail.columns
        ]
        st.dataframe(
            detail[keep].sort_values(["year", "agency"]), use_container_width=True, hide_index=True
        )

    # ------------------------------------------------------------ 迁移矩阵
    st.subheader("3 · 评级迁移矩阵")
    left, right = st.columns((3, 2))
    matrix = build_migration_matrix(
        filtered,
        from_year=year_range[0],
        to_year=year_range[1],
        score_is_effective="score_in_effect" in filtered.columns,
    )
    normalize = view_mode == "概率"
    with left:
        st.plotly_chart(
            migration_heatmap_plotly(matrix, normalize=normalize), use_container_width=True
        )
    with right:
        summary = migration_summary(matrix)
        st.markdown("**总体迁移指标**")
        st.dataframe(
            pd.DataFrame(
                {
                    "指标": ["迁移观测数", "稳定概率", "上调概率", "下调概率", "平均变动（档）"],
                    "数值": [
                        f"{summary['n_transitions']:.0f}",
                        f"{summary['p_stable']:.4f}" if pd.notna(summary["p_stable"]) else "—",
                        f"{summary['p_upgrade']:.4f}" if pd.notna(summary["p_upgrade"]) else "—",
                        (
                            f"{summary['p_downgrade']:.4f}"
                            if pd.notna(summary["p_downgrade"])
                            else "—"
                        ),
                        f"{summary['mean_delta']:.4f}" if pd.notna(summary["mean_delta"]) else "—",
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**按机构的上调 / 下调概率**")
        by_agency = upgrade_downgrade_probabilities(filtered, group_cols=["agency"])
        if not by_agency.empty:
            st.dataframe(
                by_agency[
                    ["agency", "n_transitions", "p_upgrade", "p_downgrade", "p_stable"]
                ].round(4),
                use_container_width=True,
                hide_index=True,
            )
        _download_button(
            matrix.reset_index(),
            "下载迁移矩阵 CSV",
            f"migration_matrix_{year_range[0]}_{year_range[1]}.csv",
            "dl_matrix",
        )

    # ------------------------------------------------------------ 机构分歧
    st.subheader("4 · 机构间评级分歧")
    dispersion = agency_dispersion(filtered)
    if not dispersion.empty:
        col_a, col_b = st.columns((2, 3))
        with col_a:
            split_by_year = (
                dispersion.groupby("year")
                .agg(share_split=("is_split", "mean"), mean_range=("range_score", "mean"))
                .reset_index()
            )
            st.markdown("**分歧比例的时间趋势**")
            st.line_chart(split_by_year.set_index("year"), height=260)
        with col_b:
            st.markdown("**分歧最大的国家-年份**")
            top_split = dispersion.sort_values(["range_score", "std_score"], ascending=False).head(
                12
            )
            st.dataframe(top_split.round(3), use_container_width=True, hide_index=True)
        _download_button(
            dispersion, "下载机构分歧明细 CSV", "agency_dispersion.csv", "dl_dispersion"
        )

    # ------------------------------------------------------------ 宏观指标
    st.subheader("5 · 宏观驱动指标")
    macro_candidates = [
        c
        for c in (
            "gdp_growth",
            "inflation",
            "gov_debt_gdp",
            "external_debt_gni",
            "reserves_months_imports",
            "current_account_gdp",
            "exports_gdp",
            "fx_depreciation",
            "real_interest_rate",
            "spread_bps",
            "voice_accountability",
            "political_stability",
            "government_effectiveness",
            "regulatory_quality",
            "rule_of_law",
            "control_corruption",
        )
        if c in filtered.columns
    ]
    if macro_candidates:
        chosen = st.multiselect(
            "选择要展示的指标（最多 6 个）",
            options=macro_candidates,
            default=macro_candidates[:4],
            max_selections=6,
        )
        country_macro = (
            filtered[filtered["country_iso3"] == selected_country].groupby("year")[chosen].mean()
            if chosen
            else pd.DataFrame()
        )
        if not country_macro.empty:
            st.line_chart(country_macro, height=320)
            st.caption(
                "数值为该国在所选年份下各机构记录的均值（宏观指标与国家-年份唯一，"
                "除以机构数为去重后的等价操作）。"
            )
    else:
        st.info("面板中没有可用的宏观指标列。请先运行 `python -m src.pipeline`。")

    # ------------------------------------------------------------ 预测概率
    st.subheader("6 · 预测下调 / 上调概率")
    predictions = load_oof_predictions()
    if predictions is None:
        st.info(
            "尚未生成样本外预测。请先运行完整流水线：\n\n"
            "```bash\npython -m src.pipeline\n```\n\n"
            "流水线会输出各模型的时间序列交叉验证结果与样本外概率"
            "（`reports/tables/oof_predictions_*.csv`）。"
        )
    elif "country_iso3" not in predictions.columns or "year" not in predictions.columns:
        st.warning("预测文件缺少 country_iso3 / year 列，无法按国家筛选。")
    else:
        country_pred = predictions[predictions["country_iso3"] == selected_country].sort_values(
            "year"
        )
        if country_pred.empty:
            st.info(f"预测结果中没有 {selected_country} 的记录（该国可能缺少滞后期观测）。")
        else:
            table = pd.DataFrame(index=country_pred["year"])
            table.index.name = "年份"
            if "p_downgrade" in country_pred.columns:
                table["下调概率"] = pd.to_numeric(country_pred["p_downgrade"], errors="coerce")
                table["稳定概率"] = pd.to_numeric(country_pred["p_stable"], errors="coerce")
                table["上调概率"] = pd.to_numeric(country_pred["p_upgrade"], errors="coerce")
            table["预测档位"] = country_pred["y_pred"].map(
                lambda v: score_to_grade(v, "S&P") or str(v)
            )
            table["实际档位"] = country_pred["y_true"].map(
                lambda v: score_to_grade(v, "S&P") or str(v)
            )
            if "prior_score" in country_pred.columns:
                table["预测前档位"] = country_pred["prior_score"].map(
                    lambda v: score_to_grade(v, "S&P") or str(v)
                )
            st.dataframe(table, use_container_width=True)
            if {"下调概率", "上调概率"}.issubset(table.columns):
                st.line_chart(table[["下调概率", "上调概率"]], height=300)
            st.caption(
                "概率来自**有序 Logit** 的前向链式交叉验证（严格样本外）。"
                "「下调概率」= 预测分布中低于预测前评级的所有档位概率之和；"
                "「上调概率」同理。模型对比见 `reports/tables/cv_model_comparison.csv`。"
            )

    # ------------------------------------------------------------ 下载
    st.subheader("7 · 下载数据")
    dl_cols = st.columns(3)
    with dl_cols[0]:
        _download_button(filtered, "下载当前筛选面板 CSV", "panel_filtered.csv", "dl_filtered")
    with dl_cols[1]:
        _download_button(panel, "下载完整面板 CSV", "panel_full.csv", "dl_full")
    with dl_cols[2]:
        _download_button(
            upgrade_downgrade_probabilities(filtered, group_cols=["year"]),
            "下载年度迁移概率 CSV",
            "probability_by_year.csv",
            "dl_probyear",
        )

    st.markdown("---")
    st.caption(
        "sovereign-rating-lab · MIT License · "
        "本项目不打包任何第三方评级机构的版权数据，"
        "使用者需自行确认数据来源的合法性与使用权限。"
    )


if __name__ == "__main__":
    main()
