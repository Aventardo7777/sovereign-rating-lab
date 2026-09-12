"""主权评级数据的载入、解析与校验。

**版权与合规声明（务必阅读）**
------------------------------
S&P Global Ratings、Moody's、Fitch Ratings 的评级历史数据库均为**商业授权产品**，
不提供免费的完整历史下载接口。本项目：

1. **不伪造、不打包**任何完整的真实评级历史数据；
2. 提供 :file:`data/raw/ratings_template.csv` 作为**导入模板**，用户导入自己拥有
   合法使用权的数据；
3. 提供 :file:`data/sample/ratings_sample.csv` 作为**演示用合成数据**，
   文件内含 ``provenance = synthetic_demo_only`` 标记列，**不是真实评级数据**；
4. 提供**合规的公开页面解析脚本**（:func:`fetch_public_ratings_page` /
   :func:`parse_ratings_html`）：抓取前检查 ``robots.txt``、限速、缓存原始 HTML，
   并在 provenance 登记表中记录来源 URL 与下载时间。

**用户责任**：启用任何抓取功能前，请自行确认目标站点的服务条款与版权许可。
本项目不对用户导入或抓取的数据的合法性负责。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_config, get_path, resolve_path
from src.features.rating_scale import apply_rating_mapping, normalize_agency, normalize_outlook
from src.utils.http import HttpClient
from src.utils.logging_utils import get_logger

__all__ = [
    "REQUIRED_RATING_COLUMNS",
    "SAMPLE_PROVENANCE_TAG",
    "fetch_public_ratings_page",
    "load_ratings_csv",
    "load_sample_ratings",
    "parse_ratings_html",
    "validate_imported_ratings",
    "write_provenance_manifest",
]

logger = get_logger(__name__)

REQUIRED_RATING_COLUMNS: tuple[str, ...] = ("country_iso3", "year", "agency", "rating")
""":func:`validate_imported_ratings` 要求的最小列集合。"""

OPTIONAL_RATING_COLUMNS: tuple[str, ...] = (
    "country_name",
    "outlook",
    "action",
    "action_date",
    "rating_score",
    "source_url",
    "provenance",
)

SAMPLE_PROVENANCE_TAG = "synthetic_demo_only"
"""示例数据的溯源标记——出现该值的行**不是真实评级数据**。"""


def load_ratings_csv(
    path: str | Path,
    *,
    mapping_col: str = "rating",
    agency_col: str = "agency",
    standardize: bool = True,
) -> pd.DataFrame:
    """载入用户导入的评级 CSV，并完成列标准化与评级分值映射。

    Parameters
    ----------
    path:
        CSV 路径（相对路径按项目根目录解析）。
    mapping_col, agency_col:
        用于映射 1-21 分值的列名。
    standardize:
        是否对展望列做标准化（``normalize_outlook``）。
    """
    from src.utils.io import read_csv_safely

    resolved = resolve_path(path)
    frame = read_csv_safely(resolved)
    frame = _standardize_rating_columns(frame)

    if mapping_col not in frame.columns:
        raise KeyError(
            f"评级数据缺少列 {mapping_col!r}。请参考 data/raw/ratings_template.csv 的列结构。"
        )
    if agency_col in frame.columns:
        frame[agency_col] = frame[agency_col].map(lambda a: normalize_agency(a) or a)
    elif agency_col != mapping_col:
        logger.warning("评级数据缺少机构列 %r，将使用跨机构自动匹配", agency_col)
    frame = apply_rating_mapping(
        frame,
        rating_col=mapping_col,
        agency_col=agency_col if agency_col in frame.columns else None,
        out_col="rating_score",
        reason_col="rating_map_reason",
    )
    if standardize and "outlook" in frame.columns:
        frame["outlook"] = frame["outlook"].map(normalize_outlook)
    if "year" in frame.columns:
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    if "action_date" in frame.columns:
        frame["action_date"] = pd.to_datetime(frame["action_date"], errors="coerce")

    unmapped = int(frame["rating_score"].isna().sum())
    if unmapped:
        logger.warning(
            "有 %d 条评级无法映射到 1-21 刻度（占 %.1f%%），请检查 rating_map_reason 列",
            unmapped,
            100 * unmapped / max(len(frame), 1),
        )
    return frame


def _standardize_rating_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """统一评级数据列名（复用面板模块的别名表）。"""
    from src.clean.panel import standardize_columns

    return standardize_columns(frame)


def load_sample_ratings() -> pd.DataFrame:
    """载入演示用合成评级数据。

    .. warning::
       返回的数据是**为演示流程而人工生成的合成数据**，不代表任何真实国家、
       真实机构或真实评级的实际状况。请勿用于任何实证结论。
    """
    path = get_path("sample") / "ratings_sample.csv"
    frame = load_ratings_csv(path)
    if "provenance" in frame.columns:
        tags = set(frame["provenance"].dropna().unique())
        if tags and tags != {SAMPLE_PROVENANCE_TAG}:
            logger.warning("示例数据的 provenance 标记异常: %s", tags)
    else:
        logger.warning("示例数据缺少 provenance 列，无法确认其合成属性")
    logger.info(
        "已载入示例评级数据 %d 行（%s）。该数据仅用于演示，不是完整或真实的评级历史。",
        len(frame),
        SAMPLE_PROVENANCE_TAG,
    )
    return frame


def parse_ratings_html(
    html: str,
    *,
    source_url: str,
    table_selector: str = "table",
    column_map: Mapping[str, str] | None = None,
    agency: str | None = None,
    table_index: int = 0,
) -> pd.DataFrame:
    """解析评级公开页面中的 HTML 表格。

    Parameters
    ----------
    html:
        页面 HTML 文本。
    source_url:
        来源 URL（写入 ``source_url`` 列，满足溯源要求）。
    table_selector:
        CSS 选择器。默认 ``"table"``（取第一个匹配的表格）。
    column_map:
        原始表头 -> 标准列名的映射，例如
        ``{"Country": "country_name", "Rating": "rating", "Outlook": "outlook"}``。
        未在映射中的列会被丢弃，以保证输出结构统一。
    agency:
        机构名称，会写入 ``agency`` 列。
    table_index:
        当选择器匹配多个表格时，选取第几个。

    Returns
    -------
    pandas.DataFrame
        含 ``source_url`` / ``downloaded_at`` 溯源列，以及映射后的标准列。
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要 beautifulsoup4：pip install beautifulsoup4 lxml") from exc

    soup = BeautifulSoup(html, "lxml")
    tables = soup.select(table_selector)
    if not tables:
        logger.warning("页面中未找到匹配 %r 的表格", table_selector)
        return pd.DataFrame()
    if table_index >= len(tables):
        logger.warning("页面共有 %d 个匹配表格，请求的索引 %d 越界", len(tables), table_index)
        return pd.DataFrame()

    raw = pd.read_html(str(tables[table_index]))[0]
    raw.columns = [str(c).strip() for c in raw.columns]

    mapping = dict(column_map) if column_map else {}
    if mapping:
        keep = [c for c in raw.columns if c in mapping]
        if not keep:
            logger.warning(
                "column_map 与页面表头完全不匹配。页面表头为 %s，映射键为 %s",
                list(raw.columns)[:10],
                list(mapping)[:10],
            )
            return pd.DataFrame()
        frame = raw[keep].rename(columns=mapping)
    else:
        frame = raw.copy()

    frame["source_url"] = source_url
    frame["downloaded_at"] = datetime.now(tz=UTC).isoformat()
    if agency:
        frame["agency"] = normalize_agency(agency) or agency
    logger.info("从 %s 解析出 %d 行评级记录", source_url, len(frame))
    return frame


def fetch_public_ratings_page(
    url: str,
    *,
    table_selector: str = "table",
    column_map: Mapping[str, str] | None = None,
    agency: str | None = None,
    client: HttpClient | None = None,
) -> pd.DataFrame:
    """抓取并解析公开评级页面（合规流程）。

    合规保证
    --------
    1. 抓取前检查 ``robots.txt``；被禁止时抛出 :class:`RobotsDisallowedError`；
    2. 请求间隔不小于 ``scraping.min_delay_seconds``；
    3. 原始 HTML 缓存到 ``data/raw/cache/html``；
    4. 来源 URL、抓取时间、状态码写入 ``data/raw/cache/provenance.json``。

    .. warning::
       使用前请确认目标站点的服务条款允许自动化抓取与再分发。
       本项目不对数据使用权限负责。
    """
    cfg = get_config()
    client = client or HttpClient(cfg)
    client.respect_robots = bool(cfg.get("scraping", {}).get("respect_robots_txt", True))

    result = client.get(url, check_robots=client.respect_robots)
    return parse_ratings_html(
        result.text,
        source_url=result.url,
        table_selector=table_selector,
        column_map=column_map,
        agency=agency,
    )


def validate_imported_ratings(
    frame: pd.DataFrame,
    *,
    max_unmapped_share: float = 0.05,
    require_consecutive_years: bool = True,
) -> pd.DataFrame:
    """对导入的评级数据运行质量检查。

    Returns
    -------
    pandas.DataFrame
        每行一项检查：``check`` / ``n_violations`` / ``passed`` / ``detail``。
    """
    checks: list[dict[str, Any]] = []

    def add(name: str, n: int, detail: str = "") -> None:
        checks.append(
            {"check": name, "n_violations": int(n), "passed": int(n) == 0, "detail": detail}
        )

    missing = [c for c in REQUIRED_RATING_COLUMNS if c not in frame.columns]
    if missing:
        add("必需列齐全", len(missing), f"缺少列: {missing}")
        return pd.DataFrame(checks, columns=["check", "n_violations", "passed", "detail"])

    add("必需列齐全", 0, "")

    if "rating_score" not in frame.columns:
        add("评级分值映射完成", len(frame), "尚未调用 apply_rating_mapping")
    else:
        n_unmapped = int(frame["rating_score"].isna().sum())
        share = n_unmapped / max(len(frame), 1)
        checks.append(
            {
                "check": "评级分值映射完成",
                "n_violations": n_unmapped,
                "passed": share <= max_unmapped_share,
                "detail": f"未映射占比 {share:.2%}（阈值 {max_unmapped_share:.0%}）",
            }
        )

    unknown_agency = frame["agency"].isna().sum() if "agency" in frame.columns else len(frame)
    add("机构名称可识别", int(unknown_agency), "无法归一为 S&P / Moody's / Fitch")

    year_na = int(pd.to_numeric(frame.get("year"), errors="coerce").isna().sum())
    add("年份可解析", year_na, "年份缺失或非数值")

    if "country_iso3" in frame.columns:
        bad_iso = int(frame["country_iso3"].astype(str).str.len().ne(3).sum())
        add("国家代码为 3 位 ISO3", bad_iso, "长度不等于 3 的国家代码数")

    if require_consecutive_years and {"country_iso3", "agency", "year"}.issubset(frame.columns):
        sub = frame.dropna(subset=["country_iso3", "agency", "year"])
        grouped = sub.groupby(["country_iso3", "agency"])["year"].apply(
            lambda s: sorted(pd.unique(s.astype(int)))
        )
        gaps = 0
        for years in grouped:
            if len(years) > 1:
                expected = set(range(min(years), max(years) + 1))
                gaps += len(expected - set(years))
        checks.append(
            {
                "check": "年份连续（无缺口）",
                "n_violations": gaps,
                "passed": gaps == 0,
                "detail": "缺口年份会削弱迁移矩阵的有效样本量（属于提示性检查，非致命错误）",
            }
        )

    duplicate_keys = int(frame.duplicated(subset=["country_iso3", "agency", "year"]).sum())
    add("主键 (国家, 机构, 年份) 唯一", duplicate_keys, "重复的年-机构观测数")

    result = pd.DataFrame(checks, columns=["check", "n_violations", "passed", "detail"])
    failed = result[~result["passed"]]
    if not failed.empty:
        logger.warning("评级数据校验发现 %d 项问题：%s", len(failed), list(failed["check"]))
    return result


def write_provenance_manifest(extra: Mapping[str, Any] | None = None) -> Path:
    """把抓取登记表复制到 ``data/raw``，作为数据来源附录的素材。"""
    from src.utils.io import read_json, write_json

    cfg = get_config()
    source = resolve_path(
        cfg.get("scraping", {}).get("provenance_file", "data/raw/cache/provenance.json")
    )
    target = get_path("raw") / "provenance_manifest.json"
    entries: Sequence[Any] = []
    if source.exists():
        loaded = read_json(source)
        if isinstance(loaded, list):
            entries = loaded
    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "project": cfg.get("project", {}).get("name", "sovereign-rating-lab"),
        "n_entries": len(entries),
        "entries": list(entries),
    }
    if extra:
        payload.update(dict(extra))
    write_json(payload, target)
    logger.info("已写出数据溯源清单: %s", target)
    return target
