"""评级刻度统一映射（1-21）。

背景
----
S&P、Moody's、Fitch 的长期主权评级符号体系不同，直接比较会引入偏差。本模块把
三家机构的长期本币/外币主权评级映射到一个**统一的有序数值刻度 1-21**，
分值越高代表信用质量越好。

映射表
------
======  ============  ============  =======
score   S&P / Fitch   Moody's       档位
======  ============  ============  =======
21      AAA           Aaa           投资级
20      AA+           Aa1           投资级
19      AA            Aa2           投资级
18      AA-           Aa3           投资级
17      A+            A1            投资级
16      A             A2            投资级
15      A-            A3            投资级
14      BBB+          Baa1          投资级
13      BBB           Baa2          投资级
12      BBB-          Baa3          投资级（门槛）
11      BB+           Ba1           投机级
10      BB            Ba2           投机级
9       BB-           Ba3           投机级
8       B+            B1            投机级
7       B             B2            投机级
6       B-            B3            投机级
5       CCC+          Caa1          投机级
4       CCC           Caa2          投机级
3       CCC-          Caa3          投机级
2       CC            Ca            投机级
1       C / D / SD    C             违约
======  ============  ============  =======

要点
----
* S&P 的 ``D``（default）、``SD``（selective default）、``RD``（restricted default）
  与 ``C`` 统一映射到 1 分，从而把 S&P 的 22 个符号压缩到 21 档，
  与 Moody's 的 21 档一一对应（Moody's 的 ``C`` 亦为最低档）。
* 展望（outlook）与信用观察（credit watch）**不改变**评级分值，只作为独立字段使用；
  本模块提供 :func:`normalize_outlook` 做标准化。
* 短期评级（如 ``A-1+``、``P-1``、``F1``）不在本刻度范围内，返回 ``NaN``。
* 无法识别的输入返回 ``NaN`` 而**不抛异常**，并把原因记录在
  :func:`map_rating_with_reason` 的返回值中，便于数据清洗阶段审计。
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "INVESTMENT_GRADE_THRESHOLD",
    "MAX_SCORE",
    "MIN_SCORE",
    "RATING_BUCKETS",
    "RATING_SCALE_TABLE",
    "SCORE_TO_GRADE",
    "apply_rating_mapping",
    "clean_rating_text",
    "is_investment_grade",
    "map_rating",
    "map_rating_series",
    "map_rating_with_reason",
    "normalize_agency",
    "normalize_outlook",
    "rating_action_type",
    "rating_bucket",
    "rating_scale_table",
    "score_to_grade",
]

MIN_SCORE: int = 1
MAX_SCORE: int = 21
INVESTMENT_GRADE_THRESHOLD: int = 12
""">= 该分值为投资级（BBB-/Baa3 = 12）。"""

# ---------------------------------------------------------------------------
# 原始符号 -> 分值
# ---------------------------------------------------------------------------
SP_FITCH_TO_SCORE: dict[str, int] = {
    "AAA": 21,
    "AA+": 20,
    "AA": 19,
    "AA-": 18,
    "A+": 17,
    "A": 16,
    "A-": 15,
    "BBB+": 14,
    "BBB": 13,
    "BBB-": 12,
    "BB+": 11,
    "BB": 10,
    "BB-": 9,
    "B+": 8,
    "B": 7,
    "B-": 6,
    "CCC+": 5,
    "CCC": 4,
    "CCC-": 3,
    "CC": 2,
    "C": 1,
    # 违约类：统一到最低档
    "D": 1,
    "SD": 1,
    "RD": 1,
    "DD": 1,
    "DDD": 1,
    # 复合写法（部分数据源把 S&P 的最低档写作 C/D）
    "C/D": 1,
}

MOODYS_TO_SCORE: dict[str, int] = {
    "Aaa": 21,
    "Aa1": 20,
    "Aa2": 19,
    "Aa3": 18,
    "A1": 17,
    "A2": 16,
    "A3": 15,
    "Baa1": 14,
    "Baa2": 13,
    "Baa3": 12,
    "Ba1": 11,
    "Ba2": 10,
    "Ba3": 9,
    "B1": 8,
    "B2": 7,
    "B3": 6,
    "Caa1": 5,
    "Caa2": 4,
    "Caa3": 3,
    "Ca": 2,
    "C": 1,
    "D": 1,
}

SCORE_TO_GRADE: dict[int, dict[str, str]] = {
    21: {"S&P": "AAA", "Fitch": "AAA", "Moody's": "Aaa"},
    20: {"S&P": "AA+", "Fitch": "AA+", "Moody's": "Aa1"},
    19: {"S&P": "AA", "Fitch": "AA", "Moody's": "Aa2"},
    18: {"S&P": "AA-", "Fitch": "AA-", "Moody's": "Aa3"},
    17: {"S&P": "A+", "Fitch": "A+", "Moody's": "A1"},
    16: {"S&P": "A", "Fitch": "A", "Moody's": "A2"},
    15: {"S&P": "A-", "Fitch": "A-", "Moody's": "A3"},
    14: {"S&P": "BBB+", "Fitch": "BBB+", "Moody's": "Baa1"},
    13: {"S&P": "BBB", "Fitch": "BBB", "Moody's": "Baa2"},
    12: {"S&P": "BBB-", "Fitch": "BBB-", "Moody's": "Baa3"},
    11: {"S&P": "BB+", "Fitch": "BB+", "Moody's": "Ba1"},
    10: {"S&P": "BB", "Fitch": "BB", "Moody's": "Ba2"},
    9: {"S&P": "BB-", "Fitch": "BB-", "Moody's": "Ba3"},
    8: {"S&P": "B+", "Fitch": "B+", "Moody's": "B1"},
    7: {"S&P": "B", "Fitch": "B", "Moody's": "B2"},
    6: {"S&P": "B-", "Fitch": "B-", "Moody's": "B3"},
    5: {"S&P": "CCC+", "Fitch": "CCC+", "Moody's": "Caa1"},
    4: {"S&P": "CCC", "Fitch": "CCC", "Moody's": "Caa2"},
    3: {"S&P": "CCC-", "Fitch": "CCC-", "Moody's": "Caa3"},
    2: {"S&P": "CC", "Fitch": "CC", "Moody's": "Ca"},
    1: {"S&P": "C/D", "Fitch": "C/D", "Moody's": "C"},
}

RATING_BUCKETS: dict[str, tuple[int, int]] = {
    "AAA": (21, 21),
    "AA": (18, 20),
    "A": (15, 17),
    "BBB": (12, 14),
    "BB": (9, 11),
    "B": (6, 8),
    "CCC及以下": (1, 5),
}

_AGENCY_ALIASES: dict[str, str] = {
    "s&p": "S&P",
    "sp": "S&P",
    "s and p": "S&P",
    "standard & poor's": "S&P",
    "standard and poors": "S&P",
    "standard & poors": "S&P",
    "standardpoor": "S&P",
    "moody's": "Moody's",
    "moodys": "Moody's",
    "moody": "Moody's",
    "md": "Moody's",
    "fitch": "Fitch",
    "fitch ratings": "Fitch",
    "fitchratings": "Fitch",
    "fr": "Fitch",
    "fit": "Fitch",
}

# 明确表示「无评级」的记号
_NO_RATING_TOKENS: frozenset[str] = frozenset(
    {
        "",
        "NR",
        "N/R",
        "N.A.",
        "NA",
        "N/A",
        "WR",
        "WD",
        "W/D",
        "UNRATED",
        "NOT RATED",
        "NONE",
        "NULL",
        "NAN",
        "WITHDRAWN",
        "--",
        "-",
        "N.A",
    }
)

# 短期评级记号（不在 1-21 刻度内）。注意：不把裸 ``B`` / ``NP`` 纳入，
# 否则会把 S&P 的长期评级 ``B`` 误判为短期评级。
_SHORT_TERM_PATTERN = re.compile(
    r"^(?:A-[1-3]|P-[1-3]|F[1-3]|S[1-3]|D-[1-3]|R-[1-6])[+-]?$",
    re.IGNORECASE,
)

_PAREN_PATTERN = re.compile(r"[\(（][^)）]*[\)）]")
_SUFFIX_PATTERN = re.compile(r"[\*\u2020\u2021\u00b0\u00aa!]+")
_MULTISPACE = re.compile(r"\s+")
_TRAILING_MODIFIER = re.compile(
    # 基础评级部分：字母（可含 / 分隔的复合符号，如 S&P 的 "C/D"），可带 +/- 与数字。
    # 必须使用**非贪婪**匹配，否则 "AAAu" 会被整体吞进主体、导致映射失败。
    r"^(?P<base>[A-Za-z]{1,4}?(?:/[A-Za-z]{1,4}?)?[+-]?\d?)"
    # 可选的后缀修饰符：u(未邀约) / p(临时) / sf(结构化融资) / so(担保债务) 等。
    # 只允许真正的修饰符字母，而不是任意 u/p 组合。
    r"(?:\s*(?:u{1,2}|p{1,2}|sf|so|pi|glc|local|foreign))?$",
    re.IGNORECASE,
)

_OUTLOOK_MAP: dict[str, str] = {
    "positive": "Positive",
    "pos": "Positive",
    "positive outlook": "Positive",
    "outlook positive": "Positive",
    "ppy": "Positive",
    "正面": "Positive",
    "积极": "Positive",
    "negative": "Negative",
    "neg": "Negative",
    "negative outlook": "Negative",
    "outlook negative": "Negative",
    "npy": "Negative",
    "负面": "Negative",
    "消极": "Negative",
    "stable": "Stable",
    "sta": "Stable",
    "stb": "Stable",
    "outlook stable": "Stable",
    "stable outlook": "Stable",
    "稳定": "Stable",
    "developing": "Developing",
    "dev": "Developing",
    "evolving": "Developing",
    "uncertain": "Developing",
    "发展中": "Developing",
    "watch positive": "Watch-Positive",
    "positive watch": "Watch-Positive",
    "cw positive": "Watch-Positive",
    "creditwatch positive": "Watch-Positive",
    "watch pos": "Watch-Positive",
    "观察正面": "Watch-Positive",
    "watch negative": "Watch-Negative",
    "negative watch": "Watch-Negative",
    "cw negative": "Watch-Negative",
    "creditwatch negative": "Watch-Negative",
    "watch neg": "Watch-Negative",
    "观察负面": "Watch-Negative",
    "watch developing": "Watch-Developing",
    "developing watch": "Watch-Developing",
    "cw developing": "Watch-Developing",
    "观察发展中": "Watch-Developing",
}

_NO_OUTLOOK = frozenset({"", "NONE", "NAN", "NR", "NA", "N/A", "NULL", "NOT APPLICABLE"})


def _norm_key(value: Any) -> str:
    """把任意文本归一为「小写 + 单空格」形式，便于查表。"""
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text)
    return _MULTISPACE.sub(" ", text).strip()


def normalize_agency(agency: Any) -> str | None:
    """把机构名称别名归一为 ``"S&P"`` / ``"Moody's"`` / ``"Fitch"``。

    无法识别时返回 ``None``（调用方决定是丢弃还是保留原值）。
    """
    if agency is None or (isinstance(agency, float) and np.isnan(agency)):
        return None
    key = _norm_key(agency)
    if key in _AGENCY_ALIASES:
        return _AGENCY_ALIASES[key]
    compact = key.replace(" ", "")
    for alias, canonical in _AGENCY_ALIASES.items():
        if compact == alias.replace(" ", ""):
            return canonical
    return None


def clean_rating_text(raw: Any) -> str:
    """去除评级符号中的修饰符，返回核心评级文本。

    处理内容：括号注释（``(P)``/``(pi)``/``(sf)``）、上标字符、多余空白、
    以及紧随其后的单字母修饰符（``u`` 未邀约 / ``p`` 临时）。

    Examples
    --------
    >>> clean_rating_text("AA+u")
    'AA+'
    >>> clean_rating_text("(P)AAA")
    'AAA'
    >>> clean_rating_text("Baa1  (hyb)")
    'Baa1'
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    text = str(raw).replace("\u00a0", " ").strip()
    text = _PAREN_PATTERN.sub(" ", text)
    text = _SUFFIX_PATTERN.sub(" ", text)
    text = _MULTISPACE.sub(" ", text).strip()
    # 形如 "AA+ u" / "Baa1 p" 的尾部修饰符
    parts = text.split(" ")
    if len(parts) == 2 and parts[1].lower() in {"u", "p", "sf", "so", "pi", "glc"}:
        text = parts[0]
    match = _TRAILING_MODIFIER.match(text)
    if match and match.group("base"):
        text = match.group("base")
    return text.strip()


def _first_token(text: str) -> str:
    parts = text.split(" ")
    return parts[0] if parts else ""


def _lookup_text(text: str, agency: str | None) -> int | None:
    """在给定机构（或全部机构）的映射表中查找分值。"""
    if not text:
        return None
    upper = text.upper()
    candidates: list[str | None] = []
    if agency in (None, "S&P", "Fitch"):
        candidates.append(SP_FITCH_TO_SCORE.get(upper))
    if agency in (None, "Moody's"):
        candidates.append(MOODYS_TO_SCORE.get(text))
        candidates.append(MOODYS_TO_SCORE.get(text.title()))
    for value in candidates:
        if value is not None:
            return value
    return None


def map_rating_with_reason(
    rating: Any,
    agency: Any = None,
    *,
    allow_numeric: bool = True,
) -> tuple[float, str]:
    """映射评级并返回 ``(分值, 原因)``。

    原因取值：``"ok"``、``"missing"``、``"no_rating"``、``"short_term"``、
    ``"unmapped"``、``"numeric_out_of_range"``。
    """
    canonical_agency = normalize_agency(agency)

    if rating is None:
        return (np.nan, "missing")
    if isinstance(rating, float) and np.isnan(rating):
        return (np.nan, "missing")
    if isinstance(rating, (int, np.integer)) or (
        isinstance(rating, (float, np.floating)) and float(rating).is_integer()
    ):
        if allow_numeric:
            value = int(rating)
            if MIN_SCORE <= value <= MAX_SCORE:
                return (float(value), "ok")
            return (np.nan, "numeric_out_of_range")
        return (np.nan, "unmapped")

    text = str(rating).strip()
    if not text:
        return (np.nan, "missing")
    if text.upper() in _NO_RATING_TOKENS:
        return (np.nan, "no_rating")
    if _SHORT_TERM_PATTERN.match(text.replace(" ", "")):
        return (np.nan, "short_term")

    for candidate in (text, text.upper()):
        if candidate.upper() in _NO_RATING_TOKENS:
            return (np.nan, "no_rating")

    score = _lookup_text(text, canonical_agency)
    if score is not None:
        return (float(score), "ok")

    cleaned = clean_rating_text(text)
    if cleaned and cleaned.upper() in _NO_RATING_TOKENS:
        return (np.nan, "no_rating")
    score = _lookup_text(cleaned, canonical_agency)
    if score is not None:
        return (float(score), "ok")

    token = _first_token(cleaned)
    if token and token != cleaned:
        score = _lookup_text(token, canonical_agency)
        if score is not None:
            return (float(score), "ok")

    if canonical_agency is None:
        # 机构未知时退化为「跨机构搜索」，常见于混合来源的数据
        for target in ("S&P", "Moody's", "Fitch"):
            score = _lookup_text(cleaned or text, target)
            if score is not None:
                return (float(score), "ok")

    return (np.nan, "unmapped")


def map_rating(rating: Any, agency: Any = None, *, allow_numeric: bool = True) -> float:
    """把单个评级映射为 1-21 分值；无法映射时返回 ``np.nan``。

    Examples
    --------
    >>> map_rating("AAA", "S&P")
    21.0
    >>> map_rating("Baa3", "Moody's")
    12.0
    >>> map_rating("BBB+", "Fitch")
    14.0
    >>> import math; math.isnan(map_rating("NR", "S&P"))
    True
    """
    return map_rating_with_reason(rating, agency, allow_numeric=allow_numeric)[0]


def map_rating_series(
    ratings: pd.Series,
    agency: Any = None,
    *,
    allow_numeric: bool = True,
) -> pd.Series:
    """向量化映射一整列评级。

    ``agency`` 可以是标量、也可能是与 ``ratings`` 等长的 :class:`pandas.Series`
    （此时逐行按其机构映射）。返回 ``float`` 序列，索引与原序列一致。
    """
    if isinstance(agency, pd.Series):
        agency_aligned = agency.reindex(ratings.index)
        scores = [
            map_rating_with_reason(r, a, allow_numeric=allow_numeric)[0]
            for r, a in zip(ratings.to_numpy(), agency_aligned.to_numpy(), strict=False)
        ]
        return pd.Series(scores, index=ratings.index, name="rating_score", dtype="float64")
    scores = [map_rating_with_reason(r, agency, allow_numeric=allow_numeric)[0] for r in ratings]
    return pd.Series(scores, index=ratings.index, name="rating_score", dtype="float64")


def apply_rating_mapping(
    df: pd.DataFrame,
    rating_col: str = "rating",
    agency_col: str | None = "agency",
    *,
    out_col: str = "rating_score",
    reason_col: str | None = "rating_map_reason",
    inplace: bool = False,
) -> pd.DataFrame:
    """给数据框批量添加评级分值列。

    Parameters
    ----------
    df:
        含评级列的原始数据。
    rating_col:
        评级文本列名。
    agency_col:
        机构列名；为 ``None`` 时使用跨机构自动匹配。
    out_col:
        输出分值列名。
    reason_col:
        输出映射原因列名；为 ``None`` 时不输出。
    inplace:
        为 ``True`` 时直接修改传入的 ``df``。
    """
    if rating_col not in df.columns:
        raise KeyError(f"缺少评级列: {rating_col}")

    target = df if inplace else df.copy()

    if agency_col is None or agency_col not in df.columns:
        agencies: Any = None
    else:
        normalized = df[agency_col].map(lambda a: normalize_agency(a) or a)
        unknown = df.loc[normalized.isna(), agency_col].dropna().unique()
        if len(unknown):
            from src.utils.logging_utils import get_logger

            get_logger(__name__).debug("无法识别的机构名: %s", list(unknown)[:5])
        agencies = normalized

    pairs = [
        map_rating_with_reason(r, a)
        for r, a in zip(
            df[rating_col].to_numpy(),
            (agencies.to_numpy() if isinstance(agencies, pd.Series) else [agencies] * len(df)),
            strict=False,
        )
    ]
    target[out_col] = pd.Series([p[0] for p in pairs], index=df.index, dtype="float64")
    if reason_col:
        target[reason_col] = pd.Series([p[1] for p in pairs], index=df.index, dtype="object")
    return target


def normalize_outlook(outlook: Any) -> str | None:
    """标准化展望 / 信用观察状态。

    返回 ``"Positive"`` / ``"Negative"`` / ``"Stable"`` / ``"Developing"``
    / ``"Watch-Positive"`` / ``"Watch-Negative"`` / ``"Watch-Developing"``，
    缺失时返回 ``None``。

    Examples
    --------
    >>> normalize_outlook("negative")
    'Negative'
    >>> normalize_outlook("CW-Negative")
    'Watch-Negative'
    >>> normalize_outlook("n/a") is None
    True
    """
    if outlook is None or (isinstance(outlook, float) and np.isnan(outlook)):
        return None
    text = str(outlook).strip()
    if text.upper() in _NO_OUTLOOK:
        return None
    if text in {"+", "＋"}:
        return "Positive"
    if text in {"-", "－", "–"}:
        return "Negative"
    key = _norm_key(text)
    if not key:
        return None
    if key in _OUTLOOK_MAP:
        return _OUTLOOK_MAP[key]
    compact = key.replace(" ", "")
    for alias, canonical in _OUTLOOK_MAP.items():
        if compact == alias.replace(" ", ""):
            return canonical
    return None


def score_to_grade(score: Any, agency: str | None = "S&P") -> str | None:
    """把分值转回评级符号。

    Examples
    --------
    >>> score_to_grade(21, "S&P")
    'AAA'
    >>> score_to_grade(12, "Moody's")
    'Baa3'
    """
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return None
    try:
        value = round(float(score))
    except (TypeError, ValueError):
        return None
    if value < MIN_SCORE or value > MAX_SCORE:
        return None
    canonical = normalize_agency(agency) or "S&P"
    return SCORE_TO_GRADE[value].get(canonical, SCORE_TO_GRADE[value]["S&P"])


def is_investment_grade(score: Any, threshold: int = INVESTMENT_GRADE_THRESHOLD) -> bool:
    """判断是否达到投资级门槛。缺失值返回 ``False``（便于直接求和计数）。"""
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return False
    try:
        return float(score) >= threshold
    except (TypeError, ValueError):
        return False


def rating_bucket(score: Any) -> str | None:
    """把分值归入 7 个大档（AAA / AA / A / BBB / BB / B / CCC及以下）。"""
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return None
    try:
        value = round(float(score))
    except (TypeError, ValueError):
        return None
    for name, (low, high) in RATING_BUCKETS.items():
        if low <= value <= high:
            return name
    return None


def rating_action_type(delta: Any) -> str | None:
    """根据分值变化判断评级行动类型。

    Examples
    --------
    >>> rating_action_type(2)
    'upgrade'
    >>> rating_action_type(-3)
    'downgrade'
    >>> rating_action_type(0)
    'affirm'
    """
    if delta is None or (isinstance(delta, float) and np.isnan(delta)):
        return None
    try:
        value = float(delta)
    except (TypeError, ValueError):
        return None
    if value > 0:
        return "upgrade"
    if value < 0:
        return "downgrade"
    return "affirm"


def rating_scale_table() -> pd.DataFrame:
    """返回完整的 1-21 映射表（可直接导出为数据字典 / LaTeX 表格）。"""
    rows = []
    for score in range(MAX_SCORE, MIN_SCORE - 1, -1):
        entry = SCORE_TO_GRADE[score]
        rows.append(
            {
                "rating_score": score,
                "sp": entry["S&P"],
                "fitch": entry["Fitch"],
                "moodys": entry["Moody's"],
                "is_investment_grade": score >= INVESTMENT_GRADE_THRESHOLD,
                "bucket": rating_bucket(score),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "rating_score",
            "sp",
            "fitch",
            "moodys",
            "is_investment_grade",
            "bucket",
        ],
    )


RATING_SCALE_TABLE: pd.DataFrame = rating_scale_table()
"""模块级常量形式的映射表（与 :func:`rating_scale_table` 内容一致）。"""
