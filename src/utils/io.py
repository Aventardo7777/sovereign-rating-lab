"""数据读写与完整性校验工具。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import resolve_path

__all__ = [
    "dataframe_fingerprint",
    "read_csv_safely",
    "read_json",
    "write_csv",
    "write_json",
]


def _as_path(path: str | Path) -> Path:
    return resolve_path(path)


def read_csv_safely(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """读取 CSV，自动处理编码与 ``utf-8-sig`` BOM。

    评级数据经常由 Excel 导出，容易带 BOM，这里统一处理。
    """
    resolved = _as_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {resolved}")
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            return pd.read_csv(resolved, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:  # pragma: no cover - 依赖具体文件
            last_error = exc
            continue
    raise UnicodeDecodeError(  # pragma: no cover - defensive
        "utf-8", b"", 0, 1, f"无法解码文件 {resolved}: {last_error}"
    )


def write_csv(df: pd.DataFrame, path: str | Path, *, index: bool = False, **kwargs: Any) -> Path:
    """写出 CSV（UTF-8 with BOM，兼容 Excel 直接打开）。"""
    resolved = _as_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(resolved, index=index, encoding="utf-8-sig", **kwargs)
    return resolved


def read_json(path: str | Path) -> Any:
    """读取 JSON 文件。"""
    resolved = _as_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {resolved}")
    with resolved.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(obj: Any, path: str | Path, *, indent: int = 2) -> Path:
    """写出 JSON 文件（UTF-8，保留中文）。"""
    resolved = _as_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent, default=str)
    return resolved


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """计算数据框内容指纹（用于复现性记录）。

    指纹对「列名顺序」和「行顺序」敏感；等价内容但行序不同会得到不同指纹，
    这是刻意的设计——研究流水线中的行序本身也是结果的一部分。
    """
    hasher = hashlib.sha256()
    hasher.update("|".join(map(str, df.columns)).encode("utf-8"))
    hasher.update(str(df.shape).encode("utf-8"))
    for column in df.columns:
        series = df[column]
        try:
            hasher.update(pd.util.hash_pandas_object(series, index=False).values.tobytes())
        except TypeError:  # pragma: no cover - 混合类型列
            hasher.update(series.astype(str).str.cat(sep="|").encode("utf-8"))
    return hasher.hexdigest()[:16]
