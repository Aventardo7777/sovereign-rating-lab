"""全局配置加载与路径管理.

本模块是项目中**唯一**允许读取 ``config.yaml`` 的地方。其余模块应通过
:func:`get_config` / :func:`get_path` 访问配置，从而保证：

1. 路径解析统一基于项目根目录，脚本在任意工作目录下运行都能找到数据；
2. 配置只解析一次（LRU 缓存），避免重复 IO；
3. 所有输出目录在 :func:`ensure_directories` 中被自动创建。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# 项目根目录 = 本文件（src/config.py）的上两层
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "PROJECT_ROOT",
    "ensure_directories",
    "get_config",
    "get_env",
    "get_path",
    "load_config",
    "reload_config",
    "resolve_path",
]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """从磁盘读取 YAML 配置。

    Parameters
    ----------
    path:
        配置文件路径。默认为项目根目录下的 ``config.yaml``。

    Returns
    -------
    dict
        解析后的配置字典。
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件格式错误，顶层应为映射: {config_path}")
    return cfg


@lru_cache(maxsize=4)
def get_config(path: str | None = None) -> dict[str, Any]:
    """返回缓存的配置字典。"""
    return load_config(path)


def reload_config(path: str | None = None) -> dict[str, Any]:
    """清空缓存并重新读取配置（供测试与交互式会话使用）。"""
    get_config.cache_clear()
    return get_config(path)


def resolve_path(path: str | Path) -> Path:
    """把相对路径解析为相对于项目根目录的绝对路径。"""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def get_path(key: str, config: dict[str, Any] | None = None) -> Path:
    """按 ``paths`` 段中的键取得绝对路径。

    Examples
    --------
    >>> get_path("processed").name
    'processed'
    """
    cfg = config if config is not None else get_config()
    try:
        raw = cfg["paths"][key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"config.yaml 的 paths 段中缺少键: {key}") from exc
    return resolve_path(raw)


def get_env(name: str, default: str | None = None) -> str | None:
    """读取环境变量。

    用于 API Key 等敏感信息。项目**禁止**在源码或配置文件中硬编码密钥。
    """
    value = os.environ.get(name)
    return value if value else default


def ensure_directories(config: dict[str, Any] | None = None) -> dict[str, Path]:
    """创建配置中声明的全部目录，返回 ``{key: 绝对路径}``。"""
    cfg = config if config is not None else get_config()
    created: dict[str, Path] = {}
    for key in cfg.get("paths", {}):
        directory = get_path(key, cfg)
        directory.mkdir(parents=True, exist_ok=True)
        created[key] = directory
    return created
