"""统一日志配置。"""

from __future__ import annotations

import logging
import sys

from src.config import get_config

__all__ = ["get_logger", "setup_logging"]

_CONFIGURED = False


def setup_logging(level: str | None = None, config: dict | None = None) -> logging.Logger:
    """初始化根日志器（幂等）。"""
    global _CONFIGURED
    cfg = config if config is not None else get_config()
    log_cfg = cfg.get("logging", {}) if isinstance(cfg, dict) else {}
    resolved_level = (level or log_cfg.get("level") or "INFO").upper()
    fmt = log_cfg.get("format") or "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    root = logging.getLogger("sovereign_rating_lab")
    if not _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    root.setLevel(resolved_level)
    return root


def get_logger(name: str) -> logging.Logger:
    """取得带项目前缀的子日志器。"""
    setup_logging()
    return logging.getLogger(f"sovereign_rating_lab.{name}")
