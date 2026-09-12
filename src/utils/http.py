"""合规 HTTP 客户端。

设计目标
--------
1. **尊重 robots.txt**：抓取前先获取并解析目标站点的 robots.txt，被禁止时
   直接抛出 :class:`RobotsDisallowedError`，绝不绕过。
2. **限速**：同一主机两次请求之间最小间隔 ``min_delay_seconds`` 秒。
3. **可复现**：原始响应体写入 ``data/raw/cache/html``，并记录来源 URL、
   抓取时间、HTTP 状态码到 ``provenance.json``。相同 URL 二次请求默认命中缓存。
4. **稳健**：指数退避重试，超时可控。

本模块**不做任何解析**，只负责「合规地把字节取回来」。
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from src.config import PROJECT_ROOT, get_config, resolve_path
from src.utils.io import read_json, write_json
from src.utils.logging_utils import get_logger

__all__ = [
    "FetchResult",
    "HttpClient",
    "RobotsDisallowedError",
    "check_robots_allowed",
]

logger = get_logger(__name__)


class RobotsDisallowedError(RuntimeError):
    """目标路径被 robots.txt 禁止抓取。"""


@dataclass
class FetchResult:
    """一次抓取的完整记录。"""

    url: str
    status_code: int
    text: str
    from_cache: bool
    fetched_at: str
    cache_path: Path | None = None
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


def check_robots_allowed(url: str, user_agent: str = "*", timeout: int = 15) -> bool:
    """检查 *url* 是否允许抓取。

    robots.txt 无法获取时（网络异常、404）采取**保守放行**策略：返回 True，
    但调用方会在 provenance 中记录下来，便于事后审计。
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"非法 URL: {url}")
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
        allowed = parser.can_fetch(user_agent, url)
    except Exception as exc:
        logger.warning("robots.txt 无法解析 (%s)：%s，按保守策略放行", robots_url, exc)
        return True
    if not allowed:
        logger.warning("robots.txt 禁止抓取: %s", url)
    return allowed


class HttpClient:
    """带缓存、限速与重试的 HTTP 客户端。"""

    def __init__(self, config: dict | None = None) -> None:
        cfg = config if config is not None else get_config()
        source_cfg = cfg.get("data_sources", {})
        scrape_cfg = cfg.get("scraping", {})

        self.offline: bool = bool(source_cfg.get("offline", False))
        self.timeout: int = int(source_cfg.get("request_timeout", 30))
        self.max_retries: int = int(source_cfg.get("max_retries", 3))
        self.user_agent: str = str(
            source_cfg.get("user_agent", "sovereign-rating-lab/0.1 (research)")
        )
        self.respect_robots: bool = bool(scrape_cfg.get("respect_robots_txt", True))
        self.min_delay: float = float(scrape_cfg.get("min_delay_seconds", 2.0))
        self.cache_dir: Path = resolve_path(scrape_cfg.get("cache_dir", "data/raw/cache/html"))
        self.provenance_file: Path = resolve_path(
            scrape_cfg.get("provenance_file", "data/raw/cache/provenance.json")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.user_agent})
        self._last_request_at: dict[str, float] = {}
        self._provenance: list[dict[str, Any]] = self._load_provenance()

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _hash_url(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]

    def _load_provenance(self) -> list[dict[str, Any]]:
        if self.provenance_file.exists():
            try:
                data = read_json(self.provenance_file)
                return list(data) if isinstance(data, list) else []
            except Exception as exc:
                logger.warning("provenance 文件损坏，重新开始: %s", exc)
        return []

    def _record_provenance(self, entry: dict[str, Any]) -> None:
        self._provenance.append(entry)
        try:
            write_json(self._provenance, self.provenance_file)
        except OSError as exc:  # pragma: no cover
            logger.warning("无法写入 provenance: %s", exc)

    def provenance(self) -> list[dict[str, Any]]:
        """返回本次会话的全部抓取记录（可用于生成数据来源附录）。"""
        return list(self._provenance)

    def _throttle(self, host: str) -> None:
        last = self._last_request_at.get(host)
        if last is not None:
            elapsed = time.monotonic() - last
            if elapsed < self.min_delay:
                time.sleep(self.min_delay - elapsed)
        self._last_request_at[host] = time.monotonic()

    # ------------------------------------------------------------------ fetch
    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        use_cache: bool = True,
        headers: dict[str, str] | None = None,
        check_robots: bool | None = None,
    ) -> FetchResult:
        """发起 GET 请求并返回 :class:`FetchResult`。"""
        cache_key = self._hash_url(url + "?" + str(sorted((params or {}).items())))
        cache_path = self.cache_dir / f"{cache_key}.txt"

        if use_cache and cache_path.exists():
            logger.debug("命中缓存: %s", url)
            return FetchResult(
                url=url,
                status_code=200,
                text=cache_path.read_text(encoding="utf-8", errors="replace"),
                from_cache=True,
                fetched_at=datetime.fromtimestamp(cache_path.stat().st_mtime, tz=UTC).isoformat(),
                cache_path=cache_path,
            )

        if self.offline:
            raise RuntimeError(f"离线模式 (data_sources.offline = true) 且无缓存，无法抓取: {url}")

        should_check = self.respect_robots if check_robots is None else check_robots
        if should_check and not check_robots_allowed(url, self.user_agent, self.timeout):
            raise RobotsDisallowedError(
                f"robots.txt 禁止抓取该路径: {url}。请更换合规数据源或联系数据所有者。"
            )

        parsed = urlparse(url)
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle(parsed.netloc)
            try:
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                    headers=headers,
                )
                if response.status_code >= 500:  # 服务端错误 -> 重试
                    raise requests.HTTPError(f"HTTP {response.status_code}")
                response.raise_for_status()
                cache_path.write_text(response.text, encoding="utf-8")
                result = FetchResult(
                    url=url,
                    status_code=response.status_code,
                    text=response.text,
                    from_cache=False,
                    fetched_at=datetime.now(tz=UTC).isoformat(),
                    cache_path=cache_path,
                    headers=dict(response.headers),
                )
                self._record_provenance(
                    {
                        "url": response.url,
                        "requested_url": url,
                        "params": params or {},
                        "status_code": response.status_code,
                        "fetched_at": result.fetched_at,
                        "cache_path": (
                            str(cache_path.relative_to(PROJECT_ROOT))
                            if cache_path.is_relative_to(PROJECT_ROOT)
                            else str(cache_path)
                        ),
                        "robots_checked": should_check,
                    }
                )
                return result
            except (requests.RequestException, requests.HTTPError) as exc:
                last_exc = exc
                backoff = min(2.0**attempt, 30.0)
                logger.warning(
                    "请求失败 (%d/%d) %s: %s，%.1fs 后重试",
                    attempt,
                    self.max_retries,
                    url,
                    exc,
                    backoff,
                )
                if attempt < self.max_retries:
                    time.sleep(backoff)

        raise RuntimeError(f"抓取失败，已重试 {self.max_retries} 次: {url}") from last_exc

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        use_cache: bool = True,
        check_robots: bool = False,
    ) -> Any:
        """抓取 JSON 接口（API 场景默认跳过 robots 校验）。"""
        result = self.get(url, params=params, use_cache=use_cache, check_robots=check_robots)
        try:
            return json.loads(result.text)
        except ValueError as exc:
            raise ValueError(f"响应不是合法 JSON: {url}") from exc
