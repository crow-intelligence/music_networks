"""Async HTTP fetching: per-egress rate limiting and multi-proxy retry.

When a :class:`~src.scraper.proxies.ProxyPool` is supplied, every request is
routed through a free proxy, and the pool spaces each proxy's own requests by
``delay`` seconds — so many proxies fetch in parallel while each egress IP still
honours the site's crawl-delay. A request retries across *distinct* proxies
until one returns a usable response. Without a pool, requests go direct, spaced
by a single :class:`RateLimiter` (kept for completeness; the live site bans
high-volume direct traffic).
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

import httpx

from src.scraper.proxies import ProxyPool

# A small pool of realistic desktop browser user-agents. We rotate among these
# rather than relying on a network-backed UA database, which is flaky.
_USER_AGENTS = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
)


def random_user_agent() -> str:
    """Return a random realistic desktop browser User-Agent string.

    Returns:
        One of the built-in browser UA strings.
    """
    return random.choice(_USER_AGENTS)


@dataclass
class FetchOutcome:
    """Result of a fetch attempt, carrying enough to diagnose failures.

    Attributes:
        html: Page text on success, else ``None``.
        status: Last HTTP status code seen, or ``None`` if no response arrived.
        error: Human-readable failure reason, or ``None`` on success.

    Examples:
        >>> FetchOutcome("<html>...</html>", 200, None).ok
        True
        >>> FetchOutcome(None, 404, "HTTP 404").ok
        False
    """

    html: str | None
    status: int | None
    error: str | None

    @property
    def ok(self) -> bool:
        """Whether the fetch produced usable HTML."""
        return self.html is not None


class RateLimiter:
    """Spaces request starts so they are at least ``min_interval`` apart.

    Safe for concurrent use: callers ``await acquire()`` and the limiter
    serializes them just long enough to enforce the gap.
    """

    def __init__(self, min_interval: float) -> None:
        """Initialize the limiter.

        Args:
            min_interval: Minimum seconds between consecutive request starts.
        """
        self._min_interval = min_interval
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def acquire(self) -> None:
        """Block until enough time has elapsed since the last request start."""
        async with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_allowed = time.monotonic() + self._min_interval


class Fetcher:
    """Polite async HTTP client returning a :class:`FetchOutcome`."""

    def __init__(
        self,
        *,
        delay: float = 2.0,
        concurrency: int = 4,
        timeout: float = 20.0,
        max_retries: int = 5,
        min_content_length: int = 1000,
        proxies: ProxyPool | None = None,
    ) -> None:
        """Configure the fetcher.

        Args:
            delay: Seconds between request starts on the direct path.
            concurrency: Max simultaneous in-flight requests.
            timeout: Per-request timeout in seconds.
            max_retries: Distinct attempts (proxies) per URL before giving up.
            min_content_length: Minimum body length to accept a 200 response;
                guards against tiny proxy error pages returned as 200.
            proxies: Optional proxy pool; when ``None``, requests go direct.
        """
        self._limiter = RateLimiter(delay)
        self._semaphore = asyncio.Semaphore(concurrency)
        self._timeout = timeout
        self._max_retries = max_retries
        self._min_content_length = min_content_length
        self._proxies = proxies

    async def get(self, url: str) -> FetchOutcome:
        """Fetch ``url`` via the proxy pool (or direct), retrying as needed.

        Args:
            url: Absolute URL to fetch.

        Returns:
            A :class:`FetchOutcome` with the HTML on success, or the last status
            and error reason on failure.
        """
        if self._proxies is None:
            return await self._get_direct(url)
        return await self._get_via_proxies(url)

    async def _request(self, url: str, proxy: str | None) -> tuple[int, str]:
        """Perform one HTTP GET, returning ``(status_code, body_text)``."""
        async with httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            proxy=proxy,
            headers={"User-Agent": random_user_agent()},
        ) as client:
            response = await client.get(url)
        return response.status_code, response.text

    def _accept(self, status: int, text: str) -> bool:
        """Whether a response counts as a usable success."""
        return status == 200 and len(text) >= self._min_content_length

    async def _get_via_proxies(self, url: str) -> FetchOutcome:
        """Try successive ready proxies until one returns usable HTML."""
        assert self._proxies is not None
        last_status: int | None = None
        last_error = "no proxies available"
        for _ in range(self._max_retries):
            reservation = await self._proxies.reserve()
            if reservation is None:
                await asyncio.sleep(2.0)
                continue
            proxy, wait = reservation
            if wait > 0:
                await asyncio.sleep(wait)
            async with self._semaphore:
                start = time.monotonic()
                try:
                    status, text = await self._request(url, proxy)
                except (httpx.HTTPError, OSError) as exc:
                    self._proxies.report_bad(proxy)
                    last_error = type(exc).__name__
                    continue
            if self._accept(status, text):
                self._proxies.report_ok(proxy, time.monotonic() - start)
                return FetchOutcome(text, 200, None)
            last_status = status
            if status == 404:  # a real missing page; no proxy will fix it
                return FetchOutcome(None, 404, "HTTP 404")
            # blocked / server error / truncated: blame the proxy, try the next
            self._proxies.report_bad(proxy)
            last_error = f"HTTP {status}" if status != 200 else "short response"
        return FetchOutcome(None, last_status, last_error)

    async def _get_direct(self, url: str) -> FetchOutcome:
        """Fetch directly (no proxy), spaced by the global rate limiter."""
        last_status: int | None = None
        last_error = "fetch failed"
        for attempt in range(self._max_retries):
            async with self._semaphore:
                await self._limiter.acquire()
                try:
                    status, text = await self._request(url, None)
                except (httpx.HTTPError, OSError) as exc:
                    last_error = type(exc).__name__
                    await asyncio.sleep(min(2.0 * (attempt + 1), 5.0))
                    continue
            if self._accept(status, text):
                return FetchOutcome(text, 200, None)
            last_status = status
            if status == 404:
                return FetchOutcome(None, 404, "HTTP 404")
            last_error = f"HTTP {status}"
            await asyncio.sleep(min(2.0 * (attempt + 1), 5.0))
        return FetchOutcome(None, last_status, last_error)
