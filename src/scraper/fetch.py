"""Async HTTP fetching with polite, per-domain rate limiting.

The fetcher spaces the *start* of consecutive requests by ``delay`` seconds
(honouring the site's robots.txt crawl-delay) while allowing a bounded number
of concurrent in-flight requests to hide network latency. Optional proxy
rotation is delegated to :mod:`src.scraper.proxies`.
"""

from __future__ import annotations

import asyncio
import random
import time

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
    """Polite async HTTP client returning page HTML (or ``None`` on failure)."""

    def __init__(
        self,
        *,
        delay: float = 1.5,
        concurrency: int = 4,
        timeout: float = 30.0,
        max_retries: int = 3,
        proxies: ProxyPool | None = None,
    ) -> None:
        """Configure the fetcher.

        Args:
            delay: Minimum seconds between request starts (politeness).
            concurrency: Max simultaneous in-flight requests.
            timeout: Per-request timeout in seconds.
            max_retries: Attempts per URL before giving up.
            proxies: Optional proxy pool; when ``None``, requests go direct.
        """
        self._limiter = RateLimiter(delay)
        self._semaphore = asyncio.Semaphore(concurrency)
        self._timeout = timeout
        self._max_retries = max_retries
        self._proxies = proxies

    async def get(self, url: str) -> str | None:
        """Fetch ``url`` and return its text, retrying on transient failures.

        Args:
            url: Absolute URL to fetch.

        Returns:
            The response body as text on HTTP 200, otherwise ``None`` after
            exhausting retries.
        """
        for attempt in range(self._max_retries):
            proxy = await self._proxies.acquire() if self._proxies else None
            async with self._semaphore:
                await self._limiter.acquire()
                try:
                    async with httpx.AsyncClient(
                        timeout=self._timeout,
                        follow_redirects=True,
                        proxy=proxy,
                        headers={"User-Agent": random_user_agent()},
                    ) as client:
                        response = await client.get(url)
                    if response.status_code == 200:
                        if self._proxies and proxy:
                            self._proxies.report_ok(proxy)
                        return response.text
                except (httpx.HTTPError, OSError):
                    if self._proxies and proxy:
                        self._proxies.report_bad(proxy)
            # brief backoff before retrying
            await asyncio.sleep(min(2.0 * (attempt + 1), 5.0))
        return None
