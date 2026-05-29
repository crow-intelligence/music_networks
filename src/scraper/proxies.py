"""Optional self-healing pool of free HTTP proxies with per-egress scheduling.

Free proxies are unreliable, so this pool is best-effort: it fetches candidate
proxies from public lists, hands them out **soonest-available first**, spaces
each proxy's own requests by a fixed ``delay`` (so every egress IP independently
honours the site's crawl-delay), evicts proxies that fail repeatedly, and
re-fills when the live set runs low.

Because every proxy keeps its *own* ``next_allowed`` clock, ``K`` live proxies
can run ``K`` requests in parallel while each individual IP still waits
``delay`` seconds between its own requests. The scraper runs *direct* by
default; pass ``--proxies`` to enable this.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

# Free proxy list endpoints (plain "ip:port" per line). Tried in order.
_SOURCES = (
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=http"
    "&timeout=10000&country=all&ssl=all&anonymity=all",
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
)


async def fetch_proxy_candidates(limit: int = 200) -> list[str]:
    """Download a list of candidate ``http://ip:port`` proxy URLs.

    Args:
        limit: Maximum number of candidates to return.

    Returns:
        A list of proxy URLs (possibly empty if all sources are unreachable).
    """
    proxies: list[str] = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        for source in _SOURCES:
            try:
                resp = await client.get(source)
            except httpx.HTTPError:
                continue
            if resp.status_code != 200:
                continue
            for raw in resp.text.splitlines():
                line = raw.strip()
                if line and ":" in line:
                    proxies.append(f"http://{line}")
            if proxies:
                break
    return proxies[:limit]


@dataclass
class _ProxyState:
    """Mutable bookkeeping for a single proxy URL.

    Attributes:
        url: The ``http://ip:port`` proxy URL.
        next_allowed: Monotonic time before which the proxy must not be reused.
        failures: Consecutive failure count (reset on success).
        latency: Last observed success latency in seconds.
    """

    url: str
    next_allowed: float = 0.0
    failures: int = 0
    latency: float = 0.0


class ProxyPool:
    """A self-refilling pool that schedules per-proxy request times."""

    def __init__(
        self,
        *,
        delay: float = 2.0,
        min_size: int = 10,
        max_failures: int = 3,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize the pool.

        Args:
            delay: Minimum seconds between consecutive requests *per proxy*.
            min_size: Refill when fewer than this many live proxies remain.
            max_failures: Consecutive failures before a proxy is evicted.
            clock: Monotonic time source (injectable for testing).
        """
        self._delay = delay
        self._min_size = min_size
        self._max_failures = max_failures
        self._clock = clock
        self._proxies: dict[str, _ProxyState] = {}
        self._lock = asyncio.Lock()

    @property
    def size(self) -> int:
        """Number of live proxies currently in the pool.

        Examples:
            >>> ProxyPool().size
            0
        """
        return len(self._proxies)

    def seed(self, urls: list[str]) -> int:
        """Add proxy URLs to the pool, ignoring duplicates.

        Args:
            urls: Proxy URLs (``http://ip:port``).

        Returns:
            The number of newly added proxies.

        Examples:
            >>> pool = ProxyPool()
            >>> pool.seed(["http://a:1", "http://b:2", "http://a:1"])
            2
            >>> pool.size
            2
        """
        added = 0
        for url in urls:
            if url not in self._proxies:
                self._proxies[url] = _ProxyState(url=url)
                added += 1
        return added

    async def _refill(self) -> None:
        """Repopulate the pool from the free-proxy sources."""
        self.seed(await fetch_proxy_candidates())

    async def reserve(self) -> tuple[str, float] | None:
        """Reserve the soonest-available proxy and the wait it requires.

        Returns the proxy whose ``next_allowed`` time is earliest and advances
        that proxy's clock by ``delay``, so concurrent callers spread across
        distinct proxies. The caller should ``await asyncio.sleep(wait)`` before
        issuing the request. Refilling (network I/O) happens under a lock; the
        fast path takes no lock and performs no ``await``, so it is atomic.

        Returns:
            A ``(proxy_url, wait_seconds)`` tuple, or ``None`` when no proxy is
            available (the caller should fall back or retry later).

        Examples:
            >>> import asyncio
            >>> pool = ProxyPool(delay=2.0, min_size=0, clock=lambda: 100.0)
            >>> pool.seed(["http://a:1", "http://b:2"])
            2
            >>> asyncio.run(pool.reserve())  # both immediately available
            ('http://a:1', 0.0)
            >>> asyncio.run(pool.reserve())
            ('http://b:2', 0.0)
            >>> _, wait = asyncio.run(pool.reserve())  # both cooling down now
            >>> round(wait, 1)
            2.0
        """
        if len(self._proxies) < self._min_size:
            async with self._lock:
                if len(self._proxies) < self._min_size:
                    await self._refill()
        if not self._proxies:
            return None
        state = min(self._proxies.values(), key=lambda s: s.next_allowed)
        now = self._clock()
        start = max(now, state.next_allowed)
        state.next_allowed = start + self._delay
        return state.url, max(0.0, start - now)

    def report_ok(self, url: str, latency: float = 0.0) -> None:
        """Record a successful fetch through ``url`` (resets its failure count).

        Args:
            url: The proxy that succeeded.
            latency: Observed request latency in seconds.

        Examples:
            >>> pool = ProxyPool(max_failures=2)
            >>> pool.seed(["http://a:1"])
            1
            >>> pool.report_bad("http://a:1")
            >>> pool.report_ok("http://a:1", 1.2)  # success clears the failure
            >>> pool.report_bad("http://a:1")
            >>> pool.size  # still alive: the streak was reset
            1
        """
        state = self._proxies.get(url)
        if state is not None:
            state.failures = 0
            state.latency = latency

    def report_bad(self, url: str) -> None:
        """Record a failure for ``url``, evicting it after too many in a row.

        Args:
            url: The proxy that failed.

        Examples:
            >>> pool = ProxyPool(max_failures=2)
            >>> pool.seed(["http://a:1"])
            1
            >>> pool.report_bad("http://a:1")
            >>> pool.size
            1
            >>> pool.report_bad("http://a:1")  # hits max_failures -> evicted
            >>> pool.size
            0
        """
        state = self._proxies.get(url)
        if state is None:
            return
        state.failures += 1
        if state.failures >= self._max_failures:
            del self._proxies[url]
