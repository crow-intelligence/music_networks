"""Optional self-healing pool of free HTTP proxies.

Free proxies are unreliable, so this pool is best-effort: it fetches a list of
candidate proxies, hands them out round-robin, drops ones that fail, and
re-fills when depleted. When the pool cannot be filled it yields ``None`` so
the caller falls back to a direct connection. Proxies add resilience only —
politeness (the crawl delay) is enforced by the fetcher regardless.

The scraper runs *direct* by default; pass ``--proxies`` to enable this.
"""

from __future__ import annotations

import asyncio
from collections import deque

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
            for line in resp.text.splitlines():
                line = line.strip()
                if line and ":" in line:
                    proxies.append(f"http://{line}")
            if proxies:
                break
    return proxies[:limit]


class ProxyPool:
    """A round-robin pool of free proxies with failure tracking."""

    def __init__(self, min_size: int = 10) -> None:
        """Initialize an empty pool.

        Args:
            min_size: Re-fill the pool when it drops below this many proxies.
        """
        self._min_size = min_size
        self._available: deque[str] = deque()
        self._lock = asyncio.Lock()

    async def _refill(self) -> None:
        """Repopulate the pool from the free-proxy sources."""
        candidates = await fetch_proxy_candidates()
        self._available.extend(candidates)

    async def acquire(self) -> str | None:
        """Return the next proxy URL, refilling if needed, or ``None``.

        Returns:
            A proxy URL, or ``None`` when no proxies could be obtained (the
            caller should then connect directly).
        """
        async with self._lock:
            if len(self._available) < self._min_size:
                await self._refill()
            if not self._available:
                return None
            proxy = self._available.popleft()
            # round-robin: put it back at the end so it can be reused
            self._available.append(proxy)
            return proxy

    def report_ok(self, proxy: str) -> None:
        """Mark ``proxy`` as healthy (no-op; kept for symmetry).

        Args:
            proxy: The proxy URL that succeeded.
        """

    def report_bad(self, proxy: str) -> None:
        """Remove a failing ``proxy`` from rotation.

        Args:
            proxy: The proxy URL that failed.
        """
        try:
            self._available.remove(proxy)
        except ValueError:
            pass
