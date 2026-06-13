"""Wikipedia pageviews → a cultural-salience score on :class:`~src.db.Artist`.

For each artist with a Hungarian-Wikipedia article (``Artist.huwiki_title``,
filled by :mod:`src.enrich.wikidata_enrich`), this queries the Wikimedia
Pageviews REST API for monthly views and stores the **average monthly
pageviews** in ``Artist.avg_monthly_pageviews`` — a proxy for *lasting* cultural
salience (how often people still look the act up).

Note the time window: the API only has data from **2015-07** onward, so this is a
*present-day* salience signal — it reflects how remembered an act is now,
regardless of which decade it was active in (an Omega still draws views today).

Mirrors the other enrich modules: pure, doctested helpers (:func:`encode_title`,
:func:`average_monthly_views`) plus an async :class:`PageviewsClient` reusing the
scraper's rate limiter. Resumable: an artist whose score is already set (or that
has no article / no data, recorded as ``0.0``) is skipped on re-runs.
"""

from __future__ import annotations

import asyncio
import urllib.parse
from collections.abc import Iterable

import httpx
from sqlalchemy import select, update

from src.db import Artist, create_all, get_async_engine, make_async_session
from src.enrich import USER_AGENT

REST_BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
PROJECT = "hu.wikipedia.org"
# The API has data from 2015-07-01; end is filled with the current month at runtime.
START = "2015070100"

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


def encode_title(title: str) -> str:
    """Encode an article title for the Pageviews API path segment.

    Spaces become underscores (MediaWiki page-title form) and every other
    reserved character is percent-encoded, so titles with slashes, parentheses
    or non-ASCII letters can't break the URL path.

    Args:
        title: The article title (e.g. ``"Omega (együttes)"``).

    Returns:
        The path-safe, percent-encoded title.

    Examples:
        >>> encode_title("Omega (együttes)")
        'Omega_%28egy%C3%BCttes%29'
        >>> encode_title("AC/DC")
        'AC%2FDC'
        >>> encode_title("Karácsony János")
        'Kar%C3%A1csony_J%C3%A1nos'
    """
    return urllib.parse.quote(title.replace(" ", "_"), safe="")


def average_monthly_views(items: Iterable[dict]) -> float:
    """Average the ``views`` across monthly pageview items.

    Args:
        items: The ``items`` list from a Pageviews API response (each a dict with
            a ``views`` count).

    Returns:
        The mean monthly views, or ``0.0`` when there are no items.

    Examples:
        >>> average_monthly_views([{"views": 100}, {"views": 300}])
        200.0
        >>> average_monthly_views([{"views": 0}])
        0.0
        >>> average_monthly_views([])
        0.0
    """
    counts = [int(it.get("views", 0)) for it in items]
    return sum(counts) / len(counts) if counts else 0.0


def _end_month() -> str:
    """Return the first day of the current month as a ``YYYYMMDD00`` stamp."""
    from datetime import date

    today = date.today()
    return f"{today.year}{today.month:02d}0100"


class PageviewsClient:
    """Polite async client over the Wikimedia Pageviews REST API."""

    def __init__(
        self, *, delay: float = 0.2, timeout: float = 30.0, max_retries: int = 4
    ) -> None:
        """Configure the client.

        Args:
            delay: Minimum seconds between requests (the API is generous; a small
                gap keeps us well-behaved).
            timeout: Per-request timeout in seconds.
            max_retries: Attempts for transient failures before giving up.
        """
        from src.scraper.fetch import RateLimiter

        self._limiter = RateLimiter(delay)
        self._timeout = timeout
        self._max_retries = max_retries

    async def monthly_views(
        self, title: str, *, start: str = START, end: str | None = None
    ) -> list[dict]:
        """Fetch the monthly pageview items for an article (``[]`` if none).

        A 404 means the article has no pageview data (too new, or never viewed) —
        returned as an empty list rather than an error.

        Args:
            title: The article title.
            start: Inclusive start stamp (``YYYYMMDDHH``).
            end: Inclusive end stamp; defaults to the current month.

        Returns:
            The ``items`` list (possibly empty).

        Raises:
            httpx.HTTPError: On a non-retryable error or exhausted retries.
        """
        end = end or _end_month()
        url = (
            f"{REST_BASE}/{PROJECT}/all-access/user/"
            f"{encode_title(title)}/monthly/{start}/{end}"
        )
        last_exc: httpx.HTTPError | None = None
        for attempt in range(self._max_retries):
            await self._limiter.acquire()
            try:
                async with httpx.AsyncClient(
                    timeout=self._timeout, headers={"User-Agent": USER_AGENT}
                ) as client:
                    resp = await client.get(url)
            except httpx.TransportError as exc:
                last_exc = exc
            else:
                if resp.status_code == 404:
                    return []
                if resp.status_code not in _RETRY_STATUS:
                    resp.raise_for_status()
                    return resp.json().get("items", [])
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
            await asyncio.sleep(min(2.0**attempt, 30.0))
        assert last_exc is not None
        raise last_exc


class PageviewsEnricher:
    """Fills ``Artist.avg_monthly_pageviews`` for articled artists."""

    def __init__(self, client: PageviewsClient, db_path: str = "data/music.db") -> None:
        """Create an enricher.

        Args:
            client: A configured :class:`PageviewsClient`.
            db_path: Path to the SQLite database file.
        """
        self._client = client
        self._engine = get_async_engine(db_path)
        self._session_factory = make_async_session(self._engine)

    async def init_db(self) -> None:
        """Create tables (and run column migrations) if needed."""
        await create_all(self._engine)

    async def close(self) -> None:
        """Dispose of the database engine."""
        await self._engine.dispose()

    async def run(
        self, *, limit: int | None = None, resume: bool = True
    ) -> dict[str, int]:
        """Score artists with a Hungarian-Wikipedia article by avg pageviews.

        Args:
            limit: Optional cap on artists processed this run.
            resume: Skip artists whose score is already set.

        Returns:
            Counts ``{"processed", "with_views"}``.
        """
        await self.init_db()
        async with self._session_factory() as session:
            stmt = select(Artist.id, Artist.huwiki_title).where(
                Artist.huwiki_title.is_not(None)
            )
            if resume:
                stmt = stmt.where(Artist.avg_monthly_pageviews.is_(None))
            targets = (await session.execute(stmt)).all()
        if limit is not None:
            targets = targets[:limit]

        counts = {"processed": 0, "with_views": 0}
        for artist_id, title in targets:
            try:
                items = await self._client.monthly_views(title)
            except Exception:  # noqa: BLE001 - leave NULL; a re-run retries it
                continue
            score = round(average_monthly_views(items), 1)
            async with self._session_factory() as session:
                await session.execute(
                    update(Artist)
                    .where(Artist.id == artist_id)
                    .values(avg_monthly_pageviews=score)
                )
                await session.commit()
            counts["processed"] += 1
            counts["with_views"] += score > 0
        return counts


def main() -> None:
    """CLI: fill cultural-salience pageview scores for articled artists."""
    import argparse

    ap = argparse.ArgumentParser(description="Wikipedia pageviews → artist salience.")
    ap.add_argument("--db", default="data/music.db")
    ap.add_argument("--delay", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    async def _go() -> None:
        enricher = PageviewsEnricher(
            PageviewsClient(delay=args.delay), db_path=args.db
        )
        try:
            counts = await enricher.run(limit=args.limit, resume=not args.no_resume)
            print(
                f"Scored {counts['processed']} artists "
                f"({counts['with_views']} with non-zero pageviews)."
            )
        finally:
            await enricher.close()

    asyncio.run(_go())


if __name__ == "__main__":
    main()
