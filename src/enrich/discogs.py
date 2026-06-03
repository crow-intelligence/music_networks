"""Discogs dating for the tail MusicBrainz cannot date.

Discogs is a second open release database — often stronger than MusicBrainz for
Hungarian vinyl/CD pressings — so it dates songs the MB pass missed. We search
its ``/database/search`` endpoint by artist + track, validate the artist, and
take the earliest plausible release year.

Mirrors :mod:`src.enrich.dating` / :mod:`src.enrich.musicbrainz`:

* pure parsers (`search_params`, `earliest_year`) that are deterministic and
  doctested, and
* an async :class:`DiscogsClient` reusing the scraper's
  :class:`~src.scraper.fetch.RateLimiter`, plus a :class:`DiscogsDater` that
  fills the still-undated tail, resumable through ``enrich_state`` rows with
  ``source='discogs'``.

The search endpoint requires authentication: pass a personal access token
(discogs.com → Settings → Developers) via the ``DISCOGS_TOKEN`` env var or the
``date-discogs --token`` flag. Authenticated clients may make 60 requests/min.
"""

from __future__ import annotations

import asyncio

import httpx
from sqlalchemy import and_, or_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import (
    EnrichState,
    Performer,
    Song,
    SongPerformer,
    create_all,
    get_async_engine,
    make_async_session,
)
from src.enrich import USER_AGENT
from src.enrich.genius_kaggle import parse_year
from src.enrich.match import normalize, song_key
from src.scraper.fetch import RateLimiter

DISCOGS_SEARCH = "https://api.discogs.com/database/search"
_SRC = "discogs"
MAX_ATTEMPTS = 3

# Transient HTTP statuses worth retrying (Discogs throttles with 429).
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


def search_params(artist: str, title: str, token: str) -> dict[str, str]:
    """Build the Discogs ``/database/search`` query parameters.

    Args:
        artist: The performing act name.
        title: The song title (matched against release *tracks*).
        token: A Discogs personal access token.

    Returns:
        The query-parameter dict.

    Examples:
        >>> p = search_params("Omega", "Gyöngyhajú lány", "tok")
        >>> (p["artist"], p["track"], p["type"], p["token"])
        ('Omega', 'Gyöngyhajú lány', 'release', 'tok')
    """
    return {
        "artist": artist,
        "track": title,
        "type": "release",
        "token": token,
        "per_page": "25",
    }


def earliest_year(artist: str, results: list[dict]) -> int | None:
    """Return the earliest plausible release year among matching results.

    A result counts only when our (normalized) artist appears in its
    ``"<artist> - <release>"`` title, guarding against the search returning
    loosely-related releases. Years are validated via
    :func:`~src.enrich.genius_kaggle.parse_year`.

    Args:
        artist: Our performing act name.
        results: The ``results`` list from a Discogs search response.

    Returns:
        The earliest valid year, or ``None`` if nothing qualifies.

    Examples:
        >>> res = [
        ...     {"title": "Omega - Talált tárgyak", "year": "1996"},
        ...     {"title": "Omega - Gyöngyhajú lány", "year": "1969"},
        ...     {"title": "Other Band - Comp", "year": "1970"},
        ... ]
        >>> earliest_year("Omega", res)
        1969
        >>> earliest_year("Omega", [{"title": "X - Y", "year": "1980"}]) is None
        True
        >>> earliest_year("Omega", []) is None
        True
    """
    norm_artist = normalize(artist)
    if not norm_artist:
        return None
    years = [
        year
        for result in results
        if norm_artist in normalize(result.get("title", ""))
        and (year := parse_year(str(result.get("year") or ""))) is not None
    ]
    return min(years) if years else None


class DiscogsClient:
    """Polite async client over the Discogs search API."""

    def __init__(
        self,
        token: str,
        *,
        delay: float = 1.2,
        timeout: float = 30.0,
        max_retries: int = 4,
    ) -> None:
        """Configure the client.

        Args:
            token: Discogs personal access token.
            delay: Minimum seconds between requests (~1.2s keeps under 60/min).
            timeout: Per-request timeout in seconds.
            max_retries: Attempts for transient failures before giving up.
        """
        self._token = token
        self._limiter = RateLimiter(delay)
        self._timeout = timeout
        self._max_retries = max_retries
        self._headers = {"User-Agent": USER_AGENT}

    async def search_release(self, artist: str, title: str) -> list[dict]:
        """Search releases by artist + track; return the ``results`` list.

        Retries transient failures (timeouts, 429/5xx) with backoff, honouring
        ``Retry-After``; other 4xx propagate.

        Args:
            artist: The performing act name.
            title: The song title.

        Returns:
            The search ``results`` (possibly empty).

        Raises:
            httpx.HTTPError: On a non-retryable error or exhausted retries.
        """
        params = search_params(artist, title, self._token)
        last_exc: httpx.HTTPError | None = None
        for attempt in range(self._max_retries):
            await self._limiter.acquire()
            try:
                async with httpx.AsyncClient(
                    timeout=self._timeout, headers=self._headers
                ) as client:
                    resp = await client.get(DISCOGS_SEARCH, params=params)
            except httpx.TransportError as exc:
                last_exc = exc
            else:
                if resp.status_code not in _RETRY_STATUS:
                    resp.raise_for_status()
                    return resp.json().get("results", [])
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    await asyncio.sleep(min(float(retry_after), 60.0))
                    continue
            await asyncio.sleep(min(2.0**attempt, 30.0))
        assert last_exc is not None
        raise last_exc


class DiscogsDater:
    """Dates still-undated lyrics songs via Discogs release search."""

    def __init__(self, client: DiscogsClient, db_path: str = "data/music.db"):
        """Create a dater.

        Args:
            client: A configured :class:`DiscogsClient`.
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

    async def run(self, *, limit: int | None = None) -> dict[str, int]:
        """Date undated lyrics songs, one Discogs search per artist|title.

        Args:
            limit: Optional cap on searches (distinct keys) this run.

        Returns:
            Counts ``{"looked_up", "dated", "missed"}`` (songs for dated/missed).
        """
        await self.init_db()
        groups = await self._candidate_groups()
        counts = {"looked_up": 0, "dated": 0, "missed": 0}
        for members in groups.values():
            if limit is not None and counts["looked_up"] >= limit:
                break
            counts["looked_up"] += 1
            artist, title, _ = members[0]
            song_ids = [sid for (_name, _title, sid) in members]
            try:
                results = await self._client.search_release(artist, title)
            except Exception as exc:  # noqa: BLE001 - record and retry later
                await self._mark(song_ids, "failed", str(exc)[:500])
                continue
            year = earliest_year(artist, results)
            if year is not None:
                await self._apply(song_ids, year)
                counts["dated"] += len(song_ids)
            else:
                await self._mark(song_ids, "done", None)
                counts["missed"] += len(song_ids)
        return counts

    async def _candidate_groups(self) -> dict[str, list[tuple[str, str, int]]]:
        """Group still-undated lyrics songs by ``song_key(performer, title)``.

        Value entries are ``(artist, title, song_id)``; the first member's
        artist/title drive the search, and every member is dated together.
        """
        stmt = (
            select(Performer.name, Song.title, Song.id)
            .join(SongPerformer, SongPerformer.song_id == Song.id)
            .join(Performer, Performer.id == SongPerformer.performer_id)
            .outerjoin(
                EnrichState,
                and_(EnrichState.song_id == Song.id, EnrichState.source == _SRC),
            )
            .where(
                Song.lyrics.is_not(None),
                Song.lyrics != "",
                Song.first_release_year.is_(None),
                or_(
                    EnrichState.id.is_(None),
                    and_(
                        EnrichState.status != "done",
                        EnrichState.attempts < MAX_ATTEMPTS,
                    ),
                ),
            )
        )
        groups: dict[str, list[tuple[str, str, int]]] = {}
        seen: set[int] = set()
        async with self._session_factory() as session:
            for name, title, song_id in (await session.execute(stmt)).all():
                if song_id in seen:
                    continue
                seen.add(song_id)
                groups.setdefault(song_key(name, title), []).append(
                    (name, title, song_id)
                )
        return groups

    async def _apply(self, song_ids: list[int], year: int) -> None:
        """Write the Discogs year to each song and mark its state done."""
        async with self._session_factory() as session:
            await session.execute(
                update(Song)
                .where(Song.id.in_(song_ids))
                .values(first_release_year=year, first_release_source=_SRC)
            )
            for song_id in song_ids:
                await self._upsert_state(session, song_id, "done", None)
            await session.commit()

    async def _mark(self, song_ids: list[int], status: str, error: str | None) -> None:
        """Mark a key group's enrich-state (done miss, or failed retry)."""
        async with self._session_factory() as session:
            for song_id in song_ids:
                await self._upsert_state(session, song_id, status, error)
            await session.commit()

    @staticmethod
    async def _upsert_state(
        session: AsyncSession, song_id: int, status: str, error: str | None
    ) -> None:
        """Insert or update the ``discogs`` enrich-state row for a song."""
        set_: dict = {"status": status, "last_error": error}
        if status == "failed":
            set_["attempts"] = EnrichState.attempts + 1
        await session.execute(
            sqlite_insert(EnrichState)
            .values(
                song_id=song_id,
                source=_SRC,
                status=status,
                attempts=1 if status == "failed" else 0,
                last_error=error,
            )
            .on_conflict_do_update(index_elements=["song_id", "source"], set_=set_)
        )
