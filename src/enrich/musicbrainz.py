"""MusicBrainz API client for first-release dating and the version graph.

MusicBrainz answers the question the lyrics portals cannot: *when was this
recording first released*, and *which composition (work) is it a version of*.
This module wraps the public web service (``/ws/2/``) — no auth, ~1 req/s, a
contactable User-Agent — and exposes:

* pure parsers (date/work extraction) that are deterministic and doctested, and
* an async :class:`MusicBrainzClient` that reuses the scraper's
  :class:`~src.scraper.fetch.RateLimiter` for politeness.

The dating rule is **per recording**: a recording's first release date across
all its releases. Versions of the same composition share a ``work_id`` (the
MusicBrainz Work), which is what lets remakes link back to the original.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

from src.enrich import USER_AGENT
from src.scraper.fetch import RateLimiter

# MusicBrainz web service base and Hungary's area MBID (stable identifier).
WS_BASE = "https://musicbrainz.org/ws/2/"
HUNGARY_AREA_MBID = "71bbafaa-e825-3e15-8ca9-017dcad1748b"

# Transient HTTP statuses worth retrying (MB throttles with 503 under load).
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass
class RecordingInfo:
    """A MusicBrainz recording reduced to the fields we store.

    Attributes:
        gid: The recording's MusicBrainz id (its external id).
        title: Recording title.
        artist: Joined artist-credit string.
        first_release_date: Earliest release date seen (``YYYY``/``YYYY-MM-DD``)
            or ``None``.
        work_id: MusicBrainz Work id this recording performs, or ``None``.
    """

    gid: str
    title: str
    artist: str
    first_release_date: str | None
    work_id: str | None


def date_year(date: str | None) -> int | None:
    """Extract the four-digit year from a MusicBrainz date string.

    Args:
        date: A date like ``"1969"``, ``"1969-05"`` or ``"1969-05-23"``, or
            ``None``/empty.

    Returns:
        The year as an int, or ``None`` if no four-digit year is present.

    Examples:
        >>> date_year("1969-05-23")
        1969
        >>> date_year("1969")
        1969
        >>> date_year("") is None
        True
        >>> date_year(None) is None
        True
    """
    if not date:
        return None
    head = date[:4]
    return int(head) if head.isdigit() and len(head) == 4 else None


def earliest_release_date(releases: list[dict]) -> str | None:
    """Return the earliest non-empty ``date`` among a recording's releases.

    Dates compare correctly as ``YYYY[-MM[-DD]]`` strings; partial dates sort
    before fuller ones in the same year, which is the desired "as early as
    known" behaviour.

    Args:
        releases: The ``releases`` list from a recording (each a dict that may
            carry a ``date``).

    Returns:
        The minimum date string, or ``None`` if none have a date.

    Examples:
        >>> earliest_release_date([{"date": "2000"}, {"date": "1980-06"}])
        '1980-06'
        >>> earliest_release_date([{"date": ""}, {"title": "x"}]) is None
        True
        >>> earliest_release_date([]) is None
        True
    """
    dates = [r["date"] for r in releases if r.get("date")]
    return min(dates) if dates else None


def work_id_from_relations(relations: list[dict]) -> str | None:
    """Return the first performed-work id from a recording's relations.

    Args:
        relations: The ``relations`` list from a recording fetched with
            ``inc=work-rels``.

    Returns:
        The related Work's MusicBrainz id, or ``None`` if there is no work
        relation.

    Examples:
        >>> rels = [{"target-type": "work", "work": {"id": "w-123"}}]
        >>> work_id_from_relations(rels)
        'w-123'
        >>> work_id_from_relations([{"target-type": "artist"}]) is None
        True
    """
    for rel in relations:
        if rel.get("target-type") == "work" and "work" in rel:
            return rel["work"].get("id")
    return None


def parse_recording(rec: dict) -> RecordingInfo:
    """Reduce a MusicBrainz recording JSON object to a :class:`RecordingInfo`.

    Args:
        rec: A recording object from ``/ws/2/recording`` fetched with
            ``inc=releases+work-rels`` (or a search result).

    Returns:
        The extracted :class:`RecordingInfo`.

    Examples:
        >>> rec = {
        ...     "id": "r-1",
        ...     "title": "Gyöngyhajú lány",
        ...     "artist-credit": [{"name": "Omega"}],
        ...     "releases": [{"date": "1969"}, {"date": "1998"}],
        ...     "relations": [{"target-type": "work", "work": {"id": "w-9"}}],
        ... }
        >>> info = parse_recording(rec)
        >>> (info.artist, info.first_release_date, info.work_id)
        ('Omega', '1969', 'w-9')
    """
    artist = "".join(
        part.get("name", "") + part.get("joinphrase", "")
        for part in rec.get("artist-credit", [])
    ).strip()
    # Search results expose "first-release-date" directly; browse results carry
    # the releases list we reduce ourselves. Prefer the explicit field.
    first = rec.get("first-release-date") or earliest_release_date(
        rec.get("releases", [])
    )
    return RecordingInfo(
        gid=rec["id"],
        title=rec.get("title", ""),
        artist=artist,
        first_release_date=first or None,
        work_id=work_id_from_relations(rec.get("relations", [])),
    )


class MusicBrainzClient:
    """Polite async client over the MusicBrainz web service."""

    def __init__(
        self, *, delay: float = 1.0, timeout: float = 30.0, max_retries: int = 4
    ) -> None:
        """Configure the client.

        Args:
            delay: Minimum seconds between requests (MusicBrainz asks for ~1s).
            timeout: Per-request timeout in seconds.
            max_retries: Attempts for transient failures (timeouts, 429/5xx)
                before giving up.
        """
        self._limiter = RateLimiter(delay)
        self._timeout = timeout
        self._max_retries = max_retries

    async def get_json(self, path: str, params: dict) -> dict:
        """GET ``path`` under the web service and return parsed JSON.

        Retries transient failures (connection/read timeouts and 429/5xx, which
        MusicBrainz returns when throttling) with exponential backoff; other 4xx
        errors propagate immediately.

        Args:
            path: Web-service path, e.g. ``"recording"``.
            params: Query parameters (``fmt=json`` is added automatically).

        Returns:
            The decoded JSON object.

        Raises:
            httpx.HTTPError: If a non-retryable error occurs, or retries are
                exhausted.
        """
        query = {**params, "fmt": "json"}
        last_exc: httpx.HTTPError | None = None
        for attempt in range(self._max_retries):
            await self._limiter.acquire()
            try:
                async with httpx.AsyncClient(
                    timeout=self._timeout, headers={"User-Agent": USER_AGENT}
                ) as client:
                    resp = await client.get(WS_BASE + path, params=query)
            except httpx.TransportError as exc:  # timeouts, connection drops
                last_exc = exc
            else:
                if resp.status_code not in _RETRY_STATUS:
                    resp.raise_for_status()  # non-retryable 4xx -> propagate
                    return resp.json()
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
            await asyncio.sleep(min(2.0**attempt, 30.0))
        assert last_exc is not None
        raise last_exc

    async def browse_artists_by_area(
        self, area_mbid: str = HUNGARY_AREA_MBID, *, limit: int = 100, offset: int = 0
    ) -> dict:
        """Browse artists linked to an area (paginated).

        Args:
            area_mbid: The area's MusicBrainz id (Hungary by default).
            limit: Page size (max 100).
            offset: Page offset.

        Returns:
            The raw browse response (``artist-count`` + ``artists``).
        """
        return await self.get_json(
            "artist", {"area": area_mbid, "limit": limit, "offset": offset}
        )

    async def search_recordings_by_artist(
        self, artist_mbid: str, *, limit: int = 100, offset: int = 0
    ) -> dict:
        """Search an artist's recordings (results carry ``first-release-date``).

        The search endpoint is how we get per-recording first-release dates;
        the browse endpoint cannot return them.

        Args:
            artist_mbid: The artist's MusicBrainz id.
            limit: Page size (max 100).
            offset: Page offset.

        Returns:
            The raw search response (``count`` + ``recordings``).
        """
        return await self.get_json(
            "recording",
            {"query": f"arid:{artist_mbid}", "limit": limit, "offset": offset},
        )

    async def browse_recordings_work_rels(
        self, artist_mbid: str, *, limit: int = 100, offset: int = 0
    ) -> dict:
        """Browse an artist's recordings with their work relations.

        ``inc=work-rels`` is valid for browse (``releases`` is not); this is how
        we recover the ``work_id`` that groups versions of a composition.

        Args:
            artist_mbid: The artist's MusicBrainz id.
            limit: Page size (max 100).
            offset: Page offset.

        Returns:
            The raw browse response (``recording-count`` + ``recordings``).
        """
        return await self.get_json(
            "recording",
            {
                "artist": artist_mbid,
                "inc": "work-rels",
                "limit": limit,
                "offset": offset,
            },
        )
