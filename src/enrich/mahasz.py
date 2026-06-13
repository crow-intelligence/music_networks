"""Mahász (slagerlistak.hu) yearly chart scraping → :class:`ChartEntry` rows.

The Hungarian record-industry association (Mahász) publishes year-end aggregated
charts at ``slagerlistak.hu``. This adds a **contemporaneous commercial
popularity** signal to the corpus: which acts/titles actually charted, and how
high. Coverage is **2003–2022** (the archive's range) across four lists —
``album_chart``, ``single_chart`` and two streaming variants — and includes
international acts, of which only the Hungarian ones match our data.

Mirrors the other enrich modules: pure, doctested parsers
(:func:`parse_rank`, :func:`strip_label`, :func:`parse_chart_page`) plus an async
:class:`MahaszClient` reusing the scraper's
:class:`~src.scraper.fetch.RateLimiter`. Each scraped line is stored **verbatim**
(``artist_raw``/``title_raw``/``label``) and best-effort linked to our own
:class:`~src.db.Performer` (fuzzy, via ``rapidfuzz``) and :class:`~src.db.Song`
(exact artist+title), so the source data is preserved regardless of match rate.

robots.txt blocks named AI/SEO bots but has no catch-all; chart rankings are
factual data. We scrape politely (descriptive UA, rate-limited).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import (
    ChartEntry,
    Performer,
    Song,
    SongPerformer,
    create_all,
    get_async_engine,
    make_async_session,
)
from src.enrich import USER_AGENT
from src.enrich.match import normalize, song_key
from src.scraper.fetch import RateLimiter

ARCHIVE = (
    "https://slagerlistak.hu/archivum/eves-osszesitett-slagerlistak-archiv-adatok"
)
_SRC = "mahasz"
MIN_YEAR, MAX_YEAR = 2003, 2022

# URL slug → the short chart name we store. Streaming lists only exist for recent
# years; an empty page (no table) is skipped, so listing them all is harmless.
CHARTS: dict[str, str] = {
    "album_chart": "album",
    "single_chart": "single",
    "stream": "stream",
    "stream_db": "stream_count",
}

# Fuzzy artist-match acceptance threshold (rapidfuzz token_sort_ratio, 0–100).
FUZZY_CUTOFF = 90.0

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass
class ChartRow:
    """One parsed chart line.

    Attributes:
        rank: 1-based chart position.
        artist: Performer credit as printed (e.g. ``"MÁGA ZOLTÁN"``).
        title: Album/song title as printed, or ``None``.
        label: Record label (parentheses stripped), or ``None``.
    """

    rank: int
    artist: str
    title: str | None
    label: str | None


def parse_rank(text: str) -> int | None:
    """Parse a rank cell like ``"1."`` into an int.

    Args:
        text: The rank cell text.

    Returns:
        The integer rank, or ``None`` if no digits are present.

    Examples:
        >>> parse_rank("1.")
        1
        >>> parse_rank("  12. ")
        12
        >>> parse_rank("No.") is None
        True
    """
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else None


def strip_label(text: str | None) -> str | None:
    """Strip the wrapping parentheses (and whitespace) from a label cell.

    Args:
        text: The label text, e.g. ``"(Universal Music)"``.

    Returns:
        The bare label, or ``None`` if empty.

    Examples:
        >>> strip_label("(Universal Music)")
        'Universal Music'
        >>> strip_label("  (Tom-Tom Records) ")
        'Tom-Tom Records'
        >>> strip_label("") is None
        True
        >>> strip_label(None) is None
        True
    """
    if not text:
        return None
    cleaned = text.strip().lstrip("(").rstrip(")").strip()
    return cleaned or None


def parse_chart_page(html: str) -> list[ChartRow]:
    """Parse a slagerlistak.hu year-end chart page into rows.

    The ranked list is a single ``<table>``; each data row carries the rank in
    ``td.no_sor`` and, in ``td.lemez_sor2``, a ``span.eloado`` (artist), the bare
    title text node(s), and a ``span.kiado_sor`` (label). Rows without a numeric
    rank or an artist are skipped (header / spacer rows).

    Args:
        html: The page HTML.

    Returns:
        The parsed :class:`ChartRow` list (empty if the page has no chart table,
        e.g. a streaming chart for a year before streaming data exists).

    Examples:
        >>> html = '''<table>
        ...   <tr><td class="no_sor">No.</td><td></td></tr>
        ...   <tr><td class="no_sor">1.</td>
        ...       <td class="lemez_sor2"><span class="eloado">OMEGA</span><br/>
        ...       Gammapolisz<br/><span class="kiado_sor">(Pepita)</span></td></tr>
        ... </table>'''
        >>> rows = parse_chart_page(html)
        >>> len(rows), rows[0].rank, rows[0].artist
        (1, 1, 'OMEGA')
        >>> rows[0].title, rows[0].label
        ('Gammapolisz', 'Pepita')
        >>> parse_chart_page("<p>no table here</p>")
        []
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return []
    rows: list[ChartRow] = []
    for tr in table.find_all("tr"):
        rank_cell = tr.find("td", class_="no_sor") or tr.find("td")
        rank = parse_rank(rank_cell.get_text()) if rank_cell else None
        body = tr.find("td", class_="lemez_sor2")
        if rank is None or body is None:
            continue
        eloado = body.find("span", class_="eloado")
        artist = eloado.get_text(strip=True) if eloado else ""
        if not artist:
            continue
        kiado = body.find("span", class_="kiado_sor")
        label = strip_label(kiado.get_text()) if kiado else None
        # Title = the cell's direct text nodes (between the <br>s), i.e. not the
        # artist/label spans.
        title_text = " ".join(
            t.strip() for t in body.find_all(string=True, recursive=False) if t.strip()
        )
        rows.append(ChartRow(rank, artist, title_text or None, label))
    return rows


def chart_url(slug: str, year: int) -> str:
    """Build the archive URL for a chart ``slug`` and ``year``.

    Args:
        slug: One of :data:`CHARTS` keys (``"album_chart"`` …).
        year: Four-digit year.

    Returns:
        The full page URL.

    Examples:
        >>> chart_url("album_chart", 2010).endswith("/album_chart/2010")
        True
    """
    return f"{ARCHIVE}/{slug}/{year}"


def build_performer_index(
    rows: list[tuple[int, str]],
) -> dict[str, int]:
    """Build a ``normalize(name) → performer_id`` index for exact matching.

    Args:
        rows: ``(performer_id, name)`` pairs.

    Returns:
        Map from normalized name to the first performer id seen for it.

    Examples:
        >>> build_performer_index([(1, "Omega"), (2, "Kis Grófo"), (1, "Omega")])
        {'omega': 1, 'kis grofo': 2}
    """
    index: dict[str, int] = {}
    for pid, name in rows:
        index.setdefault(normalize(name), pid)
    return index


def match_artist(
    artist_raw: str,
    index: dict[str, int],
    *,
    cutoff: float = FUZZY_CUTOFF,
) -> int | None:
    """Match a chart artist to a performer id: exact-normalized, then fuzzy.

    Args:
        artist_raw: The chart's printed artist string.
        index: ``normalize(name) → performer_id`` (see
            :func:`build_performer_index`).
        cutoff: Minimum ``rapidfuzz`` token-sort ratio (0–100) to accept a fuzzy
            match.

    Returns:
        The matched performer id, or ``None``.

    Examples:
        >>> idx = {"omega": 7, "tankcsapda": 9}
        >>> match_artist("Omega", idx)
        7
        >>> match_artist("Omega!!", idx)  # normalizes to 'omega'
        7
        >>> match_artist("Beyoncé", idx) is None
        True
    """
    key = normalize(artist_raw)
    if not key:
        return None
    if key in index:
        return index[key]
    from rapidfuzz import fuzz, process

    hit = process.extractOne(
        key, index.keys(), scorer=fuzz.token_sort_ratio, score_cutoff=cutoff
    )
    return index[hit[0]] if hit else None


class MahaszClient:
    """Polite async client for slagerlistak.hu chart pages."""

    def __init__(
        self, *, delay: float = 2.0, timeout: float = 30.0, max_retries: int = 4
    ) -> None:
        """Configure the client.

        Args:
            delay: Minimum seconds between requests.
            timeout: Per-request timeout in seconds.
            max_retries: Attempts for transient failures before giving up.
        """
        self._limiter = RateLimiter(delay)
        self._timeout = timeout
        self._max_retries = max_retries

    async def fetch(self, url: str) -> str | None:
        """GET a chart page, returning its HTML (``None`` on a 404).

        Retries transient failures (timeouts, 429/5xx) with backoff; a 404
        (year/chart combination that does not exist) returns ``None``.

        Args:
            url: The page URL.

        Returns:
            The HTML text, or ``None`` if the page is absent.

        Raises:
            httpx.HTTPError: On a non-retryable error or exhausted retries.
        """
        last_exc: httpx.HTTPError | None = None
        for attempt in range(self._max_retries):
            await self._limiter.acquire()
            try:
                async with httpx.AsyncClient(
                    timeout=self._timeout,
                    headers={"User-Agent": USER_AGENT},
                    follow_redirects=True,
                ) as client:
                    resp = await client.get(url)
            except httpx.TransportError as exc:
                last_exc = exc
            else:
                if resp.status_code == 404:
                    return None
                if resp.status_code not in _RETRY_STATUS:
                    resp.raise_for_status()
                    return resp.text
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
            await asyncio.sleep(min(2.0**attempt, 30.0))
        assert last_exc is not None
        raise last_exc


class MahaszScraper:
    """Scrapes the yearly charts into ``ChartEntry`` rows and matches them."""

    def __init__(self, client: MahaszClient, db_path: str = "data/music.db") -> None:
        """Create a scraper.

        Args:
            client: A configured :class:`MahaszClient`.
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
        self,
        *,
        years: range | None = None,
        charts: list[str] | None = None,
        resume: bool = True,
    ) -> dict[str, int]:
        """Scrape every (chart, year), upsert rows, and link them to our data.

        Args:
            years: Years to scrape (default :data:`MIN_YEAR`–:data:`MAX_YEAR`).
            charts: Slugs to scrape (default all of :data:`CHARTS`).
            resume: Skip a (chart, year) already present in ``chart_entry``.

        Returns:
            Counts ``{"pages", "entries", "matched_performers", "matched_songs"}``.
        """
        await self.init_db()
        years = years or range(MIN_YEAR, MAX_YEAR + 1)
        slugs = charts or list(CHARTS)
        perf_index, song_index = await self._load_indices()
        counts = {"pages": 0, "entries": 0, "matched_performers": 0, "matched_songs": 0}
        for slug in slugs:
            chart = CHARTS.get(slug, slug)
            for year in years:
                if resume and await self._already_scraped(chart, year):
                    continue
                html = await self._client.fetch(chart_url(slug, year))
                if not html:
                    continue
                rows = parse_chart_page(html)
                if not rows:
                    continue
                counts["pages"] += 1
                stats = await self._store(chart, year, rows, perf_index, song_index)
                counts["entries"] += stats[0]
                counts["matched_performers"] += stats[1]
                counts["matched_songs"] += stats[2]
        return counts

    async def _load_indices(self) -> tuple[dict[str, int], dict[str, int]]:
        """Build the performer-name and ``artist|title`` → song indices."""
        async with self._session_factory() as session:
            perfs = (
                await session.execute(select(Performer.id, Performer.name))
            ).all()
            songs = (
                await session.execute(
                    select(Performer.name, Song.title, Song.id)
                    .join(SongPerformer, SongPerformer.song_id == Song.id)
                    .join(Performer, Performer.id == SongPerformer.performer_id)
                )
            ).all()
        perf_index = build_performer_index([(pid, name) for pid, name in perfs])
        song_index: dict[str, int] = {}
        for name, title, sid in songs:
            song_index.setdefault(song_key(name, title), sid)
        return perf_index, song_index

    async def _already_scraped(self, chart: str, year: int) -> bool:
        """Whether any ``chart_entry`` row exists for this (chart, year)."""
        async with self._session_factory() as session:
            found = (
                await session.execute(
                    select(ChartEntry.id)
                    .where(
                        ChartEntry.source == _SRC,
                        ChartEntry.chart == chart,
                        ChartEntry.year == year,
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
        return found is not None

    async def _store(
        self,
        chart: str,
        year: int,
        rows: list[ChartRow],
        perf_index: dict[str, int],
        song_index: dict[str, int],
    ) -> tuple[int, int, int]:
        """Upsert one year's rows; return (entries, matched_perf, matched_song)."""
        entries = matched_p = matched_s = 0
        async with self._session_factory() as session:
            for row in rows:
                performer_id = match_artist(row.artist, perf_index)
                song_id = None
                if performer_id is not None and row.title:
                    song_id = song_index.get(song_key(row.artist, row.title))
                matched_p += performer_id is not None
                matched_s += song_id is not None
                await self._upsert(session, chart, year, row, performer_id, song_id)
                entries += 1
            await session.commit()
        return entries, matched_p, matched_s

    @staticmethod
    async def _upsert(
        session: AsyncSession,
        chart: str,
        year: int,
        row: ChartRow,
        performer_id: int | None,
        song_id: int | None,
    ) -> None:
        """Insert-or-update one chart entry (unique on source/chart/year/rank)."""
        await session.execute(
            sqlite_insert(ChartEntry)
            .values(
                source=_SRC,
                chart=chart,
                year=year,
                rank=row.rank,
                artist_raw=row.artist,
                title_raw=row.title,
                label=row.label,
                performer_id=performer_id,
                song_id=song_id,
            )
            .on_conflict_do_update(
                index_elements=["source", "chart", "year", "rank"],
                set_={
                    "artist_raw": row.artist,
                    "title_raw": row.title,
                    "label": row.label,
                    "performer_id": performer_id,
                    "song_id": song_id,
                },
            )
        )


def main() -> None:
    """CLI: scrape the Mahász yearly charts into the database."""
    import argparse

    ap = argparse.ArgumentParser(description="Scrape Mahász (slagerlistak.hu) charts.")
    ap.add_argument("--db", default="data/music.db")
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--from-year", type=int, default=MIN_YEAR)
    ap.add_argument("--to-year", type=int, default=MAX_YEAR)
    ap.add_argument(
        "--charts", nargs="*", choices=list(CHARTS), default=None, help="chart slugs"
    )
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    async def _go() -> None:
        scraper = MahaszScraper(MahaszClient(delay=args.delay), db_path=args.db)
        try:
            counts = await scraper.run(
                years=range(args.from_year, args.to_year + 1),
                charts=args.charts,
                resume=not args.no_resume,
            )
            print(
                f"Scraped {counts['pages']} chart pages, {counts['entries']} entries; "
                f"matched {counts['matched_performers']} to performers, "
                f"{counts['matched_songs']} to songs."
            )
        finally:
            await scraper.close()

    asyncio.run(_go())


if __name__ == "__main__":
    main()
