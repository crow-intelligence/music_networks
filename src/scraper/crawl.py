"""Crawl orchestrator: discover -> fetch pool -> single SQLite writer.

Design:
    * Discovery seeds ``crawl_state`` with band-page URLs (one per band).
    * ``crawl`` processes pending *band* rows (each yields song URLs + members),
      then pending *song* rows. Many fetchers run concurrently, but **all DB
      writes funnel through one writer coroutine** via an ``asyncio.Queue`` —
      so SQLite never sees competing writers.
    * Progress lives in ``crawl_state`` (``pending``/``done``/``failed``), so
      stopping and restarting simply resumes the remaining work.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from string import ascii_lowercase

from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import (
    Author,
    Band,
    BandMember,
    BandSong,
    Composer,
    CrawlState,
    Performer,
    Person,
    Song,
    SongAuthor,
    SongComposer,
    SongPerformer,
    create_all,
    get_async_engine,
    make_async_session,
)
from src.scraper import BASE_URL
from src.scraper.fetch import Fetcher, FetchOutcome
from src.scraper.parse import (
    BandPage,
    SongRecord,
    band_slug,
    extract_band_id,
    extract_song_id,
    parse_band_links,
    parse_band_page,
    parse_last_index_page,
    parse_song_page,
    split_aka,
)

# Default crawl alphabet: the same initials the original scraper used.
DEFAULT_INITIALS = ("09", *ascii_lowercase)

# Max times a failed URL is retried before we stop attempting it.
MAX_ATTEMPTS = 3

_WRITER_STOP = object()


@dataclass
class FetchResult:
    """A fetched+parsed unit of work handed to the writer.

    Attributes:
        kind: ``"band"`` or ``"song"``.
        url: The (relative) crawl-state URL this result is for.
        band_id: Owning band id (set for song results).
        payload: Parsed object (:class:`~src.scraper.parse.BandPage` or
            :class:`~src.scraper.parse.SongRecord`), or ``None`` on failure.
        error: Failure reason, or ``None`` on success.
    """

    kind: str
    url: str
    band_id: int | None
    payload: BandPage | SongRecord | None
    error: str | None


def absolute(url: str) -> str:
    """Make a site-relative href absolute.

    Args:
        url: An href, possibly already absolute.

    Returns:
        An absolute ``https://...`` URL.
    """
    return url if url.startswith("http") else BASE_URL + url


class Crawler:
    """Coordinates discovery and resumable scraping into the SQLite DB."""

    def __init__(
        self,
        fetcher: Fetcher,
        db_path: str = "data/music.db",
        *,
        concurrency: int = 4,
        batch_size: int = 200,
    ) -> None:
        """Create a crawler.

        Args:
            fetcher: Configured :class:`~src.scraper.fetch.Fetcher`.
            db_path: Path to the SQLite database file.
            concurrency: Number of concurrent fetch workers.
            batch_size: Pending rows loaded into memory per processing batch,
                so a full run never materializes all work at once.
        """
        self._fetcher = fetcher
        self._engine = get_async_engine(db_path)
        self._session_factory = make_async_session(self._engine)
        self._concurrency = concurrency
        self._batch_size = batch_size

    async def init_db(self) -> None:
        """Create database tables if they do not yet exist."""
        await create_all(self._engine)

    async def close(self) -> None:
        """Dispose of the database engine."""
        await self._engine.dispose()

    # -- discovery ---------------------------------------------------------
    async def discover(
        self, initials: tuple[str, ...] = DEFAULT_INITIALS, max_bands: int | None = None
    ) -> int:
        """Seed ``crawl_state`` with band-page URLs from the artist index.

        Args:
            initials: Index pages to crawl (``09`` + ``a``..``z`` by default).
            max_bands: Optional cap on the number of bands seeded.

        Returns:
            The number of band URLs newly seeded as ``pending``.
        """
        await self.init_db()
        band_urls: list[str] = []
        seen: set[str] = set()
        for ch in initials:
            html = (await self._fetcher.get(f"{BASE_URL}eloadok/{ch}")).html
            if not html:
                continue
            last = parse_last_index_page(html)
            pages = [html]
            for page in range(2, last + 1):
                extra = (
                    await self._fetcher.get(f"{BASE_URL}eloadok/{ch}&page={page}")
                ).html
                if extra:
                    pages.append(extra)
            for page_html in pages:
                for href in parse_band_links(page_html):
                    if href not in seen:
                        seen.add(href)
                        band_urls.append(href)
            if max_bands and len(band_urls) >= max_bands:
                break

        if max_bands:
            band_urls = band_urls[:max_bands]

        async with self._session_factory() as session:
            for url in band_urls:
                await session.execute(
                    sqlite_insert(CrawlState)
                    .values(url=url, kind="band", status="pending")
                    .on_conflict_do_nothing(index_elements=["url"])
                )
            await session.commit()
        return len(band_urls)

    # -- crawling ----------------------------------------------------------
    async def crawl(self, limit: int | None = None) -> dict[str, int]:
        """Process pending bands (creating song work), then pending songs.

        Args:
            limit: Optional cap on rows processed per kind (for smoke tests).

        Returns:
            A dict with the number of band and song rows processed.
        """
        bands = await self._run_kind("band", limit)
        songs = await self._run_kind("song", limit)
        return {"bands": bands, "songs": songs}

    async def _load_pending(
        self, kind: str, after_url: str, limit: int
    ) -> list[tuple]:
        """Return the next ``(url, band_id)`` rows needing work for ``kind``.

        Uses keyset pagination on ``url`` (``url > after_url``) so a single run
        visits each eligible row once, in order — rows that fail this run keep
        their ``url`` and so are not re-loaded until the next ``crawl``.
        """
        async with self._session_factory() as session:
            stmt = (
                select(CrawlState.url, CrawlState.band_id)
                .where(
                    CrawlState.kind == kind,
                    CrawlState.status != "done",
                    CrawlState.attempts < MAX_ATTEMPTS,
                    CrawlState.url > after_url,
                )
                .order_by(CrawlState.url)
                .limit(limit)
            )
            return list((await session.execute(stmt)).all())

    async def _run_kind(self, kind: str, limit: int | None) -> int:
        """Fetch+parse pending rows of ``kind`` in batches through the writer.

        Processes work in ``batch_size`` chunks (keyset-paginated by ``url``) so
        memory stays bounded regardless of how much work is outstanding.
        """
        processed = 0
        after_url = ""
        while limit is None or processed < limit:
            batch = self._batch_size
            if limit is not None:
                batch = min(batch, limit - processed)
            rows = await self._load_pending(kind, after_url, batch)
            if not rows:
                break
            after_url = rows[-1][0]
            await self._process_batch(kind, rows)
            processed += len(rows)
        return processed

    async def _process_batch(self, kind: str, rows: list[tuple]) -> None:
        """Fetch+parse one batch of rows through the single writer coroutine."""
        write_q: asyncio.Queue = asyncio.Queue()
        sem = asyncio.Semaphore(self._concurrency)

        async def worker(url: str, band_id: int | None) -> None:
            async with sem:
                outcome = await self._fetcher.get(absolute(url))
            await write_q.put(self._build_result(kind, url, band_id, outcome))

        writer_task = asyncio.create_task(self._writer(write_q))
        await asyncio.gather(*(worker(url, bid) for url, bid in rows))
        await write_q.put(_WRITER_STOP)
        await writer_task

    def _build_result(
        self, kind: str, url: str, band_id: int | None, outcome: FetchOutcome
    ) -> FetchResult:
        """Parse a fetch outcome into a :class:`FetchResult` (off the writer)."""
        if outcome.html is None:
            error = outcome.error or "fetch failed"
            return FetchResult(kind, url, band_id, None, error)
        html = outcome.html
        try:
            if kind == "band":
                payload = parse_band_page(html, band_slug(url))
            else:
                payload = parse_song_page(html)
            return FetchResult(kind, url, band_id, payload, None)
        except Exception as exc:  # noqa: BLE001 - record any parse failure
            return FetchResult(kind, url, band_id, None, f"parse error: {exc}")

    # -- the single writer -------------------------------------------------
    async def _writer(self, write_q: asyncio.Queue) -> None:
        """Consume results and apply all DB writes from one coroutine."""
        async with self._session_factory() as session:
            while True:
                item = await write_q.get()
                if item is _WRITER_STOP:
                    return
                try:
                    if item.error:
                        await self._mark_failed(session, item.url, item.error)
                    elif item.kind == "band":
                        await self._save_band(session, item)
                        await self._mark_done(session, item.url)
                    else:
                        await self._save_song(session, item)
                        await self._mark_done(session, item.url)
                    await session.commit()
                except Exception as exc:  # noqa: BLE001 - never kill the writer
                    await session.rollback()
                    await self._mark_failed(session, item.url, f"write error: {exc}")
                    await session.commit()

    async def _save_band(self, session: AsyncSession, item: FetchResult) -> None:
        """Persist a band, its members, and queue its songs as pending."""
        band_id = extract_band_id(item.url)
        slug = band_slug(item.url)
        page = item.payload
        assert isinstance(page, BandPage)
        await session.execute(
            sqlite_insert(Band)
            .values(id=band_id, slug=slug, url=item.url)
            .on_conflict_do_nothing(index_elements=["id"])
        )
        for name, uri in page.members:
            person = await self._get_or_create_person(session, name, uri)
            await session.execute(
                sqlite_insert(BandMember)
                .values(band_id=band_id, person_id=person.id)
                .on_conflict_do_nothing()
            )
        for song_url in page.song_urls:
            await session.execute(
                sqlite_insert(CrawlState)
                .values(url=song_url, kind="song", status="pending", band_id=band_id)
                .on_conflict_do_nothing(index_elements=["url"])
            )

    async def _save_song(self, session: AsyncSession, item: FetchResult) -> None:
        """Persist a song with its performers, authors, and composers."""
        song_id = extract_song_id(item.url)
        assert song_id is not None
        rec = item.payload
        assert isinstance(rec, SongRecord)
        await session.execute(
            sqlite_insert(Song)
            .values(
                id=song_id,
                title=rec.title,
                lyrics=rec.lyrics,
                year=rec.year,
                album=rec.album,
                label=rec.label,
                length=rec.length,
            )
            .on_conflict_do_update(
                index_elements=["id"],
                set_={
                    "title": rec.title,
                    "lyrics": rec.lyrics,
                    "year": rec.year,
                    "album": rec.album,
                    "label": rec.label,
                    "length": rec.length,
                },
            )
        )
        await self._link_credits(
            session, song_id, rec.performers, Performer, SongPerformer, "performer_id"
        )
        await self._link_credits(
            session, song_id, rec.authors, Author, SongAuthor, "author_id"
        )
        await self._link_credits(
            session, song_id, rec.composers, Composer, SongComposer, "composer_id"
        )
        if item.band_id is not None:
            await session.execute(
                sqlite_insert(BandSong)
                .values(band_id=item.band_id, song_id=song_id)
                .on_conflict_do_nothing()
            )

    async def _link_credits(
        self, session: AsyncSession, song_id: int, raw_names, model, link_model, fk: str
    ) -> None:
        """Get-or-create each credited name and link it to the song."""
        for raw in raw_names:
            name, aka = split_aka(raw)
            if not name:
                continue
            obj = await self._get_or_create_named(session, model, name, aka)
            await session.execute(
                sqlite_insert(link_model)
                .values(song_id=song_id, **{fk: obj.id})
                .on_conflict_do_nothing()
            )

    async def _get_or_create_named(
        self, session: AsyncSession, model, name: str, aka: str
    ):
        """Return an existing row by ``name`` or create one."""
        found = (
            await session.execute(select(model).where(model.name == name))
        ).scalar_one_or_none()
        if found is not None:
            return found
        obj = model(name=name, aka=aka)
        session.add(obj)
        await session.flush()
        return obj

    async def _get_or_create_person(self, session: AsyncSession, name: str, uri: str):
        """Return an existing person by ``uri`` or create one."""
        found = (
            await session.execute(select(Person).where(Person.uri == uri))
        ).scalar_one_or_none()
        if found is not None:
            return found
        obj = Person(name=name, uri=uri)
        session.add(obj)
        await session.flush()
        return obj

    async def _mark_done(self, session: AsyncSession, url: str) -> None:
        """Mark a crawl-state row as successfully scraped."""
        await session.execute(
            update(CrawlState).where(CrawlState.url == url).values(status="done")
        )

    async def _mark_failed(self, session: AsyncSession, url: str, error: str) -> None:
        """Mark a crawl-state row failed and bump its attempt counter."""
        await session.execute(
            update(CrawlState)
            .where(CrawlState.url == url)
            .values(
                status="failed",
                last_error=error[:500],
                attempts=CrawlState.attempts + 1,
            )
        )
