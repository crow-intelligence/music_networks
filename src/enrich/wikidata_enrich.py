"""Wikidata enrichment: Hungarian artists -> canonical :class:`Artist` rows.

Two phases, both resumable:

1. **Facts** — page through every Hungarian musician/band on Wikidata
   (:meth:`WikidataClient.fetch_hungarian_artists`) and upsert each as an
   :class:`~src.db.Artist`, idempotent on its Wikidata id. Where the artist's
   name matches one of our credit strings, link ``Performer.artist_id`` /
   ``Person.artist_id`` to it. Names are compared with the shared
   :func:`~src.enrich.match.normalize` so accents/case/punctuation don't matter.
   Artists with a Hungarian Wikipedia article are queued in ``crawl_state``
   (``kind='wd_artist'``) for phase 2.

2. **Bios** — drain the queued ``wd_artist`` rows, fetching each article's intro
   paragraph (:meth:`WikidataClient.huwiki_extract`) into ``Artist.description``.

The phase-1 upserts are idempotent (``ON CONFLICT`` / fill-if-empty) and phase 2
rides the same ``crawl_state`` mechanism the scraper and MusicBrainz enumeration
use, so the whole command can be stopped and re-run freely.
"""

from __future__ import annotations

from typing import cast

from sqlalchemy import CursorResult, func, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import (
    Artist,
    ArtistExternal,
    CrawlState,
    Performer,
    Person,
    SongPerformer,
    create_all,
    get_async_engine,
    make_async_session,
)
from src.enrich.match import normalize
from src.enrich.wikidata import ArtistInfo, WikidataClient

_WD = "wikidata"
_MB = "musicbrainz"
_KIND = "wd_artist"
_SEARCH_KIND = "wd_search"
MAX_ATTEMPTS = 3


class WikidataEnricher:
    """Enriches the corpus with Wikidata facts + Hungarian Wikipedia bios."""

    def __init__(self, client: WikidataClient, db_path: str = "data/music.db") -> None:
        """Create an enricher.

        Args:
            client: A configured :class:`~src.enrich.wikidata.WikidataClient`.
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

    async def import_facts(self, *, max_pages: int | None = None) -> dict[str, int]:
        """Fetch Hungarian artists and upsert them, linking credit strings.

        Args:
            max_pages: Optional cap on SPARQL pages fetched (for small runs).

        Returns:
            Counts ``{"fetched", "linked_performers", "linked_persons",
            "queued"}``.
        """
        await self.init_db()
        artists = await self._client.fetch_hungarian_artists(max_pages=max_pages)
        async with self._session_factory() as session:
            performers = await self._name_map(session, Performer.name, Performer.id)
            persons = await self._name_map(session, Person.name, Person.id)
            counts = {
                "fetched": len(artists),
                "linked_performers": 0,
                "linked_persons": 0,
                "queued": 0,
            }
            for info in artists:
                artist_id = await self._upsert_artist(session, info)
                key = normalize(info.name)
                if key in performers:
                    counts["linked_performers"] += await self._link(
                        session, Performer, performers[key], artist_id
                    )
                if key in persons:
                    counts["linked_persons"] += await self._link(
                        session, Person, persons[key], artist_id
                    )
                if info.huwiki_title:
                    counts["queued"] += await self._queue_bio(session, info.qid)
            await session.commit()
        return counts

    async def resolve_unlinked(
        self, *, limit: int | None = None, min_songs: int = 1
    ) -> dict[str, int]:
        """Per-name fallback: resolve still-unlinked performers via label search.

        The bulk :meth:`import_facts` only links performers whose name matches a
        Hungarian artist already in Wikidata's set. This catches the rest by
        searching each unlinked performer's name (most-recorded first), keeping
        only a confident music-entity match (:meth:`WikidataClient.resolve_name`),
        then upserting + linking it. Resumable via ``crawl_state``
        (``kind='wd_search'``): an already-attempted performer is skipped.

        Args:
            limit: Optional cap on performers attempted this run.
            min_songs: Only consider performers credited on at least this many
                songs (skip the once-off long tail).

        Returns:
            Counts ``{"attempted", "resolved"}``.
        """
        await self.init_db()
        async with self._session_factory() as session:
            candidates = (
                await session.execute(
                    select(Performer.id, Performer.name)
                    .join(SongPerformer, SongPerformer.performer_id == Performer.id)
                    .where(Performer.artist_id.is_(None))
                    .group_by(Performer.id)
                    .having(func.count(SongPerformer.song_id) >= min_songs)
                    .order_by(func.count(SongPerformer.song_id).desc())
                )
            ).all()

        counts = {"attempted": 0, "resolved": 0}
        for performer_id, name in candidates:
            if limit is not None and counts["attempted"] >= limit:
                break
            if await self._search_done(performer_id):
                continue
            counts["attempted"] += 1
            try:
                info = await self._client.resolve_name(name)
            except Exception as exc:  # noqa: BLE001 - record and retry later
                await self._mark_search(performer_id, "failed", str(exc)[:500])
                continue
            if info is None:
                await self._mark_search(performer_id, "done", None)
                continue
            async with self._session_factory() as session:
                artist_id = await self._upsert_artist(session, info)
                await self._link(session, Performer, performer_id, artist_id)
                if info.huwiki_title:
                    await self._queue_bio(session, info.qid)
                await session.commit()
            await self._mark_search(performer_id, "done", None)
            counts["resolved"] += 1
        return counts

    async def _search_done(self, performer_id: int) -> bool:
        """Whether a performer's per-name search is done / exhausted retries."""
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(CrawlState.status, CrawlState.attempts).where(
                        CrawlState.url == f"performer:{performer_id}",
                        CrawlState.kind == _SEARCH_KIND,
                    )
                )
            ).first()
        if row is None:
            return False
        status, attempts = row
        return status == "done" or attempts >= MAX_ATTEMPTS

    async def _mark_search(
        self, performer_id: int, status: str, error: str | None
    ) -> None:
        """Upsert a ``wd_search`` crawl-state row for a performer."""
        async with self._session_factory() as session:
            set_: dict = {"status": status, "last_error": error}
            if status == "failed":
                set_["attempts"] = CrawlState.attempts + 1
            await session.execute(
                sqlite_insert(CrawlState)
                .values(
                    url=f"performer:{performer_id}",
                    kind=_SEARCH_KIND,
                    status=status,
                    attempts=1 if status == "failed" else 0,
                    last_error=error,
                )
                .on_conflict_do_update(index_elements=["url"], set_=set_)
            )
            await session.commit()

    async def fetch_bios(self, *, limit: int | None = None) -> int:
        """Drain queued ``wd_artist`` rows, filling ``Artist.description``.

        Args:
            limit: Optional cap on articles fetched (for small runs).

        Returns:
            The number of descriptions written.
        """
        written = 0
        while limit is None or written < limit:
            async with self._session_factory() as session:
                qid = (
                    await session.execute(
                        select(CrawlState.url)
                        .where(
                            CrawlState.kind == _KIND,
                            CrawlState.status != "done",
                            CrawlState.attempts < MAX_ATTEMPTS,
                        )
                        .order_by(CrawlState.url)
                        .limit(1)
                    )
                ).scalar_one_or_none()
            if qid is None:
                break
            if await self._fetch_one_bio(qid):
                written += 1
        return written

    async def _fetch_one_bio(self, qid: str) -> bool:
        """Fetch + store one article's intro; mark its queue row done/failed."""
        async with self._session_factory() as session:
            artist = (
                await session.execute(select(Artist).where(Artist.wikidata_id == qid))
            ).scalar_one_or_none()
        title = artist.huwiki_title if artist else None
        if not title:
            await self._mark_bio(qid, "done", None)
            return False
        try:
            extract = await self._client.huwiki_extract(title)
        except Exception as exc:  # noqa: BLE001 - record and move on
            await self._mark_bio(qid, "failed", str(exc)[:500])
            return False
        if extract:
            async with self._session_factory() as session:
                await session.execute(
                    update(Artist)
                    .where(Artist.wikidata_id == qid)
                    .values(description=extract)
                )
                await session.commit()
        await self._mark_bio(qid, "done", None)
        return extract is not None

    @staticmethod
    async def _name_map(session: AsyncSession, name_col, id_col) -> dict[str, int]:
        """Build a ``normalize(name) -> id`` map for a credit-string table."""
        rows = (await session.execute(select(name_col, id_col))).all()
        mapping: dict[str, int] = {}
        for name, row_id in rows:
            mapping.setdefault(normalize(name), row_id)
        return mapping

    async def _upsert_artist(self, session: AsyncSession, info: ArtistInfo) -> int:
        """Upsert an :class:`Artist` (idempotent on Wikidata id); link externals.

        Structured facts from Wikidata are authoritative and overwritten on
        update; ``description`` (a fetched Wikipedia bio) is never clobbered.
        """
        fields = {
            "name": info.name,
            "kind": info.kind,
            "mb_artist_id": info.mb_artist_id,
            "begin_date": info.begin_date,
            "end_date": info.end_date,
            "birthplace": info.birthplace,
            "genre": info.genre,
            "huwiki_title": info.huwiki_title,
            "huwiki_url": info.huwiki_url,
        }
        existing = (
            await session.execute(
                select(Artist.id).where(Artist.wikidata_id == info.qid)
            )
        ).scalar_one_or_none()
        if existing is not None:
            await session.execute(
                update(Artist).where(Artist.id == existing).values(**fields)
            )
            artist_id = existing
        else:
            artist = Artist(wikidata_id=info.qid, **fields)
            session.add(artist)
            await session.flush()
            artist_id = artist.id
        await self._link_external(
            session,
            artist_id,
            _WD,
            info.qid,
            f"https://www.wikidata.org/wiki/{info.qid}",
        )
        if info.mb_artist_id:
            await self._link_external(
                session,
                artist_id,
                _MB,
                info.mb_artist_id,
                f"https://musicbrainz.org/artist/{info.mb_artist_id}",
            )
        return artist_id

    @staticmethod
    async def _link_external(
        session: AsyncSession, artist_id: int, source: str, external_id: str, url: str
    ) -> None:
        """Idempotently record an external id for an artist."""
        await session.execute(
            sqlite_insert(ArtistExternal)
            .values(
                artist_id=artist_id, source=source, external_id=external_id, url=url
            )
            .on_conflict_do_nothing(index_elements=["source", "external_id"])
        )

    @staticmethod
    async def _link(
        session: AsyncSession,
        model: type[Performer] | type[Person],
        row_id: int,
        artist_id: int,
    ) -> int:
        """Point a credit-string row at its canonical artist (only if unset).

        Returns ``1`` if a row was newly linked, else ``0``.
        """
        result = cast(
            CursorResult,
            await session.execute(
                update(model)
                .where(model.id == row_id, model.artist_id.is_(None))
                .values(artist_id=artist_id)
            ),
        )
        return result.rowcount or 0

    @staticmethod
    async def _queue_bio(session: AsyncSession, qid: str) -> int:
        """Queue a ``wd_artist`` crawl-state row for bio fetching.

        Returns ``1`` if newly queued, else ``0``.
        """
        result = cast(
            CursorResult,
            await session.execute(
                sqlite_insert(CrawlState)
                .values(url=qid, kind=_KIND, status="pending")
                .on_conflict_do_nothing(index_elements=["url"])
            ),
        )
        return result.rowcount or 0

    async def _mark_bio(self, qid: str, status: str, error: str | None) -> None:
        """Update a ``wd_artist`` crawl-state row's status/attempts."""
        async with self._session_factory() as session:
            values: dict = {"status": status, "last_error": error}
            if status == "failed":
                values["attempts"] = CrawlState.attempts + 1
            await session.execute(
                update(CrawlState).where(CrawlState.url == qid).values(**values)
            )
            await session.commit()
