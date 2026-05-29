"""MusicBrainz enumeration: artists by area -> recordings -> song versions.

This is the (b) "all occurrences" crawl: every recording of every Hungarian
artist becomes its own :class:`~src.db.Song` row (a version), carrying its own
first-release date and the ``work_id`` that groups it with other versions of
the same composition. Lyric-less rows are fine — lyrics are attached later from
zeneszoveg/Genius.

Progress is resumable via ``crawl_state`` rows with ``kind='mb_artist'`` (the
same proven mechanism the scraper uses): each artist is ``pending`` until its
recordings are stored, then ``done``. Writes are idempotent on
``song_external (source='musicbrainz', external_id=<recording id>)``.

After enumeration, :func:`link_remakes` groups versions by ``work_id`` and
points every later version at the earliest one.
"""

from __future__ import annotations

from sqlalchemy import func, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import (
    CrawlState,
    Performer,
    Song,
    SongExternal,
    SongPerformer,
    create_all,
    get_async_engine,
    make_async_session,
)
from src.enrich.musicbrainz import (
    HUNGARY_AREA_MBID,
    MusicBrainzClient,
    RecordingInfo,
    date_year,
    parse_recording,
)

_MB = "musicbrainz"
MAX_ATTEMPTS = 3


class Enumerator:
    """Enumerates a MusicBrainz area into song-version rows."""

    def __init__(
        self,
        client: MusicBrainzClient,
        db_path: str = "data/music.db",
        *,
        page_size: int = 100,
    ) -> None:
        """Create an enumerator.

        Args:
            client: A configured :class:`~src.enrich.musicbrainz.MusicBrainzClient`.
            db_path: Path to the SQLite database file.
            page_size: MusicBrainz browse page size (max 100).
        """
        self._client = client
        self._engine = get_async_engine(db_path)
        self._session_factory = make_async_session(self._engine)
        self._page_size = page_size

    async def init_db(self) -> None:
        """Create database tables if they do not yet exist."""
        await create_all(self._engine)

    async def close(self) -> None:
        """Dispose of the database engine."""
        await self._engine.dispose()

    async def seed_artists(
        self, area_mbid: str = HUNGARY_AREA_MBID, *, max_artists: int | None = None
    ) -> int:
        """Seed ``crawl_state`` with ``mb_artist`` rows for an area.

        Args:
            area_mbid: The area's MusicBrainz id (Hungary by default).
            max_artists: Optional cap on artists seeded (for small runs).

        Returns:
            The number of artist ids newly seeded as ``pending``.
        """
        await self.init_db()
        seeded = 0
        offset = 0
        while True:
            data = await self._client.browse_artists_by_area(
                area_mbid, limit=self._page_size, offset=offset
            )
            artists = data.get("artists", [])
            if not artists:
                break
            async with self._session_factory() as session:
                for artist in artists:
                    result = await session.execute(
                        sqlite_insert(CrawlState)
                        .values(url=artist["id"], kind="mb_artist", status="pending")
                        .on_conflict_do_nothing(index_elements=["url"])
                    )
                    seeded += result.rowcount or 0
                await session.commit()
            offset += len(artists)
            if max_artists and offset >= max_artists:
                break
            if offset >= data.get("artist-count", offset):
                break
        return seeded

    async def run(self, limit: int | None = None) -> int:
        """Process pending ``mb_artist`` rows, storing their recordings.

        Args:
            limit: Optional cap on artists processed (for small runs).

        Returns:
            The number of artists processed.
        """
        processed = 0
        while limit is None or processed < limit:
            async with self._session_factory() as session:
                row = (
                    await session.execute(
                        select(CrawlState.url)
                        .where(
                            CrawlState.kind == "mb_artist",
                            CrawlState.status == "pending",
                            CrawlState.attempts < MAX_ATTEMPTS,
                        )
                        .order_by(CrawlState.url)
                        .limit(1)
                    )
                ).scalar_one_or_none()
            if row is None:
                break
            await self._process_artist(row)
            processed += 1
        return processed

    async def _process_artist(self, artist_mbid: str) -> None:
        """Fetch all recordings for one artist and persist them as songs."""
        try:
            recordings = await self._fetch_all_recordings(artist_mbid)
        except Exception as exc:  # noqa: BLE001 - record and move on
            await self._mark_artist(artist_mbid, "failed", str(exc)[:500])
            return
        async with self._session_factory() as session:
            for rec in recordings:
                await self._save_recording(session, parse_recording(rec))
            await session.commit()
        await self._mark_artist(artist_mbid, "done", None)

    async def _fetch_all_recordings(self, artist_mbid: str) -> list[dict]:
        """Collect an artist's recordings with dates and work ids.

        Two passes (the API can give one or the other, not both): *search* for
        ``first-release-date``, *browse* with ``work-rels`` for ``work_id``;
        merged by recording id.
        """
        recordings: list[dict] = []
        offset = 0
        while True:
            data = await self._client.search_recordings_by_artist(
                artist_mbid, limit=self._page_size, offset=offset
            )
            page = data.get("recordings", [])
            if not page:
                break
            recordings.extend(page)
            offset += len(page)
            if offset >= data.get("count", offset):
                break

        work_by_recording = await self._fetch_work_ids(artist_mbid)
        for rec in recordings:
            if rec["id"] in work_by_recording:
                rec["relations"] = work_by_recording[rec["id"]]
        return recordings

    async def _fetch_work_ids(self, artist_mbid: str) -> dict[str, list]:
        """Map recording id -> its work relations via the browse endpoint."""
        work_by_recording: dict[str, list] = {}
        offset = 0
        while True:
            data = await self._client.browse_recordings_work_rels(
                artist_mbid, limit=self._page_size, offset=offset
            )
            page = data.get("recordings", [])
            if not page:
                break
            for rec in page:
                relations = rec.get("relations", [])
                if relations:
                    work_by_recording[rec["id"]] = relations
            offset += len(page)
            if offset >= data.get("recording-count", offset):
                break
        return work_by_recording

    async def _save_recording(
        self, session: AsyncSession, info: RecordingInfo
    ) -> None:
        """Upsert one recording as a song version, idempotent on its MB id."""
        fields = {
            "title": info.title,
            "first_release_date": info.first_release_date,
            "first_release_year": date_year(info.first_release_date),
            "first_release_source": _MB,
            "work_id": info.work_id,
        }
        existing = (
            await session.execute(
                select(SongExternal.song_id).where(
                    SongExternal.source == _MB, SongExternal.external_id == info.gid
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            await session.execute(
                update(Song).where(Song.id == existing).values(**fields)
            )
            song_id = existing
        else:
            song = Song(**fields)
            session.add(song)
            await session.flush()
            session.add(
                SongExternal(
                    song_id=song.id,
                    source=_MB,
                    external_id=info.gid,
                    url=f"https://musicbrainz.org/recording/{info.gid}",
                )
            )
            await session.flush()
            song_id = song.id
        if info.artist:
            await self._link_performer(session, song_id, info.artist)

    async def _link_performer(
        self, session: AsyncSession, song_id: int, name: str
    ) -> None:
        """Get-or-create the performing act and link it to the song."""
        performer = (
            await session.execute(select(Performer).where(Performer.name == name))
        ).scalar_one_or_none()
        if performer is None:
            performer = Performer(name=name)
            session.add(performer)
            await session.flush()
        await session.execute(
            sqlite_insert(SongPerformer)
            .values(song_id=song_id, performer_id=performer.id)
            .on_conflict_do_nothing()
        )

    async def _mark_artist(
        self, artist_mbid: str, status: str, error: str | None
    ) -> None:
        """Update an ``mb_artist`` crawl-state row's status/attempts."""
        async with self._session_factory() as session:
            values: dict = {"status": status, "last_error": error}
            if status == "failed":
                values["attempts"] = CrawlState.attempts + 1
            await session.execute(
                update(CrawlState).where(CrawlState.url == artist_mbid).values(**values)
            )
            await session.commit()


async def link_remakes(db_path: str = "data/music.db") -> int:
    """Flag remakes and link them to the earliest version of each work.

    For every ``work_id`` with more than one version, the row with the smallest
    ``first_release_year`` is the original; the rest get ``is_remake=True`` and
    ``original_song_id`` pointing at it. Songs missing a year sort last (treated
    as "not the original" when an earlier dated version exists).

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        The number of songs flagged as remakes.
    """
    engine = get_async_engine(db_path)
    factory = make_async_session(engine)
    flagged = 0
    async with factory() as session:
        work_ids = (
            await session.execute(
                select(Song.work_id)
                .where(Song.work_id.is_not(None))
                .group_by(Song.work_id)
                .having(func.count() > 1)
            )
        ).scalars().all()
        for work_id in work_ids:
            songs = list(
                (
                    await session.execute(
                        select(Song.id, Song.first_release_year).where(
                            Song.work_id == work_id
                        )
                    )
                ).all()
            )
            # Earliest dated version is the original; undated sort last.
            songs.sort(key=lambda s: (s[1] is None, s[1] if s[1] is not None else 0))
            original_id = songs[0][0]
            for song_id, _ in songs[1:]:
                await session.execute(
                    update(Song)
                    .where(Song.id == song_id)
                    .values(is_remake=True, original_song_id=original_id)
                )
                flagged += 1
        await session.commit()
    await engine.dispose()
    return flagged
