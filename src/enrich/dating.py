"""Targeted MusicBrainz dating for the lyrics-bearing corpus.

The MusicBrainz *area enumeration* (:mod:`src.enrich.enrich`) produced dated song
versions, but for a largely international population that does not overlap the
Hungarian acts we actually have lyrics for — so it cannot date them. This module
takes the opposite, targeted approach: for each song that *has lyrics* but no
``first_release_year``, it searches MusicBrainz by that song's own
``artist + title``, validates the hit strictly, and copies the earliest known
release date back onto the song.

Lookups are deduplicated by ``song_key(performer, title)`` so duplicate acts /
titles cost one request, and the result is written to every row sharing the key.
Progress is resumable through ``enrich_state`` rows with ``source='mb_date'``
(the same mechanism the scraper and enumeration use): a song is marked ``done``
whether or not a date was found (a miss is cached so it is not retried), and
``failed`` with an incremented attempt count on transient errors.

Yield is partial by nature — many Hungarian songs are absent from MusicBrainz or
carry no release date — so misses are logged rather than hidden.
"""

from __future__ import annotations

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
from src.enrich.match import song_key
from src.enrich.musicbrainz import (
    MusicBrainzClient,
    date_year,
    parse_recording,
    pick_earliest_recording,
)

_SRC = "mb_date"
_MB = "musicbrainz"
_ZSZ = "zeneszoveg"
MAX_ATTEMPTS = 3


class _Candidate:
    """One undated lyrics song: its id, performing act, title and page year."""

    __slots__ = ("song_id", "artist", "title", "page_year")

    def __init__(self, song_id: int, artist: str, title: str, page_year: int | None):
        self.song_id = song_id
        self.artist = artist
        self.title = title
        self.page_year = page_year


class RecordingDater:
    """Dates lyrics-bearing songs via targeted MusicBrainz recording search."""

    def __init__(self, client: MusicBrainzClient, db_path: str = "data/music.db"):
        """Create a dater.

        Args:
            client: A configured :class:`~src.enrich.musicbrainz.MusicBrainzClient`.
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
        self, *, limit: int | None = None, page_year_fallback: bool = False
    ) -> dict[str, int]:
        """Date undated lyrics songs, one MusicBrainz lookup per artist|title.

        Args:
            limit: Optional cap on MusicBrainz lookups (distinct artist|title
                keys) this run, for chunked polite runs.
            page_year_fallback: When a lookup finds no date, fall back to the
                song's scraped page ``year`` (less reliable; recorded with
                ``first_release_source='zeneszoveg'``).

        Returns:
            Counts ``{"looked_up", "dated", "missed"}`` (songs, not keys, for
            ``dated``/``missed``).
        """
        await self.init_db()
        groups = await self._candidate_groups()
        counts = {"looked_up": 0, "dated": 0, "missed": 0}
        for members in groups.values():
            if limit is not None and counts["looked_up"] >= limit:
                break
            counts["looked_up"] += 1
            rep = members[0]
            try:
                info = await self._lookup(rep.artist, rep.title)
            except Exception as exc:  # noqa: BLE001 - record and retry later
                await self._mark(members, "failed", str(exc)[:500])
                continue
            if info is not None and info.first_release_date:
                await self._apply(
                    members,
                    year=date_year(info.first_release_date),
                    date=info.first_release_date,
                    work_id=info.work_id,
                    source=_MB,
                )
                counts["dated"] += len(members)
            elif page_year_fallback and rep.page_year is not None:
                await self._apply(
                    members, year=rep.page_year, date=None, work_id=None, source=_ZSZ
                )
                counts["dated"] += len(members)
            else:
                await self._mark(members, "done", None)
                counts["missed"] += len(members)
        return counts

    async def _candidate_groups(self) -> dict[str, list[_Candidate]]:
        """Group undated lyrics songs by ``song_key(performer, title)``.

        One representative performer per song (the first), so each song appears
        once; songs sharing an artist|title key are dated by a single lookup.
        """
        stmt = (
            select(Song.id, Performer.name, Song.title, Song.year)
            .join(SongPerformer, SongPerformer.song_id == Song.id)
            .join(Performer, Performer.id == SongPerformer.performer_id)
            .outerjoin(
                EnrichState,
                and_(EnrichState.song_id == Song.id, EnrichState.source == _SRC),
            )
            .where(
                Song.lyrics.is_not(None),
                Song.lyrics != "",
                # Only songs with no year yet. MB lookups are the slow (~1/s)
                # bottleneck, so we spend them filling gaps — not re-querying
                # songs Genius/page already dated (run genius-years first).
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
        groups: dict[str, list[_Candidate]] = {}
        seen: set[int] = set()
        async with self._session_factory() as session:
            for song_id, name, title, year in (await session.execute(stmt)).all():
                if song_id in seen:
                    continue
                seen.add(song_id)
                groups.setdefault(song_key(name, title), []).append(
                    _Candidate(song_id, name, title, year)
                )
        return groups

    async def _lookup(self, artist: str, title: str):
        """Search MusicBrainz and return the best-matching recording or ``None``."""
        data = await self._client.search_recordings_by_name(artist, title, limit=25)
        recordings = [parse_recording(rec) for rec in data.get("recordings", [])]
        return pick_earliest_recording(artist, title, recordings)

    async def _apply(
        self,
        members: list[_Candidate],
        *,
        year: int | None,
        date: str | None,
        work_id: str | None,
        source: str,
    ) -> None:
        """Write dating to every song in a key group and mark them done."""
        values: dict = {"first_release_year": year, "first_release_source": source}
        if date is not None:
            values["first_release_date"] = date
        if work_id is not None:
            values["work_id"] = work_id
        async with self._session_factory() as session:
            for member in members:
                await session.execute(
                    update(Song).where(Song.id == member.song_id).values(**values)
                )
                await self._upsert_state(session, member.song_id, "done", None)
            await session.commit()

    async def _mark(
        self, members: list[_Candidate], status: str, error: str | None
    ) -> None:
        """Mark a key group's enrich-state (done miss, or failed retry)."""
        async with self._session_factory() as session:
            for member in members:
                await self._upsert_state(session, member.song_id, status, error)
            await session.commit()

    @staticmethod
    async def _upsert_state(
        session: AsyncSession, song_id: int, status: str, error: str | None
    ) -> None:
        """Insert or update the ``mb_date`` enrich-state row for a song."""
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
