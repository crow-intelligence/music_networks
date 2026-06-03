"""Quality-assurance reporting over the scraped SQLite database.

Answers the two questions the user asked for: what fraction of discovered songs
were successfully scraped, and how complete is the data we have per song.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import (
    CrawlState,
    Song,
    SongAuthor,
    SongComposer,
    SongExternal,
    SongPerformer,
    get_async_engine,
    make_async_session,
)


@dataclass
class QAReport:
    """Aggregated crawl/coverage statistics.

    Attributes:
        discovered: Song URLs known to the crawler.
        done: Song URLs successfully scraped.
        failed: Song URLs that exhausted retries / errored.
        pending: Song URLs not yet attempted.
        failures_by_reason: Count of failed song URLs per error message.
        songs: Stored song rows.
        with_lyrics: Songs with non-empty lyrics.
        with_year: Songs with a release year.
        with_performer: Songs linked to >=1 performer.
        with_author: Songs linked to >=1 author.
        with_composer: Songs linked to >=1 composer.
        complete: Songs with lyrics + year + performer + author + composer.
        by_source: Per-``song_external`` source, ``{"songs": n, "with_lyrics": n}``.
            Answers "where did these songs come from?" (e.g. ``zeneszoveg`` is
            the scraper, ``musicbrainz`` is metadata-only enumeration).
        lyrics_with_year: Lyrics songs with a year (page ``year`` *or*
            ``first_release_year``).
        lyrics_with_author: Lyrics songs linked to >=1 author.
        lyrics_with_composer: Lyrics songs linked to >=1 composer.
        lyrics_complete: Lyrics songs with year + performer + author + composer.
        lyrics_year_by_source: Lyrics songs' year provenance, keyed by quality
            tier — ``musicbrainz`` (authoritative), ``genius`` (good), ``page``
            (scraped page year only, weakest).
    """

    discovered: int
    done: int
    failed: int
    pending: int
    failures_by_reason: dict[str, int]
    songs: int
    with_lyrics: int
    with_year: int
    with_performer: int
    with_author: int
    with_composer: int
    complete: int
    by_source: dict[str, dict[str, int]]
    lyrics_with_year: int
    lyrics_with_author: int
    lyrics_with_composer: int
    lyrics_complete: int
    lyrics_year_by_source: dict[str, int]

    @property
    def success_rate(self) -> float:
        """Fraction of discovered songs successfully scraped (0..1)."""
        return self.done / self.discovered if self.discovered else 0.0

    def _pct(self, n: int) -> str:
        """Format ``n`` as a percentage of stored songs."""
        return f"{(100 * n / self.songs):.1f}%" if self.songs else "n/a"

    def _lpct(self, n: int) -> str:
        """Format ``n`` as a percentage of lyrics-bearing songs."""
        return f"{(100 * n / self.with_lyrics):.1f}%" if self.with_lyrics else "n/a"

    def render(self) -> str:
        """Return a human-readable multi-line report.

        Returns:
            The formatted report text.
        """
        lines = [
            "=== Scrape progress (song URLs) ===",
            f"  discovered : {self.discovered}",
            f"  done       : {self.done}  ({100 * self.success_rate:.1f}% success)",
            f"  pending    : {self.pending}",
            f"  failed     : {self.failed}",
        ]
        if self.failures_by_reason:
            lines.append("  failures by reason:")
            for reason, count in sorted(
                self.failures_by_reason.items(), key=lambda kv: -kv[1]
            ):
                lines.append(f"    {count:>6}  {reason}")
        lines.append("")
        lines.append("=== Provenance (by source) ===")
        scraped = self.by_source.get("zeneszoveg", {}).get("songs", 0)
        lines.append(f"  scraped corpus: {scraped} of {self.songs} stored rows")
        for source, counts in sorted(
            self.by_source.items(), key=lambda kv: -kv[1]["songs"]
        ):
            songs = counts["songs"]
            with_lyrics = counts["with_lyrics"]
            pct = f"{(100 * with_lyrics / songs):.1f}%" if songs else "n/a"
            lines.append(
                f"  {source:<12} : {songs} songs  ({with_lyrics} w/ lyrics, {pct})"
            )
        lines += ["", f"=== Data completeness ({self.songs} stored songs) ==="]
        fields = [
            ("with lyrics", self.with_lyrics),
            ("with year", self.with_year),
            ("with performer", self.with_performer),
            ("with author", self.with_author),
            ("with composer", self.with_composer),
            ("fully complete", self.complete),
        ]
        for label, value in fields:
            lines.append(f"  {label:<14} : {value}  ({self._pct(value)})")
        # The meaningful corpus: songs we actually have lyrics for. Percentages
        # here are out of lyrics songs, not the (enumeration-inflated) total.
        lines += ["", f"=== Lyrics corpus ({self.with_lyrics} songs with lyrics) ==="]
        lyrics_fields = [
            ("with year", self.lyrics_with_year),
            ("with author", self.lyrics_with_author),
            ("with composer", self.lyrics_with_composer),
            ("fully complete", self.lyrics_complete),
        ]
        for label, value in lyrics_fields:
            lines.append(f"  {label:<14} : {value}  ({self._lpct(value)})")
        src = self.lyrics_year_by_source
        lines.append(
            "    year by source: "
            f"musicbrainz {src.get('musicbrainz', 0)}, "
            f"discogs {src.get('discogs', 0)}, "
            f"genius {src.get('genius', 0)}, "
            f"page {src.get('page', 0)}"
        )
        return "\n".join(lines)


async def _count(session: AsyncSession, stmt) -> int:
    """Run a scalar COUNT-style statement and return the integer result."""
    return (await session.execute(stmt)).scalar_one()


async def collect(db_path: str = "data/music.db") -> QAReport:
    """Gather QA statistics from the database.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        A populated :class:`QAReport`.
    """
    engine = get_async_engine(db_path)
    session_factory = make_async_session(engine)
    try:
        async with session_factory() as session:
            discovered = await _count(
                session,
                select(func.count())
                .select_from(CrawlState)
                .where(CrawlState.kind == "song"),
            )

            async def states(status: str) -> int:
                return await _count(
                    session,
                    select(func.count())
                    .select_from(CrawlState)
                    .where(CrawlState.kind == "song", CrawlState.status == status),
                )

            done = await states("done")
            failed = await states("failed")
            pending = await states("pending")

            reason_rows = (
                await session.execute(
                    select(CrawlState.last_error, func.count())
                    .where(CrawlState.kind == "song", CrawlState.status == "failed")
                    .group_by(CrawlState.last_error)
                )
            ).all()
            failures = {(r or "unknown"): c for r, c in reason_rows}

            songs = await _count(session, select(func.count()).select_from(Song))
            with_lyrics = await _count(
                session,
                select(func.count())
                .select_from(Song)
                .where(Song.lyrics.is_not(None), Song.lyrics != ""),
            )
            with_year = await _count(
                session,
                select(func.count()).select_from(Song).where(Song.year.is_not(None)),
            )

            with_performer = await _count(
                session, select(func.count(func.distinct(SongPerformer.song_id)))
            )
            with_author = await _count(
                session, select(func.count(func.distinct(SongAuthor.song_id)))
            )
            with_composer = await _count(
                session, select(func.count(func.distinct(SongComposer.song_id)))
            )

            source_rows = (
                await session.execute(
                    select(
                        SongExternal.source,
                        func.count(),
                        func.sum(
                            case(
                                (
                                    Song.lyrics.is_not(None) & (Song.lyrics != ""),
                                    1,
                                ),
                                else_=0,
                            )
                        ),
                    )
                    .select_from(SongExternal)
                    .join(Song, Song.id == SongExternal.song_id)
                    .group_by(SongExternal.source)
                )
            ).all()
            by_source = {
                source: {"songs": int(n), "with_lyrics": int(lyr or 0)}
                for source, n, lyr in source_rows
            }

            perf_ids = select(SongPerformer.song_id)
            auth_ids = select(SongAuthor.song_id)
            comp_ids = select(SongComposer.song_id)
            complete = await _count(
                session,
                select(func.count())
                .select_from(Song)
                .where(
                    Song.lyrics.is_not(None),
                    Song.lyrics != "",
                    Song.year.is_not(None),
                    Song.id.in_(perf_ids),
                    Song.id.in_(auth_ids),
                    Song.id.in_(comp_ids),
                ),
            )

            # Lyrics corpus: the songs we actually have lyrics for. "has a year"
            # accepts either the scraped page year or the authoritative
            # first_release_year (filled by the targeted MusicBrainz dating pass).
            has_lyrics = (Song.lyrics.is_not(None), Song.lyrics != "")
            has_year = or_(Song.year.is_not(None), Song.first_release_year.is_not(None))
            lyrics_with_year = await _count(
                session,
                select(func.count()).select_from(Song).where(*has_lyrics, has_year),
            )
            lyrics_with_author = await _count(
                session,
                select(func.count())
                .select_from(Song)
                .where(*has_lyrics, Song.id.in_(auth_ids)),
            )
            lyrics_with_composer = await _count(
                session,
                select(func.count())
                .select_from(Song)
                .where(*has_lyrics, Song.id.in_(comp_ids)),
            )
            lyrics_complete = await _count(
                session,
                select(func.count())
                .select_from(Song)
                .where(
                    *has_lyrics,
                    has_year,
                    Song.id.in_(perf_ids),
                    Song.id.in_(auth_ids),
                    Song.id.in_(comp_ids),
                ),
            )

            # Year provenance among lyrics songs, by quality tier. A song counts
            # once: authoritative first_release_source if set, else page-year.
            async def year_from(source: str) -> int:
                return await _count(
                    session,
                    select(func.count())
                    .select_from(Song)
                    .where(
                        *has_lyrics,
                        Song.first_release_year.is_not(None),
                        Song.first_release_source == source,
                    ),
                )

            page_only = await _count(
                session,
                select(func.count())
                .select_from(Song)
                .where(
                    *has_lyrics,
                    Song.year.is_not(None),
                    Song.first_release_year.is_(None),
                ),
            )
            lyrics_year_by_source = {
                "musicbrainz": await year_from("musicbrainz"),
                "discogs": await year_from("discogs"),
                "genius": await year_from("genius"),
                "page": page_only,
            }

        return QAReport(
            discovered=discovered,
            done=done,
            failed=failed,
            pending=pending,
            failures_by_reason=failures,
            songs=songs,
            with_lyrics=with_lyrics,
            with_year=with_year,
            with_performer=with_performer,
            with_author=with_author,
            with_composer=with_composer,
            complete=complete,
            by_source=by_source,
            lyrics_with_year=lyrics_with_year,
            lyrics_with_author=lyrics_with_author,
            lyrics_with_composer=lyrics_with_composer,
            lyrics_complete=lyrics_complete,
            lyrics_year_by_source=lyrics_year_by_source,
        )
    finally:
        await engine.dispose()
