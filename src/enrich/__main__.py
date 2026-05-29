"""Command-line entry point for corpus enrichment.

Examples:
    Seed Hungarian artists, enumerate their recordings into dated song
    versions, link remakes, and report::

        uv run python -m src.enrich seed --max-artists 50
        uv run python -m src.enrich dates --limit 50
        uv run python -m src.enrich qa

All commands are resumable: ``seed`` and ``dates`` only do outstanding work
(tracked in ``crawl_state`` rows with ``kind='mb_artist'``).
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import func, select

from src.db import Song, SongExternal, get_engine, make_session
from src.enrich.enrich import Enumerator, link_remakes
from src.enrich.genius_kaggle import import_lyrics
from src.enrich.musicbrainz import HUNGARY_AREA_MBID, MusicBrainzClient


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the enrichment CLI."""
    parser = argparse.ArgumentParser(prog="python -m src.enrich")
    parser.add_argument("--db", default="data/music.db", help="SQLite DB path")
    parser.add_argument(
        "--delay", type=float, default=1.0, help="seconds between MB requests"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("seed", help="seed Hungarian artists from MusicBrainz")
    s.add_argument("--area", default=HUNGARY_AREA_MBID, help="MusicBrainz area id")
    s.add_argument("--max-artists", type=int, default=None, help="cap artists seeded")

    d = sub.add_parser("dates", help="enumerate recordings, then link remakes")
    d.add_argument("--limit", type=int, default=None, help="cap artists processed")

    lyr = sub.add_parser("lyrics", help="import Hungarian Genius lyrics from CSV")
    lyr.add_argument("--csv", required=True, help="path to song_lyrics.csv")

    sub.add_parser("link-remakes", help="recompute is_remake / original_song_id")
    sub.add_parser("qa", help="print dating coverage report")
    return parser


def _qa(db_path: str) -> str:
    """Build a coverage / dating-quality report for the song corpus."""
    with make_session(get_engine(db_path))() as s:

        def count(stmt) -> int:
            return s.execute(stmt).scalar_one()

        total = count(select(func.count()).select_from(Song))
        dated = count(
            select(func.count()).select_from(Song).where(Song.first_release_year.is_not(None))
        )
        with_lyrics = count(
            select(func.count()).select_from(Song).where(Song.lyrics.is_not(None))
        )
        remakes = count(
            select(func.count()).select_from(Song).where(Song.is_remake.is_(True))
        )
        works = count(
            select(func.count(func.distinct(Song.work_id))).where(
                Song.work_id.is_not(None)
            )
        )
        mb = count(
            select(func.count())
            .select_from(SongExternal)
            .where(SongExternal.source == "musicbrainz")
        )
        # zeneszoveg year vs MusicBrainz first_release_year, where both exist.
        both = list(
            s.execute(
                select(Song.year, Song.first_release_year).where(
                    Song.year.is_not(None), Song.first_release_year.is_not(None)
                )
            ).all()
        )
        disagree = sum(1 for y, fr in both if y != fr)

    def pct(n: int) -> str:
        return f"{(100 * n / total):.1f}%" if total else "n/a"

    return "\n".join(
        [
            f"=== Corpus ({total} song versions) ===",
            f"  with first_release_year : {dated}  ({pct(dated)})",
            f"  with lyrics             : {with_lyrics}  ({pct(with_lyrics)})",
            f"  from MusicBrainz        : {mb}",
            f"  distinct works          : {works}",
            f"  remakes (linked)        : {remakes}",
            "=== Dating cross-check (year vs first_release_year) ===",
            f"  songs with both         : {len(both)}",
            f"  disagree                : {disagree}"
            + (f"  ({100 * disagree / len(both):.0f}%)" if both else ""),
        ]
    )


async def _run(args: argparse.Namespace) -> None:
    """Dispatch the parsed CLI command."""
    if args.command == "qa":
        print(_qa(args.db))
        return
    if args.command == "link-remakes":
        flagged = await link_remakes(args.db)
        print(f"Linked {flagged} remakes.")
        return
    if args.command == "lyrics":
        counts = import_lyrics(args.csv, args.db)
        print(
            f"Hungarian rows: {counts['rows']} "
            f"(matched {counts['matched']}, added {counts['added']})."
        )
        return

    client = MusicBrainzClient(delay=args.delay)
    enumerator = Enumerator(client, db_path=args.db)
    try:
        if args.command == "seed":
            n = await enumerator.seed_artists(args.area, max_artists=args.max_artists)
            print(f"Seeded {n} artists.")
        elif args.command == "dates":
            n = await enumerator.run(limit=args.limit)
            flagged = await link_remakes(args.db)
            print(f"Processed {n} artists; linked {flagged} remakes.")
    finally:
        await enumerator.close()


def main() -> None:
    """Parse arguments and run the requested command."""
    args = _build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
