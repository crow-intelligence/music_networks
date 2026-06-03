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
import os

from sqlalchemy import func, select

from src.db import Song, SongExternal, get_engine, make_session
from src.enrich.dating import RecordingDater
from src.enrich.discogs import DiscogsClient, DiscogsDater
from src.enrich.enrich import Enumerator, link_remakes
from src.enrich.genius_kaggle import import_lyrics, import_years
from src.enrich.musicbrainz import HUNGARY_AREA_MBID, MusicBrainzClient
from src.enrich.wikidata import WikidataClient
from src.enrich.wikidata_enrich import WikidataEnricher


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

    gy = sub.add_parser(
        "genius-years", help="backfill song years from the Genius CSV year column"
    )
    gy.add_argument("--csv", required=True, help="path to song_lyrics.csv")

    dl = sub.add_parser(
        "date-lyrics", help="date lyrics songs via targeted MusicBrainz search"
    )
    dl.add_argument("--limit", type=int, default=None, help="cap MB lookups this run")
    dl.add_argument(
        "--page-year-fallback",
        action="store_true",
        help="on a MB miss, fall back to the scraped page year (less reliable)",
    )

    dd = sub.add_parser(
        "date-discogs", help="date the still-undated tail via Discogs (needs token)"
    )
    dd.add_argument(
        "--token",
        default=None,
        help="Discogs token (or set the DISCOGS_TOKEN env var)",
    )
    dd.add_argument("--limit", type=int, default=None, help="cap Discogs searches")

    wd = sub.add_parser(
        "wikidata", help="enrich artists/bands from Wikidata + Hungarian Wikipedia"
    )
    wd.add_argument("--max-pages", type=int, default=None, help="cap SPARQL pages")
    wd.add_argument(
        "--bio-limit", type=int, default=None, help="cap Wikipedia bios fetched"
    )
    wd.add_argument(
        "--no-extracts", action="store_true", help="skip fetching Wikipedia bios"
    )

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
            select(func.count())
            .select_from(Song)
            .where(Song.first_release_year.is_not(None))
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
    if args.command == "genius-years":
        counts = import_years(args.csv, args.db)
        print(
            f"Scanned {counts['rows']} Hungarian rows; "
            f"dated {counts['dated']} songs from Genius years."
        )
        return
    if args.command == "date-lyrics":
        dater = RecordingDater(MusicBrainzClient(delay=args.delay), db_path=args.db)
        try:
            counts = await dater.run(
                limit=args.limit, page_year_fallback=args.page_year_fallback
            )
            print(
                f"Looked up {counts['looked_up']} artist|title keys; "
                f"dated {counts['dated']} songs, {counts['missed']} without a date."
            )
        finally:
            await dater.close()
        return
    if args.command == "wikidata":
        enricher = WikidataEnricher(WikidataClient(delay=args.delay), db_path=args.db)
        try:
            counts = await enricher.import_facts(max_pages=args.max_pages)
            print(
                f"Fetched {counts['fetched']} Hungarian artists; linked "
                f"{counts['linked_performers']} performers, "
                f"{counts['linked_persons']} persons; "
                f"queued {counts['queued']} for bios."
            )
            if not args.no_extracts:
                bios = await enricher.fetch_bios(limit=args.bio_limit)
                print(f"Fetched {bios} Hungarian Wikipedia bios.")
        finally:
            await enricher.close()
        return
    if args.command == "date-discogs":
        token = args.token or os.environ.get("DISCOGS_TOKEN")
        if not token:
            raise SystemExit(
                "A Discogs token is required: pass --token or set DISCOGS_TOKEN "
                "(create one at discogs.com -> Settings -> Developers)."
            )
        # Discogs allows 60 auth'd req/min; keep at least ~1.2s between requests.
        client = DiscogsClient(token, delay=max(args.delay, 1.2))
        dater = DiscogsDater(client, db_path=args.db)
        try:
            counts = await dater.run(limit=args.limit)
            print(
                f"Looked up {counts['looked_up']} artist|title keys; "
                f"dated {counts['dated']} songs, {counts['missed']} without a date."
            )
        finally:
            await dater.close()
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
