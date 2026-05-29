"""Command-line entry point for the scraper.

Examples:
    Seed band URLs (optionally capped), then scrape, then report::

        uv run python -m src.scraper discover --max-bands 1
        uv run python -m src.scraper crawl --limit 20
        uv run python -m src.scraper qa

All commands are resumable: re-running ``crawl`` only fetches what is still
pending or has failed below the retry ceiling.
"""

from __future__ import annotations

import argparse
import asyncio

from src.scraper import qa as qa_module
from src.scraper.crawl import DEFAULT_INITIALS, Crawler
from src.scraper.fetch import Fetcher
from src.scraper.proxies import ProxyPool


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the scraper CLI."""
    parser = argparse.ArgumentParser(prog="python -m src.scraper")
    parser.add_argument("--db", default="data/music.db", help="SQLite DB path")
    parser.add_argument(
        "--delay", type=float, default=1.5, help="seconds between requests"
    )
    parser.add_argument("--concurrency", type=int, default=4, help="concurrent fetches")
    parser.add_argument(
        "--proxies", action="store_true", help="use free proxy rotation"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("discover", help="seed band URLs from the artist index")
    d.add_argument("--initials", nargs="*", default=None, help="index letters to crawl")
    d.add_argument("--max-bands", type=int, default=None, help="cap bands seeded")

    c = sub.add_parser("crawl", help="scrape pending bands then songs")
    c.add_argument("--limit", type=int, default=None, help="cap rows per kind")

    sub.add_parser("qa", help="print coverage / success-rate report")
    return parser


async def _run(args: argparse.Namespace) -> None:
    """Dispatch the parsed CLI command."""
    if args.command == "qa":
        report = await qa_module.collect(args.db)
        print(report.render())
        return

    pool = ProxyPool() if args.proxies else None
    fetcher = Fetcher(delay=args.delay, concurrency=args.concurrency, proxies=pool)
    crawler = Crawler(fetcher, db_path=args.db, concurrency=args.concurrency)
    try:
        if args.command == "discover":
            initials = tuple(args.initials) if args.initials else DEFAULT_INITIALS
            n = await crawler.discover(initials, max_bands=args.max_bands)
            print(f"Seeded {n} band URLs.")
        elif args.command == "crawl":
            counts = await crawler.crawl(limit=args.limit)
            print(f"Processed {counts['bands']} bands and {counts['songs']} songs.")
    finally:
        await crawler.close()


def main() -> None:
    """Parse arguments and run the requested command."""
    args = _build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
