"""Integration tests for the crawl writer: persistence, idempotency, resume.

These exercise :class:`~src.scraper.crawl.Crawler` end-to-end with a fake
fetcher (no network), against a throwaway SQLite file.
"""

import asyncio

from sqlalchemy import func, select

from src.db import (
    Band,
    BandMember,
    BandSong,
    Performer,
    Person,
    Song,
    SongAuthor,
    SongPerformer,
    get_engine,
    make_session,
)
from src.scraper import BASE_URL
from src.scraper.crawl import Crawler

_BAND_HTML = """
<html><body>
<a href="dalszoveg/5/teszt/a-dal-zeneszoveg.html">A dal</a>
<a href="dalszoveg/9/masok/related-zeneszoveg.html">related (other band)</a>
<a href="szemely/7/teszt-bela-adatlap.html" title="Teszt Béla (Teszt)">
  <img alt="Teszt Béla"></a>
<a href="szemely/uj.html">új személy beküldése</a>
</body></html>
"""

_INDEX_HTML = '<a href="egyuttes/1/teszt-dalszovegei.html">Teszt</a>'

_SONG_HTML = """
<html><head><title>Teszt : A dal dalszöveg, videó - Zeneszöveg.hu</title></head>
<body>
<div class="lyrics-plain-text">Első sor\nMásodik sor</div>
<div class="lyrics-header-text short"><table>
<tr><th>Előadó:</th><td>Teszt</td></tr>
<tr><th>Megjelenés:</th><td>1995</td></tr></table></div>
<div class="lyrics-header-text short"><table>
<tr><th>Szövegírók:</th><td>Katona László</td></tr>
<tr><th>Zeneszerzők:</th><td>Keressük a zeneszerzőt!</td></tr></table></div>
</body></html>
"""


class FakeFetcher:
    """Serves canned HTML by absolute URL; returns None for unknown URLs."""

    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages
        self.calls: list[str] = []

    async def get(self, url: str) -> str | None:
        self.calls.append(url)
        return self.pages.get(url)


def _pages() -> dict[str, str]:
    return {
        f"{BASE_URL}eloadok/a": _INDEX_HTML,
        f"{BASE_URL}egyuttes/1/teszt-dalszovegei.html": _BAND_HTML,
        f"{BASE_URL}dalszoveg/5/teszt/a-dal-zeneszoveg.html": _SONG_HTML,
    }


def _count(session, model) -> int:
    return session.execute(select(func.count()).select_from(model)).scalar_one()


async def _discover_and_crawl(crawler: Crawler) -> None:
    await crawler.discover(initials=("a",))
    await crawler.crawl()


def test_full_crawl_persists_everything(tmp_path) -> None:
    """A discover+crawl run stores the song, credits, member and links."""
    db = str(tmp_path / "t.db")
    fetcher = FakeFetcher(_pages())
    crawler = Crawler(fetcher, db_path=db, concurrency=2)
    asyncio.run(_discover_and_crawl(crawler))
    asyncio.run(crawler.close())

    with make_session(get_engine(db))() as s:
        song = s.execute(select(Song)).scalar_one()
        assert song.id == 5
        assert song.title == "a dal"
        assert song.lyrics == "Első sor\nMásodik sor"
        assert song.year == 1995
        # the related (other-band) song must NOT have been queued/stored
        assert _count(s, Song) == 1
        # credits
        assert s.execute(select(Performer.name)).scalar_one() == "teszt"
        assert _count(s, SongPerformer) == 1
        assert s.execute(select(SongAuthor)).scalars().all()  # author linked
        # member + band links
        person = s.execute(select(Person)).scalar_one()
        assert person.name == "Teszt Béla"
        assert _count(s, BandMember) == 1
        assert _count(s, Band) == 1
        assert _count(s, BandSong) == 1


def test_crawl_is_idempotent_and_resumable(tmp_path) -> None:
    """Re-running produces no duplicates and processes nothing the 2nd time."""
    db = str(tmp_path / "t.db")
    crawler = Crawler(FakeFetcher(_pages()), db_path=db, concurrency=2)
    asyncio.run(_discover_and_crawl(crawler))

    # Second pass: everything is already 'done', so no rows to process.
    second = asyncio.run(crawler.crawl())
    asyncio.run(crawler.close())
    assert second == {"bands": 0, "songs": 0}

    with make_session(get_engine(db))() as s:
        assert _count(s, Song) == 1
        assert _count(s, SongPerformer) == 1
        assert _count(s, BandMember) == 1


def test_failed_fetch_is_marked_not_done(tmp_path) -> None:
    """A song whose fetch fails is recorded as failed, with attempts bumped."""
    db = str(tmp_path / "t.db")
    # Serve index + band, but NOT the song page -> song fetch returns None.
    pages = _pages()
    del pages[f"{BASE_URL}dalszoveg/5/teszt/a-dal-zeneszoveg.html"]
    crawler = Crawler(FakeFetcher(pages), db_path=db, concurrency=2)
    asyncio.run(_discover_and_crawl(crawler))
    asyncio.run(crawler.close())

    from src.db import CrawlState

    with make_session(get_engine(db))() as s:
        row = s.execute(
            select(CrawlState).where(CrawlState.kind == "song")
        ).scalar_one()
        assert row.status == "failed"
        assert row.attempts == 1
        assert _count(s, Song) == 0
