"""Property-based tests for :mod:`src.scraper.parse`.

The doctests in ``parse.py`` pin concrete behaviour; these Hypothesis tests
assert the invariants that must hold for arbitrary inputs (robustness against
the messy real-world HTML the scraper will encounter).
"""

from hypothesis import given
from hypothesis import strategies as st

from src.scraper.parse import (
    SongRecord,
    extract_band_id,
    extract_song_id,
    normalize_name,
    parse_band_page,
    parse_song_page,
    parse_title,
    split_aka,
)

# Text without control chars / surrogates that break HTML or comparisons.
_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cc", "Cs")),
    max_size=80,
)


@given(performer=_text, title=_text)
def test_parse_title_never_raises_and_normalizes(performer: str, title: str) -> None:
    """A well-formed ``"performer : title …"`` always yields normalized parts."""
    raw = f"{performer} : {title} dalszöveg, videó - Zeneszöveg.hu"
    perf, tit = parse_title(raw)
    # Output is always lower-cased and whitespace-collapsed (idempotent).
    assert perf == normalize_name(perf)
    assert tit == normalize_name(tit)


@given(raw=_text)
def test_split_aka_returns_two_parts(raw: str) -> None:
    """``split_aka`` always returns a normalized name plus an aka string."""
    name, aka = split_aka(raw)
    assert name == normalize_name(name)
    assert isinstance(aka, str)


@given(song_id=st.integers(min_value=1, max_value=10_000_000))
def test_song_id_roundtrips(song_id: int) -> None:
    """A constructed song URL parses back to its id (and isn't a band id)."""
    url = f"dalszoveg/{song_id}/akela/some-song-zeneszoveg.html"
    assert extract_song_id(url) == song_id
    assert extract_band_id(url) is None


@given(band_id=st.integers(min_value=1, max_value=10_000_000))
def test_band_id_roundtrips(band_id: int) -> None:
    """A constructed band URL parses back to its id (and isn't a song id)."""
    url = f"egyuttes/{band_id}/akela-dalszovegei.html"
    assert extract_band_id(url) == band_id
    assert extract_song_id(url) is None


@given(garbage=_text)
def test_parse_song_page_never_raises(garbage: str) -> None:
    """Parsing arbitrary/garbage HTML yields a SongRecord, never an exception."""
    rec = parse_song_page(garbage)
    assert isinstance(rec, SongRecord)
    assert isinstance(rec.performers, list)


@given(garbage=_text, slug=st.text(alphabet="abcdefghij", min_size=1, max_size=8))
def test_parse_band_page_never_raises(garbage: str, slug: str) -> None:
    """Parsing arbitrary HTML yields de-duplicated, well-typed link lists."""
    page = parse_band_page(garbage, slug)
    assert len(page.song_urls) == len(set(page.song_urls))
    assert all(isinstance(m, tuple) and len(m) == 2 for m in page.members)
