"""Property-based tests for :mod:`src.enrich.wikidata` pure helpers.

Doctests cover concrete cases; these Hypothesis tests cover the invariants the
SPARQL-row parsers must satisfy for *any* input so enrichment stays stable.
"""

import urllib.parse

from hypothesis import given
from hypothesis import strategies as st

from src.enrich.wikidata import (
    HUWIKI_PREFIX,
    ArtistInfo,
    merge_artists,
    parse_artist_binding,
    qid_from_uri,
    title_from_huwiki_url,
    trim_date,
)

_QIDS = st.from_regex(r"Q[1-9][0-9]{0,8}", fullmatch=True)


@given(qid=_QIDS)
def test_qid_from_full_uri_round_trips(qid: str) -> None:
    """A full entity URI parses back to its bare id."""
    assert qid_from_uri(f"http://www.wikidata.org/entity/{qid}") == qid


@given(value=st.text())
def test_trim_date_never_exceeds_ten_chars(value: str) -> None:
    """A trimmed date is at most ``YYYY-MM-DD`` (10 chars), or ``None``."""
    out = trim_date(value)
    assert out is None or len(out) <= 10


def test_trim_date_empty_is_none() -> None:
    """Empty / falsy inputs trim to ``None``."""
    assert trim_date("") is None
    assert trim_date(None) is None


@given(
    title=st.text(
        alphabet=st.characters(min_codepoint=32, max_codepoint=0x017F),
        min_size=1,
    ).filter(lambda s: not (set(s) & set("/_")) and s.strip() == s)
)
def test_huwiki_title_round_trips_through_url(title: str) -> None:
    """Encoding a title into a huwiki URL and parsing it back is identity.

    Mirrors how Wikidata emits sitelinks: spaces become underscores and the
    rest is percent-encoded. Literal underscores are excluded — Wikipedia
    treats ``_`` and space as equivalent, so they cannot round-trip distinctly.
    """
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    url = f"{HUWIKI_PREFIX}wiki/{encoded}"
    assert title_from_huwiki_url(url) == title


def test_huwiki_title_none_passthrough() -> None:
    """A missing URL yields a missing title."""
    assert title_from_huwiki_url(None) is None


@given(qid=_QIDS, name=st.text(min_size=1), genre=st.text(min_size=1))
def test_parse_binding_keeps_single_genre(qid: str, name: str, genre: str) -> None:
    """A binding with one genre cell yields exactly that genre."""
    binding = {
        "item": {"value": f"http://www.wikidata.org/entity/{qid}"},
        "name": {"value": name},
        "genre": {"value": genre},
    }
    info = parse_artist_binding(binding)
    assert info.qid == qid
    assert info.genres == [genre]


@given(
    qid=_QIDS,
    genres=st.lists(st.text(min_size=1), min_size=0, max_size=5),
)
def test_merge_collapses_to_one_entry_per_qid(qid: str, genres: list[str]) -> None:
    """Multiple rows for one entity merge into a single, genre-unioned entry."""
    rows = [ArtistInfo(qid=qid, name="X", genres=[g]) for g in genres]
    merged = merge_artists(rows)
    assert len(merged) <= 1
    if merged:
        # Genres are de-duplicated and sorted.
        assert merged[0].genres == sorted(set(genres))


@given(
    qids=st.lists(_QIDS, min_size=1, max_size=6, unique=True),
)
def test_merge_preserves_distinct_entities(qids: list[str]) -> None:
    """Distinct entities are never collapsed together."""
    rows = [ArtistInfo(qid=q, name="X") for q in qids]
    merged = merge_artists(rows)
    assert {m.qid for m in merged} == set(qids)
    assert len(merged) == len(qids)
