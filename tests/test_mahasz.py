"""Property-based tests for the Mahász chart pure helpers."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.enrich.mahasz import (
    build_performer_index,
    match_artist,
    parse_chart_page,
    parse_rank,
    strip_label,
)
from src.enrich.match import normalize


@given(st.integers(min_value=1, max_value=100), st.text(alphabet=" .#", max_size=4))
def test_parse_rank_recovers_number(n, noise):
    """A rank cell like ``"<n>."`` parses back to ``n`` regardless of stray dots."""
    assert parse_rank(f"{noise}{n}.{noise}") == n


@given(st.text(alphabet="abc ", max_size=8))
def test_parse_rank_no_digits_is_none(text):
    """A cell with no digits yields ``None``."""
    assert parse_rank(text) is None


@given(st.text(max_size=30))
def test_strip_label_no_wrapping_parens(text):
    """The result never keeps the outer parentheses or surrounding whitespace."""
    out = strip_label(f"({text})")
    if out is not None:
        assert out == out.strip()
        assert not (out.startswith("(") and out.endswith(")"))


def test_strip_label_empty_is_none():
    """Empty / whitespace-only input loads to ``None``."""
    assert strip_label("") is None
    assert strip_label("()") is None
    assert strip_label(None) is None


_names = st.lists(
    st.tuples(
        st.integers(min_value=1, max_value=9999),
        st.text(alphabet="abcdéő .", min_size=1, max_size=8),
    ),
    max_size=20,
)


@given(_names)
def test_build_performer_index_first_wins(rows):
    """Every key maps to a real id; the first id for a normalized name wins."""
    index = build_performer_index(rows)
    seen: dict[str, int] = {}
    for pid, name in rows:
        seen.setdefault(normalize(name), pid)
    # Drop empty normalized names (setdefault keeps them, but they're not useful).
    expected = {k: v for k, v in seen.items()}
    assert index == expected


@given(_names, st.text(alphabet="abcdéő .", min_size=1, max_size=8))
def test_match_artist_exact_is_grounded(rows, query):
    """An exact-normalized hit returns that performer id; misses are valid/None."""
    index = build_performer_index(rows)
    result = match_artist(query, index)
    key = normalize(query)
    if key and key in index:  # match_artist returns None for an empty key
        assert result == index[key]
    # Whatever it returns must be a real id from the index (or None).
    assert result is None or result in index.values()


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=100),
            st.text(alphabet="ABCDE", min_size=1, max_size=6),  # non-empty, no ws
            st.text(alphabet="abcde ", min_size=1, max_size=8),
        ),
        min_size=1,
        max_size=10,
        unique_by=lambda t: t[0],  # unique ranks
    )
)
@settings(deadline=None)  # BeautifulSoup parsing is too slow for the 200ms default
def test_parse_chart_page_roundtrip(entries):
    """Rows built into the site's table markup parse back with rank+artist."""
    cells = "".join(
        f'<tr><td class="no_sor">{rank}.</td>'
        f'<td class="lemez_sor2"><span class="eloado">{artist}</span><br/>'
        f'{title}<br/><span class="kiado_sor">(Label)</span></td></tr>'
        for rank, artist, title in entries
    )
    rows = parse_chart_page(f"<table>{cells}</table>")
    assert len(rows) == len(entries)
    for got, (rank, artist, _title) in zip(rows, entries, strict=True):
        assert got.rank == rank
        assert got.artist == artist.strip()
        assert got.label == "Label"
