"""Property-based tests for the Discogs dating pure helpers.

Doctests cover concrete cases; these Hypothesis tests cover the invariants the
year picker must satisfy for *any* input, so a wrong release year is never
pinned on a song.
"""

from hypothesis import given
from hypothesis import strategies as st

from src.enrich.discogs import earliest_year, search_params
from src.enrich.match import normalize

_YEARS = st.integers(min_value=1900, max_value=2026)


@given(artist=st.text(min_size=1), title=st.text(min_size=1), token=st.text(min_size=1))
def test_search_params_roundtrips_inputs(artist: str, title: str, token: str) -> None:
    """The builder passes artist/title/token through and pins type=release."""
    p = search_params(artist, title, token)
    assert p["artist"] == artist
    assert p["track"] == title
    assert p["token"] == token
    assert p["type"] == "release"


@given(
    artist=st.text(min_size=1).filter(lambda s: normalize(s)),
    years=st.lists(_YEARS, min_size=1, max_size=6),
)
def test_earliest_year_picks_min_of_matching(artist: str, years: list[int]) -> None:
    """When every result credits our artist, the earliest year is returned."""
    results = [
        {"title": f"{artist} - Release {i}", "year": str(y)}
        for i, y in enumerate(years)
    ]
    assert earliest_year(artist, results) == min(years)


@given(artist=st.text(min_size=1).filter(lambda s: normalize(s)), year=_YEARS)
def test_earliest_year_rejects_unrelated_artist(artist: str, year: int) -> None:
    """A result whose title doesn't contain our artist is ignored."""
    title = "zzztotallyunrelatedactname - release"
    # Skip the rare case where the generated artist coincidentally appears in
    # the decoy title (e.g. a single-letter artist matching a title token).
    if normalize(artist) not in normalize(title):
        results = [{"title": title, "year": str(year)}]
        assert earliest_year(artist, results) is None


@given(artist=st.text(min_size=1).filter(lambda s: normalize(s)))
def test_earliest_year_ignores_out_of_range_years(artist: str) -> None:
    """Implausible years (e.g. 0, 3000) never produce a result."""
    results = [
        {"title": f"{artist} - A", "year": "0"},
        {"title": f"{artist} - B", "year": "3000"},
        {"title": f"{artist} - C", "year": ""},
    ]
    assert earliest_year(artist, results) is None
