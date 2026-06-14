"""Property-based tests for the genre association cross-tabs."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from src.lyrics.associations import MIN_GENRE_SONGS, category_by_genre

_CATS = ["a", "b", "c"]
_GENRES = ["Rock", "Pop", "Jazz"]


@given(
    st.dictionaries(
        st.integers(min_value=1, max_value=400),
        st.tuples(st.sampled_from(_CATS), st.sampled_from(_GENRES)),
        max_size=400,
    )
)
def test_category_by_genre_rows_are_distributions(data):
    """Each charted genre's shares sum to 1 (labels all in the category set)."""
    labels = {k: v[0] for k, v in data.items()}
    genres = {k: v[1] for k, v in data.items()}
    out = category_by_genre(labels, genres, _CATS)
    # Only genres with enough songs are charted.
    from collections import Counter

    counts = Counter(genres.values())
    for g in out["genres"]:
        assert counts[g] >= MIN_GENRE_SONGS
    for row in out["rows"]:
        shares = row["shares"]
        assert set(shares) == set(_CATS)
        assert all(0.0 <= v <= 1.0 for v in shares.values())
        if row["n"]:
            assert abs(sum(shares.values()) - 1.0) < 1e-9


def test_category_by_genre_ignores_out_of_set_labels():
    """Labels not in the category list don't count toward a genre's rows."""
    labels = {i: ("x" if i % 2 else "a") for i in range(MIN_GENRE_SONGS * 2)}
    genres = {i: "Rock" for i in range(MIN_GENRE_SONGS * 2)}
    out = category_by_genre(labels, genres, ["a", "b"])
    row = out["rows"][0]
    assert row["shares"]["a"] == 1.0  # only the 'a'-labelled half counts
    assert row["n"] == MIN_GENRE_SONGS
