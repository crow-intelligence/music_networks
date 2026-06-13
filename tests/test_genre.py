"""Property-based tests for the Discogs genre pure helpers."""

from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from src.enrich.genre import (
    aggregate,
    group_candidates,
    load_token,
    pick_genre,
)

_GENRES = ["Rock", "Pop", "Hip Hop", "Electronic", "Folk", "Jazz"]
_STYLES = ["Synth-pop", "Hard Rock", "Trap", "Schlager", "Prog Rock"]


@given(st.text(max_size=40))
def test_load_token_strips_quotes(raw):
    """A loaded token never carries wrapping quotes or surrounding whitespace."""
    tok = load_token(f'"{raw}"')
    if tok is not None:
        assert not tok.startswith(('"', "'"))
        assert tok == tok.strip()


def test_load_token_empty_is_none():
    """Empty / quote-only input loads to ``None``."""
    assert load_token("") is None
    assert load_token('""') is None
    assert load_token(None) is None


_artist = st.sampled_from(["Omega", "LGT", "Tankcsapda", "Kispál"])
_result = st.builds(
    lambda a, extra, g, s: {
        "title": f"{a} - {extra}",
        "genre": g,
        "style": s,
    },
    st.sampled_from(["Omega", "LGT", "Other Band", "Foo"]),
    st.text(alphabet="abcde ", min_size=1, max_size=6),
    st.lists(st.sampled_from(_GENRES), max_size=3),
    st.lists(st.sampled_from(_STYLES), max_size=3),
)


@given(_artist, st.lists(_result, max_size=12))
def test_pick_genre_is_grounded(artist, results):
    """The chosen genre/styles come only from artist-matching results."""
    from src.enrich.match import normalize

    genre, styles = pick_genre(artist, results)
    matching = [r for r in results if normalize(artist) in normalize(r["title"])]
    present_genres = {g for r in matching for g in r["genre"]}
    present_styles = {s for r in matching for s in r["style"]}
    if genre is None:
        assert not present_genres  # only None when nothing matched/tagged
    else:
        assert genre in present_genres
        assert set(styles) <= present_styles
        assert len(styles) == len(set(styles))  # de-duplicated


@given(
    st.lists(
        st.tuples(
            st.sampled_from(["Omega", "omega", "LGT", "lgt"]),
            st.sampled_from(["Petróleum", "petróleum", "Mindenki"]),
            st.integers(1, 10_000),
        ),
        max_size=30,
    )
)
def test_group_candidates_partitions(rows):
    """Every input row lands in exactly one group; nothing is lost."""
    groups = group_candidates(rows)
    regrouped = [m for members in groups.values() for m in members]
    assert sorted(regrouped) == sorted(rows)


@given(
    st.lists(
        st.fixed_dictionaries(
            {
                "decade": st.integers(1950, 2020),
                "genre": st.one_of(st.none(), st.sampled_from(_GENRES)),
            }
        ),
        min_size=1,
        max_size=40,
    )
)
def test_aggregate_coverage_and_shares(rows):
    """Coverage = tagged/total and per-decade genre shares sum to ~1."""
    agg = aggregate(rows)
    for scope in [agg["overall"], *agg["by_decade"]]:
        assert scope["coverage"] == (
            scope["n"] / scope["total"] if scope["total"] else 0.0
        )
        if scope["n"]:
            assert math.isclose(sum(scope["share"].values()), 1.0, abs_tol=1e-6)
    # The genres list is exactly the set of tagged genres in the input.
    tagged = {r["genre"] for r in rows if r["genre"]}
    assert set(agg["genres"]) == tagged
