"""Property-based tests for lexical-diversity metrics."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from src.lyrics.diversity import lexical_diversity_by_decade, mattr

_tokens = st.lists(st.sampled_from(["a", "b", "c", "d", "e"]), max_size=200)


@given(_tokens, st.integers(min_value=1, max_value=50))
def test_mattr_in_unit_range(tokens, window):
    """MATTR is always a ratio in [0, 1]."""
    assert 0.0 <= mattr(tokens, window=window) <= 1.0


@given(st.lists(st.sampled_from(["a", "b", "c"]), min_size=1, max_size=200))
def test_mattr_short_is_plain_ttr(tokens):
    """For sequences no longer than the window, MATTR equals plain TTR."""
    window = len(tokens) + 5
    assert mattr(tokens, window=window) == len(set(tokens)) / len(tokens)


@given(st.integers(min_value=1, max_value=40), st.integers(min_value=1, max_value=30))
def test_all_distinct_is_one(n, window):
    """A sequence of all-distinct tokens scores 1.0 at any window."""
    tokens = [f"w{i}" for i in range(n)]
    assert mattr(tokens, window=window) == 1.0


def test_repeated_single_token_is_low():
    """A single repeated token yields the minimal non-zero diversity."""
    assert mattr(["a"] * 100, window=10) == 0.1


class _Doc:
    def __init__(self, decade: int, tokens: list[str]) -> None:
        self.decade = decade
        self.tokens = tokens


@given(
    st.lists(
        st.tuples(st.sampled_from([1960, 1990, 2020]), _tokens),
        min_size=1,
        max_size=10,
    )
)
def test_by_decade_covers_all_decades(pairs):
    """Every decade present in the docs appears in the output, in [0, 1]."""
    docs = [_Doc(d, t) for d, t in pairs]
    out = lexical_diversity_by_decade(docs, window=10)
    assert set(out) == {d for d, _ in pairs}
    assert all(0.0 <= v <= 1.0 for v in out.values())
