"""Property-based tests for the lyrics analysis pure helpers.

Doctests cover concrete cases; these Hypothesis tests cover the invariants the
corpus + keyness helpers must satisfy for any input. No DB, model, or network.
"""

import math

from hypothesis import given
from hypothesis import strategies as st

from src.enrich.match import normalize
from src.lyrics.corpus import jaccard, keep_token, word_shingles
from src.lyrics.decade_keywords import log_likelihood, log_ratio
from src.lyrics.stats import song_decade


@given(
    fry=st.one_of(st.none(), st.integers(min_value=1000, max_value=9999)),
    year=st.one_of(st.none(), st.integers(min_value=1000, max_value=9999)),
)
def test_song_decade_is_multiple_of_ten_or_none(fry, year) -> None:
    """A song's decade is a multiple of ten, or None when undated."""
    out = song_decade(fry, year)
    assert out is None or out % 10 == 0


@given(text=st.text(), k=st.integers(min_value=1, max_value=4))
def test_word_shingles_count(text: str, k: int) -> None:
    """Shingle count equals max(0, words - k + 1) and each shingle has k words."""
    words = len(normalize(text).split())
    shingles = word_shingles(text, k=k)
    if words < k:
        assert shingles == set()
    else:
        assert all(len(s.split()) == k for s in shingles)
        assert len(shingles) <= words - k + 1  # <= because duplicates collapse


@given(
    a=st.sets(st.integers(), max_size=10),
    b=st.sets(st.integers(), max_size=10),
)
def test_jaccard_bounds_and_symmetry(a: set[int], b: set[int]) -> None:
    """Jaccard is in [0, 1], symmetric, and 1.0 for equal non-empty sets."""
    sa = {str(x) for x in a}
    sb = {str(x) for x in b}
    j = jaccard(sa, sb)
    assert 0.0 <= j <= 1.0
    assert j == jaccard(sb, sa)
    if sa and sa == sb:
        assert j == 1.0


@given(
    pos=st.sampled_from(["NOUN", "VERB", "ADJ", "ADV", "ADP", "DET", "PRON"]),
    is_stop=st.booleans(),
    is_alpha=st.booleans(),
)
def test_keep_token_requires_all_conditions(pos, is_stop, is_alpha) -> None:
    """A token is kept iff alphabetic, non-stop, and a content POS."""
    kept = keep_token(pos, is_stop, is_alpha)
    expected = is_alpha and not is_stop and pos in {"NOUN", "VERB", "ADJ", "ADV"}
    assert kept == expected


@given(
    a=st.integers(min_value=1, max_value=1000),
    b=st.integers(min_value=1, max_value=1000),
    c=st.integers(min_value=1000, max_value=100000),
    d=st.integers(min_value=1000, max_value=100000),
)
def test_log_likelihood_non_negative(a, b, c, d) -> None:
    """G² is always non-negative."""
    assert log_likelihood(a, b, c, d) >= -1e-9


@given(
    a=st.integers(min_value=1, max_value=1000),
    c=st.integers(min_value=1000, max_value=100000),
    d=st.integers(min_value=1000, max_value=100000),
)
def test_log_ratio_sign_matches_relative_frequency(a, c, d) -> None:
    """Equal relative frequencies give log-ratio ~0; higher target gives > 0."""
    # Same relative frequency in both corpora -> ratio 1 -> log2 == 0.
    b = max(1, round(a * d / c))
    assert abs(log_ratio(a, b, c, d) - math.log2((a / c) / (b / d))) < 1e-9
