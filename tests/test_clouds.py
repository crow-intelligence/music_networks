"""Property-based tests for the word-cloud weighting."""

from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from src.dashboard.clouds import cloud_weights

_terms = st.lists(
    st.fixed_dictionaries(
        {
            "term": st.text(alphabet="abcdef_", min_size=1, max_size=6),
            "log_likelihood": st.floats(
                min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False
            ),
        }
    ),
    max_size=120,
)


@given(_terms, st.integers(min_value=1, max_value=60))
def test_cloud_weights_invariants(terms, word_count):
    """Weights are sqrt of G², positive, capped, and deduped by display term."""
    w = cloud_weights(terms, word_count=word_count)

    # Never more entries than the cap.
    assert len(w) <= word_count
    # All weights strictly positive (zero-score terms dropped).
    assert all(v > 0 for v in w.values())

    # Mirror the contract: top-N terms, sqrt(G²) weight, underscores → spaces,
    # zero scores dropped, last write wins on a display-key collision.
    expected: dict[str, float] = {}
    for t in terms[:word_count]:
        score = t["log_likelihood"]
        if score > 0:
            expected[t["term"].replace("_", " ")] = math.sqrt(score)
    assert w.keys() == expected.keys()
    assert all(math.isclose(w[k], expected[k], rel_tol=1e-9) for k in w)


@given(_terms)
def test_no_underscores_in_keys(terms):
    """Display terms never contain the n-gram join character."""
    assert all("_" not in k for k in cloud_weights(terms))
