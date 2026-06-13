"""Property-based tests for the dashboard data-shaping helpers."""

from __future__ import annotations

import math
from collections import Counter

from hypothesis import given
from hypothesis import strategies as st

from src.dashboard.build import _recombine, frequency_block

_counts = st.dictionaries(
    st.integers(min_value=1950, max_value=2020),
    st.dictionaries(
        st.text(alphabet="abcdef_", min_size=1, max_size=6),
        st.integers(min_value=1, max_value=10_000),
        min_size=1,
        max_size=30,
    ).map(Counter),
    max_size=8,
)


@given(_counts, st.integers(min_value=1, max_value=40))
def test_frequency_block_ranks_by_count(counts, top):
    """Each decade keeps its ``top`` most-frequent terms, in descending order."""
    block = frequency_block(counts, top=top)
    assert [r["decade"] for r in block] == sorted(counts)
    for row in block:
        terms = row["terms"]
        assert len(terms) <= top
        freqs = [t["freq"] for t in terms]
        assert freqs == sorted(freqs, reverse=True)
        # Reported freqs match the source counter.
        src = counts[row["decade"]]
        assert all(src[t["term"]] == t["freq"] for t in terms)


_decade_summary = st.builds(
    lambda n, a, b: {
        "n": n,
        "mean": {"x": a, "y": b},
        "share": {"x": a, "y": b},
    },
    st.integers(min_value=0, max_value=500),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)


@given(st.lists(_decade_summary, min_size=1, max_size=8))
def test_recombine_is_weighted_average(decades):
    """The recombined overall mean is the n-weighted average of the decades."""
    out = _recombine(decades, ["mean", "share"], "n", "n")
    assert out["n"] == sum(d["n"] for d in decades)
    total_n = sum(d["n"] for d in decades) or 1
    for key in ("x", "y"):
        expected = sum(d["mean"][key] * d["n"] for d in decades) / total_n
        assert math.isclose(out["mean"][key], expected, rel_tol=1e-9, abs_tol=1e-12)
        # A weighted average never escapes the input range.
        vals = [d["mean"][key] for d in decades if d["n"] > 0]
        if vals:
            assert min(vals) - 1e-9 <= out["mean"][key] <= max(vals) + 1e-9


# Genre-style summaries: per-decade `share` is a distribution (sums to 1) over
# the `n` tagged songs, with a separate `total` (all songs) for coverage.
_genre_summary = st.builds(
    lambda total, frac, extra_n: {
        # n (tagged) <= total (all); share is a 2-genre distribution.
        "total": total + extra_n,
        "n": total,
        "share": {"Rock": frac, "Pop": 1.0 - frac},
    },
    st.integers(min_value=1, max_value=500),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    st.integers(min_value=0, max_value=200),
)


@given(st.lists(_genre_summary, min_size=1, max_size=8))
def test_recombine_genre_share_sums_to_one(decades):
    """Overall genre share is a distribution (sums to 1); coverage = n/total."""
    out = _recombine(decades, ["share"], "n", "total")
    # The Összes bar must fill 100% — the bug was dividing by `total` not `n`.
    assert math.isclose(sum(out["share"].values()), 1.0, abs_tol=1e-9)
    exp_cov = sum(d["n"] for d in decades) / sum(d["total"] for d in decades)
    assert math.isclose(out["coverage"], exp_cov, rel_tol=1e-9)
