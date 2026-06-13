"""Property-based tests for the emotion-aggregation pure helpers."""

from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from src.lyrics.emotion import (
    EMOTIONS,
    aggregate,
    chunk_sequence,
    dominant_emotion,
    mean_pool,
)

_ids = st.lists(st.integers(min_value=0, max_value=50_000), max_size=200)
_probvec = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=len(EMOTIONS),
    max_size=len(EMOTIONS),
)


@given(_ids, st.integers(min_value=1, max_value=64))
def test_chunk_sequence_partitions(ids, max_len):
    """Chunks concatenate back to the input and never exceed ``max_len``."""
    chunks = chunk_sequence(ids, max_len)
    assert chunks  # always at least one chunk (one model call per song)
    assert all(len(c) <= max_len for c in chunks)
    flat = [x for c in chunks for x in c]
    assert flat == ids


@given(
    st.lists(_probvec, min_size=1, max_size=8),
    st.lists(st.floats(min_value=0.0, max_value=100.0), min_size=1, max_size=8),
)
def test_mean_pool_within_bounds(vectors, weights):
    """Each pooled component lies within the per-dimension input range."""
    weights = weights[: len(vectors)] + [1.0] * max(0, len(vectors) - len(weights))
    pooled = mean_pool(vectors, weights)
    assert len(pooled) == len(EMOTIONS)
    for j in range(len(EMOTIONS)):
        col = [v[j] for v in vectors]
        assert min(col) - 1e-9 <= pooled[j] <= max(col) + 1e-9


@given(st.lists(_probvec, min_size=1, max_size=6))
def test_mean_pool_preserves_simplex(vectors):
    """Pooling probability vectors (sum 1) yields a vector that sums to ~1."""
    normed = []
    for v in vectors:
        s = sum(v)
        normed.append([x / s for x in v] if s > 0 else [1.0 / len(v)] * len(v))
    pooled = mean_pool(normed)
    assert math.isclose(sum(pooled), 1.0, abs_tol=1e-6)


@given(_probvec)
def test_dominant_is_argmax(probs):
    """The dominant label is the one at the highest-probability index."""
    label = dominant_emotion(probs)
    idx = EMOTIONS.index(label)
    assert probs[idx] == max(probs)


@given(
    st.lists(
        st.fixed_dictionaries({"decade": st.integers(1950, 2020), "probs": _probvec}),
        min_size=1,
        max_size=40,
    )
)
def test_aggregate_consistency(rows):
    """Decade counts partition the total; shares form a distribution."""
    agg = aggregate(rows)
    assert agg["overall"]["n"] == len(rows)
    assert sum(d["n"] for d in agg["by_decade"]) == len(rows)
    for scope in [agg["overall"], *agg["by_decade"]]:
        if scope["n"]:
            # Dominant-emotion shares are a probability distribution.
            assert math.isclose(sum(scope["share"].values()), 1.0, abs_tol=1e-6)
            # Every label present in both maps.
            assert set(scope["share"]) == set(EMOTIONS)
            assert set(scope["mean"]) == set(EMOTIONS)
