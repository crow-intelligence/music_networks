"""Property-based tests for the collaboration-network pure helpers."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.graphs.collab import (
    accumulate_edges,
    backbone,
    disparity_alpha,
    is_placeholder,
    pair_edges,
)


def test_is_placeholder_known_cases():
    """Placeholder credit strings are dropped; real names are kept."""
    for junk in ("Tradícionális", "ismeretlen, nem védett szerző", "népdal", "  "):
        assert is_placeholder(junk)
    for real in ("Bródy János", "Seress Rezső", "Ady Endre", "S. Nagy István"):
        assert not is_placeholder(real)


_people = st.lists(st.text(alphabet="abcde", min_size=1, max_size=3), max_size=8)


@given(_people)
def test_pair_edges_are_sorted_unique_pairs(creators):
    """Edges are ``(a, b)`` with ``a < b``, no self-loops, one per distinct pair."""
    edges = pair_edges(creators)
    n = len(set(creators))
    assert len(edges) == n * (n - 1) // 2
    for a, b in edges:
        assert a < b
    assert len(edges) == len(set(edges))


@given(st.lists(_people, max_size=10))
def test_accumulate_edges_counts_shared_songs(songs):
    """Each pair's weight equals the number of songs crediting both."""
    weights = accumulate_edges(songs)
    for (a, b), w in weights.items():
        expected = sum(1 for s in songs if a in set(s) and b in set(s))
        assert w == expected
        assert a < b


# Weighted edge lists -> a small graph for the disparity-filter properties.
_edges = st.lists(
    st.tuples(
        st.integers(min_value=0, max_value=6),
        st.integers(min_value=0, max_value=6),
        st.integers(min_value=1, max_value=50),
    ).filter(lambda e: e[0] != e[1]),
    max_size=20,
)


def _build(edges):
    import networkx as nx

    g = nx.Graph()
    for u, v, w in edges:
        g.add_edge(u, v, weight=w)
    return g


@given(_edges)
@settings(deadline=None)
def test_disparity_alpha_in_unit_range(edges):
    """Every alpha lies in [0, 1] and is keyed with the smaller node first."""
    alpha = disparity_alpha(_build(edges))
    for (u, v), a in alpha.items():
        assert u < v
        assert 0.0 <= a <= 1.0


@given(_edges)
@settings(deadline=None)
def test_backbone_is_a_monotone_subgraph(edges):
    """The backbone's edges are a subset of the graph's, growing with alpha."""
    g = _build(edges)
    full = {tuple(sorted(e)) for e in g.edges()}
    bb_low = backbone(g, alpha=0.01)
    bb_high = backbone(g, alpha=0.5)
    low = {tuple(sorted(e)) for e in bb_low.edges()}
    high = {tuple(sorted(e)) for e in bb_high.edges()}
    assert low <= full
    assert high <= full
    assert low <= high  # a laxer threshold keeps at least as many edges
    # No isolated nodes survive in the backbone.
    assert all(d > 0 for _, d in bb_low.degree())
