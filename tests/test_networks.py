"""Property-based tests for the per-decade word-graph payloads."""

from __future__ import annotations

import networkx as nx
from hypothesis import given
from hypothesis import strategies as st

from src.lyrics.networks import graph_to_payload

# Small alphabet of node names keeps generated graphs dense enough to exercise
# the neighbour-capping logic.
_words = st.text(alphabet="abcdef", min_size=1, max_size=3)


@st.composite
def weighted_graphs(draw: st.DrawFn) -> tuple[nx.Graph, dict[str, int]]:
    """Generate a weighted graph plus a frequency map covering its nodes."""
    edges = draw(
        st.lists(
            st.tuples(_words, _words, st.floats(0.001, 1.0)),
            min_size=1,
            max_size=30,
        )
    )
    g = nx.Graph()
    for a, b, w in edges:
        if a != b:
            g.add_edge(a, b, weight=w)
    freqs = {n: draw(st.integers(0, 1000)) for n in g.nodes()}
    return g, freqs


@given(weighted_graphs(), st.integers(1, 8), st.integers(1, 6))
def test_payload_invariants(data, top_nodes, top_neighbors):
    """Payload respects caps, stays self-contained, and ranks by weight."""
    graph, freqs = data
    seeds = list(freqs)[:2]
    p = graph_to_payload(
        graph, freqs, top_nodes=top_nodes, top_neighbors=top_neighbors, seeds=seeds
    )

    # At most `top_nodes` seeds carry an adjacency entry.
    assert len(p["adj"]) <= min(top_nodes, graph.number_of_nodes())

    for seed, nbrs in p["adj"].items():
        # Per-seed neighbour cap honoured.
        assert len(nbrs) <= top_neighbors
        # Neighbours sorted by descending weight.
        weights = [w for _, w in nbrs]
        assert weights == sorted(weights, reverse=True)
        for nb, _w in nbrs:
            # Every referenced node is present (payload is self-contained)...
            assert nb in p["nodes"]
            # ...and the edge actually exists in the source graph.
            assert graph.has_edge(seed, nb)

    # Seeds returned are exactly the requested seeds that survived the node cap.
    assert set(p["seeds"]) <= set(seeds)
    for s in p["seeds"]:
        assert s in p["adj"]


@given(weighted_graphs())
def test_node_freqs_are_ints(data):
    """All node frequencies are non-negative integers."""
    graph, freqs = data
    p = graph_to_payload(graph, freqs, seeds=[])
    assert all(isinstance(v, int) and v >= 0 for v in p["nodes"].values())
