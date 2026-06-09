"""Per-decade word-association graphs for the dashboard's §8 ego networks.

Builds a weighted **co-occurrence graph** per decade from the Phase-0 corpus
tokens with `kenon` (``build_cooccurrence_graph`` → disparity-filter
``extract_backbone``), then exports a compact per-decade adjacency payload the
static dashboard renders as fixed-radial *ego networks* (a seed word at the
centre, its strongest neighbours on a ring, edges weighted by association).

The same payloads are intended to feed the separate, force-directed *semantic
explorer* page later — so the numbers are produced once, here.

Association metric: relative skip-gram co-occurrence frequency within a fixed
window (kenon's ``weight`` edge attribute), pruned to the statistically salient
backbone. Node weight is the lemma's raw frequency in the decade.

Outputs (under ``data/processed/networks``):

* ``decade_<d>.json`` — ``{"decade", "nodes": {word: freq}, "adj":
  {word: [[neighbour, weight], ...]}, "seeds": [...]}``,
* ``index.json`` — ``{"decades": [...], "window", "metric", "seeds"}``.

Import-safe: only :func:`main` touches the corpus / filesystem.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import networkx as nx

DEFAULT_CORPUS_DIR = Path("data/processed/corpus")
DEFAULT_OUT_DIR = Path("data/processed/networks")

# Editorially chosen default seed words (mirrors the diachronic target words);
# only those actually present in a decade's backbone are offered as defaults.
DEFAULT_SEEDS = [
    "szerelem",
    "szív",
    "pénz",
    "haza",
    "szabadság",
    "isten",
    "halál",
    "város",
    "éjszaka",
    "lány",
    "élet",
    "háború",
]

# Co-occurrence window (tokens are already lemmatised content words, so a small
# window captures phrase-level association).
WINDOW = 5


def graph_to_payload(
    graph: nx.Graph,
    freqs: dict[str, int],
    *,
    top_nodes: int = 240,
    top_neighbors: int = 12,
    seeds: list[str] | None = None,
) -> dict[str, Any]:
    """Reduce a co-occurrence graph to a compact, render-ready payload.

    The ``top_nodes`` most frequent words become the selectable seeds; for each,
    its ``top_neighbors`` heaviest edges in the *full* graph are kept (a hub's
    strongest co-occurrences are the meaningful ones — unlike a disparity-filter
    backbone, which surfaces rare-word noise around hubs). Neighbour nodes are
    added to ``nodes`` so the payload is self-contained for sizing.

    Args:
        graph: Weighted graph (networkx); edge weights under the ``weight`` key.
        freqs: Raw lemma frequency per node (used for node sizing + ranking).
        top_nodes: Max seed nodes to keep (most frequent first).
        top_neighbors: Max neighbours kept per seed (heaviest edges first).
        seeds: Candidate default seeds; only those kept are returned.

    Returns:
        ``{"nodes": {word: freq}, "adj": {word: [[nb, weight], ...]}, "seeds":
        [...]}``; ``adj`` has an entry per kept seed, neighbours sorted by
        descending weight.

    Examples:
        >>> import networkx as nx
        >>> g = nx.Graph()
        >>> g.add_edge("szív", "szerelem", weight=0.9)
        >>> g.add_edge("szív", "vér", weight=0.2)
        >>> p = graph_to_payload(g, {"szív": 10, "szerelem": 5, "vér": 1},
        ...                      top_nodes=10, top_neighbors=1, seeds=["szív"])
        >>> p["adj"]["szív"]          # heaviest edge only (top_neighbors=1)
        [['szerelem', 0.9]]
        >>> p["seeds"]
        ['szív']
        >>> "szív" in p["nodes"] and "szerelem" in p["nodes"]
        True
    """
    ranked = sorted(graph.nodes(), key=lambda n: freqs.get(n, 0), reverse=True)
    kept = ranked[:top_nodes]
    nodes: dict[str, int] = {}
    adj: dict[str, list[list[Any]]] = {}
    for n in kept:
        edges = [
            (nb, float(graph[n][nb].get("weight", 0.0))) for nb in graph.neighbors(n)
        ]
        edges.sort(key=lambda e: e[1], reverse=True)
        top = edges[:top_neighbors]
        adj[n] = [[nb, round(w, 8)] for nb, w in top]
        nodes[n] = int(freqs.get(n, 0))
        for nb, _ in top:
            nodes.setdefault(nb, int(freqs.get(nb, 0)))
    keptset = set(kept)
    present = [s for s in (seeds or []) if s in keptset]
    return {"nodes": nodes, "adj": adj, "seeds": present}


def build_decade_graph(
    tokens: list[str],
    *,
    window: int = WINDOW,
    backbone: bool = False,
    min_alpha_ptile: float = 0.5,
    min_degree: int = 2,
) -> nx.Graph:
    """Build the weighted co-occurrence graph for one decade's tokens.

    Args:
        tokens: Flat lemma stream for the decade (Phase-0 ``tokens`` concatenated).
        window: Skip-gram co-occurrence window.
        backbone: If true, apply kenon's disparity-filter backbone. Off by
            default: ego previews center on frequent hubs, whose strongest raw
            co-occurrences are the meaningful neighbours (the backbone instead
            surfaces rare-word noise around hubs). The backbone is for the
            global force-directed explorer view.
        min_alpha_ptile: Disparity-filter threshold (only if ``backbone``).
        min_degree: Minimum node degree kept (only if ``backbone``).

    Returns:
        The weighted graph (a ``networkx.Graph``).
    """
    from kenon import build_cooccurrence_graph, extract_backbone

    graph = build_cooccurrence_graph(tokens, window=window)
    if backbone:
        graph = extract_backbone(
            graph, min_alpha_ptile=min_alpha_ptile, min_degree=min_degree
        )
    return graph


def build_networks(
    corpus_dir: Path | str = DEFAULT_CORPUS_DIR,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    *,
    window: int = WINDOW,
    top_nodes: int = 240,
    top_neighbors: int = 12,
    seeds: list[str] | None = None,
) -> list[int]:
    """Build and write per-decade word-graph payloads from the corpus.

    Args:
        corpus_dir: Phase-0 corpus directory (``decade_*.jsonl``).
        out_dir: Output directory for the JSON payloads + ``index.json``.
        window: Co-occurrence window.
        top_nodes: Max nodes kept per decade payload.
        top_neighbors: Max neighbours kept per node.
        seeds: Default seed words (falls back to :data:`DEFAULT_SEEDS`).

    Returns:
        The list of decades that produced a non-empty graph.
    """
    from src.lyrics.corpus import load_corpus

    corpus_dir, out_dir = Path(corpus_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = seeds or DEFAULT_SEEDS

    docs = load_corpus(corpus_dir)
    by_decade: dict[int, list[str]] = {}
    for doc in docs:
        by_decade.setdefault(doc.decade, []).extend(doc.tokens)

    written: list[int] = []
    for decade in sorted(by_decade):
        tokens = by_decade[decade]
        graph = build_decade_graph(tokens, window=window)
        if graph.number_of_nodes() == 0:
            continue
        freqs = Counter(tokens)
        payload = graph_to_payload(
            graph,
            freqs,
            top_nodes=top_nodes,
            top_neighbors=top_neighbors,
            seeds=seeds,
        )
        payload["decade"] = decade
        (out_dir / f"decade_{decade}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        written.append(decade)
        print(
            f"  {decade}s : {len(payload['nodes'])} nodes, "
            f"{sum(len(v) for v in payload['adj'].values())} edges"
        )

    metric = "relatív együtt-előfordulás (skip-gram ablak)"
    (out_dir / "index.json").write_text(
        json.dumps(
            {
                "decades": written,
                "window": window,
                "metric": metric,
                "seeds": seeds,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return written


def main() -> None:
    """Build the per-decade word graphs from the cached corpus."""
    import argparse

    ap = argparse.ArgumentParser(description="Build per-decade word graphs (kenon).")
    ap.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--window", type=int, default=WINDOW)
    args = ap.parse_args()

    written = build_networks(args.corpus_dir, args.out_dir, window=args.window)
    if not written:
        raise SystemExit(
            f"No corpus at {args.corpus_dir}; run `python -m src.lyrics.corpus`."
        )
    print(f"Wrote graphs for {len(written)} decades -> {args.out_dir}")


if __name__ == "__main__":
    main()
