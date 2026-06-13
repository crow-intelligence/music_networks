"""Songwriter collaboration networks from the lyrics database.

Builds the project's namesake network: **who creates songs together**. A node is
a *person* (a lyricist or composer, unified across the two roles by normalized
name); an edge joins two people **co-credited on the same song**, weighted by how
many songs they share. The network is built both **aggregate** (all credited
songs) and **diachronically** (one graph per release decade), and each graph is
reduced to its **disparity-filter backbone** (Serrano, Boguñá & Vespignani 2009)
— the multiscale skeleton of statistically significant collaborations.

The pure graph helpers (edge enumeration, weight accumulation, the disparity
filter) are deterministic and doctested; the database loaders and the networkx
build/export live behind functions so the module stays import-safe (no work, and
no heavy imports, at import time).

Outputs (``data/processed/collab/``): ``<slice>_full.graphml`` +
``<slice>_backbone.graphml`` per slice, and a ``summary.json`` with per-slice
sizes and top collaborators.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.enrich.match import normalize
from src.lyrics.stats import AUTHORITATIVE_SOURCES, song_decade

if TYPE_CHECKING:
    import networkx as nx

DEFAULT_COLLAB_DIR = Path("data/processed/collab")

# Credit strings that are placeholders, not people — excluded from the network.
# Matched on the normalized form; substrings catch the "unknown"/"traditional"
# families, exact entries catch specific portal placeholders.
_PLACEHOLDER_EXACT = frozenset(
    normalize(x)
    for x in (
        "tradícionális",
        "ismeretlen, nem védett szerző",
        "népdal átdolgozás",
        "népdal",
        "népzene",
        "néphagyomány",
        "ismeretlen szerző",
        "ismeretlen",
    )
)
_PLACEHOLDER_SUBSTR = ("ismeretlen", "tradicional", "traditional")

# Disparity-filter significance level: an edge is kept in the backbone when it is
# significant (alpha below this) from at least one of its endpoints.
DEFAULT_ALPHA = 0.05


def is_placeholder(name: str) -> bool:
    """Whether a credit string is a non-person placeholder to drop.

    Args:
        name: An author/composer credit string.

    Returns:
        ``True`` for empty/unknown/traditional-style placeholders.

    Examples:
        >>> is_placeholder("Tradícionális")
        True
        >>> is_placeholder("ismeretlen, nem védett szerző")
        True
        >>> is_placeholder("Bródy János")
        False
        >>> is_placeholder("   ")
        True
    """
    key = normalize(name)
    if not key or key in _PLACEHOLDER_EXACT:
        return True
    return any(sub in key for sub in _PLACEHOLDER_SUBSTR)


def pair_edges(creators: Iterable[str]) -> list[tuple[str, str]]:
    """Enumerate the unordered co-credit pairs among a song's creators.

    Args:
        creators: The creator keys credited on one song (duplicates ignored).

    Returns:
        Sorted ``(a, b)`` pairs with ``a < b``; empty for fewer than two distinct
        creators.

    Examples:
        >>> pair_edges(["b", "a", "a"])
        [('a', 'b')]
        >>> pair_edges(["c", "a", "b"])
        [('a', 'b'), ('a', 'c'), ('b', 'c')]
        >>> pair_edges(["solo"])
        []
    """
    people = sorted(set(creators))
    return [
        (people[i], people[j])
        for i in range(len(people))
        for j in range(i + 1, len(people))
    ]


def accumulate_edges(songs: Iterable[Iterable[str]]) -> Counter[tuple[str, str]]:
    """Accumulate co-credit edge weights across many songs.

    Args:
        songs: An iterable of per-song creator-key collections.

    Returns:
        A counter mapping each ``(a, b)`` pair to the number of shared songs.

    Examples:
        >>> w = accumulate_edges([["a", "b"], ["a", "b", "c"], ["a"]])
        >>> w[("a", "b")], w[("a", "c")], w[("b", "c")]
        (2, 1, 1)
    """
    weights: Counter[tuple[str, str]] = Counter()
    for creators in songs:
        weights.update(pair_edges(creators))
    return weights


def disparity_alpha(graph: nx.Graph) -> dict[tuple[Any, Any], float]:
    """Disparity-filter significance (alpha) for every edge of a weighted graph.

    For a node ``i`` of degree ``k`` and strength ``s`` (sum of incident edge
    weights), a neighbour ``j`` carries normalized weight ``p = w_ij / s`` and
    significance ``(1 - p) ** (k - 1)`` — the probability a uniform random split
    of ``i``'s strength would give an edge at least this strong. An undirected
    edge takes the **minimum** alpha over its two endpoints (significant from
    either side). Degree-1 endpoints contribute no alpha (theirs is always 1).

    Args:
        graph: A weighted undirected ``networkx`` graph (``weight`` edge attr).

    Returns:
        ``{(u, v): alpha}`` with ``u < v``; smaller alpha = more significant.

    Examples:
        >>> import networkx as nx
        >>> g = nx.Graph()
        >>> g.add_edge("a", "b", weight=10)
        >>> g.add_edge("a", "c", weight=1)
        >>> g.add_edge("a", "d", weight=1)
        >>> alpha = disparity_alpha(g)
        >>> round(alpha[("a", "b")], 4)  # b dominates a's strength -> significant
        0.0278
        >>> alpha[("a", "b")] < alpha[("a", "c")]  # the strong tie is keener
        True
    """
    alpha: dict[tuple[Any, Any], float] = {}
    for i in graph.nodes():
        degree = graph.degree(i)
        if degree <= 1:
            continue
        strength = sum(graph[i][j].get("weight", 1) for j in graph[i])
        if strength <= 0:
            continue
        for j in graph[i]:
            p = graph[i][j].get("weight", 1) / strength
            a = (1.0 - p) ** (degree - 1)
            edge = (i, j) if i < j else (j, i)
            alpha[edge] = min(alpha.get(edge, 1.0), a)
    return alpha


def backbone(graph: nx.Graph, alpha: float = DEFAULT_ALPHA) -> nx.Graph:
    """Extract the disparity-filter backbone at significance ``alpha``.

    Keeps edges that are significant (alpha below the threshold) from at least
    one endpoint, carrying node and edge attributes (plus the computed ``alpha``)
    onto the backbone. Nodes with no surviving edge are dropped.

    Args:
        graph: A weighted undirected ``networkx`` graph.
        alpha: Significance level (default :data:`DEFAULT_ALPHA`).

    Returns:
        The backbone subgraph.

    Examples:
        >>> import networkx as nx
        >>> g = nx.Graph()
        >>> g.add_edge("a", "b", weight=10)
        >>> g.add_edge("a", "c", weight=1)
        >>> g.add_edge("a", "d", weight=1)
        >>> bb = backbone(g, alpha=0.05)
        >>> sorted(bb.edges())
        [('a', 'b')]
    """
    import networkx as nx

    scores = disparity_alpha(graph)
    bb = nx.Graph()
    for (u, v), a in scores.items():
        if a < alpha:
            bb.add_edge(u, v, **graph[u][v], alpha=round(a, 6))
    for node in bb.nodes():
        bb.nodes[node].update(graph.nodes[node])
    return bb


# ---------------------------------------------------------------------------
# Database loading
# ---------------------------------------------------------------------------
def load_song_creators(
    db_path: str = "data/music.db",
) -> tuple[dict[int, set[str]], dict[int, int | None], dict[str, str]]:
    """Load each song's creator set, its decade, and a key→display-name map.

    Creators are the song's authors and composers unified by
    :func:`~src.enrich.match.normalize` (so a person credited as both lyricist
    and composer is one node); placeholders are dropped. Decade follows the same
    authoritative-then-page-year policy as the corpus (``None`` if undatable).

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        ``(creators, decade, display)`` where ``creators[song_id]`` is the set of
        normalized creator keys, ``decade[song_id]`` is the release decade or
        ``None``, and ``display[key]`` is a representative original spelling.
    """
    from src.db import (
        Author,
        Composer,
        Song,
        SongAuthor,
        SongComposer,
        get_engine,
        make_session,
    )

    creators: dict[int, set[str]] = {}
    display_counts: dict[str, Counter[str]] = {}

    def add(song_id: int, name: str) -> None:
        if is_placeholder(name):
            return
        key = normalize(name)
        creators.setdefault(song_id, set()).add(key)
        display_counts.setdefault(key, Counter())[name] += 1

    engine = get_engine(db_path)
    try:
        with make_session(engine)() as session:
            for sid, name in session.execute(
                Author.__table__.select()
                .with_only_columns(SongAuthor.song_id, Author.name)
                .select_from(SongAuthor.__table__)
                .join(Author.__table__, Author.id == SongAuthor.author_id)
            ):
                add(sid, name)
            for sid, name in session.execute(
                Composer.__table__.select()
                .with_only_columns(SongComposer.song_id, Composer.name)
                .select_from(SongComposer.__table__)
                .join(Composer.__table__, Composer.id == SongComposer.composer_id)
            ):
                add(sid, name)
            decade: dict[int, int | None] = {}
            for sid, fry, source, year in session.execute(
                Song.__table__.select().with_only_columns(
                    Song.id,
                    Song.first_release_year,
                    Song.first_release_source,
                    Song.year,
                )
            ):
                if sid not in creators:
                    continue
                auth = fry if source in AUTHORITATIVE_SOURCES else None
                decade[sid] = song_decade(auth, year)
    finally:
        engine.dispose()

    display = {
        key: counts.most_common(1)[0][0] for key, counts in display_counts.items()
    }
    return creators, decade, display


def build_graph(song_creators: Iterable[set[str]], display: dict[str, str]) -> nx.Graph:
    """Build a weighted collaboration graph from per-song creator sets.

    Args:
        song_creators: One creator-key set per song in the slice.
        display: ``key → display name`` map (sets each node's ``label``).

    Returns:
        A ``networkx`` graph: nodes carry ``label`` and ``songs`` (credit count);
        edges carry ``weight`` (shared-song count).
    """
    import networkx as nx

    song_creators = [set(s) for s in song_creators]
    node_songs: Counter[str] = Counter()
    for creators in song_creators:
        node_songs.update(creators)
    graph = nx.Graph()
    for key, n in node_songs.items():
        graph.add_node(key, label=display.get(key, key), songs=n)
    for (u, v), w in accumulate_edges(song_creators).items():
        graph.add_edge(u, v, weight=w)
    return graph


# ---------------------------------------------------------------------------
# Network summary
# ---------------------------------------------------------------------------
def top_collaborators(graph: nx.Graph, k: int = 15) -> list[dict[str, Any]]:
    """Rank a graph's people by weighted degree (collaboration strength).

    Args:
        graph: A built collaboration graph.
        k: How many to return.

    Returns:
        Up to ``k`` rows ``{"name", "strength", "partners", "songs"}``, strongest
        first. ``strength`` sums incident edge weights; ``partners`` is the degree.
    """
    rows = []
    for node in graph.nodes():
        strength = sum(graph[node][j].get("weight", 1) for j in graph[node])
        rows.append(
            {
                "name": graph.nodes[node].get("label", node),
                "strength": int(strength),
                "partners": graph.degree(node),
                "songs": int(graph.nodes[node].get("songs", 0)),
            }
        )
    rows.sort(key=lambda r: (r["strength"], r["partners"]), reverse=True)
    return rows[:k]


def summarize(graph: nx.Graph, bb: nx.Graph) -> dict[str, Any]:
    """Summarize a slice: full/backbone sizes, communities, top collaborators.

    Args:
        graph: The full collaboration graph.
        bb: Its disparity-filter backbone.

    Returns:
        A JSON-serializable summary dict.
    """
    import networkx as nx

    communities = (
        nx.community.louvain_communities(bb, weight="weight", seed=42)
        if bb.number_of_edges()
        else []
    )
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "backbone_nodes": bb.number_of_nodes(),
        "backbone_edges": bb.number_of_edges(),
        "backbone_communities": len(communities),
        "top_collaborators": top_collaborators(graph),
    }


def build_networks(
    db_path: str = "data/music.db",
    out_dir: Path | str = DEFAULT_COLLAB_DIR,
    *,
    alpha: float = DEFAULT_ALPHA,
    min_decade: int = 1950,
) -> dict[str, Any]:
    """Build the aggregate + per-decade collaboration networks and backbones.

    Args:
        db_path: Path to the SQLite database file.
        out_dir: Output directory for the ``.graphml`` files + ``summary.json``.
        alpha: Disparity-filter significance level.
        min_decade: Drop per-decade slices before this decade (too sparse).

    Returns:
        The summary dict (also written to ``summary.json``).
    """
    import networkx as nx

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    creators, decade, display = load_song_creators(db_path)

    slices: dict[str, list[set[str]]] = {"aggregate": list(creators.values())}
    by_decade: dict[int, list[set[str]]] = {}
    for sid, people in creators.items():
        dec = decade.get(sid)
        if dec is not None and dec >= min_decade:
            by_decade.setdefault(dec, []).append(people)
    for dec in sorted(by_decade):
        slices[f"decade_{dec}"] = by_decade[dec]

    summary: dict[str, Any] = {"alpha": alpha, "slices": {}}
    for name, song_sets in slices.items():
        graph = build_graph(song_sets, display)
        bb = backbone(graph, alpha)
        nx.write_graphml(graph, out_dir / f"{name}_full.graphml")
        nx.write_graphml(bb, out_dir / f"{name}_backbone.graphml")
        summary["slices"][name] = summarize(graph, bb)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    """CLI: build the collaboration networks and print a summary report."""
    import argparse

    ap = argparse.ArgumentParser(description="Build songwriter collaboration networks.")
    ap.add_argument("--db", default="data/music.db")
    ap.add_argument("--out-dir", default=str(DEFAULT_COLLAB_DIR))
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--min-decade", type=int, default=1950)
    args = ap.parse_args()

    summary = build_networks(
        args.db, args.out_dir, alpha=args.alpha, min_decade=args.min_decade
    )
    print(f"Collaboration networks -> {args.out_dir} (alpha={summary['alpha']})")
    for name, s in summary["slices"].items():
        print(
            f"  {name:16s} nodes={s['nodes']:6d} edges={s['edges']:7d} -> "
            f"backbone nodes={s['backbone_nodes']:5d} "
            f"edges={s['backbone_edges']:6d} ({s['backbone_communities']} comm.)"
        )
    agg_top = summary["slices"]["aggregate"]["top_collaborators"][:8]
    print("  top collaborators (aggregate, by strength):")
    for r in agg_top:
        name = r["name"][:28]
        print(f"    {name:28s} strength={r['strength']:5d} partners={r['partners']}")


if __name__ == "__main__":
    main()
