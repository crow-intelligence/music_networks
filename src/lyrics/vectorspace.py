"""Word-vector drift trajectories for the essay's "vektortér" chart.

Rebuilds the per-decade chronowords embeddings (PPMI + truncated SVD, the pipeline
in :mod:`src.lyrics.usage_change`), Procrustes-aligns every decade into one shared
reference frame, then projects all ``(word, decade)`` unit vectors together with a
single 2D reducer so each target word traces a *path* across the decades — showing
the *direction* of its drift.

Caveat (kept deliberately, as an illustration): cross-decade alignment is only
~0.25 cosine for our small per-decade corpora, so the paths are approximate, not a
precise measurement — they convey movement, not exact coordinates.

Projection: ``umap`` (needs the ``nlp-ml`` extra) → metric ``mds`` on cosine
distances → ``pca`` fallbacks (``--method``). Reuses ``usage_change`` helpers; needs
only the cached corpus, no DB. Cached JSON, gitignored (like allotax/networks).

Run: ``uv run python -m src.lyrics.vectorspace`` (re-run when the corpus changes).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_CORPUS_DIR = "data/processed/corpus"
DEFAULT_DIR = Path("data/processed/vectorspace")
MIN_DECADE = 1950
TOP_MOVERS = 6


def _normalize_coords(coords: Any) -> Any:
    """Scale 2D coords into ``[0, 1]`` preserving aspect ratio (single span).

    Examples:
        >>> _normalize_coords([[0, 0], [2, 1]]).tolist()
        [[0.0, 0.0], [1.0, 0.5]]
    """
    import numpy as np

    c = np.asarray(coords, dtype=float)
    mins = c.min(axis=0)
    span = float((c.max(axis=0) - mins).max()) or 1.0
    return (c - mins) / span


def _unit(v: Any) -> Any:
    """L2-normalise a vector onto the unit sphere.

    chronowords magnitudes are ~uniform; only the *direction* carries the semantic
    signal, so trajectories are compared as unit vectors.

    Examples:
        >>> _unit([3.0, 4.0]).tolist()
        [0.6, 0.8]
        >>> _unit([0.0, 0.0]).tolist()
        [0.0, 0.0]
    """
    import numpy as np

    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else a


def _path_length(vectors: Any) -> float:
    """Total length of the poly-line through ``vectors`` (0 for < 2 points).

    Examples:
        >>> _path_length([[0, 0], [3, 4]])
        5.0
        >>> _path_length([[1, 1]])
        0.0
    """
    import numpy as np

    v = np.asarray(vectors, dtype=float)
    if len(v) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(v, axis=0), axis=1).sum())


def _project(x: Any, method: str, seed: int = 42) -> tuple[Any, str]:
    """Project rows of ``x`` to 2D, trying ``method`` then mds/pca fallbacks.

    Returns ``(coords, used_method)``. ``umap`` needs the ``nlp-ml`` extra; ``mds``
    (metric, cosine) and ``pca`` are core deps.
    """
    import numpy as np

    n = len(x)
    order = [method] + [m for m in ("umap", "mds", "pca") if m != method]
    for m in order:
        try:
            if m == "umap":
                from umap import UMAP

                nn = max(2, min(15, n - 1))
                coords = UMAP(
                    n_components=2, metric="cosine", n_neighbors=nn,
                    min_dist=0.35, random_state=seed,
                ).fit_transform(x)
                return coords, "umap"
            if m == "mds":
                from sklearn.manifold import MDS
                from sklearn.metrics.pairwise import cosine_distances

                dist = cosine_distances(x)
                coords = MDS(
                    n_components=2, dissimilarity="precomputed", random_state=seed,
                    normalized_stress="auto", n_init=4,
                ).fit_transform(dist)
                return coords, "mds"
            if m == "tsne":
                from sklearn.manifold import TSNE

                perp = min(30, max(5, (n - 1) // 3))
                coords = TSNE(
                    n_components=2, metric="cosine", perplexity=perp,
                    random_state=seed, init="pca",
                ).fit_transform(x)
                return coords, "tsne"
            if m == "pca":
                from sklearn.decomposition import PCA

                return PCA(n_components=2).fit_transform(x), "pca"
        except Exception as exc:  # noqa: BLE001 — any missing dep / numerical fail
            print(f"  {m} projection failed ({type(exc).__name__}); falling back")
    # dependency-free last resort: PCA via numpy SVD
    xc = np.asarray(x, dtype=float)
    xc = xc - xc.mean(axis=0)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    return xc @ vt[:2].T, "pca-numpy"


def export_vectorspace(
    corpus_dir: str = DEFAULT_CORPUS_DIR,
    out_dir: Path | str = DEFAULT_DIR,
    *,
    target_words: list[str] | None = None,
    method: str = "umap",
    seed: int = 42,
    top_movers: int = TOP_MOVERS,
) -> Path:
    """Compute per-word drift trajectories → ``vectorspace.json``.

    Each target word's Procrustes-aligned, unit-normalised vector per decade is
    pooled and projected to a single shared 2D plane; the ordered points form the
    word's trajectory. ``drift`` (poly-line length on the unit sphere) ranks the
    movers for emphasis. Paths are approximate (see the module note), shown to
    convey the *direction* of drift.
    """
    import numpy as np

    from src.lyrics.corpus import load_corpus
    from src.lyrics.usage_change import (
        MIN_DECADE_SONGS,
        TARGET_WORDS,
        align_pair,
        decade_token_streams,
        reliable_decades,
        train_decade_models,
    )

    words = target_words or list(TARGET_WORDS)
    docs = load_corpus(corpus_dir)
    streams = decade_token_streams(docs)
    decades = [
        d for d in reliable_decades(streams, MIN_DECADE_SONGS) if d >= MIN_DECADE
    ]
    if len(decades) < 2:
        raise SystemExit("vectorspace: need >= 2 modellable decades")

    # Reference frame = the most-populous decade (richest, most stable vocabulary).
    ref = max(decades, key=lambda d: len(streams[d]))
    print(f"vectorspace: {len(decades)} decades, ref={ref}, method={method}")
    ref_model = train_decade_models(streams, [ref]).get(ref)
    if ref_model is None:
        raise SystemExit("vectorspace: reference decade failed to train")

    pooled: list[tuple[str, int, Any]] = []  # (word, decade, aligned unit vector)
    for d in decades:
        model = ref_model if d == ref else train_decade_models(streams, [d]).get(d)
        if model is None:
            continue
        if d == ref:
            aligned = np.asarray(model.embeddings, dtype=float)
        else:
            aligner, _ = align_pair(model, ref_model)
            aligned = np.asarray(aligner.transform(model.embeddings), dtype=float)
        idx = {w: i for i, w in enumerate(model.vocabulary)}
        for w in words:
            if w in idx:
                pooled.append((w, d, _unit(aligned[idx[w]])))
        if model is not ref_model:
            del model

    if len(pooled) < 3:
        raise SystemExit("vectorspace: too few (word, decade) vectors to project")

    matrix = np.array([v for _, _, v in pooled], dtype=float)
    coords, used = _project(matrix, method, seed)
    coords = _normalize_coords(coords)

    by_word: dict[str, dict[str, Any]] = {}
    for (w, d, vec), (x, y) in zip(pooled, coords, strict=True):
        entry = by_word.setdefault(w, {"vecs": [], "decs": [], "path": []})
        entry["vecs"].append(vec)
        entry["decs"].append(d)
        entry["path"].append(
            {"decade": d, "x": round(float(x), 4), "y": round(float(y), 4)}
        )

    def _word_drift(info: dict[str, Any]) -> float:
        order = sorted(range(len(info["decs"])), key=lambda i: info["decs"][i])
        return _path_length([info["vecs"][i] for i in order])

    out_words = [
        {
            "word": w,
            "drift": round(_word_drift(info), 4),
            "path": sorted(info["path"], key=lambda p: p["decade"]),
        }
        for w, info in by_word.items()
    ]
    out_words.sort(key=lambda z: -z["drift"])
    top: list[str] = [str(w["word"]) for w in out_words[:top_movers]]
    payload = {
        "method": used,
        "ref": ref,
        "decades": decades,
        "top_movers": top,
        "words": out_words,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "vectorspace.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"vectorspace: {len(out_words)} words via {used} → {path}\n"
        f"  top movers: {', '.join(top)}"
    )
    return path


def load_vectorspace(vspace_dir: Path | str = DEFAULT_DIR) -> dict[str, Any]:
    """Load ``vectorspace.json`` (empty dict if absent)."""
    path = Path(vspace_dir) / "vectorspace.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """CLI: ``python -m src.lyrics.vectorspace``."""
    import argparse

    ap = argparse.ArgumentParser(description="Export word-vector drift trajectories.")
    ap.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR)
    ap.add_argument("--out-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--method", default="umap", choices=["umap", "mds", "pca", "tsne"])
    ap.add_argument("--words", nargs="*", default=None, help="target lemmas")
    args = ap.parse_args()
    export_vectorspace(
        args.corpus_dir, Path(args.out_dir), target_words=args.words, method=args.method
    )


if __name__ == "__main__":
    main()
