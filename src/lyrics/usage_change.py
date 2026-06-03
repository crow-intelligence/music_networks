"""Phase 3 — diachronic word-usage change with chronowords.

For each decade we train a `chronowords` :class:`~chronowords.algebra.svd.SVDAlgebra`
model (PPMI + truncated SVD) on that decade's **lemma tokens**, then align
consecutive decades with `chronowords`'
:class:`~chronowords.alignment.procrustes.ProcrustesAligner`. A target word's
cosine similarity between its aligned representation in decade *D* and decade
*D+1* measures how stable its meaning is; ``1 - similarity`` is the **shift
intensity** the dashboard plots.

The module produces exactly the data shapes the crowintelligence.org/analysis
dashboard uses, adapted from presidential *periods* to song *decades*:

* ``semantic_change`` — per target word, shift intensity at each transition,
* ``shift_heatmap`` — word × transition intensity matrix,
* ``cumulative_drift`` — running drift from the baseline decade,
* ``nearest_neighbors`` — per word per decade, top-N similar lemmas,
* ``vocab_stats`` — vocabulary size / embedding dim / song count per decade,
* ``alignment_stats`` — aligned-word count + mean similarity per transition.

Pre-1980 decades are thin (roadmap caveat), so decades below ``MIN_DECADE_SONGS``
songs are dropped from the diachronic models and flagged. The pure helpers are
doctested; chronowords is imported inside functions so the module stays
import-safe without the ``nlp-ml`` extra installed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.lyrics.corpus import SongDoc, load_corpus

DEFAULT_USAGE_DIR = Path("data/processed/usage")

# SVD on a decade needs a reasonable amount of text to be meaningful.
MIN_DECADE_SONGS = 30
N_COMPONENTS = 100
NEIGHBORS = 10
# Count-Min Sketch width. chronowords' 1M default saturates on the largest
# decades (~10k songs), collapsing their PPMI matrix to a near-empty rank ~100
# space (most word vectors become zero). 8M restores full-rank embeddings; the
# extra memory (~0.6 GB of counters) is well worth correct vectors.
CMS_WIDTH = 8_000_000

# Hungarian lemma target words tracked for meaning change. Lemmatized + lowercase
# to match the corpus ``tokens`` view. Override with --words on the CLI.
TARGET_WORDS = (
    "szerelem",  # love
    "szív",  # heart
    "pénz",  # money
    "haza",  # homeland
    "szabadság",  # freedom
    "isten",  # god
    "halál",  # death
    "város",  # city
    "éjszaka",  # night
    "lány",  # girl
    "élet",  # life
    "háború",  # war
)


def decade_token_streams(docs: list[SongDoc]) -> dict[int, list[list[str]]]:
    """Group documents' lemma token lists by decade.

    Args:
        docs: The corpus (see :func:`src.lyrics.corpus.load_corpus`).

    Returns:
        Map of decade to the list of per-song token lists in that decade.

    Examples:
        >>> from src.lyrics.corpus import SongDoc
        >>> docs = [
        ...     SongDoc(1, 1990, "", ["a", "b"]),
        ...     SongDoc(2, 1990, "", ["c"]),
        ...     SongDoc(3, 2000, "", ["d"]),
        ... ]
        >>> streams = decade_token_streams(docs)
        >>> sorted(streams)
        [1990, 2000]
        >>> streams[1990]
        [['a', 'b'], ['c']]
    """
    out: dict[int, list[list[str]]] = {}
    for doc in docs:
        out.setdefault(doc.decade, []).append(doc.tokens)
    return out


def reliable_decades(
    streams: dict[int, list[list[str]]], min_songs: int = MIN_DECADE_SONGS
) -> list[int]:
    """Return the sorted decades with at least ``min_songs`` songs.

    Args:
        streams: Per-decade token streams (see :func:`decade_token_streams`).
        min_songs: Minimum song count for a decade to be modelled.

    Returns:
        Sorted list of decades that clear the threshold.

    Examples:
        >>> streams = {1980: [[]], 1990: [[]] * 40, 2000: [[]] * 50}
        >>> reliable_decades(streams, min_songs=30)
        [1990, 2000]
        >>> reliable_decades({}, min_songs=1)
        []
    """
    return sorted(d for d, songs in streams.items() if len(songs) >= min_songs)


def shift_intensity(similarity: float) -> float:
    """Convert an aligned cosine similarity into a shift intensity in ``[0, 2]``.

    A similarity of 1 means no change (intensity 0); lower similarity means more
    meaning change. Negative similarities are allowed and yield intensity > 1.

    Args:
        similarity: Aligned cosine similarity in ``[-1, 1]``.

    Returns:
        ``1 - similarity``, floored at 0.

    Examples:
        >>> shift_intensity(1.0)
        0.0
        >>> shift_intensity(0.25)
        0.75
        >>> shift_intensity(1.2)
        0.0
    """
    return max(0.0, 1.0 - similarity)


def cumulative_drift(shifts: list[float | None]) -> list[float | None]:
    """Accumulate per-transition shifts into a drift-from-baseline series.

    The returned series is prefixed with ``0.0`` for the baseline decade, then one
    running total per transition. A missing transition (``None``) is carried
    forward as ``None`` and does not advance the running total.

    Args:
        shifts: Per-transition shift intensities (``None`` where unavailable).

    Returns:
        Drift values, one per decade (``len(shifts) + 1`` entries).

    Examples:
        >>> cumulative_drift([0.2, 0.3])
        [0.0, 0.2, 0.5]
        >>> cumulative_drift([0.2, None, 0.1])
        [0.0, 0.2, None, 0.30000000000000004]
        >>> cumulative_drift([])
        [0.0]
    """
    out: list[float | None] = [0.0]
    total = 0.0
    for shift in shifts:
        if shift is None:
            out.append(None)
            continue
        total += shift
        out.append(total)
    return out


def transition_labels(decades: list[int]) -> list[str]:
    """Build ``"D→D'"`` labels for consecutive decade pairs.

    Args:
        decades: Sorted decades.

    Returns:
        One arrow label per consecutive pair.

    Examples:
        >>> transition_labels([1990, 2000, 2010])
        ['1990→2000', '2000→2010']
        >>> transition_labels([1990])
        []
    """
    return [f"{a}→{b}" for a, b in zip(decades, decades[1:], strict=False)]


def train_decade_models(
    streams: dict[int, list[list[str]]],
    decades: list[int],
    *,
    n_components: int = N_COMPONENTS,
) -> dict[int, Any]:
    """Train one SVDAlgebra model per decade on its joined lemma tokens.

    Args:
        streams: Per-decade token streams (see :func:`decade_token_streams`).
        decades: The decades to model (see :func:`reliable_decades`).
        n_components: SVD dimensionality per decade model.

    Returns:
        Map of decade to a trained ``SVDAlgebra`` (decades whose vocabulary is
        empty are skipped).
    """
    from chronowords.algebra.svd import SVDAlgebra

    models: dict[int, Any] = {}
    for decade in decades:
        lines = (" ".join(tokens) for tokens in streams[decade])
        model = SVDAlgebra(n_components=n_components, cms_width=CMS_WIDTH)
        try:
            model.train(lines)
        except ValueError:
            # No words cleared the frequency threshold for this decade.
            continue
        if model.vocabulary:
            models[decade] = model
    return models


def align_pair(model_a: Any, model_b: Any) -> tuple[Any, Any]:
    """Procrustes-align ``model_a`` onto ``model_b``.

    The alignment uses the **full shared vocabulary** as anchor words (the
    standard diachronic-embedding approach). Chronowords' default anchor
    selection slices an unsorted top-rank window, which makes the anchor set —
    and hence the shift scores — wildly inconsistent across decade pairs; passing
    the explicit intersection fixes that.

    Args:
        model_a: Source-decade ``SVDAlgebra``.
        model_b: Target-decade ``SVDAlgebra``.

    Returns:
        ``(aligner, metrics)`` — a fitted ``ProcrustesAligner`` and its
        ``AlignmentMetrics`` (mean similarity, aligned-word count, error).
    """
    from chronowords.alignment.procrustes import ProcrustesAligner

    shared = sorted(set(model_a.vocabulary) & set(model_b.vocabulary))
    aligner = ProcrustesAligner()
    metrics = aligner.fit(
        model_a.embeddings,
        model_b.embeddings,
        model_a.vocabulary,
        model_b.vocabulary,
        anchor_words=shared,
    )
    return aligner, metrics


def analyze(
    docs: list[SongDoc],
    *,
    target_words: tuple[str, ...] = TARGET_WORDS,
    min_songs: int = MIN_DECADE_SONGS,
    n_components: int = N_COMPONENTS,
    neighbors: int = NEIGHBORS,
) -> dict[str, Any]:
    """Run the full diachronic analysis and return the dashboard data blocks.

    Args:
        docs: The corpus.
        target_words: Hungarian lemmas to track for meaning change.
        min_songs: Minimum songs for a decade to be modelled.
        n_components: SVD dimensionality.
        neighbors: Nearest-neighbour count per word per decade.

    Returns:
        A dict with ``decades``, ``transitions``, ``semantic_change``,
        ``shift_heatmap``, ``cumulative_drift``, ``nearest_neighbors``,
        ``vocab_stats`` and ``alignment_stats``.
    """
    streams = decade_token_streams(docs)
    decades = reliable_decades(streams, min_songs)
    models = train_decade_models(streams, decades, n_components=n_components)
    decades = [d for d in decades if d in models]
    labels = transition_labels(decades)

    # Align each consecutive pair; record per-word cross-decade similarity.
    pair_word_sim: list[dict[str, float]] = []
    alignment_stats: list[dict[str, Any]] = []
    for a, b in zip(decades, decades[1:], strict=False):
        aligner, metrics = align_pair(models[a], models[b])
        sims: dict[str, float] = {}
        for word in target_words:
            sim = aligner.get_word_similarity(
                word, models[a].embeddings, models[b].embeddings
            )
            # Skip None (word absent) and NaN (zero-norm vector → invalid divide).
            if sim is not None and math.isfinite(sim):
                sims[word] = float(sim)
        pair_word_sim.append(sims)
        alignment_stats.append(
            {
                "from": a,
                "to": b,
                "aligned_words": int(metrics.num_aligned_words),
                "avg_similarity": round(float(metrics.average_cosine_similarity), 4),
            }
        )

    semantic_change: list[dict[str, Any]] = []
    heatmap_values: list[list[float | None]] = []
    drift: list[dict[str, Any]] = []
    for word in target_words:
        shifts: list[float | None] = []
        for sims in pair_word_sim:
            sim = sims.get(word)
            shifts.append(None if sim is None else shift_intensity(sim))
        semantic_change.append(
            {
                "word": word,
                "shifts": [
                    {"transition": label, "shift": shift}
                    for label, shift in zip(labels, shifts, strict=True)
                ],
            }
        )
        heatmap_values.append([s if s is not None else None for s in shifts])
        drift.append({"word": word, "drift": cumulative_drift(shifts)})

    nearest_neighbors = _nearest_neighbors(models, decades, target_words, neighbors)
    vocab_stats = [
        {
            "decade": d,
            "vocab_size": len(models[d].vocabulary),
            "embedding_dim": int(models[d].embeddings.shape[1]),
            "n_songs": len(streams[d]),
        }
        for d in decades
    ]

    return {
        "decades": decades,
        "transitions": labels,
        "target_words": list(target_words),
        "semantic_change": semantic_change,
        "shift_heatmap": {"words": list(target_words), "values": heatmap_values},
        "cumulative_drift": drift,
        "nearest_neighbors": nearest_neighbors,
        "vocab_stats": vocab_stats,
        "alignment_stats": alignment_stats,
    }


def _nearest_neighbors(
    models: dict[int, Any],
    decades: list[int],
    target_words: tuple[str, ...],
    n: int,
) -> list[dict[str, Any]]:
    """Per word per decade, the top-``n`` similar lemmas in that decade's model."""
    out: list[dict[str, Any]] = []
    for word in target_words:
        per_decade: list[dict[str, Any]] = []
        for decade in decades:
            sims = models[decade].most_similar(word, n=n)
            per_decade.append(
                {
                    "decade": decade,
                    "neighbors": [
                        {"word": s.word, "similarity": round(s.similarity, 4)}
                        for s in sims
                    ],
                }
            )
        out.append({"word": word, "by_decade": per_decade})
    return out


def save_artifacts(
    data: dict[str, Any], out_dir: Path | str = DEFAULT_USAGE_DIR
) -> None:
    """Write each analysis block to its own JSON file under ``out_dir``.

    Args:
        data: The dict returned by :func:`analyze`.
        out_dir: Output directory (created if missing).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key in (
        "semantic_change",
        "shift_heatmap",
        "cumulative_drift",
        "nearest_neighbors",
        "vocab_stats",
        "alignment_stats",
    ):
        (out / f"{key}.json").write_text(
            json.dumps(data[key], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    meta = {
        "decades": data["decades"],
        "transitions": data["transitions"],
        "target_words": data["target_words"],
    }
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    """Run the diachronic analysis from the cached corpus and write JSON."""
    import argparse

    parser = argparse.ArgumentParser(description="Diachronic word-usage change.")
    parser.add_argument("--corpus-dir", default="data/processed/corpus")
    parser.add_argument("--out-dir", default=str(DEFAULT_USAGE_DIR))
    parser.add_argument("--min-songs", type=int, default=MIN_DECADE_SONGS)
    parser.add_argument(
        "--words", nargs="*", default=list(TARGET_WORDS), help="target lemmas"
    )
    args = parser.parse_args()

    docs = load_corpus(args.corpus_dir)
    if not docs:
        raise SystemExit(f"No corpus docs in {args.corpus_dir}; build it first.")

    data = analyze(docs, target_words=tuple(args.words), min_songs=args.min_songs)
    save_artifacts(data, args.out_dir)
    print(
        f"Modelled {len(data['decades'])} decades "
        f"({', '.join(map(str, data['decades']))}) -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
