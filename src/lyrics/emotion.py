"""Per-song emotion classification over the lyrics corpus.

Runs the Hungarian emotion classifier
`visegradmedia-emotion/Emotion_RoBERTa_hungarian6 <https://huggingface.co/visegradmedia-emotion/Emotion_RoBERTa_hungarian6>`_
(an ``xlm-roberta-base`` sequence classifier, 6 labels) over the cleaned
natural-language ``text`` view of each :class:`~src.lyrics.corpus.SongDoc`, then
aggregates the predictions **overall** and **by decade** for the dashboard.

Label order was verified empirically against unambiguous Hungarian probe
sentences (one clearly-joyful, one fearful, …): the model's ``LABEL_i`` map to
:data:`EMOTIONS` ``[anger, fear, disgust, sadness, joy, none]`` — matching the
model card's stated order. ``none`` means "no clear emotion" (the model's sixth
class), not "missing data".

The run is **resumable**: each song's prediction is appended to
``per_song.jsonl`` as it completes, and a re-run skips song ids already present.
A song longer than the model's window is split into token chunks whose softmax
probabilities are length-weighted-averaged back into one per-song vector.

Pure helpers (chunking, pooling, aggregation) are deterministic and doctested;
the heavy ``torch``/``transformers`` dependencies are imported inside functions
so the module stays import-safe.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.lyrics.corpus import DEFAULT_CORPUS_DIR, SongDoc, load_corpus

# Verified label order (see module docstring): config LABEL_0..5 in this order.
EMOTIONS: tuple[str, ...] = ("anger", "fear", "disgust", "sadness", "joy", "none")

MODEL_NAME = "visegradmedia-emotion/Emotion_RoBERTa_hungarian6"
DEFAULT_EMOTION_DIR = Path("data/processed/emotion")
_PER_SONG = "per_song.jsonl"

# The model is xlm-roberta-base (max position 514). Leave room for <s>/</s>, so
# chunk the content token ids in blocks of this size.
MODEL_MAX_TOKENS = 512
CHUNK_LEN = MODEL_MAX_TOKENS - 2
# Cap how many chunks of a single (very long) song we score — lyrics rarely need
# more than this many windows, and it bounds the worst-case runtime per song.
MAX_CHUNKS = 6


def chunk_sequence(ids: list[int], max_len: int = CHUNK_LEN) -> list[list[int]]:
    """Split a token-id list into consecutive chunks of at most ``max_len``.

    Args:
        ids: The content token ids (no special tokens).
        max_len: Maximum ids per chunk.

    Returns:
        A list of chunks; ``[[]]`` for empty input (one empty chunk), so every
        song yields at least one model call.

    Examples:
        >>> chunk_sequence([1, 2, 3, 4, 5], max_len=2)
        [[1, 2], [3, 4], [5]]
        >>> chunk_sequence([], max_len=2)
        [[]]
        >>> chunk_sequence([7], max_len=512)
        [[7]]
    """
    if not ids:
        return [[]]
    if max_len < 1:
        raise ValueError("max_len must be >= 1")
    return [ids[i : i + max_len] for i in range(0, len(ids), max_len)]


def mean_pool(
    vectors: list[list[float]], weights: Sequence[float] | None = None
) -> list[float]:
    """Length-weighted average of equal-length probability vectors.

    Args:
        vectors: One probability vector per chunk (all the same length).
        weights: Optional per-vector weights (e.g. chunk token counts). Defaults
            to uniform. Non-positive total weight falls back to a uniform mean.

    Returns:
        The pooled vector (same length as the inputs).

    Raises:
        ValueError: If ``vectors`` is empty or vectors differ in length.

    Examples:
        >>> mean_pool([[1.0, 0.0], [0.0, 1.0]])
        [0.5, 0.5]
        >>> mean_pool([[1.0, 0.0], [0.0, 1.0]], weights=[3.0, 1.0])
        [0.75, 0.25]
        >>> mean_pool([[0.2, 0.8]])
        [0.2, 0.8]
    """
    if not vectors:
        raise ValueError("need at least one vector to pool")
    width = len(vectors[0])
    if any(len(v) != width for v in vectors):
        raise ValueError("vectors must be the same length")
    if weights is None or sum(weights) <= 0:
        weights = [1.0] * len(vectors)
    total = sum(weights)
    # Normalize the weights first (so they sum to 1) before applying them: this
    # keeps the result a true convex combination — always within [min, max] —
    # even when a weight is so small that ``value * weight`` would underflow.
    norm = [w / total for w in weights]
    return [
        sum(v[j] * w for v, w in zip(vectors, norm, strict=True))
        for j in range(width)
    ]


def dominant_emotion(probs: list[float], labels: tuple[str, ...] = EMOTIONS) -> str:
    """Return the highest-probability label (ties broken by lowest index).

    Args:
        probs: A probability vector aligned with ``labels``.
        labels: The label names (defaults to :data:`EMOTIONS`).

    Returns:
        The label with the greatest probability.

    Examples:
        >>> dominant_emotion([0.1, 0.0, 0.0, 0.0, 0.7, 0.2])
        'joy'
        >>> dominant_emotion([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
        'anger'
    """
    if len(probs) != len(labels):
        raise ValueError("probs and labels must align")
    best = max(range(len(probs)), key=lambda i: probs[i])
    return labels[best]


def aggregate(
    per_song: list[dict[str, Any]], labels: tuple[str, ...] = EMOTIONS
) -> dict[str, Any]:
    """Aggregate per-song predictions into overall + per-decade summaries.

    For each scope (overall, and each decade) two views are produced:

    * ``mean`` — the mean per-song probability for every label (a soft profile),
    * ``share`` — the fraction of songs whose *dominant* emotion is each label.

    Args:
        per_song: Rows with at least ``decade`` and ``probs`` (a vector aligned
            with ``labels``); a ``label`` field is used for ``share`` when
            present, else recomputed via :func:`dominant_emotion`.
        labels: The label names.

    Returns:
        ``{"labels": [...], "overall": {...}, "by_decade": [{"decade", "n",
        "mean", "share"}, ...]}`` with decades sorted ascending.

    Examples:
        >>> rows = [
        ...     {"decade": 1990, "probs": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]},
        ...     {"decade": 1990, "probs": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        ...     {"decade": 2000, "probs": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]},
        ... ]
        >>> agg = aggregate(rows)
        >>> agg["overall"]["n"]
        3
        >>> d90 = next(d for d in agg["by_decade"] if d["decade"] == 1990)
        >>> d90["n"], d90["share"]["joy"], d90["share"]["anger"]
        (2, 0.5, 0.5)
        >>> round(d90["mean"]["joy"], 3)
        0.5
    """

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        means = {lab: 0.0 for lab in labels}
        shares: Counter[str] = Counter()
        for row in rows:
            probs = row["probs"]
            for lab, p in zip(labels, probs, strict=True):
                means[lab] += p
            label = row.get("label") or dominant_emotion(probs, labels)
            shares[label] += 1
        return {
            "n": n,
            "mean": {lab: (means[lab] / n if n else 0.0) for lab in labels},
            "share": {lab: (shares[lab] / n if n else 0.0) for lab in labels},
        }

    by_decade: dict[int, list[dict[str, Any]]] = {}
    for row in per_song:
        by_decade.setdefault(int(row["decade"]), []).append(row)

    return {
        "labels": list(labels),
        "overall": summarize(per_song),
        "by_decade": [
            {"decade": dec, **summarize(by_decade[dec])} for dec in sorted(by_decade)
        ],
    }


class EmotionClassifier:
    """Lazy wrapper over the Hungarian emotion transformer (CPU/GPU)."""

    def __init__(self, model_name: str = MODEL_NAME, *, device: str | None = None):
        """Load the tokenizer and model in eval mode.

        Args:
            model_name: Hugging Face model id.
            device: Torch device string (``"cpu"``/``"cuda"``); auto-detected
                when ``None``.
        """
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        # transformers' factory return types are loosely typed (union incl. None);
        # treat the handles as Any so static checking doesn't choke on them.
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.device == "cpu":
            torch.set_num_threads(max(1, (torch.get_num_threads() or 1)))

    def classify_batch(self, texts: list[str]) -> list[list[float]]:
        """Classify a batch of texts into per-text probability vectors.

        Each text is tokenized without truncation, split into ≤512-token chunks
        (capped at :data:`MAX_CHUNKS`), and the chunks' softmax probabilities are
        length-weighted-averaged back into one vector per text. All chunks across
        the batch are scored in one forward pass for throughput.

        Args:
            texts: Cleaned natural-language lyric texts.

        Returns:
            One probability vector (aligned with :data:`EMOTIONS`) per input text.
        """
        torch = self._torch
        tok = self.tokenizer
        cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
        pad_id = tok.pad_token_id or 0

        # Build the flat list of chunk token-id rows, remembering which text and
        # how many tokens each chunk has (for length weighting).
        rows: list[list[int]] = []
        owners: list[int] = []
        lengths: list[int] = []
        for ti, text in enumerate(texts):
            ids = tok(text, add_special_tokens=False)["input_ids"]
            for chunk in chunk_sequence(ids, CHUNK_LEN)[:MAX_CHUNKS]:
                rows.append([cls_id, *chunk, sep_id])
                owners.append(ti)
                lengths.append(max(1, len(chunk)))

        width = max(len(r) for r in rows)
        input_ids = [r + [pad_id] * (width - len(r)) for r in rows]
        attn = [[1] * len(r) + [0] * (width - len(r)) for r in rows]
        with torch.inference_mode():
            logits = self.model(
                input_ids=torch.tensor(input_ids, device=self.device),
                attention_mask=torch.tensor(attn, device=self.device),
            ).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

        # Pool each text's chunk vectors (weighted by chunk token count).
        out: list[list[float]] = []
        for ti in range(len(texts)):
            vecs = [probs[i] for i in range(len(owners)) if owners[i] == ti]
            wts = [lengths[i] for i in range(len(owners)) if owners[i] == ti]
            out.append(mean_pool(vecs, wts))
        return out


def _done_ids(path: Path) -> set[int]:
    """Return the set of song ids already present in ``per_song.jsonl``."""
    done: set[int] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["song_id"]))
            except (ValueError, KeyError, json.JSONDecodeError):
                continue  # skip a torn final line from an interrupted run
    return done


def classify_corpus(
    corpus_dir: Path | str = DEFAULT_CORPUS_DIR,
    out_dir: Path | str = DEFAULT_EMOTION_DIR,
    *,
    batch_size: int = 16,
    limit: int | None = None,
    resume: bool = True,
) -> int:
    """Classify every corpus song's emotion, appending results as it goes.

    Args:
        corpus_dir: Directory of ``decade_*.jsonl`` corpus files.
        out_dir: Output directory for ``per_song.jsonl`` (created if absent).
        batch_size: Songs per forward pass (chunks are batched internally).
        limit: Optional cap on songs processed this run.
        resume: Skip song ids already present in ``per_song.jsonl``.

    Returns:
        The number of songs classified in this run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_song_path = out_dir / _PER_SONG

    docs: list[SongDoc] = load_corpus(corpus_dir)
    done = _done_ids(per_song_path) if resume else set()
    todo = [d for d in docs if d.song_id not in done and d.text.strip()]
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        return 0

    clf = EmotionClassifier()
    written = 0
    with per_song_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            vectors = clf.classify_batch([d.text for d in batch])
            for doc, probs in zip(batch, vectors, strict=True):
                rounded = [round(p, 4) for p in probs]
                handle.write(
                    json.dumps(
                        {
                            "song_id": doc.song_id,
                            "decade": doc.decade,
                            "label": dominant_emotion(probs),
                            "probs": rounded,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            handle.flush()
            written += len(batch)
    return written


def build_artifacts(
    out_dir: Path | str = DEFAULT_EMOTION_DIR,
) -> dict[str, Any]:
    """Aggregate ``per_song.jsonl`` into ``emotion.json`` for the dashboard.

    Args:
        out_dir: Directory holding ``per_song.jsonl``; ``emotion.json`` is
            written alongside it.

    Returns:
        The aggregated dict (see :func:`aggregate`).
    """
    out_dir = Path(out_dir)
    per_song_path = out_dir / _PER_SONG
    rows: list[dict[str, Any]] = []
    if per_song_path.exists():
        with per_song_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:  # tolerate a torn final line if a run is still appending
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    agg = aggregate(rows)
    (out_dir / "emotion.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return agg


def main() -> None:
    """CLI: ``run`` the classifier or ``aggregate`` the cached predictions."""
    import argparse

    ap = argparse.ArgumentParser(description="Corpus emotion classification.")
    sub = ap.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="classify corpus songs (resumable)")
    r.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    r.add_argument("--out-dir", default=str(DEFAULT_EMOTION_DIR))
    r.add_argument("--batch-size", type=int, default=16)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--no-resume", action="store_true")

    a = sub.add_parser("aggregate", help="build emotion.json from predictions")
    a.add_argument("--out-dir", default=str(DEFAULT_EMOTION_DIR))

    args = ap.parse_args()
    if args.command == "run":
        n = classify_corpus(
            args.corpus_dir,
            args.out_dir,
            batch_size=args.batch_size,
            limit=args.limit,
            resume=not args.no_resume,
        )
        print(f"Classified {n} songs -> {Path(args.out_dir) / _PER_SONG}")
        agg = build_artifacts(args.out_dir)
        print(f"Aggregated {agg['overall']['n']} songs -> emotion.json")
    elif args.command == "aggregate":
        agg = build_artifacts(args.out_dir)
        print(f"Aggregated {agg['overall']['n']} songs -> emotion.json")


if __name__ == "__main__":
    main()
