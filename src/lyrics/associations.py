"""Genre association cross-tabs for the dashboard.

Joins the per-song genre tags (Discogs) with other per-song signals to answer
"which X goes with which genre":

* **emotion × genre** — the dominant-emotion mix within each genre,
* **topic × genre** — the topic mix within each genre (needs per-song topic ids),
* **genre × lexical diversity** — MATTR computed per genre.

All are honest only on the Discogs-genred subset. The matrix builders are pure
and doctested; the small JSONL loaders keep the IO out of them.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

# A genre needs at least this many songs (in the joined subset) to be charted.
MIN_GENRE_SONGS = 40
MAX_GENRES = 9


def load_genre_map(genre_dir: Path | str) -> dict[int, str]:
    """Load ``song_id → genre`` for tagged songs from the genre artifact.

    Args:
        genre_dir: Directory holding the genre ``per_song.jsonl``.

    Returns:
        ``{song_id: genre}`` for songs that got a (non-null) genre.
    """
    out: dict[int, str] = {}
    path = Path(genre_dir) / "per_song.jsonl"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("genre"):
                        out[int(rec["song_id"])] = rec["genre"]
    return out


def _top_genres(genre_of: dict[int, str], keys: set[int]) -> list[str]:
    """Genres present among ``keys``, frequency-ordered, capped + min-filtered."""
    counts = Counter(genre_of[k] for k in keys if k in genre_of)
    return [g for g, n in counts.most_common(MAX_GENRES) if n >= MIN_GENRE_SONGS]


def category_by_genre(
    labels_of: dict[int, str],
    genre_of: dict[int, str],
    categories: list[str],
) -> dict[str, Any]:
    """Build a genre × category share matrix (rows = genre, cols = category).

    For each genre, the fraction of its songs falling in each category (e.g. each
    dominant emotion, or each topic) — every row sums to 1.

    Args:
        labels_of: ``song_id → category`` (e.g. dominant emotion, topic name).
        genre_of: ``song_id → genre``.
        categories: The column order (categories not in it are ignored).

    Returns:
        ``{"genres", "categories", "rows": [{genre, n, shares: {cat: frac}}]}``,
        genres frequency-ordered.

    Examples:
        >>> lab = {1: "joy", 2: "joy", 3: "anger", 4: "joy"}
        >>> gen = {1: "Pop", 2: "Pop", 3: "Pop", 4: "Rock"}
        >>> out = category_by_genre(lab, gen, ["anger", "joy"])  # doctest: +SKIP
    """
    keys = set(labels_of) & set(genre_of)
    genres = _top_genres(genre_of, keys)
    catset = set(categories)
    rows = []
    for g in genres:
        members = [k for k in keys if genre_of[k] == g and labels_of[k] in catset]
        n = len(members)
        counts = Counter(labels_of[k] for k in members)
        rows.append(
            {
                "genre": g,
                "n": n,
                "shares": {c: (counts.get(c, 0) / n if n else 0.0) for c in categories},
            }
        )
    return {"genres": genres, "categories": categories, "rows": rows}


def mattr_by_genre(
    tokens_of: dict[int, list[str]],
    genre_of: dict[int, str],
    *,
    window: int = 100,
) -> dict[str, Any]:
    """Compute MATTR (lexical diversity) per genre over the joined songs.

    Args:
        tokens_of: ``song_id → token list`` (corpus tokens).
        genre_of: ``song_id → genre``.
        window: MATTR window (see :func:`src.lyrics.diversity.mattr`).

    Returns:
        ``{"rows": [{genre, n, mattr}], ...}`` frequency-ordered, for genres with
        enough songs.
    """
    from src.lyrics.diversity import mattr

    keys = set(tokens_of) & set(genre_of)
    genres = _top_genres(genre_of, keys)
    rows = []
    for g in genres:
        members = [k for k in keys if genre_of[k] == g]
        toks: list[str] = []
        for k in members:
            toks.extend(tokens_of[k])
        rows.append(
            {"genre": g, "n": len(members), "mattr": round(mattr(toks, window), 4)}
        )
    return {"rows": rows}


def load_emotion_labels(emotion_dir: Path | str) -> dict[int, str]:
    """Load ``song_id → dominant-emotion label`` from the emotion artifact."""
    out: dict[int, str] = {}
    path = Path(emotion_dir) / "per_song.jsonl"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("label"):
                        out[int(rec["song_id"])] = rec["label"]
    return out
