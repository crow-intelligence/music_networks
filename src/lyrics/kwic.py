"""KWIC (keyword-in-context) concordances for the scrollytelling essay.

For a handful of target words, collect SHORT in-context windows (±N words) from
the corpus lyrics, per decade — so the essay can show *how* a word is used in each
era. Only brief windows are stored (never full lyrics).

Hungarian is heavily inflected and the corpus ``text`` is surface form, so each
target is matched by a **stem** (``szerelem`` → ``szerel`` catches *szerelmem*,
*szerelmes*, …). Matching is on the surface text (readable), not the lemma tokens.

The pure helpers are doctested; corpus I/O lives in :func:`build_kwic`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEFAULT_CORPUS_DIR = Path("data/processed/corpus")
DEFAULT_KWIC_DIR = Path("data/processed/kwic")
MIN_DECADE = 1950

# Essay target words → a match prefix (or tuple of prefixes) trimmed for Hungarian
# inflection. Tuples exclude homographs: "szerelem"/"szerelm" catches the love
# inflections (szerelmes, szerelmet) but NOT szerel-ik / szerel-vény (mount/train).
DEFAULT_TARGETS: dict[str, str | tuple[str, ...]] = {
    "szerelem": ("szerelem", "szerelm"),
    "lány": "lány",
    "csaj": "csaj",
    "pénz": "pénz",
    "éjszaka": "éjszak",
    "szabadság": "szabadság",
    "élet": "élet",
}

# Lowercased substrings that mark a window as junk even when the keyword itself is a
# real hit — e.g. "árvalányhaj" (feather-grass, a plant) can sit in a "lány" window's
# wide context. A window is dropped if any of these appears in its pre+kw+post.
_CONTEXT_BLOCKLIST: tuple[str, ...] = ("árvalányhaj",)

_STRIP = re.compile(r"^\W+|\W+$", re.UNICODE)


def _norm(token: str) -> str:
    """Lowercase a word and strip leading/trailing punctuation.

    Examples:
        >>> _norm("Szerelem,")
        'szerelem'
        >>> _norm("(éjszakát)")
        'éjszakát'
    """
    return _STRIP.sub("", token).lower()


def concordances(
    text: str,
    stem: str | tuple[str, ...],
    *,
    window: int = 12,
    limit: int | None = None,
) -> list[dict[str, str]]:
    """Find ``±window``-word windows around words whose stem matches ``stem``.

    Args:
        text: Surface lyric text (whitespace-separated).
        stem: A lowercased match prefix, or a tuple of prefixes — a word matches
            if its normalised form starts with any of them. Tuples let a target
            keep its true inflections while excluding homographs.
        window: Words of context kept on each side.
        limit: Max windows to return (``None`` = all).

    Returns:
        ``[{"pre", "kw", "post"}]`` — the words before, the matched word, after.

    Examples:
        >>> c = concordances("Egy szép szerelem dala szól ma", "szerel")
        >>> c[0]["pre"], c[0]["kw"], c[0]["post"]
        ('Egy szép', 'szerelem', 'dala szól ma')
        >>> [c["kw"] for c in concordances(
        ...     "a szerelvény és a szerelmem", ("szerelem", "szerelm"))]
        ['szerelmem']
    """
    words = text.split()
    out: list[dict[str, str]] = []
    for i, word in enumerate(words):
        if not _norm(word).startswith(stem):
            continue
        out.append(
            {
                "pre": " ".join(words[max(0, i - window) : i]),
                "kw": word,
                "post": " ".join(words[i + 1 : i + 1 + window]),
            }
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def build_kwic(
    corpus_dir: Path | str = DEFAULT_CORPUS_DIR,
    out_dir: Path | str = DEFAULT_KWIC_DIR,
    *,
    targets: dict[str, str | tuple[str, ...]] | None = None,
    window: int = 12,
    per_decade: int = 8,
) -> dict[str, dict[str, list[dict[str, str]]]]:
    """Collect per-(word, decade) concordances from the corpus into ``kwic.json``.

    Keeps **at most one window per song** per word (so a decade's windows come from
    distinct songs — no same-song overlaps), skips windows whose context matches
    :data:`_CONTEXT_BLOCKLIST`, and de-duplicates identical windows across songs.

    Args:
        corpus_dir: Per-decade corpus JSONL dir (records carry surface ``text``).
        out_dir: Output dir (``kwic.json`` is written here).
        targets: ``word → prefix(es)`` map (defaults to :data:`DEFAULT_TARGETS`).
        window: Context words per side.
        per_decade: Max distinct windows kept per word per decade.

    Returns:
        ``{word: {decade_str: [{"pre","kw","post"}]}}`` (also written to disk).
    """
    targets = targets or DEFAULT_TARGETS
    result: dict[str, dict[str, list[dict[str, str]]]] = {w: {} for w in targets}
    for path in sorted(Path(corpus_dir).glob("decade_*.jsonl")):
        decade = int(path.stem.split("_")[1])
        if decade < MIN_DECADE:
            continue
        seen = {w: set() for w in targets}
        bucket: dict[str, list[dict[str, str]]] = {w: [] for w in targets}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                text = (json.loads(line).get("text") or "").strip()
                if not text:
                    continue
                for word, stem in targets.items():
                    if len(bucket[word]) >= per_decade:
                        continue
                    # At most ONE window per song per word (no same-song overlaps):
                    # take the first match whose context is non-empty, not blocklisted
                    # and not already seen, then move on to the next song.
                    for c in concordances(text, stem, window=window):
                        if not c["pre"] and not c["post"]:
                            continue  # need some context
                        blob = f"{c['pre']} {c['kw']} {c['post']}".lower()
                        if any(bad in blob for bad in _CONTEXT_BLOCKLIST):
                            continue
                        key = f"{c['pre']}|{c['kw']}|{c['post']}".lower()
                        if key in seen[word]:
                            continue
                        seen[word].add(key)
                        bucket[word].append(c)
                        break
        for word in targets:
            if bucket[word]:
                result[word][str(decade)] = bucket[word]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kwic.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def load_kwic(kwic_dir: Path | str = DEFAULT_KWIC_DIR) -> dict[str, Any]:
    """Load ``kwic.json`` (empty dict if absent)."""
    path = Path(kwic_dir) / "kwic.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """CLI: ``python -m src.lyrics.kwic``."""
    import argparse

    ap = argparse.ArgumentParser(description="Build KWIC concordances for the essay.")
    ap.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_KWIC_DIR))
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--per-decade", type=int, default=8)
    args = ap.parse_args()
    res = build_kwic(
        args.corpus_dir, args.out_dir, window=args.window, per_decade=args.per_decade
    )
    total = sum(len(v) for w in res.values() for v in w.values())
    out = Path(args.out_dir) / "kwic.json"
    print(f"KWIC: {len(res)} words, {total} decade-buckets -> {out}")


if __name__ == "__main__":
    main()
