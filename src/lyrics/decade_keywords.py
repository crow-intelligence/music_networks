"""Per-decade keyness: which words are distinctive of each decade.

Consumes the Phase-0 corpus (:mod:`src.lyrics.corpus`, already lemmatized,
stop-worded and n-gram-folded) and scores every term in a decade against the
*rest* of the corpus with two complementary statistics:

* **log-likelihood (G²)** — significance of the frequency difference
  (Rayson & Garside), and
* **log-ratio** — effect size, ``log2`` of the relative-frequency ratio
  (Hardie), with +0.5 smoothing when a term is absent from the reference.

Both are pure, deterministic and doctested. The module is import-safe (no
top-level execution); :func:`main` writes per-decade TSVs.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.lyrics.corpus import DEFAULT_CORPUS_DIR, SongDoc, load_corpus

DEFAULT_KEYWORDS_DIR = Path("data/processed/keywords")


def log_likelihood(a: int, b: int, c: int, d: int) -> float:
    """Log-likelihood (G²) for a term's target vs reference frequency.

    Args:
        a: Term frequency in the target corpus.
        b: Term frequency in the reference corpus.
        c: Total tokens in the target corpus.
        d: Total tokens in the reference corpus.

    Returns:
        The G² statistic (``0.0`` when the term is equally frequent in both).

    Examples:
        >>> log_likelihood(10, 10, 100, 100)
        0.0
        >>> round(log_likelihood(20, 0, 100, 100), 4)
        27.7259
    """
    e1 = c * (a + b) / (c + d)
    e2 = d * (a + b) / (c + d)
    term1 = a * math.log(a / e1) if a > 0 else 0.0
    term2 = b * math.log(b / e2) if b > 0 else 0.0
    return 2 * (term1 + term2)


def log_ratio(a: int, b: int, c: int, d: int) -> float:
    """Log-ratio effect size: ``log2`` of the relative-frequency ratio.

    A term absent from the reference (``b == 0``) is smoothed with +0.5 to keep
    the ratio finite (Hardie's recommendation).

    Args:
        a: Term frequency in the target corpus.
        b: Term frequency in the reference corpus.
        c: Total tokens in the target corpus.
        d: Total tokens in the reference corpus.

    Returns:
        The base-2 log of ``(a/c) / (b/d)``.

    Examples:
        >>> log_ratio(20, 10, 100, 100)
        1.0
        >>> log_ratio(10, 20, 100, 100)
        -1.0
        >>> round(log_ratio(10, 0, 100, 100), 4)
        4.3219
    """
    b_smoothed = b if b > 0 else 0.5
    target_rel = a / c
    ref_rel = b_smoothed / d
    return math.log2(target_rel / ref_rel)


@dataclass
class Keyword:
    """A distinctive term for a decade.

    Attributes:
        term: The (lemmatized, possibly n-gram) term.
        log_likelihood: G² significance vs the rest of the corpus.
        log_ratio: Effect size (positive = over-represented this decade).
        freq: Raw frequency in the decade.
    """

    term: str
    log_likelihood: float
    log_ratio: float
    freq: int


def decade_counts(docs: list[SongDoc]) -> dict[int, Counter]:
    """Aggregate token frequencies per decade.

    Args:
        docs: The Phase-0 corpus.

    Returns:
        ``decade -> Counter(token -> frequency)``.

    Examples:
        >>> d = [SongDoc(1, 1990, "", ["a", "a", "b"]),
        ...      SongDoc(2, 2000, "", ["b", "c"])]
        >>> counts = decade_counts(d)
        >>> counts[1990]["a"], counts[2000]["c"]
        (2, 1)
    """
    out: dict[int, Counter] = {}
    for doc in docs:
        out.setdefault(doc.decade, Counter()).update(doc.tokens)
    return out


def keyness(
    target: Counter, reference: Counter, *, min_freq: int = 5, top_n: int | None = None
) -> list[Keyword]:
    """Score target terms against a reference corpus, ranked by log-likelihood.

    Only over-represented terms (positive log-ratio) at or above ``min_freq`` in
    the target are returned.

    Args:
        target: Token counts for the decade.
        reference: Token counts for the rest of the corpus.
        min_freq: Minimum target frequency to consider a term.
        top_n: Optional cap on the number of keywords returned.

    Returns:
        Keywords sorted by descending log-likelihood.
    """
    c = sum(target.values())
    d = sum(reference.values())
    keywords: list[Keyword] = []
    for term, a in target.items():
        if a < min_freq:
            continue
        b = reference.get(term, 0)
        lr = log_ratio(a, b, c, d)
        if lr <= 0:
            continue
        keywords.append(Keyword(term, log_likelihood(a, b, c, d), lr, a))
    keywords.sort(key=lambda k: k.log_likelihood, reverse=True)
    return keywords[:top_n] if top_n is not None else keywords


def keywords_by_decade(
    docs: list[SongDoc], *, min_freq: int = 5, top_n: int | None = 100
) -> dict[int, list[Keyword]]:
    """Compute distinctive keywords for every decade (vs the rest of the corpus).

    Args:
        docs: The Phase-0 corpus.
        min_freq: Minimum target frequency per term.
        top_n: Keywords kept per decade.

    Returns:
        ``decade -> list[Keyword]`` (each list ranked by log-likelihood).
    """
    counts = decade_counts(docs)
    totals: Counter = Counter()
    for counter in counts.values():
        totals.update(counter)
    result: dict[int, list[Keyword]] = {}
    for decade, target in counts.items():
        reference = totals - target  # rest-of-corpus counts
        result[decade] = keyness(target, reference, min_freq=min_freq, top_n=top_n)
    return result


def main() -> None:
    """Compute per-decade keywords from the cached corpus and write TSVs."""
    docs = load_corpus(DEFAULT_CORPUS_DIR)
    if not docs:
        raise SystemExit(
            f"No corpus at {DEFAULT_CORPUS_DIR}; "
            "run `python -m src.lyrics.corpus` first."
        )
    DEFAULT_KEYWORDS_DIR.mkdir(parents=True, exist_ok=True)
    by_decade = keywords_by_decade(docs)
    for decade, keywords in sorted(by_decade.items()):
        path = DEFAULT_KEYWORDS_DIR / f"decade_{decade}.tsv"
        with path.open("w", encoding="utf-8") as handle:
            handle.write("term\tlog_likelihood\tlog_ratio\tfreq\n")
            for kw in keywords:
                handle.write(
                    f"{kw.term}\t{kw.log_likelihood:.4f}\t{kw.log_ratio:.4f}\t{kw.freq}\n"
                )
        print(f"  {decade}s : {len(keywords)} keywords -> {path}")


if __name__ == "__main__":
    main()
