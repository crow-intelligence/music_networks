"""Lexical-diversity metrics for the per-decade corpus.

Plain type–token ratio (TTR = types / tokens) is **not** comparable across our
decades, because it falls as a text grows and the decades differ in size by an
order of magnitude — a raw TTR would mostly rank decades by corpus size. We use
**MATTR** (Moving-Average Type–Token Ratio; Covington & McFall, 2010): the mean
TTR over a sliding fixed-length window, which is (approximately) independent of
total length and reads as "what fraction of words are distinct within any
``window``-token span".

Import-safe: pure functions only, no work at import time.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.lyrics.corpus import SongDoc


def mattr(tokens: list[str], window: int = 100) -> float:
    """Moving-Average Type–Token Ratio of a token sequence.

    The mean over every length-``window`` contiguous window of
    (distinct tokens / window). For sequences shorter than ``window`` this
    degrades gracefully to the plain TTR of the whole sequence.

    Args:
        tokens: The token sequence (order matters).
        window: Sliding-window length in tokens.

    Returns:
        A diversity score in ``[0, 1]`` (``0.0`` for an empty sequence).

    Examples:
        >>> mattr(["a", "b", "a", "b"], window=2)   # every 2-window is all-distinct
        1.0
        >>> mattr(["a", "a", "a"], window=2)         # each 2-window has 1 type
        0.5
        >>> mattr(["a", "b", "c"], window=10)        # shorter than window -> plain TTR
        1.0
        >>> mattr([], window=5)
        0.0
    """
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(tokens)) / n

    counts: Counter[str] = Counter()
    distinct = 0
    for tok in tokens[:window]:
        if counts[tok] == 0:
            distinct += 1
        counts[tok] += 1
    total = distinct
    n_windows = 1
    for i in range(window, n):
        add, rem = tokens[i], tokens[i - window]
        if counts[add] == 0:
            distinct += 1
        counts[add] += 1
        counts[rem] -= 1
        if counts[rem] == 0:
            distinct -= 1
        total += distinct
        n_windows += 1
    return total / (n_windows * window)


def lexical_diversity_by_decade(
    docs: Iterable[SongDoc], window: int = 100
) -> dict[int, float]:
    """Compute MATTR per decade over the concatenated decade token stream.

    Args:
        docs: Phase-0 corpus documents (each with ``decade`` and ``tokens``).
        window: MATTR window length.

    Returns:
        ``{decade: mattr}`` for every decade present in ``docs``.

    Examples:
        >>> class D:
        ...     def __init__(self, decade, tokens):
        ...         self.decade, self.tokens = decade, tokens
        >>> out = lexical_diversity_by_decade(
        ...     [D(1990, ["a", "a"]), D(1990, ["b"]), D(2000, ["x", "y"])],
        ...     window=2)
        >>> sorted(out)
        [1990, 2000]
        >>> out[2000]
        1.0
    """
    by_decade: dict[int, list[str]] = {}
    for doc in docs:
        by_decade.setdefault(doc.decade, []).extend(doc.tokens)
    return {dec: mattr(toks, window=window) for dec, toks in by_decade.items()}
