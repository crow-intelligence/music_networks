"""Allotaxonograph comparisons between decades (rank-turbulence divergence, keyflux).

For every pair of decades, render keyflux's **allotaxonometer** diamond — the Dodds
rank-rank map (which words sit where in each decade's frequency ranking) plus the
ranked list of words that drove the shift (rank-turbulence divergence) — to a cached
PNG under ``data/processed/allotax/``. The dashboard shows these with two decade
dropdowns, so any two decades can be compared. Matplotlib/keyflux imports stay inside
:func:`build_allotax` so importing this module is cheap.

Run: ``python -m src.lyrics.allotax`` (re-run when the corpus changes).
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

DEFAULT_ALLOTAX_DIR = Path("data/processed/allotax")
DEFAULT_CORPUS_DIR = "data/processed/corpus"
MIN_DECADE = 1950
ALPHA = 1 / 3

# Hungarian decade labels (match the dashboard's DECADE_HU).
_DECADE_HU = {
    1950: "1950-es",
    1960: "1960-as",
    1970: "1970-es",
    1980: "1980-as",
    1990: "1990-es",
    2000: "2000-es",
    2010: "2010-es",
    2020: "2020-as",
}


def decade_label(decade: int) -> str:
    """Hungarian decade label for a plot title.

    Examples:
        >>> decade_label(1980)
        '1980-as évek'
        >>> decade_label(2020)
        '2020-as évek'
    """
    return f"{_DECADE_HU.get(decade, str(decade))} évek"


def build_allotax(
    corpus_dir: str = DEFAULT_CORPUS_DIR,
    out_dir: Path | str = DEFAULT_ALLOTAX_DIR,
    *,
    alpha: float = ALPHA,
    min_decade: int = MIN_DECADE,
    dpi: int = 110,
) -> list[int]:
    """Render an allotaxonometer PNG for every decade pair; return the decades.

    Writes ``<out_dir>/<lo>_<hi>.png`` (earlier decade left) for each unordered pair
    of decades ``>= min_decade``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from keyflux import RankedList, allotaxonometer

    from src.lyrics.corpus import load_corpus
    from src.lyrics.decade_keywords import decade_counts

    counts = decade_counts(load_corpus(corpus_dir))
    decades = sorted(d for d in counts if d >= min_decade)
    ranked = {
        d: RankedList.from_counts(counts[d], label=decade_label(d)) for d in decades
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = list(combinations(decades, 2))
    for lo, hi in pairs:
        # keyflux puts list1 on the diamond's RIGHT, so pass (later, earlier) to get
        # the earlier decade on the LEFT and the later one on the RIGHT.
        fig = allotaxonometer(
            ranked[hi],
            ranked[lo],
            alpha=alpha,
            labels=(decade_label(hi), decade_label(lo)),
        )
        fig.savefig(out / f"{lo}_{hi}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    print(f"Allotaxonographs: {len(pairs)} decade pairs → {out}")
    return decades


def export_data(
    corpus_dir: str = DEFAULT_CORPUS_DIR,
    out_dir: Path | str = DEFAULT_ALLOTAX_DIR,
    *,
    alpha: float = ALPHA,
    min_decade: int = MIN_DECADE,
    top: int = 160,
) -> Path:
    """Export per-decade-pair rank-turbulence-divergence data → ``allotax_data.json``.

    For the interactive (HTML+JS) allotaxonograph: for each unordered decade pair,
    the ``top`` words by divergence contribution, each with its rank in the earlier
    (``r1``) and later (``r2``) decade, its contribution, and its side (``d``: -1 =
    earlier-favoured, +1 = later-favoured, 0 = shared).
    """
    from keyflux import RankedList, rtd

    from src.lyrics.corpus import load_corpus
    from src.lyrics.decade_keywords import decade_counts

    counts = decade_counts(load_corpus(corpus_dir))
    decades = sorted(d for d in counts if d >= min_decade)
    ranked = {d: RankedList.from_counts(counts[d]) for d in decades}
    dside = {"system1": -1, "system2": 1, "shared": 0}
    pairs: dict[str, Any] = {}
    for lo, hi in combinations(decades, 2):
        res = rtd(ranked[lo], ranked[hi], alpha=alpha)  # list1=lo (earlier)
        rows = sorted(res.contributions, key=lambda c: -abs(c.contribution))[:top]
        pairs[f"{lo}_{hi}"] = {
            "div": round(res.divergence, 4),
            "lo": lo,
            "hi": hi,
            "words": [
                {
                    "w": c.type,
                    "r1": round(c.rank1),
                    "r2": round(c.rank2),
                    "c": round(c.contribution, 6),
                    "d": dside.get(c.direction, 0),
                }
                for c in rows
            ],
        }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "allotax_data.json"
    path.write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")
    print(f"Allotax data: {len(pairs)} pairs × top {top} words → {path}")
    return path


def main() -> None:
    """CLI: ``python -m src.lyrics.allotax`` — writes the interactive data JSON.

    The dashboard draws the allotaxonograph client-side from this JSON. Pass
    ``--png`` to also render the (offline) static allotaxonometer PNGs.
    """
    import argparse

    ap = argparse.ArgumentParser(description="Export per-decade-pair allotaxonometry.")
    ap.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR)
    ap.add_argument("--out-dir", default=str(DEFAULT_ALLOTAX_DIR))
    ap.add_argument("--png", action="store_true", help="also render static PNGs")
    args = ap.parse_args()
    export_data(args.corpus_dir, Path(args.out_dir))
    if args.png:
        build_allotax(args.corpus_dir, Path(args.out_dir))


if __name__ == "__main__":
    main()
