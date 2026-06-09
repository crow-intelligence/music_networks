"""Build-time renderer for the per-decade *distinctive word* clouds.

Each decade in the keyness block becomes one word-cloud **image** whose word
sizes encode the distinctiveness score (G², the log-likelihood from
:mod:`src.lyrics.decade_keywords`) and whose typeface evokes that era. The font
is rasterised into the PNG at build time, so the shipped dashboard needs **no
runtime web fonts** for the clouds — only the static images under
``data/dashboard/assets/clouds/``.

The clouds are the *expressive* view of section §3; the ranked list rendered
next to them in the template stays the *analytical* (and screen-reader / no-JS)
fallback, so no information lives only in the image.

This module is import-safe (no work at import time) and depends only on
``wordcloud`` / ``Pillow`` plus stdlib, so it can be collected for doctests.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# Era → display font (file in the build-only font dir) + a human label for the
# caption. Trimmed to the decades the corpus actually covers; any decade without
# an entry falls back to ``_FALLBACK_FONT`` (the page-serif-like Playfair).
DECADE_FONTS: dict[int, tuple[str, str]] = {
    1960: ("RozhaOne-Regular.ttf", "Rozha One — beat / pszichedelikus"),
    1970: ("Pacifico-Regular.ttf", "Pacifico — diszkó / funk"),
    1980: ("Audiowide-Regular.ttf", "Audiowide — new wave / szintipop"),
    1990: ("SpecialElite-Regular.ttf", "Special Elite — írógép / grunge"),
    2000: ("Exo2.ttf", "Exo 2 — Y2K / techno"),
    2010: ("Poppins-Regular.ttf", "Poppins — letisztult indie"),
    2020: ("Anton-Regular.ttf", "Anton — trap / kortárs"),
}
_FALLBACK_FONT = "PlayfairDisplay.ttf"

# Per-decade ink colors (subset of the site categorical palette). All are dark
# enough to stay AA-contrast on the cream background; words never rely on color
# alone (size + the ranked list carry the data).
# All hues are AA-contrast (>= 4.5:1) on the cream background.
DECADE_INK: dict[int, list[str]] = {
    1960: ["#7a1f1f", "#aa3333", "#856404"],
    1970: ["#a8431a", "#856404", "#7a1f1f"],
    1980: ["#34568b", "#7d3c98", "#2a6f7a"],
    1990: ["#2c2a24", "#5a5346", "#486b2f"],
    2000: ["#2a6f7a", "#34568b", "#2f5d50"],
    2010: ["#2f5d50", "#486b2f", "#34568b"],
    2020: ["#1a1a17", "#7a1f1f", "#2c2a24"],
}
_DEFAULT_INK = ["#7a1f1f", "#2f5d50", "#34568b"]
_PAPER = "#f7f5ef"


def cloud_weights(
    terms: list[dict[str, Any]], word_count: int = 40
) -> dict[str, float]:
    """Map the top distinctive terms to word-cloud weights via a ``sqrt`` scale.

    Font area in the cloud scales with the weight, so taking ``sqrt`` of the G²
    log-likelihood keeps the strongest word from dwarfing the rest. ``_`` joins
    in n-gram lemmas are turned back into spaces for display.

    Args:
        terms: Per-decade term dicts (``term`` + ``log_likelihood``), strongest
            first, as produced by :func:`src.dashboard.build.keyness_block`.
        word_count: How many top terms to include.

    Returns:
        Mapping of display term → positive weight, suitable for
        ``WordCloud.generate_from_frequencies``.

    Examples:
        >>> w = cloud_weights(
        ...     [{"term": "szív", "log_likelihood": 100.0},
        ...      {"term": "nagy_szerelem", "log_likelihood": 25.0}],
        ...     word_count=40)
        >>> sorted(w)
        ['nagy szerelem', 'szív']
        >>> w["szív"] == 10.0 and w["nagy szerelem"] == 5.0
        True
    """
    out: dict[str, float] = {}
    for t in terms[:word_count]:
        score = max(0.0, float(t.get("log_likelihood", 0.0)))
        weight = math.sqrt(score)
        if weight > 0:
            out[str(t["term"]).replace("_", " ")] = weight
    return out


def _ink_func(decade: int):
    """Return a deterministic wordcloud ``color_func`` for ``decade``."""
    palette = DECADE_INK.get(decade, _DEFAULT_INK)

    def color(word: str, **_: Any) -> str:
        return palette[sum(ord(c) for c in word) % len(palette)]

    return color


def render_clouds(
    keyness: list[dict[str, Any]],
    out_dir: Path | str,
    *,
    font_dir: Path | str,
    word_count: int = 40,
    width: int = 900,
    height: int = 460,
    scale: int = 2,
) -> dict[int, dict[str, Any]]:
    """Render one era-styled word-cloud PNG per decade and return a manifest.

    Args:
        keyness: The keyness block (one row per decade with ``terms``).
        out_dir: Directory the PNGs are written to (created if absent).
        font_dir: Build-only directory holding the era ``.ttf`` files.
        word_count: Max words per cloud (default 40).
        width: Cloud width in px before ``scale``.
        height: Cloud height in px before ``scale``.
        scale: Super-sampling factor for crisp text on HiDPI screens.

    Returns:
        ``{decade: {"file": <name>.png, "n": int, "font": str,
        "width": int, "height": int}}`` for decades that produced a cloud.
        ``file`` is the bare filename (the template prefixes ``assets/clouds/``).
    """
    from wordcloud import WordCloud

    out_dir, font_dir = Path(out_dir), Path(font_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[int, dict[str, Any]] = {}

    for row in keyness:
        decade = int(row["decade"])
        weights = cloud_weights(row.get("terms", []), word_count)
        if not weights:
            continue
        font_name, font_label = DECADE_FONTS.get(
            decade, (_FALLBACK_FONT, "Playfair Display")
        )
        font_path = font_dir / font_name
        if not font_path.exists():
            font_path = font_dir / _FALLBACK_FONT
            font_label = "Playfair Display"
        wc = WordCloud(
            font_path=str(font_path),
            width=width,
            height=height,
            scale=scale,
            background_color=_PAPER,
            mode="RGB",
            prefer_horizontal=0.88,  # mostly 0°, some 90° — never random angles
            max_words=word_count,
            relative_scaling=0.5,
            min_font_size=14,
            margin=4,
            color_func=_ink_func(decade),
            random_state=42,  # deterministic layout across builds
        )
        wc.generate_from_frequencies(weights)
        fname = f"decade_{decade}.png"
        wc.to_file(str(out_dir / fname))
        manifest[decade] = {
            "file": fname,
            "n": len(weights),
            "font": font_label,
            "width": width * scale,
            "height": height * scale,
        }
    return manifest


def main() -> None:
    """Render clouds from a keyness JSON file (debug/standalone use)."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Render decade word-cloud images.")
    ap.add_argument("keyness_json", help="JSON file: the keyness block (list).")
    ap.add_argument("--out-dir", default="data/dashboard/assets/clouds")
    ap.add_argument("--font-dir", default="src/dashboard/assets/build-fonts")
    ap.add_argument("--word-count", type=int, default=40)
    args = ap.parse_args()

    keyness = json.loads(Path(args.keyness_json).read_text(encoding="utf-8"))
    manifest = render_clouds(
        keyness,
        args.out_dir,
        font_dir=args.font_dir,
        word_count=args.word_count,
    )
    print(f"Rendered {len(manifest)} clouds -> {args.out_dir}")


if __name__ == "__main__":
    main()
