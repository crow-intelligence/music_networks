"""Source per-decade era backdrops (music-playback media) from Wikimedia Commons.

One-off helper: for each decade the essay expresses its era through the dominant
music-playback *medium* (vinyl → reel-to-reel → turntable → cassette → CD → MP3 →
smartphone → earbuds). This queries the Commons API for a freely-licensed landscape
photo per medium, downloads a 1600px render into ``assets/images/media-<decade>.jpg``
(+ a ``hero.jpg``), and writes ``assets/images/credits.json`` with attribution.

The essay build only *copies* whatever is already in ``assets/images`` — this script
is what populates it. Re-run to refresh; commit nothing under ``data/``.

Run: ``uv run python -m src.essay.fetch_images``.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

_DEST = Path(__file__).with_name("assets") / "images"
_UA = {
    "User-Agent": (
        "music-networks-essay/1.0 "
        "(https://crowintelligence.org; zoltan.varju@crowintelligence.org)"
    )
}
_API = "https://commons.wikimedia.org/w/api.php"
# Free licences we accept (substring match on LicenseShortName, case-insensitive).
_FREE = ("cc0", "public domain", "cc by", "cc by-sa")
# Search terms per target file (first free landscape hit wins).
_TARGETS: dict[str, list[str]] = {
    "hero.jpg": ["vintage microphone dark studio", "old radio microphone"],
    "media-1950.jpg": ["vinyl record 45 rpm single", "gramophone record close up"],
    "media-1960.jpg": ["reel to reel tape recorder", "reel-to-reel tape machine"],
    "media-1970.jpg": ["turntable record player", "hifi turntable vinyl"],
    "media-1980.jpg": ["compact cassette tape", "audio cassette tape"],
    "media-1990.jpg": ["compact disc CD", "cd disc close up"],
    "media-2000.jpg": ["portable mp3 player", "portable music player device"],
    "media-2010.jpg": ["smartphone music headphones", "smartphone music app"],
    "media-2020.jpg": ["wireless earbuds", "true wireless earphones"],
}


def _strip(v: str) -> str:
    """Strip HTML tags/entities from a Commons metadata value."""
    return html.unescape(re.sub(r"<[^>]+>", "", v or "")).strip()


def _search(client: Any, term: str, limit: int = 10) -> list[dict[str, Any]]:
    """Return Commons imageinfo records for a search term."""
    r = client.get(
        _API,
        params={
            "action": "query",
            "generator": "search",
            "gsrsearch": term,
            "gsrnamespace": "6",
            "gsrlimit": str(limit),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": "1600",
            "format": "json",
        },
        timeout=30,
        headers=_UA,
    )
    r.raise_for_status()
    return list(r.json().get("query", {}).get("pages", {}).values())


def _pick(pages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose the best free, landscape, decent-resolution image from results."""
    best = None
    for p in pages:
        ii = (p.get("imageinfo") or [{}])[0]
        if not str(ii.get("mime", "")).startswith("image/"):
            continue
        em = ii.get("extmetadata", {})
        lic = str(em.get("LicenseShortName", {}).get("value", "")).lower()
        if not any(f in lic for f in _FREE):
            continue
        w, h = ii.get("width", 0), ii.get("height", 0)
        if w < 1000:
            continue
        landscape = w >= h * 1.15
        cand = {
            "title": p.get("title", ""),
            "url": ii.get("thumburl") or ii.get("url"),
            "page": ii.get("descriptionurl", ""),
            "license": _strip(em.get("LicenseShortName", {}).get("value", "")),
            "author": _strip(em.get("Artist", {}).get("value", "")) or "ismeretlen",
            "landscape": landscape,
        }
        if landscape:
            return cand
        best = best or cand  # fall back to first free portrait if no landscape
    return best


def main() -> None:
    """Fetch one era image per decade + a hero into ``assets/images``."""
    import httpx
    from PIL import Image

    _DEST.mkdir(parents=True, exist_ok=True)
    credits: dict[str, Any] = {}
    with httpx.Client(follow_redirects=True) as client:
        for fname, terms in _TARGETS.items():
            chosen = None
            for term in terms:
                chosen = _pick(_search(client, term))
                if chosen:
                    break
            if not chosen:
                print(f"  ! no free image for {fname} ({terms[0]})")
                continue
            img_bytes = client.get(chosen["url"], timeout=60, headers=_UA).content
            tmp = _DEST / (fname + ".tmp")
            tmp.write_bytes(img_bytes)
            im = Image.open(tmp).convert("RGB")
            if im.width > 1600:
                im = im.resize((1600, round(im.height * 1600 / im.width)))
            im.save(_DEST / fname, quality=82)
            tmp.unlink()
            credits[fname] = {
                "title": chosen["title"].removeprefix("File:").rsplit(".", 1)[0],
                "author": chosen["author"],
                "license": chosen["license"],
                "source": chosen["page"],
            }
            author = chosen["author"][:36]
            print(f"  ok {fname} <- {credits[fname]['license']} ({author})")
    (_DEST / "credits.json").write_text(
        json.dumps(credits, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(credits)} images + credits.json → {_DEST}")


if __name__ == "__main__":
    main()
