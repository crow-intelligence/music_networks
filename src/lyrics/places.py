"""Extract Hungarian place names from the lyrics and aggregate them for the map.

Pipeline (two CLI steps; heavy geo deps are the opt-in ``geo`` extra):

1. ``build-gazetteer`` — stream a Geofabrik **Hungary OSM extract**
   (``hungary-latest.osm.pbf``) with :mod:`osmium`, keep ``place=*`` nodes
   (settlements + notable suburbs) with a name + coordinates, and write
   ``data/processed/places/gazetteer.json``.
2. ``aggregate`` — longest-match the gazetteer against each song's **surface**
   lyrics (``corpus.SongDoc.text`` — original casing/diacritics; the lemmatised
   tokens drop proper nouns, so they are useless here), count mentions per place
   and per decade, bin coordinates into **H3** cells, project them to 2-D, and
   write ``data/processed/places/places.json`` for the dashboard.

Matching is gazetteer longest-match on lowercased text (**diacritics kept** — so
``Eger`` ≠ ``egér``): a name matches a token when it is a prefix of the token and
the remainder is a standard Hungarian inflectional suffix (``Budapest``→
``Budapesten``/``budapesti``). Precision comes from a min-length floor, a
settlement-type filter, and an editable blocklist of common-word homographs
(``Pápa``=pope, ``Tata``=grandpa, …) — not from a capitalisation cue, so lowercase
mentions are still caught.

The pure helpers (:func:`tokenize`, :func:`strip_suffix`, :func:`match_at`,
:func:`extract_mentions`, :func:`project`) are doctested; the geo imports stay
inside the CLI functions so importing this module never needs the ``geo`` extra.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_PLACES_DIR = Path("data/processed/places")
DEFAULT_PBF = Path("data/misc/hungary-latest.osm.pbf")
PBF_URL = "https://download.geofabrik.de/europe/hungary-latest.osm.pbf"
DEFAULT_BLOCKLIST = Path("data/misc/place_blocklist.txt")
DEFAULT_OUTLINE = Path("data/misc/hungary.geojson")
# Curated gazetteers merged on top of the OSM settlements (gitignored, like the
# blocklist): Hungarian geographic features (lakes/rivers/regions — on the map) and
# foreign places (countries/cities — ranked list only, no coords).
DEFAULT_GEO = Path("data/misc/places_geo.json")
DEFAULT_FOREIGN = Path("data/misc/places_foreign.json")
# A place whose `kind` is one of these is "foreign" — listed, not put on the HU map.
FOREIGN_KINDS = frozenset({"ország", "város"})

# OSM ``place`` values we keep — settlements only. Streets/squares (the user's
# renamed edge cases — Moszkva tér, Vöröskatona utca) and Budapest-internal
# districts (suburb/quarter/…) are excluded. City/town are always kept; villages
# must clear ``MIN_VILLAGE_POP`` (the long tail of tiny villages is where
# common-word homographs — Velem, Visz, Boldog — cluster).
PLACE_TYPES = frozenset({"city", "town", "village"})
KEEP_ALWAYS = frozenset({"city", "town"})
MIN_NAME_LEN = 4  # drop 1–3 char names (Vác, Bő, Ág…) — too collision-prone
MIN_VILLAGE_POP = 1000  # keeps Hortobágy (1208); drops Velem (383), Visz (157)
H3_RES = 5  # ~8 km edge; a few hundred cells over Hungary

# Standard Hungarian inflectional suffixes a toponym's final token may carry. ``""``
# is exact match. Diacritics are kept throughout (we never deaccent).
SUFFIXES: frozenset[str] = frozenset(
    {
        "",
        # superessive / locative -n
        "n",
        "on",
        "en",
        "ön",
        "an",
        "ott",
        "ett",
        "ött",
        # inessive / illative / elative
        "ban",
        "ben",
        "ba",
        "be",
        "ból",
        "ből",
        # adessive / allative / ablative
        "nál",
        "nél",
        "hoz",
        "hez",
        "höz",
        "tól",
        "től",
        # sublative / delative
        "ra",
        "re",
        "ról",
        "ről",
        # dative / causal / terminative / instrumental
        "nak",
        "nek",
        "ért",
        "ig",
        "val",
        "vel",
        # NB: the adjectival -i ("budapesti", "pécsi") is deliberately EXCLUDED — it
        # collides with the very common placename+i SURNAMES (Szegedi, Pécsi, Ecsédi),
        # and the lowercase adjective is already cut by the capitalisation cue.
    }
)

_WORD = re.compile(r"[a-záéíóöőúüűA-ZÁÉÍÓÖŐÚÜŰ]+")
# NB: we deliberately do NOT model the stem-final a/e→á/é lengthening (Buda→Budá-n)
# — registering the lengthened base caused false positives (Sáta matched "Sátán" =
# Satan). We keep only the plain base, so we catch nominative + ``-i`` forms and a
# few suffixes that don't lengthen; the oblique cases of a/e-final names are missed
# (an acceptable precision trade for those mostly-homograph names: Baja, Pápa, …).


def tokenize(text: str) -> list[str]:
    """Split ``text`` into lowercased word tokens (Hungarian letters; accents kept).

    Examples:
        >>> tokenize("Budapesten, a Duna-parton jó.")
        ['budapesten', 'a', 'duna', 'parton', 'jó']
        >>> tokenize("")
        []
    """
    return [m.group(0).lower() for m in _WORD.finditer(text)]


def strip_suffix(token: str, base: str) -> bool:
    """Return whether ``token`` is ``base`` + a valid Hungarian suffix (or exact).

    Examples:
        >>> strip_suffix("budapesten", "budapest")
        True
        >>> strip_suffix("budapesti", "budapest")     # -i excluded (surname clash)
        False
        >>> strip_suffix("budapest", "budapest")
        True
        >>> strip_suffix("budapestx", "budapest")     # 'x' is not a suffix
        False
        >>> strip_suffix("szeged", "szegedre")         # token shorter than base
        False
    """
    if not token.startswith(base):
        return False
    return token[len(base) :] in SUFFIXES


@dataclass
class Gazetteer:
    """Compiled matcher: a char-trie of single-token names + a multiword index.

    Attributes:
        trie: Char-trie over single-token base names; ``"$"`` keys hold the
            place key at a terminal.
        multi: ``first_token -> [(token_tuple, place_key), …]`` (longest first)
            for the rare multi-word names.
        places: ``place_key -> {"name", "lat", "lon"}``.
    """

    trie: dict[str, Any] = field(default_factory=dict)
    multi: dict[str, list[tuple[tuple[str, ...], str]]] = field(default_factory=dict)
    places: dict[str, dict[str, Any]] = field(default_factory=dict)


def _trie_longest_base(token: str, trie: dict[str, Any]) -> str | None:
    """Longest terminal base that is a prefix of ``token`` with a valid suffix.

    Returns the matched place key, or ``None``.

    Examples:
        >>> g = build_index([
        ...     {"name": "Pécs", "lat": 0, "lon": 0, "place": "city"},
        ...     {"name": "Pécsvárad", "lat": 0, "lon": 0, "place": "town"},
        ... ], blocklist=set())
        >>> _trie_longest_base("pécsváradon", g.trie)   # longest wins
        'pécsvárad'
        >>> _trie_longest_base("pécsen", g.trie)        # Pécs + -en
        'pécs'
        >>> _trie_longest_base("pécsx", g.trie) is None
        True
    """
    node = trie
    best: str | None = None
    i, n = 0, len(token)
    while True:
        if "$" in node and token[i:] in SUFFIXES:
            best = node["$"]  # longest base so far (i increases each step)
        if i >= n:
            break
        node = node.get(token[i])
        if node is None:
            break
        i += 1
    return best


def build_index(
    entries: list[dict[str, Any]],
    *,
    blocklist: set[str],
    min_len: int = MIN_NAME_LEN,
    min_village_pop: int = MIN_VILLAGE_POP,
) -> Gazetteer:
    """Compile gazetteer ``entries`` into a :class:`Gazetteer` matcher.

    ``entries`` are ``{"name", "lat", "lon", "place", "pop"}``. Dropped: names
    shorter than ``min_len``; non-settlement ``place`` types; villages below
    ``min_village_pop``; names whose lowercased form is in ``blocklist``.

    Examples:
        >>> g = build_index([
        ...     {"name": "Szeged", "lat": 46.25, "lon": 20.15, "place": "city"},
        ...     {"name": "Bő", "lat": 0, "lon": 0, "place": "village"},  # too short
        ...     {"name": "Pápa", "lat": 0, "lon": 0, "place": "town"},   # blocked
        ...     {"name": "Velem", "lat": 0, "lon": 0, "place": "village", "pop": 383},
        ... ], blocklist={"pápa"})
        >>> sorted(g.places)
        ['szeged']
    """
    gaz = Gazetteer()
    for e in entries:
        name = (e.get("name") or "").strip()
        if not name or len(name) < min_len:
            continue
        # Curated entries (geographic features / foreign places) bypass the OSM
        # place-type + village-population gates; min-length + blocklist still apply.
        if not e.get("curated"):
            place = e.get("place")
            if place not in PLACE_TYPES:
                continue
            if place not in KEEP_ALWAYS and (e.get("pop") or 0) < min_village_pop:
                continue
        toks = tokenize(name)
        if not toks:
            continue
        key = " ".join(toks)
        if key in blocklist or name.lower() in blocklist:
            continue
        if key in gaz.places:  # dedup: keep the higher-population entry
            if (e.get("pop") or 0) <= (gaz.places[key].get("pop") or 0):
                continue
        gaz.places[key] = {
            "name": name,
            "lat": float(e.get("lat") or 0.0),
            "lon": float(e.get("lon") or 0.0),
            "pop": e.get("pop") or 0,
            "kind": e.get("kind") or "település",
        }
    # Build the matchers from the surviving places.
    for key in gaz.places:
        toks = key.split(" ")
        if len(toks) == 1:
            node = gaz.trie
            for ch in toks[0]:
                node = node.setdefault(ch, {})
            node["$"] = key
        else:
            gaz.multi.setdefault(toks[0], []).append((tuple(toks), key))
    for first in gaz.multi:
        gaz.multi[first].sort(key=lambda tk: -len(tk[0]))
    return gaz


def match_at(tokens: list[str], i: int, gaz: Gazetteer) -> tuple[str, int] | None:
    """Longest gazetteer match starting at ``tokens[i]`` → ``(place_key, span)``.

    Multi-word names match earlier tokens exactly and allow a suffix on the last;
    single-token names match via :func:`_trie_longest_base`. The longest span wins.

    Examples:
        >>> g = build_index([
        ...     {"name": "Szeged", "lat": 0, "lon": 0, "place": "city"},
        ...     {"name": "Hortobágy", "lat": 0, "lon": 0, "place": "village",
        ...      "pop": 1208},
        ... ], blocklist=set())
        >>> match_at(["szegedre", "megyek"], 0, g)
        ('szeged', 1)
        >>> match_at(["a", "hortobágyon"], 1, g)
        ('hortobágy', 1)
        >>> match_at(["nincs", "itt"], 0, g) is None
        True
    """
    best: tuple[str, int] | None = None
    # Multi-word (rare): longest candidate first.
    for toks, key in gaz.multi.get(tokens[i], ()):  # type: ignore[arg-type]
        k = len(toks)
        if i + k > len(tokens):
            continue
        if tuple(tokens[i : i + k - 1]) == toks[:-1] and strip_suffix(
            tokens[i + k - 1], toks[-1]
        ):
            if best is None or k > best[1]:
                best = (key, k)
    # Single-token.
    key = _trie_longest_base(tokens[i], gaz.trie)
    if key is not None and (best is None or best[1] < 1):
        best = (key, 1)
    return best


def extract_mentions(
    text: str, gaz: Gazetteer, *, require_cap: bool = True
) -> list[str]:
    """Return the place keys mentioned in ``text`` (non-overlapping longest-match).

    With ``require_cap`` (default) a match only counts when its first surface token
    is **Capitalised** — the proper-noun cue that separates the toponym ``Velem``
    from the everyday word ``velem`` ("with me"). Matching itself is case-insensitive
    (diacritics kept).

    Examples:
        >>> g = build_index([
        ...     {"name": "Szeged", "lat": 0, "lon": 0, "place": "city"},
        ...     {"name": "Pécs", "lat": 0, "lon": 0, "place": "city"},
        ... ], blocklist=set())
        >>> extract_mentions("Szegedről Pécsre, aztán haza.", g)
        ['szeged', 'pécs']
        >>> extract_mentions("ma szegedről jövök", g)   # lowercase → not a place
        []
    """
    surface = _WORD.findall(text)
    tokens = [t.lower() for t in surface]
    out: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        m = match_at(tokens, i, gaz)
        if m is None:
            i += 1
            continue
        if require_cap and not surface[i][:1].isupper():
            i += 1
            continue
        out.append(m[0])
        i += m[1]
    return out


def count_corpus(
    docs: list[Any], gaz: Gazetteer
) -> tuple[Counter[str], dict[int, Counter[str]]]:
    """Count place mentions over corpus ``docs`` → ``(overall, by_decade)``.

    ``docs`` are :class:`~src.lyrics.corpus.SongDoc` (uses ``.text`` + ``.decade``).
    """
    overall: Counter[str] = Counter()
    by_decade: dict[int, Counter[str]] = defaultdict(Counter)
    for doc in docs:
        for key in extract_mentions(doc.text, gaz):
            overall[key] += 1
            by_decade[doc.decade][key] += 1
    return overall, by_decade


def project(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Equirectangular projection (km) of ``(lat, lon)`` about ``(lat0, lon0)``.

    x grows east, y grows **north** (screen flips it later).

    Examples:
        >>> x, y = project(47.5, 19.0, 47.0, 19.0)
        >>> round(x, 1), round(y, 1)
        (0.0, 55.7)
    """
    r = 111.32  # km per degree latitude
    x = (lon - lon0) * r * math.cos(math.radians(lat0))
    y = (lat - lat0) * r
    return x, y


def _load_blocklist(path: Path) -> set[str]:
    """Read a blocklist file (one lowercased name/line, ``#`` comments)."""
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip().lower()
        if s:
            out.add(s)
    return out


# ---------------------------------------------------------------------------
# CLI step 1: gazetteer from the OSM PBF (needs the ``geo`` extra: osmium)
# ---------------------------------------------------------------------------
def fetch_pbf(dest: Path = DEFAULT_PBF, url: str = PBF_URL) -> Path:
    """Download the Geofabrik Hungary PBF to ``dest`` (skip if it already exists)."""
    if dest.exists():
        return dest
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} → {dest} …")
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
    return dest


def build_gazetteer(
    pbf_path: Path = DEFAULT_PBF, out_dir: Path = DEFAULT_PLACES_DIR
) -> Path:
    """Stream ``place=*`` nodes from the PBF → ``gazetteer.json``. Returns the path."""
    import osmium
    import osmium.filter
    import osmium.osm

    entries: list[dict[str, Any]] = []
    fp = osmium.FileProcessor(str(pbf_path)).with_filter(
        osmium.filter.KeyFilter("place")
    )
    for o in fp:
        if not isinstance(o, osmium.osm.Node):
            continue
        place = o.tags.get("place")
        name = o.tags.get("name")
        if not name or place not in PLACE_TYPES:
            continue
        try:
            lat, lon = o.location.lat, o.location.lon
        except (osmium.InvalidLocationError, AttributeError):
            continue
        pop_raw = re.sub(r"\D", "", o.tags.get("population") or "")
        entries.append(
            {
                "name": name,
                "lat": lat,
                "lon": lon,
                "place": place,
                "pop": int(pop_raw) if pop_raw else 0,
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "gazetteer.json"
    out.write_text(json.dumps(entries, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"Gazetteer: {len(entries)} place nodes → {out}")
    return out


# ---------------------------------------------------------------------------
# CLI step 2: match the corpus + H3-aggregate → places.json
# ---------------------------------------------------------------------------
def _load_outline(path: Path, lat0: float, lon0: float) -> list[list[list[float]]]:
    """Project a Hungary boundary GeoJSON to ground ``[x, y]`` rings (or ``[]``)."""
    if not path.exists():
        return []
    gj = json.loads(path.read_text(encoding="utf-8"))
    rings: list[list[list[float]]] = []

    def add(coords: list[Any]) -> None:
        pts = (project(c[1], c[0], lat0, lon0) for c in coords)
        rings.append([[round(x, 2), round(y, 2)] for x, y in pts])

    feats = gj.get("features", [gj]) if gj.get("type") == "FeatureCollection" else [gj]
    for f in feats:
        geom = f.get("geometry", f)
        t, co = geom.get("type"), geom.get("coordinates", [])
        if t == "Polygon":
            add(co[0])
        elif t == "MultiPolygon":
            for poly in co:
                add(poly[0])
    return rings


def aggregate(
    *,
    db_path: str = "data/music.db",
    corpus_dir: str = "data/processed/corpus",
    places_dir: Path = DEFAULT_PLACES_DIR,
    blocklist_path: Path = DEFAULT_BLOCKLIST,
    outline_path: Path = DEFAULT_OUTLINE,
    geo_path: Path = DEFAULT_GEO,
    foreign_path: Path = DEFAULT_FOREIGN,
    res: int = H3_RES,
) -> Path:
    """Match the gazetteer over the corpus, H3-bin mentions, write ``places.json``."""
    import h3

    from src.lyrics.corpus import load_corpus

    # Settlements (OSM) + curated geographic features + curated foreign places, all
    # in one index → one corpus pass; split afterwards by `kind`.
    entries = json.loads((places_dir / "gazetteer.json").read_text(encoding="utf-8"))
    for curated_path in (geo_path, foreign_path):
        if curated_path.exists():
            for e in json.loads(curated_path.read_text(encoding="utf-8")):
                entries.append({**e, "curated": True})
    gaz = build_index(entries, blocklist=_load_blocklist(blocklist_path))
    docs = load_corpus(corpus_dir)
    overall, by_decade = count_corpus(docs, gaz)
    if not overall:
        raise SystemExit("No place mentions found — check the gazetteer/corpus.")

    def is_foreign(key: str) -> bool:
        return gaz.places[key].get("kind") in FOREIGN_KINDS

    # H3 binning — HU places only (foreign places have no real coordinates).
    cell_total: Counter[str] = Counter()
    cell_names: dict[str, Counter[str]] = defaultdict(Counter)
    cell_dec: dict[str, Counter[int]] = defaultdict(Counter)
    cell_of: dict[str, str] = {}
    for key, cnt in overall.items():
        if is_foreign(key):
            continue
        p = gaz.places[key]
        cell = h3.latlng_to_cell(p["lat"], p["lon"], res)
        cell_of[key] = cell
        cell_total[cell] += cnt
        cell_names[cell][p["name"]] += cnt
    for dec, ctr in by_decade.items():
        for key, cnt in ctr.items():
            if key in cell_of:
                cell_dec[cell_of[key]][dec] += cnt

    # Projection centred on the mentioned-cell bounding box.
    cents = {c: h3.cell_to_latlng(c) for c in cell_total}
    lat0 = sum(la for la, _ in cents.values()) / len(cents)
    lon0 = sum(lo for _, lo in cents.values()) / len(cents)

    hexes = []
    for cell, cnt in cell_total.most_common():
        clat, clon = cents[cell]
        cx, cy = project(clat, clon, lat0, lon0)
        ring = (project(la, lo, lat0, lon0) for la, lo in h3.cell_to_boundary(cell))
        verts = [[round(x, 2), round(y, 2)] for x, y in ring]
        hexes.append(
            {
                "cx": round(cx, 2),
                "cy": round(cy, 2),
                "verts": verts,
                "count": cnt,
                "top": [n for n, _ in cell_names[cell].most_common(3)],
                "byDecade": {str(d): c for d, c in sorted(cell_dec[cell].items())},
            }
        )

    hu_ranked = [(k, c) for k, c in overall.most_common() if not is_foreign(k)]
    foreign_ranked = [(k, c) for k, c in overall.most_common() if is_foreign(k)]

    def row(k: str, c: int) -> dict[str, Any]:
        p = gaz.places[k]
        return {"name": p["name"], "kind": p["kind"], "count": c}

    payload = {
        "res": res,
        "maxCount": max(cell_total.values()),
        "hexes": hexes,
        "outline": _load_outline(outline_path, lat0, lon0),
        "top_places": [row(k, c) for k, c in hu_ranked[:40]],
        "by_decade": [
            {"decade": d, "n": int(sum(c for k, c in ctr.items() if not is_foreign(k)))}
            for d, ctr in sorted(by_decade.items())
        ],
        "foreign": {
            "top": [row(k, c) for k, c in foreign_ranked[:40]],
            "by_decade": [
                {"decade": d, "n": int(sum(c for k, c in ctr.items() if is_foreign(k)))}
                for d, ctr in sorted(by_decade.items())
            ],
        },
    }
    places_dir.mkdir(parents=True, exist_ok=True)
    out = places_dir / "places.json"
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(
        f"Places: {len(hu_ranked)} HU + {len(foreign_ranked)} foreign distinct, "
        f"{sum(overall.values())} mentions, {len(hexes)} H3 cells (res {res}) → {out}"
    )
    def fmt(rows: list[dict[str, Any]]) -> str:
        return ", ".join(f"{p['name']}({p['count']})" for p in rows[:15])

    print("Top HU:", fmt(payload["top_places"]))
    print("Top foreign:", fmt(payload["foreign"]["top"]))
    return out


def main() -> None:
    """CLI: ``python -m src.lyrics.places {build-gazetteer,aggregate}``."""
    import argparse

    ap = argparse.ArgumentParser(description="Hungarian place names in the lyrics.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("build-gazetteer", help="OSM PBF → gazetteer.json")
    g.add_argument("--pbf", default=str(DEFAULT_PBF))
    g.add_argument(
        "--no-fetch", action="store_true", help="don't auto-download the PBF"
    )
    a = sub.add_parser("aggregate", help="match corpus + H3 → places.json")
    a.add_argument("--db", default="data/music.db")
    a.add_argument("--corpus-dir", default="data/processed/corpus")
    a.add_argument("--res", type=int, default=H3_RES)
    args = ap.parse_args()

    if args.cmd == "build-gazetteer":
        pbf = Path(args.pbf)
        if not args.no_fetch:
            pbf = fetch_pbf(pbf)
        build_gazetteer(pbf)
    elif args.cmd == "aggregate":
        aggregate(db_path=args.db, corpus_dir=args.corpus_dir, res=args.res)


if __name__ == "__main__":
    main()
