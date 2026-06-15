"""Phase 4 — build the static lyrics-analysis dashboard.

Assembles every cached analysis artifact into one ``DATA`` JSON blob and injects
it into :mod:`~src.dashboard` ``template.html``, producing a **single
self-contained** ``data/dashboard/index.html`` (no server, no build step, no JS
dependencies — charts are drawn as inline SVG by vanilla JS in the template).
This mirrors the crowintelligence.org/analysis dashboard, adapted from
presidential *periods* to song *decades*.

Inputs:

* per-decade descriptive stats — :func:`src.lyrics.stats.collect_decade_stats`,
* per-decade keyness — :func:`src.lyrics.decade_keywords.keywords_by_decade`,
* topics — ``data/processed/topics/{topic_info,topics_over_time}.json``,
* diachronic usage change — ``data/processed/usage/*.json``.

Only aggregate numbers / keywords are emitted — never full lyrics (copyright).
The data-shaping helpers are pure and doctested; this module imports only
project + stdlib code (no heavy ML deps), so it stays import-safe.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from src.lyrics.decade_keywords import Keyword
from src.lyrics.stats import DecadeStat

DEFAULT_OUT = Path("data/dashboard/index.html")
DEFAULT_TOPICS_DIR = Path("data/processed/topics")
DEFAULT_USAGE_DIR = Path("data/processed/usage")
DEFAULT_NETWORKS_DIR = Path("data/processed/networks")
DEFAULT_EMOTION_DIR = Path("data/processed/emotion")
DEFAULT_GENRE_DIR = Path("data/processed/genre")
DEFAULT_COLLAB_DIR = Path("data/processed/collab")
DEFAULT_RHYME_DIR = Path("data/processed/rhyme")
_TEMPLATE = Path(__file__).with_name("template.html")
_ASSETS_SRC = Path(__file__).with_name("assets")
DEFAULT_FONT_DIR = _ASSETS_SRC / "build-fonts"

# Decades below this song count are flagged as statistically thin in the UI.
LOW_N_SONGS = 50

# How many distinctive terms to surface per decade (word cloud + ranked list).
WORD_COUNT = 40

# The pre-1950 decades are too sparse to be meaningful (single/double-digit song
# counts); the dashboard shows only decades from this year on. The 1950s clears
# the low-N bar (~150 corpus songs) and is included; underlying artifacts for the
# dropped earlier decades are left intact.
MIN_DECADE = 1950


def decade_overview(stats: list[DecadeStat]) -> list[dict[str, Any]]:
    """Turn per-decade stats into JSON rows, flagging thin decades.

    Args:
        stats: Per-decade counts (see :func:`src.lyrics.stats.collect_decade_stats`).

    Returns:
        One row per decade with counts and a ``low_n`` flag, sorted by decade.

    Examples:
        >>> from src.lyrics.stats import DecadeStat
        >>> rows = decade_overview([DecadeStat(1990, 200, 80, 50, 40),
        ...                         DecadeStat(1960, 10, 5, 4, 3)])
        >>> [r["decade"] for r in rows]
        [1960, 1990]
        >>> rows[0]["low_n"], rows[1]["low_n"]
        (True, False)
    """
    return [
        {
            "decade": s.decade,
            "songs": s.songs,
            "performers": s.performers,
            "authors": s.authors,
            "composers": s.composers,
            "low_n": s.songs < LOW_N_SONGS,
        }
        for s in sorted(stats, key=lambda s: s.decade)
    ]


def keyness_block(
    by_decade: dict[int, list[Keyword]], top: int = 15
) -> list[dict[str, Any]]:
    """Reduce per-decade keyness to the top over-represented terms per decade.

    Keeps only positively keyed terms (``log_ratio > 0``), strongest first.

    Args:
        by_decade: Output of :func:`src.lyrics.decade_keywords.keywords_by_decade`.
        top: How many terms to keep per decade.

    Returns:
        One row per decade with its top distinctive terms.

    Examples:
        >>> from src.lyrics.decade_keywords import Keyword
        >>> kw = {1990: [Keyword("a", 50.0, 2.0, 9), Keyword("b", 80.0, -1.0, 3),
        ...               Keyword("c", 30.0, 1.0, 4)]}
        >>> block = keyness_block(kw, top=2)
        >>> [t["term"] for t in block[0]["terms"]]
        ['a', 'c']
    """
    out: list[dict[str, Any]] = []
    for decade in sorted(by_decade):
        keyed = [k for k in by_decade[decade] if k.log_ratio > 0]
        keyed.sort(key=lambda k: k.log_likelihood, reverse=True)
        out.append(
            {
                "decade": decade,
                "terms": [
                    {
                        "term": k.term,
                        "log_likelihood": round(k.log_likelihood, 2),
                        "log_ratio": round(k.log_ratio, 2),
                        "freq": k.freq,
                    }
                    for k in keyed[:top]
                ],
            }
        )
    return out


def frequency_block(counts: dict[int, Counter], top: int = 15) -> list[dict[str, Any]]:
    """Reduce per-decade token counts to the most frequent content words.

    The complement of :func:`keyness_block`: instead of what is *distinctive* of
    a decade, this is simply what is *most said* in it (the corpus tokens are
    already content-word-filtered and n-gram-folded, so function words are gone).

    Args:
        counts: Per-decade token counts, from
            :func:`src.lyrics.decade_keywords.decade_counts`.
        top: How many terms to keep per decade.

    Returns:
        One row per decade with its most frequent terms (``term`` + ``freq``),
        most frequent first.

    Examples:
        >>> from collections import Counter
        >>> counts = {1990: Counter({"szív": 9, "szeret": 5, "kép": 1})}
        >>> block = frequency_block(counts, top=2)
        >>> [(t["term"], t["freq"]) for t in block[0]["terms"]]
        [('szív', 9), ('szeret', 5)]
    """
    out: list[dict[str, Any]] = []
    for decade in sorted(counts):
        out.append(
            {
                "decade": decade,
                "terms": [
                    {"term": term, "freq": freq}
                    for term, freq in counts[decade].most_common(top)
                ],
            }
        )
    return out


def _load_json(path: Path, default: Any) -> Any:
    """Return parsed JSON at ``path``, or ``default`` if the file is absent."""
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def assemble_data(
    *,
    overview: list[dict[str, Any]],
    keyness: list[dict[str, Any]],
    frequency: list[dict[str, Any]],
    topic_info: list[dict[str, Any]],
    topics_over_time: list[dict[str, Any]],
    usage: dict[str, Any],
    networks: dict[str, Any] | None = None,
    emotion: dict[str, Any] | None = None,
    genre: dict[str, Any] | None = None,
    collab: dict[str, Any] | None = None,
    rhyme: dict[str, Any] | None = None,
    associations: dict[str, Any] | None = None,
    popularity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the single ``DATA`` blob the template renders.

    Topic ``-1`` (BERTopic outliers) is dropped from the topic blocks.

    Args:
        overview: From :func:`decade_overview`.
        keyness: From :func:`keyness_block`.
        frequency: From :func:`frequency_block`.
        topic_info: Topic table (``topic_info.json``).
        topics_over_time: Per-(topic, decade) frequencies.
        usage: The diachronic-analysis blocks (may be empty).
        networks: The per-decade word-graph blocks (may be empty/``None``).
        emotion: The aggregated emotion blocks (may be empty/``None``).
        genre: The aggregated genre blocks (may be empty/``None``).
        collab: The collaboration-network blocks (may be empty/``None``).
        rhyme: The rhyme-analysis blocks (may be empty/``None``).
        associations: The genre association cross-tabs (may be empty/``None``).
        popularity: The artist popularity/salience blocks (may be empty/``None``).

    Returns:
        The dashboard data dict.

    Examples:
        >>> info = [{"topic_id": -1, "name": "out", "keywords": [], "size": 1},
        ...         {"topic_id": 0, "name": "Szerelem", "keywords": ["a"], "size": 9}]
        >>> ot = [{"topic_id": -1, "decade": 1990, "frequency": 1},
        ...       {"topic_id": 0, "decade": 1990, "frequency": 5}]
        >>> data = assemble_data(overview=[], keyness=[], frequency=[],
        ...                      topic_info=info, topics_over_time=ot, usage={})
        >>> [t["topic_id"] for t in data["topics"]["info"]]
        [0]
        >>> data["topics"]["over_time"][0]["topic_id"]
        0
    """
    info = [t for t in topic_info if t.get("topic_id") != -1]
    over_time = [t for t in topics_over_time if t.get("topic_id") != -1]
    return {
        "overview": overview,
        "keyness": keyness,
        "frequency": frequency,
        "topics": {"info": info, "over_time": over_time},
        "usage": usage,
        "networks": networks or {},
        "emotion": emotion or {},
        "genre": genre or {},
        "collab": collab or {},
        "rhyme": rhyme or {},
        "associations": associations or {},
        "popularity": popularity or {},
    }


def emit_assets(
    out_dir: Path,
    keyness: list[dict[str, Any]],
    frequency: list[dict[str, Any]],
    *,
    font_dir: Path = DEFAULT_FONT_DIR,
    word_count: int = WORD_COUNT,
) -> None:
    """Render the decade word-cloud images and copy shipped assets.

    Renders, per decade, **two** era-styled cloud PNGs under
    ``<out_dir>/assets/clouds`` — a *most-frequent* cloud (``freq_decade_*.png``)
    and a *distinctive* cloud (``decade_*.png``) — copies the self-hosted runtime
    fonts next to the HTML, and annotates each keyness/frequency row **in place**
    with its ``cloud`` entry (so the shared dicts the template serialises gain
    the image reference).

    Args:
        out_dir: The dashboard output directory (where ``index.html`` lives).
        keyness: The keyness block; rows are mutated to add ``cloud``.
        frequency: The frequency block; rows are mutated to add ``cloud``.
        font_dir: Build-only directory of era ``.ttf`` files.
        word_count: Max words per cloud.
    """
    from src.dashboard.clouds import render_clouds

    assets = out_dir / "assets"
    src_fonts = _ASSETS_SRC / "fonts"
    if src_fonts.is_dir():
        dst_fonts = assets / "fonts"
        dst_fonts.mkdir(parents=True, exist_ok=True)
        for font in src_fonts.iterdir():
            if font.is_file():
                shutil.copy2(font, dst_fonts / font.name)

    # Clear stale clouds first so dropped decades don't leave orphan images.
    clouds_dir = assets / "clouds"
    if clouds_dir.exists():
        shutil.rmtree(clouds_dir)

    for rows, weight_key, prefix in (
        (keyness, "log_likelihood", "decade_"),
        (frequency, "freq", "freq_decade_"),
    ):
        manifest = render_clouds(
            rows,
            clouds_dir,
            font_dir=font_dir,
            word_count=word_count,
            weight_key=weight_key,
            prefix=prefix,
        )
        for row in rows:
            cloud = manifest.get(int(row["decade"]))
            if cloud is not None:
                row["cloud"] = cloud


def render_html(data: dict[str, Any], template: str) -> str:
    """Inject ``data`` into the template's ``/*__DATA__*/`` placeholder.

    Args:
        data: The assembled dashboard data.
        template: The HTML template text.

    Returns:
        The self-contained HTML document.

    Examples:
        >>> render_html({"a": 1}, "x = /*__DATA__*/;")
        'x = {"a": 1};'
    """
    blob = json.dumps(data, ensure_ascii=False)
    return template.replace("/*__DATA__*/", blob)


def load_usage(usage_dir: Path) -> dict[str, Any]:
    """Load the diachronic-usage artifacts into one dict (empty if missing).

    Args:
        usage_dir: Directory with the ``usage_change`` JSON files.

    Returns:
        A dict with the usage blocks; ``{}`` if no ``meta.json`` is present.
    """
    if not (usage_dir / "meta.json").exists():
        return {}
    meta = _load_json(usage_dir / "meta.json", {})
    return {
        **meta,
        "semantic_change": _load_json(usage_dir / "semantic_change.json", []),
        "shift_heatmap": _load_json(usage_dir / "shift_heatmap.json", {}),
        "cumulative_drift": _load_json(usage_dir / "cumulative_drift.json", []),
        "nearest_neighbors": _load_json(usage_dir / "nearest_neighbors.json", []),
        "vocab_stats": _load_json(usage_dir / "vocab_stats.json", []),
        "alignment_stats": _load_json(usage_dir / "alignment_stats.json", []),
    }


def load_networks(networks_dir: Path) -> dict[str, Any]:
    """Load the per-decade kenon word-graph artifacts into one dict.

    Args:
        networks_dir: Directory with ``index.json`` + ``decade_*.json`` graphs.

    Returns:
        ``{}`` if no ``index.json`` is present, otherwise
        ``{"decades": [...], "graphs": {decade: <node-link dict>}}``.
    """
    index = networks_dir / "index.json"
    if not index.exists():
        return {}
    meta = _load_json(index, {})
    graphs: dict[str, Any] = {}
    for decade in meta.get("decades", []):
        graph = _load_json(networks_dir / f"decade_{decade}.json", None)
        if graph is not None:
            graphs[str(decade)] = graph
    return {"decades": meta.get("decades", []), "graphs": graphs}


def _recombine(
    decades: list[dict[str, Any]], keys: list[str], weight: str, denom: str
) -> dict[str, Any]:
    """N-weighted recombination of per-decade summaries into one overall summary.

    Each ``mean``/``share`` map is a weighted average over the decade ``weight``
    counts, so it is divided by the **weight** total — keeping the property that a
    ``share`` map sums to 1.0 overall, exactly as it does per decade. When
    ``denom`` differs from ``weight`` (genre: tagged ``n`` vs all ``total``) a
    separate ``coverage`` = weight/denom ratio is added. The overall reflects
    exactly the decades kept after :data:`MIN_DECADE` filtering.

    Args:
        decades: Per-decade summary dicts.
        keys: Which map fields to recombine (e.g. ``["mean", "share"]``).
        weight: The per-decade count field to weight a map by (and divide by).
        denom: The per-decade count field for the ``coverage`` denominator.

    Returns:
        A summary dict with the summed integer counts and recombined maps.
    """
    total = {f: sum(int(d.get(f, 0)) for d in decades) for f in {weight, denom}}
    out: dict[str, Any] = dict(total)
    div = total[weight] or 1
    for key in keys:
        labels = decades[0].get(key, {}) if decades else {}
        out[key] = {
            lab: sum(
                d.get(key, {}).get(lab, 0.0) * int(d.get(weight, 0)) for d in decades
            )
            / div
            for lab in labels
        }
    if denom != weight:  # genre carries a coverage ratio
        out["coverage"] = (total[weight] / total[denom]) if total[denom] else 0.0
    return out


def load_emotion(emotion_dir: Path) -> dict[str, Any]:
    """Load aggregated emotion blocks, scoped to :data:`MIN_DECADE`+.

    Args:
        emotion_dir: Directory with ``emotion.json`` (see
            :func:`src.lyrics.emotion.build_artifacts`).

    Returns:
        ``{}`` if absent, else ``{"labels", "overall", "by_decade"}`` with
        decades below :data:`MIN_DECADE` dropped and ``overall`` recomputed.
    """
    agg = _load_json(emotion_dir / "emotion.json", {})
    if not agg:
        return {}
    kept = [d for d in agg.get("by_decade", []) if d["decade"] >= MIN_DECADE]
    overall = _recombine(kept, ["mean", "share"], "n", "n") if kept else {}
    return {"labels": agg.get("labels", []), "overall": overall, "by_decade": kept}


def load_genre(genre_dir: Path) -> dict[str, Any]:
    """Load aggregated genre blocks, scoped to :data:`MIN_DECADE`+.

    Args:
        genre_dir: Directory with ``genre.json`` (see
            :func:`src.enrich.genre.build_artifacts`).

    Returns:
        ``{}`` if absent, else ``{"genres", "overall", "by_decade"}`` with
        decades below :data:`MIN_DECADE` dropped and ``overall`` recomputed.
    """
    agg = _load_json(genre_dir / "genre.json", {})
    if not agg:
        return {}
    kept = [d for d in agg.get("by_decade", []) if d["decade"] >= MIN_DECADE]
    overall = _recombine(kept, ["share"], "n", "total") if kept else {}
    return {"genres": agg.get("genres", []), "overall": overall, "by_decade": kept}


def load_collab(collab_dir: Path) -> dict[str, Any]:
    """Load the collaboration-network summary, scoped to :data:`MIN_DECADE`+.

    Args:
        collab_dir: Directory with ``summary.json`` (see
            :func:`src.graphs.collab.build_networks`): centrality rankings +
            community breakdown per slice (``aggregate`` + ``decade_<d>``).

    Returns:
        ``{}`` if absent, else ``{"alpha", "slices"}`` with per-decade slices
        below :data:`MIN_DECADE` dropped.
    """
    summary = _load_json(collab_dir / "summary.json", {})
    if not summary:
        return {}
    kept: dict[str, Any] = {}
    for name, sl in summary.get("slices", {}).items():
        if name == "aggregate":
            kept[name] = sl
        elif name.startswith("decade_") and int(name.split("_")[1]) >= MIN_DECADE:
            kept[name] = sl
    return {"alpha": summary.get("alpha"), "slices": kept}


def load_rhyme(rhyme_dir: Path) -> dict[str, Any]:
    """Load the rhyme-analysis summary, scoped to :data:`MIN_DECADE`+.

    Args:
        rhyme_dir: Directory with ``rhyme.json`` (see
            :func:`src.lyrics.rhyme.build_artifacts`).

    Returns:
        ``{}`` if absent, else the aggregate with per-decade rows below
        :data:`MIN_DECADE` dropped.
    """
    agg = _load_json(rhyme_dir / "rhyme.json", {})
    if not agg:
        return {}
    agg["by_decade"] = [
        d for d in agg.get("by_decade", []) if d["decade"] >= MIN_DECADE
    ]
    return agg


def load_popularity(db_path: str) -> dict[str, Any]:
    """Load artist popularity/salience blocks from the DB (empty if unenriched).

    Args:
        db_path: SQLite DB with the ``artist``/``chart_entry`` enrichment.

    Returns:
        ``{"salience", "charts", "era"}`` (see
        :func:`src.lyrics.popularity.build_popularity`); ``{}`` if nothing scored.
    """
    from src.lyrics.popularity import build_popularity

    pop = build_popularity(db_path)
    return pop if pop.get("salience") or pop.get("charts") else {}


def load_associations(
    docs: list[Any],
    *,
    emotion_dir: Path,
    genre_dir: Path,
    topics_dir: Path,
) -> dict[str, Any]:
    """Build the genre association cross-tabs (emotion / topic / lexical div).

    Args:
        docs: The loaded corpus (for the lexical-diversity-by-genre view).
        emotion_dir: Emotion artifacts dir.
        genre_dir: Genre artifacts dir.
        topics_dir: Topic artifacts dir (``assignments.json`` + ``topic_info``).

    Returns:
        ``{"emotion", "lexdiv", "topic"}`` blocks (each present only when its
        inputs exist); ``{}`` if there are no genre tags.
    """
    from src.lyrics.associations import (
        category_by_genre,
        load_emotion_labels,
        load_genre_map,
        mattr_by_genre,
    )
    from src.lyrics.emotion import EMOTIONS

    genre_of = load_genre_map(genre_dir)
    if not genre_of:
        return {}
    out: dict[str, Any] = {}

    emotion_labels = load_emotion_labels(emotion_dir)
    if emotion_labels:
        out["emotion"] = category_by_genre(emotion_labels, genre_of, list(EMOTIONS))

    if docs:
        out["lexdiv"] = mattr_by_genre({d.song_id: d.tokens for d in docs}, genre_of)

    assignments = _load_json(topics_dir / "assignments.json", {})
    topic_info = _load_json(topics_dir / "topic_info.json", [])
    if assignments and topic_info:
        name_of = {
            t["topic_id"]: t["name"] for t in topic_info if t.get("topic_id") != -1
        }
        # Categories = the largest topics (a 9×N heatmap stays readable).
        ranked = sorted(
            (t for t in topic_info if t.get("topic_id") != -1),
            key=lambda t: t.get("size", 0),
            reverse=True,
        )
        cats = [t["name"] for t in ranked[:14]]
        topic_label = {
            int(sid): name_of[tid]
            for sid, tid in assignments.items()
            if int(tid) in name_of
        }
        out["topic"] = category_by_genre(topic_label, genre_of, cats)
    return out


def build(
    *,
    db_path: str = "data/music.db",
    corpus_dir: str = "data/processed/corpus",
    topics_dir: Path | str = DEFAULT_TOPICS_DIR,
    usage_dir: Path | str = DEFAULT_USAGE_DIR,
    networks_dir: Path | str = DEFAULT_NETWORKS_DIR,
    emotion_dir: Path | str = DEFAULT_EMOTION_DIR,
    genre_dir: Path | str = DEFAULT_GENRE_DIR,
    collab_dir: Path | str = DEFAULT_COLLAB_DIR,
    rhyme_dir: Path | str = DEFAULT_RHYME_DIR,
    out_path: Path | str = DEFAULT_OUT,
) -> Path:
    """Build the dashboard HTML from all cached artifacts.

    Args:
        db_path: SQLite DB for descriptive stats.
        corpus_dir: Corpus JSONL dir for keyness.
        topics_dir: Topic artifacts dir.
        usage_dir: Diachronic-usage artifacts dir.
        networks_dir: Per-decade word-graph artifacts dir.
        emotion_dir: Emotion artifacts dir (``emotion.json``).
        genre_dir: Genre artifacts dir (``genre.json``).
        collab_dir: Collaboration-network artifacts dir (``summary.json``).
        rhyme_dir: Rhyme-analysis artifacts dir (``rhyme.json``).
        out_path: Output HTML path.

    Returns:
        The written HTML path.
    """
    from src.lyrics.corpus import load_corpus
    from src.lyrics.decade_keywords import decade_counts, keywords_by_decade
    from src.lyrics.stats import collect_decade_stats

    topics_dir, usage_dir, networks_dir, out_path = (
        Path(topics_dir),
        Path(usage_dir),
        Path(networks_dir),
        Path(out_path),
    )

    overview = [
        r
        for r in decade_overview(collect_decade_stats(db_path))
        if r["decade"] >= MIN_DECADE
    ]
    docs = load_corpus(corpus_dir)
    all_keyness = (
        keyness_block(keywords_by_decade(docs), top=WORD_COUNT) if docs else []
    )
    keyness = [r for r in all_keyness if r["decade"] >= MIN_DECADE]
    all_frequency = frequency_block(decade_counts(docs), top=WORD_COUNT) if docs else []
    frequency = [r for r in all_frequency if r["decade"] >= MIN_DECADE]
    topics_over_time = [
        r
        for r in _load_json(topics_dir / "topics_over_time.json", [])
        if r.get("decade", 0) >= MIN_DECADE
    ]

    usage = load_usage(usage_dir)
    if docs:
        from src.lyrics.diversity import lexical_diversity_by_decade

        lexdiv = lexical_diversity_by_decade(docs)
        for row in usage.get("vocab_stats", []):
            score = lexdiv.get(row.get("decade"))
            if score is not None:
                row["lexdiv"] = round(score, 3)

    networks = load_networks(networks_dir)
    if networks.get("decades"):
        networks["decades"] = [d for d in networks["decades"] if d >= MIN_DECADE]
        networks["graphs"] = {
            k: v for k, v in networks["graphs"].items() if int(k) >= MIN_DECADE
        }

    emotion = load_emotion(Path(emotion_dir))
    genre = load_genre(Path(genre_dir))
    collab = load_collab(Path(collab_dir))
    rhyme = load_rhyme(Path(rhyme_dir))
    associations = load_associations(
        docs,
        emotion_dir=Path(emotion_dir),
        genre_dir=Path(genre_dir),
        topics_dir=topics_dir,
    )
    popularity = load_popularity(db_path)

    data = assemble_data(
        overview=overview,
        keyness=keyness,
        frequency=frequency,
        topic_info=_load_json(topics_dir / "topic_info.json", []),
        topics_over_time=topics_over_time,
        usage=usage,
        networks=networks,
        emotion=emotion,
        genre=genre,
        collab=collab,
        rhyme=rhyme,
        associations=associations,
        popularity=popularity,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Render cloud images + copy assets; this annotates keyness/frequency rows in
    # place (shared dict refs in `data`) before the blob is serialised.
    emit_assets(out_path.parent, keyness, frequency)

    html = render_html(data, _TEMPLATE.read_text(encoding="utf-8"))
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main() -> None:
    """Build the dashboard and report where it was written."""
    import argparse

    parser = argparse.ArgumentParser(description="Build the static dashboard.")
    parser.add_argument("--db", default="data/music.db")
    parser.add_argument("--corpus-dir", default="data/processed/corpus")
    parser.add_argument("--topics-dir", default=str(DEFAULT_TOPICS_DIR))
    parser.add_argument("--usage-dir", default=str(DEFAULT_USAGE_DIR))
    parser.add_argument("--networks-dir", default=str(DEFAULT_NETWORKS_DIR))
    parser.add_argument("--emotion-dir", default=str(DEFAULT_EMOTION_DIR))
    parser.add_argument("--genre-dir", default=str(DEFAULT_GENRE_DIR))
    parser.add_argument("--collab-dir", default=str(DEFAULT_COLLAB_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    path = build(
        db_path=args.db,
        corpus_dir=args.corpus_dir,
        topics_dir=args.topics_dir,
        usage_dir=args.usage_dir,
        networks_dir=args.networks_dir,
        emotion_dir=args.emotion_dir,
        genre_dir=args.genre_dir,
        collab_dir=args.collab_dir,
        out_path=args.out,
    )
    folder = path.parent.resolve()
    clouds = len(list((folder / "assets" / "clouds").glob("*.png")))
    print(f"Dashboard written -> {path.resolve()}")
    print(
        f"Standalone site ({clouds} clouds + self-hosted fonts, no external deps) "
        f"-> copy this folder to your web root:\n  {folder}"
    )


if __name__ == "__main__":
    main()
