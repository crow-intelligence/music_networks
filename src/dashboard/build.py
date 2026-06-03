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
from pathlib import Path
from typing import Any

from src.lyrics.decade_keywords import Keyword
from src.lyrics.stats import DecadeStat

DEFAULT_OUT = Path("data/dashboard/index.html")
DEFAULT_TOPICS_DIR = Path("data/processed/topics")
DEFAULT_USAGE_DIR = Path("data/processed/usage")
_TEMPLATE = Path(__file__).with_name("template.html")

# Decades below this song count are flagged as statistically thin in the UI.
LOW_N_SONGS = 50


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


def _load_json(path: Path, default: Any) -> Any:
    """Return parsed JSON at ``path``, or ``default`` if the file is absent."""
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def assemble_data(
    *,
    overview: list[dict[str, Any]],
    keyness: list[dict[str, Any]],
    topic_info: list[dict[str, Any]],
    topics_over_time: list[dict[str, Any]],
    usage: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the single ``DATA`` blob the template renders.

    Topic ``-1`` (BERTopic outliers) is dropped from the topic blocks.

    Args:
        overview: From :func:`decade_overview`.
        keyness: From :func:`keyness_block`.
        topic_info: Topic table (``topic_info.json``).
        topics_over_time: Per-(topic, decade) frequencies.
        usage: The diachronic-analysis blocks (may be empty).

    Returns:
        The dashboard data dict.

    Examples:
        >>> info = [{"topic_id": -1, "name": "out", "keywords": [], "size": 1},
        ...         {"topic_id": 0, "name": "Szerelem", "keywords": ["a"], "size": 9}]
        >>> ot = [{"topic_id": -1, "decade": 1990, "frequency": 1},
        ...       {"topic_id": 0, "decade": 1990, "frequency": 5}]
        >>> data = assemble_data(overview=[], keyness=[], topic_info=info,
        ...                      topics_over_time=ot, usage={})
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
        "topics": {"info": info, "over_time": over_time},
        "usage": usage,
    }


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


def build(
    *,
    db_path: str = "data/music.db",
    corpus_dir: str = "data/processed/corpus",
    topics_dir: Path | str = DEFAULT_TOPICS_DIR,
    usage_dir: Path | str = DEFAULT_USAGE_DIR,
    out_path: Path | str = DEFAULT_OUT,
) -> Path:
    """Build the dashboard HTML from all cached artifacts.

    Args:
        db_path: SQLite DB for descriptive stats.
        corpus_dir: Corpus JSONL dir for keyness.
        topics_dir: Topic artifacts dir.
        usage_dir: Diachronic-usage artifacts dir.
        out_path: Output HTML path.

    Returns:
        The written HTML path.
    """
    from src.lyrics.corpus import load_corpus
    from src.lyrics.decade_keywords import keywords_by_decade
    from src.lyrics.stats import collect_decade_stats

    topics_dir, usage_dir, out_path = (
        Path(topics_dir),
        Path(usage_dir),
        Path(out_path),
    )

    overview = decade_overview(collect_decade_stats(db_path))
    docs = load_corpus(corpus_dir)
    keyness = keyness_block(keywords_by_decade(docs)) if docs else []
    data = assemble_data(
        overview=overview,
        keyness=keyness,
        topic_info=_load_json(topics_dir / "topic_info.json", []),
        topics_over_time=_load_json(topics_dir / "topics_over_time.json", []),
        usage=load_usage(usage_dir),
    )

    html = render_html(data, _TEMPLATE.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    path = build(
        db_path=args.db,
        corpus_dir=args.corpus_dir,
        topics_dir=args.topics_dir,
        usage_dir=args.usage_dir,
        out_path=args.out,
    )
    print(f"Dashboard written -> {path.resolve()}")


if __name__ == "__main__":
    main()
