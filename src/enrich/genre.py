"""Per-song genre tagging via Discogs.

MusicBrainz genre coverage for Hungarian acts is poor (most native artists carry
no genre/tag at all), but Discogs tags **every release** with a coarse ``genre``
(Rock / Pop / Hip Hop / Electronic / Folk, World, & Country / …) and finer
``style`` values — and the dating pass already proved Discogs matches Hungarian
records well. So genre comes from the *same* artist+track search the
:class:`~src.enrich.discogs.DiscogsDater` runs, reusing :class:`DiscogsClient`.

Like :mod:`src.lyrics.emotion`, predictions are written to a resumable
``per_song.jsonl`` artifact (resume skips song ids already present) rather than
mutating the DB — keeping the Genius ``Song.genre`` column untouched and the
provenance clean. :func:`build_artifacts` aggregates it by decade for the
dashboard.

The pure picker (:func:`pick_genre`) is deterministic and doctested; the network
client is imported/constructed only when a run actually happens.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.enrich.match import normalize, song_key
from src.lyrics.corpus import DEFAULT_CORPUS_DIR, load_corpus

DEFAULT_GENRE_DIR = Path("data/processed/genre")
_PER_SONG = "per_song.jsonl"


def load_token(raw: str | None) -> str | None:
    """Strip wrapping quotes/whitespace from a Discogs token.

    The project's ``.env`` stores ``discogs_token`` wrapped in quotes, which
    Discogs rejects when passed verbatim as a query parameter. This normalizes it.

    Args:
        raw: The raw token string (or ``None``).

    Returns:
        The cleaned token, or ``None`` if empty.

    Examples:
        >>> load_token('"abcDEF123"')
        'abcDEF123'
        >>> load_token("  tok  ")
        'tok'
        >>> load_token("") is None
        True
        >>> load_token(None) is None
        True
    """
    if not raw:
        return None
    cleaned = raw.strip().strip("\"'").strip()
    return cleaned or None


def pick_genre(
    artist: str, results: list[dict[str, Any]]
) -> tuple[str | None, list[str]]:
    """Choose a primary genre + styles for an artist from Discogs search results.

    Only results whose ``"<artist> - <release>"`` title contains our (normalized)
    artist count — the same guard :func:`~src.enrich.discogs.earliest_year` uses
    against loosely-related hits. The **most frequent** ``genre`` across those
    releases wins (ties broken by first appearance); the returned styles are the
    most common styles among the matching releases that carry the winning genre.

    Args:
        artist: Our performing act name.
        results: The ``results`` list from a Discogs release search.

    Returns:
        ``(genre, styles)`` — ``genre`` is ``None`` when nothing qualifies;
        ``styles`` is a (possibly empty) list ordered by frequency.

    Examples:
        >>> res = [
        ...     {"title": "Omega - Élő Omega", "genre": ["Rock"],
        ...      "style": ["Prog Rock"]},
        ...     {"title": "Omega - Gammapolis", "genre": ["Rock", "Pop"],
        ...      "style": ["Prog Rock", "Synth-pop"]},
        ...     {"title": "Other Band - Comp", "genre": ["Jazz"], "style": []},
        ... ]
        >>> pick_genre("Omega", res)
        ('Rock', ['Prog Rock', 'Synth-pop'])
        >>> pick_genre("Nobody", res)
        (None, [])
        >>> pick_genre("Omega", [])
        (None, [])
    """
    norm_artist = normalize(artist)
    if not norm_artist:
        return None, []
    matching = [r for r in results if norm_artist in normalize(r.get("title", ""))]
    genre_counts: Counter[str] = Counter()
    for r in matching:
        for g in r.get("genre") or []:
            genre_counts[g] += 1
    if not genre_counts:
        return None, []
    # Most frequent genre; Counter.most_common preserves insertion order on ties.
    top_genre = genre_counts.most_common(1)[0][0]
    style_counts: Counter[str] = Counter()
    for r in matching:
        if top_genre in (r.get("genre") or []):
            for s in r.get("style") or []:
                style_counts[s] += 1
    styles = [s for s, _ in style_counts.most_common()]
    return top_genre, styles


def _done_ids(path: Path) -> set[int]:
    """Return the set of song ids already present in ``per_song.jsonl``."""
    done: set[int] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["song_id"]))
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
    return done


def _candidate_rows(db_path: str, song_ids: set[int]) -> list[tuple[str, str, int]]:
    """Fetch ``(performer_name, title, song_id)`` for the given song ids.

    Args:
        db_path: SQLite path.
        song_ids: Restrict to these song ids (the corpus universe).

    Returns:
        One row per song that has both a performer and a title; a song with
        several performers keeps its first-seen credit.
    """
    from src.db import Performer, Song, SongPerformer, get_engine, make_session

    engine = get_engine(db_path)
    seen: set[int] = set()
    out: list[tuple[str, str, int]] = []
    try:
        with make_session(engine)() as session:
            rows = session.execute(
                Song.__table__.select()
                .with_only_columns(Performer.name, Song.title, Song.id)
                .select_from(Song.__table__)
                .join(SongPerformer.__table__, SongPerformer.song_id == Song.id)
                .join(Performer.__table__, Performer.id == SongPerformer.performer_id)
                .where(Song.lyrics.is_not(None))
            )
            for name, title, song_id in rows:
                if song_id not in song_ids or song_id in seen:
                    continue
                seen.add(song_id)
                out.append((name, title, song_id))
    finally:
        engine.dispose()
    return out


def group_candidates(
    rows: list[tuple[str, str, int]],
) -> dict[str, list[tuple[str, str, int]]]:
    """Group ``(name, title, song_id)`` rows by ``song_key`` (one search each).

    Args:
        rows: Candidate rows.

    Returns:
        ``song_key -> [(name, title, song_id), ...]``; every member of a group
        gets the group's genre (they are the same artist+title).

    Examples:
        >>> rows = [("Omega", "Petróleum", 1), ("omega", "petróleum", 2),
        ...         ("LGT", "Mindenki", 3)]
        >>> g = group_candidates(rows)
        >>> sorted(len(v) for v in g.values())
        [1, 2]
    """
    groups: dict[str, list[tuple[str, str, int]]] = {}
    for name, title, song_id in rows:
        groups.setdefault(song_key(name, title), []).append((name, title, song_id))
    return groups


def aggregate(per_song: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-song genres into overall + per-decade proportions.

    Args:
        per_song: Rows with ``decade`` and ``genre`` (``None``/absent = untagged,
            which is excluded from the proportions but counted in ``coverage``).

    Returns:
        ``{"genres": [...], "overall": {...}, "by_decade": [...]}`` where each
        scope carries ``n`` (tagged songs), ``coverage`` (tagged / total) and a
        ``share`` per genre. Genres are ordered by overall frequency.

    Examples:
        >>> rows = [
        ...     {"decade": 1970, "genre": "Rock"},
        ...     {"decade": 1970, "genre": "Pop"},
        ...     {"decade": 1970, "genre": None},
        ...     {"decade": 2010, "genre": "Hip Hop"},
        ... ]
        >>> agg = aggregate(rows)
        >>> agg["genres"][0] in {"Rock", "Pop", "Hip Hop"}
        True
        >>> d70 = next(d for d in agg["by_decade"] if d["decade"] == 1970)
        >>> d70["n"], round(d70["coverage"], 3)
        (2, 0.667)
        >>> d70["share"]["Rock"]
        0.5
    """
    genre_totals: Counter[str] = Counter()
    for row in per_song:
        if row.get("genre"):
            genre_totals[row["genre"]] += 1
    genres = [g for g, _ in genre_totals.most_common()]

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(rows)
        tagged = [r for r in rows if r.get("genre")]
        n = len(tagged)
        counts: Counter[str] = Counter(r["genre"] for r in tagged)
        return {
            "total": total,
            "n": n,
            "coverage": (n / total if total else 0.0),
            "share": {g: (counts.get(g, 0) / n if n else 0.0) for g in genres},
        }

    by_decade: dict[int, list[dict[str, Any]]] = {}
    for row in per_song:
        by_decade.setdefault(int(row["decade"]), []).append(row)

    return {
        "genres": genres,
        "overall": summarize(per_song),
        "by_decade": [
            {"decade": dec, **summarize(by_decade[dec])} for dec in sorted(by_decade)
        ],
    }


class DiscogsGenreTagger:
    """Tags corpus songs with a Discogs genre, resumable via the JSONL artifact."""

    def __init__(
        self,
        client,
        *,
        db_path: str = "data/music.db",
        corpus_dir: Path | str = DEFAULT_CORPUS_DIR,
        out_dir: Path | str = DEFAULT_GENRE_DIR,
    ):
        """Create a tagger.

        Args:
            client: A configured :class:`~src.enrich.discogs.DiscogsClient`.
            db_path: SQLite path (for performer/title lookups).
            corpus_dir: Corpus dir (defines which songs to tag + their decade).
            out_dir: Output dir for ``per_song.jsonl`` / ``genre.json``.
        """
        self._client = client
        self._db_path = db_path
        self._corpus_dir = Path(corpus_dir)
        self._out_dir = Path(out_dir)

    async def run(
        self, *, limit: int | None = None, resume: bool = True
    ) -> dict[str, int]:
        """Tag corpus songs, one Discogs search per artist|title group.

        Args:
            limit: Optional cap on searches (distinct keys) this run.
            resume: Skip song ids already present in ``per_song.jsonl``.

        Returns:
            Counts ``{"looked_up", "tagged", "untagged"}`` (songs for tag counts).
        """
        self._out_dir.mkdir(parents=True, exist_ok=True)
        per_song_path = self._out_dir / _PER_SONG

        docs = load_corpus(self._corpus_dir)
        decade_of = {d.song_id: d.decade for d in docs}
        done = _done_ids(per_song_path) if resume else set()
        wanted = {sid for sid in decade_of if sid not in done}
        rows = [
            (name, title, sid)
            for name, title, sid in _candidate_rows(self._db_path, wanted)
        ]
        groups = group_candidates(rows)

        counts = {"looked_up": 0, "tagged": 0, "untagged": 0}
        with per_song_path.open("a", encoding="utf-8") as handle:
            for members in groups.values():
                if limit is not None and counts["looked_up"] >= limit:
                    break
                counts["looked_up"] += 1
                artist, title, _ = members[0]
                try:
                    results = await self._client.search_release(artist, title)
                except Exception:  # noqa: BLE001 - leave undone; a re-run retries
                    continue
                genre, styles = pick_genre(artist, results)
                for _name, _title, song_id in members:
                    handle.write(
                        json.dumps(
                            {
                                "song_id": song_id,
                                "decade": decade_of[song_id],
                                "genre": genre,
                                "styles": styles,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    counts["tagged" if genre else "untagged"] += 1
                handle.flush()
        return counts


def build_artifacts(out_dir: Path | str = DEFAULT_GENRE_DIR) -> dict[str, Any]:
    """Aggregate ``per_song.jsonl`` into ``genre.json`` for the dashboard.

    Args:
        out_dir: Directory holding ``per_song.jsonl``.

    Returns:
        The aggregated dict (see :func:`aggregate`).
    """
    out_dir = Path(out_dir)
    per_song_path = out_dir / _PER_SONG
    rows: list[dict[str, Any]] = []
    if per_song_path.exists():
        with per_song_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:  # tolerate a torn final line if a run is still appending
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    agg = aggregate(rows)
    (out_dir / "genre.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return agg


def main() -> None:
    """CLI: ``run`` the Discogs genre tagger or ``aggregate`` the predictions."""
    import argparse
    import asyncio
    import os

    ap = argparse.ArgumentParser(description="Discogs genre tagging.")
    ap.add_argument("--db", default="data/music.db")
    ap.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_GENRE_DIR))
    sub = ap.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="tag corpus songs via Discogs (needs token)")
    r.add_argument("--token", default=None, help="Discogs token (or DISCOGS_TOKEN env)")
    r.add_argument("--delay", type=float, default=1.2)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--no-resume", action="store_true")

    sub.add_parser("aggregate", help="build genre.json from predictions")

    args = ap.parse_args()
    if args.command == "aggregate":
        agg = build_artifacts(args.out_dir)
        print(f"Aggregated {agg['overall']['n']} tagged songs -> genre.json")
        return

    token = load_token(args.token or os.environ.get("DISCOGS_TOKEN"))
    if not token:
        raise SystemExit("A Discogs token is required: pass --token or DISCOGS_TOKEN.")

    from src.enrich.discogs import DiscogsClient

    async def _go() -> None:
        client = DiscogsClient(token, delay=max(args.delay, 1.2))
        tagger = DiscogsGenreTagger(
            client, db_path=args.db, corpus_dir=args.corpus_dir, out_dir=args.out_dir
        )
        counts = await tagger.run(limit=args.limit, resume=not args.no_resume)
        print(
            f"Looked up {counts['looked_up']} keys; tagged {counts['tagged']} songs, "
            f"{counts['untagged']} without a genre."
        )
        agg = build_artifacts(args.out_dir)
        print(f"Aggregated {agg['overall']['n']} tagged songs -> genre.json")

    asyncio.run(_go())


if __name__ == "__main__":
    main()
