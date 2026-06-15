"""Artist popularity & cultural-salience stats for the dashboard.

Joins the two external popularity signals already in the DB for the §Népszerűség
views:

* **cultural salience** — average monthly Hungarian-Wikipedia pageviews per
  artist (a *present-day* "how remembered" proxy, from
  :mod:`src.enrich.pageviews`), and
* **commercial success** — Mahász year-end chart appearances, 2003–2022 (from
  :mod:`src.enrich.mahasz` / the ``chart_entry`` table).

The ranking/aggregation helpers are pure and doctested; the database read lives
in :func:`build_popularity`, so the module stays import-safe.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.enrich.match import normalize
from src.utils import year_to_decade

# Performer credit strings that are category buckets, not acts — kept out of the
# chart ranking.
_CHART_JUNK = frozenset(
    normalize(x)
    for x in (
        "filmzenék",
        "filmzene",
        "válogatás",
        "various",
        "various artists",
        "unknown artist",
        "magyar nóták",
        "népdal",
        "vers",
        "egyéb szövegek",
    )
)


def prettify_name(name: str) -> str:
    """Title-case an all-lowercase credit string; leave styled names alone.

    Scraped performer names are inconsistently cased. Only names that are
    *entirely* lowercase are title-cased (so ``halász judit`` becomes
    ``Halász Judit``), while intentional stylings (``DESH``, ``Tankcsapda``,
    ``LGT``) are left untouched.

    Args:
        name: A credit string.

    Returns:
        The display name.

    Examples:
        >>> prettify_name("halász judit")
        'Halász Judit'
        >>> prettify_name("ismerős arcok")
        'Ismerős Arcok'
        >>> prettify_name("DESH")
        'DESH'
        >>> prettify_name("Tankcsapda")
        'Tankcsapda'
    """
    if name and name == name.lower():
        return " ".join(w[:1].upper() + w[1:] for w in name.split(" ") if w)
    return name


# Lowercase connector words when title-casing an all-caps chart credit.
_CONNECTORS = frozenset(
    {"meg", "a", "az", "és", "s", "feat", "ft", "vs", "x", "the", "of", "and"}
)


def title_case_hu(name: str) -> str:
    """Title-case an all-caps credit string, keeping connector words lowercase.

    MAHASZ prints every act in capitals (``RÚZSA MAGDOLNA``); this renders it as
    a name (``Rúzsa Magdolna``), lowercasing Hungarian connectors (``meg``, ``a``,
    ``és`` …). A few stage-name acronyms come out slightly off (``DESH`` →
    ``Desh``), which is a fair trade for restoring the accents our scraped
    performer table had lost.

    Args:
        name: The (typically all-caps) credit string.

    Returns:
        The title-cased display name.

    Examples:
        >>> title_case_hu("RÚZSA MAGDOLNA")
        'Rúzsa Magdolna'
        >>> title_case_hu("KOWALSKY MEG A VEGA")
        'Kowalsky meg a Vega'
        >>> title_case_hu("ÁKOS")
        'Ákos'
    """
    words = name.split()
    out = []
    for i, w in enumerate(words):
        if i > 0 and w.lower().strip(".") in _CONNECTORS:
            out.append(w.lower())
        else:
            out.append(w[:1].upper() + w[1:].lower())
    return " ".join(out)


def salience_ranking(
    rows: list[tuple[str, float | None]], top: int = 20
) -> list[dict[str, Any]]:
    """Rank artists by cultural-salience score (pageviews), highest first.

    Args:
        rows: ``(name, score)`` pairs; ``None``/zero scores are dropped.
        top: How many to return.

    Returns:
        Rows ``{"name", "views"}`` (rounded), most-viewed first.

    Examples:
        >>> salience_ranking([("A", 100.0), ("B", 900.4), ("C", None)], top=2)
        [{'name': 'B', 'views': 900}, {'name': 'A', 'views': 100}]
    """
    ranked = sorted(
        ((n, float(s)) for n, s in rows if s), key=lambda x: x[1], reverse=True
    )
    return [{"name": prettify_name(n), "views": round(v)} for n, v in ranked[:top]]


def chart_ranking(
    rows: list[tuple[str, int | None]], top: int = 20
) -> list[dict[str, Any]]:
    """Rank acts by MAHASZ chart appearances (ties broken by best peak rank).

    The credit string is the chart's **own** printed name (``artist_raw``), which
    keeps its accents — unlike some matched performer names in our scraped table.
    Category-bucket credits (compilations, "filmzenék", …) are excluded.

    Args:
        rows: ``(artist_raw, rank)`` per chart entry.
        top: How many acts to return.

    Returns:
        Rows ``{"name", "entries", "peak"}`` — number of appearances and the best
        (lowest) rank reached — most appearances first; names title-cased.

    Examples:
        >>> r = chart_ranking([("OMEGA", 3), ("OMEGA", 1), ("BEATRICE", 5),
        ...                    ("VÁLOGATÁS", 1)], top=2)
        >>> [(x["name"], x["entries"], x["peak"]) for x in r]
        [('Omega', 2, 1), ('Beatrice', 1, 5)]
    """
    accents = lambda s: sum(ord(ch) > 127 for ch in s)  # noqa: E731
    by: dict[str, list[Any]] = {}
    for name, rank in rows:
        disp = name.rstrip(":").strip()
        key = normalize(disp)
        if not key or key in _CHART_JUNK:
            continue
        rec = by.setdefault(key, [disp, 0, None])
        if accents(disp) > accents(rec[0]):  # keep the best-accented spelling
            rec[0] = disp
        rec[1] += 1
        if rank is not None:
            rec[2] = rank if rec[2] is None else min(rec[2], rank)
    ranked = sorted(by.values(), key=lambda r: (r[1], -(r[2] or 9999)), reverse=True)
    return [
        {"name": title_case_hu(d), "entries": c, "peak": b} for d, c, b in ranked[:top]
    ]


def salience_by_era(
    rows: list[tuple[float | None, str | None]],
    *,
    min_decade: int = 1950,
    min_artists: int = 8,
) -> list[dict[str, Any]]:
    """Average salience by the artist's birth/formation decade.

    A rough "which generation is most remembered today" signal: artists are
    bucketed by the decade of their ``begin_date`` (birth for a person,
    formation for a group). Decades below ``min_decade`` or with fewer than
    ``min_artists`` are dropped.

    Args:
        rows: ``(score, begin_date)`` pairs (``begin_date`` is ``YYYY-…``).
        min_decade: Earliest decade to keep.
        min_artists: Minimum artists for a decade to be reported.

    Returns:
        Rows ``{"decade", "mean_views", "n"}``, decades ascending.

    Examples:
        >>> rows = [(100.0, "1965-01-01"), (300.0, "1968"), (50.0, "1300")]
        >>> salience_by_era(rows, min_artists=2)
        [{'decade': 1960, 'mean_views': 200, 'n': 2}]
    """
    buckets: dict[int, list[float]] = defaultdict(list)
    for score, begin in rows:
        if not score or not begin:
            continue
        head = begin[:4]
        if not head.isdigit():
            continue
        decade = year_to_decade(int(head))
        if decade >= min_decade:
            buckets[decade].append(float(score))
    out = []
    for decade in sorted(buckets):
        vals = buckets[decade]
        if len(vals) >= min_artists:
            out.append(
                {
                    "decade": decade,
                    "mean_views": round(sum(vals) / len(vals)),
                    "n": len(vals),
                }
            )
    return out


def build_popularity(db_path: str = "data/music.db") -> dict[str, Any]:
    """Read the DB and assemble the popularity blocks for the dashboard.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        ``{"salience": [...], "charts": [...], "era": [...]}`` (empty lists if the
        enrichment hasn't run).
    """
    from sqlalchemy import select

    from src.db import Artist, ChartEntry, get_engine, make_session

    engine = get_engine(db_path)
    try:
        with make_session(engine)() as session:
            artists = session.execute(
                select(
                    Artist.name, Artist.avg_monthly_pageviews, Artist.begin_date
                ).where(Artist.avg_monthly_pageviews.is_not(None))
            ).all()
            charts = session.execute(
                select(ChartEntry.artist_raw, ChartEntry.rank).where(
                    ChartEntry.performer_id.is_not(None)
                )
            ).all()
    finally:
        engine.dispose()

    return {
        "salience": salience_ranking([(n, v) for n, v, _ in artists]),
        "charts": chart_ranking([(n, r) for n, r in charts]),
        "era": salience_by_era([(v, b) for _, v, b in artists]),
    }
