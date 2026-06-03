"""Descriptive per-decade statistics over the dated lyrics corpus.

Powers the dashboard's "Data overview" page and doubles as a quick standalone
report. For each decade it counts songs and the distinct **performers**,
**authors**, **composers** and **works** credited that decade, plus overall
dated-coverage. Pure SQL aggregation over :mod:`src.db` — no NLP/ML deps.

A song's decade comes from its best available year (``first_release_year`` if
set, else the scraped page ``year``), bucketed by :func:`src.utils.year_to_decade`.
Only songs with non-empty lyrics and a usable year are counted.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.db import (
    Song,
    SongAuthor,
    SongComposer,
    SongPerformer,
    get_engine,
    make_session,
)
from src.utils import year_to_decade


def song_decade(first_release_year: int | None, year: int | None) -> int | None:
    """Return a song's decade from its best available year, or ``None``.

    Prefers the authoritative ``first_release_year`` over the scraped page
    ``year``; returns ``None`` when neither is a usable four-digit year.

    Args:
        first_release_year: Authoritative first-release year, or ``None``.
        year: Scraped page year, or ``None``.

    Returns:
        The decade (e.g. ``1980``), or ``None`` if undated.

    Examples:
        >>> song_decade(1987, 1990)
        1980
        >>> song_decade(None, 1995)
        1990
        >>> song_decade(None, None) is None
        True
        >>> song_decade(None, 12) is None
        True
    """
    best = first_release_year if first_release_year is not None else year
    if best is None:
        return None
    decade = year_to_decade(best)
    return decade or None


@dataclass
class DecadeStat:
    """Per-decade corpus counts.

    Attributes:
        decade: The decade (e.g. ``1990``).
        songs: Dated lyrics songs in the decade.
        performers: Distinct performers credited that decade.
        authors: Distinct authors (lyricists) credited that decade.
        composers: Distinct composers credited that decade.
    """

    decade: int
    songs: int
    performers: int
    authors: int
    composers: int


def _decade_col():
    """SQL expression bucketing a song's best year into its decade."""
    # ``year - (year % 10)`` floors to the decade using integer ops only.
    # (SQLAlchemy 2.0 renders ``/`` as true/float division, which would not
    # floor — so we avoid division here.)
    best = func.coalesce(Song.first_release_year, Song.year)
    return (best - best % 10).label("decade")


def _dated_filter():
    """Conditions selecting dated, lyrics-bearing songs (>= year 1000)."""
    best = func.coalesce(Song.first_release_year, Song.year)
    return (Song.lyrics.is_not(None), Song.lyrics != "", best >= 1000)


def _distinct_by_decade(session: Session, link_model, fk) -> dict[int, int]:
    """Map decade -> count of distinct ``fk`` ids credited that decade."""
    decade = _decade_col()
    rows = session.execute(
        select(decade, func.count(func.distinct(fk)))
        .select_from(Song)
        .join(link_model, link_model.song_id == Song.id)
        .where(*_dated_filter())
        .group_by(decade)
    ).all()
    return {int(d): int(n) for d, n in rows}


def collect_decade_stats(db_path: str = "data/music.db") -> list[DecadeStat]:
    """Gather per-decade song / performer / author / composer / work counts.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        One :class:`DecadeStat` per decade, ordered oldest-first.
    """
    engine = get_engine(db_path)
    decade = _decade_col()
    try:
        with make_session(engine)() as session:
            song_rows = session.execute(
                select(decade, func.count(Song.id))
                .where(*_dated_filter())
                .group_by(decade)
            ).all()
            songs = {int(d): int(n) for d, n in song_rows}

            performers = _distinct_by_decade(
                session, SongPerformer, SongPerformer.performer_id
            )
            authors = _distinct_by_decade(session, SongAuthor, SongAuthor.author_id)
            composers = _distinct_by_decade(
                session, SongComposer, SongComposer.composer_id
            )
    finally:
        engine.dispose()

    return [
        DecadeStat(
            decade=d,
            songs=songs[d],
            performers=performers.get(d, 0),
            authors=authors.get(d, 0),
            composers=composers.get(d, 0),
        )
        for d in sorted(songs)
    ]


def render_decade_stats(stats: list[DecadeStat]) -> str:
    """Render decade stats as a fixed-width table.

    Args:
        stats: The per-decade stats (see :func:`collect_decade_stats`).

    Returns:
        A human-readable multi-line table.

    Examples:
        >>> out = render_decade_stats([DecadeStat(1990, 100, 40, 25, 18)])
        >>> lines = out.splitlines()
        >>> lines[0].split("|")[0].strip()
        'decade'
        >>> [c.strip() for c in lines[2].split("|")]
        ['1990', '100', '40', '25', '18']
        >>> lines[-1].startswith("TOTAL") and "100" in lines[-1]
        True
    """
    header = (
        f"{'decade':6} | {'songs':>6} | {'performers':>10} | {'authors':>7} | "
        f"{'composers':>9}"
    )
    sep = "-------+--------+------------+---------+----------"
    lines = [header, sep]
    for s in stats:
        lines.append(
            f"{s.decade:>6} | {s.songs:>6} | {s.performers:>10} | {s.authors:>7} | "
            f"{s.composers:>9}"
        )
    total = sum(s.songs for s in stats)
    lines.append(f"{'TOTAL':6} | {total:>6} | {'':>10} | {'':>7} | {'':>9}")
    return "\n".join(lines)


def main() -> None:
    """Print the per-decade descriptive report for the default database."""
    print(render_decade_stats(collect_decade_stats()))


if __name__ == "__main__":
    main()
