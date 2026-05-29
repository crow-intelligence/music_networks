"""Import the Hungarian subset of the Genius lyrics dump.

The Genius "song lyrics with language information" Kaggle dump is one ~9 GB CSV
of ~5M rows. We stream it (the lyrics column has embedded newlines, so a real
CSV parser is required), keep only ``language == 'hu'`` rows, and attach their
lyrics + coarse genre to the corpus:

* if a row matches an existing song version (by normalized ``artist|title``),
  fill its ``lyrics`` (only when empty — never clobber) and ``genre``;
* otherwise insert a new ``source='genius'`` song version.

This complements MusicBrainz, which supplies dates/structure but no lyrics.
``year`` from Genius is unreliable (page year, not first release) and is
deliberately ignored for dating.
"""

from __future__ import annotations

import csv
import sys
from collections.abc import Iterator

from sqlalchemy import select

from src.db import (
    Performer,
    Song,
    SongExternal,
    SongPerformer,
    get_engine,
    make_session,
)
from src.enrich.match import song_key

_GENIUS = "genius"

# Lyrics fields blow past the default CSV field limit; raise it as high as the
# platform allows.
def _raise_csv_limit() -> None:
    """Raise the CSV field-size limit to the platform maximum."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2


def clean_row(row: dict) -> dict | None:
    """Reduce a raw Genius CSV row to the fields we store, or ``None`` to skip.

    Args:
        row: A CSV row as produced by ``csv.DictReader``.

    Returns:
        A dict with ``artist``/``title``/``lyrics``/``genre``/``genius_id``, or
        ``None`` if it lacks the essentials (artist, title, non-empty lyrics).

    Examples:
        >>> raw = {"artist": "Omega", "title": "Gyöngyhajú lány",
        ...        "lyrics": "Egyszer volt...", "tag": "pop", "id": "42"}
        >>> clean_row(raw) == {"artist": "Omega", "title": "Gyöngyhajú lány",
        ...     "lyrics": "Egyszer volt...", "genre": "pop", "genius_id": "42"}
        True
        >>> clean_row({"artist": "X", "title": "Y", "lyrics": "  "}) is None
        True
    """
    artist = (row.get("artist") or "").strip()
    title = (row.get("title") or "").strip()
    lyrics = (row.get("lyrics") or "").strip()
    if not artist or not title or not lyrics:
        return None
    return {
        "artist": artist,
        "title": title,
        "lyrics": lyrics,
        "genre": (row.get("tag") or "").strip() or None,
        "genius_id": (row.get("id") or "").strip(),
    }


def iter_hungarian_rows(csv_path: str) -> Iterator[dict]:
    """Stream cleaned Hungarian (``language == 'hu'``) rows from the dump.

    Args:
        csv_path: Path to ``song_lyrics.csv``.

    Yields:
        Cleaned row dicts (see :func:`clean_row`) for Hungarian songs only.
    """
    _raise_csv_limit()
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("language") != "hu":
                continue
            cleaned = clean_row(row)
            if cleaned is not None:
                yield cleaned


def _existing_key_map(session) -> dict[str, int]:
    """Map normalized ``artist|title`` -> song id for current song versions."""
    rows = session.execute(
        select(Performer.name, Song.title, Song.id)
        .join(SongPerformer, SongPerformer.song_id == Song.id)
        .join(Performer, Performer.id == SongPerformer.performer_id)
    ).all()
    mapping: dict[str, int] = {}
    for performer_name, title, song_id in rows:
        mapping.setdefault(song_key(performer_name, title), song_id)
    return mapping


def import_lyrics(csv_path: str, db_path: str = "data/music.db") -> dict[str, int]:
    """Import Hungarian Genius lyrics into the corpus.

    Matches each row to an existing version by ``artist|title``; fills lyrics
    (only if empty) + genre on a match, else inserts a new ``genius`` version.

    Args:
        csv_path: Path to ``song_lyrics.csv``.
        db_path: Path to the SQLite database file.

    Returns:
        Counts ``{"matched": int, "added": int, "rows": int}``.
    """
    engine = get_engine(db_path)
    matched = added = rows = 0
    with make_session(engine)() as session:
        keymap = _existing_key_map(session)
        for row in iter_hungarian_rows(csv_path):
            rows += 1
            key = song_key(row["artist"], row["title"])
            song_id = keymap.get(key)
            if song_id is not None:
                song = session.get(Song, song_id)
                if song is not None and not (song.lyrics or "").strip():
                    song.lyrics = row["lyrics"]
                    song.lyrics_source = _GENIUS
                if song is not None and song.genre is None:
                    song.genre = row["genre"]
                matched += 1
            else:
                song = Song(
                    title=row["title"],
                    lyrics=row["lyrics"],
                    lyrics_source=_GENIUS,
                    genre=row["genre"],
                )
                session.add(song)
                session.flush()
                if row["genius_id"]:
                    session.add(
                        SongExternal(
                            song_id=song.id,
                            source=_GENIUS,
                            external_id=row["genius_id"],
                            url=f"https://genius.com/songs/{row['genius_id']}",
                        )
                    )
                _link_performer(session, song.id, row["artist"])
                keymap[key] = song.id
                added += 1
            if rows % 1000 == 0:
                session.commit()
        session.commit()
    engine.dispose()
    return {"matched": matched, "added": added, "rows": rows}


def _link_performer(session, song_id: int, name: str) -> None:
    """Get-or-create the performing act and link it to the song."""
    performer = session.execute(
        select(Performer).where(Performer.name == name)
    ).scalar_one_or_none()
    if performer is None:
        performer = Performer(name=name)
        session.add(performer)
        session.flush()
    exists = session.execute(
        select(SongPerformer).where(
            SongPerformer.song_id == song_id,
            SongPerformer.performer_id == performer.id,
        )
    ).scalar_one_or_none()
    if exists is None:
        session.add(SongPerformer(song_id=song_id, performer_id=performer.id))
