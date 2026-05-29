"""Integration tests for the Genius lyrics importer.

Exercises matching (attach to an existing version) vs insertion (new version),
language filtering, and the no-clobber rule, against a tiny synthetic CSV and a
throwaway DB — no network and no 9 GB download needed.
"""

import csv

from sqlalchemy import select

from src.db import (
    Performer,
    Song,
    SongExternal,
    SongPerformer,
    get_engine,
    make_session,
)
from src.enrich.genius_kaggle import import_lyrics

_HEADER = ["id", "title", "artist", "tag", "language", "lyrics"]
_ROWS = [
    # matches the pre-seeded MusicBrainz version (Omega / Gyöngyhajú lány)
    ["1", "Gyöngyhajú lány", "Omega", "rock", "hu", "Egyszer volt, hol nem volt"],
    # Hungarian, no existing match -> inserted as a new genius version
    ["2", "Új dal", "Valaki", "pop", "hu", "Friss szöveg"],
    # non-Hungarian -> skipped
    ["3", "Yesterday", "The Beatles", "pop", "en", "Yesterday..."],
    # Hungarian but empty lyrics -> skipped
    ["4", "Üres", "Senki", "pop", "hu", "   "],
]


def _write_csv(path) -> str:
    csv_path = str(path / "song_lyrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_HEADER)
        writer.writerows(_ROWS)
    return csv_path


def _seed_existing_version(db: str) -> None:
    """Insert one MusicBrainz-style version with no lyrics yet."""
    with make_session(get_engine(db))() as s:
        song = Song(title="Gyöngyhajú lány", first_release_year=1969)
        s.add(song)
        s.flush()
        performer = Performer(name="Omega")
        s.add(performer)
        s.flush()
        s.add(SongPerformer(song_id=song.id, performer_id=performer.id))
        s.commit()


def test_import_matches_and_inserts(tmp_path) -> None:
    """Hungarian rows attach lyrics to a match, else insert a new version."""
    db = str(tmp_path / "t.db")
    get_engine(db)  # creates the file
    from src.db import Base

    Base.metadata.create_all(get_engine(db))
    _seed_existing_version(db)

    counts = import_lyrics(_write_csv(tmp_path), db)
    assert counts == {"rows": 2, "matched": 1, "added": 1}

    with make_session(get_engine(db))() as s:
        # the existing Omega version now carries the lyrics + genre, same row
        omega = s.execute(
            select(Song).where(Song.title == "Gyöngyhajú lány")
        ).scalar_one()
        assert omega.first_release_year == 1969  # untouched
        assert omega.lyrics == "Egyszer volt, hol nem volt"
        assert omega.lyrics_source == "genius"
        assert omega.genre == "rock"
        # the unmatched hu row became a new genius version with provenance
        new = s.execute(select(Song).where(Song.title == "Új dal")).scalar_one()
        assert new.lyrics == "Friss szöveg"
        ext = s.execute(
            select(SongExternal).where(SongExternal.source == "genius")
        ).scalar_one()
        assert ext.song_id == new.id and ext.external_id == "2"
        # only two songs total: the en row and empty-lyrics row were skipped
        assert len(s.execute(select(Song)).scalars().all()) == 2
