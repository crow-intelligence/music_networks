"""Integration tests for the Genius lyrics importer.

Exercises matching (attach to an existing version) vs insertion (new version),
language filtering, and the no-clobber rule, against a tiny synthetic CSV and a
throwaway DB — no network and no 9 GB download needed.
"""

import csv

from hypothesis import given
from hypothesis import strategies as st
from sqlalchemy import select

from src.db import (
    Performer,
    Song,
    SongExternal,
    SongPerformer,
    get_engine,
    make_session,
)
from src.enrich.genius_kaggle import import_lyrics, import_years, parse_year

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


@given(value=st.integers(min_value=-10000, max_value=10000).map(str))
def test_parse_year_only_accepts_plausible_years(value: str) -> None:
    """Parsed years are always within the plausible window, else ``None``."""
    out = parse_year(value)
    assert out is None or 1900 <= out <= 2026


_YEAR_HEADER = ["id", "title", "artist", "tag", "language", "lyrics", "year"]
_YEAR_ROWS = [
    # matches the undated zeneszoveg song -> should be dated from Genius year
    ["1", "Gyöngyhajú lány", "Omega", "rock", "hu", "Egyszer volt", "1969"],
    # matches a song that already has an authoritative year -> must NOT change
    ["2", "Régi dal", "Beatrice", "rock", "hu", "Szöveg", "1980"],
    # hu row with no matching song -> ignored
    ["3", "Ismeretlen", "Senki", "pop", "hu", "x", "2001"],
    # non-Hungarian -> skipped
    ["4", "Yesterday", "The Beatles", "pop", "en", "y", "1965"],
]


def _write_year_csv(path) -> str:
    csv_path = str(path / "song_lyrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_YEAR_HEADER)
        writer.writerows(_YEAR_ROWS)
    return csv_path


def _seed_for_years(db: str) -> None:
    """Seed one undated lyrics song and one already authoritatively dated."""
    with make_session(get_engine(db))() as s:
        undated = Song(
            title="Gyöngyhajú lány", lyrics="van", lyrics_source="zeneszoveg"
        )
        dated = Song(
            title="Régi dal",
            lyrics="van",
            lyrics_source="zeneszoveg",
            first_release_year=1976,
            first_release_source="musicbrainz",
        )
        s.add_all([undated, dated])
        s.flush()
        omega = Performer(name="Omega")
        beatrice = Performer(name="Beatrice")
        s.add_all([omega, beatrice])
        s.flush()
        s.add_all(
            [
                SongPerformer(song_id=undated.id, performer_id=omega.id),
                SongPerformer(song_id=dated.id, performer_id=beatrice.id),
            ]
        )
        s.commit()


def test_import_years_fills_only_undated(tmp_path) -> None:
    """Genius years date undated lyrics songs without clobbering MB dates."""
    db = str(tmp_path / "y.db")
    from src.db import Base

    Base.metadata.create_all(get_engine(db))
    _seed_for_years(db)

    counts = import_years(_write_year_csv(tmp_path), db)
    # three hu rows streamed (the en row is filtered out before counting)
    assert counts == {"rows": 3, "dated": 1}

    with make_session(get_engine(db))() as s:
        undated = s.execute(
            select(Song).where(Song.title == "Gyöngyhajú lány")
        ).scalar_one()
        assert undated.first_release_year == 1969
        assert undated.first_release_source == "genius"
        # the already-authoritative song is untouched
        dated = s.execute(select(Song).where(Song.title == "Régi dal")).scalar_one()
        assert dated.first_release_year == 1976
        assert dated.first_release_source == "musicbrainz"
