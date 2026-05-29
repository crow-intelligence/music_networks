"""Centralized SQLite schema and engine factories for the project.

This replaces the per-script MariaDB connection strings and duplicated ORM
models that used to live in ``scraper.py``, ``process_songs.py`` and
``person_graph.py``. Everything now reads/writes a single SQLite file
(``data/music.db``) in WAL mode.

The schema captures, per song *version*, its lyrics and metadata plus the
people involved through role-specific link tables: ``performer`` / ``author`` /
``composer`` (credit strings) and ``person`` (band members). Each ``song`` row
is one occurrence/recording, identified across sources by ``song_external``
``(source, external_id)``; versions of the same composition share ``work_id``
and remakes link back via ``original_song_id``. ``crawl_state`` tracks scraping
progress and ``enrich_state`` tracks per-source enrichment — both resumable.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import ForeignKey, String, UnicodeText, UniqueConstraint, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
)

# Default on-disk location of the single SQLite database.
DEFAULT_DB_PATH = Path("data/music.db")


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


# ---------------------------------------------------------------------------
# Domain tables
# ---------------------------------------------------------------------------
class Song(Base):
    """A single song *version* (one recording / one occurrence).

    Keyed by a surrogate autoincrement id so the same logical song can exist as
    multiple rows across sources (zeneszoveg, MusicBrainz, Genius) and as
    multiple remakes. Source-specific ids live in :class:`SongExternal`, which
    is what makes re-imports idempotent.

    Dating: ``year`` is whatever a lyrics page reported (the release the page
    hangs off — unreliable for first publication, kept only for comparison).
    ``first_release_year`` is the authoritative version-level date (this
    recording's first release, from MusicBrainz). Versions of the same
    composition share ``work_id``; remakes point back to the earliest version
    via ``original_song_id``.
    """

    __tablename__ = "song"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(1000), index=True)
    lyrics: Mapped[str | None] = mapped_column(UnicodeText)
    lyrics_source: Mapped[str | None] = mapped_column(String(40))
    year: Mapped[int | None] = mapped_column(index=True)
    album: Mapped[str | None] = mapped_column(String(1000))
    label: Mapped[str | None] = mapped_column(String(1000))
    length: Mapped[str | None] = mapped_column(String(20))
    # Authoritative version-level first-release dating (from MusicBrainz).
    first_release_year: Mapped[int | None] = mapped_column(index=True)
    first_release_date: Mapped[str | None] = mapped_column(String(10))
    first_release_source: Mapped[str | None] = mapped_column(String(40))
    # Composition grouping + remake linking.
    work_id: Mapped[str | None] = mapped_column(String(40), index=True)
    is_remake: Mapped[bool] = mapped_column(default=False)
    original_song_id: Mapped[int | None] = mapped_column(
        ForeignKey("song.id"), index=True
    )
    genre: Mapped[str | None] = mapped_column(String(100))


class SongExternal(Base):
    """A source-specific id for a :class:`Song` (provenance + idempotency).

    Each row maps one external identity (e.g. ``source='zeneszoveg'``,
    ``external_id='10119'``) to a song row. Unique on ``(source, external_id)``
    so re-imports upsert instead of duplicating.
    """

    __tablename__ = "song_external"
    __table_args__ = (UniqueConstraint("source", "external_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    song_id: Mapped[int] = mapped_column(ForeignKey("song.id"), index=True)
    source: Mapped[str] = mapped_column(String(40), index=True)
    external_id: Mapped[str] = mapped_column(String(200), index=True)
    url: Mapped[str | None] = mapped_column(String(600))


class Performer(Base):
    """A performing act, as credited in a song's "Előadó" field."""

    __tablename__ = "performer"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(400), unique=True, index=True)
    aka: Mapped[str | None] = mapped_column(String(400))


class Author(Base):
    """A lyricist, as credited in a song's "Szövegírók" field."""

    __tablename__ = "author"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(400), unique=True, index=True)
    aka: Mapped[str | None] = mapped_column(String(400))


class Composer(Base):
    """A composer, as credited in a song's "Zeneszerzők" field."""

    __tablename__ = "composer"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(400), unique=True, index=True)
    aka: Mapped[str | None] = mapped_column(String(400))


class Person(Base):
    """A band member, scraped from a ``szemely/`` page; keyed by its URI."""

    __tablename__ = "person"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(400), index=True)
    uri: Mapped[str] = mapped_column(String(400), unique=True, index=True)


class Band(Base):
    """A band/act page (``egyuttes/<id>/``); the crawl unit for members."""

    __tablename__ = "band"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    slug: Mapped[str] = mapped_column(String(400), index=True)
    url: Mapped[str] = mapped_column(String(600))


# --- association tables (each unique on its pair) -------------------------
class SongPerformer(Base):
    """Link: a song was performed by a performer."""

    __tablename__ = "song_performer"
    __table_args__ = (UniqueConstraint("song_id", "performer_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    song_id: Mapped[int] = mapped_column(ForeignKey("song.id"), index=True)
    performer_id: Mapped[int] = mapped_column(ForeignKey("performer.id"), index=True)


class SongAuthor(Base):
    """Link: a song was written by an author (lyricist)."""

    __tablename__ = "song_author"
    __table_args__ = (UniqueConstraint("song_id", "author_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    song_id: Mapped[int] = mapped_column(ForeignKey("song.id"), index=True)
    author_id: Mapped[int] = mapped_column(ForeignKey("author.id"), index=True)


class SongComposer(Base):
    """Link: a song was composed by a composer."""

    __tablename__ = "song_composer"
    __table_args__ = (UniqueConstraint("song_id", "composer_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    song_id: Mapped[int] = mapped_column(ForeignKey("song.id"), index=True)
    composer_id: Mapped[int] = mapped_column(ForeignKey("composer.id"), index=True)


class BandMember(Base):
    """Link: a person is a member of a band."""

    __tablename__ = "band_member"
    __table_args__ = (UniqueConstraint("band_id", "person_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    band_id: Mapped[int] = mapped_column(ForeignKey("band.id"), index=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("person.id"), index=True)


class BandSong(Base):
    """Link: a song was listed on a band's page."""

    __tablename__ = "band_song"
    __table_args__ = (UniqueConstraint("band_id", "song_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    band_id: Mapped[int] = mapped_column(ForeignKey("band.id"), index=True)
    song_id: Mapped[int] = mapped_column(ForeignKey("song.id"), index=True)


# ---------------------------------------------------------------------------
# Crawl bookkeeping (resumability)
# ---------------------------------------------------------------------------
class CrawlState(Base):
    """One row per discovered URL, tracking whether it has been scraped.

    ``status`` is one of ``pending`` / ``done`` / ``failed``. The crawler only
    fetches ``pending`` rows (and ``failed`` rows below the retry ceiling), so
    stopping and restarting simply resumes where it left off.
    """

    __tablename__ = "crawl_state"

    url: Mapped[str] = mapped_column(String(600), primary_key=True)
    kind: Mapped[str] = mapped_column(String(20), index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    attempts: Mapped[int] = mapped_column(default=0)
    last_error: Mapped[str | None] = mapped_column(String(500))
    band_id: Mapped[int | None] = mapped_column(index=True)


class EnrichState(Base):
    """Per-song, per-source enrichment progress (resumable, like CrawlState).

    ``status`` is one of ``pending`` / ``done`` / ``failed``. One row per
    ``(song_id, source)`` lets each enrichment source (e.g. ``musicbrainz``,
    ``genius``) be retried independently and resumed after a stop.
    """

    __tablename__ = "enrich_state"
    __table_args__ = (UniqueConstraint("song_id", "source"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    song_id: Mapped[int] = mapped_column(ForeignKey("song.id"), index=True)
    source: Mapped[str] = mapped_column(String(40), index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    attempts: Mapped[int] = mapped_column(default=0)
    last_error: Mapped[str | None] = mapped_column(String(500))


def _enable_sqlite_pragmas(dbapi_conn, _record) -> None:
    """Enable WAL mode and a busy timeout on every new SQLite connection.

    WAL lets readers and the single writer coexist; ``busy_timeout`` makes
    writers wait briefly for a lock instead of failing immediately.
    """
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def _ensure_parent(db_path: Path) -> None:
    """Create the database file's parent directory if it does not exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)


def get_engine(db_path: Path | str = DEFAULT_DB_PATH) -> Engine:
    """Create a synchronous SQLite engine (for the analysis scripts).

    Args:
        db_path: Filesystem path to the SQLite database file.

    Returns:
        A SQLAlchemy :class:`~sqlalchemy.engine.Engine` with WAL enabled.
    """
    from sqlalchemy import create_engine

    db_path = Path(db_path)
    _ensure_parent(db_path)
    engine = create_engine(f"sqlite:///{db_path}")
    event.listen(engine, "connect", _enable_sqlite_pragmas)
    return engine


def get_async_engine(db_path: Path | str = DEFAULT_DB_PATH) -> AsyncEngine:
    """Create an asynchronous SQLite engine (for the scraper's writer).

    Args:
        db_path: Filesystem path to the SQLite database file.

    Returns:
        A SQLAlchemy :class:`~sqlalchemy.ext.asyncio.AsyncEngine` (aiosqlite),
        with WAL enabled on connect.
    """
    db_path = Path(db_path)
    _ensure_parent(db_path)
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    event.listen(engine.sync_engine, "connect", _enable_sqlite_pragmas)
    return engine


async def create_all(engine: AsyncEngine) -> None:
    """Create every table if it does not already exist.

    Args:
        engine: The async engine whose database should be initialized.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def make_async_session(engine: AsyncEngine) -> async_sessionmaker:
    """Build an :class:`async_sessionmaker` bound to ``engine``.

    Args:
        engine: The async engine to bind sessions to.

    Returns:
        A configured :class:`async_sessionmaker`.
    """
    return async_sessionmaker(engine, expire_on_commit=False)


def make_session(engine: Engine) -> sessionmaker:
    """Build a synchronous :class:`sessionmaker` bound to ``engine``.

    Args:
        engine: The sync engine to bind sessions to.

    Returns:
        A configured :class:`sessionmaker`.
    """
    return sessionmaker(engine, expire_on_commit=False)
