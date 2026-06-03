"""Property-based tests for the targeted-dating pure helpers.

Doctests cover concrete cases; these Hypothesis tests cover the invariants the
MusicBrainz query builder and match-validating date picker must satisfy for
*any* input, so a wrong release year is never pinned on a song.
"""

from hypothesis import given
from hypothesis import strategies as st

from src.enrich.match import normalize
from src.enrich.musicbrainz import (
    RecordingInfo,
    pick_earliest_recording,
    recording_query,
)

# Text without the two characters the query builder must escape, so the plain
# (unescaped) round-trip below is exact.
_PLAIN = st.text().filter(lambda s: '"' not in s and "\\" not in s)


@given(artist=_PLAIN, title=_PLAIN)
def test_query_is_exact_when_nothing_to_escape(artist: str, title: str) -> None:
    """Inputs free of quotes/backslashes appear verbatim inside the phrases."""
    assert recording_query(artist, title) == (
        f'artist:"{artist}" AND recording:"{title}"'
    )


@given(artist=st.text(), title=st.text())
def test_query_escapes_quotes_and_backslashes(artist: str, title: str) -> None:
    """No raw double-quote survives except the four structural delimiters."""
    query = recording_query(artist, title)
    # Remove escaped backslashes first, then escaped quotes; what remains are
    # only the 4 structural delimiters.
    stripped = query.replace("\\\\", "").replace('\\"', "")
    assert stripped.count('"') == 4


def _rec(title: str, artist: str, date: str | None) -> RecordingInfo:
    """Build a minimal RecordingInfo for picker tests."""
    return RecordingInfo(
        gid=f"{title}|{artist}|{date}",
        title=title,
        artist=artist,
        first_release_date=date,
        work_id=None,
    )


@given(
    title=st.text(min_size=1),
    artist=st.text(min_size=1),
)
def test_pick_requires_a_title_match(title: str, artist: str) -> None:
    """With no title-matching candidate, the picker returns ``None``."""
    others = [_rec(title + " (remix) zzz", artist, "1980")]
    # Only keep the case where titles genuinely differ after normalization.
    if normalize(others[0].title) != normalize(title):
        assert pick_earliest_recording(artist, title, others) is None


@given(
    title=st.text(min_size=1).filter(lambda s: normalize(s)),
    artist=st.text(min_size=1).filter(lambda s: normalize(s)),
    dates=st.lists(
        st.integers(min_value=1900, max_value=2025).map(lambda y: f"{y}"),
        min_size=1,
        max_size=6,
    ),
)
def test_pick_returns_earliest_dated_match(
    title: str, artist: str, dates: list[str]
) -> None:
    """Among valid matches, the earliest release date wins."""
    recs = [_rec(title, artist, d) for d in dates]
    chosen = pick_earliest_recording(artist, title, recs)
    assert chosen is not None
    assert chosen.first_release_date == min(dates)


@given(
    title=st.text(min_size=1).filter(lambda s: normalize(s)),
    artist=st.text(min_size=1).filter(lambda s: normalize(s)),
)
def test_pick_rejects_wrong_artist(title: str, artist: str) -> None:
    """A title match credited to an unrelated artist is not accepted."""
    foreign = "zzztotallyunrelatedactname"
    # Guard: ensure our artist isn't accidentally a substring of the decoy.
    if normalize(artist) not in normalize(foreign):
        recs = [_rec(title, foreign, "1990")]
        assert pick_earliest_recording(artist, title, recs) is None
