"""Property-based tests for :mod:`src.enrich.match`.

Doctests cover concrete spellings; these Hypothesis tests cover the invariants
the normalizer must satisfy for *any* input so the matching key stays stable.
"""

from hypothesis import given
from hypothesis import strategies as st

from src.enrich.match import normalize, song_key, strip_accents


@given(text=st.text())
def test_normalize_is_idempotent(text: str) -> None:
    """Normalizing an already-normalized string changes nothing."""
    once = normalize(text)
    assert normalize(once) == once


@given(text=st.text())
def test_normalize_output_is_clean(text: str) -> None:
    """Output is lowercase, trimmed, and free of double spaces."""
    out = normalize(text)
    assert out == out.lower()
    assert out == out.strip()
    assert "  " not in out


@given(text=st.text())
def test_strip_accents_preserves_ascii_letters(text: str) -> None:
    """Plain ASCII letters/digits survive accent stripping unchanged."""
    for ch in text:
        if ch.isascii() and ch.isalnum():
            assert ch in strip_accents(ch)


@given(artist=st.text(), title=st.text())
def test_song_key_has_single_separator(artist: str, title: str) -> None:
    """The key splits unambiguously: exactly one ``|``."""
    assert song_key(artist, title).count("|") == 1


@given(
    artist=st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
    title=st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
)
def test_song_key_round_trips_normalized_parts(artist: str, title: str) -> None:
    """Each side of the key equals the normalized input part."""
    left, right = song_key(artist, title).split("|")
    assert left == normalize(artist)
    assert right == normalize(title)
