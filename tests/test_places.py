"""Property + unit tests for the place-name gazetteer matcher (``places``)."""

from __future__ import annotations

import string

from hypothesis import given
from hypothesis import strategies as st

from src.lyrics.places import (
    SUFFIXES,
    build_index,
    extract_mentions,
    strip_suffix,
    tokenize,
)

# A small fixed gazetteer of unambiguous names (none are common words).
_ENTRIES = [
    {"name": "Szeged", "lat": 46.25, "lon": 20.15, "place": "city"},
    {"name": "Debrecen", "lat": 47.53, "lon": 21.62, "place": "city"},
    {"name": "Pécs", "lat": 46.07, "lon": 18.23, "place": "city"},
    {"name": "Pécsvárad", "lat": 46.16, "lon": 18.5, "place": "town"},
    {"name": "Hortobágy", "lat": 47.58, "lon": 21.15, "place": "village", "pop": 1208},
]
_GAZ = build_index(_ENTRIES, blocklist=set())

# Suffixes that don't change the stem (no a/e-lengthening corner cases).
_SAFE_SUFFIXES = sorted(s for s in SUFFIXES if s)

_names = st.sampled_from(["Szeged", "Debrecen", "Pécsvárad", "Hortobágy"])
_filler = st.text(alphabet=string.ascii_lowercase + " ", min_size=0, max_size=12)


@given(_names, st.sampled_from(_SAFE_SUFFIXES), _filler, _filler)
def test_capitalised_inflected_name_is_found(name, suffix, pre, post):
    """A capitalised name + a valid suffix is detected (key = lowercased name)."""
    text = f"{pre} {name}{suffix} {post}"
    assert name.lower() in extract_mentions(text, _GAZ)


@given(_names, st.sampled_from(_SAFE_SUFFIXES))
def test_lowercase_occurrence_is_not_a_place(name, suffix):
    """The same token in lowercase is not a place (the capitalisation cue)."""
    assert extract_mentions(f"valami {name.lower()}{suffix} szó", _GAZ) == []


@given(_names, st.text(alphabet="xqwy", min_size=1, max_size=4))
def test_invalid_suffix_is_rejected(name, junk):
    """A name followed by a non-suffix remainder in the same token is not matched."""
    if junk in SUFFIXES:  # the strategy can't produce a real suffix, but be safe
        return
    assert name.lower() not in extract_mentions(f"{name}{junk} valami", _GAZ)


def test_longest_match_wins():
    """``Pécsváradon`` resolves to Pécsvárad, not the shorter Pécs."""
    assert extract_mentions("A Pécsváradon jártam.", _GAZ) == ["pécsvárad"]


def test_blocklist_and_min_length_and_village_pop():
    """Blocked, too-short, and under-populated village names are all dropped."""
    gaz = build_index(
        [
            {"name": "Szeged", "lat": 0, "lon": 0, "place": "city"},
            {"name": "Pápa", "lat": 0, "lon": 0, "place": "town"},  # blocked
            {"name": "Bő", "lat": 0, "lon": 0, "place": "village"},  # too short
            {"name": "Velem", "lat": 0, "lon": 0, "place": "village", "pop": 383},
        ],
        blocklist={"pápa"},
    )
    assert sorted(gaz.places) == ["szeged"]
    # None of the dropped names match even when capitalised.
    assert extract_mentions("Pápán Velemben Bőn Szegeden", gaz) == ["szeged"]


@given(st.text(min_size=0, max_size=80))
def test_extract_mentions_never_crashes_and_returns_known_keys(text):
    """For arbitrary text, every returned key is a real gazetteer entry."""
    for key in extract_mentions(text, _GAZ):
        assert key in _GAZ.places


@given(st.text(alphabet=string.ascii_letters + "áéíóöőúüű ", max_size=40))
def test_tokenize_lowercases_and_keeps_accents(text):
    """Tokens are lowercase and contain only letters (diacritics preserved)."""
    for tok in tokenize(text):
        assert tok == tok.lower() and tok.isalpha()


def test_strip_suffix_excludes_adjectival_i():
    """The surname-colliding adjectival ``-i`` is deliberately not a valid suffix."""
    assert strip_suffix("szegeden", "szeged")
    assert not strip_suffix("szegedi", "szeged")
