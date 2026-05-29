"""Pure text-normalization helpers for matching songs across sources.

The same song appears under slightly different spellings across zeneszoveg,
MusicBrainz and Genius (case, accents, punctuation, stray whitespace). These
helpers collapse those differences into a stable key so the same
``(artist, title)`` lands on the same bucket regardless of source.

All functions here are deterministic and I/O-free — safe to import, doctest and
cover with Hypothesis.
"""

from __future__ import annotations

import re
import unicodedata

# Matches any run of characters that is not a letter, digit or space, so it can
# be squashed away (punctuation, brackets, quotes, etc.).
_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    """Remove diacritics, mapping accented letters to their base form.

    Uses NFKD decomposition then drops combining marks, which handles the
    Hungarian vowels (``á é í ó ú``) including the double-acute ``ő``/``ű``.

    Args:
        text: Arbitrary input text.

    Returns:
        ``text`` with diacritics removed.

    Examples:
        >>> strip_accents("Gyöngyhajú lány")
        'Gyongyhaju lany'
        >>> strip_accents("őrült űrhajó")
        'orult urhajo'
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize(text: str) -> str:
    """Normalize text to a comparable form: lowercased, deaccented, depunctuated.

    Punctuation is dropped, accents stripped, case folded, and internal
    whitespace collapsed to single spaces with the ends trimmed.

    Args:
        text: Arbitrary input text.

    Returns:
        The normalized string (possibly empty).

    Examples:
        >>> normalize("Gyöngyhajú Lány")
        'gyongyhaju lany'
        >>> normalize("  Pancsoló   kislány! ")
        'pancsolo kislany'
        >>> normalize("Üres??!")
        'ures'
    """
    deaccented = strip_accents(text).lower()
    no_punct = _NON_WORD.sub(" ", deaccented)
    return _WHITESPACE.sub(" ", no_punct).strip()


def song_key(artist: str, title: str) -> str:
    """Build a normalized ``artist|title`` matching key.

    Args:
        artist: The performing artist/act name.
        title: The song title.

    Returns:
        A ``"<artist>|<title>"`` key, both parts normalized. The ``|``
        separator never appears inside a part (punctuation is stripped), so the
        split is unambiguous.

    Examples:
        >>> song_key("Beatrice", "Pancsoló kislány")
        'beatrice|pancsolo kislany'
        >>> song_key("Omega", "Gyöngyhajú lány")
        'omega|gyongyhaju lany'
    """
    return f"{normalize(artist)}|{normalize(title)}"
