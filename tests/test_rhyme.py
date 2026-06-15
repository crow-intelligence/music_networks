"""Property-based tests for the rhyme-analysis pure helpers."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from src.lyrics.rhyme import (
    aggregate,
    classify_rhyme,
    common_suffix,
    last_word,
    scheme_label,
    song_examples,
    split_stanzas,
    to_phonemes,
)

_words = st.text(alphabet="aábcdeégyiíklmnoóprstuúzs ", min_size=1, max_size=10)


@given(_words)
def test_to_phonemes_reconstructs_word(word):
    """Joining the phonemes recovers the word's Hungarian letters, lowercased."""
    expected = "".join(c for c in word.lower() if c.isalpha())
    assert "".join(to_phonemes(word)) == expected


def test_to_phonemes_keeps_digraphs():
    """Multigraphs are single tokens."""
    assert to_phonemes("hosszú") == ["h", "o", "s", "sz", "ú"]
    assert to_phonemes("gyöngy") == ["gy", "ö", "n", "gy"]


@given(_words, _words)
def test_classify_is_symmetric(a, b):
    """Rhyme classification (the kind) is order-independent."""
    ra, rb = classify_rhyme(a, b), classify_rhyme(b, a)
    assert (ra is None) == (rb is None)
    if ra and rb:
        assert ra[0] == rb[0]


@given(_words)
def test_identical_word_never_rhymes(word):
    """A word does not rhyme with itself (önrím is excluded)."""
    assert classify_rhyme(word, word) is None


def test_classify_known_pairs():
    """Hand-checked Hungarian rhyme cases."""
    assert classify_rhyme("virág", "világ") == ("tiszta", False)  # genuine pure
    assert classify_rhyme("látok", "várok")[1] is True  # -ok suffix => ragrím
    assert classify_rhyme("ház", "vár") == ("asszonánc", False)  # vowel-only
    assert classify_rhyme("alma", "körte") is None


@given(_words, _words)
def test_common_suffix_is_a_suffix(a, b):
    """The common suffix ends both words and is commutative."""
    cs = common_suffix(a, b)
    assert a.lower().endswith(cs) and b.lower().endswith(cs)
    assert common_suffix(a, b) == common_suffix(b, a)


@given(st.lists(_words, max_size=8))
def test_scheme_label_shape(words):
    """The scheme has one label per line; identical words share a letter."""
    label = scheme_label(words)
    assert len(label) == len(words)
    # Two identical non-empty words must carry the same letter.
    for i, wi in enumerate(words):
        for j, wj in enumerate(words):
            if wi and wj and wi.lower() == wj.lower():
                assert label[i] == label[j]


def test_aggregate_schemes_total_covers_shown():
    """``schemes_total`` is every quatrain, so it's >= the sum of the top-8 shown."""
    rows = [
        {"song_id": i, "decade": 2010, "lines": 4, "rhymed_lines": 4, "density": 1.0,
         "types": {"tiszta": 2}, "schemes": {f"S{i % 11}": 1}}
        for i in range(40)
    ]
    agg = aggregate(rows)
    shown = sum(c for _, c in agg["schemes"])
    assert agg["schemes_total"] == 40  # one quatrain per row
    assert agg["schemes_total"] >= shown
    assert len(agg["schemes"]) <= 8


def test_song_examples_match_their_scheme():
    """A mined stanza's scheme label matches the key it is filed under."""
    lyr = (
        "szállj a virág\nüt a nagy ház\nszállj a világ\nfáj a vad láz\n\n"
        " rövid kis dal\nzeng a halk hang\n"
    )
    pairs, stanzas = song_examples(lyr)
    assert stanzas  # at least one clean 4-line stanza mined
    for scheme, lines in stanzas.items():
        assert len(lines) == 4
        assert scheme_label([last_word(ln) for ln in lines]) == scheme
    # Each type example carries the rhyming word pair AND its two source lines.
    for ex in pairs.values():
        assert len(ex["words"]) == 2 and len(ex["lines"]) == 2
        assert all(last_word(ln) in ex["words"] for ln in ex["lines"])


@given(st.text(max_size=200))
def test_split_stanzas_drops_blanks_and_markers(text):
    """Every stanza line is non-empty, trimmed, and not a full bracket marker."""
    from src.lyrics.rhyme import _BRACKET_LINE

    for stanza in split_stanzas(text):
        assert stanza  # non-empty stanza
        for line in stanza:
            assert line.strip() == line and line
            assert not _BRACKET_LINE.match(line)
