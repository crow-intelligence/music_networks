"""Property tests for the KWIC concordance helper."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from src.lyrics.kwic import _norm, concordances

_word = st.text(alphabet="abcdeé ", min_size=1, max_size=8)


@given(st.lists(_word, min_size=1, max_size=30), st.integers(min_value=1, max_value=6))
def test_concordance_windows_are_bounded_and_match(words, window):
    """Every window's keyword matches the stem and context stays within bounds."""
    text = " ".join(w for w in words if w.strip())
    stem = "ab"
    cs = concordances(text, stem, window=window)
    toks = text.split()
    for c in cs:
        assert _norm(c["kw"]).startswith(stem)
        assert len(c["pre"].split()) <= window
        assert len(c["post"].split()) <= window
    # Count matches independently — every stem-matching token yields one window.
    expected = sum(1 for t in toks if _norm(t).startswith(stem))
    assert len(cs) == expected


def test_concordance_marks_keyword_and_context():
    """A concrete window splits pre / kw / post correctly."""
    c = concordances("a szép szerelem dala szól", "szerel")[0]
    assert c["kw"] == "szerelem"
    assert c["pre"] == "a szép"
    assert c["post"] == "dala szól"
