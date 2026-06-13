"""Property-based tests for the Wikipedia-pageviews pure helpers."""

from __future__ import annotations

import math
import urllib.parse

from hypothesis import given
from hypothesis import strategies as st

from src.enrich.pageviews import average_monthly_views, encode_title


@given(st.text(min_size=1, max_size=40))
def test_encode_title_is_path_safe_and_reversible(title):
    """Encoding has no spaces and decodes back to the underscore form."""
    encoded = encode_title(title)
    assert " " not in encoded
    # Percent-decoding recovers the title with spaces turned into underscores.
    assert urllib.parse.unquote(encoded) == title.replace(" ", "_")


@given(
    st.lists(
        st.fixed_dictionaries(
            {"views": st.integers(min_value=0, max_value=10_000_000)}
        ),
        max_size=120,
    )
)
def test_average_monthly_views_is_the_mean(items):
    """The score is the arithmetic mean of the view counts (0 when empty)."""
    out = average_monthly_views(items)
    if not items:
        assert out == 0.0
    else:
        counts = [it["views"] for it in items]
        assert math.isclose(out, sum(counts) / len(counts), rel_tol=1e-9)
        assert min(counts) <= out <= max(counts)
