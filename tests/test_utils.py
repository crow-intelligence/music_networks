"""Property-based tests for :mod:`src.utils`.

Doctests cover concrete examples; Hypothesis covers the invariants that should
hold for *every* input.
"""

from hypothesis import given
from hypothesis import strategies as st

from src.utils import year_to_decade


@given(year=st.integers(min_value=1000, max_value=9999))
def test_decade_is_multiple_of_ten(year: int) -> None:
    """Any four-digit year maps to a decade divisible by ten."""
    assert year_to_decade(year) % 10 == 0


@given(year=st.integers(min_value=1000, max_value=9999))
def test_decade_within_ten_years_below(year: int) -> None:
    """The decade is the year floored to the nearest ten."""
    decade = year_to_decade(year)
    assert decade <= year < decade + 10


@given(year=st.integers(min_value=0, max_value=999))
def test_non_four_digit_years_return_zero(year: int) -> None:
    """Years with fewer than four digits are rejected with ``0``."""
    assert year_to_decade(year) == 0
