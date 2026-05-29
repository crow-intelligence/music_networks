"""Shared, import-safe helpers for the music-networks analysis scripts.

This module is the reference for the project's conventions: Google-style
docstrings, full type hints (checked with ``ty``), and ``doctest`` examples
that double as executable documentation. It contains no module-level side
effects, so it is safe to import and to collect with ``--doctest-modules``.
"""


def year_to_decade(year: int) -> int:
    """Round a four-digit year down to its decade.

    Args:
        year: A calendar year. Expected to be a four-digit integer such as
            ``1987``.

    Returns:
        The decade the year falls in (e.g. ``1980``), or ``0`` when ``year``
        is not a four-digit number.

    Examples:
        >>> year_to_decade(1987)
        1980
        >>> year_to_decade(2000)
        2000
        >>> year_to_decade(2019)
        2010
        >>> year_to_decade(42)
        0
    """
    if len(str(year)) == 4:
        return int(str(year)[:-1] + "0")
    return 0
