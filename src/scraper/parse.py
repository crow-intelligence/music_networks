"""Pure HTML/URL parsing helpers for the zeneszoveg.hu scraper.

Every function here is deterministic and free of I/O, which makes them safe to
import, to doctest, and to cover with Hypothesis. The CSS selectors and title
format were verified against live pages on 2026-05-29.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

#: Suffixes appended to the ``<title>`` of a song page, stripped during parsing.
_TITLE_SUFFIXES = (
    " dalszöveg, videó - Zeneszöveg.hu",
    " dalszöveg - Zeneszöveg.hu",
)

#: Placeholder strings the site shows when a field has no real data.
PLACEHOLDERS = ("Keressük",)


def _attr(tag, key: str) -> str:
    """Return a tag attribute as a plain string.

    BeautifulSoup may return a list for multi-valued attributes (or ``None``
    when absent); this normalizes both to a single string.

    Args:
        tag: A BeautifulSoup ``Tag``.
        key: The attribute name.

    Returns:
        The attribute value as a string, or ``""`` if absent.
    """
    value = tag.get(key)
    if isinstance(value, list):
        return " ".join(value)
    return value if isinstance(value, str) else ""


@dataclass
class SongRecord:
    """Structured data extracted from a single song page.

    Attributes:
        title: Lower-cased song title.
        performers: Performer ("Előadó") credit strings.
        authors: Lyricist ("Szövegírók") credit strings.
        composers: Composer ("Zeneszerzők") credit strings.
        lyrics: The plain-text lyrics, or ``None`` if absent/placeholder.
        year: Release year ("Megjelenés"), or ``None``.
        album: Album name, or ``None``.
        label: Record label ("Kiadó"), or ``None``.
        length: Track length string ("Hossz", e.g. ``"2:59"``), or ``None``.
    """

    title: str
    performers: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    composers: list[str] = field(default_factory=list)
    lyrics: str | None = None
    year: int | None = None
    album: str | None = None
    label: str | None = None
    length: str | None = None


@dataclass
class BandPage:
    """Links discovered on a band ("egyuttes") page.

    Attributes:
        song_urls: Relative song-page hrefs (start with ``dalszoveg``).
        members: ``(name, uri)`` pairs for band members (``szemely/`` links).
    """

    song_urls: list[str] = field(default_factory=list)
    members: list[tuple[str, str]] = field(default_factory=list)


def extract_song_id(url: str) -> int | None:
    """Extract the numeric song id from a ``dalszoveg`` URL.

    Args:
        url: A song URL or href, absolute or relative.

    Returns:
        The numeric id, or ``None`` if the URL is not a song URL.

    Examples:
        >>> extract_song_id("dalszoveg/3485/akela/a-farkasok-dala-zeneszoveg.html")
        3485
        >>> extract_song_id("https://www.zeneszoveg.hu/dalszoveg/101559/akela/x.html")
        101559
        >>> extract_song_id("egyuttes/138/akela-dalszovegei.html") is None
        True
    """
    m = re.search(r"dalszoveg/(\d+)/", url)
    return int(m.group(1)) if m else None


def extract_band_id(url: str) -> int | None:
    """Extract the numeric band id from an ``egyuttes`` URL.

    Args:
        url: A band URL or href, absolute or relative.

    Returns:
        The numeric id, or ``None`` if the URL is not a band URL.

    Examples:
        >>> extract_band_id("egyuttes/138/akela-dalszovegei.html")
        138
        >>> extract_band_id("dalszoveg/3485/akela/x.html") is None
        True
    """
    m = re.search(r"egyuttes/(\d+)/", url)
    return int(m.group(1)) if m else None


def band_slug(url: str) -> str:
    """Return the band slug from an ``egyuttes`` URL.

    Args:
        url: A band URL or href.

    Returns:
        The slug with the ``-dalszovegei.html`` suffix removed, or ``""``.

    Examples:
        >>> band_slug("egyuttes/138/akela-dalszovegei.html")
        'akela'
        >>> band_slug("egyuttes/2046/a-jo-lacibetyar-dalszovegei.html")
        'a-jo-lacibetyar'
    """
    tail = url.rstrip("/").split("/")[-1]
    return tail.replace("-dalszovegei.html", "")


def normalize_name(raw: str) -> str:
    """Collapse whitespace and lower-case a name or title.

    Args:
        raw: An arbitrary name/title string.

    Returns:
        The string with runs of whitespace collapsed to single spaces,
        trimmed and lower-cased.

    Examples:
        >>> normalize_name("  A   Farkasok   Dala ")
        'a farkasok dala'
    """
    return " ".join(raw.split()).lower()


def parse_title(raw_title: str) -> tuple[str, str]:
    """Split a song page ``<title>`` into ``(performer, title)``.

    Args:
        raw_title: The raw ``<title>`` text, of the form
            ``"Performer : Title dalszöveg, videó - Zeneszöveg.hu"``.

    Returns:
        A ``(performer, title)`` tuple, both normalized (lower-cased,
        whitespace-collapsed). If no ``":"`` is present, the performer is
        empty and the whole string (minus suffix) is treated as the title.

    Examples:
        >>> parse_title("Akela : A farkasok dala dalszöveg, videó - Zeneszöveg.hu")
        ('akela', 'a farkasok dala')
        >>> parse_title("Edda Művek : Kstyx dalszöveg - Zeneszöveg.hu")
        ('edda művek', 'kstyx')
    """
    text = raw_title
    for suffix in _TITLE_SUFFIXES:
        text = text.replace(suffix, "")
    performer, _, title = text.partition(":")
    if not title:
        return "", normalize_name(performer)
    return normalize_name(performer), normalize_name(title)


def split_aka(raw: str) -> tuple[str, str]:
    """Separate a parenthesised alias from a credit name.

    Args:
        raw: A credit string, possibly containing ``"(alias)"``.

    Returns:
        A ``(name, aka)`` tuple. ``name`` is normalized; ``aka`` is the raw
        alias text, or ``"None"`` when there is no parenthesised part.

    Examples:
        >>> split_aka("Bródy János (Hobo)")
        ('bródy jános', 'Hobo')
        >>> split_aka("Katona László")
        ('katona lászló', 'None')
    """
    if "(" in raw and ")" in raw:
        start, end = raw.index("("), raw.index(")")
        aka = raw[start + 1 : end]
        name = raw.replace(f"({aka})", "")
        return normalize_name(name), aka
    return normalize_name(raw), "None"


def _is_placeholder(value: str) -> bool:
    """Return ``True`` if ``value`` is one of the site's placeholder strings.

    Examples:
        >>> _is_placeholder("Keressük a zeneszerzőt!")
        True
        >>> _is_placeholder("Magneoton")
        False
    """
    return any(value.startswith(p) for p in PLACEHOLDERS)


def _split_multi(value: str) -> list[str]:
    r"""Split a newline-separated multi-value metadata field into clean items.

    Args:
        value: The raw cell text (credits are newline-separated).

    Returns:
        A list of non-empty, non-placeholder items.

    Examples:
        >>> _split_multi("Katona László\nIsmeretlen\n")
        ['Katona László', 'Ismeretlen']
        >>> _split_multi("Keressük a zeneszerzőt!")
        []
    """
    items = [part.strip() for part in value.split("\n")]
    return [it for it in items if it and not _is_placeholder(it)]


def parse_metadata(soup: BeautifulSoup) -> dict[str, str]:
    """Collect the label -> value pairs from a song page's metadata boxes.

    Labels are normalized by stripping non-word characters (so ``"Előadó:"``
    becomes ``"Előadó"``), matching the original scraper's behaviour.

    Args:
        soup: Parsed song page.

    Returns:
        A mapping of normalized label to raw value text.
    """
    data: dict[str, str] = {}
    for box in soup.find_all("div", {"class": "lyrics-header-text short"}):
        table = box.find("table")
        if not table:
            continue
        for label in table.select("th, td.attrLabels"):
            sibling = label.find_next_sibling()
            if sibling is None:
                continue
            key = re.sub(r"\W+", "", label.get_text().strip())
            data[key] = sibling.get_text().strip()
    return data


def parse_song_page(html: str) -> SongRecord:
    r"""Parse a full song page into a :class:`SongRecord`.

    Args:
        html: Raw HTML of a ``dalszoveg`` page.

    Returns:
        A populated :class:`SongRecord`. Missing/placeholder fields are left
        as ``None`` or empty lists.

    Examples:
        >>> html = '''
        ... <html><head><title>Akela : Teszt dalszöveg, videó - Zeneszöveg.hu
        ... </title></head><body>
        ... <div class="lyrics-plain-text">Egy sor\nMásik sor</div>
        ... <div class="lyrics-header-text short"><table>
        ... <tr><th>Előadó:</th><td>Akela</td></tr>
        ... <tr><th>Megjelenés:</th><td>1995</td></tr></table></div>
        ... <div class="lyrics-header-text short"><table>
        ... <tr><th>Szövegírók:</th><td>Katona László</td></tr>
        ... <tr><th>Zeneszerzők:</th><td>Keressük a zeneszerzőt!</td></tr>
        ... </table></div></body></html>'''
        >>> rec = parse_song_page(html)
        >>> rec.title
        'teszt'
        >>> rec.performers, rec.authors, rec.composers
        (['Akela'], ['Katona László'], [])
        >>> rec.year
        1995
        >>> rec.lyrics
        'Egy sor\nMásik sor'
    """
    soup = BeautifulSoup(html, "lxml")

    title_tag = soup.find("title")
    _performer_from_title, title = parse_title(
        title_tag.get_text() if title_tag else ""
    )

    lyrics_div = soup.find("div", class_="lyrics-plain-text")
    lyrics: str | None = None
    if lyrics_div:
        text = lyrics_div.get_text().strip()
        if text and not _is_placeholder(text):
            lyrics = text

    meta = parse_metadata(soup)

    year: int | None = None
    if (raw_year := meta.get("Megjelenés", "")).isdigit():
        year = int(raw_year)

    album = meta.get("Album") or None
    if album and _is_placeholder(album):
        album = None
    label = meta.get("Kiadó") or None
    if label and _is_placeholder(label):
        label = None
    length = meta.get("Hossz") or None
    if length and _is_placeholder(length):
        length = None

    return SongRecord(
        title=title,
        performers=_split_multi(meta.get("Előadó", "")),
        authors=_split_multi(meta.get("Szövegírók", "")),
        composers=_split_multi(meta.get("Zeneszerzők", "")),
        lyrics=lyrics,
        year=year,
        album=album,
        label=label,
        length=length,
    )


def _member_name(anchor) -> str:
    """Best-effort member name for a ``szemely/`` anchor.

    The visible link is an image, so the name lives in the anchor's ``title``
    (falling back to the image ``alt`` or link text). A trailing ``"(band)"``
    qualifier is dropped.

    Examples:
        >>> from bs4 import BeautifulSoup
        >>> a = BeautifulSoup(
        ...     '<a title="Kovács Attila (Akela)"><img alt="x"></a>', "lxml"
        ... ).a
        >>> _member_name(a)
        'Kovács Attila'
    """
    name = _attr(anchor, "title")
    if not name:
        img = anchor.find("img")
        name = (_attr(img, "alt") if img else "") or anchor.get_text()
    name = name.strip()
    if "(" in name:
        name = name[: name.index("(")].strip()
    return name


def parse_band_page(html: str, slug: str) -> BandPage:
    """Extract this band's song links and members from a band page.

    Band pages also embed "related songs" from *other* acts, so song links are
    filtered to those whose URL contains this band's ``slug`` segment. The
    "submit new person" link (``szemely/uj.html``) is ignored.

    Args:
        html: Raw HTML of an ``egyuttes`` page.
        slug: This band's slug (see :func:`band_slug`), used to keep only the
            band's own songs.

    Returns:
        A :class:`BandPage` with de-duplicated song URLs (order preserved) and
        ``(name, uri)`` member pairs.

    Examples:
        >>> html = '''<html><body>
        ... <a href="dalszoveg/1/akela/egy-zeneszoveg.html">Egy</a>
        ... <a href="dalszoveg/1/akela/egy-zeneszoveg.html">dup</a>
        ... <a href="dalszoveg/2/masok/related-zeneszoveg.html">related</a>
        ... <a href="szemely/9/teszt-bela-adatlap.html" title="Teszt Béla (Akela)">
        ...   <img alt="Teszt Béla"></a>
        ... <a href="szemely/uj.html">új személy beküldése</a>
        ... </body></html>'''
        >>> page = parse_band_page(html, "akela")
        >>> page.song_urls
        ['dalszoveg/1/akela/egy-zeneszoveg.html']
        >>> page.members
        [('Teszt Béla', 'szemely/9/teszt-bela-adatlap.html')]
    """
    soup = BeautifulSoup(html, "lxml")
    song_urls: list[str] = []
    members: list[tuple[str, str]] = []
    seen_songs: set[str] = set()
    seen_members: set[str] = set()
    segment = f"/{slug}/"

    for a in soup.find_all("a"):
        href = _attr(a, "href")
        if not href:
            continue
        if href.startswith("dalszoveg") and segment in href and href not in seen_songs:
            seen_songs.add(href)
            song_urls.append(href)
        elif (
            href.startswith("szemely/")
            and href != "szemely/uj.html"
            and href not in seen_members
        ):
            seen_members.add(href)
            members.append((_member_name(a), href))

    return BandPage(song_urls=song_urls, members=members)


def parse_band_links(html: str) -> list[str]:
    """Extract band-page hrefs from an artist-index listing page.

    Args:
        html: Raw HTML of an ``/eloadok/<initial>`` page.

    Returns:
        De-duplicated ``egyuttes/<id>/...`` hrefs in document order. The
        "submit new band" link (``egyuttes/uj.html``, no numeric id) is skipped.

    Examples:
        >>> html = '''<a href="egyuttes/1/a-dalszovegei.html">A</a>
        ... <a href="egyuttes/1/a-dalszovegei.html">dup</a>
        ... <a href="egyuttes/uj.html">submit new band</a>
        ... <a href="szemely/9/x.html">person</a>'''
        >>> parse_band_links(html)
        ['egyuttes/1/a-dalszovegei.html']
    """
    soup = BeautifulSoup(html, "lxml")
    links: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a"):
        href = _attr(a, "href")
        if extract_band_id(href) is not None and href not in seen:
            seen.add(href)
            links.append(href)
    return links


def parse_last_index_page(html: str) -> int:
    """Find the highest pagination page number on an artist-index page.

    The site paginates with ``&page=N`` and labels the final page link
    "Utolsó oldal". When no pagination is present, there is a single page.

    Args:
        html: Raw HTML of an ``/eloadok/<initial>`` page.

    Returns:
        The number of the last page (``1`` if unpaginated).

    Examples:
        >>> html = '<a href="eloadok/a&page=3">Utolsó oldal→</a>'
        >>> parse_last_index_page(html)
        3
        >>> parse_last_index_page("<html>no pagination</html>")
        1
    """
    soup = BeautifulSoup(html, "lxml")
    last = 1
    for a in soup.find_all("a"):
        m = re.search(r"[?&]page=(\d+)", _attr(a, "href"))
        if m:
            last = max(last, int(m.group(1)))
    return last
