"""Wikidata (+ Hungarian Wikipedia) client for artist/band enrichment.

Where MusicBrainz answers *when* a recording was released, Wikidata answers
*who* the act is: a musician's birth/death, a band's formation/dissolution,
country, birthplace, genre, and the link to the Hungarian Wikipedia article. It
is the right source for this because it holds the same facts as huwiki infoboxes
as clean structured triples, exposes them via a stable SPARQL endpoint, and
cross-links to MusicBrainz through property ``P434`` — so the artists we already
enumerated from MusicBrainz join up with no fuzzy matching.

This module mirrors :mod:`src.enrich.musicbrainz`:

* pure parsers (``parse_artist_binding``, ``merge_artists`` and the small
  field helpers) that are deterministic and doctested, and
* an async :class:`WikidataClient` that reuses the scraper's
  :class:`~src.scraper.fetch.RateLimiter` for politeness and honours the WDQS
  usage policy (descriptive User-Agent, gzip, ``Retry-After`` on 429).

The Hungarian-artist SPARQL query lives in :func:`hungarian_artists_query`; the
intro bio paragraph comes from the MediaWiki ``extracts`` API via
:meth:`WikidataClient.huwiki_extract`.
"""

from __future__ import annotations

import asyncio
import urllib.parse
from collections.abc import Iterable
from dataclasses import dataclass, field

import httpx

from src.enrich import USER_AGENT
from src.scraper.fetch import RateLimiter

# Wikidata Query Service SPARQL endpoint and the Hungarian Wikipedia API.
WDQS_ENDPOINT = "https://query.wikidata.org/sparql"
HUWIKI_API = "https://hu.wikipedia.org/w/api.php"
HUWIKI_PREFIX = "https://hu.wikipedia.org/"

# Transient HTTP statuses worth retrying (WDQS throttles with 429/5xx).
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

# Occupations (P106) that make a Hungarian human an "artist" for our purposes:
# musician, singer, composer, songwriter, lyricist, guitarist, pianist, rapper.
_MUSIC_OCCUPATIONS = (
    "Q639669",
    "Q177220",
    "Q36834",
    "Q753110",
    "Q12144794",
    "Q855091",
    "Q486748",
    "Q2252262",
)


@dataclass
class ArtistInfo:
    """A Wikidata artist reduced to the fields we store.

    Attributes:
        qid: The Wikidata entity id (e.g. ``"Q151944"``), our external id.
        name: Preferred label (Hungarian, English fallback).
        kind: ``"person"`` or ``"group"``.
        mb_artist_id: MusicBrainz artist id (``P434``) if linked, else ``None``.
        begin_date: Birth (person) or inception (group) date, partial-ISO.
        end_date: Death (person) or dissolution (group) date, partial-ISO.
        birthplace: Place-of-birth label (persons), else ``None``.
        genres: Distinct genre labels (``P136``), sorted.
        huwiki_url: Hungarian Wikipedia article URL, or ``None``.
        huwiki_title: Hungarian Wikipedia article title, or ``None``.
    """

    qid: str
    name: str
    kind: str | None = None
    mb_artist_id: str | None = None
    begin_date: str | None = None
    end_date: str | None = None
    birthplace: str | None = None
    genres: list[str] = field(default_factory=list)
    huwiki_url: str | None = None
    huwiki_title: str | None = None

    @property
    def genre(self) -> str | None:
        """Genres joined into a single comma-separated string, or ``None``."""
        return ", ".join(self.genres) if self.genres else None


def qid_from_uri(uri: str) -> str:
    """Extract the bare entity id from a Wikidata entity URI.

    Args:
        uri: A full entity URI, e.g. ``"http://www.wikidata.org/entity/Q42"``.

    Returns:
        The trailing entity id (the part after the last ``/``).

    Examples:
        >>> qid_from_uri("http://www.wikidata.org/entity/Q151944")
        'Q151944'
        >>> qid_from_uri("Q42")
        'Q42'
    """
    return uri.rsplit("/", 1)[-1]


def trim_date(value: str | None) -> str | None:
    """Reduce a Wikidata dateTime literal to its ``YYYY-MM-DD`` head.

    Wikidata returns dates as full timestamps (``"1945-08-15T00:00:00Z"``);
    we keep only the date part. Leading ``+`` (used by xsd:dateTime) is dropped.

    Args:
        value: A Wikidata date literal, or ``None``/empty.

    Returns:
        The ``YYYY-MM-DD`` (or shorter) head, or ``None`` if absent.

    Examples:
        >>> trim_date("1945-08-15T00:00:00Z")
        '1945-08-15'
        >>> trim_date("+1969-01-01T00:00:00Z")
        '1969-01-01'
        >>> trim_date("") is None
        True
        >>> trim_date(None) is None
        True
    """
    if not value:
        return None
    head = value.lstrip("+")[:10]
    return head or None


def title_from_huwiki_url(url: str | None) -> str | None:
    """Recover the article title from a Hungarian Wikipedia URL.

    Args:
        url: A ``https://hu.wikipedia.org/wiki/<Title>`` URL, or ``None``.

    Returns:
        The percent-decoded title with underscores turned back into spaces, or
        ``None`` if no URL was given.

    Examples:
        >>> title_from_huwiki_url("https://hu.wikipedia.org/wiki/Omega_(egy%C3%BCttes)")
        'Omega (együttes)'
        >>> title_from_huwiki_url("https://hu.wikipedia.org/wiki/Kar%C3%A1csony_J%C3%A1nos")
        'Karácsony János'
        >>> title_from_huwiki_url(None) is None
        True
    """
    if not url:
        return None
    slug = url.rsplit("/wiki/", 1)[-1]
    return urllib.parse.unquote(slug).replace("_", " ")


def _binding_str(binding: dict, key: str) -> str | None:
    """Return ``binding[key]["value"]`` if present, else ``None``."""
    cell = binding.get(key)
    return cell["value"] if cell else None


def parse_artist_binding(binding: dict) -> ArtistInfo:
    """Reduce one SPARQL result binding to an :class:`ArtistInfo`.

    Each binding carries at most one ``genre`` cell; :func:`merge_artists`
    folds the per-genre duplicate rows for the same entity back together.

    Args:
        binding: One element of ``results.bindings`` from a WDQS JSON response.

    Returns:
        The extracted :class:`ArtistInfo` (genres holding 0 or 1 entry).

    Examples:
        >>> b = {
        ...     "item": {"value": "http://www.wikidata.org/entity/Q151944"},
        ...     "name": {"value": "Omega"},
        ...     "kind": {"value": "group"},
        ...     "begin": {"value": "1962-01-01T00:00:00Z"},
        ...     "genre": {"value": "progresszív rock"},
        ...     "article": {"value": "https://hu.wikipedia.org/wiki/Omega_(egy%C3%BCttes)"},
        ... }
        >>> info = parse_artist_binding(b)
        >>> (info.qid, info.name, info.kind, info.begin_date)
        ('Q151944', 'Omega', 'group', '1962-01-01')
        >>> (info.genres, info.huwiki_title)
        (['progresszív rock'], 'Omega (együttes)')
    """
    qid = qid_from_uri(binding["item"]["value"])
    genre = _binding_str(binding, "genre")
    url = _binding_str(binding, "article")
    return ArtistInfo(
        qid=qid,
        name=_binding_str(binding, "name") or qid,
        kind=_binding_str(binding, "kind"),
        mb_artist_id=_binding_str(binding, "mbid"),
        begin_date=trim_date(_binding_str(binding, "begin")),
        end_date=trim_date(_binding_str(binding, "end")),
        birthplace=_binding_str(binding, "birthplace"),
        genres=[genre] if genre else [],
        huwiki_url=url,
        huwiki_title=title_from_huwiki_url(url),
    )


def merge_artists(infos: Iterable[ArtistInfo]) -> list[ArtistInfo]:
    """Collapse per-genre duplicate rows into one entry per entity.

    Optional fields like ``P136`` (genre) multiply a SPARQL result into one row
    per value. This groups by ``qid``, keeping the first non-null scalar field
    and unioning genres (sorted, de-duplicated). Input order is preserved.

    Args:
        infos: Per-binding :class:`ArtistInfo` objects (see
            :func:`parse_artist_binding`).

    Returns:
        One merged :class:`ArtistInfo` per distinct ``qid``.

    Examples:
        >>> a = ArtistInfo(qid="Q1", name="X", genres=["rock"])
        >>> b = ArtistInfo(qid="Q1", name="X", begin_date="1980", genres=["pop"])
        >>> merged = merge_artists([a, b])
        >>> len(merged), merged[0].genres, merged[0].begin_date
        (1, ['pop', 'rock'], '1980')
    """
    by_qid: dict[str, ArtistInfo] = {}
    for info in infos:
        current = by_qid.get(info.qid)
        if current is None:
            by_qid[info.qid] = ArtistInfo(
                qid=info.qid,
                name=info.name,
                kind=info.kind,
                mb_artist_id=info.mb_artist_id,
                begin_date=info.begin_date,
                end_date=info.end_date,
                birthplace=info.birthplace,
                genres=list(info.genres),
                huwiki_url=info.huwiki_url,
                huwiki_title=info.huwiki_title,
            )
            continue
        current.kind = current.kind or info.kind
        current.mb_artist_id = current.mb_artist_id or info.mb_artist_id
        current.begin_date = current.begin_date or info.begin_date
        current.end_date = current.end_date or info.end_date
        current.birthplace = current.birthplace or info.birthplace
        current.huwiki_url = current.huwiki_url or info.huwiki_url
        current.huwiki_title = current.huwiki_title or info.huwiki_title
        for genre in info.genres:
            if genre not in current.genres:
                current.genres.append(genre)
    for info in by_qid.values():
        info.genres.sort()
    return list(by_qid.values())


def hungarian_artists_query(*, limit: int, offset: int) -> str:
    """Build the paginated SPARQL query for Hungarian musicians and bands.

    The inner sub-select paginates over *distinct entities* (so ``LIMIT``/
    ``OFFSET`` stay correct despite the genre fan-out in the outer query):
    Hungarian-citizen humans with a music occupation, plus musical groups whose
    origin/country is Hungary. The outer query attaches optional facts and labels.

    Args:
        limit: Number of distinct entities per page.
        offset: Entity offset for pagination.

    Returns:
        A SPARQL query string.

    Examples:
        >>> q = hungarian_artists_query(limit=10, offset=20)
        >>> "LIMIT 10 OFFSET 20" in q
        True
        >>> "wd:Q28" in q and "wdt:P434" in q
        True
    """
    occupations = " ".join(f"wd:{qid}" for qid in _MUSIC_OCCUPATIONS)
    return f"""
SELECT DISTINCT ?item ?kind ?name ?mbid ?begin ?end ?birthplace ?genre ?article
WHERE {{
  {{
    SELECT DISTINCT ?item ?kind WHERE {{
      {{
        ?item wdt:P27 wd:Q28 ; wdt:P106 ?occ .
        VALUES ?occ {{ {occupations} }}
        BIND("person" AS ?kind)
      }}
      UNION
      {{
        ?item wdt:P31 wd:Q215380 .
        {{ ?item wdt:P495 wd:Q28 }} UNION {{ ?item wdt:P17 wd:Q28 }}
        BIND("group" AS ?kind)
      }}
    }}
    ORDER BY ?item
    LIMIT {limit} OFFSET {offset}
  }}
  OPTIONAL {{ ?item wdt:P434 ?mbid . }}
  OPTIONAL {{ ?item wdt:P569 ?birth . }}
  OPTIONAL {{ ?item wdt:P571 ?inception . }}
  OPTIONAL {{ ?item wdt:P570 ?death . }}
  OPTIONAL {{ ?item wdt:P576 ?dissolution . }}
  BIND(COALESCE(?birth, ?inception) AS ?begin)
  BIND(COALESCE(?death, ?dissolution) AS ?end)
  OPTIONAL {{ ?item wdt:P19 ?birthplaceItem . }}
  OPTIONAL {{ ?item wdt:P136 ?genreItem . }}
  OPTIONAL {{
    ?article schema:about ?item ; schema:isPartOf <{HUWIKI_PREFIX}> .
  }}
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "hu,en" .
    ?item rdfs:label ?name .
    ?birthplaceItem rdfs:label ?birthplace .
    ?genreItem rdfs:label ?genre .
  }}
}}
""".strip()


class WikidataClient:
    """Polite async client over WDQS and the Hungarian Wikipedia API."""

    def __init__(
        self, *, delay: float = 1.0, timeout: float = 60.0, max_retries: int = 4
    ) -> None:
        """Configure the client.

        Args:
            delay: Minimum seconds between requests (politeness; WDQS limits by
                query *time* per minute, so a ~1s gap is comfortable).
            timeout: Per-request timeout in seconds (SPARQL can be slow).
            max_retries: Attempts for transient failures (timeouts, 429/5xx)
                before giving up.
        """
        self._limiter = RateLimiter(delay)
        self._timeout = timeout
        self._max_retries = max_retries
        self._headers = {
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        }

    async def _get(self, url: str, params: dict, accept: str) -> httpx.Response:
        """GET ``url`` with politeness + transient-retry, honouring Retry-After.

        Args:
            url: Absolute URL to fetch.
            params: Query parameters.
            accept: ``Accept`` header value (JSON flavour expected back).

        Returns:
            The successful :class:`httpx.Response`.

        Raises:
            httpx.HTTPError: On a non-retryable error or exhausted retries.
        """
        headers = {**self._headers, "Accept": accept}
        last_exc: httpx.HTTPError | None = None
        for attempt in range(self._max_retries):
            await self._limiter.acquire()
            try:
                async with httpx.AsyncClient(
                    timeout=self._timeout, headers=headers, follow_redirects=True
                ) as client:
                    resp = await client.get(url, params=params)
            except httpx.TransportError as exc:  # timeouts, connection drops
                last_exc = exc
            else:
                if resp.status_code not in _RETRY_STATUS:
                    resp.raise_for_status()  # non-retryable 4xx -> propagate
                    return resp
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    await asyncio.sleep(min(float(retry_after), 60.0))
                    continue
            await asyncio.sleep(min(2.0**attempt, 30.0))
        assert last_exc is not None
        raise last_exc

    async def sparql(self, query: str) -> list[dict]:
        """Run a SPARQL query and return its result bindings.

        Args:
            query: A SPARQL ``SELECT`` query.

        Returns:
            The ``results.bindings`` list (one dict per row), possibly empty.
        """
        resp = await self._get(
            WDQS_ENDPOINT,
            {"query": query, "format": "json"},
            accept="application/sparql-results+json",
        )
        return resp.json().get("results", {}).get("bindings", [])

    async def fetch_hungarian_artists(
        self, *, page_size: int = 500, max_pages: int | None = None
    ) -> list[ArtistInfo]:
        """Page through all Hungarian musicians/bands and merge the results.

        Args:
            page_size: Distinct entities requested per SPARQL page.
            max_pages: Optional cap on pages fetched (for small/test runs).

        Returns:
            Merged :class:`ArtistInfo` objects (one per Wikidata entity).
        """
        merged: dict[str, ArtistInfo] = {}
        offset = 0
        page = 0
        while max_pages is None or page < max_pages:
            bindings = await self.sparql(
                hungarian_artists_query(limit=page_size, offset=offset)
            )
            if not bindings:
                break
            for info in merge_artists(parse_artist_binding(b) for b in bindings):
                merged[info.qid] = info
            distinct = {b["item"]["value"] for b in bindings}
            offset += len(distinct)
            page += 1
            if len(distinct) < page_size:
                break
        return list(merged.values())

    async def huwiki_extract(self, title: str) -> str | None:
        """Fetch the intro (lead) paragraph of a Hungarian Wikipedia article.

        Args:
            title: The article title (see :func:`title_from_huwiki_url`).

        Returns:
            The plain-text intro extract, or ``None`` if the article is missing
            or has no extract.
        """
        resp = await self._get(
            HUWIKI_API,
            {
                "action": "query",
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "redirects": 1,
                "format": "json",
                "titles": title,
            },
            accept="application/json",
        )
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            if "missing" in page:
                continue
            extract = (page.get("extract") or "").strip()
            if extract:
                return extract
        return None
