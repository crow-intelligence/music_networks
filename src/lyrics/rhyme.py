"""Rhyme analysis of Hungarian song lyrics.

Hungarian orthography is close to phonemic, so rhymes can be detected fairly well
from spelling — provided the **digraphs** (``cs gy ly ny sz ty zs dz`` + ``dzs``)
are tokenized as single consonants and vowel **length** (``a/á`` …) is handled.
From the raw, line-broken lyrics this module:

* splits into stanzas / lines and takes each line's final word,
* classifies a line-pair's rhyme as **tiszta** (pure — last syllable incl. coda
  matches), **asszonánc** (assonance — vowels match, coda differs), or neither,
  flagging **ragrím** (suffix rhyme — the match is merely a shared grammatical
  ending, the agglutinating language's "cheap" rhyme),
* labels each stanza's **rhyme scheme** (``AABB`` / ``ABAB`` / ``ABBA`` …),
* and reduces a song to a **rhyme density** + type/scheme counts.

Aggregated **by decade and by genre** (rap should rhyme far more densely than
folk). The pure helpers are deterministic and doctested; the DB/corpus loading is
behind functions so the module stays import-safe.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypeGuard

DEFAULT_RHYME_DIR = Path("data/processed/rhyme")

VOWELS = set("aáeéiíoóöőuúüű")
# Length-folding for assonance (quality kept; only length normalized).
_DEACUTE = str.maketrans("áéíóőúű", "aeioöuü")
# Hungarian multigraphs, longest first, tokenized as single consonants.
_DIGRAPHS = ("dzs", "cs", "dz", "gy", "ly", "ny", "sz", "ty", "zs")
_HU_LETTERS = set("aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz")

# Common grammatical endings that make a "ragrím" (suffix rhyme) when shared.
_SUFFIXES = frozenset(
    {
        "ban",
        "ben",
        "ba",
        "be",
        "ból",
        "ből",
        "ra",
        "re",
        "ról",
        "ről",
        "nak",
        "nek",
        "val",
        "vel",
        "tól",
        "től",
        "hoz",
        "hez",
        "höz",
        "nál",
        "nél",
        "ig",
        "ként",
        "ul",
        "ül",
        "kor",
        "ok",
        "ek",
        "ök",
        "ak",
        "om",
        "em",
        "öm",
        "od",
        "ed",
        "öd",
        "ja",
        "je",
        "unk",
        "ünk",
        "tok",
        "tek",
        "tök",
        "uk",
        "ük",
        "tam",
        "tem",
        "tál",
        "tél",
        "ott",
        "ett",
        "ött",
        "na",
        "ne",
        "nék",
        "lak",
        "lek",
        "juk",
        "jük",
        "ván",
        "vén",
        "va",
        "ve",
        "ság",
        "ség",
        "ás",
        "és",
        "tlan",
        "tlen",
        "talan",
        "telen",
    }
)

_BRACKET_LINE = re.compile(r"^\s*[\[(].*[\])]\s*$")
_WORD = re.compile(r"[a-záéíóöőúüűA-ZÁÉÍÓÖŐÚÜŰ]+")


def to_phonemes(word: str) -> list[str]:
    """Tokenize a Hungarian word into phonemes (digraphs as single units).

    Args:
        word: A word (case/punctuation are ignored).

    Returns:
        Lowercase phoneme tokens; multigraphs (``sz``, ``dzs`` …) stay together.

    Examples:
        >>> to_phonemes("Egészség")
        ['e', 'g', 'é', 'sz', 's', 'é', 'g']
        >>> to_phonemes("asszony")
        ['a', 's', 'sz', 'o', 'ny']
        >>> to_phonemes("Ady!")
        ['a', 'd', 'y']
    """
    text = "".join(ch for ch in word.lower() if ch in _HU_LETTERS)
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i : i + 3] in _DIGRAPHS:
            out.append(text[i : i + 3])
            i += 3
        elif text[i : i + 2] in _DIGRAPHS:
            out.append(text[i : i + 2])
            i += 2
        else:
            out.append(text[i])
            i += 1
    return out


def rhyme_tail(phonemes: list[str], nuclei: int = 1) -> list[str]:
    """Return the rhyme tail: from the ``nuclei``-th-last vowel to the end.

    Args:
        phonemes: Phonemes (see :func:`to_phonemes`).
        nuclei: How many trailing vowel nuclei to include.

    Returns:
        The tail phonemes (the whole word if it has too few vowels).

    Examples:
        >>> rhyme_tail(to_phonemes("virág"))
        ['á', 'g']
        >>> rhyme_tail(to_phonemes("szerelem"), nuclei=2)
        ['e', 'l', 'e', 'm']
        >>> rhyme_tail(to_phonemes("brr"))
        ['b', 'r', 'r']
    """
    vowel_idx = [i for i, p in enumerate(phonemes) if p[0] in VOWELS]
    if len(vowel_idx) < nuclei:
        return list(phonemes)
    return phonemes[vowel_idx[-nuclei] :]


def _fold(phonemes: list[str]) -> list[str]:
    """Length-fold the vowels of a phoneme list (for assonance comparison)."""
    return [p.translate(_DEACUTE) for p in phonemes]


def common_suffix(a: str, b: str) -> str:
    """Longest common trailing substring of two words (lowercased).

    Args:
        a: First word.
        b: Second word.

    Returns:
        The shared ending (possibly empty).

    Examples:
        >>> common_suffix("látok", "várok")
        'ok'
        >>> common_suffix("virág", "világ")
        'ág'
        >>> common_suffix("alma", "körte")
        ''
    """
    a, b = a.lower(), b.lower()
    i = 0
    while i < len(a) and i < len(b) and a[-1 - i] == b[-1 - i]:
        i += 1
    return a[len(a) - i :] if i else ""


def classify_rhyme(a: str, b: str) -> tuple[str, bool] | None:
    """Classify the rhyme between two line-final words.

    Args:
        a: First line-final word.
        b: Second line-final word.

    Returns:
        ``None`` if they do not rhyme (or are the same word); otherwise
        ``(kind, ragrim)`` where ``kind`` is ``"tiszta"`` (pure) or
        ``"asszonánc"`` (assonance) and ``ragrim`` is ``True`` when the rhyme is
        merely a shared grammatical suffix.

    Examples:
        >>> classify_rhyme("virág", "világ")      # pure, genuine
        ('tiszta', False)
        >>> classify_rhyme("látok", "várok")      # pure but a suffix rhyme
        ('tiszta', True)
        >>> classify_rhyme("ház", "vár")          # vowels match, coda differs
        ('asszonánc', False)
        >>> classify_rhyme("alma", "körte") is None
        True
        >>> classify_rhyme("szív", "szív") is None   # önrím (identical) ignored
        True
    """
    pa, pb = to_phonemes(a), to_phonemes(b)
    if not pa or not pb or pa == pb:
        return None
    ta, tb = rhyme_tail(pa), rhyme_tail(pb)
    if not any(p[0] in VOWELS for p in ta):
        return None
    ragrim = common_suffix(a, b) in _SUFFIXES
    if ta == tb:
        return ("tiszta", ragrim)
    fa, fb = _fold(ta), _fold(tb)
    va = [p for p in fa if p[0] in VOWELS]
    vb = [p for p in fb if p[0] in VOWELS]
    if va and va == vb:
        return ("asszonánc", ragrim)
    return None


def split_stanzas(lyrics: str) -> list[list[str]]:
    r"""Split raw lyrics into stanzas of content lines.

    Blank lines separate stanzas; ``[Verse]`` / ``(Refrén)`` marker lines and
    empty lines are dropped. Handles both ``\n`` and ``/`` line separators.

    Args:
        lyrics: Raw lyrics text.

    Returns:
        A list of stanzas, each a list of non-empty content lines.

    Examples:
        >>> split_stanzas("[Verse]\negy\nkettő\n\nhárom\nnégy")
        [['egy', 'kettő'], ['három', 'négy']]
        >>> split_stanzas("a / b / c")
        [['a', 'b', 'c']]
    """
    text = lyrics.replace("/", "\n")
    stanzas: list[list[str]] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [
            ln.strip()
            for ln in block.splitlines()
            if ln.strip() and not _BRACKET_LINE.match(ln)
        ]
        if lines:
            stanzas.append(lines)
    return stanzas


def last_word(line: str) -> str | None:
    """Return a line's final word (letters only), or ``None``.

    Examples:
        >>> last_word("Ha átölelsz és átölellek én.")
        'én'
        >>> last_word("...!!!") is None
        True
    """
    words = _WORD.findall(line)
    return words[-1] if words else None


def scheme_label(words: list[str | None]) -> str:
    """Label a stanza's rhyme scheme (``AABB`` / ``ABAB`` / ``ABCB`` …).

    Each line gets the letter of the first earlier line that **purely** rhymes
    with it (or repeats the same final word — common in refrains); otherwise a
    fresh letter. Only pure (``tiszta``) rhyme assigns a shared letter — assonance
    is too permissive and would collapse distinct rhymes. A line with no usable
    final word gets ``.``.

    Args:
        words: The stanza's per-line final words (``None`` allowed).

    Returns:
        The scheme string.

    Examples:
        >>> scheme_label(["virág", "ház", "világ", "láz"])
        'ABAB'
        >>> scheme_label(["szív", "ív", "rím", "rím"])
        'AABB'
    """
    reps: list[tuple[str, str]] = []
    labels: list[str] = []
    nxt = 0
    for w in words:
        if not w:
            labels.append(".")
            continue
        letter = None
        for rep_word, rep_letter in reps:
            res = classify_rhyme(w, rep_word)
            if w.lower() == rep_word.lower() or (res and res[0] == "tiszta"):
                letter = rep_letter
                break
        if letter is None:
            letter = chr(ord("A") + nxt)
            nxt += 1
            reps.append((w, letter))
        labels.append(letter)
    return "".join(labels)


def analyze_lyrics(lyrics: str) -> dict[str, Any]:
    r"""Compute a song's rhyme statistics from its raw lyrics.

    Args:
        lyrics: Raw lyrics text.

    Returns:
        ``{"lines", "rhymed_lines", "density", "types", "schemes"}`` where
        ``density`` is rhymed-lines / lines, ``types`` counts rhyming pairs by
        ``tiszta`` / ``asszonánc`` / ``ragrím``, and ``schemes`` counts the
        schemes of four-line stanzas.

    Examples:
        >>> st = analyze_lyrics("virág\nház\nvilág\nláz")
        >>> st["lines"], st["rhymed_lines"], st["schemes"]
        (4, 4, {'ABAB': 1})
        >>> st["density"]
        1.0
    """
    types: Counter[str] = Counter()
    schemes: Counter[str] = Counter()
    total_lines = 0
    rhymed_lines = 0
    for stanza in split_stanzas(lyrics):
        ends = [last_word(ln) for ln in stanza]
        total_lines += len(ends)
        letters = scheme_label(ends)
        counts = Counter(letters)
        rhymed_lines += sum(1 for c in letters if c != "." and counts[c] > 1)
        if len(ends) == 4 and "." not in letters:
            schemes[letters] += 1
        # Count rhyming pairs within the stanza by type.
        for i in range(len(ends)):
            for j in range(i + 1, len(ends)):
                a, b = ends[i], ends[j]
                if not a or not b:
                    continue
                got = classify_rhyme(a, b)
                if got:
                    kind, ragrim = got
                    types[kind] += 1
                    if ragrim:
                        types["ragrím"] += 1
    return {
        "lines": total_lines,
        "rhymed_lines": rhymed_lines,
        "density": round(rhymed_lines / total_lines, 4) if total_lines else 0.0,
        "types": dict(types),
        "schemes": dict(schemes),
    }


# ---------------------------------------------------------------------------
# Corpus run + aggregation
# ---------------------------------------------------------------------------
def iter_corpus_lyrics(
    db_path: str, corpus_dir: Path | str
) -> Iterator[tuple[int, int, str]]:
    """Yield ``(song_id, decade, raw_lyrics)`` for the (Hungarian, dated) corpus.

    Reuses the corpus song-id set (Hungarian-filtered, de-duplicated, dated) and
    the dating policy, but returns the **raw** line-broken lyrics rhyme analysis
    needs (the corpus's cached ``text`` has line breaks stripped).

    Args:
        db_path: SQLite path.
        corpus_dir: Corpus JSONL directory (defines the song-id universe).

    Yields:
        ``(song_id, decade, raw_lyrics)``.
    """
    from src.lyrics.corpus import iter_dated_lyrics, load_corpus

    corpus_ids = {d.song_id for d in load_corpus(corpus_dir)}
    for song_id, decade, lyrics in iter_dated_lyrics(db_path):
        if song_id in corpus_ids:
            yield song_id, decade, lyrics


def _done_ids(path: Path) -> set[int]:
    """Song ids already present in ``per_song.jsonl`` (for resuming)."""
    done: set[int] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    done.add(int(json.loads(line)["song_id"]))
                except (ValueError, KeyError, json.JSONDecodeError):
                    continue
    return done


def run(
    db_path: str = "data/music.db",
    corpus_dir: Path | str = "data/processed/corpus",
    out_dir: Path | str = DEFAULT_RHYME_DIR,
    *,
    limit: int | None = None,
    resume: bool = True,
) -> int:
    """Analyze every corpus song's rhymes, appending to ``per_song.jsonl``.

    Args:
        db_path: SQLite path.
        corpus_dir: Corpus JSONL directory.
        out_dir: Output directory.
        limit: Optional cap on songs this run.
        resume: Skip song ids already analyzed.

    Returns:
        The number of songs analyzed this run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_song = out_dir / "per_song.jsonl"
    done = _done_ids(per_song) if resume else set()
    written = 0
    with per_song.open("a", encoding="utf-8") as handle:
        for song_id, decade, lyrics in iter_corpus_lyrics(db_path, corpus_dir):
            if song_id in done:
                continue
            if limit is not None and written >= limit:
                break
            stats = analyze_lyrics(lyrics)
            handle.write(
                json.dumps(
                    {"song_id": song_id, "decade": decade, **stats},
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    return written


def _mean(values: list[float]) -> float:
    """Mean of a list (0.0 if empty)."""
    return sum(values) / len(values) if values else 0.0


def aggregate(
    rows: list[dict[str, Any]], genre_of: dict[int, str] | None = None
) -> dict[str, Any]:
    """Aggregate per-song rhyme rows by decade and (optionally) genre.

    Args:
        rows: Per-song rhyme dicts (see :func:`analyze_lyrics` + ``song_id`` /
            ``decade``).
        genre_of: Optional ``song_id → genre`` map for the by-genre breakdown.

    Returns:
        ``{"overall", "by_decade", "by_genre", "schemes"}``. Each scope carries
        mean ``density``, the rhyme-type mix (shares), and ``n`` songs;
        ``schemes`` is the overall four-line scheme distribution (top first).

    Examples:
        >>> rows = [
        ...     {"song_id": 1, "decade": 2010, "lines": 4, "rhymed_lines": 4,
        ...      "density": 1.0, "types": {"tiszta": 2}, "schemes": {"AABB": 1}},
        ...     {"song_id": 2, "decade": 2010, "lines": 4, "rhymed_lines": 2,
        ...      "density": 0.5, "types": {"asszonánc": 1}, "schemes": {}},
        ... ]
        >>> agg = aggregate(rows, {1: "Hip Hop", 2: "Pop"})
        >>> agg["overall"]["n"], agg["overall"]["density"]
        (2, 0.75)
        >>> agg["by_genre"]["Hip Hop"]["density"]
        1.0
        >>> agg["schemes"][0]
        ['AABB', 1]
        >>> agg["schemes_total"]
        1
    """

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        type_tot: Counter[str] = Counter()
        for r in group:
            type_tot.update(r.get("types", {}))
        base = type_tot["tiszta"] + type_tot["asszonánc"]
        return {
            "n": len(group),
            "density": round(_mean([r["density"] for r in group]), 4),
            "types": {
                "tiszta": round(type_tot["tiszta"] / base, 4) if base else 0.0,
                "asszonánc": round(type_tot["asszonánc"] / base, 4) if base else 0.0,
                "ragrím": round(type_tot["ragrím"] / base, 4) if base else 0.0,
            },
        }

    by_decade: dict[int, list[dict[str, Any]]] = {}
    by_genre: dict[str, list[dict[str, Any]]] = {}
    schemes: Counter[str] = Counter()
    for r in rows:
        by_decade.setdefault(int(r["decade"]), []).append(r)
        schemes.update(r.get("schemes", {}))
        if genre_of and (g := genre_of.get(int(r["song_id"]))):
            by_genre.setdefault(g, []).append(r)

    return {
        "overall": summarize(rows),
        "by_decade": [
            {"decade": d, **summarize(by_decade[d])} for d in sorted(by_decade)
        ],
        "by_genre": {g: summarize(rs) for g, rs in by_genre.items()},
        "schemes": [list(x) for x in schemes.most_common(8)],
        # Total four-line stanzas analyzed, so the dashboard can show each scheme
        # as a share of all quatrains (not just of the top-8 shown).
        "schemes_total": sum(schemes.values()),
    }


def build_artifacts(
    out_dir: Path | str = DEFAULT_RHYME_DIR,
    genre_dir: Path | str = "data/processed/genre",
) -> dict[str, Any]:
    """Aggregate ``per_song.jsonl`` into ``rhyme.json`` (joining genre).

    Args:
        out_dir: Directory holding ``per_song.jsonl``.
        genre_dir: Directory with the genre ``per_song.jsonl`` (for the by-genre
            breakdown).

    Returns:
        The aggregated dict (see :func:`aggregate`).
    """
    out_dir = Path(out_dir)
    rows: list[dict[str, Any]] = []
    per_song = out_dir / "per_song.jsonl"
    if per_song.exists():
        with per_song.open(encoding="utf-8") as handle:
            rows = [json.loads(ln) for ln in handle if ln.strip()]
    genre_of: dict[int, str] = {}
    genre_path = Path(genre_dir) / "per_song.jsonl"
    if genre_path.exists():
        with genre_path.open(encoding="utf-8") as handle:
            for ln in handle:
                if ln.strip():
                    rec = json.loads(ln)
                    if rec.get("genre"):
                        genre_of[int(rec["song_id"])] = rec["genre"]
    agg = aggregate(rows, genre_of)
    (out_dir / "rhyme.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return agg


# ---------------------------------------------------------------------------
# Example mining (real, short, attributed rhyme examples for the dashboard)
# ---------------------------------------------------------------------------
_MIN_EX_WORD = 3
_MAX_EX_LINE = 60
_HU_ACCENTS = set("áéíóöőúű")
# Leading verse/refrain markers to strip from a displayed example line
# (e.g. "2. Érzem…", "R. Még nem…") — they're not part of the lyric.
_ENUM_MARKER = re.compile(r"^\s*(?:\d{1,2}|[Rr]|Ref)\.\s*")
# Skipped only when choosing the *showcased* pick (kept among candidates), so a
# front-page example isn't crude — the analysis itself keeps every word.
_HARSH = frozenset({"fasz", "geci", "kurva", "bazmeg", "baszni", "baszik", "picsa"})


def _clean_ex_word(w: str | None) -> TypeGuard[str]:
    """Whether a line-final word is presentable in an example (letters, len≥3).

    Examples:
        >>> _clean_ex_word("virág"), _clean_ex_word("ó"), _clean_ex_word(None)
        (True, False, False)
    """
    if not w or len(w) < _MIN_EX_WORD:
        return False
    return all(c in _HU_LETTERS for c in w.lower())


def _clean_ex_line(line: str) -> bool:
    """Whether a stanza line is short and well-formed enough to showcase.

    Examples:
        >>> _clean_ex_line("Szállj el kismadár")
        True
        >>> _clean_ex_line("a")
        False
    """
    return 3 <= len(line) <= _MAX_EX_LINE and len(_WORD.findall(line)) >= 2


def _hungarianish(lines: list[str]) -> bool:
    """Whether a stanza looks Hungarian (has an accented vowel) — drops English.

    Examples:
        >>> _hungarianish(["szállj a virág", "fáj a láz"])
        True
        >>> _hungarianish(["Legends dyin' too young", "so why believe in you"])
        False
    """
    return any(c in _HU_ACCENTS for line in lines for c in line.lower())


def _title_case_hu(title: str) -> str:
    """Capitalize a title's first letter — zeneszoveg titles arrive lowercased.

    Existing capitals are left intact (so e.g. "Ha én Rózsa Volnék" is unchanged).

    Examples:
        >>> _title_case_hu("az első szerelem")
        'Az első szerelem'
        >>> _title_case_hu("Ha én Rózsa Volnék")
        'Ha én Rózsa Volnék'
    """
    for i, ch in enumerate(title):
        if ch.isalpha():
            return title[:i] + ch.upper() + title[i + 1:]
    return title


def _pair_category(a: str, b: str) -> str | None:
    """Map a rhyming word pair to the example category to showcase it under.

    A suffix rhyme is shown as ``ragrím`` (its defining trait); otherwise the
    genuine ``tiszta`` / ``asszonánc`` kind.

    Examples:
        >>> _pair_category("virág", "világ")
        'tiszta'
        >>> _pair_category("látok", "várok")
        'ragrím'
        >>> _pair_category("ház", "vár")
        'asszonánc'
    """
    got = classify_rhyme(a, b)
    if not got:
        return None
    kind, ragrim = got
    return "ragrím" if ragrim else kind


def song_examples(
    lyrics: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    r"""Mine one example rhyme pair per type + one stanza per scheme from a song.

    Args:
        lyrics: Raw line-broken lyrics.

    Returns:
        ``(pairs, stanzas)`` where ``pairs`` maps a type (``asszonánc`` /
        ``tiszta`` / ``ragrím``) to ``{"words": [a, b], "lines": [lineA, lineB]}``
        (the rhyming word pair + the two source lines, for a hover excerpt), and
        ``stanzas`` maps a four-line scheme (``AABB`` …) to the first clean
        four-line stanza found. All shown lines have ≥2 words and verse markers
        stripped.

    Examples:
        >>> lyr = "szállj a virág\nüt a ház\nszállj a világ\nfáj a láz"
        >>> pairs, stanzas = song_examples(lyr)
        >>> pairs["tiszta"]["words"]
        ['virág', 'világ']
        >>> pairs["tiszta"]["lines"]
        ['szállj a virág', 'szállj a világ']
        >>> stanzas["ABAB"]
        ['szállj a virág', 'üt a ház', 'szállj a világ', 'fáj a láz']
    """
    pairs: dict[str, dict[str, Any]] = {}
    stanzas: dict[str, list[str]] = {}
    for stanza in split_stanzas(lyrics):
        disp = [_ENUM_MARKER.sub("", ln).strip() for ln in stanza]
        ends = [last_word(ln) for ln in stanza]
        for i, a in enumerate(ends):
            if not _clean_ex_word(a) or not _clean_ex_line(disp[i]):
                continue
            for off, b in enumerate(ends[i + 1 :]):
                j = i + 1 + off
                if not _clean_ex_word(b) or a.lower() == b.lower():
                    continue
                if not _clean_ex_line(disp[j]):
                    continue
                cat = _pair_category(a, b)
                if cat and cat not in pairs:
                    pairs[cat] = {"words": [a, b], "lines": [disp[i], disp[j]]}
        if len(ends) == 4 and all(_clean_ex_word(w) for w in ends):
            letters = scheme_label(ends)
            if "." not in letters and letters not in stanzas:
                if all(_clean_ex_line(ln) for ln in disp) and _hungarianish(disp):
                    stanzas[letters] = disp
    return pairs, stanzas


def build_examples(
    db_path: str = "data/music.db",
    out_dir: Path | str = DEFAULT_RHYME_DIR,
    *,
    scan_cap: int = 4000,
    per_category: int = 6,
) -> dict[str, Any]:
    """Mine real, attributed rhyme examples, biased toward well-known artists.

    Scans songs in descending order of their performer's Wikipedia salience
    (``Artist.avg_monthly_pageviews``), so examples come from recognizable songs.
    Writes ``examples.json`` with a chosen ``pick`` + ``candidates`` per rhyme
    type and per four-line scheme. The ``pick`` avoids the harshest words; every
    example is a short excerpt with ``performer`` / ``title`` / ``url`` attribution.

    Args:
        db_path: SQLite path.
        out_dir: Output directory (``examples.json`` is written here).
        scan_cap: Maximum songs to scan (salience-ordered).
        per_category: Candidates to keep per type / per scheme.

    Returns:
        The examples dict (also written to ``examples.json``).
    """
    from sqlalchemy import select

    from src.db import (
        Artist,
        Performer,
        Song,
        SongExternal,
        SongPerformer,
        get_engine,
        make_session,
    )
    from src.enrich.match import prettify_name

    engine = get_engine(db_path)
    try:
        with make_session(engine)() as session:
            # Best (highest-salience) performer per song.
            best: dict[int, tuple[str, float]] = {}
            for sid, name, views in session.execute(
                select(Song.id, Performer.name, Artist.avg_monthly_pageviews)
                .join(SongPerformer, SongPerformer.song_id == Song.id)
                .join(Performer, Performer.id == SongPerformer.performer_id)
                .outerjoin(Artist, Artist.id == Performer.artist_id)
            ):
                score = float(views) if views is not None else -1.0
                if sid not in best or score > best[sid][1]:
                    best[sid] = (name, score)
            urls = {
                sid: url
                for sid, url in session.execute(
                    select(SongExternal.song_id, SongExternal.url).where(
                        SongExternal.source == "zeneszoveg"
                    )
                )
            }
            order = sorted(best, key=lambda s: best[s][1], reverse=True)

            type_cands: dict[str, list[dict[str, Any]]] = {}
            scheme_cands: dict[str, list[dict[str, Any]]] = {}
            # At most one candidate per performer per category, so the list spans
            # many recognizable artists instead of the single most-salient one.
            type_seen: dict[str, set[str]] = {}
            scheme_seen: dict[str, set[str]] = {}
            scanned = 0
            for sid in order:
                if scanned >= scan_cap:
                    break
                song = session.get(Song, sid)
                if not song or not song.lyrics:
                    continue
                scanned += 1
                name, views = best[sid]
                attr = {
                    "performer": prettify_name(name),
                    "title": _title_case_hu(song.title),
                    "url": urls.get(sid),
                    "views": round(views) if views >= 0 else None,
                }
                pairs, stanzas = song_examples(song.lyrics)
                for cat, ex in pairs.items():
                    bucket = type_cands.setdefault(cat, [])
                    seen = type_seen.setdefault(cat, set())
                    if len(bucket) < per_category and name not in seen:
                        bucket.append({**ex, **attr})
                        seen.add(name)
                for scheme, lines in stanzas.items():
                    bucket = scheme_cands.setdefault(scheme, [])
                    seen = scheme_seen.setdefault(scheme, set())
                    if len(bucket) < per_category and name not in seen:
                        bucket.append({"lines": lines, **attr})
                        seen.add(name)
    finally:
        engine.dispose()

    def _harsh(strings: list[str]) -> bool:
        toks = {w.lower() for s in strings for w in _WORD.findall(s)}
        return bool(toks & _HARSH)

    def _block(cands: list[dict[str, Any]], field: str) -> dict[str, Any]:
        clean = [c for c in cands if not _harsh(c[field])]
        return {"pick": (clean or cands)[0], "candidates": cands}

    examples = {
        "types": {k: _block(v, "words") for k, v in type_cands.items()},
        "schemes": {k: _block(v, "lines") for k, v in scheme_cands.items()},
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return examples


def main() -> None:
    """CLI: ``run`` rhyme analysis, or ``aggregate`` the cached results."""
    import argparse

    ap = argparse.ArgumentParser(description="Hungarian lyrics rhyme analysis.")
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="analyze corpus songs (resumable)")
    r.add_argument("--db", default="data/music.db")
    r.add_argument("--corpus-dir", default="data/processed/corpus")
    r.add_argument("--out-dir", default=str(DEFAULT_RHYME_DIR))
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--no-resume", action="store_true")
    sub.add_parser("aggregate", help="build rhyme.json from per-song results")
    ex = sub.add_parser(
        "examples", help="mine real attributed rhyme examples (examples.json)"
    )
    ex.add_argument("--db", default="data/music.db")
    ex.add_argument("--out-dir", default=str(DEFAULT_RHYME_DIR))
    ex.add_argument("--scan-cap", type=int, default=4000)

    args = ap.parse_args()
    if args.command == "run":
        n = run(
            args.db,
            args.corpus_dir,
            args.out_dir,
            limit=args.limit,
            resume=not args.no_resume,
        )
        print(f"Analyzed {n} songs -> {Path(args.out_dir) / 'per_song.jsonl'}")
        agg = build_artifacts(args.out_dir)
        print(
            f"Overall rhyme density {agg['overall']['density']:.2f} "
            f"over {agg['overall']['n']} songs."
        )
    elif args.command == "aggregate":
        agg = build_artifacts()
        print(f"Aggregated {agg['overall']['n']} songs -> rhyme.json")
    elif args.command == "examples":
        ex = build_examples(args.db, args.out_dir, scan_cap=args.scan_cap)
        print(
            f"Mined {len(ex['types'])} type + {len(ex['schemes'])} scheme "
            f"examples -> {Path(args.out_dir) / 'examples.json'}"
        )


if __name__ == "__main__":
    main()
