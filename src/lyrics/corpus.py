"""Build the per-decade lyrics corpus that the analysis layer consumes.

From the SQLite ``song`` table this produces, for each dated, Hungarian,
non-duplicate song, **two views**:

* ``text`` — cleaned natural-language lyrics (for transformer embeddings), and
* ``tokens`` — huspacy lemmas (POS-filtered to content words, stop-words removed)
  with significant **bi/tri-grams** folded into single tokens by a gensim
  ``Phrases`` model (for keyness / chronowords / c-TF-IDF).

Pipeline: load dated lyrics → ``langdetect`` Hungarian-only filter → MinHash-LSH
near-duplicate removal (``datasketch``) → lemmatize (``hu_core_news_lg``) →
gensim ``Phrases``. Output is cached as per-decade JSONL under
``data/processed/corpus/``.

The pure helpers (cleaning, shingling, Jaccard, POS filter, decade bucketing) are
deterministic and doctested. The heavy dependencies (spaCy model, datasketch) are
imported *inside* functions so this module stays import-safe without the ``nlp``
extra installed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from src.db import Song, get_engine, make_session
from src.enrich.match import normalize
from src.lyrics.stats import song_decade

DEFAULT_CORPUS_DIR = Path("data/processed/corpus")

# Content-word POS tags (Universal Dependencies) we keep, mirroring the legacy
# pipeline's Adj/V/Adv/N filter.
KEEP_POS = frozenset({"NOUN", "VERB", "ADJ", "ADV"})
_HU_MODEL = "hu_core_news_lg"

# Bracketed structure annotations on lyrics pages (e.g. "[Refrén]", "[Verse 2]").
_BRACKETED = re.compile(r"\[[^\]]*\]")
_WS = re.compile(r"\s+")


@dataclass
class SongDoc:
    """One corpus document: a song's two text views plus its decade.

    Attributes:
        song_id: The :class:`~src.db.Song` id.
        decade: Release decade (e.g. ``1990``).
        text: Cleaned natural-language lyrics (for embeddings).
        tokens: Lemmatized content tokens with n-grams folded in (for counts).
    """

    song_id: int
    decade: int
    text: str
    tokens: list[str]


def clean_lyrics(raw: str) -> str:
    """Clean raw lyrics into natural text for embedding.

    Drops bracketed section markers (``[Refrén]``), turns the site's ``/`` line
    separators into spaces, and collapses whitespace. Case and diacritics are
    kept (the transformer wants natural text).

    Args:
        raw: Raw lyrics text.

    Returns:
        Cleaned single-spaced text.

    Examples:
        >>> clean_lyrics("[Verse]  Egy sor / Másik sor   [Refrén] Hajrá")
        'Egy sor Másik sor Hajrá'
        >>> clean_lyrics("   ")
        ''
    """
    no_marks = _BRACKETED.sub(" ", raw)
    no_slash = no_marks.replace("/", " ")
    return _WS.sub(" ", no_slash).strip()


def word_shingles(text: str, k: int = 3) -> set[str]:
    """Return the set of ``k``-word shingles of ``text`` (normalized).

    Used as the MinHash input for near-duplicate detection. Text is normalized
    (lowercased, deaccented, depunctuated) first so trivial spelling/scrape
    differences don't change the shingles.

    Args:
        text: Input text.
        k: Shingle size in words.

    Returns:
        The set of space-joined ``k``-word shingles (empty if too few words).

    Examples:
        >>> sorted(word_shingles("a b c d", k=2))
        ['a b', 'b c', 'c d']
        >>> word_shingles("egy", k=2)
        set()
        >>> word_shingles("Egy KÉT egy", k=1) == {"egy", "ket"}
        True
    """
    words = normalize(text).split()
    if len(words) < k:
        return set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    """Return the Jaccard similarity of two sets (0.0 for two empty sets).

    Args:
        a: First set.
        b: Second set.

    Returns:
        ``|a ∩ b| / |a ∪ b|``, or ``0.0`` when both are empty.

    Examples:
        >>> jaccard({"a", "b"}, {"b", "c"})
        0.3333333333333333
        >>> jaccard(set(), set())
        0.0
        >>> jaccard({"a"}, {"a"})
        1.0
    """
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def keep_token(pos: str, is_stop: bool, is_alpha: bool) -> bool:
    """Whether a token survives content-word filtering.

    Args:
        pos: Universal-Dependencies POS tag (spaCy ``token.pos_``).
        is_stop: spaCy ``token.is_stop``.
        is_alpha: spaCy ``token.is_alpha``.

    Returns:
        ``True`` for non-stop alphabetic tokens whose POS is in :data:`KEEP_POS`.

    Examples:
        >>> keep_token("NOUN", False, True)
        True
        >>> keep_token("NOUN", True, True)
        False
        >>> keep_token("ADP", False, True)
        False
        >>> keep_token("NOUN", False, False)
        False
    """
    return is_alpha and not is_stop and pos in KEEP_POS


def iter_dated_lyrics(
    db_path: str = "data/music.db", *, limit: int | None = None
) -> Iterator[tuple[int, int, str]]:
    """Yield ``(song_id, decade, lyrics)`` for every dated, lyrics-bearing song.

    Args:
        db_path: Path to the SQLite database file.
        limit: Optional cap on songs yielded (for small/test runs).

    Yields:
        ``(song_id, decade, lyrics)`` tuples (decade via :func:`song_decade`).
    """
    engine = get_engine(db_path)
    yielded = 0
    try:
        with make_session(engine)() as session:
            rows = session.execute(
                Song.__table__.select().with_only_columns(
                    Song.id, Song.first_release_year, Song.year, Song.lyrics
                )
            )
            for song_id, fry, year, lyrics in rows:
                if limit is not None and yielded >= limit:
                    return
                if not lyrics or not lyrics.strip():
                    continue
                decade = song_decade(fry, year)
                if decade is None:
                    continue
                yield song_id, decade, lyrics
                yielded += 1
    finally:
        engine.dispose()


def is_hungarian(text: str, threshold: float = 0.90) -> bool:
    """Whether ``text`` is detected as Hungarian above ``threshold`` probability.

    Uses ``langdetect`` with a fixed seed for determinism. Very short / empty
    text returns ``False``.

    Args:
        text: Text to classify.
        threshold: Minimum probability for the top language to be ``hu``.

    Returns:
        ``True`` if the most probable language is Hungarian above ``threshold``.
    """
    from langdetect import DetectorFactory, detect_langs

    DetectorFactory.seed = 0
    stripped = text.strip()
    if len(stripped) < 20:
        return False
    try:
        langs = detect_langs(stripped)
    except Exception:  # noqa: BLE001 - langdetect raises on unusable input
        return False
    return bool(langs) and langs[0].lang == "hu" and langs[0].prob >= threshold


def dedup_canonical_ids(
    items: list[tuple[int, str]], *, threshold: float = 0.8, num_perm: int = 128
) -> set[int]:
    """Return one canonical song id per near-duplicate-lyrics cluster.

    MinHash-LSH (``datasketch``) over 3-word shingles groups near-identical
    lyrics; from each cluster the smallest id is kept. Songs with too little
    text to shingle are always kept.

    Args:
        items: ``(song_id, lyrics)`` pairs.
        threshold: Jaccard threshold for the LSH index.
        num_perm: MinHash permutations.

    Returns:
        The set of song ids to keep (canonical representatives).
    """
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: dict[int, MinHash] = {}
    keep: set[int] = set()
    for song_id, lyrics in sorted(items):  # ascending id => smallest kept first
        shingles = word_shingles(lyrics, k=3)
        if not shingles:
            keep.add(song_id)
            continue
        mh = MinHash(num_perm=num_perm)
        for sh in shingles:
            mh.update(sh.encode("utf-8"))
        if lsh.query(mh):  # a near-duplicate already kept => skip this one
            continue
        lsh.insert(str(song_id), mh)
        minhashes[song_id] = mh
        keep.add(song_id)
    return keep


def lemmatize(nlp, texts: Iterable[str]) -> Iterator[list[str]]:
    """Lemmatize cleaned texts into content-word token lists.

    Args:
        nlp: A loaded spaCy/huspacy pipeline.
        texts: Cleaned lyric texts.

    Yields:
        Per-text lists of lowercased content lemmas (see :func:`keep_token`).
    """
    for doc in nlp.pipe(texts, batch_size=64):
        yield [
            token.lemma_.lower()
            for token in doc
            if keep_token(token.pos_, token.is_stop, token.is_alpha)
        ]


def fold_ngrams(
    token_lists: list[list[str]], *, min_count: int = 20, threshold: float = 10.0
) -> list[list[str]]:
    """Fold significant bi- then tri-grams into single tokens (gensim Phrases).

    Args:
        token_lists: Per-document lemma lists.
        min_count: Ignore n-grams rarer than this.
        threshold: Phrase score threshold (higher = fewer, stronger phrases).

    Returns:
        The token lists with detected multiword expressions joined by ``_``.
    """
    from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS, Phrases

    bigram = Phrases(
        token_lists,
        min_count=min_count,
        threshold=threshold,
        connector_words=ENGLISH_CONNECTOR_WORDS,
    )
    bigrammed = [bigram[doc] for doc in token_lists]
    trigram = Phrases(bigrammed, min_count=min_count, threshold=threshold)
    return [list(trigram[doc]) for doc in bigrammed]


def build_corpus(
    db_path: str = "data/music.db",
    *,
    hungarian_only: bool = True,
    dedup: bool = True,
    limit: int | None = None,
) -> list[SongDoc]:
    """Build the full per-decade corpus (filter → dedup → lemmatize → n-grams).

    Args:
        db_path: Path to the SQLite database file.
        hungarian_only: Drop non-Hungarian lyrics via :func:`is_hungarian`.
        dedup: Remove near-duplicate lyrics via :func:`dedup_canonical_ids`.
        limit: Optional cap on songs read (for small/test runs).

    Returns:
        One :class:`SongDoc` per surviving song.
    """
    import spacy

    rows = list(iter_dated_lyrics(db_path, limit=limit))
    if hungarian_only:
        rows = [(sid, dec, ly) for sid, dec, ly in rows if is_hungarian(ly)]
    if dedup:
        keep = dedup_canonical_ids([(sid, ly) for sid, dec, ly in rows])
        rows = [(sid, dec, ly) for sid, dec, ly in rows if sid in keep]

    cleaned = [clean_lyrics(ly) for _, _, ly in rows]
    nlp = spacy.load(_HU_MODEL, disable=["parser", "ner"])
    token_lists = list(lemmatize(nlp, cleaned))
    token_lists = fold_ngrams(token_lists)

    return [
        SongDoc(song_id=sid, decade=dec, text=text, tokens=tokens)
        for (sid, dec, _), text, tokens in zip(rows, cleaned, token_lists, strict=True)
    ]


def save_corpus(docs: list[SongDoc], out_dir: Path | str = DEFAULT_CORPUS_DIR) -> None:
    """Write the corpus to one JSONL file per decade under ``out_dir``.

    Args:
        docs: The built corpus (see :func:`build_corpus`).
        out_dir: Output directory (created if missing); one
            ``decade_<d>.jsonl`` per decade, one :class:`SongDoc` per line.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_decade: dict[int, list[SongDoc]] = {}
    for doc in docs:
        by_decade.setdefault(doc.decade, []).append(doc)
    for decade, group in by_decade.items():
        path = out / f"decade_{decade}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for doc in group:
                handle.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")


def load_corpus(out_dir: Path | str = DEFAULT_CORPUS_DIR) -> list[SongDoc]:
    """Load a corpus previously written by :func:`save_corpus`.

    Args:
        out_dir: Directory holding the ``decade_<d>.jsonl`` files.

    Returns:
        The corpus as a list of :class:`SongDoc`, ordered by decade then file.
    """
    out = Path(out_dir)
    docs: list[SongDoc] = []
    for path in sorted(out.glob("decade_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    docs.append(SongDoc(**json.loads(line)))
    return docs


def main() -> None:
    """Build the corpus from the default DB and cache it to JSONL."""
    docs = build_corpus()
    save_corpus(docs)
    from collections import Counter

    by_decade = Counter(doc.decade for doc in docs)
    print(f"Built {len(docs)} corpus docs -> {DEFAULT_CORPUS_DIR}")
    for decade in sorted(by_decade):
        print(f"  {decade}s : {by_decade[decade]}")


if __name__ == "__main__":
    main()
