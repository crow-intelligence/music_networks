# Diachronic lyrics analytics + dashboard — roadmap

Analysis layer over the Hungarian song-lyrics corpus (`data/music.db`). Groups
lyrics by decade and surfaces, in a dashboard:

- **(a) Data overview** — songs per decade; distinct performers / authors /
  composers per decade; dated-coverage by source.
- **(b) Topics over time** — BERTopic, prevalence per decade.
- **(c) Distinctive words per decade** — keyness (log-ratio + log-likelihood G²).
- **(d) Word-usage change** — diachronic semantic change via Crow Intelligence's
  [`chronowords`](https://github.com/crow-intelligence/chronowords).

NLP foundation: **huspacy** lemmatization → significant **bi/tri-gram** detection
(gensim `Phrases`) → vectorization.

## Locked design decisions
- **Lemmatization:** huspacy `hu_core_news_lg` (replaces the old emagyar HTTP
  service / dropped spaCy-2 model).
- **Semantic change:** `chronowords` `SVDAlgebra` (PPMI + SVD + Procrustes
  alignment across decades) — same method as the crowintelligence.org/analysis
  presidential-rhetoric page. (kenon = synchronic; optional, only for per-decade
  semantic-network visuals.)
- **Embeddings:** `NYTK/sentence-transformers-experimental-hubert-hungarian`
  (Hungarian-native huBERT). `max_seq_length=128` ⇒ chunk songs into ≤128-token
  windows and mean-pool.
- **Lemmatize for the count layer, not for embeddings** — BERTopic embeds natural
  text; keyness / chronowords / c-TF-IDF use lemmatized + n-gram tokens.
- **Language filter:** `langdetect` (seeded), Hungarian-only.
- **Dedup:** MinHash-LSH (`datasketch`) over normalized-lyrics shingles — keep one
  canonical doc per near-duplicate cluster. (Embeddings are *not* used for dedup —
  they over-merge same-artist songs; reserve them for optional cover detection.)
- **Dashboard:** Streamlit + Plotly, themed to crowintelligence.org/analysis
  (minimal light, sans-serif, crow branding; semantic-shift heatmaps +
  intensity-over-time line charts + interactive legend toggles).

## Caveats
- Only ~40% of songs are dated (growing); **pre-1980 decades are thin** — flag
  low-n decades; chronowords/SVD is reliable mainly for 1990s–2020s.
- **Lyrics are copyrighted** — the dashboard shows aggregates only, never full
  reproduced lyrics.

## Phases (each independently useful)
- **Phase 0 — corpus foundation** (`src/lyrics/corpus.py`): SQLite → langdetect
  hu filter → MinHash-LSH dedup → two views per song (raw text for embeddings;
  huspacy lemmas, POS-filtered N/V/Adj/Adv, stop-worded) → gensim `Phrases`
  bi/trigrams. Cache to `data/processed/` by decade.
- **Phase 1 — descriptive stats + keyness**:
  - 1a `src/lyrics/stats.py` (**done**): per-decade song / performer / author /
    composer counts. `uv run python -m src.lyrics.stats`.
  - 1b modernize `src/lyrics/decade_keywords.py`: per-decade keyness (corpus-toolkit
    log-ratio + log-likelihood G²) on the Phase-0 corpus.
- **Phase 2 — topics over time** (`src/lyrics/topics_bertopic.py`): BERTopic with
  huBERT embeddings (chunked) + custom CountVectorizer (lemmatized n-grams) for
  c-TF-IDF; `topics_over_time(timestamps=decade)`.
- **Phase 3 — word-usage change** (`src/lyrics/usage_change.py`): chronowords
  `SVDAlgebra` per decade + Procrustes alignment + change scores / neighbours.
- **Phase 4 — dashboard** (`src/dashboard/app.py` + `.streamlit/config.toml`):
  pages (1) Data overview, (2) Topics over time, (3) Distinctive words,
  (4) Word-usage explorer. Each annotates decade sample size.

## Dependencies
Opt-in `[project.optional-dependencies] nlp` extra (mirrors `deeplearning`):
huspacy (+ `hu_core_news_lg`), bertopic, sentence-transformers, umap-learn,
hdbscan, chronowords, datasketch, streamlit, plotly. (`langdetect`, `gensim`,
`corpus-toolkit`, `nltk`, `networkx`, `sklearn` already present; `kenon` optional.)
`uv sync --extra nlp` + download the huspacy model.

## Build order
**Now:** Phase 0 + Phase 1, on the current ~35k dated songs (re-run later as the
crawl + dating enlarge the corpus). Then Phase 2 → 3 → 4. Only `huspacy` +
`datasketch` are added now; heavier ML deps come with their phases.

## Conventions (per CLAUDE.md)
Type hints; clean `uv run ty check`; Google docstrings + `Examples:` doctests on
pure helpers; hypothesis tests for them; **no model loading at import time** (keep
modules import-safe, add to `testpaths`). Heavy ML deps stay in the `nlp` extra.

---

## Appendix — LLM-assisted dating (separate experiment)
Optionally fill the year gap MB/Discogs miss with an LLM (Gemini-in-Sheets /
Claude), but **validate first**: a self-eval showed LLMs are reliable for famous
songs yet **confidently wrong on the obscure long tail** (which is most of the
undated set) and ambiguous for covers/folk. Plan: eval harness on ~300 known-year
songs → measure exact/±1/±2 accuracy → only then bulk-fill, tagged
`first_release_source='llm'` at the **lowest** priority tier, above a confidence
threshold (optionally requiring Claude+Gemini agreement). Files: a sample
exporter + scorer, `export-undated` / `import-llm-years` CLI mirroring
`import_years`, and an `llm` entry in the qa year-by-source line.
