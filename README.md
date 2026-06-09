# Analyzing Hungarian Pop Songs as Networks

Scrapes Hungarian song lyrics and credits from [zeneszoveg.hu](https://www.zeneszoveg.hu),
**dates** each song from open music databases, and analyzes the lyrics **by
decade** — distinctive words, topics, semantic change, and word-association
networks. The result is a single, **standalone data-essay dashboard**
(*Magyar dalszövegek hálózata*) you can drop onto any static web host.

It's the renewed, larger version of a
[2020 Crow Intelligence analysis](https://blog.crowintelligence.org/hu/2020/08/19/hatvan-ev-dalszovegei/)
and is related to the academic
[HuMus project](https://doi.org/10.1007/s44427-026-00030-x).

## Tech stack

- **Python 3.12** (via [pyenv](https://github.com/pyenv/pyenv); pinned in `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** for dependencies and environments
- **[ruff](https://docs.astral.sh/ruff/)** (lint + format), **[ty](https://github.com/astral-sh/ty)** (types),
  **pytest + Hypothesis + doctest** (tests)
- **SQLite** storage (`data/music.db`) via SQLAlchemy 2.0
- NLP/analysis: **huspacy** (lemmas), **BERTopic** + huBERT (topics),
  **chronowords** (diachronic SVD embeddings), **[kenon](https://pypi.org/project/kenon/)**
  (co-occurrence graphs), **wordcloud** (era clouds)

See [`CLAUDE.md`](CLAUDE.md) for coding conventions (Google-style docstrings,
doctests, property tests, import-safety).

## Setup

```bash
uv sync                       # create the venv and install dependencies
uv sync --extra deeplearning  # also install tensorflow (only the legacy net3d.py needs it)
```

Run anything inside the environment with `uv run …`:

```bash
uv run pytest                 # doctests + property tests
uv run ruff check . && uv run ruff format .
uv run ty check src/lyrics src/dashboard
```

**Secrets** (e.g. a Discogs API token) live in a **gitignored `.env`**; never
commit tokens. The enrichment CLI reads `DISCOGS_TOKEN` from the environment or
`--token`.

## Pipeline

```
scrape  →  enrich/date  →  corpus  →  analysis  →  dashboard
```

### 1. Scrape lyrics → SQLite

The scraper is async, polite, and **resumable** — stop any time and re-run; it
only fetches what's still outstanding (progress is tracked in the DB).

```bash
uv run python -m src.scraper discover                        # seed artist initials
uv run python -m src.scraper crawl                           # scrape pending bands, then songs
uv run python -m src.scraper --proxies crawl                 # via rotating free proxies
uv run python -m src.scraper qa                              # success rate + completeness
```

> **Scraping etiquette.** zeneszoveg.hu's `robots.txt` sets a 2-second crawl-delay
> and blocks AI-training crawlers, citing lyrics copyright. The scraper paces
> requests accordingly and uses a normal browser user-agent; how much you scrape,
> and what you do with the lyrics, is your call as the project owner. The
> dashboard only ever publishes **aggregate** numbers — never full lyrics.

### 2. Enrich & date the songs

A song's **decade** is the backbone of the analysis, so dating quality matters.
Decades are taken **only** from authoritative first-release years —
`musicbrainz`, `discogs`, or hand-verified `manual` — falling back to the scraped
zeneszoveg page year (≈88% same-decade accurate). A **Genius page-year is never
trusted** (it's the lyrics-page upload year and mis-dates old songs into recent
decades).

```bash
uv run python -m src.enrich seed                             # seed Hungarian artists (MusicBrainz)
uv run python -m src.enrich dates                            # enumerate recordings → dated versions
uv run python -m src.enrich date-lyrics                      # date lyrics songs via targeted MB search
DISCOGS_TOKEN=$(grep -E '^discogs_token=' .env | cut -d= -f2-) \
  uv run python -m src.enrich date-discogs                   # date the MB-missed tail via Discogs
uv run python -m src.enrich wikidata                         # artist/band facts + Hungarian Wikipedia bios
uv run python -m src.enrich qa                               # dating-coverage report
```

Dating is resumable (cached in `enrich_state`); MusicBrainz is ~1 req/s and
Discogs ~60 req/min, so full passes run for hours — leave them in the background.
Create a Discogs personal access token at *discogs.com → Settings → Developers*.

### 3. Build the per-decade corpus

Hungarian-only filtering, MinHash-LSH near-duplicate removal, huspacy
lemmatization (content words), and gensim n-grams → one JSONL per decade.

```bash
uv run python -m src.lyrics.corpus                           # → data/processed/corpus/decade_*.jsonl
```

### 4. Analysis artifacts

```bash
uv run python -m src.lyrics.topics_bertopic                  # BERTopic topics → data/processed/topics/
uv run python -m src.lyrics.usage_change                     # diachronic semantic change → data/processed/usage/
uv run python -m src.lyrics.networks                         # kenon word graphs → data/processed/networks/
```

(Distinctive-word **keyness** and **descriptive stats** are computed on the fly
by the dashboard build from the corpus + DB.) After re-clustering topics,
re-author `data/processed/topics/topic_names.json` (`{topic_id: name}`) from each
topic's keywords — topic ids are not stable across runs.

### 5. Build & publish the dashboard

```bash
uv run python -m src.dashboard.build                         # → data/dashboard/
python -m http.server 8000 -d data/dashboard                 # preview at http://localhost:8000
```

`data/dashboard/` is **fully standalone** — `index.html` (inline CSS/JS, data
baked in) + `assets/fonts/` (self-hosted, subset woff2) + `assets/clouds/`
(per-decade word-cloud PNGs with fonts rendered in). No build step, no JS
dependencies, no asset CDNs. **To publish: copy the `data/dashboard/` folder to
your web root.** (It's a generated artifact under the gitignored `data/`, so
regenerate it any time with the build command.)

The dashboard shows, per decade (1960s–2020s):

- **Adatok évtizedenként** — songs / performers / lyricists / composers (bars)
- **Megkülönböztető szavak** — distinctive words (G² keyness) as era-styled word
  clouds, with a ranked-list fallback
- **Témák az időben** — BERTopic topics over time (count or within-decade share)
- **Jelentésváltozás / Eltérés-hőtérkép / Kumulatív sodródás** — diachronic
  semantic change (chronowords)
- **Szomszédos szavak** — per-decade word-association *ego networks* (kenon)
- **Szókincs- és modell-statisztika** — vocabulary size + lexical diversity (MATTR)
- **Módszertan** — methodology and sources

It's responsive and accessibility-checked (keyboard-operable, AA-contrast,
colorblind-safe palette).

## Legacy

The people-collaboration network code under `src/graphs/`, `src/graph_viz/`, and
`src/data_normalization/` predates the current tooling and is being modernized
incrementally (typing, docstrings, a move onto `src/db.py` + SQLite).
