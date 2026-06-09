# CLAUDE.md

Guidance for working in this repository. Read before making changes.

## What this project is

Analysis of Hungarian pop-song lyrics as networks. Lyrics are scraped from
zeneszoveg.hu into a MariaDB database, then processed into per-decade corpora,
keyword/topic models, and various network graphs/visualizations. Code lives as a
collection of scripts under `src/` (no installable package).

- Data store: **MariaDB** via `docker-compose.yml` (MySQL on `:3306`,
  phpMyAdmin on `:8080`). Code connects with the `mysql+pymysql://` SQLAlchemy URL.
- `src/data_tasks/` scraping, `src/lyrics/` NLP/corpus, `src/graphs/` network
  analysis, `src/graph_viz/` rendering, `src/data_normalization/` cleaning.

## Toolchain (use these — do not reintroduce the old workflow)

- **Python 3.12**, provided by **pyenv** (`.python-version` → `3.12.13`).
- **uv** manages the environment and dependencies. There is no `requirements.txt`
  and no `pip install`; everything is driven by `pyproject.toml` + `uv.lock`.
- **ruff** for linting **and** formatting (replaces black/isort).
- **ty** (Astral) for static type checking.
- **pytest + hypothesis + doctest** for testing.

### Everyday commands

```bash
uv sync                       # create/update the venv from pyproject + lock
uv sync --extra deeplearning  # also install tensorflow (only net3d.py needs it)
uv add <pkg>                  # add a dependency (updates pyproject + lock)
uv run pytest                 # run doctests + hypothesis tests
uv run ruff check .           # lint
uv run ruff format .          # format
uv run ty check <path>        # type-check
uv run python src/...py       # run a script inside the env
```

## Code conventions

- **Type hints on all new code**, kept clean under `uv run ty check`.
- **Google-style docstrings** (enforced by ruff's pydocstyle, `convention=google`).
- **Doctests**: every public function/method gets a docstring with an
  `Examples:` block of `>>>` doctests that demonstrate how to call it. These are
  documentation *and* tests — `pytest --doctest-modules` runs them.
- **Property-based tests**: cover invariants with **hypothesis** in `tests/`,
  alongside concrete doctests. Be strict — test edge cases, not just happy paths.
- Reference implementation showing all of the above: `src/utils.py`
  (`year_to_decade`) with its tests in `tests/test_utils.py`. Match that style.

### Import-safety rule (important for doctests)

`pytest` runs with `--doctest-modules`, but only over the modules listed in
`[tool.pytest.ini_options] testpaths` in `pyproject.toml`. Several legacy scripts
execute code at **import time** (e.g. `src/lyrics/process_songs.py` opens a DB
connection and runs queries at module level), so they cannot be collected safely.

When you touch such a script: move its top-level execution under
`if __name__ == "__main__":` (and into functions), then add the now import-safe
module to `testpaths` so its doctests run.

## Scraper (`src/scraper/`, data in `src/db.py`)

The scraper is async, polite, and resumable, writing to a single SQLite file
(`data/music.db`, WAL mode). It replaces the old threaded MariaDB scraper
(`src/data_tasks/`, removed).

```bash
uv run python -m src.scraper discover                 # seed band URLs (all initials)
uv run python -m src.scraper discover --initials a --max-bands 1   # small/targeted
uv run python -m src.scraper crawl                    # scrape pending bands, then songs
uv run python -m src.scraper crawl --limit 50         # bounded chunk
uv run python -m src.scraper --proxies crawl          # crawl via rotating free proxies
uv run python -m src.scraper qa                        # success % + completeness report
# global flags: --db PATH  --delay SECONDS (default 2.0)  --concurrency N  --proxies
# NOTE: global flags are defined on the top-level parser, so they must come
#       BEFORE the subcommand: `--proxies crawl`, not `crawl --proxies`.
```

The live site IP-bans high-volume direct traffic (TCP connection refused), so a
bulk crawl needs `--proxies`. Failed rows stay eligible while `attempts < 3`, so
re-running `--proxies crawl` mops up the ~30% that miss on a given pass; repeat
until `qa` plateaus.

Key design points (see `src/scraper/crawl.py`):
- **Resumable via the DB**: `crawl_state` tracks each URL as
  `pending`/`done`/`failed`. Stop any time; re-running `crawl` only fetches
  what's left. All writes are idempotent (`ON CONFLICT`).
- **Single-writer**: many async fetchers feed one writer coroutine through a
  queue — the only SQLite-safe way to write concurrently.
- **Politeness**: per-domain rate limiter honours the site's 2s crawl-delay;
  real browser UA. robots.txt blocks AI-bot UAs and asserts lyrics copyright —
  scrape responsibly (the user owns that decision).
- **Pure parsers** in `parse.py` (doctested) are the easy place to adapt if the
  site's HTML changes. Selectors were verified live on 2026-05-29.
- `tensorflow` extra and `--proxies` (free, self-healing pool) are opt-in.

## Dashboard (`src/dashboard/`)

A single-page data essay ("Magyar dalszövegek hálózata"). The source of truth is
`template.html` — inline `<style>` + one `<script>` that draws **raw SVG with
vanilla JS** (no D3/npm/bundler). `build.py` injects a `DATA` JSON blob into the
template's `/*__DATA__*/` placeholder and now emits a **folder**:
`data/dashboard/index.html` + `assets/` (self-hosted woff2 fonts + per-decade
word-cloud PNGs). Charts are interactive (hover tooltips, toggleable/keyboard
legends, small-multiples, cloud↔list and network↔list toggles); a11y-checked
(axe clean), responsive (wide charts scroll inside contained regions), AA palette.

```bash
uv run python -m src.lyrics.networks      # (re)build per-decade word graphs (kenon) -> data/processed/networks/
uv run python -m src.dashboard.build      # assemble DATA + render clouds + copy assets -> data/dashboard/
python -m http.server 8000 -d data/dashboard   # preview
```

Build inputs / pieces:
- **§3 word clouds** — `src/dashboard/clouds.py` renders one era-styled cloud
  image per decade (font size ∝ √G²), using build-only TTFs in
  `src/dashboard/assets/build-fonts/` (one open-licensed display face per decade;
  baked into the PNG, never shipped as web fonts). The ranked list stays as the
  analytical / screen-reader fallback.
- **§8 ego networks** — `src/lyrics/networks.py` builds per-decade co-occurrence
  graphs with `kenon` and exports compact adjacency payloads. Ego previews use
  the **full** co-occurrence top-neighbours (not the disparity-filter backbone —
  the backbone surfaces rare-word noise around frequent hubs). The standalone
  force-directed *explorer* page is a planned follow-up that reuses these graphs.
- Runtime fonts (IM Fell English, Courier Prime) are self-hosted subset woff2 in
  `src/dashboard/assets/fonts/`, copied next to the HTML at build time — the page
  has **no external network dependency**.
- **Lexical diversity** in §stats — `src/lyrics/diversity.py` computes per-decade
  **MATTR** (moving-average TTR, window 100; length-robust, unlike plain TTR),
  merged into the vocab table at build time.
- `MIN_DECADE = 1960` in `build.py` scopes the dashboard to 1960s+ (the sparse
  1900–1950 decades are filtered out; underlying artifacts kept).
- **Dating policy (important):** a song's decade comes from an authoritative
  first-release year (`AUTHORITATIVE_SOURCES = ("musicbrainz","discogs")` in
  `src/lyrics/stats.py`, reused by `corpus.iter_dated_lyrics` and the overview
  SQL) when available, **else the scraped zeneszoveg `Song.year`** as a
  decade-level fallback (~88% same-decade-accurate vs MB/Discogs). A
  **non-authoritative `first_release_year` (e.g. Genius page-year) is never
  trusted** — only `Song.year` is the fallback (the Genius page-year piled old
  songs into the 2010s/2020s and was cleared from the DB). To add precise dates
  to the fallback tail, run MB/Discogs enrichment (`src/enrich/{dating,discogs,
  musicbrainz}.py`). Caveat: MB/Discogs date the *matched recording*, so a song
  that only survives on a later compilation can still date late.
- **Secrets / API tokens** live in a **gitignored `.env`** (e.g. a Discogs
  personal access token). The enrich CLI reads `DISCOGS_TOKEN` from the env or
  `--token`; pass it explicitly, e.g.
  `DISCOGS_TOKEN=$(grep -E '^discogs_token=' .env | cut -d= -f2-) uv run python -m src.enrich date-discogs`.
  Never commit `.env` or paste tokens into code/logs.
- **Topic names need re-curation after any re-clustering.** `topic_names.json`
  maps `topic_id → name`, but ids are not stable across BERTopic re-runs, so a
  stale curated file silently mislabels new clusters — re-author it (from each
  topic's keywords) as the **last** step after a recompute. The current
  `topic_names.json` curates the **29-topic** clustering of the 43k-doc corpus;
  the original 28-topic curation is parked at `topic_names.curated-OLD-28topics.json`.
- New modules are import-safe (heavy deps imported inside functions) and carry
  doctests + hypothesis tests (`tests/test_networks.py`, `tests/test_clouds.py`,
  `tests/test_diversity.py`).

## Migration notes / known follow-ups

- Dependencies were unpinned from their 2020 versions and resolved fresh on 3.12,
  so major libs jumped versions (SQLAlchemy 1.3→2.0, numpy 1.x→2.x, pandas
  1.x→3.0). The toolchain is green, but the legacy scripts are **not yet verified
  at runtime** against these new majors — expect breaking-API fixes when you run
  them (e.g. SQLAlchemy 2.0 moved `declarative_base`).
- The old `hu_core_ud_lg` spaCy 2.x model is dropped (incompatible with spaCy 3).
  Wire up a current Hungarian model (huspacy / emagyar) at runtime when needed.
