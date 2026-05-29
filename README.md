# Analyzing Hungarian Pop Songs as Networks

Scrapes Hungarian song lyrics and credits from [zeneszoveg.hu](https://www.zeneszoveg.hu),
stores them in a single SQLite database, and analyzes them two ways: the
**language** of the lyrics over the decades, and the **collaboration network** of
the people who make the songs (performers, lyricists, composers, band members).

## Tech stack

- **Python 3.12** (via [pyenv](https://github.com/pyenv/pyenv); pinned in `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** for dependencies and environments
- **[ruff](https://docs.astral.sh/ruff/)** for linting + formatting
- **[ty](https://github.com/astral-sh/ty)** for type checking
- **pytest + Hypothesis + doctest** for tests
- **SQLite** for storage (`data/music.db`), via SQLAlchemy 2.0

See [`CLAUDE.md`](CLAUDE.md) for the coding conventions (Google-style docstrings,
doctests, property tests).

## Setup

```bash
uv sync                      # create the venv and install dependencies
uv sync --extra deeplearning # also install tensorflow (only net3d.py needs it)
```

Then run anything inside the environment with `uv run …`:

```bash
uv run pytest                # doctests + property tests
uv run ruff check . && uv run ruff format .
uv run ty check src/scraper src/db.py
```

## Scraping

The scraper is async, polite, and **resumable** — stop it any time and re-run; it
only fetches what's still outstanding. Progress is tracked in the database, so no
work is repeated.

```bash
uv run python -m src.scraper discover                          # seed all artist initials
uv run python -m src.scraper discover --initials a --max-bands 1   # small/targeted run
uv run python -m src.scraper crawl                             # scrape pending bands, then songs
uv run python -m src.scraper crawl --limit 50                  # bounded chunk
uv run python -m src.scraper qa                                # success rate + data completeness
```

Useful global flags: `--db PATH`, `--delay SECONDS` (default 1.5), `--concurrency N`,
`--proxies` (opt-in free-proxy rotation).

> **Note on scraping etiquette.** zeneszoveg.hu's `robots.txt` sets a 2-second
> crawl-delay and blocks AI-training crawlers, citing lyrics copyright. The
> scraper paces requests accordingly and uses a normal browser user-agent; how
> much you scrape, and what you do with the lyrics, is your call as the project
> owner.

## Pipeline

1. **Scrape** (`src/scraper/`) → SQLite (`src/db.py`).
2. **Lyrics** (`src/lyrics/`) → per-decade corpora → keyword/keyness, skip-gram
   PageRank, LDA topics, word clouds. Lemmatization uses a local Hungarian
   morphological tagger service (emagyar) on `http://127.0.0.1:5000`.
3. **People networks** (`src/graphs/`, `src/data_normalization/`) → co-credit
   graph → disparity-filter backbone → community detection, ego networks, and
   3D / D3 visualizations (`src/graph_viz/`).

> The analysis scripts under `src/lyrics/`, `src/graphs/`, etc. predate the
> tooling migration and are being modernized incrementally (typing, docstrings,
> and a move onto `src/db.py` + SQLite).
