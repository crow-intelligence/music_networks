#!/bin/bash
# Orchestrate the artist-enrichment steps that depend on the Wikidata artist
# table, in order, unattended:
#   1. pageviews  — score artists already fetched (runs immediately)
#   2. wait for the bulk `wikidata` run (facts + bios) to finish
#   3. wd-resolve — per-name fallback for performers the bulk pass missed
#   4. pageviews  — score the newly-resolved artists
# Writes data/processed/ENRICH_FINALIZED when done. Launched detached.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=data/processed/enrich_finalize.log
mkdir -p data/processed
echo "$(date '+%F %T') enrich-finalizer started" >>"$LOG"

# 1. Cultural-salience pageviews for the artists the bulk pass already fetched.
uv run python -m src.enrich.pageviews >>"$LOG" 2>&1
echo "$(date '+%F %T') pageviews (pass 1) done" >>"$LOG"

# 2. Wait for the bulk Wikidata enumeration (facts + bios) to finish.
while pgrep -f 'src\.enrich wikidata' >/dev/null 2>&1; do
  sleep 300
done
echo "$(date '+%F %T') bulk wikidata finished" >>"$LOG"

# 3. Per-name fallback for still-unlinked performers credited on >= 2 songs.
uv run python -m src.enrich wd-resolve --min-songs 2 >>"$LOG" 2>&1
echo "$(date '+%F %T') wd-resolve done" >>"$LOG"

# 4. Score the artists the fallback just added.
uv run python -m src.enrich.pageviews >>"$LOG" 2>&1
echo "$(date '+%F %T') pageviews (pass 2) done" >>"$LOG"

touch data/processed/ENRICH_FINALIZED
echo "$(date '+%F %T') ENRICH FINALIZED" >>"$LOG"
