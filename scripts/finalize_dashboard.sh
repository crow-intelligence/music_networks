#!/bin/bash
# Unattended finalizer for the emotion + genre runs.
#
# Every tick it: (1) restarts either run if it died before finishing (both are
# resumable — a no-op if already complete), (2) re-aggregates and rebuilds the
# dashboard so it reflects progress. When both runs are complete (no growth for
# two consecutive ticks and neither process alive) it does a final rebuild and
# writes data/processed/FINALIZED.
#
# Launched detached (setsid/nohup) by the assistant; safe to re-run any time.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=data/processed/finalize.log
TICK=1200 # 20 min
mkdir -p data/processed
echo "$(date '+%F %T') finalizer started" >>"$LOG"

DISCOGS_TOKEN=$(grep -iE '^discogs_token=' .env | cut -d= -f2- | tr -d "\"'")
export DISCOGS_TOKEN

emo_running() { pgrep -f 'src\.lyrics\.emotion run' >/dev/null 2>&1; }
gen_running() { pgrep -f 'src\.enrich\.genre run' >/dev/null 2>&1; }
count() { wc -l <"$1" 2>/dev/null || echo 0; }

restart_emotion() {
  TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false \
    setsid nohup uv run python -m src.lyrics.emotion run --batch-size 32 \
    >>data/processed/emotion/run.log 2>&1 </dev/null &
  echo "$(date '+%F %T') restarted emotion run" >>"$LOG"
}
restart_genre() {
  setsid nohup uv run python -m src.enrich.genre run \
    >>data/processed/genre/run.log 2>&1 </dev/null &
  echo "$(date '+%F %T') restarted genre run" >>"$LOG"
}

prev_e=-1
prev_g=-1
stable=0
while :; do
  sleep "$TICK"
  e=$(count data/processed/emotion/per_song.jsonl)
  g=$(count data/processed/genre/per_song.jsonl)

  # Resurrect a run that died before finishing (resume is idempotent).
  if ! emo_running && [ "$e" -gt "$prev_e" ] 2>/dev/null; then :; fi
  if ! emo_running; then restart_emotion; fi
  if ! gen_running; then restart_genre; fi

  uv run python -m src.lyrics.emotion aggregate >>"$LOG" 2>&1
  uv run python -m src.enrich.genre aggregate >>"$LOG" 2>&1
  uv run python -m src.dashboard.build >>"$LOG" 2>&1
  echo "$(date '+%F %T') rebuilt: emotion=$e genre=$g" >>"$LOG"

  # Completion: counts unchanged for two ticks (the restarts above are quick
  # no-ops once everything is processed).
  if [ "$e" = "$prev_e" ] && [ "$g" = "$prev_g" ]; then
    stable=$((stable + 1))
  else
    stable=0
  fi
  prev_e=$e
  prev_g=$g
  if [ "$stable" -ge 2 ]; then break; fi
done

uv run python -m src.lyrics.emotion aggregate >>"$LOG" 2>&1
uv run python -m src.enrich.genre aggregate >>"$LOG" 2>&1
uv run python -m src.dashboard.build >>"$LOG" 2>&1
touch data/processed/FINALIZED
echo "$(date '+%F %T') FINALIZED (emotion=$prev_e genre=$prev_g)" >>"$LOG"
