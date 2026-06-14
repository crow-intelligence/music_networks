#!/bin/bash
# Watch the XLM-EMO emotion re-run; restart it if it dies before finishing,
# re-aggregate + rebuild the dashboard periodically, and do a final rebuild once
# it's complete (writes data/processed/EMOTION_REFIT_DONE). Launched detached.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=data/processed/emotion_finalize.log
TICK=1200
echo "$(date '+%F %T') emotion finalizer started" >>"$LOG"

running() { pgrep -f 'src\.lyrics\.emotion run' >/dev/null 2>&1; }
count() { wc -l <data/processed/emotion/per_song.jsonl 2>/dev/null || echo 0; }

restart() {
  TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false \
    setsid nohup uv run python -m src.lyrics.emotion run --batch-size 32 \
    >>data/processed/emotion/run.log 2>&1 </dev/null &
  echo "$(date '+%F %T') restarted emotion run" >>"$LOG"
}

prev=-1
stable=0
while :; do
  sleep "$TICK"
  n=$(count)
  if ! running; then restart; fi
  uv run python -m src.lyrics.emotion aggregate >>"$LOG" 2>&1
  uv run python -m src.dashboard.build >>"$LOG" 2>&1
  echo "$(date '+%F %T') rebuilt: emotion=$n" >>"$LOG"
  if [ "$n" = "$prev" ]; then stable=$((stable + 1)); else stable=0; fi
  prev=$n
  if [ "$stable" -ge 2 ]; then break; fi
done

uv run python -m src.lyrics.emotion aggregate >>"$LOG" 2>&1
uv run python -m src.dashboard.build >>"$LOG" 2>&1
touch data/processed/EMOTION_REFIT_DONE
echo "$(date '+%F %T') EMOTION REFIT DONE (n=$prev)" >>"$LOG"
