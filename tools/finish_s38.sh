#!/usr/bin/env bash
# Waits for `run_s38.sh` to finish, then runs checkpoint selection on both seeds.
# Selection is on the fixed `SAVE_EVERY_N_STEPS` grid, never on `val_loss`.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python

while pgrep -f "semkine/train.py --config configs/semkine/s38_jointreadout" > /dev/null 2>&1; do
  sleep 60
done

# #region agent log
DBG=/data1/lyq/code/mesh/EventHands1/.cursor/debug-ea00d3.log
log_dbg() { echo "{\"sessionId\":\"ea00d3\",\"runId\":\"s38\",\"hypothesisId\":\"$1\",\"message\":\"$2\",\"location\":\"tools/finish_s38.sh\",\"data\":$3,\"timestamp\":$(date +%s000)}" >> "$DBG"; }
# #endregion

for seed in 3407 3408; do
  dir="outputs/semkine/s38_jointreadout_s${seed}"
  [ -d "$dir" ] || continue
  echo "[$(date +%H:%M:%S)] selecting ${seed}"
  $PY tools/select_checkpoint.py --run-dir "$dir" > "logs/s38_select_s${seed}.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] selected ${seed} rc=$rc"
  # #region agent log
  sel=$(ls "$dir"/selection_*.json 2>/dev/null | head -1)
  ra=$( [ -n "$sel" ] && $PY -c "import json;d=json.load(open('$sel'));s=d.get('selected') or {};print(json.dumps({'step':s.get('step'),'ra':s.get('mpjpe_ra_mm'),'grid_median':d.get('grid_median')}))" 2>/dev/null || echo '{}' )
  log_dbg "H5" "seed selected" "{\"seed\":${seed},\"rc\":${rc},\"selection\":${ra:-{}}}"
  # #endregion
done
echo "S38_SELECTION_DONE"
