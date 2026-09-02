#!/usr/bin/env bash
# Waits for `run_s36.sh` to finish, then runs checkpoint selection on both seeds.
#
# Selection is on the fixed `SAVE_EVERY_N_STEPS` grid, never on `val_loss`, which is why the
# config saves every 500 steps and monitors nothing.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python

while pgrep -f "semkine/train.py --config configs/semkine/s36_eventgnn" > /dev/null 2>&1; do
  sleep 60
done

for seed in 3407 3408; do
  dir="outputs/semkine/s36_eventgnn_s${seed}"
  [ -d "$dir" ] || continue
  echo "[$(date +%H:%M:%S)] selecting ${seed}"
  $PY tools/select_checkpoint.py --run-dir "$dir" > "logs/s36_select_s${seed}.log" 2>&1
  echo "[$(date +%H:%M:%S)] selected ${seed} rc=$?"
done
echo "S36_SELECTION_DONE"
