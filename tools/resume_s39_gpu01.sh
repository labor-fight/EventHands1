#!/usr/bin/env bash
# S39 continuation under the GPU 0-1 constraint:
#   1. wait for the in-flight seed-3407 training to finish (it frees GPU 0 and 4);
#   2. wait for GPU 1 to be released by the external job currently on it;
#   3. train seed 3408 on CUDA_VISIBLE_DEVICES=0,1;
#   4. run checkpoint selection for both seeds (fixed grid, never val_loss).
set -uo pipefail
cd "$(dirname "$0")/.."

export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED=1

PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
mkdir -p logs

# #region agent log
DBG=/data1/lyq/code/mesh/EventHands1/.cursor/debug-ea00d3.log
log_dbg() { echo "{\"sessionId\":\"ea00d3\",\"runId\":\"s39\",\"hypothesisId\":\"$1\",\"message\":\"$2\",\"location\":\"tools/resume_s39_gpu01.sh\",\"data\":$3,\"timestamp\":$(date +%s000)}" >> "$DBG"; }
# #endregion

while pgrep -f "semkine/train.py --config configs/semkine/s39_so3fk_s3407" > /dev/null 2>&1; do
  sleep 30
done
echo "[$(date +%H:%M:%S)] seed 3407 training finished"

while true; do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
  [ "$mem" -lt 500 ] && break
  sleep 60
done
echo "[$(date +%H:%M:%S)] GPU 1 is free"
# #region agent log
log_dbg "H8" "seed 3408 starting on GPU 0,1" "{\"seed\":3408}"
# #endregion

export CUDA_VISIBLE_DEVICES=0,1
name=s39_so3fk_s3408
$PY semkine/train.py --config "configs/semkine/${name}.yaml" \
    --run-name "$name" --output-dir "outputs/semkine/${name}" \
    > "logs/${name}.log" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] $name done rc=$rc"
# #region agent log
log_dbg "H8" "seed train done" "{\"seed\":3408,\"rc\":${rc}}"
# #endregion

for seed in 3407 3408; do
  dir="outputs/semkine/s39_so3fk_s${seed}"
  [ -d "$dir" ] || continue
  echo "[$(date +%H:%M:%S)] selecting ${seed}"
  $PY tools/select_checkpoint.py --run-dir "$dir" > "logs/s39_select_s${seed}.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] selected ${seed} rc=$rc"
  # #region agent log
  sel=$(ls "$dir"/selection_*.json 2>/dev/null | head -1)
  ra=$( [ -n "$sel" ] && $PY -c "import json;d=json.load(open('$sel'));s=d.get('selected') or {};print(json.dumps({'step':s.get('step'),'ra':s.get('mpjpe_ra_mm'),'abs':s.get('mpjpe_abs_mm'),'grid_median':d.get('grid_median')}))" 2>/dev/null || echo '{}' )
  log_dbg "H8" "seed selected" "{\"seed\":${seed},\"rc\":${rc},\"selection\":${ra:-{}}}"
  # #endregion
done
echo "S39_SELECTION_DONE"
