#!/usr/bin/env bash
# S38: train the joint-query readout arm on GPU 0 and 1, one seed after the other.
#
# NCCL_P2P_DISABLE is not a preference: a bare 1024-element all_reduce between GPU 0 and 1
# hung for 118 s on this box under the defaults (tools/probe_ddp.py). Any dual-card run
# here needs this set.
set -euo pipefail
cd "$(dirname "$0")/.."

export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1

PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
mkdir -p logs

# #region agent log
DBG=/data1/lyq/code/mesh/EventHands1/.cursor/debug-ea00d3.log
log_dbg() { echo "{\"sessionId\":\"ea00d3\",\"runId\":\"s38\",\"hypothesisId\":\"$1\",\"message\":\"$2\",\"location\":\"tools/run_s38.sh\",\"data\":$3,\"timestamp\":$(date +%s000)}" >> "$DBG"; }
# #endregion

for seed in 3407 3408; do
  cfg=configs/semkine/s38_jointreadout_s${seed}.yaml
  name=s38_jointreadout_s${seed}
  echo "[$(date +%H:%M:%S)] $name"
  # #region agent log
  log_dbg "H5" "seed train start" "{\"seed\":${seed}}"
  # #endregion
  $PY semkine/train.py --config "$cfg" \
      --run-name "$name" --output-dir "outputs/semkine/${name}" \
      > "logs/${name}.log" 2>&1
  echo "[$(date +%H:%M:%S)] $name done"
  # #region agent log
  log_dbg "H5" "seed train done" "{\"seed\":${seed},\"rc\":0}"
  # #endregion
done
echo "S38_TRAIN_DONE"
