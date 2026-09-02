#!/usr/bin/env bash
# S37: train the render-free FK-direct arm on GPU 0 and 1, one seed after the other.
#
# NCCL_P2P_DISABLE is not a preference. On this box the process group comes up in ~1.3 s and then
# the first collective never returns: a bare 1024-element `all_reduce` between GPU 0 and 1 hung
# for 118 s under the default settings (tools/probe_ddp.py reproduces both halves). Any dual-card
# run here needs this set.
set -euo pipefail
cd "$(dirname "$0")/.."

export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1

PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
mkdir -p logs

# #region agent log
DBG=/data1/lyq/code/mesh/EventHands1/.cursor/debug-ea00d3.log
log_dbg() { echo "{\"sessionId\":\"ea00d3\",\"runId\":\"s37\",\"hypothesisId\":\"$1\",\"location\":\"tools/run_s37.sh\",\"message\":\"$2\",\"data\":$3,\"timestamp\":$(date +%s000)}" >> "$DBG"; }
# #endregion

for seed in 3407 3408; do
  cfg=configs/semkine/s37_fkdirect_s${seed}.yaml
  name=s37_fkdirect_s${seed}
  echo "[$(date +%H:%M:%S)] $name"
  # #region agent log
  log_dbg "H1" "seed train start" "{\"seed\":${seed}}"
  # #endregion
  $PY semkine/train.py --config "$cfg" \
      --run-name "$name" --output-dir "outputs/semkine/${name}" \
      > "logs/${name}.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] $name done rc=$rc"
  # #region agent log
  log_dbg "H1" "seed train done" "{\"seed\":${seed},\"rc\":${rc}}"
  # #endregion
done
echo "S37_TRAIN_DONE"
