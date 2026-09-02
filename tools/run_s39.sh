#!/usr/bin/env bash
# S39: train the SO(3)+FK-loss arm, one seed after the other.
#
# GPU 0 + 1 only -- the user's standing constraint for this box. (Seed 3407 of the first
# S39 run was trained on 0 + 4 while GPU 1 was busy; that excursion was stopped on user
# instruction. All 8 cards are identical L20s, so the seed-3407 result is protocol-valid;
# seed 3408 ran on 0 + 1 via tools/resume_s39_gpu01.sh.)
#
# NCCL_P2P_DISABLE is not a preference: a bare 1024-element all_reduce between two GPUs
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
log_dbg() { echo "{\"sessionId\":\"ea00d3\",\"runId\":\"s39\",\"hypothesisId\":\"$1\",\"message\":\"$2\",\"location\":\"tools/run_s39.sh\",\"data\":$3,\"timestamp\":$(date +%s000)}" >> "$DBG"; }
# #endregion

for seed in 3407 3408; do
  cfg=configs/semkine/s39_so3fk_s${seed}.yaml
  name=s39_so3fk_s${seed}
  echo "[$(date +%H:%M:%S)] $name"
  # #region agent log
  log_dbg "H8" "seed train start" "{\"seed\":${seed}}"
  # #endregion
  $PY semkine/train.py --config "$cfg" \
      --run-name "$name" --output-dir "outputs/semkine/${name}" \
      > "logs/${name}.log" 2>&1
  echo "[$(date +%H:%M:%S)] $name done"
  # #region agent log
  log_dbg "H8" "seed train done" "{\"seed\":${seed},\"rc\":0}"
  # #endregion
done
echo "S39_TRAIN_DONE"
