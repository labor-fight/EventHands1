#!/usr/bin/env bash
# S40 / S41 / S42: the three fine-tuning arms, sequentially, on GPU 0-1 only.
#
# GPU 0-1 is the user's standing constraint for this box. GPU 1 is shared with another
# user's jobs, so every launch first waits for it to be free. Selection for an arm runs
# right after that arm's two seeds, so results are readable arm by arm.
#
# NCCL_P2P_DISABLE is not a preference: a bare 1024-element all_reduce between two GPUs
# hung for 118 s on this box under the defaults (tools/probe_ddp.py).
set -uo pipefail
cd "$(dirname "$0")/.."

export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1

PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
mkdir -p logs

# #region agent log
DBG=/data1/lyq/code/mesh/EventHands1/.cursor/debug-ea00d3.log
log_dbg() { echo "{\"sessionId\":\"ea00d3\",\"runId\":\"s40s41s42\",\"hypothesisId\":\"$1\",\"message\":\"$2\",\"location\":\"tools/run_s40_41_42.sh\",\"data\":$3,\"timestamp\":$(date +%s000)}" >> "$DBG"; }
# #endregion

wait_gpu1() {
  while true; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    [ "$mem" -lt 500 ] && return
    sleep 60
  done
}

run_arm() {
  local arm=$1 hyp=$2
  for seed in 3407 3408; do
    name=${arm}_s${seed}
    echo "[$(date +%H:%M:%S)] waiting for GPU 1 (${name})"
    wait_gpu1
    echo "[$(date +%H:%M:%S)] ${name} training"
    # #region agent log
    log_dbg "$hyp" "seed train start" "{\"arm\":\"${arm}\",\"seed\":${seed}}"
    # #endregion
    $PY semkine/train.py --config "configs/semkine/${name}.yaml" \
        --run-name "$name" --output-dir "outputs/semkine/${name}" \
        > "logs/${name}.log" 2>&1
    rc=$?
    echo "[$(date +%H:%M:%S)] ${name} done rc=$rc"
    # #region agent log
    log_dbg "$hyp" "seed train done" "{\"arm\":\"${arm}\",\"seed\":${seed},\"rc\":${rc}}"
    # #endregion
  done
  for seed in 3407 3408; do
    dir="outputs/semkine/${arm}_s${seed}"
    [ -d "$dir" ] || continue
    echo "[$(date +%H:%M:%S)] selecting ${arm} s${seed}"
    $PY tools/select_checkpoint.py --run-dir "$dir" > "logs/${arm}_select_s${seed}.log" 2>&1
    rc=$?
    echo "[$(date +%H:%M:%S)] selected ${arm} s${seed} rc=$rc"
    # #region agent log
    sel=$(ls "$dir"/selection_*.json 2>/dev/null | head -1)
    ra=$( [ -n "$sel" ] && $PY -c "import json;d=json.load(open('$sel'));s=d.get('selected') or {};print(json.dumps({'step':s.get('step'),'ra':s.get('mpjpe_ra_mm'),'abs':s.get('mpjpe_abs_mm'),'grid_median':d.get('grid_median')}))" 2>/dev/null || echo '{}' )
    log_dbg "$hyp" "seed selected" "{\"arm\":\"${arm}\",\"seed\":${seed},\"rc\":${rc},\"selection\":${ra:-{}}}"
    # #endregion
  done
  echo "${arm}_DONE"
}

run_arm s40_fkdir6 H11
run_arm s41_attnpool H12
run_arm s42_absfk H13
echo "S40_41_42_ALL_DONE"
