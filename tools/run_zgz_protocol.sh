#!/usr/bin/env bash
# zgz protocol (2026-09-05): train one arm on the nine-subject default split (`splits_semkine.json`)
# and select on the held-out subject zgz. Usage:
#
#   tools/run_zgz_protocol.sh <arm> [seed ...]        # default seeds 3407 3408
#   tools/run_zgz_protocol.sh s36_eventgnn            # -> outputs/semkine/s36_eventgnn_s{3407,3408}
#
# `<arm>` names `configs/semkine/<arm>_s3407.yaml`; every other seed is that file with `--seed`.
# GPUs 4-7 only, so the two seeds run in parallel as two dual-card jobs (first seed on 6,7, second
# on 4,5), then each is selected on one card. Two cards per run is not negotiable (the schedule is
# tuned for the 2 x 512 effective batch), and NCCL_P2P_DISABLE=1 is mandatory on this box (see
# run_s36.sh). Selection is on the fixed SAVE_EVERY_N_STEPS grid, recursive RA on
# val_core = zgz_global + zgz_local, never on val_loss.
#
# History: the S36 rows in outputs/semkine/s36_eventgnn_s{3407,3408} were produced by the 09-05
# version of this script (`s36_eventgnn` and a second arm per seed, one seed at a time).
set -uo pipefail
cd "$(dirname "$0")/.."

ARM="${1:-s36_eventgnn}"
shift || true
SEEDS=("$@")
[ ${#SEEDS[@]} -gt 0 ] || SEEDS=(3407 3408)
[ ${#SEEDS[@]} -le 2 ] || { echo "at most two seeds fit on GPUs 4-7 at once"; exit 2; }
CFG=configs/semkine/${ARM}_s3407.yaml
[ -f "$CFG" ] || { echo "no config $CFG"; exit 2; }

export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED=1
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
mkdir -p logs

train() {   # gpus seed
  local gpus=$1 seed=$2
  local name=${ARM}_s${seed}
  echo "[$(date +%H:%M:%S)] train ${name} on GPUs ${gpus} (${CFG} --seed ${seed})"
  CUDA_VISIBLE_DEVICES=$gpus $PY semkine/train.py --config "$CFG" --seed "$seed" \
      --run-name "$name" --output-dir "outputs/semkine/${name}" > "logs/${name}.log" 2>&1
  echo "[$(date +%H:%M:%S)] train ${name} done rc=$?"
}

select_run() {   # gpu seed
  local name=${ARM}_s$2 dir=outputs/semkine/${ARM}_s$2
  if ls "$dir"/*step=6000.ckpt > /dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] select ${name} on GPU $1"
    CUDA_VISIBLE_DEVICES=$1 $PY tools/select_checkpoint.py --run-dir "$dir" > "logs/${name}_select.log" 2>&1
    echo "[$(date +%H:%M:%S)] select ${name} done rc=$?"
  else
    echo "[$(date +%H:%M:%S)] ${name}: grid incomplete, not selecting"
  fi
}

GPU_PAIRS=("6,7" "4,5")
SELECT_GPUS=(6 4)
for i in "${!SEEDS[@]}"; do
  train "${GPU_PAIRS[$i]}" "${SEEDS[$i]}" &
done
wait
for i in "${!SEEDS[@]}"; do
  select_run "${SELECT_GPUS[$i]}" "${SEEDS[$i]}" &
done
wait
echo "[$(date +%H:%M:%S)] ${ARM} finished"
echo "ZGZ_PROTOCOL_DONE"
