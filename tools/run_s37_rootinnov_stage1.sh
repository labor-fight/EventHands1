#!/usr/bin/env bash
# S37 root innovation, stage 1 (docs/S37_EXPERIMENT_RECORDS.md [ROOTINNOV]): both seeds in parallel on
# GPUs 6,7 / 4,5, each from its own config (it warm-starts from that seed's selected S37 step), then
# selection on the 250-step grid, the main row, and the pre-registered mechanism gates.
# `tools/run_zgz_protocol.sh` is not used because it only selects a complete 6000-step grid and
# gives every seed the 3407 config.
set -uo pipefail
cd "$(dirname "$0")/.."
export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED=1
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
ARM=s37_rootinnov
mkdir -p logs

train() {   # gpus seed
  local name=${ARM}_s$2
  echo "[$(date +%F' '%T)] train ${name} on GPUs $1"
  CUDA_VISIBLE_DEVICES=$1 $PY semkine/train.py --config "configs/semkine/${name}.yaml" \
      --run-name "$name" --output-dir "outputs/semkine/${name}" > "logs/${name}.log" 2>&1
  echo "[$(date +%F' '%T)] train ${name} done rc=$?"
}

select_run() {   # gpu seed
  local name=${ARM}_s$2
  if ls "outputs/semkine/${name}"/*step=1500.ckpt > /dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES=$1 $PY tools/select_checkpoint.py --run-dir "outputs/semkine/${name}" \
        > "logs/${name}_select.log" 2>&1
    echo "[$(date +%F' '%T)] select ${name} done rc=$?"
  else
    echo "[$(date +%F' '%T)] ${name}: grid incomplete, not selecting"
    return 1
  fi
}

train "6,7" 3407 &
train "4,5" 3408 &
wait
select_run 6 3407 & p1=$!
select_run 4 3408 & p2=$!
wait $p1 && wait $p2 || { echo "selection failed"; exit 1; }
CUDA_VISIBLE_DEVICES=6 $PY tools/make_s36_row.py --run ${ARM} > logs/row_${ARM}_zgzproto.log 2>&1
echo "[$(date +%F' '%T)] main row done rc=$?"
CUDA_VISIBLE_DEVICES=6 $PY tools/probe_s37_rootinnov.py \
    --runs outputs/semkine/${ARM}_s3407 outputs/semkine/${ARM}_s3408 \
    --out outputs/semkine/probe_${ARM}.json > logs/probe_${ARM}.log 2>&1
echo "[$(date +%F' '%T)] gates done rc=$?"
echo "S37_ROOTINNOV_STAGE1_DONE"
