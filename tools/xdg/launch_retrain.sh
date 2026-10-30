#!/usr/bin/env bash
# XDG-1: three fresh Q1-derived runs on GPU2-7 (two cards each), then the formal row on the first card of each pair
# and the CPU mechanism evaluation. Pre-registration: outputs/xdg/retrain/protocol.json (frozen before launch).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
LOG=outputs/xdg/retrain/logs
mkdir -p "$LOG"
export NCCL_P2P_DISABLE=1 NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 2,3,4,5,6,7

pipeline() {
  local run=$1 cards=$2 port=$3 first=${2%%,*}
  CUDA_VISIBLE_DEVICES=$cards MASTER_PORT=$port "$PY" -u tools/xdl/train.py --config "configs/xdg/$run.yaml" \
    --frozen-manifest outputs/xdg/retrain/frozen_manifest.json > "$LOG/train_$run.log" 2>&1
  CUDA_VISIBLE_DEVICES=$first "$PY" -u tools/xdl/make_row.py --run "$run" --config "configs/xdg/$run.yaml" \
    --device cuda:0 > "$LOG/row_$run.log" 2>&1
  for domain in zgz_local zgz_global check; do
    CUDA_VISIBLE_DEVICES="" XDG_THREADS=16 "$PY" -u tools/xdg/eval_retrain.py evaluate --run "$run" --domain "$domain" \
      > "$LOG/eval_${run}__$domain.log" 2>&1 &
  done
  wait
  echo "$run done" >> "$LOG/pipeline_done.txt"
}

pipeline xdg_q1t0_s3407 2,3 29720 &
pipeline xdg_q1dz_s3407 4,5 29721 &
pipeline xdg_q1r_s3407 6,7 29722 &
wait
"$PY" tools/xdg/eval_retrain.py summarize > "$LOG/summarize.log" 2>&1
echo "XDG-1 complete: outputs/xdg/retrain/result.json"
