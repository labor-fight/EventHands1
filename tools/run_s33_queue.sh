#!/usr/bin/env bash
# S33 queue. Seed 3407 is already training on GPU1 when this starts; this fills GPU0 as soon as the
# S31 sensitivity probe releases it, then selects each seed the moment its training ends. Selection
# is the fixed 12-step grid on val_core under the retired 5/2/3 manifest, same as every arm it is
# being compared against.
set -u
cd /data1/lyq/code/mesh/EventHands1
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
MANIFEST=_retired_splits_semkine_5v2v3.json

wait_for() { sleep 30; while pgrep -f "$1" >/dev/null; do sleep 60; done; }

select_seed() {  # run name, gpu
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$2" $PY tools/select_checkpoint.py \
        --run-dir "outputs/semkine/$1" --manifest "$MANIFEST" --step-ms 50 \
        > "logs/s33_sel_$1.log" 2>&1
    echo "SELECTED $1"
    grep -E "^selected|^grid" "logs/s33_sel_$1.log"
}

# GPU0 is held by the probe; take it for seed 3408 the moment it lets go.
wait_for "run_closed_loop_probe"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 $PY semkine/train.py \
    --config configs/semkine/s33_cellgnn_render_s3408.yaml > logs/s33_render_s3408.log 2>&1
echo "TRAIN_3408_DONE"

# 3407 started ~an hour earlier on GPU1, so by here it is finished or nearly so.
wait_for "s33_cellgnn_render_s3407.yaml"
select_seed s33_cellgnn_render_s3407 1
select_seed s33_cellgnn_render_s3408 0
echo "S33_QUEUE_DONE"
