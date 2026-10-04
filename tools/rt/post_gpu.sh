#!/bin/sh
# Root-tracking round: the post-processing chain of tools/rt/post_jobs.py, run directly on one GPU
# (for results that gate the next decision). Usage: tools/rt/post_gpu.sh RUN GPU
RUN=$1; GPU=$2
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
RD=/data1/lyq/code/mesh/EventHands1/outputs/semkine/$RUN
cd /data1/lyq/code/mesh/EventHands1 || exit 1
export CUDA_VISIBLE_DEVICES=$GPU
$PY tools/select_checkpoint.py --run-dir "$RD" &&
$PY tools/tracking/evalx.py eval --run-dir "$RD" --ckpt last --controls --tf --perturb &&
$PY tools/tracking/evalx.py eval --run-dir "$RD" --ckpt selected
