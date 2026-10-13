#!/bin/bash
# S43: evaluate trained runs (last checkpoint) on both pre-registered protocols.
#   1) the S39 stack (W200 clip-run-start + ramp1000 spec + Count50) with raw CL and TF: tools/trk6/screen_eval.sh,
#      output under outputs/s43/full/<run>/ (the main-table protocol)
#   2) the raw W50 protocol with controls, TF and root perturbation (rt_s37 / S38 numbers are on it):
#      outputs/semkine/<run>/evalx_val_core_last_tf_pert.json
# usage: tools/s43/post_eval.sh GPU RUN [RUN ...]
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
cd /data1/lyq/code/mesh/EventHands1
GPU=$1; shift
for run in "$@"; do
  tools/trk6/screen_eval.sh $GPU outputs/s43/full $run
  CUDA_VISIBLE_DEVICES=$GPU $PY tools/tracking/evalx.py eval --run-dir outputs/semkine/$run --ckpt last --controls --tf \
    --perturb 2>&1 | grep -E "RA=|wrote|retention"
done
