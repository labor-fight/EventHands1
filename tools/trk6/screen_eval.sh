#!/bin/bash
# trk6 / S42: evaluate runs on the full S39 stack (W200 clip + ramp1000 spec + Count50), with raw CL and TF, last ckpt.
# usage: tools/trk6/screen_eval.sh GPU OUTROOT RUN [RUN ...]
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
cd /data1/lyq/code/mesh/EventHands1
GPU=$1; OUT=$2; shift 2
for run in "$@"; do
  CUDA_VISIBLE_DEVICES=$GPU $PY tools/dt/filter_eval.py --run-dir outputs/semkine/$run --ckpt last --split val_core \
    --window-ms 200 --clip-run-start --count-mode last50 --filter-spec outputs/dt3/gate_specs/ramp1000.json \
    --raw --tf --out-dir $OUT/$run 2>&1 | grep "RA="
done
