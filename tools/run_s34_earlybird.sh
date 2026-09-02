#!/usr/bin/env bash
# S34 early-bird probe. A full seed is ~3 h at this width; the step=2000 checkpoint lands in ~1 h
# and a single-arm probe on it costs six minutes. That is enough to call the scale question early,
# which is the practice that saved hours on the constP unroll failure.
#
# Reads against the S31 anchor it has to beat: teacher-forced 16.03, recursive 34.64, x2.16.
# Registered call: recursive < 30 with amplification < 2.0 => scale was a real confound, let both
# seeds finish. Recursive still >= 30 with amplification still ~2.2 => the pooled readout is the
# ceiling independent of size, kill the run and change the readout's shape instead.
set -u
cd /data1/lyq/code/mesh/EventHands1
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
CK=outputs/semkine/s34_cellgnn_scale_s3407/s34_cellgnn_scale_s3407-step=2000.ckpt

while [ ! -f "$CK" ]; do sleep 120; done
sleep 30   # let the write finish

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 $PY tools/run_closed_loop_probe.py \
    --arm "s34_scale=$CK:configs/semkine/s34_cellgnn_scale_s3407.yaml" \
    --manifest _retired_splits_semkine_5v2v3.json --step-ms 50 --prev-noise 1,4 \
    --out outputs/semkine/s34_earlybird_50ms.json 2>&1 | tee logs/s34_earlybird.log
echo "S34_EARLYBIRD_DONE"
