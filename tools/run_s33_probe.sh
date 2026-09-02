#!/usr/bin/env bash
# Once selection lands, probe the S33 winners the same way S31 was probed. The question is not
# "is it worse" -- the grid already says that -- but *which* failure it is:
#
#   update_ratio well below 1 with a flat sensitivity slope => the render lattice became a
#     shortcut: the pose is recoverable from the render channel alone, so copying it minimises the
#     single-step loss and the events stop being read.
#   update_ratio near S31's with the same steep slope => the channel changed nothing and the
#     16 px lattice itself is the ceiling.
set -u
cd /data1/lyq/code/mesh/EventHands1
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
MANIFEST=_retired_splits_semkine_5v2v3.json
SEL=selection_val_core_step50__retired_splits_semkine_5v2v3.json

while pgrep -f select_checkpoint >/dev/null; do sleep 45; done

CK=$($PY -c "
import json
d=json.load(open('outputs/semkine/s33_cellgnn_render_s3407/$SEL'))
print(d['selected']['ckpt'])")
echo "probing $CK"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 $PY tools/run_closed_loop_probe.py \
    --arm "s33_render=$CK:configs/semkine/s33_cellgnn_render_s3407.yaml" \
    --manifest "$MANIFEST" --step-ms 50 --prev-noise 0.5,1,2,4 \
    --out outputs/semkine/s33_gain_slope_50ms.json 2>&1 | tee logs/s33_slope.log
echo "S33_PROBE_DONE"
