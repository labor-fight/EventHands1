#!/bin/sh
# DT2 official deployment-loop timing on an idle GPU (GPU 4, node-1 cores 60-63,132-135): dt_dz_l3_s3407 full ladder,
# then dt_dz_l3w128_s3407 v0 / v5 only.
cd /data1/lyq/code/mesh/EventHands1 || exit 1
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
C=60-63,132-135
date
taskset -c $C nice -n 5 $PY tools/dt/bench_loop.py --run-dir outputs/semkine/dt_dz_l3_s3407 --gpu 4 --cores $C \
  --variants v0,v1,v2,v3,v4,v4g,v5 --packets 300 --rounds 5 --tag idle
date
taskset -c $C nice -n 5 $PY tools/dt/bench_loop.py --run-dir outputs/semkine/dt_dz_l3w128_s3407 --gpu 4 --cores $C \
  --variants v0,v5 --packets 300 --rounds 5 --tag idle_w128
date
