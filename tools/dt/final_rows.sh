#!/bin/sh
# DT round: main-table rows (tools/report_table.py format) on ONE otherwise idle GPU, one arm after another, latency
# measured by evalx.latency_model (1.75 ms full-model anchor). Usage: tools/dt/final_rows.sh GPU "CPU_LIST" [arms...]
# Row names: <arm>_tf_pert (raw tracker) and <arm>_filt (the preregistered 0.5/1.0/0.5 filter; same checkpoints, the filter
# evaluation json is linked into the run directory as evalx_val_core_last_filt.json; latency is the tracker's, the filter adds
# ~0.1 ms of host arithmetic).
GPU=$1; CPUS=$2; shift; shift
cd /data1/lyq/code/mesh/EventHands1 || exit 1
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
S=outputs/semkine
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2
for arm in "$@"; do
  set -x
  taskset -c $CPUS $PY tools/tracking/evalx.py row --arm $arm --runs $S/${arm}_s3407 $S/${arm}_s3408 --ckpt last --variant tf_pert
  if [ -f outputs/dt/filter/${arm}_s3407/r0.5_f1.0_t0.5.json ] && [ -f outputs/dt/filter/${arm}_s3408/r0.5_f1.0_t0.5.json ]; then
    for s in 3407 3408; do ln -sf $PWD/outputs/dt/filter/${arm}_s$s/r0.5_f1.0_t0.5.json $S/${arm}_s$s/evalx_val_core_last_filt.json; done
    taskset -c $CPUS $PY tools/tracking/evalx.py row --arm $arm --runs $S/${arm}_s3407 $S/${arm}_s3408 --ckpt last --variant filt
  fi
  set +x
done
