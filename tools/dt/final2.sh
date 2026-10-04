#!/bin/sh
# DT2 final rows (tools/report_table.py format) on ONE otherwise idle GPU. Usage: tools/dt/final2.sh GPU "CPU_LIST" [extra arms...]
# Rows (all --ckpt last, three seeds 3407/3408/3409 unless an arm has fewer finished seeds):
#   dt2_dz_l3_tf_pert   the bare tracker (recorded evalx_val_core_last_tf_pert.json)
#   dt2_dz_l3_dt2filt   + the preregistered constant filter (0.5, 1.0, 0.5)        outputs/dt2/filter_sweep/<run>/r0.5_f1.0_t0.5.json
#   dt2_dz_l3_dt2afilt  + the recommended AdaptiveFilter (gains chosen on zgz, two folds) .../afad1f21_rn0.3-0.5-0.8_f0.8_t0.5.json
#   <arm>_tf_pert       for every extra arm given (its finished seeds)
# Latency = evalx.latency_model (eager, batch 1, bare tracker, 1.75 ms anchor), the main table's definition; the filter is not
# in it (its cost is in the deployment bench, outputs/dt2/reports/D_bench_idle.json).
GPU=$1; CPUS=$2; shift; shift
cd /data1/lyq/code/mesh/EventHands1 || exit 1
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
S=outputs/semkine
F=$PWD/outputs/dt2/filter_sweep
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2
for s in 3407 3408 3409; do
  ln -sf $F/dt_dz_l3_s$s/r0.5_f1.0_t0.5.json $S/dt_dz_l3_s$s/evalx_val_core_last_dt2filt.json
  ln -sf $F/dt_dz_l3_s$s/afad1f21_rn0.3-0.5-0.8_f0.8_t0.5.json $S/dt_dz_l3_s$s/evalx_val_core_last_dt2afilt.json
done
R3="$S/dt_dz_l3_s3407 $S/dt_dz_l3_s3408 $S/dt_dz_l3_s3409"
set -x
for v in tf_pert dt2filt dt2afilt; do
  taskset -c $CPUS $PY tools/tracking/evalx.py row --arm dt2_dz_l3 --runs $R3 --ckpt last --variant $v
done
for arm in "$@"; do
  runs=""
  for s in 3407 3408 3409; do
    [ -f $S/${arm}_s$s/evalx_val_core_last_tf_pert.json ] && runs="$runs $S/${arm}_s$s"
  done
  [ -n "$runs" ] && taskset -c $CPUS $PY tools/tracking/evalx.py row --arm dt2_${arm#dt_} --runs $runs --ckpt last --variant tf_pert
done
set +x
