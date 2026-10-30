#!/usr/bin/env bash
# MRF launcher (docs/EXPERIMENT_RECORDS.md#mrf-prereg-20261024). Physical GPUs 4-7 only, CPU free.
#
#   tools/mrf/run.sh a        Stage A on the CPU
#   tools/mrf/run.sh b        Stage B probes (14 processes, at most two per GPU, a queue) + B-vertex (A0, A3)
#   tools/mrf/run.sh c        Stage C (single / loop / thin on A0 and A3), one process per GPU
#   tools/mrf/run.sh d        Stage D: select -> preflight / formal training -> evaluation (after apply-wiring)
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=outputs/mrf
mkdir -p "$O/logs"
case "${1:-}" in
  a)
    CUDA_VISIBLE_DEVICES="" $PY tools/mrf/a_audit.py --out "$O/a" > "$O/logs/a.log" 2>&1
    ;;
  b)
    $PY tools/mrf/gpu_queue.py b > "$O/logs/b_queue.log" 2>&1
    ;;
  c)
    $PY tools/mrf/gpu_queue.py c > "$O/logs/c_queue.log" 2>&1
    ;;
  d)
    $PY tools/mrf/d_stage.py select
    $PY tools/mrf/d_stage.py train > "$O/logs/d_train.log" 2>&1
    $PY tools/mrf/d_stage.py evaluate > "$O/logs/d_eval.log" 2>&1
    ;;
  *)
    echo "usage: tools/mrf/run.sh a|b|c|d" >&2
    exit 2
    ;;
esac
