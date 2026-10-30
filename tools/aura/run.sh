#!/usr/bin/env bash
# AURA launcher (docs/EXPERIMENT_RECORDS.md#aura-prereg-20261024). Physical GPUs 4-7 only, CPU free.
#
#   tools/aura/run.sh a        Stage A on the CPU (four packet sets in parallel, then combine)
#   tools/aura/run.sh b        Stage B on the CPU (zgz, sel, fit), then summarize b
#   tools/aura/run.sh gpu      the GPU stages in the background: waits for AGG778 R2 and MRF to complete, then the R2
#                              precondition, Stage C, C2, summaries, Stage D (select, wiring, train, evaluate), final
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=outputs/aura
mkdir -p "$O/logs"
case "${1:-}" in
  a)
    for s in zgz_e3 zgz_e2 sel_e3 fit_noise; do
      CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 $PY tools/aura/a_audit.py --set $s --out "$O/a" --threads 8 \
        > "$O/logs/a_$s.log" 2>&1 &
    done
    wait
    CUDA_VISIBLE_DEVICES="" $PY tools/aura/a_audit.py --combine --out "$O/a" > "$O/logs/a_combine.log" 2>&1
    ;;
  b)
    for s in zgz sel fit; do
      CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 $PY tools/aura/b_operator.py --set $s --procs ${B_PROCS:-40} \
        --out "$O/b" > "$O/logs/b_$s.log" 2>&1
    done
    $PY tools/aura/summarize.py b > "$O/logs/b_summary.log" 2>&1
    ;;
  gpu)
    nohup $PY tools/aura/orchestrate.py > "$O/logs/orchestrator.log" 2>&1 &
    echo "orchestrator pid $!"
    ;;
  *)
    echo "usage: tools/aura/run.sh a|b|gpu" >&2
    exit 2
    ;;
esac
