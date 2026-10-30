#!/usr/bin/env bash
# AGG778 R2 launcher (docs/EXPERIMENT_RECORDS.md#agg778-r2-prereg-20261024). Physical GPUs 4-7 only, CPU free.
#
#   tools/agg778/r2_run.sh probes   the eight pre-registered probe processes, two per GPU
#   tools/agg778/r2_run.sh d        Stage D: select (R2 validity checked) -> preflight / formal training -> evaluation
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8
O=outputs/agg778/r2
mkdir -p "$O/logs" "$O/probe"
run() {
  CUDA_VISIBLE_DEVICES=$1 $PY tools/agg778/r2_task.py --cond "$2" --regime "$3" --group "$4" \
    --out "$O/probe/$2_$3_$4" > "$O/logs/probe_$2_$3_$4.log" 2>&1
}
case "${1:-}" in
  probes)
    run 4 A3 e3 g1 & run 4 A0 e3 g1 &
    run 5 A3 e3 g2 & run 5 A0 e3 g2 &
    run 6 A3 e3 g3 & run 6 A0 e3 g3 &
    run 7 A3 noise g1 & run 7 A3 e3 gb &
    wait
    ;;
  d)
    $PY tools/agg778/r2_d.py select
    $PY tools/agg778/r2_d.py train > "$O/logs/d_train.log" 2>&1
    $PY tools/agg778/r2_d.py evaluate > "$O/logs/d_eval.log" 2>&1
    ;;
  *)
    echo "usage: tools/agg778/r2_run.sh probes|d" >&2
    exit 2
    ;;
esac
