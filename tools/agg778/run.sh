#!/usr/bin/env bash
# AGG778 launcher (docs/EXPERIMENT_RECORDS.md#agg778-prereg-20261024). Physical GPUs 4-7 only, CPU free.
#
#   tools/agg778/run.sh ab      Stage A (GPU 4) with Stage B in parallel (B-task A3 GPU 5, B-vertex A0 / A3 GPU 6 / 7;
#                               then B-task A0 and the P4 reproduction on GPU 4). B is read only after A passes.
#   tools/agg778/run.sh c       Stage C (GPU 4), after Stage B
#   tools/agg778/run.sh d       Stage D: select -> preflight / formal training (GPUs 4-5, 6-7) -> evaluation (4-7)
#   tools/agg778/run.sh summary Stage-B summary (CPU) / final verdict and the nine-column table (CPU)
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8
O=outputs/agg778
mkdir -p "$O/logs"
case "${1:-}" in
  ab)
    CUDA_VISIBLE_DEVICES=5 $PY tools/agg778/b_task.py --cond A3 --out $O/b_task/A3 > $O/logs/b_task_A3.log 2>&1 &
    CUDA_VISIBLE_DEVICES=6 $PY tools/agg778/b_vertex.py --cond A0 --out $O/b_vertex/A0 > $O/logs/b_vertex_A0.log 2>&1 &
    CUDA_VISIBLE_DEVICES=7 $PY tools/agg778/b_vertex.py --cond A3 --out $O/b_vertex/A3 > $O/logs/b_vertex_A3.log 2>&1 &
    CUDA_VISIBLE_DEVICES=4 $PY tools/agg778/a_audit.py --out $O/a > $O/logs/a.log 2>&1
    CUDA_VISIBLE_DEVICES=4 $PY tools/agg778/b_vertex.py --p4repro --out $O/b_vertex/p4repro > $O/logs/p4repro.log 2>&1
    CUDA_VISIBLE_DEVICES=4 $PY tools/agg778/b_task.py --cond A0 --out $O/b_task/A0 > $O/logs/b_task_A0.log 2>&1
    wait
    ;;
  c)
    CUDA_VISIBLE_DEVICES=4 $PY tools/agg778/c_robust.py --out $O/c > $O/logs/c.log 2>&1
    ;;
  d)
    $PY tools/agg778/d_stage.py select
    $PY tools/agg778/d_stage.py train > $O/logs/d_train.log 2>&1
    $PY tools/agg778/d_stage.py evaluate > $O/logs/d_eval.log 2>&1
    ;;
  summary)
    $PY tools/agg778/summarize.py "${2:-b}"
    ;;
  *)
    echo "usage: tools/agg778/run.sh ab|c|d|summary [b|final]" >&2
    exit 2
    ;;
esac
