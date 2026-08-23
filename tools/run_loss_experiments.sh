#!/usr/bin/env bash
# Unattended driver for one ablation arm at a time, whatever its control is: pass
# EXPERIMENTS as run basenames and CONTROL as the arm to compare against.
#
# Each experiment occupies both cards: its two replicates train concurrently, one per
# card, single-GPU each (no DDP) at the same effective batch as the control, so dRA
# against that control is not confounded by a different batch layout.
#
# GPU health is checked before training, since GPU 7 fails `torch.cuda.init` on this
# host even though nvidia-smi lists it. Pass GPUS="a b" to pick the pair.
#
# Each experiment runs four steps and aborts the whole driver if any of them fails,
# leaving the logs in place:
#   1. train both replicates concurrently
#   2. assert all 6 step checkpoints exist per replicate
#   3. score all 12 checkpoints closed-loop
#   4. write the pooled report with the pre-registered gate
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python}
GPUS=${GPUS:-"6 1"}
EXPERIMENTS=${EXPERIMENTS:-"so3fk"}
CONTROL=${CONTROL:-ch_both}
GATE_MM=${GATE_MM:--1.1}
EXPECTED_CKPTS=${EXPECTED_CKPTS:-6}
LOGDIR=${LOGDIR:-outputs/hand_data51/loss_exp_logs}

read -ra GPU_ARR <<< "$GPUS"
(( ${#GPU_ARR[@]} == 2 )) || { echo "need exactly 2 GPUs, got '$GPUS'" >&2; exit 2; }
mkdir -p "$LOGDIR"

# Fail before burning an hour of training rather than after.
for gpu in "${GPU_ARR[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu $PY -c "import torch; torch.zeros(8, device='cuda')" \
        >/dev/null 2>&1 || { echo "GPU $gpu cannot initialize CUDA" >&2; exit 2; }
done
echo "GPUs healthy: ${GPU_ARR[*]}"

for base in $EXPERIMENTS; do
    runs=("$base" "${base}_rep2")
    echo
    echo "=============================================================="
    echo "experiment $base: replicates ${runs[*]} on GPUs ${GPU_ARR[*]}"
    echo "started $(date -Is)"
    echo "=============================================================="

    # --- 1. train both replicates, one per card ---
    pids=()
    for i in 0 1; do
        run=${runs[$i]}
        gpu=${GPU_ARR[$i]}
        log="$LOGDIR/train_$run.log"
        echo "  train $run -> GPU $gpu ($log)"
        CUDA_VISIBLE_DEVICES=$gpu $PY model/train_abs.py \
            --config "configs/eventhands_track_render51_$run.yaml" --devices 1 \
            > "$log" 2>&1 &
        pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
    if (( fail )); then
        echo "  TRAIN FAILED ($fail/2); see $LOGDIR/train_*.log" >&2
        exit 1
    fi
    echo "  training done $(date -Is)"

    # --- 2. every checkpoint of the fixed step grid must be present ---
    for run in "${runs[@]}"; do
        d="outputs/hand_data51/track_render51_$run"
        n=$(ls "$d"/track_render51_"$run"-step=*.ckpt 2>/dev/null | wc -l)
        if (( n != EXPECTED_CKPTS )); then
            echo "  $run: expected $EXPECTED_CKPTS step checkpoints, found $n" >&2
            exit 1
        fi
        echo "  $run: $n checkpoints ok"
    done

    # --- 3. closed-loop score all 12 checkpoints ---
    echo "  evaluating $((2 * EXPECTED_CKPTS)) checkpoints..."
    RUNS="${runs[*]}" GPUS="$GPUS" PER_GPU=${EVAL_PER_GPU:-3} PY="$PY" \
        bash tools/eval_ch_ablation.sh > "$LOGDIR/eval_$base.log" 2>&1 || {
        echo "  EVAL FAILED; see $LOGDIR/eval_$base.log" >&2; exit 1; }

    # --- 4. pooled report for this experiment vs the control ---
    $PY tools/report_ch_ablation.py --arms "$CONTROL" "$base" \
        --control "$CONTROL" --gate-mm "$GATE_MM" \
        --out "outputs/hand_data51/report_loss_$base.md" \
        > "$LOGDIR/report_$base.log" 2>&1 || {
        echo "  REPORT FAILED; see $LOGDIR/report_$base.log" >&2; exit 1; }
    echo "  report -> outputs/hand_data51/report_loss_$base.md"
    echo "experiment $base complete $(date -Is)"
done

echo "all experiments finished $(date -Is)"
