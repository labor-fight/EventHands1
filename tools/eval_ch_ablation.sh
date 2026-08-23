#!/usr/bin/env bash
# Closed-loop evaluation of every render-channel ablation checkpoint.
#
# The rollout is batch-size-1 and sequential, so one process leaves a card almost
# idle: pack several evals per GPU and spread runs over the free cards.
set -euo pipefail

cd "$(dirname "$0")/.."
# Absolute interpreter: a bare `python` silently resolves to the system one when this
# runs detached from an activated shell, and every rollout dies on `import torch`.
PY=${PY:-/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python}
GPUS=${GPUS:-"0 2 3 4 5 6 7"}
PER_GPU=${PER_GPU:-6}
STEP_MS=${STEP_MS:-50}
RUNS=${RUNS:-"ch_sil ch_sil_rep2 ch_inv ch_inv_rep2 ch_both ch_both_rep2"}

read -ra GPU_ARR <<< "$GPUS"
n_gpu=${#GPU_ARR[@]}
i=0
pids=()

for run in $RUNS; do
    dir="outputs/hand_data51/track_render51_$run"
    cfg="configs/eventhands_track_render51_$run.yaml"
    [[ -d $dir ]] || { echo "skip missing $dir"; continue; }
    for ckpt in "$dir"/track_render51_"$run"-step=*.ckpt; do
        [[ -e $ckpt ]] || continue
        step=$(basename "$ckpt" | sed -E 's/.*-step=([0-9]+)\.ckpt/\1/')
        out="$dir/eval_step$step"
        gpu=${GPU_ARR[$((i % n_gpu))]}
        i=$((i + 1))
        mkdir -p "$out"
        CUDA_VISIBLE_DEVICES=$gpu $PY model/eval_track.py \
            --config "$cfg" --ckpt "$ckpt" --step-ms "$STEP_MS" --out-dir "$out" \
            > "$out/eval.log" 2>&1 &
        pids+=($!)
        # Throttle so no card hosts more than PER_GPU concurrent rollouts.
        while (( $(jobs -rp | wc -l) >= n_gpu * PER_GPU )); do wait -n; done
    done
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "done: ${#pids[@]} evals, $fail failures"
exit $fail
