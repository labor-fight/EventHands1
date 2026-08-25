#!/usr/bin/env bash
# Serial training queue for one GPU. Two of these, on GPU 0 and GPU 1, run the S19/S20 arms of one
# seed each: the arms are independent, so splitting by seed keeps both replicates of every arm on
# the same hardware and finishes each arm's pair at the same time.
#
#   tools/run_keg_queue.sh <gpu> <seed> [config ...]
#
# Skips a config whose run directory carries the `.run_complete` marker this script writes on a
# clean exit, so the queue can be re-run after an interruption without redoing finished work.
#
# The marker exists because counting checkpoints cannot tell a finished run from a failed one. A
# collapsed arm has a full grid too, and the S22 unroll arm was archived precisely because it had
# one; a count-based rule would have skipped its replacement.
set -u
gpu=$1; seed=$2; shift 2
# `KEG_REPO` lets the script be copied elsewhere before launching, which is worth doing: bash reads
# a running script incrementally by byte offset, so editing this file mid-run makes the shell resume
# at a shifted position and fail on a line it never meant to execute.
cd "${KEG_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"

for cfg in "$@"; do
  name=$(basename "$cfg" .yaml)
  dir="outputs/semkine/${name}_s${seed}"
  want=$(python - "configs/semkine/${name}.yaml" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))["TRAIN"]
print(int(c["MAX_STEPS"]) // int(c["SAVE_EVERY_N_STEPS"]))
PY
)
  have=$(ls "$dir"/*step=*.ckpt 2>/dev/null | wc -l)
  if [ -f "$dir/.run_complete" ]; then
    echo "[gpu$gpu seed$seed] $name completed earlier ($have/$want checkpoints), skipping"
    continue
  fi
  if [ "$have" -ge "$want" ]; then
    echo "[gpu$gpu seed$seed] $name has a full grid ($have/$want) but no .run_complete marker."
    echo "  Refusing to guess. Either 'touch $dir/.run_complete' to accept it, or move the grid"
    echo "  aside if it is a failed arm being replaced."
    continue
  fi
  # Move an interrupted run's partial grid out of the way. Lightning would keep it and write
  # `-v1` duplicates beside it, leaving a selection grid whose points come from two different
  # runs -- a silent way to select a checkpoint that was never on the trajectory being claimed.
  if [ "$have" -gt 0 ]; then
    stale="$dir/interrupted_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$stale"
    mv "$dir"/*.ckpt "$stale"/ 2>/dev/null
    echo "[gpu$gpu seed$seed] $name: moved $have partial checkpoints to $stale"
  fi
  mkdir -p "$dir"
  echo "[gpu$gpu seed$seed] $name starting ($(date +%H:%M:%S))"
  CUDA_VISIBLE_DEVICES=$gpu python semkine/train.py \
      --config "configs/semkine/${name}.yaml" --seed "$seed" \
      --output-dir "$dir" --run-name "${name}_s${seed}" > "$dir/train.log" 2>&1
  rc=$?
  got=$(ls "$dir"/*step=*.ckpt 2>/dev/null | wc -l)
  if [ "$rc" -eq 0 ] && [ "$got" -ge "$want" ]; then
    date +%Y-%m-%dT%H:%M:%S > "$dir/.run_complete"
  fi
  echo "[gpu$gpu seed$seed] $name exit=$rc ckpts=$got ($(date +%H:%M:%S))"
done
echo "[gpu$gpu seed$seed] queue done"
