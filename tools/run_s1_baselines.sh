#!/usr/bin/env bash
# S1 baseline family. Two phases because the tracking arms warm-start from the absolute ones,
# exactly as the frozen `track_render51` did (its INIT_FROM pointed at `abs_full51`).
#
# Arms x replicates:
#   A0  s1_abs_base       seeds 3407, 3408    absolute, no domrand
#   A0d s1_abs_domrand    seeds 3407, 3408    absolute, + domrand   (also the S13 anchor)
#   A1  s1_track_base     seeds 3407, 3408    tracking,  no domrand
#   A2  s1_track_domrand  seeds 3407, 3408    tracking,  + domrand  (main dense baseline)
#
# Two replicates per arm is the minimum the noise floor allows: same-config scatter is 0.4 mm
# and the observed replicate range reached 2.43 mm, so a single run cannot be compared to
# another single run.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
STEPS=${STEPS:-4000}
BSZ=${BSZ:-1024}
NW=${NW:-8}
OUT=outputs/semkine
mkdir -p "$OUT/logs"

launch () {   # launch <gpu> <config> <seed> <run_name> [extra args...]
  local gpu=$1 cfg=$2 seed=$3 name=$4; shift 4
  echo "GPU$gpu -> $name"
  CUDA_VISIBLE_DEVICES=$gpu NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
    nohup $PY semkine/train.py --config "$cfg" --seed "$seed" --run-name "$name" \
      --output-dir "$OUT/$name" --devices 1 --max-steps "$STEPS" \
      --batch-size "$BSZ" --num-workers "$NW" "$@" \
      > "$OUT/logs/$name.log" 2>&1 &
}

phase=${1:-abs}
if [ "$phase" = "abs" ]; then
  launch 0 configs/semkine/s1_abs_base.yaml    3407 s1_abs_base_s3407
  launch 1 configs/semkine/s1_abs_base.yaml    3408 s1_abs_base_s3408
  launch 2 configs/semkine/s1_abs_domrand.yaml 3407 s1_abs_domrand_s3407
  launch 4 configs/semkine/s1_abs_domrand.yaml 3408 s1_abs_domrand_s3408
elif [ "$phase" = "track" ]; then
  # INIT_FROM is passed as a config override through the run's own yaml copy, so the
  # provenance of each warm start is recorded next to its checkpoints.
  for s in 3407 3408; do
    for arm in base domrand; do
      src="$OUT/s1_abs_${arm}_s${s}/last.ckpt"
      [ -f "$src" ] || { echo "missing warm start $src" >&2; exit 1; }
      cfg="$OUT/s1_track_${arm}_s${s}.yaml"
      $PY - "$arm" "$s" "$src" "$cfg" <<'EOF'
import sys
from pathlib import Path
arm, seed, src, dst = sys.argv[1:5]
t = Path(f"configs/semkine/s1_track_{arm}.yaml").read_text()
t = t.replace("  RENDER_CHUNK: 256", f"  RENDER_CHUNK: 256\n  INIT_FROM: {src}")
Path(dst).write_text(t)
EOF
    done
  done
  launch 0 "$OUT/s1_track_base_s3407.yaml"    3407 s1_track_base_s3407
  launch 1 "$OUT/s1_track_base_s3408.yaml"    3408 s1_track_base_s3408
  launch 2 "$OUT/s1_track_domrand_s3407.yaml" 3407 s1_track_domrand_s3407
  launch 4 "$OUT/s1_track_domrand_s3408.yaml" 3408 s1_track_domrand_s3408
else
  echo "usage: $0 {abs|track}" >&2; exit 2
fi

wait
echo "phase $phase done"
