#!/usr/bin/env bash
# S26: the S1 dense baseline family retrained under the canonical nine-subject split.
#
# Same recipe as run_s1_baselines.sh, same configs, same seeds, same fixed step budget. The only
# variable is the subject split, which is now the capture-time one: nine subjects train, zgz held
# out. The configs need no edit because `splits_semkine.json` is the loader's default and now holds
# that split, so this measures exactly one thing against the s1_*_s340{7,8} runs.
#
# Two seeds per arm is not optional here. Same-config scatter on this data is 0.4 mm and the observed
# replicate range reached 2.43 mm, so a single run against a single run cannot support a claim.
#
# Phases are sequential: the tracking arms warm-start from the absolute ones, as S1 did.
#   ./tools/run_s26_9subj.sh abs     # ~80 min, GPUs 0 and 1
#   ./tools/run_s26_9subj.sh track   # ~72 min, GPUs 0 and 1
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
STEPS=${STEPS:-4000}
BSZ=${BSZ:-1024}
NW=${NW:-8}
GPUS=(${GPUS:-0 1})
SEEDS=(3407 3408)
OUT=outputs/semkine
mkdir -p "$OUT/logs"

launch () {   # launch <gpu> <config> <seed> <run_name>
  local gpu=$1 cfg=$2 seed=$3 name=$4
  echo "GPU$gpu -> $name"
  CUDA_VISIBLE_DEVICES=$gpu NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
    nohup $PY semkine/train.py --config "$cfg" --seed "$seed" --run-name "$name" \
      --output-dir "$OUT/$name" --devices 1 --max-steps "$STEPS" \
      --batch-size "$BSZ" --num-workers "$NW" \
      > "$OUT/logs/$name.log" 2>&1 &
}

# Refuse to run against the wrong split rather than produce a run that looks fine and is not
# comparable. This is the check whose absence let five-subject runs pass for nine-subject ones.
$PY - <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, "."); sys.path.insert(0, "model")
from config import load_config
root = Path(load_config("configs/semkine/s1_abs_domrand.yaml")["DATA"]["ROOT"])
m = json.loads((root / "splits_semkine.json").read_text())
tr, held = m["train"]["subjects"], m["val_core"]["subjects"]
assert len(tr) == 9 and held == ["zgz"], f"expected 9 train subjects and zgz held out, got {tr} / {held}"
assert m["_leakage"]["clean"], "split manifest reports leakage"
print(f"split ok: {len(tr)} train subjects, held out {held}")
EOF

phase=${1:-abs}
if [ "$phase" = "abs" ]; then
  for i in 0 1; do
    launch "${GPUS[$i]}" configs/semkine/s1_abs_domrand.yaml "${SEEDS[$i]}" "s26_abs_9subj_s${SEEDS[$i]}"
  done
elif [ "$phase" = "track" ]; then
  # INIT_FROM goes through a per-replicate yaml copy so each warm start's provenance sits next to
  # its own checkpoints, matching how the S1 tracking arms were built.
  for s in "${SEEDS[@]}"; do
    src="$OUT/s26_abs_9subj_s${s}/last.ckpt"
    [ -f "$src" ] || { echo "missing warm start $src; run the abs phase first" >&2; exit 1; }
    $PY - "$src" "$OUT/s26_track_9subj_s${s}.yaml" <<'EOF'
import sys
from pathlib import Path
src, dst = sys.argv[1:3]
t = Path("configs/semkine/s1_track_domrand.yaml").read_text()
t = t.replace("  RENDER_CHUNK: 256", f"  RENDER_CHUNK: 256\n  INIT_FROM: {src}")
Path(dst).write_text(t)
EOF
  done
  for i in 0 1; do
    s=${SEEDS[$i]}
    launch "${GPUS[$i]}" "$OUT/s26_track_9subj_s${s}.yaml" "$s" "s26_track_9subj_s${s}"
  done
else
  echo "usage: $0 {abs|track}" >&2; exit 2
fi

wait
echo "phase $phase done"
