#!/usr/bin/env bash
# Three independent 2 x 512 jobs on the user's GPU2-7 allocation.
# Debug is the default. Formal training requires a recorded passing learning gate.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${XKF_PYTHON:-/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python}
MODE=${1:-debug}
STEPS=${2:-200}
OUT=${3:-outputs/xkf/debug_20261015_r1}
mkdir -p "$OUT/logs"
export NCCL_P2P_DISABLE=1 NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
if [[ "$MODE" != debug && "$MODE" != full ]]; then
  echo 'Usage: launch.sh debug [steps] [output-dir] | launch.sh full 6000 [output-dir]' >&2
  exit 2
fi
if [[ "$MODE" == full ]]; then
  : "${XKF_DEBUG_GATE:?A passing debug learning-gate JSON is required}"
  : "${XKF_FROZEN_MANIFEST:?A final frozen source/config manifest is required}"
  "$PY" - "$XKF_DEBUG_GATE" "$XKF_FROZEN_MANIFEST" <<'PY'
import hashlib, json, pathlib, sys
gate, manifest = [json.loads(pathlib.Path(p).read_text()) for p in sys.argv[1:]]
if gate.get('status') != 'PASS' or not gate.get('all_three_seeds_passed'):
    raise SystemExit('Full training denied: all three debug learning evaluations must pass.')
for name, digest in gate['source_sha256'].items():
    if hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() != digest:
        raise SystemExit(f'Debugged implementation changed: {name}')
if gate['source_sha256'] != manifest['source_sha256']:
    raise SystemExit('Formal manifest differs from the implementation that passed debug.')
PY
  [[ "$STEPS" == 6000 ]] || { echo 'Formal endpoint must be step6000' >&2; exit 2; }
fi

arm() {
  local seed=$1 cards=$2 port=$3 run=xkf_s$1
  local dest="$OUT/$run"
  local args=(--config "configs/xkf/$run.yaml" --output-dir "$dest" --run-name "$run")
  if [[ "$MODE" == debug ]]; then
    args+=(--preflight --max-steps "$STEPS" --frozen-manifest "$OUT/frozen_manifest.json")
  else
    args+=(--frozen-manifest "$XKF_FROZEN_MANIFEST")
  fi
  if [[ -e "$dest/training_metadata.json" ]]; then
    echo "Refusing to overwrite existing training: $dest" >&2
    return 2
  fi
  CUDA_VISIBLE_DEVICES=$cards MASTER_PORT=$port "$PY" -u tools/xkf/train.py "${args[@]}" \
    > "$OUT/logs/$run.log" 2>&1
}
arm 3407 2,3 29731 & p1=$!
arm 3408 4,5 29732 & p2=$!
arm 3409 6,7 29733 & p3=$!
printf '%s\n' "$p1" "$p2" "$p3" > "$OUT/launcher_pids.txt"
failed=0
wait "$p1" || failed=1
wait "$p2" || failed=1
wait "$p3" || failed=1
if [[ "$failed" == 1 ]]; then
  echo "$MODE failed; inspect $OUT/logs" > "$OUT/status.txt"
  exit 1
fi
echo "$MODE training complete; learning evaluation still required" > "$OUT/status.txt"
