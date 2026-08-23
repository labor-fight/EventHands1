#!/usr/bin/env bash
# Formal δ-trust arm: same checkpoint, inference-time shrinkage. No training.
# Registered control that S9 / S12 must beat in a paired comparison.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
CKPT=${1:?usage: $0 <track_ckpt> [rho=0.5] [split=val] [gpu=3] [config]}
RHO=${2:-0.5}
SPLIT=${3:-val}
GPU=${4:-3}
CFG=${5:-}
if [ -z "$CFG" ]; then
  CFG=$(dirname "$CKPT")
  CFG=$(ls "$CFG"/*.yaml 2>/dev/null | head -1)
  [ -n "$CFG" ] || CFG=configs/semkine/s1_track_domrand.yaml
fi
OUT=outputs/semkine/s1_delta_trust
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPU NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  $PY semkine/eval_track.py --config "$CFG" --ckpt "$CKPT" --split "$SPLIT" \
    --delta-trust "$RHO" --out-dir "$OUT" --arm "delta_trust_${RHO}" --buckets
echo "wrote $OUT"
