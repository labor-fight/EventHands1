#!/usr/bin/env bash
# S36: train the event-graph arm on GPU 0 and 1, one seed after the other.
#
# NCCL_P2P_DISABLE is not a preference. On this box the process group comes up in ~1.3 s and then
# the first collective never returns: a bare 1024-element `all_reduce` between GPU 0 and 1 hung for
# 118 s under the default settings, and the run died before its first batch. With P2P off the same
# probe finishes in under 5 s. `tools/probe_ddp.py` reproduces both halves in about a minute.
# Any dual-card run here needs this set.
set -euo pipefail
cd "$(dirname "$0")/.."

export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1

PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
mkdir -p logs

for seed in 3407 3408; do
  cfg=configs/semkine/s36_eventgnn_s${seed}.yaml
  [ -f "$cfg" ] || { cfg=configs/semkine/s36_eventgnn_s3407.yaml; extra="--seed ${seed}"; }
  name=s36_eventgnn_s${seed}
  echo "[$(date +%H:%M:%S)] $name"
  $PY semkine/train.py --config "$cfg" ${extra:-} \
      --run-name "$name" --output-dir "outputs/semkine/${name}" \
      > "logs/${name}.log" 2>&1
  echo "[$(date +%H:%M:%S)] $name done"
  unset extra
done
