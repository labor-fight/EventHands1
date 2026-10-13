#!/bin/bash
# trk6 / S42: main-table rows exactly as S39's (outputs/dt3/rows.sh): every run's S39-stack evaluation (filter_eval json:
# W200 clip-run-start + ramp1000 spec + Count50, last checkpoint) is linked as evalx_val_core_last_w200cfg.json, then
# `evalx.py row --variant w200cfg` writes outputs/semkine/<arm>_w200cfg_main_row.json (N-seed mean, eager latency x 1.75
# anchor on the GPU it runs on, thop MACs, params). Run on an idle GPU.
# usage: tools/trk6/make_rows.sh GPU EVALROOT ARM RUN [RUN ...]
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
cd /data1/lyq/code/mesh/EventHands1
GPU=$1; ROOT=$2; ARM=$3; shift 3
dirs=()
for run in "$@"; do
  f=$(ls $ROOT/$run/af9f41a0*.json | head -1)
  [ -f "$f" ] || { echo "no S39-stack evaluation for $run under $ROOT"; exit 1; }
  ln -sfn "$(readlink -f $f)" outputs/semkine/$run/evalx_val_core_last_w200cfg.json
  dirs+=(outputs/semkine/$run)
done
CUDA_VISIBLE_DEVICES=$GPU $PY tools/tracking/evalx.py row --arm $ARM --runs "${dirs[@]}" --ckpt last --variant w200cfg 2>&1 \
  | grep -v "Warn\|pkg_res\|declare_namespace"
