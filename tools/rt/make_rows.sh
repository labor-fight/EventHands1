#!/bin/sh
# Root-tracking round: main rows (tools/report_table.py format) of the final comparison, latency measured
# on one otherwise idle GPU, one arm after another. Usage: tools/rt/make_rows.sh GPU
GPU=$1
cd /data1/lyq/code/mesh/EventHands1 || exit 1
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
S=outputs/semkine
export CUDA_VISIBLE_DEVICES=$GPU
set -x
# last step of every run (no selection), the round's headline
$PY tools/tracking/evalx.py row --arm rt_s37 --runs $S/rt_s37_s3407 $S/rt_s37_s3408 --ckpt last --variant tf_pert
$PY tools/tracking/evalx.py row --arm rt_cnntrack --runs $S/rt_cnntrack_s3407 $S/rt_cnntrack_s3408 --ckpt last --variant tf_pert
$PY tools/tracking/evalx.py row --arm rt_cnn --runs $S/rt_cnn_s3407 $S/rt_cnn_s3408 $S/rt_cnn_s3409 --ckpt last --variant tf_pert
$PY tools/rt/pair_eval.py row --arm rt_anchor --runs $S/rt_anchor_s3407 $S/rt_anchor_s3408 $S/rt_anchor_s3409 --ckpt last --variant tf_pert
$PY tools/rt/pair_eval.py row --arm rt_cnnf --runs $S/rt_cnnf_s3407 $S/rt_cnnf_s3408 $S/rt_cnnf_s3409 --ckpt last --variant tf_pert
# zgz-selected step, the convention of the historic main rows
$PY tools/tracking/evalx.py row --arm rt_s37_sel --runs $S/rt_s37_s3407 $S/rt_s37_s3408 --ckpt selected
$PY tools/tracking/evalx.py row --arm rt_cnn_sel --runs $S/rt_cnn_s3407 $S/rt_cnn_s3408 $S/rt_cnn_s3409 --ckpt selected
