#!/bin/bash
# trk6 / S42: formal cost measurements on an idle GPU, one process at a time.
#  1. full-chain latency (tools/trk6/bench_chain.py), S39 and the delivery candidate alternated twice;
#  2. main-table rows (tools/trk6/make_rows.sh -> evalx row: eager latency x 1.75 anchor, thop MACs, params) for S39
#     re-measured (arm trk6_s39ref, same S39-stack evaluations as the official row) and the S42 arms.
# usage: tools/trk6/cost_run.sh GPU
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
cd /data1/lyq/code/mesh/EventHands1
G=$1
mkdir -p outputs/trk6/cost
for i in 1 2; do
  CUDA_VISIBLE_DEVICES=$G $PY tools/trk6/bench_chain.py --run-dir outputs/semkine/dt_dz_l3_s3407 --ckpt step=6000 \
    --max-steps 800 --passes 3 --out outputs/trk6/cost/chain_s39_r$i.json 2>&1 | grep "total p50"
  CUDA_VISIBLE_DEVICES=$G $PY tools/trk6/bench_chain.py --run-dir outputs/semkine/trk6_dzl3w192self_6k_s3407 --ckpt last \
    --max-steps 800 --passes 3 --out outputs/trk6/cost/chain_dzl3w192self_r$i.json 2>&1 | grep "total p50"
  CUDA_VISIBLE_DEVICES=$G $PY tools/trk6/bench_chain.py --run-dir outputs/semkine/trk6_g112self_6k_s3407 --ckpt last \
    --max-steps 800 --passes 3 --out outputs/trk6/cost/chain_g112self_r$i.json 2>&1 | grep "total p50"
done
R=outputs/semkine
tools/trk6/make_rows.sh $G outputs/dt3/gate_eval/ramp1000_w200 trk6_s39ref dt_dz_l3_s3407 dt_dz_l3_s3408 dt_dz_l3_s3409 \
  | grep -E '"latency|"macs|"params' | tr -d ' ' | tr '\n' ' '; echo " <- trk6_s39ref"
for arm in dzl3w192self_6k dzl3w192_6k g112_6k g112self_6k; do
  tools/trk6/make_rows.sh $G outputs/trk6/full trk6_$arm trk6_${arm}_s3407 trk6_${arm}_s3408 trk6_${arm}_s3409 \
    | grep -E '"latency|"macs|"params' | tr -d ' ' | tr '\n' ' '; echo " <- trk6_$arm"
done
echo done
