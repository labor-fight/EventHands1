#!/usr/bin/env bash
# docs/S37_ROTW_CNNROOT_PREREG.md: part A (S37 retrained with LOSS.LAMBDA_R x10, two seeds; x30, seed
# 3407) and part B (absolute CNN, two seeds, then its root fused into S37's closed loop), all eight GPUs.
set -uo pipefail
cd "$(dirname "$0")/.."
export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED=1
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
mkdir -p logs
ts() { date "+%F %T"; }

train_s37() {   # gpus config seed name
  echo "[$(ts)] train $4 on GPUs $1"
  CUDA_VISIBLE_DEVICES=$1 $PY semkine/train.py --config "$2" --seed "$3" --run-name "$4" \
      --output-dir "outputs/semkine/$4" > "logs/$4.log" 2>&1
  echo "[$(ts)] train $4 done rc=$?"
}

train_cnn() {   # gpu seed name   (the S26 absolute recipe, tools/run_s26_9subj.sh)
  echo "[$(ts)] train $3 on GPU $1"
  CUDA_VISIBLE_DEVICES=$1 NCCL_IB_DISABLE=1 $PY semkine/train.py --config configs/semkine/s1_abs_domrand.yaml \
      --seed "$2" --run-name "$3" --output-dir "outputs/semkine/$3" --devices 1 --max-steps 4000 \
      --batch-size 1024 --num-workers 8 > "logs/$3.log" 2>&1
  echo "[$(ts)] train $3 done rc=$?"
}

select_run() {   # gpu name
  CUDA_VISIBLE_DEVICES=$1 $PY tools/select_checkpoint.py --run-dir "outputs/semkine/$2" > "logs/$2_select.log" 2>&1
  echo "[$(ts)] select $2 done rc=$?"
}

decomp() {   # gpu name
  CUDA_VISIBLE_DEVICES=$1 $PY .experiments/s37_debug_20260928/probe_decomp.py --run "outputs/semkine/$2" \
      --variants base,oracle_rot --out "outputs/semkine/decomp_$2.json" > "logs/decomp_$2.log" 2>&1
  echo "[$(ts)] decomp $2 done rc=$?"
}

( train_s37 0,1 configs/semkine/s37_rotw10_s3407.yaml 3407 s37_rotw10_s3407; select_run 0 s37_rotw10_s3407; decomp 0 s37_rotw10_s3407 ) & a1=$!
( train_s37 2,3 configs/semkine/s37_rotw10_s3407.yaml 3408 s37_rotw10_s3408; select_run 2 s37_rotw10_s3408; decomp 2 s37_rotw10_s3408 ) & a2=$!
( train_s37 4,5 configs/semkine/s37_rotw30_s3407.yaml 3407 s37_rotw30_s3407; select_run 4 s37_rotw30_s3407; decomp 4 s37_rotw30_s3407 ) & a3=$!
( train_cnn 6 3407 s37diag_cnnabs_s3407; select_run 6 s37diag_cnnabs_s3407 ) & b1=$!
( train_cnn 7 3408 s37diag_cnnabs_s3408; select_run 7 s37diag_cnnabs_s3408 ) & b2=$!

wait $b1; wait $b2
CUDA_VISIBLE_DEVICES=6 $PY tools/probe_s37_cnnroot.py \
    --pair outputs/semkine/s37_routed_s3407=outputs/semkine/s37diag_cnnabs_s3407 \
    --pair outputs/semkine/s37_routed_s3408=outputs/semkine/s37diag_cnnabs_s3408 \
    --out outputs/semkine/probe_s37_cnnroot.json > logs/probe_s37_cnnroot.log 2>&1
echo "[$(ts)] part B fusion done rc=$?"

wait $a1; wait $a2; wait $a3
CUDA_VISIBLE_DEVICES=0 $PY tools/make_s36_row.py --run s37_rotw10 > logs/row_s37_rotw10_zgzproto.log 2>&1
echo "[$(ts)] part A main row done rc=$?"
echo "S37_ROTW_CNNROOT_DONE"
