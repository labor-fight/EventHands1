#!/usr/bin/env bash
# S39 (docs/S39_COVMAP_PREREG.md) on the zgz protocol with a configurable GPU pair: the same
# steps as `tools/run_zgz_protocol.sh` (two cards per seed, NCCL_P2P_DISABLE=1, selection on the
# fixed grid afterwards) but the two seeds run one after the other on ONE pair, because on
# 2026-09-21 only GPUs 1 and 3 were free (0, 2, 4-7 held by another project).
#
#   tools/run_s39.sh                       # arm s39_covmap, seeds 3407 3408, GPUs 1,3
#   S39_GPUS=6,7 tools/run_s39.sh s39_covmap 3408
set -uo pipefail
cd "$(dirname "$0")/.."

ARM="${1:-s39_covmap}"
shift || true
SEEDS=("$@")
[ ${#SEEDS[@]} -gt 0 ] || SEEDS=(3407 3408)
GPUS="${S39_GPUS:-1,3}"
CFG=configs/semkine/${ARM}_s3407.yaml
[ -f "$CFG" ] || { echo "no config $CFG"; exit 2; }

export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED=1
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
mkdir -p logs

for seed in "${SEEDS[@]}"; do
  name=${ARM}_s${seed}
  echo "[$(date +%F' '%T)] train ${name} on GPUs ${GPUS} (${CFG} --seed ${seed})"
  CUDA_VISIBLE_DEVICES=$GPUS $PY semkine/train.py --config "$CFG" --seed "$seed" \
      --run-name "$name" --output-dir "outputs/semkine/${name}" > "logs/${name}.log" 2>&1
  echo "[$(date +%F' '%T)] train ${name} done rc=$?"
done

IFS=',' read -r G0 G1 <<< "$GPUS"
SEL=("$G0" "$G1")
for i in "${!SEEDS[@]}"; do
  name=${ARM}_s${SEEDS[$i]}; dir=outputs/semkine/${name}
  if ls "$dir"/*step=6000.ckpt > /dev/null 2>&1; then
    gpu=${SEL[$((i % 2))]}
    echo "[$(date +%F' '%T)] select ${name} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES=$gpu $PY tools/select_checkpoint.py --run-dir "$dir" > "logs/${name}_select.log" 2>&1 &
  else
    echo "[$(date +%F' '%T)] ${name}: grid incomplete, not selecting"
  fi
done
wait
for seed in "${SEEDS[@]}"; do
  grep -h "^selected\|^grid" "logs/${ARM}_s${seed}_select.log" 2>/dev/null | sed "s/^/    s${seed}: /"
done
echo "[$(date +%F' '%T)] ${ARM} finished"
echo "S39_DONE"
