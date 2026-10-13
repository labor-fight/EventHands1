#!/usr/bin/env bash
# Train one [XYZ-ASSOC] arm on a GPU pair, then run the selection grid on the first GPU of the pair
# (selection is for comparability with the historical selected-step numbers only; the
# pre-registered primary readout is the last step). Usage: launch_run.sh <name> <gpus>
set -uo pipefail
cd /data1/lyq/code/mesh/EventHands1
NAME=$1; GPUS=$2
SEED=${NAME##*_s}
CFG=configs/xa/${NAME}.yaml
[ -f "$CFG" ] || { echo "no config $CFG"; exit 2; }
export NCCL_P2P_DISABLE=1 PYTHONUNBUFFERED=1
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
mkdir -p logs outputs/xa
echo "[$(date +%F' '%T)] train ${NAME} on GPUs ${GPUS}" | tee -a outputs/xa/launch.log
CUDA_VISIBLE_DEVICES=$GPUS $PY semkine/train.py --config "$CFG" --seed "$SEED" \
    --run-name "$NAME" --output-dir "outputs/semkine/${NAME}" > "logs/${NAME}.log" 2>&1
rc=$?
echo "[$(date +%F' '%T)] train ${NAME} done rc=${rc}" | tee -a outputs/xa/launch.log
if ls outputs/semkine/${NAME}/*step=6000.ckpt > /dev/null 2>&1; then
  G1=${GPUS%%,*}
  echo "[$(date +%F' '%T)] select ${NAME} on GPU ${G1}" | tee -a outputs/xa/launch.log
  CUDA_VISIBLE_DEVICES=$G1 $PY tools/select_checkpoint.py --run-dir "outputs/semkine/${NAME}" \
      > "logs/${NAME}_select.log" 2>&1
  echo "[$(date +%F' '%T)] select ${NAME} done rc=$?" | tee -a outputs/xa/launch.log
else
  echo "[$(date +%F' '%T)] ${NAME}: no step=6000 ckpt, not selecting" | tee -a outputs/xa/launch.log
fi
echo "XA_RUN_DONE ${NAME}"
