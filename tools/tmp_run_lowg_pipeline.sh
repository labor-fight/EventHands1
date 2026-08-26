#!/usr/bin/env bash
# Wait for the two BN-fixed LowG arms to finish training, then select checkpoints on the fixed grid
# and measure retention, so the GPUs are never idle between stages.
set -u
cd /data1/lyq/code/mesh/EventHands1
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
TAGS="0p11 1p05"

for t in $TAGS; do
  while [ ! -f "logs/bnfix_${t}.exit" ]; do sleep 60; done
  echo "$(date +%H:%M:%S) train $t done: $(cat logs/bnfix_${t}.exit)"
done

i=0
for t in $TAGS; do
  gpu=$(echo "0 1" | cut -d' ' -f$((i + 1)))
  CUDA_VISIBLE_DEVICES=$gpu $PY tools/select_checkpoint.py \
    --run-dir outputs/semkine/s24_lowg_${t}_bnfix_s3407 \
    --config configs/semkine/s24_lowg_${t}.yaml \
    --split val_core --step-ms 50 > logs/sel_${t}_bnfix.log 2>&1 &
  i=$((i + 1))
done
wait
echo "$(date +%H:%M:%S) selection done"

ARMS=""
for t in $TAGS; do
  C=$($PY -c "import json;print(json.load(open('outputs/semkine/s24_lowg_${t}_bnfix_s3407/selection_val_core_step50.json'))['selected']['ckpt'])" 2>/dev/null)
  [ -n "$C" ] && ARMS="$ARMS --arm lowg_${t}=$C:configs/semkine/s24_lowg_${t}.yaml"
done
# The baseline's own selected checkpoint, so retention and RA are compared on one protocol.
B=outputs/semkine/s1_track_domrand_s3407/s1_track_domrand_s3407-step=3500.ckpt
[ -f "$B" ] && ARMS="$ARMS --arm baseline=$B:configs/semkine/s1_track_domrand.yaml"

CUDA_VISIBLE_DEVICES=0 $PY tools/tmp_probe_echo_gain.py $ARMS \
  --split val_core --step-ms 50 --out outputs/semkine/lowg_gain_bnfix.json \
  > logs/lowg_gain_bnfix.log 2>&1
echo "$(date +%H:%M:%S) gain probe done"
echo LOWG_PIPELINE_COMPLETE
