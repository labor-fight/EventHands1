#!/bin/bash
# trk6 / S42 job queue: launch "CONFIG SEED" lines of QUEUEFILE one by one whenever fewer than MAXJOBS trk6 trainings run,
# on the GPU in 4..7 with the fewest trk6 trainings (GPUs 0-3 are never used). Each job: SEMKINE_FAST_PIPE=1, no CPU pinning.
# usage: nohup tools/trk6/queue.sh QUEUEFILE MAXJOBS > outputs/trk6/logs/queue.log 2>&1 &
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
cd /data1/lyq/code/mesh/EventHands1
Q=$1; MAX=${2:-6}
n_running() { pgrep -af "[s]emkine/train.py --config configs/trk6/" | grep -o "run-name [^ ]*" | sort -u | wc -l; }
gpu_of_pid() { tr '\0' '\n' < /proc/$1/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p'; }
pick_gpu() {
  declare -A c=([4]=0 [5]=0 [6]=0 [7]=0)
  for r in $(pgrep -af "[s]emkine/train.py --config configs/trk6/" | grep -o "run-name [^ ]*" | sort -u | cut -d" " -f2); do
    p=$(pgrep -f "[s]emkine/train.py .*--run-name $r " | head -1)
    g=$(gpu_of_pid $p); [ -n "$g" ] && [ -n "${c[$g]+x}" ] && c[$g]=$((c[$g]+1))
  done
  best=4; for g in 4 5 6 7; do [ ${c[$g]} -lt ${c[$best]} ] && best=$g; done; echo $best
}
while read -r cfg seed; do
  [ -z "$cfg" ] && continue
  while [ "$(n_running)" -ge "$MAX" ]; do sleep 60; done
  g=$(pick_gpu); name=$(basename $cfg .yaml); run=trk6_${name}_s$seed
  CUDA_VISIBLE_DEVICES=$g SEMKINE_FAST_PIPE=1 nohup $PY semkine/train.py --config $cfg --seed $seed --num-workers 16 \
    --run-name $run --output-dir outputs/semkine/$run > outputs/trk6/logs/$run.log 2>&1 < /dev/null &
  echo "$(date +%T) launched $run on GPU $g"
  sleep 90
done < $Q
echo "$(date +%T) queue empty"
