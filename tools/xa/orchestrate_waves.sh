#!/usr/bin/env bash
# [XYZ-ASSOC] waves 2 and 3: wait for the previous wave's three trainings to finish, then launch.
# Wave 2: xa_orc_s3407 (2,3), xa_orc_s3408 (4,5), xa_c0_s3407 (6,7)
# Wave 3: xa_c0_s3408 (2,3), xa_ctr_s3407 (4,5), xa_ctr_s3408 (6,7)   [T1-e passed 2026-10-09]
# Abort switch: touch outputs/xa/STOP before a wave starts to skip it.
set -uo pipefail
cd /data1/lyq/code/mesh/EventHands1
LOG=outputs/xa/launch.log

wait_done() {   # names...
  while true; do
    ok=1
    for n in "$@"; do
      grep -q "train ${n} done rc=" "$LOG" || ok=0
    done
    [ $ok -eq 1 ] && return 0
    sleep 30
  done
}
launch() {   # name gpus
  nohup tools/xa/launch_run.sh "$1" "$2" > "logs/launch_$1.out" 2>&1 &
  echo "[$(date +%F' '%T)] orchestrator launched $1 on $2 pid $!" | tee -a "$LOG"
}

wait_done xa_c1_s3407 xa_c1_s3408 xa_c1r_s3407
if [ -f outputs/xa/STOP ]; then echo "[$(date +%F' '%T)] STOP present, wave 2 skipped" | tee -a "$LOG"; exit 0; fi
echo "[$(date +%F' '%T)] WAVE2_LAUNCH" | tee -a "$LOG"
launch xa_orc_s3407 2,3
launch xa_orc_s3408 4,5
launch xa_c0_s3407 6,7
sleep 120
wait_done xa_orc_s3407 xa_orc_s3408 xa_c0_s3407
if [ -f outputs/xa/STOP ]; then echo "[$(date +%F' '%T)] STOP present, wave 3 skipped" | tee -a "$LOG"; exit 0; fi
echo "[$(date +%F' '%T)] WAVE3_LAUNCH" | tee -a "$LOG"
launch xa_c0_s3408 2,3
launch xa_ctr_s3407 4,5
launch xa_ctr_s3408 6,7
sleep 120
wait_done xa_c0_s3408 xa_ctr_s3407 xa_ctr_s3408
echo "[$(date +%F' '%T)] ALL_WAVES_DONE" | tee -a "$LOG"
