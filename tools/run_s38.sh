#!/usr/bin/env bash
# S38 (docs/S38_MESH3D_PREREG.md): the two arms on the zgz protocol, one after the other, each with
# both seeds in parallel exactly as `tools/run_zgz_protocol.sh` runs them (GPUs 6,7 and 4,5, two
# cards per seed, NCCL_P2P_DISABLE=1, selection on the fixed grid afterwards).
#
#   tools/run_s38.sh                       # mesh3d then rootlever
#   tools/run_s38.sh s38_rootlever         # only the arms named
#   S38_MIN_FREE_MB=0 tools/run_s38.sh     # do not wait for the GPUs
#
# At 2026-09-20 01:00 GPUs 0-7 were all held by another project (train_evtcv4d.py, ~41 of 46 GB
# per card), so by default this script first waits until every one of GPUs 4-7 reports at least
# S38_MIN_FREE_MB (default 12000) MiB free -- the mesh graph arms peak at 9.0 GB per card at
# 2 x 512 (S37 measurement; S38a adds ~40 MB of edge vectors). It polls once a minute and logs
# when it starts, so a `nohup tools/run_s38.sh > logs/run_s38_outer.log 2>&1 &` can be left alone.
set -uo pipefail
cd "$(dirname "$0")/.."

ARMS=("$@")
[ ${#ARMS[@]} -gt 0 ] || ARMS=(s38_mesh3d s38_rootlever)
MIN_FREE=${S38_MIN_FREE_MB:-12000}
mkdir -p logs

for arm in "${ARMS[@]}"; do
  [ -f "configs/semkine/${arm}_s3407.yaml" ] || { echo "no config for ${arm}"; exit 2; }
done

gpus_free() {   # every one of GPUs 4-7 has >= MIN_FREE MiB free
  local ok=1
  while IFS=, read -r idx free; do
    idx=${idx// /}; free=${free// /}
    case "$idx" in 4|5|6|7) [ "$free" -ge "$MIN_FREE" ] || ok=0 ;; esac
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  [ "$ok" = 1 ]
}

if [ "$MIN_FREE" -gt 0 ]; then
  echo "[$(date +%F' '%T)] waiting for GPUs 4-7 to have >= ${MIN_FREE} MiB free each"
  until gpus_free; do sleep 60; done
  echo "[$(date +%F' '%T)] GPUs 4-7 free"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | sed 's/^/    /'
fi

for arm in "${ARMS[@]}"; do
  echo "[$(date +%F' '%T)] ${arm}: tools/run_zgz_protocol.sh ${arm}"
  tools/run_zgz_protocol.sh "${arm}" > "logs/run_${arm}_outer.log" 2>&1
  echo "[$(date +%F' '%T)] ${arm} finished rc=$? (logs/run_${arm}_outer.log)"
  grep -h "^selected\|^grid" logs/${arm}_s340?_select.log 2>/dev/null | sed 's/^/    /'
done
echo "S38_DONE"
