#!/usr/bin/env bash
# K-star CPU ladder driver (docs/EXPERIMENT_RECORDS.md#kstar-prereg-20261025). CPU only; decisions applied in order.
#   KS0 A/A, KS1 fusion ceiling, KS2 stale / persistence and the KS3 zgz validity check run in parallel;
#   KS3 (needs KS2's measured residual profile), then KS4 PA-fit and the descriptive capture curve.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
O=outputs/kstar
L=$O/logs
mkdir -p "$L"
log() { echo "$(date -Iseconds) $*" >> "$L/ladder.log"; }

log "start"
$PY tools/kstar/aa_audit.py --out $O/s0_aa > "$L/ks0.log" 2>&1; log "ks0 rc=$?"
$PY tools/kstar/fusion_ceiling.py --out $O/s1_fusion_ceiling > "$L/ks1.log" 2>&1 & P1=$!
$PY tools/kstar/head_probe.py stale --out $O/s2_stale > "$L/ks2.log" 2>&1 & P2=$!
$PY tools/kstar/corr_rstar.py --validate-zgz --out $O/s3_v0 > "$L/ks3v.log" 2>&1 & P3=$!
wait $P1; log "ks1 rc=$?"
wait $P2; log "ks2 rc=$?"
wait $P3; log "ks3v rc=$?"
$PY tools/kstar/corr_rstar.py --out $O/s3_corr_rstar > "$L/ks3.log" 2>&1 & P4=$!
$PY tools/kstar/head_probe.py pafit --out $O/s4_pafit > "$L/ks4.log" 2>&1 & P5=$!
$PY tools/kstar/head_probe.py capture --out $O/s4_capture > "$L/ks4c.log" 2>&1 & P6=$!
wait $P4; log "ks3 rc=$?"
wait $P5; log "ks4 rc=$?"
wait $P6; log "ks4c rc=$?"
for s in ks1 ks2 ks3 ks4 ks5; do
  $PY tools/kstar/decide.py $s > "$L/decide_$s.log" 2>&1; log "decide $s rc=$?"
done
log "done"
