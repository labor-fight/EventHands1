#!/usr/bin/env bash
# S39 post-training pipeline (docs/S39_COVMAP_PREREG.md section 5). Waits for `tools/run_s39.sh` to
# print S39_DONE (both seeds trained and selected), then on GPUs 1 and 3:
#
#   1. main row                                       tools/make_s36_row.py --run s39_covmap      (GPU 1)
#      results table                                  tools/report_table.py
#   2. rotation / articulation decomposition, grid    tools/probe_s37_meshgraph.py (zgz_global)   (GPU 3)
#      covmap mechanism gate                          tools/probe_s39_gate.py --ablate covmap     (GPU 1)
#   3. closed loop + per-step debug log (S37_DBG=1)   tools/run_closed_loop_probe.py              (GPU 3)
#      -> the same per-step instrumentation as the baseline run, for the before/after comparison
#         of the root rotation left after each step (debug session ae9d53, H7 verification)
#
#   nohup tools/finish_s39.sh > logs/finish_s39_outer.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
export PYTHONUNBUFFERED=1
mkdir -p logs outputs/semkine
G0="${S39_GPU0:-1}"; G1="${S39_GPU1:-3}"

if [ -z "${S39_NO_WAIT:-}" ]; then
  echo "[$(date +%F' '%T)] waiting for S39_DONE in logs/run_s39_outer.log"
  until grep -q "^S39_DONE" logs/run_s39_outer.log 2>/dev/null; do sleep 120; done
  echo "[$(date +%F' '%T)] S39_DONE seen"
fi
for s in 3407 3408; do
  ls outputs/semkine/s39_covmap_s${s}/selection_val_core_step50*.json > /dev/null 2>&1 || { echo "s39_covmap_s${s}: no selection JSON, aborting"; exit 2; }
done
sel() { $PY -c "import json,glob,sys; print(json.load(open(sorted(glob.glob(sys.argv[1]+'/selection_val_core_step50*.json'))[-1]))['selected']['ckpt'])" "$1"; }

echo "[$(date +%F' '%T)] 1. main row (GPU $G0) + rotation decomposition (GPU $G1)"
CUDA_VISIBLE_DEVICES=$G0 $PY tools/make_s36_row.py --run s39_covmap > logs/row_s39_covmap_zgzproto.log 2>&1 &
CUDA_VISIBLE_DEVICES=$G1 $PY tools/probe_s37_meshgraph.py --grid --seq zgz_global \
    --runs outputs/semkine/s39_covmap_s3407 outputs/semkine/s39_covmap_s3408 \
           outputs/semkine/s37_routed_s3407 outputs/semkine/s37_routed_s3408 \
    --out outputs/semkine/probe_s39_rotdecomp.json > logs/probe_s39_rotdecomp.log 2>&1 &
wait
$PY tools/report_table.py s37_routed s39_covmap --label s37_routed="S37 路由读出（当前臂）" --label s39_covmap="S39 覆盖图 root 读出" | tee logs/table_s39.log

echo "[$(date +%F' '%T)] 2. covmap gate (GPU $G0) + closed loop with per-step log (GPU $G1)"
CUDA_VISIBLE_DEVICES=$G0 $PY tools/probe_s39_gate.py --ablate covmap \
    --runs outputs/semkine/s39_covmap_s3407 outputs/semkine/s39_covmap_s3408 \
    --out outputs/semkine/probe_s39_gate.json > logs/probe_s39_gate.log 2>&1 &
S37_DBG=1 CUDA_VISIBLE_DEVICES=$G1 $PY tools/run_closed_loop_probe.py --split val_core \
    --arm s39_covmap_3407="$(sel outputs/semkine/s39_covmap_s3407)":configs/semkine/s39_covmap_s3407.yaml \
    --arm s39_covmap_3408="$(sel outputs/semkine/s39_covmap_s3408)":configs/semkine/s39_covmap_s3407.yaml \
    --out outputs/semkine/closed_loop_s39_vs_routed.json > logs/closed_loop_s39.log 2>&1 &
wait

echo "[$(date +%F' '%T)] 3. summary"
$PY - <<'EOF' | tee logs/finish_s39_summary.log
import json, glob
def J(p):
    try: return json.load(open(p))
    except Exception as e: return {"_error": str(e)}
print("== S39 vs pre-registered gates (docs/S39_COVMAP_PREREG.md section 3); control s37_routed 20.74 (19.23 / 22.26) ==")
ra = {}
for s in ("3407", "3408"):
    d = json.load(open(sorted(glob.glob(f"outputs/semkine/s39_covmap_s{s}/selection_val_core_step50*.json"))[-1]))
    ra[s] = d["selected"]["mpjpe_ra_mm"]; print(f"  seed {s}: step {d['selected']['step']} RA {ra[s]:.2f} (grid median {d['grid_median']:.2f}) per-seq {d['selected']['per_seq']}")
m = (ra["3407"] + ra["3408"]) / 2
print(f"  two-seed mean {m:.2f} -> accuracy gate (< 20.74 and 3407 <= 19.23, 3408 <= 22.26): {'PASS' if m < 20.74 and ra['3407'] <= 19.23 and ra['3408'] <= 22.26 else 'FAIL'}")
rot = J("outputs/semkine/probe_s39_rotdecomp.json")
for k, v in rot.items():
    if "closed_loop" in v:
        c, g, t = v["closed_loop"], v.get("grid_summary", {}), v["teacher_forced"]
        print(f"  {k}: zgz_global closed rot p50 {c['rot_p50_deg']:.1f} deg (gate <= 8 for s39) fingers {c['ra_rotaligned']:.1f} | grid rot {g.get('rot_p50_mean', float('nan')):.1f} +- {g.get('rot_p50_sd', float('nan')):.1f} | TF RA {t['ra']:.2f} rot {t['rot_p50_deg']:.1f}")
gate = J("outputs/semkine/probe_s39_gate.json")
for k, v in gate.items():
    if "degradation_mm" in v:
        print(f"  {k} covmap gate: live {v['live']['ra_mm']:.2f} -> covmap zeroed {v['ablated']['ra_mm']:.2f} ({v['degradation_mm']:+.2f} mm, gate >= 1.5) {'PASS' if v['gate_passed'] else 'FAIL'}")
cl = J("outputs/semkine/closed_loop_s39_vs_routed.json").get("arms", {})
for k, v in cl.items():
    print(f"  {k}: TF {v['teacher_forced']['ra_mm']:.2f} recursive {v['recursive']['ra_mm']:.2f} amplification {v['amplification']:.2f} (routed 3407: 9.02 / 19.26 / 2.14)")
EOF
echo "[$(date +%F' '%T)] FINISH_S39_DONE"
