#!/usr/bin/env bash
# S38 post-training pipeline (docs/S38_MESH3D_PREREG.md section 6). Waits for `tools/run_s38.sh` to
# print S38_DONE (both arms trained and selected), then on GPUs 4 and 6:
#
#   1. main rows for both arms                       tools/make_s36_row.py         (6: mesh3d, 4: rootlever)
#   2. rotation / articulation decomposition, grid   tools/probe_s37_meshgraph.py  (4)
#      mechanism gate, observations zeroed           tools/probe_s38_gate.py       (6)
#   3. closed loop + prev-noise sweep vs meshgraph   tools/run_closed_loop_probe.py (6)
#   4. a compact summary against the pre-registered gates, appended to logs/finish_s38_summary.log
#
#   nohup tools/finish_s38.sh > logs/finish_s38_outer.log 2>&1 &
#   S38_NO_WAIT=1 tools/finish_s38.sh        # runs are already selected; skip the wait
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
export PYTHONUNBUFFERED=1
mkdir -p logs outputs/semkine

ARMS=(s38_mesh3d s38_rootlever)
RUNS=()
for arm in "${ARMS[@]}"; do for s in 3407 3408; do RUNS+=("outputs/semkine/${arm}_s${s}"); done; done

if [ -z "${S38_NO_WAIT:-}" ]; then
  echo "[$(date +%F' '%T)] waiting for S38_DONE in logs/run_s38_outer.log"
  until grep -q "^S38_DONE" logs/run_s38_outer.log 2>/dev/null; do sleep 120; done
  echo "[$(date +%F' '%T)] S38_DONE seen"
fi
for r in "${RUNS[@]}"; do
  ls "$r"/selection_val_core_step50*.json > /dev/null 2>&1 || { echo "$r: no selection JSON, aborting"; exit 2; }
done

echo "[$(date +%F' '%T)] 1. main rows"
CUDA_VISIBLE_DEVICES=6 $PY tools/make_s36_row.py --run s38_mesh3d    > logs/row_s38_mesh3d_zgzproto.log    2>&1 &
CUDA_VISIBLE_DEVICES=4 $PY tools/make_s36_row.py --run s38_rootlever > logs/row_s38_rootlever_zgzproto.log 2>&1 &
wait

echo "[$(date +%F' '%T)] 2. rotation decomposition (GPU 4) and mechanism gate (GPU 6)"
CUDA_VISIBLE_DEVICES=4 $PY tools/probe_s37_meshgraph.py --grid --seq zgz_local \
    --runs "${RUNS[@]}" --out outputs/semkine/probe_s38_rotdecomp.json > logs/probe_s38_rotdecomp.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 $PY tools/probe_s38_gate.py --runs "${RUNS[@]}" \
    --out outputs/semkine/probe_s38_gate.json > logs/probe_s38_gate.log 2>&1 &
wait

echo "[$(date +%F' '%T)] 3. closed loop + prev-noise sweep vs meshgraph controls (GPU 6)"
sel() { $PY -c "import json,glob,sys; print(json.load(open(sorted(glob.glob(sys.argv[1]+'/selection_val_core_step50*.json'))[-1]))['selected']['ckpt'])" "$1"; }
CUDA_VISIBLE_DEVICES=6 $PY tools/run_closed_loop_probe.py --split val_core --prev-noise 0.5,1,2,4 \
    --out outputs/semkine/closed_loop_s38_vs_meshgraph.json \
    --arm m3d_3407="$(sel outputs/semkine/s38_mesh3d_s3407)":configs/semkine/s38_mesh3d_s3407.yaml \
    --arm m3d_3408="$(sel outputs/semkine/s38_mesh3d_s3408)":configs/semkine/s38_mesh3d_s3407.yaml \
    --arm rl_3407="$(sel outputs/semkine/s38_rootlever_s3407)":configs/semkine/s38_rootlever_s3407.yaml \
    --arm rl_3408="$(sel outputs/semkine/s38_rootlever_s3408)":configs/semkine/s38_rootlever_s3407.yaml \
    --arm mg_3407=outputs/semkine/s37_meshgraph_s3407/s37_meshgraph_s3407-step=3000.ckpt:configs/semkine/s37_meshgraph_s3407.yaml \
    --arm mg_3408=outputs/semkine/s37_meshgraph_s3408/s37_meshgraph_s3408-step=5000.ckpt:configs/semkine/s37_meshgraph_s3407.yaml \
    > logs/closed_loop_s38.log 2>&1

echo "[$(date +%F' '%T)] 4. summary"
$PY - <<'EOF' | tee logs/finish_s38_summary.log
import json, glob
from pathlib import Path
def J(p):
    try: return json.load(open(p))
    except Exception as e: return {"_error": str(e)}
print("== S38 vs pre-registered gates (docs/S38_MESH3D_PREREG.md section 4) ==")
print("controls: s37_meshgraph RA 24.89 (20.48 / 29.30); s37_fkgraph 22.10 (22.64 / 21.56); s37_routed 20.74")
rot = J("outputs/semkine/probe_s38_rotdecomp.json")
gate = J("outputs/semkine/probe_s38_gate.json")
cl = J("outputs/semkine/closed_loop_s38_vs_meshgraph.json").get("arms", {})
for arm in ("s38_mesh3d", "s38_rootlever"):
    row = J(f"outputs/semkine/{arm}_main_row.json")
    ra = {}
    for s in ("3407", "3408"):
        sel = sorted(glob.glob(f"outputs/semkine/{arm}_s{s}/selection_val_core_step50*.json"))
        if sel:
            d = json.load(open(sel[-1]))
            ra[s] = (d["selected"]["step"], d["selected"]["mpjpe_ra_mm"], d["grid_median"])
    print(f"\n-- {arm}")
    for s, (step, v, med) in ra.items():
        print(f"  seed {s}: selected step {step}  recursive RA {v:.2f}  (grid median {med:.2f})")
    if len(ra) == 2:
        m = sum(v for _, v, _ in ra.values()) / 2
        ok = m < 22.10 and ra["3407"][1] <= 22.64 and ra["3408"][1] <= 21.56
        print(f"  two-seed mean {m:.2f}  -> accuracy gate (< 22.10 and per seed <= fkgraph) {'PASS' if ok else 'FAIL'}"
              f"   vs meshgraph {m - 24.89:+.2f}, vs routed {m - 20.74:+.2f}")
    if "two_seed_mean" in row:
        t = row["two_seed_mean"]
        print(f"  main row: local {t['local']['mpjpe_ra_mm']:.2f} global {t['global']['mpjpe_ra_mm']:.2f} "
              f"| latency {row['latency_ms_scaled_full1p75']:.2f} ms | forward_packet {row['macs_forward_packet']/1e9:.3f} G "
              f"| params {row['params_total']/1e6:.2f} M")
    for s in ("3407", "3408"):
        r = rot.get(f"{arm}_s{s}")
        if r and "closed_loop" in r:
            c, g = r["closed_loop"], r["grid_summary"]
            print(f"  seed {s} rotation: closed p50 {c['rot_p50_deg']:.1f} deg (gate <= 22), grid p50 {g['rot_p50_mean']:.1f} +- {g['rot_p50_sd']:.1f} "
                  f"(gate sd <= 4) | fingers closed {c['ra_rotaligned']:.1f} grid {g['rotaligned_mean']:.1f} +- {g['rotaligned_sd']:.1f} (18-23) "
                  f"| TF RA {r['teacher_forced']['ra']:.2f} rot {r['teacher_forced']['rot_p50_deg']:.1f} | corr(RA,rot) {g['corr_ra_rot']:.2f}")
        q = gate.get(f"{arm}_s{s}")
        if q and "degradation_mm" in q:
            print(f"  seed {s} mechanism gate: live {q['live']['ra_mm']:.2f} -> zeroed {q['observations_zeroed']['ra_mm']:.2f} "
                  f"({q['degradation_mm']:+.2f} mm, gate >= 1.5) {'PASS' if q['gate_passed'] else 'FAIL'}")
    for s, lab in (("3407", "m3d_3407" if arm == "s38_mesh3d" else "rl_3407"),
                   ("3408", "m3d_3408" if arm == "s38_mesh3d" else "rl_3408")):
        a = cl.get(lab)
        if a:
            x4 = a.get("prev_noise_sensitivity", {}).get("x4", {})
            print(f"  seed {s} closed loop: TF {a['teacher_forced']['ra_mm']:.2f} recursive {a['recursive']['ra_mm']:.2f} "
                  f"amplification {a['amplification']:.2f} | single-step RA {x4.get('ra_mm', float('nan')):.2f} "
                  f"@ conditioning error {x4.get('cond_err_ra_mm', float('nan')):.1f} mm (gate <= 16.9 @ 26 mm)")
print("\ncontrols meshgraph in the same closed-loop probe:", {k: round(v['recursive']['ra_mm'], 2) for k, v in cl.items() if k.startswith('mg_')})
EOF
echo "[$(date +%F' '%T)] FINISH_S38_DONE"
