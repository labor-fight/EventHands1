"""Evidence-window sweep (zero training): outputs/dt2/wscan/<run>/<tag>_w<W>.json against the 50 ms records."""
import json, glob, sys
from pathlib import Path
import numpy as np
REPO = Path("/data1/lyq/code/mesh/EventHands1")
SEEDS = (3407, 3408, 3409)
TAGS = {"raw": "bare", "r0.5_f1.0_t0.5": "const 0.5/1/0.5", "afad1f21_rn0.3-0.5-0.8_f0.8_t0.5": "adaptive"}
def path(seed, tag, w):
    if w == 50:
        return REPO / f"outputs/dt2/filter_sweep/dt_dz_l3_s{seed}/{tag}.json"
    return REPO / f"outputs/dt2/wscan/dt_dz_l3_s{seed}/{tag}_w{w}.json"
def met(d):
    m = d["model"]; j = m["jitter"]
    return dict(RA=m["overall"]["mpjpe_ra_mm"], g=m["zgz_global"]["mpjpe_ra_mm"][0], l=m["zgz_local"]["mpjpe_ra_mm"][0],
                rot=m["overall"]["root_rot_deg"], acc=j["acc_err_ra_mm"], accr=j["acc_ratio_ra"],
                rg=m["zgz_global"]["motion"]["root_speed_ratio"], rl=m["zgz_local"]["motion"]["root_speed_ratio"],
                fg=m["zgz_global"]["motion"]["finger_speed_ratio"], fl=m["zgz_local"]["motion"]["finger_speed_ratio"],
                tf=d.get("tf", {}).get("overall", {}).get("mpjpe_ra_mm", float("nan")),
                fail=sum(m[s]["failure"]["episodes"] for s in ("zgz_global", "zgz_local")),
                lo=m["zgz_local"]["by_events"]["[0,500)"]["mpjpe_ra_mm"])
print("| W ms | filter | seeds | RA | g / l | local <500 ev | rot | acc_ra | acc ratio | root speed g/l | finger speed g/l | fail |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for w in (50, 75, 100, 150, 200, 300):
    for tag, name in TAGS.items():
        ms = [met(json.loads(path(s, tag, w).read_text())) for s in SEEDS if path(s, tag, w).exists()]
        if not ms: continue
        M = {k: np.mean([x[k] for x in ms]) for k in ms[0]}
        print(f"| {w} | {name} | {len(ms)} | {M['RA']:.3f} | {M['g']:.2f} / {M['l']:.2f} | {M['lo']:.2f} | {M['rot']:.2f} | {M['acc']:.2f} | {M['accr']:.2f} | {M['rg']:.2f} / {M['rl']:.2f} | {M['fg']:.3f} / {M['fl']:.3f} | {M['fail']:.0f} |")
