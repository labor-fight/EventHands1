"""Two-fold check of the evidence-window choice: folds = even / odd 10 s blocks of each zgz sequence (as in the B-package
filter sweep). For each seed and filter: pick W on fold A (min RA), report fold B at that W against fold B at W = 50; and
the other way round. Out-of-fold gain = mean of the two reports."""
import json, numpy as np
from pathlib import Path
REPO = Path("/data1/lyq/code/mesh/EventHands1")
SEEDS = (3407, 3408, 3409); WS = (50, 75, 100, 150, 200, 300)
TAGS = {"raw": "bare", "r0.5_f1.0_t0.5": "const", "afad1f21_rn0.3-0.5-0.8_f0.8_t0.5": "adaptive"}
def npz(seed, tag, w):
    p = REPO / (f"outputs/dt2/filter_sweep/dt_dz_l3_s{seed}/{tag}.npz" if w == 50 else f"outputs/dt2/wscan/dt_dz_l3_s{seed}/{tag}_w{w}.npz")
    return np.load(p) if p.exists() else None
def fold_ra(z):
    out = {0: [], 1: []}
    for s in ("zgz_global", "zgz_local"):
        e, ra = z[f"model|{s}|end"], z[f"model|{s}|mpjpe_ra_mm"]
        f = (e // 10000) % 2
        for k in (0, 1): out[k].append(ra[f == k])
    return {k: float(np.concatenate(v).mean()) for k, v in out.items()}
for tag, name in TAGS.items():
    gains, picks = [], []
    for seed in SEEDS:
        fr = {w: fold_ra(npz(seed, tag, w)) for w in WS if npz(seed, tag, w) is not None}
        if len(fr) < len(WS): print(f"  {name} s{seed}: only {sorted(fr)} available"); continue
        for sel, rep in ((0, 1), (1, 0)):
            w_star = min(fr, key=lambda w: fr[w][sel])
            gains.append(fr[w_star][rep] - fr[50][rep]); picks.append(w_star)
    if gains:
        print(f"{name:9s} out-of-fold dRA vs W=50: {np.mean(gains):+.3f} mm (per selection: {', '.join(f'{g:+.2f}' for g in gains)}); picked W: {picks}")
