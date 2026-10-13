import json, sys, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, "."); sys.path.insert(0, "model")
from semkine.dataset import _read_meta51, sequences_for_split
root = Path("/data1/lyq/code/mesh/EventHands/data/hand_data51")
mani = root / "splits_semkine.json"
out = {}
for split in ("train", "val_core"):
    seqs = sequences_for_split(root, split, mani)
    norms, steps, angs_x, quats = [], [], [], []
    for s, d in seqs:
        p = _read_meta51(str(root / d / s) + ".meta")
        aux = np.load(str(root / d / s) + "_aux.npz", allow_pickle=True)
        runs = np.asarray(aux["valid_runs_ms"]).reshape(-1, 2)
        for a, b in runs:
            idx = np.arange(a + 49, b, 50)
            if len(idx) < 2: continue
            aa = p[idx, 3:6].astype(np.float64)
            n = np.linalg.norm(aa, axis=1); norms.append(n)
            Rm = R.from_rotvec(aa)
            rel = (Rm[1:] * Rm[:-1].inv()).magnitude()   # camera-frame step angle
            steps.append(np.rad2deg(rel))
            # axis-angle additive step vs true step
            angs_x.append(np.linalg.norm(aa[1:] - aa[:-1], axis=1))
            quats.append(Rm.as_quat())
    n = np.concatenate(norms); st = np.concatenate(steps); dx = np.concatenate(angs_x)
    q = np.concatenate(quats)
    mean_R = R.from_quat(q).mean()
    dev = np.rad2deg((R.from_quat(q) * mean_R.inv()).magnitude())
    out[split] = {"n_steps": int(len(n)),
        "aa_norm_deg_pct": {p_: float(np.rad2deg(np.percentile(n, p_))) for p_ in (1, 5, 25, 50, 75, 95, 99, 100)},
        "frac_norm_gt_150deg": float((n > np.deg2rad(150)).mean()),
        "frac_norm_gt_170deg": float((n > np.deg2rad(170)).mean()),
        "step_deg_pct(50ms)": {p_: float(np.percentile(st, p_)) for p_ in (50, 90, 99)},
        "step_deg_mean": float(st.mean()),
        "aa_add_step_norm_over_true(median)": float(np.median(dx / np.maximum(np.deg2rad(st), 1e-6))),
        "dev_from_mean_rot_deg_pct": {p_: float(np.percentile(dev, p_)) for p_ in (50, 90, 99, 100)},
        "mean_rot_rotvec": mean_R.as_rotvec().tolist()}
print(json.dumps(out, indent=1))
