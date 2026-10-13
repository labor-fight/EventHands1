#!/usr/bin/env python3
"""trk6 / S42: the pre-registered adoption gates (docs/EXPERIMENT_RECORDS.md#s42-prereg section 5) of an arm vs S39.

Reads, per seed, the S39-stack evaluation (filter_eval json + npz: W200 clip + ramp1000 + Count50) of the arm and of S39,
the cost files (`--arm-cost` / `--s39-cost`: JSON with params, macs, chain_p50_ms) and prints every gate with its numbers.
    python tools/trk6/gate_check.py --arm-root R --arm-prefix trk6_g112self_6k_s --s39-root outputs/trk6/s39_ctl \
        --s39-prefix dt_dz_l3_s [--seeds 3407 3408 3409] [--arm-cost A.json --s39-cost B.json] [--out gates.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "model")); sys.path.insert(0, str(REPO / "tools"))
from mano_layer import ManoLayer                                      # noqa: E402
from trk6.diag_traj import betas_of, ra_err                           # noqa: E402
from trk6.gain_scan import gains, ra_joints                           # noqa: E402
from trk6.lag_scan import lag_stats                                   # noqa: E402

MANO = None
BETAS = {}


def f0(x):
    return x[0] if isinstance(x, list) else x


def filt_json(run_dir):
    js = [f for f in glob.glob(str(Path(run_dir) / "af9f41a0*.json"))]
    if len(js) != 1:
        raise FileNotFoundError(f"{run_dir}: expected one S39-stack json, got {js}")
    return js[0]


def seed_metrics(run_dir):
    f = filt_json(run_dir)
    d = json.load(open(f))["model"]
    z = np.load(f.replace(".json", ".npz"))
    out = {}
    for part, seq in (("g", "zgz_global"), ("l", "zgz_local")):
        s = d[seq]
        out[part + "J"], out[part + "V"] = f0(s["mpjpe_ra_mm"]), f0(s["mpvpe_ra_mm"])
        out[part + "acc"] = s["jitter"]["acc_err_ra_mm"]
        out[part + "rs"], out[part + "fs"] = s["motion"]["root_speed_ratio"], s["motion"]["finger_speed_ratio"]
        b = BETAS.setdefault(seq, torch.from_numpy(betas_of(seq)).view(1, -1))
        p, g = z[f"model|{seq}|pred"].astype(np.float32), z[f"model|{seq}|gt"].astype(np.float32)
        run, el = z[f"model|{seq}|run"], z[f"model|{seq}|elapsed"] / 1000.0
        out[part + "k5"] = gains(ra_joints(MANO, p, b), ra_joints(MANO, g, b), run)[5][0]
        out[part + "lag"] = lag_stats(MANO, p, g, run, b)[2]
        j, _ = ra_err(MANO, p, g, b, torch.device("cpu"))
        age = 8.0 if part == "g" else 4.0
        out[part + "old"] = float(j[el >= age].mean())
        out[part + "n_old"] = int((el >= age).sum())
    n_g, n_l = d["zgz_global"]["n_frames"], d["zgz_local"]["n_frames"]
    out["RA"] = (out["gJ"] * n_g + out["lJ"] * n_l) / (n_g + n_l)
    return out


def main():
    global MANO
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-root", required=True)
    ap.add_argument("--arm-prefix", required=True)
    ap.add_argument("--s39-root", default="outputs/trk6/s39_ctl")
    ap.add_argument("--s39-prefix", default="dt_dz_l3_s")
    ap.add_argument("--seeds", nargs="+", default=["3407", "3408", "3409"])
    ap.add_argument("--arm-cost", default="")
    ap.add_argument("--s39-cost", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    MANO = ManoLayer(str(REPO / "assets/mano_right.npz"), add_mean=False).eval()
    A = {s: seed_metrics(Path(a.arm_root) / f"{a.arm_prefix}{s}") for s in a.seeds}
    S = {s: seed_metrics(Path(a.s39_root) / f"{a.s39_prefix}{s}") for s in a.seeds}
    mean = lambda D, k: float(np.mean([D[s][k] for s in a.seeds]))                      # noqa: E731
    keys = ("RA", "lJ", "lV", "gJ", "gV", "lacc", "gacc", "lrs", "lfs", "grs", "gfs", "lk5", "gk5", "llag", "glag", "lold", "gold")
    print(f"{'':10s} " + " ".join(f"{k:>7s}" for k in keys))
    for name, D in (("S39", S), ("arm", A)):
        print(f"{name:10s} " + " ".join(f"{mean(D, k):7.3f}" for k in keys))
        for s in a.seeds:
            print(f"  {s:8s} " + " ".join(f"{D[s][k]:7.3f}" for k in keys))
    d_l = [A[s]["lJ"] - S[s]["lJ"] for s in a.seeds]
    gates = {}
    gates["1_local_J_V_below_10"] = (mean(A, "lJ") < 10.0 and mean(A, "lV") < 10.0, f"lJ {mean(A, 'lJ'):.3f}, lV {mean(A, 'lV'):.3f}")
    gates["2_local_paired_gain"] = (np.mean(d_l) <= -0.2 and all(x < 0 for x in d_l), f"d lJ {np.mean(d_l):+.3f} ({', '.join(f'{x:+.3f}' for x in d_l)})")
    gates["3_global_not_worse"] = (mean(A, "gJ") <= mean(S, "gJ") + 0.2, f"gJ {mean(A, 'gJ'):.3f} vs {mean(S, 'gJ'):.3f}")
    gates["4_jitter"] = (mean(A, "lacc") <= mean(S, "lacc") and mean(A, "gacc") <= mean(S, "gacc"),
                         f"acc_err_ra l {mean(A, 'lacc'):.3f} vs {mean(S, 'lacc'):.3f}, g {mean(A, 'gacc'):.3f} vs {mean(S, 'gacc'):.3f}")
    gates["5_lag_response"] = (mean(A, "llag") <= mean(S, "llag") + 0.1 and mean(A, "glag") <= mean(S, "glag") + 0.1
                               and mean(A, "lk5") >= mean(S, "lk5") - 0.02,
                               f"lag l {mean(A, 'llag'):.3f} vs {mean(S, 'llag'):.3f}, g {mean(A, 'glag'):.3f} vs {mean(S, 'glag'):.3f}; "
                               f"k5 l {mean(A, 'lk5'):.3f} vs {mean(S, 'lk5'):.3f}")
    if a.arm_cost and a.s39_cost:
        ca, cs = json.load(open(a.arm_cost)), json.load(open(a.s39_cost))
        gates["6_cost"] = (ca["params"] < cs["params"] and ca["macs"] < cs["macs"] and ca["chain_p50_ms"] < cs["chain_p50_ms"],
                           f"params {ca['params']} vs {cs['params']}, MACs {ca['macs'] / 1e9:.3f} vs {cs['macs'] / 1e9:.3f} G, "
                           f"chain p50 {ca['chain_p50_ms']:.3f} vs {cs['chain_p50_ms']:.3f} ms")
    else:
        gates["6_cost"] = (None, "cost files not given")
    gates["7_long_closed_loop"] = (mean(A, "gold") <= mean(S, "gold") + 0.2 and mean(A, "lold") <= mean(S, "lold"),
                                   f"global age>=8s {mean(A, 'gold'):.3f} vs {mean(S, 'gold'):.3f}; local age>=4s {mean(A, 'lold'):.3f} vs {mean(S, 'lold'):.3f}")
    print()
    for k, (ok, msg) in gates.items():
        print(f"{k:24s} {'PASS' if ok else ('n/a' if ok is None else 'FAIL'):5s} {msg}")
    allok = all(v[0] for v in gates.values() if v[0] is not None) and gates["6_cost"][0] is not None
    print("ADOPT" if allok else "NOT ADOPTED (or cost pending)")
    if a.out:
        Path(a.out).write_text(json.dumps({"arm": A, "s39": S, "gates": {k: [None if v[0] is None else bool(v[0]), v[1]] for k, v in gates.items()}}, indent=1))


if __name__ == "__main__":
    main()
