#!/usr/bin/env python3
"""S43 verdict numbers (docs/EXPERIMENT_RECORDS.md#s43-prereg), all read from evaluation artifacts:

  S39 stack   outputs/s43/full/<run>/af9f41a0*_w200_clip.npz      (tools/trk6/screen_eval.sh)
  W50 raw     outputs/semkine/<run>/evalx_val_core_last_tf_pert.npz (evalx eval --controls --tf --perturb)
  S39 / rt_s37 references from their recorded artifacts (same two protocols)

    python tools/s43/verdict.py --out outputs/s43/reports/verdict.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SEEDS = (3407, 3408, 3409)
KEYS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "root_rot_deg", "transl_mm", "mpjpe_abs_mm")
S39_STACK = "af9f41a0_rn0.015-0.03-0.06-0.09-0.172185-0.261517-0.404576-0.5-0.8_fn0.04-0.08-0.16-0.24-0.4-0.56-0.8_tn0.025-0.05-0.1-0.15-0.25-0.35-0.5_w200_clip.npz"


def metrics(npz: Path, mode: str = "model") -> dict:
    z = np.load(npz)
    out, tot = {}, {k: [0.0, 0] for k in KEYS}
    for s in ("zgz_global", "zgz_local"):
        for k in KEYS:
            v = z[f"{mode}|{s}|{k}"]
            out[f"{s.split('_')[1]}.{k}"] = float(v.mean())
            tot[k][0] += float(v.sum())
            tot[k][1] += len(v)
    for k, (a, n) in tot.items():
        out[f"overall.{k}"] = a / n
    return out


def stack_npz(run: str) -> Path:
    for root in (REPO / "outputs" / "s43" / "full" / run, REPO / "outputs" / "trk6" / "s39_repro" / run):
        p = root / S39_STACK
        if p.exists():
            return p
    raise FileNotFoundError(run)


def arm(prefix: str, proto: str) -> dict:
    res = {}
    for s in SEEDS:
        run = f"{prefix}_s{s}"
        try:
            p = stack_npz(run) if proto == "stack" else REPO / "outputs" / "semkine" / run / "evalx_val_core_last_tf_pert.npz"
            res[s] = metrics(p)
        except (FileNotFoundError, KeyError):
            pass
    return res


def mean(d: dict, k: str) -> float:
    return float(np.mean([v[k] for v in d.values()]))


def paired(a: dict, b: dict, k: str):
    common = sorted(set(a) & set(b))
    d = [a[s][k] - b[s][k] for s in common]
    return {"seeds": common, "diff": d, "mean": float(np.mean(d)) if d else None,
            "n_neg": int(sum(x < 0 for x in d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "outputs" / "s43" / "reports" / "verdict.json"))
    ap.add_argument("--candidate", default=None, help="the arm prefix the G1/G2/S39 verdicts use (default: best by RA)")
    ap.add_argument("--arms", nargs="*", default=[], help="further arm prefixes to tabulate")
    a = ap.parse_args()
    names = list(dict.fromkeys(["s43n_sp", "s43n_spself", "s43n_sptr", "s43n_spselftr"] + a.arms + ["dt_dz_l3"]))
    arms = {p: arm(p, "stack") for p in names}
    raw = {p: arm(p, "raw") for p in names if p != "dt_dz_l3"}
    raw["rt_s37"] = arm("rt_s37", "raw")
    rep = {"stack": {p: {"per_seed": v, "mean": {k: mean(v, k) for k in next(iter(v.values()))}} for p, v in arms.items() if v},
           "raw": {p: {"per_seed": v, "mean": {k: mean(v, k) for k in next(iter(v.values()))}} for p, v in raw.items() if v}}
    st = rep["stack"]
    full = [p for p in st if p.startswith("s43n") and len(arms[p]) == 3]
    cand = a.candidate or (min(full, key=lambda p: st[p]["mean"]["overall.mpjpe_ra_mm"]) if full else None)
    g = {}
    s39 = arms["dt_dz_l3"]
    if "s43n_sp" in st and len(arms["s43n_sp"]) == 3:
        m = st["s43n_sp"]["mean"]
        g["G1"] = {"ra": m["overall.mpjpe_ra_mm"], "local": m["local.mpjpe_ra_mm"],
                   "pass": m["overall.mpjpe_ra_mm"] <= 8.71 + 0.30 and m["local.mpjpe_ra_mm"] <= 10.77}
    if arms["s43n_spself"] and arms["s43n_sp"]:
        p = paired(arms["s43n_spself"], arms["s43n_sp"], "overall.mpjpe_ra_mm")
        g["G3"] = dict(p, local=paired(arms["s43n_spself"], arms["s43n_sp"], "local.mpjpe_ra_mm"),
                       pass_=bool(p["mean"] is not None and len(p["seeds"]) == 3 and p["mean"] <= -0.2 and p["n_neg"] == 3))
    for tr, base in (("s43n_sptr", "s43n_sp"), ("s43n_spselftr", "s43n_spself")):
        if arms[tr] and arms[base]:
            t = paired(arms[tr], arms[base], "overall.transl_mm")
            ratio = [arms[tr][s]["overall.transl_mm"] / arms[base][s]["overall.transl_mm"] for s in t["seeds"]]
            r = paired(arms[tr], arms[base], "overall.mpjpe_ra_mm")
            g[f"G7:{tr}"] = {"transl_ratio": ratio, "transl_ratio_mean": float(np.mean(ratio)), "ra": r,
                             "global_transl": paired(arms[tr], arms[base], "global.transl_mm"),
                             "local_transl": paired(arms[tr], arms[base], "local.transl_mm"),
                             "pass": bool(len(ratio) == 3 and float(np.mean(ratio)) <= 0.8 and all(x < 1 for x in ratio)
                                          and r["mean"] <= 0.2)}
    if cand:
        rc = rep["raw"].get(cand, {}).get("mean")
        rs = rep["raw"].get("rt_s37", {}).get("mean")
        if rc and rs:
            g["G2"] = {"candidate": cand, "ra": [rc["overall.mpjpe_ra_mm"], rs["overall.mpjpe_ra_mm"]],
                       "root_deg": [rc["overall.root_rot_deg"], rs["overall.root_rot_deg"]],
                       "local_transl": [rc["local.transl_mm"], rs["local.transl_mm"]],
                       "global_transl": [rc["global.transl_mm"], rs["global.transl_mm"]],
                       "pass": {"ra": rc["overall.mpjpe_ra_mm"] <= rs["overall.mpjpe_ra_mm"] - 10,
                                "root": rc["overall.root_rot_deg"] <= 8.0, "local_transl": rc["local.transl_mm"] <= 50.0,
                                "global_transl": rc["global.transl_mm"] <= 40.0}}
        if len(arms[cand]) == 3:
            p = paired(arms[cand], s39, "overall.mpjpe_ra_mm")
            g["S39"] = {"candidate": cand, "paired": p, "local": paired(arms[cand], s39, "local.mpjpe_ra_mm"),
                        "global": paired(arms[cand], s39, "global.mpjpe_ra_mm"),
                        "better": bool(p["mean"] <= -0.2 and p["n_neg"] == 3), "tie": bool(abs(p["mean"]) <= 0.3)}
    rep["gates"], rep["candidate"] = g, cand
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    fmt = lambda d: " ".join(f"{k.split('.')[0][0]}{k.split('.')[1][:5]}={v:.3f}" for k, v in d.items() if k.endswith(("ra_mm", "deg", "transl_mm")))  # noqa: E731
    for proto in ("stack", "raw"):
        print(f"== {proto}")
        for p, v in rep[proto].items():
            print(f"{p:16s} n={len(v['per_seed'])} RA={v['mean']['overall.mpjpe_ra_mm']:.3f} "
                  f"g={v['mean']['global.mpjpe_ra_mm']:.3f} l={v['mean']['local.mpjpe_ra_mm']:.3f} "
                  f"Vg={v['mean']['global.mpvpe_ra_mm']:.3f} Vl={v['mean']['local.mpvpe_ra_mm']:.3f} "
                  f"rot={v['mean']['overall.root_rot_deg']:.2f} Tg={v['mean']['global.transl_mm']:.1f} "
                  f"Tl={v['mean']['local.transl_mm']:.1f}  per-seed RA "
                  + " / ".join(f"{x['overall.mpjpe_ra_mm']:.2f}" for x in v["per_seed"].values()))
    print(json.dumps(g, indent=1, default=float))


if __name__ == "__main__":
    main()
