#!/usr/bin/env python3
"""DT2 round: paired comparison of arms against a control under the rules of docs/DT2_PREREG.md section 3.

Reads outputs/semkine/<arm>_s<seed>/evalx_val_core_last_tf_pert.json (last step, no selection) for the arm and the
control on the same seeds. Seeds missing for an arm are skipped and reported.

    python tools/dt/dt2_compare.py --base dt_dz_l3 --arms dt_dz_l3_nfh dt_dz_l3_w50 --seeds 3407 3408 3409
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEQS = ("zgz_global", "zgz_local")
NAN = float("nan")


def load(arm: str, seed: int, name: str):
    p = REPO / "outputs/semkine" / f"{arm}_s{seed}" / name
    return json.loads(p.read_text()) if p.exists() else None


def metrics(d: dict) -> dict:
    m = d["model"]
    nf = {s: m[s]["n_frames"] for s in SEQS}
    tot = sum(nf.values())
    fsr = sum(m[s]["motion"]["finger_speed_ratio"] * nf[s] for s in SEQS) / tot
    j = m["jitter"]
    k1 = d.get("perturb", {}).get("10", {}).get("div_deg", [float("nan")])[0] / 10.0
    return {
        "RA": m["overall"]["mpjpe_ra_mm"], "RA_g": m["zgz_global"]["mpjpe_ra_mm"][0],
        "RA_l": m["zgz_local"]["mpjpe_ra_mm"][0], "MPVPE": m["overall"]["mpvpe_ra_mm"],
        "ABS": m["overall"]["mpjpe_abs_mm"], "rot": m["overall"]["root_rot_deg"],
        "jit": j["jit_pred_mm"], "acc": j["acc_err_mm"], "acc_ra": j.get("acc_err_ra_mm", NAN),
        "accr_ra": j.get("acc_ratio_ra", NAN), "rsr_g": m["zgz_global"]["motion"]["root_speed_ratio"],
        "rsr_l": m["zgz_local"]["motion"]["root_speed_ratio"], "fsr": fsr,
        "fail": sum(m[s]["failure"]["episodes"] for s in SEQS),
        "amp": d["amplification"]["mpjpe_ra_mm"], "TF": d["tf"]["overall"]["mpjpe_ra_mm"], "k1": k1,
    }


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def compare(base: str, arm: str, seeds: list, name: str) -> dict:
    rows = []
    for s in seeds:
        a, b = load(arm, s, name), load(base, s, name)
        if a is None or b is None:
            continue
        rows.append((s, metrics(a), metrics(b)))
    if not rows:
        return {"arm": arm, "seeds": [], "status": "no results"}
    used = [r[0] for r in rows]
    d = {k: [r[1][k] - r[2][k] for r in rows] for k in rows[0][1]}
    A = {k: mean([r[1][k] for r in rows]) for k in rows[0][1]}
    B = {k: mean([r[2][k] for r in rows]) for k in rows[0][1]}
    dra = d["RA"]
    m = mean(dra)
    if m <= -1.1 and all(x < 0 for x in dra):
        verdict = "BETTER" if (len(dra) >= 3 or m <= -2.5) else "better-pending-3409"
    elif m >= 1.1:
        verdict = "WORSE"
    else:
        verdict = "tie-candidate" if (all(x < 0 for x in dra) and m <= -0.5) else "tie"
    rel = lambda k: (A[k] - B[k]) / B[k]                                             # noqa: E731
    guards = []
    if mean(d["ABS"]) > 2.0: guards.append(f"G1 ABS {mean(d['ABS']):+.2f}")
    if mean(d["RA_l"]) > 1.0: guards.append(f"G2 RA_local {mean(d['RA_l']):+.2f}")
    if rel("jit") > 0.05: guards.append(f"G3a jit {100*rel('jit'):+.1f}%")
    if rel("acc") > 0.05: guards.append(f"G3b acc_err {100*rel('acc'):+.1f}%")
    if rel("acc_ra") == rel("acc_ra") and rel("acc_ra") > 0.05: guards.append(f"G3c acc_err_ra {100*rel('acc_ra'):+.1f}%")
    if A["rsr_g"] < 0.9: guards.append(f"G4a root speed g {A['rsr_g']:.2f}")
    if mean(d["fsr"]) < -0.05: guards.append(f"G4b finger speed {mean(d['fsr']):+.3f}")
    if sum(d["fail"]) > 0: guards.append(f"G5 failures +{sum(d['fail']):.0f}")
    if mean(d["amp"]) > 0.05: guards.append(f"G6 amp {mean(d['amp']):+.3f}")
    return {"arm": arm, "seeds": used, "verdict": verdict, "guards": guards, "dRA": dra, "dRA_mean": m,
            "arm_mean": A, "base_mean": B, "delta_mean": {k: mean(v) for k, v in d.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="dt_dz_l3")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408, 3409])
    ap.add_argument("--name", default="evalx_val_core_last_tf_pert.json")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = [compare(a.base, arm, a.seeds, a.name) for arm in a.arms]
    print(f"control {a.base}; paired on available seeds; last step; rules docs/DT2_PREREG.md section 3\n")
    print("| arm | seeds | dRA per seed | dRA mean | RA (arm/base) | RA g / l (arm) | dRA g / l | rot deg | ABS | acc_err_ra | root speed g/l | finger speed | TF RA | amp | verdict | guardrails |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res:
        if not r["seeds"]:
            print(f"| {r['arm']} | - | | | | | | | | | | | | | no results | |")
            continue
        A, B, D = r["arm_mean"], r["base_mean"], r["delta_mean"]
        print(f"| {r['arm']} | {','.join(map(str, r['seeds']))} | {' / '.join(f'{x:+.2f}' for x in r['dRA'])} | "
              f"**{r['dRA_mean']:+.2f}** | {A['RA']:.2f} / {B['RA']:.2f} | {A['RA_g']:.2f} / {A['RA_l']:.2f} | "
              f"{D['RA_g']:+.2f} / {D['RA_l']:+.2f} | {A['rot']:.2f} ({D['rot']:+.2f}) | {A['ABS']:.1f} ({D['ABS']:+.1f}) | "
              f"{A['acc_ra']:.2f} ({D['acc_ra']:+.2f}) | {A['rsr_g']:.2f} / {A['rsr_l']:.2f} | {A['fsr']:.3f} ({D['fsr']:+.3f}) | "
              f"{A['TF']:.2f} | {A['amp']:.3f} | {r['verdict']} | {'; '.join(r['guards']) or '-'} |")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
