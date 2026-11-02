#!/usr/bin/env python3
"""S45-RT decision and report (docs/EXPERIMENT_RECORDS.md#s45rt-prereg-20261031); CPU only, run once after every
registered reading. Writes outputs/s45rt/20261031/summary.json and report.md. Single seed 3407, descriptive.

Each arm (T1b, T1p) against S45:
  R_s = (S45_s - arm_s) / (S45_s - T1d_s), s in 5000 / 5500 / 6000, each the median over the nine training local
      recordings of the recording's mean E1 RA
  C   = median RA after 20 feeds of the same packet from the training-mean finger pose / after 1 feed (training local,
      step 6000; S45-TR readout tl_f1)
  zgz (XCL single-seed rule, as tools/s45px/decide.py): per domain, E1 gain >= 5 mm (local) / 8 mm (global) at 6000,
      E2 and E3 gains > 0, and >= 2 of the three late checkpoints clear the margin
  class: DEPLOYABLE (min R >= 0.3, C <= 0.8, zgz support in a domain) > IN_SAMPLE (min R >= 0.3, C <= 0.8)
      > NULL (max R < 0.15) > WEAK (otherwise); NOT_TRAINED when T0P failed its gate (T1p only)
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.s45rt.common import OUT, S45CT, S45_EVAL, S45TR, STEPS, now, read, require, save, verify_protocol  # noqa: E402

TH = {"R_min": 0.3, "C_conv": 0.8, "R_null": 0.15, "zgz_local_mm": 5.0, "zgz_global_mm": 8.0, "late_needed": 2}
NEXT = {
    "DEPLOYABLE": "deployable routing gives S45 closed-loop correction: merge the router into S45 as a second head "
                  "(single-stage multi-task), add own-loop priors (direction 3) and a second seed; new pre-registration",
    "IN_SAMPLE": "the mechanism holds on the training subjects only: zero-training router accuracy / coverage on zgz",
    "WEAK": "partial: for T1b, T1p's result decides; for T1p, a longer-window router (new pre-registration)",
    "NULL": "this router's errors are not absorbed by training; both NULL -> stop the deployable-routing line",
    "NOT_TRAINED": "T0P failed its gate: T1p not trained (it would repeat T1b)",
}


def trace(directory):
    m = read(Path(directory) / "report.json")["metrics"]
    return {d: m[d]["mpjpe_ra_mm"] for d in ("local", "global")}


def zgz(net):
    if net == "s45":
        e1 = {s: trace(S45_EVAL / f"e1_step{s}_repeat1_none") for s in STEPS}
        return {"e1": e1, "e2": e1[6000], "e3": trace(S45_EVAL / "e3_step6000_repeat1_none")}
    ev = (S45TR if net == "t1b" else OUT) / "evaluations" / net
    return {"e1": {s: trace(ev / f"e1_step{s}") for s in STEPS}, "e2": trace(ev / "e2_step6000"),
            "e3": trace(ev / "e3_step6000")}


def insample_path(net, step):
    if net in ("s45", "t1d"):
        return (S45TR / "readouts" / net if step == 6000 else S45CT / "readouts" / f"{net}_{step}") / "tl_e1/states.npz"
    if net == "t1b" and step == 6000:
        return S45TR / "readouts/t1b/tl_e1/states.npz"
    return OUT / "readouts" / (net if step == 6000 else f"{net}_{step}") / "tl_e1/states.npz"


def insample(net, step):
    with np.load(insample_path(net, step), allow_pickle=False) as z:
        recs = sorted({k.split("/")[0] for k in z.files})
        per = {r: float(z[f"{r}/mpjpe_ra_mm"].mean()) for r in recs}
    require(len(per) == 9, "nine training local recordings expected")
    return per


def convergence(net):
    from semkine import eval_track as ET
    from tools.s45loc.analyze import FK
    path = (S45TR / "readouts" / net if net in ("s45", "t1d", "t1b") else OUT / "readouts" / net) / "tl_f1/states.npz"
    fk = FK()
    with np.load(path, allow_pickle=False) as z:
        recs = sorted({k.split("/")[0] for k in z.files})

        def ra(q, k):
            a, b = fk.joints(q, z[f"{q}/mean_iter{k}"]), fk.joints(q, z[f"{q}/target"])
            return float(np.linalg.norm(ET.root_align(a) - ET.root_align(b), axis=-1).mean())
        r1, r20 = float(np.median([ra(q, 1) for q in recs])), float(np.median([ra(q, 20) for q in recs]))
    return {"ra_feed1_mm": r1, "ra_feed20_mm": r20, "C": r20 / r1}


def classify(arm, ref):
    med = {s: float(np.median(list(insample(arm, s).values()))) for s in STEPS}
    R = {s: (ref["s45"][s] - med[s]) / (ref["s45"][s] - ref["t1d"][s]) for s in STEPS}
    conv = convergence(arm)
    z, zs = zgz(arm), ref["zgz"]
    zout = {}
    for dom, th in (("local", TH["zgz_local_mm"]), ("global", TH["zgz_global_mm"])):
        gain = {s: zs["e1"][s][dom] - z["e1"][s][dom] for s in STEPS}
        g2, g3 = zs["e2"][dom] - z["e2"][dom], zs["e3"][dom] - z["e3"][dom]
        zout[dom] = {"e1_gain_by_step": {str(s): v for s, v in gain.items()}, "e2_gain": g2, "e3_gain": g3,
                     "support": gain[6000] >= th and g2 > 0 and g3 > 0 and sum(gain[s] >= th for s in STEPS) >= TH["late_needed"]}
    support = any(v["support"] for v in zout.values())
    if min(R.values()) >= TH["R_min"] and conv["C"] <= TH["C_conv"]:
        cls = "DEPLOYABLE" if support else "IN_SAMPLE"
    elif max(R.values()) < TH["R_null"]:
        cls = "NULL"
    else:
        cls = "WEAK"
    return {"insample_median_by_step": {str(s): v for s, v in med.items()}, "R": {str(s): v for s, v in R.items()},
            "convergence": conv, "zgz": zout, "zgz_three": {"e1": {str(s): v for s, v in z["e1"].items()},
                                                            "e2": z["e2"], "e3": z["e3"]},
            "class": cls, "next": NEXT[cls]}


def main():
    verify_protocol()
    target = OUT / "summary.json"
    require(not target.exists(), "the decision is read once")
    ref = {"s45": {s: float(np.median(list(insample("s45", s).values()))) for s in STEPS},
           "t1d": {s: float(np.median(list(insample("t1d", s).values()))) for s in STEPS},
           "zgz": zgz("s45"), "s45_convergence": convergence("s45"), "t1d_convergence": convergence("t1d")}
    gate_path = OUT / "t0p/gate.json"
    gate = read(gate_path) if gate_path.is_file() else None
    arms = {"t1b": classify("t1b", ref)}
    if gate is not None and gate.get("status") == "PASS":
        arms["t1p"] = classify("t1p", ref)
    else:
        arms["t1p"] = {"class": "NOT_TRAINED", "next": NEXT["NOT_TRAINED"]}
    summary = {"computed_utc": now(), "thresholds": TH, "t0p_gate": gate,
               "references": {k: ({str(s): v for s, v in ref[k].items()} if k in ("s45", "t1d") else ref[k])
                              for k in ("s45", "t1d", "s45_convergence", "t1d_convergence")},
               "arms": arms, "single_seed_descriptive": True}
    save(target, summary, exclusive=True)
    lines = ["# S45-RT 结果（自动生成；单 seed 3407，主行固定 step 6000）", "",
             "| 臂 | R@5000 / 5500 / 6000 | C（RA 喂 1 → 喂 20） | zgz local / global 方向支持 | 类别 |", "|---|---|---|---|---|"]
    for arm, v in arms.items():
        if v["class"] == "NOT_TRAINED":
            lines.append(f"| {arm} | — | — | — | NOT_TRAINED |")
            continue
        R, c = v["R"], v["convergence"]
        lines.append(f"| {arm} | {R['5000']:+.2f} / {R['5500']:+.2f} / {R['6000']:+.2f} | {c['C']:.2f}"
                     f"（{c['ra_feed1_mm']:.1f} → {c['ra_feed20_mm']:.1f}） | {v['zgz']['local']['support']} / "
                     f"{v['zgz']['global']['support']} | {v['class']} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
