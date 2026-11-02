#!/usr/bin/env python3
"""S45-AT decomposition (docs/EXPERIMENT_RECORDS.md#s45at-prereg-20261031); CPU only, run once after every registered
reading. Writes outputs/s45at/20261031/summary.json and report.md. Single seed 3407, descriptive.

Effects (positive = the second network is better, mm of RA):
  depth_s45   S45 - dz            depth_t1d   T1d - t1ddz         interaction  depth_t1d - depth_s45
  window      S45 - w200          noise       |S45 - s45samp| (training-path noise of the paired design)
  heldout     (t1dnoylf - T1d) on ylf_local minus the median of (t1dnoylf - T1d) over the other eight recordings
Each on zgz local / global E1 (steps 5000 / 5500 / 6000) and the training local median E1 (same steps); E3 at 6000
is descriptive. Floors: zgz local max(5, noise), zgz global max(8, noise), in-sample max(2.5, noise), the noise being
the largest |S45 - s45samp| over the three steps of the same readout. Class per effect and readout: detected = same
sign and |effect| >= floor at all three steps; suggestive = |effect at 6000| >= floor and >= 2 of 3 steps with its
sign; otherwise not detected.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.s45at.common import OUT, S45CT, S45TR, S45_EVAL, now, read, require, save, verify_protocol  # noqa: E402

STEPS = (5000, 5500, 6000)
FLOOR = {"local": 5.0, "global": 8.0, "insample": 2.5}
HELD = "ylf_local"


def trace(directory):
    m = read(Path(directory) / "report.json")["metrics"]
    return {d: m[d]["mpjpe_ra_mm"] for d in ("local", "global")}


def zgz(net):
    if net == "s45":
        e1 = {s: trace(S45_EVAL / f"e1_step{s}_repeat1_none") for s in STEPS}
        return {"e1": e1, "e3": trace(S45_EVAL / "e3_step6000_repeat1_none")}
    ev = (S45TR if net == "t1d" else OUT) / "evaluations" / net
    return {"e1": {s: trace(ev / f"e1_step{s}") for s in STEPS}, "e3": trace(ev / "e3_step6000")}


def insample(net, step):
    if net in ("s45", "t1d"):
        base = (S45TR / "readouts" / net) if step == 6000 else (S45CT / "readouts" / f"{net}_{step}")
    else:
        base = OUT / "readouts" / (net if step == 6000 else f"{net}_{step}")
    with np.load(base / "tl_e1" / "states.npz", allow_pickle=False) as z:
        recs = sorted({k.split("/")[0] for k in z.files})
        per = {r: float(z[f"{r}/mpjpe_ra_mm"].mean()) for r in recs}
    require(len(per) == 9 and HELD in per, "nine training local recordings expected")
    return per


def series(net):
    z = zgz(net)
    ins = {s: insample(net, s) for s in STEPS}
    return {"local": {s: z["e1"][s]["local"] for s in STEPS}, "global": {s: z["e1"][s]["global"] for s in STEPS},
            "insample": {s: float(np.median(list(ins[s].values()))) for s in STEPS}, "e3": z["e3"], "per": ins}


def classify(effect, floor):
    e6 = effect[6000]
    same = [np.sign(effect[s]) == np.sign(e6) and e6 != 0 for s in STEPS]
    if all(same) and all(abs(effect[s]) >= floor for s in STEPS):
        return "detected"
    if abs(e6) >= floor and sum(same) >= 2:
        return "suggestive"
    return "not detected"


def effect(a, b, floors, readouts=("local", "global", "insample")):
    out = {}
    for r in readouts:
        eff = {s: a[r][s] - b[r][s] for s in STEPS}
        out[r] = {"by_step": {str(s): v for s, v in eff.items()}, "floor": floors[r], "class": classify(eff, floors[r])}
    out["e3"] = {d: a["e3"][d] - b["e3"][d] for d in ("local", "global")}
    return out


def main():
    verify_protocol()
    target = OUT / "summary.json"
    require(not target.exists(), "the decomposition is read once")
    S = {n: series(n) for n in ("s45", "t1d", "dz", "t1ddz", "t1dnoylf", "s45samp", "w200")}
    noise = {r: max(abs(S["s45"][r][s] - S["s45samp"][r][s]) for s in STEPS) for r in ("local", "global", "insample")}
    floors = {r: max(FLOOR[r], noise[r]) for r in FLOOR}
    eff = {"depth_s45": effect(S["s45"], S["dz"], floors), "depth_t1d": effect(S["t1d"], S["t1ddz"], floors),
           "window": effect(S["s45"], S["w200"], floors)}
    inter = {}
    for r in ("local", "global", "insample"):
        e = {s: (S["t1d"][r][s] - S["t1ddz"][r][s]) - (S["s45"][r][s] - S["dz"][r][s]) for s in STEPS}
        inter[r] = {"by_step": {str(s): v for s, v in e.items()}, "floor": floors[r], "class": classify(e, floors[r])}
    eff["interaction"] = inter
    held = {}
    for s in STEPS:
        a, b = S["t1dnoylf"]["per"][s], S["t1d"]["per"][s]
        control = float(np.median([a[r] - b[r] for r in a if r != HELD]))
        held[s] = (a[HELD] - b[HELD]) - control
    eff["heldout"] = {"insample_penalty_by_step": {str(s): v for s, v in held.items()}, "floor": floors["insample"],
                      "class": classify({s: v for s, v in held.items()}, floors["insample"]),
                      "ylf_local": {str(s): {"t1dnoylf": S["t1dnoylf"]["per"][s][HELD], "t1d": S["t1d"]["per"][s][HELD]}
                                    for s in STEPS}}
    eff["heldout_zgz"] = effect(S["t1d"], S["t1dnoylf"], floors, ("local", "global"))
    summary = {"computed_utc": now(), "noise": noise, "floors": floors, "effects": eff,
               "series": {n: {k: ({str(s): v for s, v in val.items()} if k in ("local", "global", "insample") else val)
                              for k, val in d.items() if k != "per"} for n, d in S.items()},
               "decomposition_zgz_local_6000": {
                   "s45_total": S["s45"]["local"][6000], "routing_oracle": S["s45"]["local"][6000] - S["t1d"]["local"][6000],
                   "depth_under_oracle_routing": eff["depth_t1d"]["local"]["by_step"]["6000"],
                   "depth_under_s45_association": eff["depth_s45"]["local"]["by_step"]["6000"],
                   "window_50_to_200": eff["window"]["local"]["by_step"]["6000"],
                   "noise_local": noise["local"]},
               "single_seed_descriptive": True}
    save(target, summary, exclusive=True)
    lines = ["# S45-AT 归因结果（自动生成；单 seed 3407）", "",
             f"噪声（S45 − s45samp 三点最大绝对差）：zgz local {noise['local']:.2f} / global {noise['global']:.2f} / "
             f"样本内 {noise['insample']:.2f} mm；门槛 local {floors['local']:.2f} / global {floors['global']:.2f} / "
             f"样本内 {floors['insample']:.2f} mm", "",
             "| 效应 | zgz local（5000 / 5500 / 6000） | zgz global | 样本内 | 类别（local / global / 样本内） |", "|---|---|---|---|---|"]
    for name in ("depth_s45", "depth_t1d", "interaction", "window"):
        e = eff[name]
        f = lambda r: " / ".join(f"{e[r]['by_step'][str(s)]:+.2f}" for s in STEPS)  # noqa: E731
        lines.append(f"| {name} | {f('local')} | {f('global')} | {f('insample')} | "
                     f"{e['local']['class']} / {e['global']['class']} / {e['insample']['class']} |")
    h = eff["heldout"]
    lines.append(f"| heldout（ylf 留出罚分） | — | — | " + " / ".join(f"{h['insample_penalty_by_step'][str(s)]:+.2f}" for s in STEPS)
                 + f" | {h['class']} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
