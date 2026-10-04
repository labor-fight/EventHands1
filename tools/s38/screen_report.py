#!/usr/bin/env python3
"""S38 screening / full-budget report (docs/S38_ROOT_TRACKING_PREREG.md section 5).

Every arm is compared with S37 at the same seeds, last step, GPU evaluation
(`tools/tracking/evalx.py eval --ckpt last --controls --tf --perturb`). The registered gates:
  effective   paired RA mean <= -1.1 mm with every seed negative, and paired closed-loop root error mean < 0
  guards      MPJPE-local (zgz_local RA) seed mean not worse than S37's by >= 1.1 mm;
              absolute translation error seed mean not worse than S37's by more than 10 %
Drift and robustness figures are reported beside them; the grid median (`tools/select_checkpoint.py`)
is a robustness figure and never selects a checkpoint.

    python tools/s38/screen_report.py --budget 2k      # writes outputs/s38/reports/screen_2k.{md,json}
    python tools/s38/screen_report.py --budget 6k
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "outputs" / "semkine"
#: the S37 baseline per budget: run prefix and the evaluation file (the 2k runs were first evaluated on CPU;
#: `_gpu` is their GPU re-evaluation, the device every S38 arm is evaluated on)
BASE = {"2k": ("rt_s37_2k", "evalx_val_core_last_tf_pert_gpu.json"),
        "6k": ("rt_s37", "evalx_val_core_last_tf_pert.json")}
ARMS = {"2k": ["s38_s37abs_2k", "s38_spdelta_2k", "s38_spabs_2k"],
        "6k": ["s38_s37abs", "s38_spdelta", "s38_spabs"]}
LABEL = {"s38_s37abs": "S37 graph + M1 (root measurement only)", "s38_spdelta": "sparse pyramid + S37 delta root (encoder only)",
         "s38_spabs": "sparse pyramid + M1 (candidate)", "s38_spe7": "sparse pyramid, absolute 51D (encoder control)",
         "rt_e7": "S37 graph, absolute 51D (E7)", "rt_cnn": "ResNet18 / LNES, absolute 51D (dense reference)",
         "rt_s37": "S37 routed readout (baseline)"}
SEEDS = ("3407", "3408", "3409")


def metrics(js: dict) -> dict:
    m = js["model"]
    g, l = m["zgz_global"], m["zgz_local"]
    out = {"RA": m["overall"]["mpjpe_ra_mm"], "RA_global": g["mpjpe_ra_mm"][0], "RA_local": l["mpjpe_ra_mm"][0],
           "MPVPE_global": g["mpvpe_ra_mm"][0], "MPVPE_local": l["mpvpe_ra_mm"][0],
           "rot": m["overall"]["root_rot_deg"], "rot_global": g["root_rot_deg"][0], "rot_local": l["root_rot_deg"][0],
           "transl_mm": m["overall"]["transl_mm"], "abs_mpjpe": m["overall"]["mpjpe_abs_mm"],
           "slope_global": float(np.mean(g["motion"]["root_err_slope_deg_per_10s"])),
           "rootspeed_global": g["motion"]["root_speed_ratio"], "rootspeed_local": l["motion"]["root_speed_ratio"],
           "fingerspeed": float(np.mean([g["motion"]["finger_speed_ratio"], l["motion"]["finger_speed_ratio"]])),
           "fail_episodes": g["failure"]["episodes"] + l["failure"]["episodes"],
           "bad_frac": float(np.mean([g["failure"]["bad_step_frac"], l["failure"]["bad_step_frac"]]))}
    if "tf" in js:
        out["tf_RA"], out["tf_rot"] = js["tf"]["overall"]["mpjpe_ra_mm"], js["tf"]["overall"]["root_rot_deg"]
        out["amp_RA"], out["amp_rot"] = js["amplification"]["mpjpe_ra_mm"], js["amplification"]["root_rot_deg"]
    if "hold" in js:
        out["hold_rot"] = js["hold"]["overall"]["root_rot_deg"]
    if "perturb" in js:
        out["ret10"] = js["perturb"]["10"]["retention_k1_k2_k5_k10_k20"]
        out["ret20"] = js["perturb"]["20"]["retention_k1_k2_k5_k10_k20"]
    return out


def load(prefix: str, seed: str, fname: str):
    run = RUNS / f"{prefix}_s{seed}"
    f = run / fname
    if not f.exists():
        return None
    out = metrics(json.loads(f.read_text()))
    out["step"] = json.loads(f.read_text())["step"]
    sel = sorted(glob.glob(str(run / "selection_val_core_step50*.json")))
    if sel:
        out["grid_median"] = json.loads(Path(sel[0]).read_text())["grid_median"]
    return out


def verdict(arm: dict, base: dict, seeds) -> dict:
    d_ra = [arm[s]["RA"] - base[s]["RA"] for s in seeds]
    d_rot = [arm[s]["rot"] - base[s]["rot"] for s in seeds]
    mean = lambda k, src: float(np.mean([src[s][k] for s in seeds]))  # noqa: E731
    out = {"dRA": d_ra, "dRA_mean": float(np.mean(d_ra)), "drot": d_rot, "drot_mean": float(np.mean(d_rot)),
           "dRA_local_mean": mean("RA_local", arm) - mean("RA_local", base),
           "transl_ratio": mean("transl_mm", arm) / mean("transl_mm", base)}
    out["effective"] = bool(out["dRA_mean"] <= -1.1 and all(v < 0 for v in d_ra) and out["drot_mean"] < 0)
    out["tie"] = bool(abs(out["dRA_mean"]) < 1.1)
    out["guard_local"] = bool(out["dRA_local_mean"] < 1.1)
    out["guard_transl"] = bool(out["transl_ratio"] <= 1.10)
    out["pass"] = out["effective"] and out["guard_local"] and out["guard_transl"]
    return out


def fmt(v, n=2):
    return "-" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", choices=("2k", "6k"), required=True)
    ap.add_argument("--arms", nargs="*", default=None)
    a = ap.parse_args()
    bprefix, bfile = BASE[a.budget]
    base = {s: load(bprefix, s, bfile) for s in SEEDS}
    base = {s: v for s, v in base.items() if v is not None}
    rows, report = [], {"budget": a.budget, "baseline": bprefix, "arms": {}}
    lines = [f"# S38 {a.budget} report: last step, GPU, paired with {bprefix} at the same seeds", ""]
    cols = ["RA", "RA_global", "RA_local", "rot", "rot_global", "rot_local", "transl_mm", "abs_mpjpe", "tf_rot",
            "amp_rot", "slope_global", "rootspeed_global", "fingerspeed", "fail_episodes", "grid_median"]
    lines.append("| arm | seed | step | " + " | ".join(cols) + " | ret10 k1/k5/k20 |")
    lines.append("|" + "---|" * (len(cols) + 4))

    def add(name, data):
        for s, v in data.items():
            r10 = v.get("ret10")
            lines.append(f"| {name} | {s} | {v['step']} | " + " | ".join(fmt(v.get(c)) for c in cols)
                         + f" | {'/'.join(f'{r10[i]:.2f}' for i in (0, 2, 4)) if r10 else '-'} |")

    add(bprefix, base)
    for arm in (a.arms or ARMS[a.budget]):
        data = {s: load(arm, s, "evalx_val_core_last_tf_pert.json") for s in SEEDS}
        data = {s: v for s, v in data.items() if v is not None}
        add(arm, data)
        seeds = [s for s in data if s in base]
        rep = {"label": LABEL.get(arm.replace("_2k", ""), arm), "seeds": seeds, "per_seed": data}
        if seeds:
            rep["verdict"] = verdict(data, base, seeds)
        report["arms"][arm] = rep
    lines += ["", "## Registered gates (paired with the baseline at the same seeds)", "",
              "| arm | seeds | dRA per seed | dRA mean | droot mean (deg) | dRA-local mean | translation ratio | effective | guards | pass |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, rep in report["arms"].items():
        v = rep.get("verdict")
        if not v:
            continue
        lines.append(f"| {arm} | {','.join(rep['seeds'])} | {', '.join(f'{x:+.2f}' for x in v['dRA'])} | {v['dRA_mean']:+.2f} | "
                     f"{v['drot_mean']:+.2f} | {v['dRA_local_mean']:+.2f} | {v['transl_ratio']:.3f} | {v['effective']} | "
                     f"{v['guard_local'] and v['guard_transl']} | **{v['pass']}** |")
    out = REPO / "outputs" / "s38" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"screen_{a.budget}.md").write_text("\n".join(lines) + "\n")
    (out / f"screen_{a.budget}.json").write_text(json.dumps(report, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
