#!/usr/bin/env python3
"""DT round diagnostic D0d (docs/DT_RENDER_TRACK_PREREG.md section 1): structure of the translation error.

The tracker's absolute translation is what separates its absolute MPJPE from its root-aligned MPJPE. From the
per-step arrays every closed-loop evaluation saves (`evalx_val_core_last_tf_pert.npz`: keys
`<mode>|<seq>|{pred,gt,run}`, modes `model` = closed loop, `tf` = teacher forced, `hold` = initial pose held)
this tool reports, per run, mode and sequence:

  bias_xyz      mean of (pred - gt) translation per axis, mm
  sd_xyz        standard deviation of the same
  z_rel_bias    depth bias divided by the mean ground-truth depth
  depth_gain    OLS slope of predicted depth on ground-truth depth (1 = depth measured; < 1 = the prediction
                is pulled toward a mean depth, the signature of a depth cue that is not read)
  corr_ez_gz    correlation of the depth error with the ground-truth depth
  run_offset_sd / within_run_sd   spread of the per-segment mean error vs spread inside segments
                (a constant per-segment offset is a calibration-like error, within-run spread is noise / drift)
  transl_mm     mean 3D translation error

    python tools/dt/transl_structure.py --runs rt_cnntrack_s3407 rt_cnntrack_s3408 --out-prefix outputs/dt/reports/transl_structure
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "outputs" / "semkine"
SEQS = ("zgz_global", "zgz_local")


def structure(pred: np.ndarray, gt: np.ndarray, run: np.ndarray) -> dict:
    e = (pred[:, :3] - gt[:, :3]) * 1000.0
    gz = gt[:, 2] * 1000.0
    pz = pred[:, 2] * 1000.0
    runs = np.unique(run)
    run_mean = np.array([e[run == r].mean(0) for r in runs])
    within = np.mean([e[run == r].std(0) for r in runs], axis=0)
    slope = float(np.polyfit(gz, pz, 1)[0]) if gz.std() > 1e-9 else float("nan")
    corr = float(np.corrcoef(e[:, 2], gz)[0, 1]) if gz.std() > 1e-9 and e[:, 2].std() > 1e-9 else float("nan")
    return {"n": int(len(e)), "transl_mm": float(np.linalg.norm(e, axis=1).mean()),
            "bias_x_mm": float(e[:, 0].mean()), "bias_y_mm": float(e[:, 1].mean()), "bias_z_mm": float(e[:, 2].mean()),
            "sd_x_mm": float(e[:, 0].std()), "sd_y_mm": float(e[:, 1].std()), "sd_z_mm": float(e[:, 2].std()),
            "gt_depth_mm": float(gz.mean()), "z_rel_bias": float(e[:, 2].mean() / max(gz.mean(), 1.0)),
            "depth_gain": slope, "corr_ez_gz": corr, "n_runs": int(len(runs)),
            "run_offset_sd_z_mm": float(run_mean[:, 2].std()) if len(runs) > 1 else float("nan"),
            "within_run_sd_z_mm": float(within[2])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run names (globs allowed) under outputs/semkine")
    ap.add_argument("--npz", default="evalx_val_core_last_tf_pert.npz")
    ap.add_argument("--out-prefix", default=str(REPO / "outputs/dt/reports/transl_structure"))
    a = ap.parse_args()
    names = []
    for r in a.runs:
        names += sorted(Path(p).name for p in glob.glob(str(RUNS / r)))
    out = {}
    for name in names:
        f = RUNS / name / a.npz
        if not f.exists():
            continue
        z = np.load(f)
        out[name] = {}
        for mode in ("model", "tf", "hold"):
            for seq in SEQS:
                k = f"{mode}|{seq}|pred"
                if k not in z.files:
                    continue
                out[name][f"{mode}|{seq}"] = structure(z[k], z[f"{mode}|{seq}|gt"], z[f"{mode}|{seq}|run"])
    cols = ["transl_mm", "bias_x_mm", "bias_y_mm", "bias_z_mm", "sd_z_mm", "gt_depth_mm", "z_rel_bias", "depth_gain",
            "corr_ez_gz", "run_offset_sd_z_mm", "within_run_sd_z_mm"]
    lines = ["# Translation error structure of recorded closed-loop runs", "",
             "Source: the saved per-step arrays of each run's `evalx_val_core_last_tf_pert.npz`; mm; "
             "`depth_gain` = slope of predicted on ground-truth depth.", "",
             "| run | mode | sequence | n | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 4)]
    for name, d in out.items():
        for key, v in d.items():
            mode, seq = key.split("|")
            lines.append(f"| {name} | {mode} | {seq} | {v['n']} | " + " | ".join(f"{v[c]:.3f}" for c in cols) + " |")
    Path(a.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_prefix + ".json").write_text(json.dumps(out, indent=1))
    Path(a.out_prefix + ".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
