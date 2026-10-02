#!/usr/bin/env python3
"""DT round diagnostic D0f (docs/DT_RENDER_TRACK_PREREG.md): depth accuracy of a trained checkpoint on its OWN
training sequences, by sequence depth.

zgz_local lies at 699-742 mm, outside the training depths (<= 710 mm). If the depth error of the recorded models
also grows toward the top of the training range *in sample*, the failure is a weak depth cue plus a pull toward the
mean; if it stays small in sample and only appears at the unseen depth, it is extrapolation. Per unique training
sequence (the `_v2`-`_v4` duplicates are skipped) this reports the ground-truth mean depth, the teacher-forced and
closed-loop translation error and depth bias, and the cross-sequence slope of depth bias on depth. In-sample, so it is
a diagnostic of a recorded run, never an evaluation.

    python tools/dt/depth_insample.py --run-dir outputs/semkine/rt_cnntrack_s3407
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "tracking"), str(REPO / "tools" / "dt")]
import evalx as EX                                                    # noqa: E402
from config import load_config                                        # noqa: E402
from mano_layer import ManoLayer                                      # noqa: E402
from model import MNISTModel                                          # noqa: E402
from semkine.dataset import sequences_for_split                       # noqa: E402
from transl_structure import structure                                # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--out-dir", default=str(REPO / "outputs/dt/ablate"))
    cli = ap.parse_args()
    run = Path(cli.run_dir)
    cfg = load_config(next(iter(sorted(run.glob("*.yaml")))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, step, _ = EX.find_ckpt(run, cli.ckpt)
    model = MNISTModel.load_from_checkpoint(str(ckpt), cfg=cfg, map_location=device).to(device).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    seqs = [(s, d) for s, d in sequences_for_split(root, "train", Path(cfg["DATA"]["SPLITS_MANIFEST"]))
            if not any(s.endswith(f"_v{k}") for k in (2, 3, 4))]
    a = argparse.Namespace(controls=False, tf=True, perturb=False, window_mode="fixed", window_ms=EX.STEP,
                           min_events=0, max_window_ms=300)
    results = {}
    for mode in ("tf", "model"):
        rng = np.random.default_rng(0)          # evalx.evaluate: one stream per mode, sequences in order
        for s_, d_ in seqs:
            results[(mode, s_)] = EX.run_sequence(model, cfg, root, d_, s_, device, rng, mode, "fixed", EX.STEP, 0, 300)
        print(f"  {run.name} {mode} done", flush=True)
    rows = []
    for s_, _ in seqs:
        row = {"seq": s_}
        for mode in ("tf", "model"):
            r = results[(mode, s_)]
            st = structure(r["pred"], r["gt"], r["run"])
            row[mode] = {k: st[k] for k in ("n", "transl_mm", "bias_z_mm", "gt_depth_mm", "z_rel_bias", "depth_gain")}
        rows.append(row)
    rows.sort(key=lambda r: r["tf"]["gt_depth_mm"])
    gd = np.array([r["tf"]["gt_depth_mm"] for r in rows])
    out = {"run": run.name, "ckpt": str(ckpt), "step": step, "rows": rows}
    for mode in ("tf", "model"):
        bz = np.array([r[mode]["bias_z_mm"] for r in rows])
        slope, icpt = np.polyfit(gd, bz, 1)
        out[f"{mode}_bias_vs_depth_slope"] = float(slope)
        out[f"{mode}_bias_vs_depth_intercept_mm"] = float(icpt)
        out[f"{mode}_bias_at_720_mm"] = float(slope * 720 + icpt)
    lines = [f"# In-sample depth accuracy, {run.name} (step {step}), unique training sequences by depth", "",
             "| sequence | GT depth mm | TF transl mm | TF z bias mm | closed-loop transl mm | closed-loop z bias mm |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['seq']} | {r['tf']['gt_depth_mm']:.0f} | {r['tf']['transl_mm']:.1f} | {r['tf']['bias_z_mm']:+.1f} | "
                     f"{r['model']['transl_mm']:.1f} | {r['model']['bias_z_mm']:+.1f} |")
    lines += ["", f"Linear fit of the z bias on the sequence depth: TF slope {out['tf_bias_vs_depth_slope']:+.3f} "
              f"(extrapolated to 720 mm: {out['tf_bias_at_720_mm']:+.1f} mm); closed loop slope {out['model_bias_vs_depth_slope']:+.3f} "
              f"(at 720 mm: {out['model_bias_at_720_mm']:+.1f} mm)."]
    od = Path(cli.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / f"{run.name}_depth_insample.json").write_text(json.dumps(out, indent=1))
    (od / f"{run.name}_depth_insample.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
