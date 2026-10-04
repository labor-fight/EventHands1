#!/usr/bin/env python3
"""DT2 diagnostic: inference-time scaling of the two root-rotation update terms of a trained dense tracker.

root delta = alpha * F_root (event / render CNN, rows 3:6 of the trunk's fc) + beta * G_root (event-blind prev_mlp, rows 3:6
of its last layer). alpha = beta = 1 is the trained model. The RA decomposition showed the root error is a constant-like
bias shared by the three seeds and present under teacher forcing; prev_mlp pulls the state toward the training mean. This
scan shows how the closed-loop RA, the root error and the teacher-forced RA move with each term. Nothing is retrained and
nothing here is adopted: a diagnostic of recorded runs on zgz (development = test subject).

    python tools/dt/ab_scan.py --run-dir outputs/semkine/dt_dz_l3_s3407 --alphas 0.8 1 1.2 --betas 0.5 0.75 1 1.25 1.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "tracking")]
import evalx as EX                                                    # noqa: E402
from config import load_config                                        # noqa: E402
from mano_layer import ManoLayer                                      # noqa: E402
from model import MNISTModel                                          # noqa: E402
from semkine.dataset import sequences_for_split                       # noqa: E402

ROOT = [3, 4, 5]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--alphas", nargs="+", type=float, default=[1.0])
    ap.add_argument("--betas", nargs="+", type=float, default=[1.0])
    ap.add_argument("--tf", action="store_true")
    ap.add_argument("--out-dir", default=str(REPO / "outputs/dt2/abscan"))
    cli = ap.parse_args()
    run = Path(cli.run_dir)
    cfg = load_config(next(iter(sorted(run.glob("*.yaml")))))
    device = torch.device("cuda")
    ckpt, step, _ = EX.find_ckpt(run, cli.ckpt)
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    seqs = sequences_for_split(root, "val_core", Path(cfg["DATA"]["SPLITS_MANIFEST"]))
    a = argparse.Namespace(controls=False, tf=cli.tf, perturb=False, window_mode="fixed", window_ms=EX.STEP,
                           min_events=0, max_window_ms=300)
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for al in cli.alphas:
        for be in cli.betas:
            model = MNISTModel.load_from_checkpoint(str(ckpt), cfg=cfg, map_location=device).to(device).eval()
            with torch.no_grad():
                fc = model.rn.fc
                fc.weight[ROOT, :] *= al
                fc.bias[ROOT] *= al
                model.prev_mlp[2].weight[ROOT, :] *= be
                model.prev_mlp[2].bias[ROOT] *= be
            summary, arrays = EX.evaluate(model, cfg, mano, root, seqs, device, a, label=f"{run.name} a{al} b{be}")
            tag = f"{run.name}_a{al:g}_b{be:g}"
            out = {"run": run.name, "ckpt": str(ckpt), "step": step, "alpha": al, "beta": be, **summary}
            np.savez_compressed(out_dir / f"{tag}.npz", **arrays)
            (out_dir / f"{tag}.json").write_text(json.dumps(out, indent=1))
            m = summary["model"]
            tf = summary.get("tf", {}).get("overall", {}).get("mpjpe_ra_mm", float("nan"))
            print(f"{tag}: RA {m['overall']['mpjpe_ra_mm']:.3f} (g {m['zgz_global']['mpjpe_ra_mm'][0]:.2f} "
                  f"l {m['zgz_local']['mpjpe_ra_mm'][0]:.2f})  rot {m['overall']['root_rot_deg']:.2f}  TF {tf:.3f}  "
                  f"acc_err_ra {m['jitter'].get('acc_err_ra_mm', float('nan')):.2f}  "
                  f"root speed g/l {m['zgz_global']['motion']['root_speed_ratio']:.2f}/{m['zgz_local']['motion']['root_speed_ratio']:.2f}",
                  flush=True)
            del model


if __name__ == "__main__":
    main()
