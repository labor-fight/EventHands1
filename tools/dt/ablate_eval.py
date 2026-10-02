#!/usr/bin/env python3
"""DT round diagnostic D0e (docs/DT_RENDER_TRACK_PREREG.md): inference-time ablation of the event-blind `prev_mlp`.

Hypothesis (registered with the translation-structure diagnostic): `prev_mlp` (51 -> 64 -> 51, reads the raw previous
state, zero-initialised last layer) learned a regression-to-the-mean pull on the depth, because the training prev noise
(up to 50 mm) makes the previous depth unreliable. At a depth outside the training range (zgz_local, 699-742 mm against a
training maximum of 710 mm) that pull is a large systematic translation step. Zeroing the pull's output rows in the
trained checkpoint, with nothing else changed, shows how much of the closed-loop and teacher-forced translation error it
carries. Nothing is retrained and nothing is selected: a diagnostic of a recorded run.

    python tools/dt/ablate_eval.py --run-dir outputs/semkine/rt_cnntrack_s3407 --ablations none pm_all pm_transl pm_z pm_root
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

#: ablation -> output rows of prev_mlp's last layer that are zeroed (weights and bias)
ROWS = {"none": [], "pm_all": list(range(51)), "pm_transl": [0, 1, 2], "pm_z": [2], "pm_root": [3, 4, 5],
        "pm_transl_root": [0, 1, 2, 3, 4, 5], "pm_fingers": list(range(6, 51))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--ablations", nargs="+", default=["none", "pm_all", "pm_transl", "pm_z", "pm_root"])
    ap.add_argument("--out-dir", default=str(REPO / "outputs/dt/ablate"))
    ap.add_argument("--tf", action="store_true")
    cli = ap.parse_args()
    run = Path(cli.run_dir)
    cfg = load_config(next(iter(sorted(run.glob("*.yaml")))) if list(run.glob("*.yaml")) else
                      json.loads((run / "training_metadata.json").read_text())["config_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, step, _ = EX.find_ckpt(run, cli.ckpt)
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    seqs = sequences_for_split(root, "val_core", Path(cfg["DATA"]["SPLITS_MANIFEST"]))
    a = argparse.Namespace(controls=False, tf=cli.tf, perturb=False, window_mode="fixed", window_ms=EX.STEP,
                           min_events=0, max_window_ms=300)
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in cli.ablations:
        model = MNISTModel.load_from_checkpoint(str(ckpt), cfg=cfg, map_location=device).to(device).eval()
        rows = ROWS[name]
        with torch.no_grad():
            if rows:
                model.prev_mlp[2].weight[rows, :] = 0.0
                model.prev_mlp[2].bias[rows] = 0.0
        summary, arrays = EX.evaluate(model, cfg, mano, root, seqs, device, a, label=f"{run.name} {name}")
        out = {"run": run.name, "ckpt": str(ckpt), "step": step, "ablation": name, "zeroed_rows": rows, **summary}
        tag = f"{run.name}_{name}"
        np.savez_compressed(out_dir / f"{tag}.npz", **arrays)
        (out_dir / f"{tag}.json").write_text(json.dumps(out, indent=1))
        m = summary["model"]
        print(f"{tag}: RA {m['overall']['mpjpe_ra_mm']:.3f}  abs {m['overall']['mpjpe_abs_mm']:.2f}  "
              f"transl global/local {m['zgz_global']['transl_mm'][0]:.1f}/{m['zgz_local']['transl_mm'][0]:.1f}  "
              f"abs global/local {m['zgz_global']['mpjpe_abs_mm'][0]:.1f}/{m['zgz_local']['mpjpe_abs_mm'][0]:.1f}  "
              f"rot {m['overall']['root_rot_deg']:.2f}", flush=True)
        del model


if __name__ == "__main__":
    main()
