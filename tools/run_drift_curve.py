#!/usr/bin/env python3
"""Error against time-since-reinitialisation, and recovery from a displaced start.

Two arms with the same single-step error can behave completely differently in a loop, and the shape
of the divergence says which. An arm that can *observe* where the hand is pulls back toward the
truth and its error saturates; an arm that can only integrate increments has no such term, and its
error grows with the length of the run no matter how good each increment is.

* `drift` reports RA and absolute error bucketed by elapsed milliseconds since the run started,
  which `track_sequence` already computes. A rising curve is an uncorrected integrator.
* `recovery` reruns each arm with the initialisation noise scaled up. An arm with an absolute
  observation converges back from a displaced start; an integrator carries the displacement forever,
  so its error at the end of a run is roughly the error it began with.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from config import load_config                                    # noqa: E402
from mano_layer import ManoLayer                                  # noqa: E402
from model import MNISTModel                                      # noqa: E402
from semkine import eval_track as ET                              # noqa: E402
from semkine.dataset import sequences_for_split                   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="LABEL=CKPT:CONFIG")
    ap.add_argument("--split", default="val_core")
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--noise-scale", default="1.0,4.0", help="comma-separated init noise scales")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scales = [float(s) for s in a.noise_scale.split(",")]
    rows = {}
    for spec in a.arm:
        label, rest = spec.split("=", 1)
        ckpt, cfg_path = rest.rsplit(":", 1)
        cfg = load_config(cfg_path)
        model = MNISTModel.load_from_checkpoint(
            ckpt, cfg=cfg, map_location=device).to(device).eval()
        mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
        root = Path(cfg["DATA"]["ROOT"])
        seqs = sequences_for_split(root, a.split, None)

        out = {}
        for sc in scales:
            rng = np.random.default_rng(0)
            per = [ET.track_sequence(model, mano, cfg, root, d, s, a.step_ms, device, rng,
                                     sc, 1.0, None) for s, d in seqs]
            per = [r for r in per if r]
            n = sum(r["n_frames"] for r in per)
            acc = defaultdict(lambda: [0.0, 0.0, 0])
            for r in per:
                for ms, d in r["drift"].items():
                    a_ = acc[int(ms)]
                    a_[0] += d["mpjpe_ra_mm"] * d["n"]
                    a_[1] += d["mpjpe_abs_mm"] * d["n"]
                    a_[2] += d["n"]
            curve = {ms: {"ra_mm": v[0] / v[2], "abs_mm": v[1] / v[2], "n": v[2]}
                     for ms, v in sorted(acc.items())}
            out[f"noise_x{sc:g}"] = {
                "ra_mm": sum(r["mpjpe_ra_mm"] * r["n_frames"] for r in per) / n,
                "abs_mm": sum(r["mpjpe_abs_mm"] * r["n_frames"] for r in per) / n,
                "drift": curve,
            }
            head = list(curve.items())[:6]
            print(f"{label} noise x{sc:g}  RA={out[f'noise_x{sc:g}']['ra_mm']:.3f}  "
                  f"abs={out[f'noise_x{sc:g}']['abs_mm']:.3f}", flush=True)
            for ms, v in head:
                print(f"    t+{ms:>5d}ms  RA={v['ra_mm']:8.3f}  abs={v['abs_mm']:8.3f}  n={v['n']}")
        rows[label] = out
        del model
        torch.cuda.empty_cache()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"step_ms": a.step_ms, "arms": rows}, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
