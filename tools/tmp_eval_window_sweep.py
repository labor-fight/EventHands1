#!/usr/bin/env python3
"""The evidence window as a free knob, under the frozen recursive protocol.

§10.3 measured that the window, not the update rate, is what moves the recursive error, and the
`bias(W)` / `G(W)` sweep in §10.5 found the two move in *opposite* directions: a longer window
lowers retention but raises the single-step bias, because LNES stores extra evidence only as stale
pixels at full weight. That predicts an interior optimum in W, and the probe put it near 20 ms --
below the 50 ms every recorded number was measured at.

The probe runs a truncated chain on a subset of instants, so it can rank arms but cannot be quoted
against the frozen selection number. This re-runs the *exact* selection protocol
(`tools/select_checkpoint.py`: same split, same pooling by frame count, same init noise, same seed)
on one checkpoint across W, so W = 50 has to reproduce the recorded figure before any other column
is believed.

Temporary: `/tmp`-class diagnostic per the experiment contract; delete after the verdict lands.
"""
from __future__ import annotations

import argparse
import json
import sys
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
from semkine.dataset import sequences_for_split, splits_manifest   # noqa: E402

KEY = "mpjpe_ra_mm"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val_core")
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--windows", default="10,20,30,50,100")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference", type=float, default=None,
                    help="recorded figure the W == step-ms column must reproduce")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = load_config(a.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel.load_from_checkpoint(a.ckpt, cfg=cfg, map_location=device).to(device).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    seqs = sequences_for_split(root, a.split, splits_manifest(cfg))
    wins = [int(x) for x in a.windows.split(",")]

    rows = {}
    for w in wins:
        rng = np.random.default_rng(a.seed)          # same seed per column, as selection does
        res = [ET.track_sequence(model, mano, cfg, root, d, s, a.step_ms, device, rng,
                                 1.0, 1.0, None, window_ms=w)
               for s, d in seqs]
        res = [r for r in res if r]
        n = sum(r["n_frames"] for r in res)
        row = {"window_ms": w, "n_frames": n,
               **{k: float(sum(r[k] * r["n_frames"] for r in res) / max(n, 1))
                  for k in ("mpjpe_ra_mm", "mpjpe_abs_mm", "mpvpe_ra_mm")},
               "per_seq": {r["seq"]: r[KEY] for r in res}}
        rows[w] = row
        print(f"  W={w:<5d} RA={row[KEY]:8.4f}  abs={row['mpjpe_abs_mm']:8.3f}  "
              f"n={n}", flush=True)

    base = rows.get(a.step_ms)
    if a.reference is not None and base is not None:
        d = abs(base[KEY] - a.reference)
        print(f"\nregression: W={a.step_ms} gives {base[KEY]:.4f} vs recorded {a.reference:.4f} "
              f"(|Δ|={d:.4f}) -> {'OK' if d < 5e-3 else 'MISMATCH'}")
        if d >= 5e-3:
            raise SystemExit("W == step-ms column does not reproduce the recorded figure; "
                             "the decoupling changed the default path")

    best = min(rows.values(), key=lambda r: r[KEY])
    print(f"\nbest W={best['window_ms']}  RA={best[KEY]:.4f}"
          + (f"  ({best[KEY] - base[KEY]:+.4f} vs W={a.step_ms})" if base else ""))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"ckpt": a.ckpt, "split": a.split, "step_ms": a.step_ms, "seed": a.seed,
         "windows": wins, "rows": list(rows.values()), "best_window_ms": best["window_ms"]},
        indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
