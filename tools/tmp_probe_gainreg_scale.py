"""Calibrate `TRACK.GAIN_REG_W` before spending a sweep on it.

The penalty is a dimensionless squared retention (~0.05 at the measured G=0.22) while the task loss
carries `LAMBDA_T = 3e4`-scale weights, so the two are not comparable a priori and a guessed lambda
grid would most likely be entirely inert or entirely dominant. This prints the task loss and the
penalty at the checkpoint the sweep starts from, which turns the grid into "penalty is x% of the
task loss" instead of a guess.

Also reports the penalty measured with the trained weights, which must land near the probe's
`gain_rand` for the arm -- that is the check that the training-time term and the evaluation-time
diagnostic are the same quantity.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from model import MNISTModel                                       # noqa: E402
from semkine.dataset import build_dataset                          # noqa: E402
from semkine.events import EventPacket, collate_packets            # noqa: E402
from config import load_config                                     # noqa: E402


def _collate(items):
    if isinstance(items[0], EventPacket):
        return collate_packets(items)
    if isinstance(items[0], tuple) and isinstance(items[0][0], EventPacket):
        return tuple(collate_packets(list(col)) for col in zip(*items))
    return torch.utils.data.default_collate(items)


def _to_dev(batch, dev):
    if hasattr(batch, "events"):
        for f in batch.__dataclass_fields__:
            v = getattr(batch, f)
            if torch.is_tensor(v):
                setattr(batch, f, v.to(dev))
        return batch
    return [v.to(dev) if torch.is_tensor(v) else v for v in batch]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/semkine/s1_track_domrand.yaml")
    ap.add_argument("--ckpt", default=None, help="if given, also report the trained-weight gain")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    cfg.setdefault("TRACK", {})["GAIN_REG_W"] = 1.0
    cfg["TRACK"]["GAIN_REG_SCALE"] = args.scale

    comp = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    ds = build_dataset(cfg, "train", comp, train=True)
    raw = ds.input_mode != "legacy_lnes"
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                        drop_last=True, collate_fn=_collate if raw else None)

    model = MNISTModel(cfg).to(dev)
    tag = "init"
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded {args.ckpt}  missing={len(missing)} unexpected={len(unexpected)}")
        tag = "ckpt"
    model.train()

    task, pen = [], []
    for i, batch in enumerate(loader):
        if i >= args.batches:
            break
        batch = _to_dev(batch, dev)
        pred, y, betas, packed = model._predict_batch(batch)
        loss, _ = model._compute_loss(pred, y, betas)
        g2 = model._gain_penalty(packed, pred)
        task.append(float(loss.detach()))
        pen.append(float(g2.detach()))

    L, G2 = float(np.mean(task)), float(np.mean(pen))
    G = G2 ** 0.5
    print(f"\n{tag}: task_loss={L:.6g}  penalty(G^2)={G2:.6g}  implied_gain={G:.4f}")
    print(f"{'lambda':>10} {'lam*G^2':>12} {'% of task':>10}")
    grid = {}
    for frac in (0.01, 0.03, 0.1, 0.3, 1.0):
        lam = frac * L / max(G2, 1e-12)
        grid[frac] = lam
        print(f"{lam:10.4g} {lam * G2:12.4g} {100 * frac:9.1f}%")


if __name__ == "__main__":
    main()
