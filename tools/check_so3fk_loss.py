#!/usr/bin/env python3
"""Zero-training gate for the SO(3)+FK objective, on a real validation batch.

Two things are checked before any GPU hours are spent:

  1. Magnitudes. A unit mix-up is the failure mode this loss is most exposed to,
     because L_rot is dimensionless, L_trans is in metres and L_FK is a metre-scale
     distance. L_trans sitting far below L_FK is *expected*: SmoothL1 is quadratic
     below 1 rad, so a centimetre error contributes ~5e-5 while the same error shows
     up in L_FK at face value. What would be a real bug is an L_FK three orders of
     magnitude above L_trans, which would mean joints are in millimetres.

  2. Gradients. The whole point of L_FK is that it back-propagates through MANO into
     the trunk, so a stray detach would silently turn it into a constant.

Run against an untrained model: the numbers are the initial-loss scale the training
LR was tuned for, not a quality claim.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from config import load_config  # noqa: E402
from fastevc import build_hand_data51_datasets  # noqa: E402
from model import MNISTModel  # noqa: E402


def real_val_batch(cfg, bsz, device):
    mano = np.load(cfg["MANO"]["NPZ"])
    val = build_hand_data51_datasets(cfg, "val", mano["hands_components"].astype(np.float32),
                                     train=False)
    for ds in val:
        ds.fixed_window = int(cfg.get("EVAL", {}).get("WINDOW_MS", 100))
        ds.speed_aug = ds.polarity_flip = ds.pixel_polarity_swap = False
    batch = next(iter(DataLoader(ConcatDataset(val), batch_size=bsz, shuffle=False)))
    return [t.to(device) if torch.is_tensor(t) else t for t in batch]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eventhands_track_render51_so3fk.yaml")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(int(cfg.get("SEED", 0)))
    device = torch.device(args.device)
    model = MNISTModel(cfg).to(device).train()

    batch = real_val_batch(cfg, args.batch_size, device)
    x, prevpos, y, betas, camera_K = model._unpack_batch(batch)
    pred = model(x, prevpos, betas=betas, camera_K=camera_K)
    loss, parts = model._compute_loss(pred, y, betas)

    print(f"\nconfig       = {args.config}")
    print(f"LOSS.TYPE    = {model.loss_type}  "
          f"(w_rot={model.rot_weight:g}, w_trans={model.trans_weight:g}, w_fk={model.fk_weight:g})")
    print(f"batch        = {x.shape[0]} real val samples, render channels {model.render_channels}")

    print("\n-- section 9: magnitudes --")
    print(f"loss_rot         = {float(parts['loss_rot']):.6f}   (dimensionless, 1-cos^2)")
    print(f"loss_trans       = {float(parts['loss_trans']):.6f}   (SmoothL1 on metres)")
    print(f"loss_fk          = {float(parts['loss_fk']):.6f}   (mean joint distance, metres)")
    print(f"2 * loss_fk      = {2.0 * float(parts['loss_fk']):.6f}")
    print(f"loss_total       = {float(loss):.6f}")
    print(f"log10(total)     = {float(loss.log10()):.6f}   "
          f"(what training_step back-propagates, LOSS.LOG10={model.log10_loss})")
    print(f"\nlegacy_loss_51d  = {float(parts['legacy_loss_51d']):.6f}   (old weighted MSE, log only)")
    print(f"rotation_err     = {float(parts['rotation_error_deg']):.2f} deg   (metric only)")
    print(f"joint_err        = {float(parts['joint_error_mm']):.2f} mm    (metric only)")

    t = y[:, model.slices.transl].float()
    print(f"\nGT translation   = mean |t| {t.abs().mean():.4f}, "
          f"z {t[:, 2].mean():.4f}  -> unit is METRES, no extra normalization applied")
    ratio = float(parts["loss_fk"]) / max(float(parts["loss_trans"]), 1e-12)
    print(f"L_FK / L_trans   = {ratio:.1f}  "
          f"({'expected: SmoothL1 quadratic region' if ratio < 1e4 else 'SUSPECT unit mismatch'})")

    print("\n-- section 7: gradients --")
    probes = {
        "conv1.weight": model.conv1.weight,
        "rn.layer4[0].conv1.weight": model.rn.layer4[0].conv1.weight,
        "rn.fc.weight (51D head)": model.rn.fc.weight,
        "prev_mlp[2].weight": model.prev_mlp[2].weight,
    }
    ok = True
    for term in ("loss_fk", "loss_rot", "loss_trans", "TOTAL"):
        tensor = loss if term == "TOTAL" else parts[term]
        grads = torch.autograd.grad(tensor, list(probes.values()), retain_graph=True,
                                    allow_unused=True)
        bits = []
        for name, g in zip(probes, grads):
            n = 0 if g is None else int(torch.count_nonzero(g))
            finite = g is not None and bool(torch.isfinite(g).all())
            ok &= n > 0 and finite
            bits.append(f"{name}: {n} nonzero{'' if finite else ' NON-FINITE'}")
        print(f"  d({term})/d: " + " | ".join(bits))

    # Live rows of the 51D head say which output dimensions each term can move.
    for term, want in (("loss_trans", 3), ("loss_rot", 48), ("loss_fk", 51)):
        g = torch.autograd.grad(parts[term], model.rn.fc.weight, retain_graph=True)[0]
        live = int((g != 0).any(dim=1).sum())
        ok &= live == want
        print(f"  {term}: {live}/51 head rows live (expected {want})")

    print(f"\nlegacy_loss_51d requires_grad = {parts['legacy_loss_51d'].requires_grad} "
          "(must be False: log only)")
    ok &= not parts["legacy_loss_51d"].requires_grad
    print("\nVERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
