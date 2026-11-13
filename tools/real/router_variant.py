#!/usr/bin/env python3
"""Stronger EvHandTrack routers on the real data (docs/EXPERIMENT_RECORDS.md#real-rfs-20261112; user 2026-11-12:
"想想怎么提升路由质量").

tools/real/evht_router.py unchanged (r_s2d's recipe on the real training data: S45's data contract and augmentation in
depth scale mode, the real XAG sources, all 12 training subjects, two-GPU DDP x 512 packets, Adam 1e-3, 500-step warm-up
then constant, BF16, seed 3407, 6000 steps) with only the network design and the output folder changed. The variant is
named by the environment variable ROUTER_VARIANT (Lightning relaunches rank 1 with the same argv and environment):

  r_s2dx2   stem s2d, trunk 32/64/96/128, fpn 64, head 128          (wider r_s2d)
  r_s1x2    stem s1 (full-resolution 3x3 first convolution), trunk 32/64/96/128, fpn 64, head 128

    ROUTER_VARIANT=r_s2dx2 NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=a,b python tools/real/router_variant.py --phase formal
    python tools/real/router_variant.py --cost            # parameters / MACs of every design on a real packet size
Output: outputs/real/20261110/routers/<variant>/r_s2d_final.ckpt (router kind "pfr", design in the checkpoint).
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VARIANTS = {
    "r_s2d": {"stem": "s2d"},
    "r_s2dx2": {"stem": "s2d", "trunk": [32, 64, 96, 128], "fpn": 64, "head": 128},
    "r_s1x2": {"stem": "s1", "trunk": [32, 64, 96, 128], "fpn": 64, "head": 128},
}


def cost():
    import torch
    from tools.real import evht as X
    X.enter()
    from semkine import pfo_router as PR
    from semkine import s45sm as SM
    npz = str(ROOT / "assets/mano_right.npz")
    for name, design in VARIANTS.items():
        net = PR.build(npz, **design).eval()
        macs = {"n": 0}

        def hook(m, i, o):
            if isinstance(m, torch.nn.Conv2d):
                macs["n"] += o.numel() // o.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
            elif isinstance(m, torch.nn.Linear):
                macs["n"] += i[0].numel() // i[0].shape[-1] * m.in_features * m.out_features
        hs = [m.register_forward_hook(hook) for m in net.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
        n = 2048
        ev = torch.zeros(n, 5)
        ev[:, 1] = torch.randint(0, X.W, (n,)).float()
        ev[:, 2] = torch.randint(0, X.H, (n,)).float()
        ev[:, 3] = torch.rand(n) * 0.05
        ev[:, 4] = torch.randint(0, 2, (n,)).float()
        with torch.no_grad():
            net(ev, torch.tensor([0, n]), torch.tensor([0.05]))
        for h in hs:
            h.remove()
        print(f"{name:10s} params {SM.n_params(net):>9,d}  MACs per packet (2048 nodes) {macs['n'] / 1e6:8.1f} M  {design}")


def main():
    if "--cost" in sys.argv:
        cost()
        return
    name = os.environ.get("ROUTER_VARIANT")
    if name not in VARIANTS or name == "r_s2d":
        raise SystemExit("set ROUTER_VARIANT to one of " + ", ".join(k for k in VARIANTS if k != "r_s2d"))
    from tools.real import evht_router as EVR
    from tools.real import evht as X
    from tools.real import ns
    EVR.DESIGN = dict(VARIANTS[name])
    X.ROUTER_DIR = ns.OUT / "routers" / name
    EVR.main()


if __name__ == "__main__":
    main()
