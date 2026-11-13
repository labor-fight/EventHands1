#!/usr/bin/env python3
"""Simulated-data r_s2dx2 router (docs/EXPERIMENT_RECORDS.md#rfsu-20261113; user 2026-11-13: "论文用 r_s2dx2，真实数据，
仿真数据，和 EvRealHands ... 都换成 r_s2dx2").

tools/pfo/router_train.py unchanged -- r_s2d's simulated recipe: S45's training data and augmentation in depth scale mode
with the XAG per-event sources, held-out subject ylf (sr1's eight subjects), two-GPU DDP x 512 packets, Adam 1e-3,
500-step warm-up then constant, BF16, seed 3407, 6000 steps (40 for the preflight) -- with only the network design
(tools/real/router_variant.py's r_s2dx2: stem s2d, trunk 32/64/96/128, fpn 64, head 128) and the output folder changed.
Lightning relaunches rank 1 with this file's argv, so the registration below runs in both ranks.

    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=a,b python tools/rfsu/router_sim.py --router r_s2dx2 --phase formal
Output: outputs/rfsu/20261113/sim/routers/r_s2dx2/r_s2dx2_final.ckpt (router kind "pfr", design in the checkpoint).
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = ROOT / "outputs/rfsu/20261113/sim"
X2 = {"stem": "s2d", "trunk": [32, 64, 96, 128], "fpn": 64, "head": 128}


def checkpoint(phase="formal"):
    return OUT / "routers" / ("r_s2dx2" + ("_preflight" if phase == "preflight" else "")) / "r_s2dx2_final.ckpt"


def main():
    from tools.pfo import router_train as RT
    from tools.real.router_variant import VARIANTS
    if VARIANTS["r_s2dx2"] != X2:
        raise SystemExit("r_s2dx2 design differs from tools/real/router_variant.py")
    RT.ROUTERS["r_s2dx2"] = {"design": dict(X2), "kd": None, "steps": 6000}
    RT.OUT = OUT                        # router_dir() reads the module global: outputs/rfsu/20261113/sim/routers/<name>
    RT.main()


if __name__ == "__main__":
    main()
