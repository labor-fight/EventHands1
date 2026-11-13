#!/usr/bin/env python3
"""ESMNet without the part loss on the simulated data (docs/EXPERIMENT_RECORDS.md#nopart-prereg-20261113; user 2026-11-13:
"如果训练时也去掉部位损失，c_i 本身会不会变好或变差……你现在进行重新训练，最后给我一个完整的结论").

tools/rfsu/router_sim.py (the paper's simulated r_s2dx2: r_s2d's simulated recipe through tools/pfo/router_train.py, X2
design, seed 3407, the same data stream and initial weights) with one change: the part-loss weight is 0
(semkine/s45sm.py WEIGHTS["part"]: the loss is c_i smooth-L1 + 0.5 on-hand BCE). The part head stays in the network,
untrained and never read, so every other weight starts identical to r_s2dx2. The checkpoint carries `nopart: True`;
semkine/nopart.py then decodes it with c_i alone (nearest rest-template face centroid over all faces).
Lightning relaunches rank 1 with this file's argv, so the changes below run in both ranks.

    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=a,b python tools/nopart/router_sim.py --router r_s2dx2np --phase formal|preflight
Output: outputs/nopart/20261113/sim/routers/r_s2dx2np[_preflight]/r_s2dx2np_final.ckpt (router kind "pfr").
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = ROOT / "outputs/nopart/20261113/sim"
NAME = "r_s2dx2np"
X2 = {"stem": "s2d", "trunk": [32, 64, 96, 128], "fpn": 64, "head": 128}


def checkpoint(phase="formal"):
    return OUT / "routers" / (NAME + ("_preflight" if phase == "preflight" else "")) / f"{NAME}_final.ckpt"


def main():
    from semkine import pfo_router as PR
    from semkine import s45sm as SM
    from tools.pfo import router_train as RT
    from tools.real.router_variant import VARIANTS
    if VARIANTS["r_s2dx2"] != X2:
        raise SystemExit("r_s2dx2 design differs from tools/real/router_variant.py")
    if SM.WEIGHTS != {"part": 1.0, "coord": 1.0, "onhand": 0.5}:
        raise SystemExit("unexpected router loss weights: " + str(SM.WEIGHTS))
    SM.WEIGHTS["part"] = 0.0                       # the only change: no part loss (c_i + 0.5 on-hand)
    RT.ROUTERS[NAME] = {"design": dict(X2), "kd": None, "steps": 6000}
    RT.OUT = OUT
    native_save = PR.save_checkpoint

    def save_checkpoint(net, path, **meta):
        return native_save(net, path, nopart=True, part_loss_weight=0.0, **meta)
    PR.save_checkpoint = save_checkpoint
    print("nopart router: loss weights", SM.WEIGHTS, flush=True)
    RT.main()


if __name__ == "__main__":
    main()
