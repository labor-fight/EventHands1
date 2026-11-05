#!/usr/bin/env python3
"""PFO router training (docs/EXPERIMENT_RECORDS.md#pfo-prereg-20261103, router sections): a slim SRNet
(semkine/pfo_router.py) trained with sr1's recipe unchanged except the architecture (and, when registered, T0P
distillation): S45's training data and augmentation in depth scale mode with the XAG per-event sources, held-out subject
ylf (sr1's eight subjects), two-GPU DDP x 512 packets, Adam 1e-3, 500-step warm-up then constant, BF16, seed 3407,
6000 steps (40 for the preflight). Loss = semkine/s45sm.py `loss` (part CE + point smooth-L1 + 0.5 on-hand BCE)
[+ the S45-SM srkd distillation terms from the frozen T0P]. The data, augmentation, module and fit helpers are the S45-SM
router trainer's (tools/s45sm/router_train.py, imported, unchanged).

    python tools/pfo/router_train.py --router NAME --phase preflight|formal
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.pfo.common import OUT, T0P, now, require, save, sha  # noqa: E402

#: router name -> design of semkine/pfo_router.py SlimSR, distillation, steps; registered in the router section
ROUTERS = {
    "r_s2d": {"design": {"stem": "s2d"}, "kd": None, "steps": 6000},
    "r_s2dns": {"design": {"stem": "s2d", "smooth": "none"}, "kd": None, "steps": 6000},
    "r_s2dw": {"design": {"stem": "s2d", "trunk": [24, 32, 64, 96], "smooth": "none"}, "kd": None, "steps": 6000},
}
KD = {"teacher": "t0p", "T": 2.0, "lambda_part": 1.0, "lambda_coord": 0.5, "lambda_onhand": 0.5}


def router_dir(name, phase="formal"):
    return OUT / "routers" / (name + ("_preflight" if phase == "preflight" else ""))


def checkpoint(name, phase="formal"):
    return router_dir(name, phase) / f"{name}_final.ckpt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--router", required=True, choices=tuple(ROUTERS))
    p.add_argument("--phase", required=True, choices=("preflight", "formal"))
    a = p.parse_args()
    import pytorch_lightning as pl
    import torch
    from semkine import pfo_router as PR
    from semkine import s45rt_router as RT
    from tools.s45sm import router_train as S
    spec = ROUTERS[a.router]
    out, target = router_dir(a.router, a.phase), checkpoint(a.router, a.phase)
    if os.environ.get("LOCAL_RANK") in (None, "0"):
        require(not out.exists(), "fresh-only output already exists: " + str(out))
        out.mkdir(parents=True)
    pl.seed_everything(3407, workers=True)
    cfg = S._data({"dz": True})
    loader = S._loader(S.dataset(cfg, "ylf"))
    torch.manual_seed(3407)
    net = PR.build(cfg["MANO"]["NPZ"], **spec["design"])
    teacher = None
    if spec["kd"]:
        teacher, _ = RT.load(T0P)
        for q in teacher.parameters():
            q.requires_grad_(False)
        teacher = teacher.eval()
    module = S.SRModule(net, None, teacher, spec["kd"])
    steps = int(spec["steps"]) if a.phase == "formal" else 40
    meta = {"heldout": "ylf", "design": net.design, "kd": spec["kd"], "router": a.router}
    if os.environ.get("LOCAL_RANK") in (None, "0"):
        save(out / "launch.json", {"router": a.router, "phase": a.phase, "spec": spec, "trainer_sha256": sha(__file__),
                                   "module_sha256": sha(ROOT / "semkine/pfo_router.py"), "started_utc": now()})
    S._fit(module, loader, out, steps, S.Finish(target, a.phase, PR.save_checkpoint, meta))


if __name__ == "__main__":
    main()
