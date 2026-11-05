#!/usr/bin/env python3
"""PCR router training (docs/EXPERIMENT_RECORDS.md#pcr-prereg-20261105): r_s2d's recipe (tools/pfo/router_train.py:
S45's training data and augmentation in depth scale mode with the XAG per-event sources, held-out subject ylf, two-GPU
DDP x 512 packets, Adam 1e-3, 500-step warm-up then constant, BF16, seed 3407, 6000 steps; 40 for the preflight; loss =
semkine/s45sm.py `loss`) with the single registered change of the network input: semkine/pcr_router.py PCRNet reads the
packet's previous state as a part map. Arm pcrb additionally replaces half of the training priors by S45-SM pfasrdz's own
closed-loop states (semkine/s45tr_prior.py `install_n4`, the N4 bank sampler: same augmentation realisation, no noise).

    python tools/pcr/train.py --router pcr|pcrb --phase preflight|formal
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

from tools.pcr.common import OUT, PFASRDZ_BANK, now, require, save, sha  # noqa: E402

ROUTERS = {"pcr": {"design": {"stem": "s2d"}, "bank": None, "steps": 6000},
           "pcrb": {"design": {"stem": "s2d"}, "bank": {"root": str(PFASRDZ_BANK), "p": 0.5}, "steps": 6000}}
PARAMS = 137508
CODE = ("tools/pcr/train.py", "semkine/pcr_router.py", "semkine/pfo_router.py", "semkine/s45sm.py",
        "tools/s45sm/router_train.py", "tools/pfo/router_train.py", "semkine/s45tr_prior.py", "semkine/dataset.py",
        "semkine/domrand.py")


def router_dir(name, phase="formal"):
    return OUT / "routers" / (name + ("_preflight" if phase == "preflight" else ""))


def checkpoint(name, phase="formal"):
    return router_dir(name, phase) / f"{name}_final.ckpt"


def module_class():
    import torch
    from semkine import pcr_router as PCR
    from semkine import s45sm as SM
    from tools.s45sm import router_train as S

    class PCRModule(S.SRModule):
        def training_step(self, batch, idx):
            batch = self._augmented(batch)
            prior = PCR.prior_maps(batch.prev_state, batch.betas, batch.camera_K, self.net.vert_part)
            out = self.net(batch.events, batch.ptr, batch.delta_t_s, prior)
            tgt = SM.targets(self.net, batch, out["src"], out["mask"])
            total, parts = SM.loss(self.net, out, tgt)
            self.log("train_loss", total, prog_bar=True, sync_dist=True)
            for k, v in parts.items():
                self.log("train_" + k, v, sync_dist=True)
            with torch.no_grad():
                v = tgt["valid"]
                if bool(v.any()):
                    pred = out["part"].argmax(-1)
                    self.log("train_part_acc", (pred == tgt["part"])[v].float().mean(), sync_dist=True)
                    g = lambda q: torch.where(q == 0, torch.zeros_like(q), (q - 1) // 3 + 1)  # noqa: E731
                    self.log("train_finger_acc", (g(pred) == g(tgt["part"]))[v].float().mean(), sync_dist=True)
                    err = (out["coord"].float() * 0.01 - tgt["point"]).norm(dim=-1)[v]
                    self.log("train_coord_err_mm", err.median() * 1000, sync_dist=True)
                    self.log("prior_in_norm", self.net.prior_in.weight.float().norm(), sync_dist=True)
            return total
    return PCRModule


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--router", required=True, choices=tuple(ROUTERS))
    p.add_argument("--phase", required=True, choices=("preflight", "formal"))
    a = p.parse_args()
    import pytorch_lightning as pl
    import torch
    from semkine import pcr_router as PCR
    from tools.s45sm import router_train as S
    spec = ROUTERS[a.router]
    out, target = router_dir(a.router, a.phase), checkpoint(a.router, a.phase)
    rank0 = os.environ.get("LOCAL_RANK") in (None, "0")
    if rank0:
        require(not out.exists(), "fresh-only output already exists: " + str(out))
        out.mkdir(parents=True)
    pl.seed_everything(3407, workers=True)
    cfg = S._data({"dz": True})
    prior_source = "dataset (TRACK mixed noise)"
    if spec["bank"]:
        from semkine import s45tr_prior as P
        P.install_n4(spec["bank"]["root"], float(spec["bank"]["p"]))
        prior_source = f"n4 bank {spec['bank']['root']} p={spec['bank']['p']} (else dataset noise)"
    loader = S._loader(S.dataset(cfg, "ylf"))
    torch.manual_seed(3407)
    net = PCR.build(cfg["MANO"]["NPZ"], **spec["design"])
    require(sum(q.numel() for q in net.parameters()) == PARAMS, "parameter count differs from the registration")
    module = module_class()(net, None, None, None)
    steps = int(spec["steps"]) if a.phase == "formal" else 40
    meta = {"heldout": "ylf", "design": net.design, "router": a.router, "bank": spec["bank"], "prior_source": prior_source}
    if rank0:
        save(out / "launch.json", {"router": a.router, "phase": a.phase, "spec": spec, "prior_source": prior_source,
                                   "code_sha256": {c: sha(ROOT / c) for c in CODE}, "params": PARAMS,
                                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "started_utc": now()})
    S._fit(module, loader, out, steps, S.Finish(target, a.phase, PCR.save_checkpoint, meta))


if __name__ == "__main__":
    main()
