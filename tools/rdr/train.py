#!/usr/bin/env python3
"""RDR router training (docs/EXPERIMENT_RECORDS.md#rdr-prereg-20261110): the PFO router recipe (tools/pfo/router_train.py:
S45's training data and augmentation in depth scale mode with the XAG per-event sources, held-out subject ylf, Adam 1e-3,
500-step warm-up then constant, BF16, seed 3407, 6000 steps) on ONE GPU with 1024 packets per step (the recipe's two
ranks x 512; here 4 accumulated loader batches of 256, which keeps the worker queues small), so that the routers train
side by side on GPUs 4-7.

  v0  control: r_s2d's own network (semkine/pfo_router.py, stem "s2d") in this trainer
  v1  roi                      semkine/rdr_router.py
  v2  roi + set term (weight 1)
  v3  roi + set term + patch head

    CUDA_VISIBLE_DEVICES=<g> python tools/rdr/train.py --variant v1 --phase preflight|formal [--workers 28]
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

OUT = ROOT / "outputs/rdr/20261110"
VARIANTS = {"v0": {"net": "r_s2d", "set": 0.0}, "v1": {"net": "roi", "set": 0.0}, "v2": {"net": "roi", "set": 1.0},
            "v3": {"net": "roi_hier", "set": 1.0}}
STEPS = {"formal": 6000, "preflight": 20}


def router_dir(name, phase="formal"):
    return OUT / "routers" / (name + ("_preflight" if phase == "preflight" else ""))


def checkpoint(name, phase="formal"):
    return router_dir(name, phase) / f"{name}_final.ckpt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    p.add_argument("--phase", required=True, choices=tuple(STEPS))
    p.add_argument("--workers", type=int, default=26)
    p.add_argument("--prefetch", type=int, default=1)
    p.add_argument("--batch", type=int, default=256, help="packets per loader batch; batch x accum = 1024 per optimiser step")
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--devices", type=int, default=1, help="2 = DDP over two GPUs (set NCCL_P2P_DISABLE=1)")
    a = p.parse_args()
    import pytorch_lightning as pl
    import torch
    from torch.utils.data import DataLoader
    from semkine import pfo_router as PR
    from semkine import rdr_router as RD
    from semkine import s45sm as SM
    from semkine.events import collate_packets
    from tools.pfo.common import now, require, save, sha
    from tools.s45sm import router_train as S
    spec = VARIANTS[a.variant]
    require(a.batch * a.accum * a.devices == 1024, "1024 packets per optimiser step")
    rank0 = os.environ.get("LOCAL_RANK") in (None, "0")
    out, target = router_dir(a.variant, a.phase), checkpoint(a.variant, a.phase)
    if rank0:
        require(not out.exists(), "fresh-only output already exists: " + str(out))
        out.mkdir(parents=True)
    pl.seed_everything(3407, workers=True)
    cfg = S._data({"dz": True})
    loader = DataLoader(S.dataset(cfg, "ylf"), batch_size=a.batch, shuffle=True, num_workers=a.workers, pin_memory=True,
                        persistent_workers=True, prefetch_factor=a.prefetch, drop_last=True, collate_fn=collate_packets)
    torch.manual_seed(3407)
    if spec["net"] == "r_s2d":
        net, save_fn = PR.build(cfg["MANO"]["NPZ"], stem="s2d"), PR.save_checkpoint
    else:
        net, save_fn = RD.build(cfg["MANO"]["NPZ"], hier=spec["net"] == "roi_hier"), RD.save_checkpoint
    params = SM.n_params(net)
    require(params <= RD.BUDGET_PARAMS, f"{a.variant}: {params} parameters exceed r_s2d's {RD.BUDGET_PARAMS}")

    class Module(S._RouterBase):
        def training_step(self, batch, idx):
            o = self.net(batch.events, batch.ptr, batch.delta_t_s)
            tgt = RD.targets(self.net, batch, o["src"], o["mask"])
            total, parts = RD.loss(self.net, o, tgt, spec["set"])
            self.log("train_loss", total, prog_bar=True, sync_dist=a.devices > 1)
            for k, v in parts.items():
                self.log("train_" + k, v)
            with torch.no_grad():
                v = tgt["valid"]
                if bool(v.any()):
                    part = (self.net.predicted_part(o) if hasattr(self.net, "predicted_part") else o["part"].argmax(-1))
                    self.log("train_part_acc", (part == tgt["part"])[v].float().mean())
                    err = (o["coord"].float() * 0.01 - tgt["point"]).norm(dim=-1)[v]
                    self.log("train_coord_err_mm", err.median() * 1000)
                    self.log("train_set_cm", RD.set_term(o, tgt))
                    if "inside" in o:
                        self.log("train_inside", o["inside"][v].float().mean())
            return total

    module = Module(net)
    meta = {"heldout": "ylf", "variant": a.variant, "spec": spec, "design": getattr(net, "design", None)}
    if rank0:
        save(out / "launch.json", {"variant": a.variant, "phase": a.phase, "spec": spec, "params": params, "workers": a.workers, "prefetch": a.prefetch, "batch": a.batch, "accum": a.accum,
                                   "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "trainer_sha256": sha(__file__),
                                   "devices": a.devices, "module_sha256": sha(ROOT / "semkine/rdr_router.py"),
                                   "started_utc": now()})
    from pytorch_lightning.strategies import DDPStrategy
    trainer = pl.Trainer(accelerator="gpu", devices=a.devices,
                         strategy=DDPStrategy(find_unused_parameters=False) if a.devices > 1 else None, precision="bf16", max_steps=STEPS[a.phase], accumulate_grad_batches=a.accum, enable_progress_bar=False,
                         logger=pl.loggers.TensorBoardLogger(str(out), name="logs"), log_every_n_steps=20,
                         enable_checkpointing=False, callbacks=[S.Finish(target, a.phase, save_fn, meta)],
                         num_sanity_val_steps=0)
    trainer.fit(module, train_dataloaders=loader)


if __name__ == "__main__":
    main()
