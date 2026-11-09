#!/usr/bin/env python3
"""TPD problem 1, LC router fine-tune (docs/EXPERIMENT_RECORDS.md#tpd-lc-prereg-20261105): r_s2d's recipe (the S45-SM router
trainer's data, targets and loss, tools/s45sm/router_train.py, imported unchanged: S45's training dataset in depth scale
mode with the XAG per-event sources, held-out subject ylf, two-GPU DDP x 512, BF16, seed 3407) with:
  network   semkine/tpd_lc.py LCRouter from r_s2d's final weights (history columns of the stem at zero)
  data      the current window fixed to 50 ms and the 200 ms history (`lc`) or an empty history (`lc0`, the control)
  schedule  Adam 1e-3, 100-step warm-up then cosine to 0, 3000 steps (40 for the preflight); deterministic kernels and
            Lightning's Trainer with deterministic=True

    python tools/tpd/router_lc.py --router lc|lc0 --phase preflight|formal
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.tpd.common import OUT, now, require, save, sha  # noqa: E402

R_S2D = ROOT / "outputs/pfo/20261103/routers/r_s2d/r_s2d_final.ckpt"
ROUTERS = {"lc": {"empty": False}, "lc0": {"empty": True}}
STEPS = {"formal": 3000, "preflight": 40}
LR, WARMUP = 1e-3, 100
HELDOUT = "ylf"


def router_dir(name, phase="formal"):
    return OUT / "routers" / (name + ("_preflight" if phase == "preflight" else ""))


def checkpoint(name, phase="formal"):
    return router_dir(name, phase) / f"{name}_final.ckpt"


def factor(step, total):
    if step < WARMUP:
        return (step + 1) / WARMUP
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - WARMUP) / max(1, total - WARMUP))))


def dataset(cfg, empty):
    import numpy as np
    from torch.utils.data import Subset
    from semkine import tpd_lc as LC
    from semkine.dataset import build_dataset
    comps = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    ds = LC.upgrade(build_dataset(cfg, "train", comps, train=True), empty=empty)
    require(len(ds.sequences) == 72, "training split changed")
    held = np.array([s.startswith(HELDOUT + "_") for s, _ in ds.sequences])
    keep = np.flatnonzero(~held[ds.index[:, 0]])
    require(held.sum() == 8 and 0 < keep.size < len(ds), "held-out subject selection changed")
    return Subset(ds, keep.tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--router", required=True, choices=tuple(ROUTERS))
    p.add_argument("--phase", required=True, choices=tuple(STEPS))
    a = p.parse_args()
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.strategies import DDPStrategy
    from torch.utils.data import DataLoader
    from semkine import s45sm as SM
    from semkine import tpd_lc as LC
    from tools.s45sm import router_train as S
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    out, target = router_dir(a.router, a.phase), checkpoint(a.router, a.phase)
    rank0 = os.environ.get("LOCAL_RANK") in (None, "0")
    if rank0:
        require(not out.exists(), "fresh-only output already exists: " + str(out))
        out.mkdir(parents=True)
    pl.seed_everything(3407, workers=True)
    cfg = S._data({"dz": True})
    loader = DataLoader(dataset(cfg, ROUTERS[a.router]["empty"]), batch_size=512, shuffle=True, num_workers=16,
                        pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True,
                        collate_fn=LC.collate_hist)
    net = LC.from_r_s2d(R_S2D)
    steps = STEPS[a.phase]

    class LCModule(S.SRModule):
        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            return batch.to(device)

        def training_step(self, batch, idx):
            b = batch.base
            out = self.net(b.events, b.ptr, b.delta_t_s, batch.hist_events, batch.hist_ptr)
            tgt = SM.targets(self.net, b, out["src"], out["mask"])
            total, parts = SM.loss(self.net, out, tgt)
            B = b.batch_size
            self.log("train_loss", total, prog_bar=True, sync_dist=True, batch_size=B)
            for k, v in parts.items():
                self.log("train_" + k, v, sync_dist=True, batch_size=B)
            with torch.no_grad():
                v = tgt["valid"]
                if bool(v.any()):
                    self.log("train_part_acc", (out["part"].argmax(-1) == tgt["part"])[v].float().mean(),
                             sync_dist=True, batch_size=B)
            return total

        def configure_optimizers(self):
            opt = torch.optim.Adam(self.net.parameters(), lr=LR)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: factor(s, steps))
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    class Finish(pl.Callback):
        def on_train_end(self, trainer, module):
            if trainer.global_rank == 0:
                step = int(trainer.global_step)
                LC.save_checkpoint(module.net, target, step=step, phase=a.phase, lr=LR, warmup=WARMUP, steps=steps,
                                   params=SM.n_params(module.net), **meta)
                save(target.with_suffix(".json"), {"status": "COMPLETE", "step": step, "phase": a.phase,
                                                   "checkpoint_sha256": sha(target), "params": SM.n_params(module.net),
                                                   "completed_at_utc": now()})

    module = LCModule(net, None, None, None)
    meta = {"heldout": HELDOUT, "init": str(R_S2D), "init_sha256": sha(R_S2D), "router": a.router,
            "empty_history": ROUTERS[a.router]["empty"], "packet_ms": LC.PACKET_MS}
    if rank0:
        save(out / "launch.json", {"router": a.router, "phase": a.phase, "spec": ROUTERS[a.router], "steps": steps,
                                   "lr": LR, "warmup": WARMUP, "trainer_sha256": sha(__file__),
                                   "module_sha256": sha(ROOT / "semkine/tpd_lc.py"), "init_sha256": sha(R_S2D),
                                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "started_utc": now()})
    trainer = pl.Trainer(accelerator="gpu", devices=2, precision="bf16", max_steps=steps,
                         strategy=DDPStrategy(find_unused_parameters=False), enable_progress_bar=False,
                         logger=pl.loggers.TensorBoardLogger(str(out), name="logs"), log_every_n_steps=20,
                         enable_checkpointing=False, deterministic=True, benchmark=False,
                         callbacks=[Finish()], num_sanity_val_steps=0)
    trainer.fit(module, train_dataloaders=loader)


if __name__ == "__main__":
    main()
