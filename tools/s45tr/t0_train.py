#!/usr/bin/env python3
"""S45-TR T0: train the event -> MANO correspondence predictor (semkine/s45tr_t0.py) on one GPU pair.

Data: S45's training dataset and augmentation (configs/c1sf_20261029/r2/c1sf_so3_s3407.yaml) with the XAG per-event
sources (`DATA.XAG_SOURCE_ROOT: outputs/xag/sources`), minus every sequence of the held-out subject ylf (the T0 gate
reads ylf only). zgz is never read. Two-GPU DDP, 512 packets per GPU, Adam 1e-3, 500-step linear warmup then constant,
BF16 autocast, seed 3407; 50 steps (preflight) or 6000 steps (formal). The final network is written to
outputs/s45tr/20261030/t0/t0_final.ckpt (formal) as a plain state dict with the template and the GNN settings.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
from pytorch_lightning.strategies import DDPStrategy  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from config import load_config  # noqa: E402
from semkine import s45tr_t0 as T0  # noqa: E402
from semkine.dataset import build_dataset  # noqa: E402
from semkine.events import collate_packets  # noqa: E402
from tools.frozen import accepts  # noqa: E402
from tools.s45tr.common import (OUT, S45_CONFIG, S45_CONFIG_SHA256, T0_CHECKPOINT, T0_DIR, T1A_GNN,  # noqa: E402
                                XAG_SOURCES, now, require, save, sha, verify_protocol)

HELDOUT = "ylf"
LR, WARMUP = 1e-3, 500


def dataset(cfg):
    components = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    ds = build_dataset(cfg, "train", components, train=True)
    held = np.array([s.startswith(HELDOUT + "_") for s, _ in ds.sequences])
    keep = np.flatnonzero(~held[ds.index[:, 0]])
    require(held.sum() == 8 and 0 < keep.size < len(ds), "held-out subject selection changed")
    return ds, Subset(ds, keep.tolist())


class T0Module(pl.LightningModule):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.gen = torch.Generator().manual_seed(3407)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)

    def training_step(self, batch, idx):
        out = self.net(batch.events, batch.ptr, batch.delta_t_s)
        tgt = T0.targets(self.net, batch, out["src"], out["mask"])
        total, parts = T0.loss(self.net, out, tgt, generator=self.gen)
        self.log("train_loss", total, prog_bar=True, sync_dist=True)
        for k, v in parts.items():
            self.log("train_" + k, v, sync_dist=True)
        live = out["mask"]
        self.log("train_onhand_share", tgt["valid"][live].float().mean(), sync_dist=True)
        return total

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


class Finish(pl.Callback):
    def __init__(self, target, phase):
        self.target, self.phase = target, phase

    def on_train_end(self, trainer, module):
        if trainer.global_rank == 0:
            net = module.net
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                        "faces": net.faces.cpu(), "tmpl": net.tmpl.cpu(), "gnn": net.gnn_cfg,
                        "step": int(trainer.global_step), "phase": self.phase, "heldout": HELDOUT,
                        "weights": T0.WEIGHTS, "face_nodes": T0.FACE_NODES, "lr": LR, "warmup": WARMUP}, self.target)
            save(self.target.with_suffix(".json"), {"status": "COMPLETE", "step": int(trainer.global_step),
                                                    "phase": self.phase, "checkpoint_sha256": sha(self.target),
                                                    "completed_at_utc": now()})


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--phase", required=True, choices=("preflight", "formal"))
    a = p.parse_args()
    verify_protocol()
    require(accepts(S45_CONFIG, S45_CONFIG_SHA256), "S45 config changed")
    out = T0_DIR if a.phase == "formal" else OUT / "t0_preflight"
    target = T0_CHECKPOINT if a.phase == "formal" else out / "t0_final.ckpt"
    if os.environ.get("LOCAL_RANK") in (None, "0"):
        require(not out.exists(), "Fresh-only output already exists: " + str(out))
        out.mkdir(parents=True)
    pl.seed_everything(3407, workers=True)
    cfg = load_config(str(S45_CONFIG))
    cfg["DATA"]["XAG_SOURCE_ROOT"] = str(XAG_SOURCES)
    _, train = dataset(cfg)
    loader = DataLoader(train, batch_size=512, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True,
                        prefetch_factor=2, drop_last=True, collate_fn=collate_packets)
    verts, faces = T0.template(cfg["MANO"]["NPZ"])
    torch.manual_seed(3407)
    net = T0.T0Net(faces, verts, T1A_GNN)
    steps = 6000 if a.phase == "formal" else 50
    trainer = pl.Trainer(accelerator="gpu", devices=2, precision="bf16", max_steps=steps,
                         strategy=DDPStrategy(find_unused_parameters=False), enable_progress_bar=False,
                         logger=pl.loggers.TensorBoardLogger(str(out), name="logs"), log_every_n_steps=20,
                         enable_checkpointing=False, callbacks=[Finish(target, a.phase)], num_sanity_val_steps=0)
    trainer.fit(T0Module(net), train_dataloaders=loader)


if __name__ == "__main__":
    main()
