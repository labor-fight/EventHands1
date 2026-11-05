#!/usr/bin/env python3
"""ACU s1c: training of the causal point-graph router PR (semkine/acu_pr.py, docs/EXPERIMENT_RECORDS.md#acu-s1c-prereg-20261104).

r_s2d's recipe (tools/pfo/router_train.py -> tools/s45sm/router_train.py helpers, imported unchanged): S45's training data
and augmentation in depth scale mode with the XAG per-event sources, held-out subject ylf, two-GPU DDP x 512 packets, Adam
1e-3, 500-step warm-up then constant, BF16, seed 3407, 6000 steps (40 for the preflight); loss = semkine/s45sm.py `loss`
(part CE + point smooth-L1 + 0.5 on-hand BCE) on the window nodes. Changes: the network (PR), the packets carry the 50 ms
history (semkine/acu_hist.py), the window nodes are the causal stride nodes, Lightning's Trainer is deterministic.

    python tools/acu/router_pr.py --phase preflight|formal
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.acu.common import OUT, now, require, save, sha  # noqa: E402

NAME = "pr"
STEPS = {"preflight": 40, "formal": 6000}


def router_dir(phase="formal"):
    return OUT / "routers" / (NAME + ("_preflight" if phase == "preflight" else ""))


def checkpoint(phase="formal"):
    return router_dir(phase) / f"{NAME}_final.ckpt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True, choices=tuple(STEPS))
    a = p.parse_args()
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.strategies import DDPStrategy
    from torch.utils.data import DataLoader
    from semkine import acu_hist as AH
    from semkine import acu_pr as PR
    from semkine import s45sm as SM
    from tools.s45sm import router_train as S
    out, target = router_dir(a.phase), checkpoint(a.phase)
    rank0 = os.environ.get("LOCAL_RANK") in (None, "0")
    if rank0:
        require(not out.exists(), "fresh-only output already exists: " + str(out))
        out.mkdir(parents=True)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(3407, workers=True)
    AH.install_dataset()
    cfg = S._data({"dz": True})
    loader = DataLoader(S.dataset(cfg, "ylf"), batch_size=512, shuffle=True, num_workers=16, pin_memory=True,
                        persistent_workers=True, prefetch_factor=2, drop_last=True, collate_fn=AH.collate_hist)
    torch.manual_seed(3407)
    net = PR.build(cfg["MANO"]["NPZ"])

    class PRModule(S._RouterBase):
        def training_step(self, batch, idx):
            o = self.net(batch.events, batch.ptr, batch.delta_t_s, batch.hist_events, batch.hist_ptr, batch.k_win(),
                         batch.k_hist)
            tgt = SM.targets(self.net, batch, o["src"], o["mask"])
            total, parts = SM.loss(self.net, o, tgt)
            self.log("train_loss", total, prog_bar=True, sync_dist=True)
            for k, v in parts.items():
                self.log("train_" + k, v, sync_dist=True)
            with torch.no_grad():
                v = tgt["valid"]
                if bool(v.any()):
                    self.log("train_part_acc", (o["part"].argmax(-1) == tgt["part"])[v].float().mean(), sync_dist=True)
            return total

    module = PRModule(net)
    meta = {"heldout": "ylf", "design": net.design, "router": NAME}
    if rank0:
        save(out / "launch.json", {"router": NAME, "phase": a.phase, "design": net.design, "params": PR.n_params(net),
                                   "trainer_sha256": sha(__file__), "module_sha256": sha(ROOT / "semkine/acu_pr.py"),
                                   "hist_sha256": sha(ROOT / "semkine/acu_hist.py"), "started_utc": now()})
    finish = S.Finish(target, a.phase, PR.save_checkpoint, meta)
    trainer = pl.Trainer(accelerator="gpu", devices=2, precision="bf16", max_steps=STEPS[a.phase], deterministic=True,
                         strategy=DDPStrategy(find_unused_parameters=False), enable_progress_bar=False,
                         logger=pl.loggers.TensorBoardLogger(str(out), name="logs"), log_every_n_steps=20,
                         enable_checkpointing=False, callbacks=[finish], num_sanity_val_steps=0)
    trainer.fit(module, train_dataloaders=loader)


if __name__ == "__main__":
    main()
