#!/usr/bin/env python3
"""PFO2 deterministic training (docs/EXPERIMENT_RECORDS.md#pfo2-amend1-20261103): tools/pfo2/train.py's procedure with
Lightning's Trainer built with `deterministic=True` in every DDP rank.

semkine/train.py builds `pl.Trainer` without `deterministic`; Lightning then calls `torch.use_deterministic_algorithms(False)`
and undoes the wrapper's flag (PFO's train*.py had this gap; the S45-line wrappers pass deterministic=True). With the
flag kept, two runs of the same config are bitwise identical (diagnostics/amend1/). Used for the base B2, its single-change
references and every later PFO2 arm; f1mp keeps F's procedure (tools/pfo2/train.py) because it is paired with F.

    python tools/pfo2/train_det.py --config configs/pfo2_20261103/pfo2_<arm>_s<seed>[_preflight|_head500].yaml
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

POLICY = {"trainer_deterministic": True, "benchmark": False}


def install_policy():
    import pytorch_lightning as pl
    from tools.pfo2.common import now, save, sha
    native = pl.Trainer

    class PolicyRecord(pl.Callback):
        def on_fit_start(self, trainer, model):
            import torch
            if trainer.global_rank == 0:
                out = Path(model.cfg["TRAIN"]["OUTPUT_DIR"])
                save(out / "pfo2_det.json", {**POLICY, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                             "entry_sha256": sha(__file__), "recorded_at_utc": now()}, exclusive=True)

    class Trainer(native):
        def __init__(self, *args, **kwargs):
            kwargs.update(deterministic=True, benchmark=False)
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), PolicyRecord()]
            super().__init__(*args, **kwargs)
    pl.Trainer = Trainer


def main():
    install_policy()
    from tools.pfo2 import train as T2
    T2.main()


if __name__ == "__main__":
    main()
