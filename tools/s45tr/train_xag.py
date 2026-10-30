#!/usr/bin/env python3
"""S45-TR entry for the arms with MODEL.XAG_* keys (T1d, T1b); amendment 1 (docs/EXPERIMENT_RECORDS.md#s45tr-amend1-20261030).

semkine/train.py adds two XAG audits to any config with MODEL.XAG_* keys outside the XCL study: it imports
`tools.xag.train_audit` (`XAGTrainAudit`, `write`), a package retired in the 2026-10-19 XCL cleanup, and its
pair-stream audit of the first batches requires the XAG source fields in every packet. This entry registers an S45-TR
module under that import name (a provenance callback that records the XAG source build and changes nothing, and the
JSON writer the pair-stream audit calls). For T1b, whose packets carry no ground-truth source fields by design, the
pair-stream audit hashes the base packet fields only (`U1aPairStreamCallback`). Then it runs tools/s45tr/train.py
unchanged; Lightning re-executes this file for DDP rank 1, so both ranks register the same modules.
"""
from __future__ import annotations

from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.s45tr import train as W  # noqa: E402  (strict environment, re-execs once)
import pytorch_lightning as pl  # noqa: E402
import semkine.train as ORIGINAL  # noqa: E402
from tools.s45tr.common import now, save, sha  # noqa: E402


class XAGTrainAudit(pl.Callback):
    """Provenance only: which XAG source build the run read; never changes training."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def on_fit_start(self, trainer, model):
        if trainer.global_rank == 0:
            root = model.cfg["DATA"].get("XAG_SOURCE_ROOT")
            save(self.output_dir / "xag_binding.json", {
                "arm": model.cfg["S45TR"]["ARM"], "config_sha256": sha(model.cfg["_config_path"]),
                "xag_assoc": model.cfg["MODEL"]["XAG_ASSOC"], "xag_gnn": model.cfg["MODEL"]["XAG_GNN"],
                "xag_source_root": root,
                "xag_source_build_summary_sha256": sha(Path(root) / "build_summary.json") if root else None,
                "recorded_at_utc": now()})


def write(path, rows):
    save(path, rows)


def register(arm):
    package = sys.modules.get("tools.xag") or types.ModuleType("tools.xag")
    package.__path__ = []
    module = types.ModuleType("tools.xag.train_audit")
    module.XAGTrainAudit, module.write = XAGTrainAudit, write
    package.train_audit = module
    sys.modules["tools.xag"], sys.modules["tools.xag.train_audit"] = package, module
    if arm == "t1b":
        ORIGINAL.XYZPairStreamCallback = ORIGINAL.U1aPairStreamCallback


def main():
    import os
    arm = os.environ.get("S45TR_ARM")
    if "--arm" in sys.argv:
        arm = sys.argv[sys.argv.index("--arm") + 1]
    if arm not in ("t1d", "t1b"):
        raise SystemExit("train_xag.py is the entry of the XAG-key arms t1d / t1b only")
    register(arm)
    W.main()


if __name__ == "__main__":
    main()
