"""MRF shared identity helpers (docs/EXPERIMENT_RECORDS.md#mrf-prereg-20261024).

The frozen modules, packets, anchors, scoring and statistics are AGG778's (`tools/agg778/common.py`, unchanged); this
module adds the MRF protocol check and receipts. model/model.py is frozen twice, outside `files`: Stages A-C run on the
file AGG778 R2 froze (`model_wiring.pre`); Stage D runs on the same file with the `MODEL.XYZ_RF` wiring applied
(`model_wiring.post`), which is applied only after AGG778 R2 is complete (`tools/mrf/wiring.py`).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from tools.agg778 import common as C

REPO = C.REPO
OUT = REPO / "outputs" / "mrf"
PROTOCOL = OUT / "protocol.json"
GPUS = ("4", "5", "6", "7")


def require_gpu():
    """One process = one physical GPU of 4-7 (user authorisation 2026-10-24)."""
    import torch
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if v not in GPUS:
        raise SystemExit(f"MRF jobs run on exactly one physical GPU 4-7 (CUDA_VISIBLE_DEVICES={v!r})")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit("CUDA_DEVICE_ORDER must be PCI_BUS_ID so that indices are physical")
    return torch.device("cuda")


def verify_protocol(stage: str) -> dict:
    """Every entry: the frozen file SHA-256 of outputs/mrf/protocol.json must match the tree (Stage D: model/model.py
    must be the wired file)."""
    if not PROTOCOL.is_file():
        raise SystemExit("MRF protocol.json missing: pre-register and freeze first")
    p = json.loads(PROTOCOL.read_text())
    expect = dict(p["files"])
    expect["model/model.py"] = p["model_wiring"]["post" if stage == "D" else "pre"]
    bad = [k for k, v in expect.items() if C.sha(REPO / k) != v]
    if bad:
        raise SystemExit(f"MRF frozen sources changed ({stage}): {bad}")
    return p


def receipt(out_dir, extra: dict, scripts=()):
    rec = {"scripts": {str(Path(s).resolve().relative_to(REPO)): C.sha(s) for s in (__file__, C.__file__, *scripts)},
           "protocol_sha256": C.sha(PROTOCOL) if PROTOCOL.is_file() else None,
           "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), **extra}
    C.save_json(Path(out_dir) / "receipt.json", rec, exclusive=False)
