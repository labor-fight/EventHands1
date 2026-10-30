"""AURA shared identity helpers (docs/EXPERIMENT_RECORDS.md#aura-prereg-20261024).

The frozen C1 module, packets, anchors, scoring and statistics are AGG778's (`tools/agg778/common.py`, imported
unchanged). This module adds the AURA protocol check, receipts and the R2 / MRF completion markers. model/model.py is
frozen outside `files`: Stages A-C accept the file AGG778 R2 froze or that file with the MRF wiring (`model_states`),
Stage D the AURA-wired file of whichever state it started from (`tools/aura/wiring.py`).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from tools.agg778 import common as C

REPO = C.REPO
OUT = REPO / "outputs" / "aura"
PROTOCOL = OUT / "protocol.json"
GPUS = ("4", "5", "6", "7")
#: model/model.py: AGG778 R2's frozen file, and that file with the MRF `MODEL.XYZ_RF` wiring applied
MODEL_PRE = {"r2": "384490909cf31266cdfbcada910405f96546de629baa607a91e6647c325eb284",
             "mrf": "f5716a1b1e97d5f918af77be6ffb33ad75a2f1b27544a73fb8b02a883244b925"}
#: the five finger groups of the 16 LBS joints (0 wrist / palm, 1-3 index, 4-6 middle, 7-9 little, 10-12 ring,
#: 13-15 thumb)
FINGER_OF_PART = (0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5)


def require_gpu():
    """One process = one physical GPU of 4-7 (user authorisation 2026-10-24)."""
    import torch
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if v not in GPUS:
        raise SystemExit(f"AURA jobs run on exactly one physical GPU 4-7 (CUDA_VISIBLE_DEVICES={v!r})")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit("CUDA_DEVICE_ORDER must be PCI_BUS_ID so that indices are physical")
    return torch.device("cuda")


def verify_protocol(stage: str) -> dict:
    """Every entry: the frozen file SHA-256 of outputs/aura/protocol.json must match the tree; model/model.py must be
    one of the accepted states of the stage (A-C: R2's or MRF-wired; D: the AURA-wired file)."""
    if not PROTOCOL.is_file():
        raise SystemExit("AURA protocol.json missing: pre-register and freeze first")
    p = json.loads(PROTOCOL.read_text())
    bad = [k for k, v in p["files"].items() if C.sha(REPO / k) != v]
    if bad:
        raise SystemExit(f"AURA frozen sources changed ({stage}): {bad}")
    have = C.sha(REPO / "model" / "model.py")
    ok = p["model_wiring"]["post"].values() if stage == "D" else p["model_wiring"]["pre"].values()
    if have not in ok:
        raise SystemExit(f"model/model.py ({have[:12]}) is not an accepted state for AURA stage {stage}")
    return p


def receipt(out_dir, extra: dict, scripts=()):
    rec = {"scripts": {str(Path(s).resolve().relative_to(REPO)): C.sha(s) for s in (__file__, C.__file__, *scripts)},
           "protocol_sha256": C.sha(PROTOCOL) if PROTOCOL.is_file() else None,
           "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), **extra}
    C.save_json(Path(out_dir) / "receipt.json", rec, exclusive=False)


def _alive(pattern: str) -> bool:
    """Some process runs a script under `pattern`: one of its argv elements is that path (relative, or absolute and
    containing it). A shell that only mentions the path inside a command string (e.g. a `pgrep -f` watcher loop, which
    matches itself and never ends) does not count."""
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        if any(a.startswith(pattern) or (a.startswith("/") and " " not in a and "/" + pattern in a) for a in argv):
            return True
    return False


def _readouts_complete(base: Path, arms) -> bool:
    want = ("row_step6000_repeat1_none", "e1_step6000_repeat1_none", "e1_step6000_repeat2_none",
            "e2_step6000_repeat1_none", "e3_step6000_repeat1_none", "gt_pose_step6000_repeat1_none",
            "gt_transl_step6000_repeat1_none", "e1_step5000_repeat1_none", "e1_step5500_repeat1_none")
    return all((base / "evaluations" / a / w / "report.json").is_file() for a in arms for w in want)


def r2_complete() -> dict:
    """AGG778 R2 is complete: probe summary, D selection (possibly empty), all selected arms' readouts, no R2 process."""
    r2 = REPO / "outputs" / "agg778" / "r2"
    out = {"summary": (r2 / "summary.json").is_file(), "selection": (r2 / "d" / "selection.json").is_file()}
    arms = json.loads((r2 / "d" / "selection.json").read_text())["arms"] if out["selection"] else None
    out["readouts"] = None if arms is None else _readouts_complete(r2 / "d", arms)
    out["no_process"] = not _alive("tools/agg778/r2_")
    out["complete"] = bool(out["summary"] and out["selection"] and out["readouts"] and out["no_process"])
    return out


def mrf_complete() -> dict:
    """MRF is complete: B summary, C results, D selection and all selected arms' readouts, no MRF process."""
    m = REPO / "outputs" / "mrf"
    out = {"b_summary": (m / "b" / "summary.json").is_file(), "c_summary": (m / "c" / "summary.json").is_file(),
           "selection": (m / "d" / "selection.json").is_file()}
    arms = json.loads((m / "d" / "selection.json").read_text())["arms"] if out["selection"] else None
    out["readouts"] = None if arms is None else _readouts_complete(m / "d", arms)
    out["no_process"] = not _alive("tools/mrf/")
    out["complete"] = bool(out["b_summary"] and out["c_summary"] and out["selection"] and out["readouts"]
                           and out["no_process"])
    return out
