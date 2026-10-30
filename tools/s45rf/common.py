"""S45-RF identity helpers (docs/EXPERIMENT_RECORDS.md#s45rf-prereg-20261030); CPU only, no scoring.

S45-RF = zero-training probes of where a deployable restoring force could come from, on frozen S45-TR networks:
  P1  is a network's one-step correction of S45's own closed-loop states driven by the packet's events (a pull toward
      the pose the events show) or by the incoming state alone?
  P2  does the frozen correspondence predictor T0's routing carry an informative, unbiased innovation once pooled?
      (registered separately, before its first reading)
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.s45tr.common import now, read, require, save, sha  # noqa: E402,F401

OUT = ROOT / "outputs/s45rf/20261030"
P1 = OUT / "p1"
PHYSICAL_GPUS = (2, 3, 4, 5, 6, 7)
LEDGER = ROOT / "docs/EXPERIMENT_RECORDS.md"
S45TR = ROOT / "outputs/s45tr/20261030"
#: S45's own zgz E1 trace: the common incoming states of S45-TR's E2 (key "<seq>/incoming")
S45_E1_TRACE = ROOT / "outputs/c1sf/20261029/evaluations/so3/e1_step6000_repeat1_none/states.npz"
S45_E3_TRACE = ROOT / "outputs/c1sf/20261029/evaluations/so3/e3_step6000_repeat1_none/states.npz"
#: S45-DEEP's frozen training-mean finger pose (pose_basis.mean, 45D axis-angle)
POSE_BASIS = ROOT / "outputs/s45loc/20261029/deep/protocol.json"
SEQS = ("zgz_global", "zgz_local")


def pose_mean():
    import numpy as np
    return np.asarray(read(POSE_BASIS)["pose_basis"]["mean"], np.float64)


def ledger_section(tag):
    """The text between a ledger record's BEGIN / END markers (the frozen pre-registration)."""
    text = LEDGER.read_text()
    begin, end = f"<!-- EXPERIMENT_RECORD:{tag}:BEGIN -->", f"<!-- EXPERIMENT_RECORD:{tag}:END -->"
    require(text.count(begin) == 1 and text.count(end) == 1, "ledger section missing or duplicated: " + tag)
    return text[text.index(begin):text.index(end) + len(end)]
