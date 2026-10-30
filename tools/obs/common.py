"""OBS shared helpers (docs/EXPERIMENT_RECORDS.md#obs-prereg-20261026): event-label time-alignment audit (OBS-A) and
individual-calibration upper bound (OBS-C). CPU only.

Two clocks for the source frames of a stream (N frames, header stamps from `aux["gt_timestamp_ns"]`):
  H  c_n^H = header_n - header_0                      (the labels' clock: `tools/prepare_hand_data.py` interpolates the
                                                       pseudo-GT to 1 ms at these times; v2e's mask-dir path maps the
                                                       interpolated frames onto these stamps)
  U  c_n^U(p) = n * D / p,  D = header_{N-1} - header_0      (v2e without the mask-time mapping: interpTimes = positions *
                                                       D / max(position); with SuperSloMo the last emitted position is
                                                       p = N - 2 + (m - 1) / m, m the last batch's upsampling factor,
                                                       without it p = N - 1). OA1 treats p as a nuisance (`pmax_candidates`);
                                                       OA2 / OA3 use p = N - 1, the TBC definition of e(n).
If a stream's events are on U, an event at time t shows the hand at true time t + e(t), e(t) = interp(t; c^U -> c^H) - t,
while the label stored at t shows the hand at t. Event times are `ms index * 1000 + tsub` microseconds, relative to frame 0.

Data helpers (sequence directories, the E3 W200 samples, the frozen S39 loaders) are imported read-only from
`tools/mv2/common.py`. Every tool verifies the frozen OBS protocol (`outputs/obs/protocol.json`) before it reads anything;
`--smoke` runs print shapes, counts and timings only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from mv2 import common as MC  # noqa: E402  (read-only: data / S39 helpers)
from semkine.dataset import sequences_for_split  # noqa: E402

OUT = REPO / "outputs" / "obs"
PROTOCOL = OUT / "protocol.json"
DATA_ROOT = MC.DATA_ROOT
SEEDS = MC.SEEDS
STEP = MC.STEP
TRAIN_SUBJECTS = MC.FIT7 + MC.SELECT
BASE_TRAIN = tuple(MC.base_seqs(TRAIN_SUBJECTS))            # 18 base training recordings
ZGZ = ("zgz_local", "zgz_global")
BASE = BASE_TRAIN + ZGZ                                     # the 20 base streams OBS-A's verdict is about
PREREG_BEGIN = '<a id="obs-prereg-20261026"></a>'
PREREG_END = "<!-- OBS-PREREG:END -->"


# ----------------------------------------------------------------------------------------------- protocol / io
def sha(path) -> str:
    return MC.sha(path)


def prereg_text() -> Optional[str]:
    doc = (REPO / "docs" / "EXPERIMENT_RECORDS.md").read_text(encoding="utf-8")
    m = re.search(re.escape(PREREG_BEGIN) + r"(.*?)" + re.escape(PREREG_END), doc, re.S)
    return m.group(1) if m else None


def verify_protocol(require: bool = True) -> dict:
    """Every frozen file must match its registered SHA-256 and the pre-registration text its digest."""
    if not PROTOCOL.exists():
        if require:
            raise SystemExit("outputs/obs/protocol.json missing: freeze before reading (tools/obs/freeze.py)")
        return {}
    proto = json.loads(PROTOCOL.read_text())
    bad = [f for f, h in proto["files"].items() if not (REPO / f).exists() or sha(REPO / f) != h]
    bad += [c for c, h in proto.get("checkpoints", {}).items() if not Path(c).exists() or sha(c) != h]
    if bad:
        raise SystemExit(f"frozen files changed since the OBS freeze: {bad}")
    txt = prereg_text()
    if txt is None or hashlib.sha256(txt.encode()).hexdigest() != proto["prereg_sha256"]:
        raise SystemExit("the OBS pre-registration text changed after the freeze")
    return proto


def verify_extra(anchor: str) -> dict:
    """verify_protocol() plus the digest of a further pre-registration block registered with `amend.py prereg`."""
    proto = verify_protocol()
    ent = proto.get("extra_preregs", {}).get(anchor)
    if ent is None:
        raise SystemExit(f"pre-registration {anchor} not registered in outputs/obs/protocol.json (tools/obs/amend.py prereg)")
    doc = (REPO / "docs" / "EXPERIMENT_RECORDS.md").read_text(encoding="utf-8")
    m = re.search(re.escape(f'<a id="{anchor}"></a>') + r"(.*?)" + re.escape(ent["end"]), doc, re.S)
    if m is None or hashlib.sha256(m.group(1).encode()).hexdigest() != ent["sha256"]:
        raise SystemExit(f"the pre-registration {anchor} changed after it was registered")
    return proto


def save_json(path, obj, exclusive: bool = True):
    MC.save_json(path, obj, exclusive=exclusive)


def receipt(out_dir, extra: dict, scripts: Iterable = ()):
    MC.receipt(out_dir, extra, scripts)


def force_cpu():
    """OBS is CPU only."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ----------------------------------------------------------------------------------------------- streams
def all_streams() -> List[Tuple[str, str]]:
    """(sequence, split dir) of every stream of hand_data51 that the train and val_core splits list: base training
    recordings, their _v2/_v3/_v4 variants and the two zgz recordings."""
    out = {}
    for split in ("train", "val_core"):
        for name, d in sequences_for_split(DATA_ROOT, split):
            out[name] = d
    return sorted(out.items())


def stream_kind(seq: str) -> str:
    if seq in ZGZ:
        return "zgz"
    if seq in BASE_TRAIN:
        return "base_train"
    return "variant"


def load_aux(seq: str, d: str) -> dict:
    a = np.load(DATA_ROOT / d / f"{seq}_aux.npz", allow_pickle=True)
    return {k: a[k] for k in a.files}


# ----------------------------------------------------------------------------------------------- clocks
def clocks_us(aux: dict) -> Tuple[np.ndarray, np.ndarray]:
    """(c_H, c_U) in microseconds (float64) of every source frame; c_U with p = N - 1 (the TBC definition)."""
    ts = np.asarray(aux["gt_timestamp_ns"], dtype=np.int64)
    c_h = (ts - ts[0]).astype(np.float64) / 1e3
    return c_h, clock_u(c_h, len(ts) - 1.0)


def clock_u(c_h: np.ndarray, pmax: float) -> np.ndarray:
    """U clock of the N source frames when the last emitted interpolation position is `pmax`: n * D / pmax."""
    return np.arange(len(c_h), dtype=np.float64) * (float(c_h[-1]) / float(pmax))


def pmax_candidates(n: int, m_max: int = 200) -> List[float]:
    """Last emitted interpolation positions v2e can produce for N = n source frames: n - 1 (no SuperSloMo) and
    n - 2 + (m - 1) / m for a last-batch upsampling factor m = 1 .. m_max."""
    return [n - 1.0] + [n - 2.0 + (m - 1.0) / m for m in range(1, m_max + 1)]


def u_to_h(t_us: np.ndarray, c_h: np.ndarray, c_u: np.ndarray) -> np.ndarray:
    """True (H-clock) time of an event stamped t on the U clock: same source-frame position, piecewise linear."""
    return np.interp(np.asarray(t_us, dtype=np.float64), c_u, c_h)


def h_to_u(t_us: np.ndarray, c_h: np.ndarray, c_u: np.ndarray) -> np.ndarray:
    return np.interp(np.asarray(t_us, dtype=np.float64), c_h, c_u)


def e_ms_at(t_ms: np.ndarray, c_h: np.ndarray, c_u: np.ndarray) -> np.ndarray:
    """e(t) in ms at U-clock times t (ms): true time - t."""
    t_us = np.asarray(t_ms, dtype=np.float64) * 1e3
    return (u_to_h(t_us, c_h, c_u) - t_us) / 1e3


def tol_us(t_us: np.ndarray) -> np.ndarray:
    """Hit tolerance: 3 us + 2 ulp of float32 seconds at t (the emulator's timestamps are float32 seconds)."""
    s = np.asarray(t_us, dtype=np.float64) / 1e6
    ulp = np.spacing(s.astype(np.float32)).astype(np.float64) * 1e6
    return 3.0 + 2.0 * ulp


def event_times_us(seq: str, d: str) -> np.ndarray:
    """Every event's timestamp in microseconds relative to frame 0 (int64): ms bin * 1000 + tsub."""
    offsets = np.load(DATA_ROOT / d / f"{seq}_offsets.npy", mmap_mode="r")
    tsub = np.load(DATA_ROOT / d / f"{seq}_tsub.npy", mmap_mode="r")
    if int(offsets[-1]) != len(tsub):
        raise RuntimeError(f"{seq}: tsub rows {len(tsub)} != events {int(offsets[-1])}")
    counts = np.diff(np.asarray(offsets, dtype=np.int64))
    t = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    t *= 1000
    sub = np.asarray(tsub, dtype=np.int64)
    if len(sub) and int(sub.max()) >= 1000:
        raise RuntimeError(f"{seq}: tsub >= 1000 us")
    t += sub
    return t


def layers_us(t: np.ndarray) -> np.ndarray:
    """Distinct event timestamps (the emulator's common time layers), sorted."""
    if len(t) and np.all(t[1:] >= t[:-1]):
        keep = np.empty(len(t), dtype=bool)
        keep[0] = True
        keep[1:] = t[1:] != t[:-1]
        return t[keep]
    return np.unique(t)


def nearest_dist(sorted_vals: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Distance from each query to the nearest value of a sorted array."""
    sv = np.asarray(sorted_vals, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if len(sv) == 0:
        return np.full(len(q), np.inf)
    i = np.searchsorted(sv, q)
    lo = np.clip(i - 1, 0, len(sv) - 1)
    hi = np.clip(i, 0, len(sv) - 1)
    return np.minimum(np.abs(q - sv[lo]), np.abs(sv[hi] - q))


def block_ci(values: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 20261026) -> Tuple[float, float]:
    return MC.block_bootstrap_ci(np.asarray(values, dtype=np.float64), np.asarray(groups), n_boot=n_boot, seed=seed)


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
