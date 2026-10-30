"""K-star shared helpers (docs/EXPERIMENT_RECORDS.md#kstar-prereg-20261025).

A thin layer over already-validated pieces: the XVM helpers (`tools/xvm/common.py`: float64 MANO, SAE, endpoint grid,
evaluator scoring), the S39 three-seed closed-loop archives (`outputs/s44/p10/s39_s{seed}_control.npz`) and the frozen
XVM patch heads. CPU only. Every probe verifies the frozen protocol (`outputs/kstar/protocol.json`) before it reads
anything, and refuses to overwrite a result.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from xvm import common as XC  # noqa: E402

OUT = REPO / "outputs" / "kstar"
PROTOCOL = OUT / "protocol.json"
ZGZ = XC.ZGZ
FIT_SUBJ = XC.TRAIN_SUBJ
SELECT_SUBJ = XC.SELECT_SUBJ
SEEDS = (3407, 3408, 3409)
ARCHIVE = REPO / "outputs" / "s44" / "p10"
HEADS = REPO / "outputs" / "xvm" / "s2" / "p4" / "A_m600" / "heads.pt"
BLOCK = 20                     # endpoints per bootstrap block (XVM / AGG778 convention), within a run
N_BOOT = 2000
BOOT_SEED = 20261025


def sha(path) -> str:
    return XC.sha(path)


def verify_protocol(require=True) -> dict:
    """Every frozen file must match its registered SHA-256; refuses to run otherwise."""
    if not PROTOCOL.exists():
        if require:
            raise SystemExit("outputs/kstar/protocol.json missing: freeze before reading (tools/kstar/freeze.py)")
        return {}
    proto = json.loads(PROTOCOL.read_text())
    bad = [f for f, h in proto["files"].items() if not (REPO / f).exists() or sha(REPO / f) != h]
    if bad:
        raise SystemExit(f"frozen files changed since the K-star freeze: {bad}")
    import re
    doc = (REPO / "docs" / "EXPERIMENT_RECORDS.md").read_text()
    m = re.search(r'<a id="kstar-prereg-20261025"></a>(.*?)<!-- KSTAR-PREREG:END -->', doc, re.S)
    if not m or hashlib.sha256(m.group(1).encode()).hexdigest() != proto["prereg_sha256"]:
        raise SystemExit("the K-star pre-registration text changed after the freeze")
    return proto


def save_json(path: Path, obj, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise SystemExit(f"{path} exists; preserve it")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_np))
    os.replace(tmp, path)


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def receipt(out_dir: Path, extra: dict, scripts=(), inputs=()):
    rec = {"scripts": {str(Path(s).resolve().relative_to(REPO)): sha(s) for s in (__file__, *scripts)},
           "inputs": {str(Path(s).resolve().relative_to(REPO)): sha(s) for s in inputs},
           "protocol_sha256": sha(PROTOCOL) if PROTOCOL.exists() else None,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), **extra}
    save_json(Path(out_dir) / "receipt.json", rec, exclusive=False)


# ------------------------------------------------------------------------------------------------ S39 archive
def archive(seed: int) -> dict:
    """{seq: {pred, prev, gt, end, run, ...}} of one S39 seed (zgz, 50 ms endpoints, own closed loop)."""
    z = np.load(ARCHIVE / f"s39_s{seed}_control.npz")
    out = {}
    for k in z.files:
        seq, key = k.split("|")
        out.setdefault(seq, {})[key] = z[k]
    return out


def archive_files():
    return [ARCHIVE / f"s39_s{s}_control.npz" for s in SEEDS]


def zgz_betas() -> dict:
    """GT betas of the two zgz sequences (the evaluator's privileged input)."""
    from s44 import evflow as EF
    return {s: np.asarray(EF.Seq(s, d).betas, np.float64) for s, d in XC.zgz_seqs()}


# ------------------------------------------------------------------------------------------------ statistics
def blocks_of(run: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Block id per endpoint: consecutive endpoints of one run, `block` at a time."""
    run = np.asarray(run)
    ids = np.zeros(len(run), np.int64)
    b, k, prev = 0, 0, None
    for i, r in enumerate(run):
        if prev is not None and (r != prev or k == block):
            b += 1
            k = 0
        ids[i] = b
        k += 1
        prev = r
    return ids


def block_ci(values: np.ndarray, block_ids: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED, q=(2.5, 97.5)):
    """Mean and percentile block-bootstrap CI of per-endpoint `values` (blocks resampled with replacement)."""
    values = np.asarray(values, np.float64)
    ub = np.unique(block_ids)
    sums = np.array([values[block_ids == b].sum() for b in ub])
    cnts = np.array([(block_ids == b).sum() for b in ub])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ub), size=(n_boot, len(ub)))
    boot = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.percentile(boot, q)
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(values)),
            "n_blocks": int(len(ub))}
