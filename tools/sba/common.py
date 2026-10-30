"""SBA probe helpers (docs/EXPERIMENT_RECORDS.md#sba-prereg-20261022).

Thin layer over `semkine/sba.py` (the observation model and solver), the S44 sequence access (`tools/s44/evflow.py`)
and the XVM evaluation helpers (`tools/xvm/common.py`: endpoint grid, root-aligned scoring, R_art, increment prior).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from xvm import common as C  # noqa: E402
from s44 import evflow as EF  # noqa: E402
from semkine import sba as SB  # noqa: E402
from semkine import eval_track as ET  # noqa: E402
from mano_layer import ManoLayer  # noqa: E402

OUT = REPO / "outputs" / "sba"
CALIB = OUT / "s0" / "calibration.json"
BANK = REPO / "outputs" / "xdl" / "packet_bank.json"
ZGZ = ("zgz_local", "zgz_global")
#: protocol initial-state noise (configs/dt/dt_dz_l3.yaml TRACK.PREV_NOISE_*), as `ET.sample_init_noise`
INIT_CFG = {"TRACK": {"PREV_NOISE_T": 0.005, "PREV_NOISE_R": 0.05, "PREV_NOISE_POSE": 0.05}}
#: tracker state uncertainty carried to the next window (the training "small noise" class)
STATE_SIG = (0.005, 0.05, 0.05)
#: XDI's step prior scales (translation m, root rad, fingers rad), used as the start-knot prior in P2
XDI_SIG = (0.05, 0.3, 0.3)
#: S39 anchor (fused arm), deliberately loose and fixed a priori (not calibrated on zgz)
S39_SIG = (0.02, 0.10, 0.15)
#: absolute finger-pose prior weight in the standalone loop (training-GT Gaussian)
W_ABS = 0.05
POOLS_TRACK = (4, 2, 1, 1, 1, 1, 1, 1)
POOLS_WIDE = (8, 4, 2, 1, 1, 1, 1, 1)
MIN_EVENTS = 100


def require_gpu() -> torch.device:
    return C.require_gpu()


def sha(path) -> str:
    return C.sha(path)


def receipt(out_dir: Path, extra: dict, scripts=()):
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {"scripts": {str(Path(s).resolve().relative_to(REPO)): sha(s)
                       for s in (__file__, REPO / "semkine" / "sba.py", *scripts)},
           "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), **extra}
    (out_dir / "receipt.json").write_text(json.dumps(rec, indent=1, sort_keys=True))


def sig_prec(sig, dev) -> torch.Tensor:
    t, r, f = sig
    s = torch.full((51,), f, device=dev)
    s[:3], s[3:6] = t, r
    return 1.0 / s ** 2


class Ctx:
    def __init__(self, dev):
        self.dev = dev
        self.mano = ManoLayer(str(EF.MANO_NPZ), add_mean=False).to(dev).eval()
        self.geo = SB.Geometry(self.mano, dev)
        self.P50 = C.increment_precision(dev).float()          # (51, 51) precision of 50 ms additive increments
        self.mean = self.mano.hands_mean.detach().float().reshape(15, 3)

    def inc_prec(self, ms: float) -> torch.Tensor:
        """Increment precision of an `ms` window: covariance ~ (ms / 50)^2 (constant-velocity scaling)."""
        return self.P50 * (50.0 / float(ms)) ** 2

    def wrist(self, x51: torch.Tensor, betas) -> torch.Tensor:
        _, j = self.geo.fk(x51.view(-1, 51), betas)
        return j[:, 0]


def load_model(path: Path = CALIB) -> SB.Model:
    c = json.loads(Path(path).read_text())
    return SB.Model(kappa_sil=c["kappa_sil"], kappa_int=c["kappa_int"], signed=bool(c["signed"]),
                    dispersion=float(c["dispersion"]), blur=float(c["blur"]), outlier=float(c["outlier"]))


class SeqData:
    """One recording on the device: events (host mmap), offsets, GT per ms, betas, source K."""

    def __init__(self, name: str, d: str, dev):
        self.sq = EF.Seq(name, d)
        self.name = name
        self.dev = dev
        self.betas = torch.tensor(self.sq.betas, dtype=torch.float32, device=dev)[None]
        self.K = torch.tensor(self.sq.K, dtype=torch.float32, device=dev)
        self.pos51 = self.sq.pos51
        self.runs = self.sq.runs

    def gt(self, ms: int) -> torch.Tensor:
        return torch.tensor(np.asarray(self.pos51[ms], np.float32), device=self.dev)

    def events(self, m_prev: int, m1: int):
        off = self.sq.offsets
        a0, a1 = int(off[m_prev + 1]), int(off[m1 + 1])
        ev = np.asarray(self.sq.events[a0:a1])
        cnt = np.diff(off[m_prev + 1:m1 + 2]).astype(np.int64)
        ms = np.repeat(np.arange(m_prev + 1, m1 + 1, dtype=np.int64), cnt)
        return ev, ms

    def run_start(self, ms: int) -> int:
        for a, b in self.runs:
            if a <= ms < b:
                return int(a)
        return int(ms)

    def run_bounds(self, ms: int):
        """(first, last) valid ms of the run containing ms."""
        for a, b in self.runs:
            if a <= ms < b:
                return int(a), int(b) - 1
        return int(ms), int(ms)


def window(ctx: Ctx, sd: SeqData, m_prev: int, m1: int, x0: torch.Tensor, x1: torch.Tensor,
           events=None, mcp=SB.MCP_JOINTS) -> SB.Window:
    """Window of events (m_prev, m1] with crop / forearm capsule / hand box from the initial trajectory x0 -> x1.
    `events` (optional): (ev, ms) of another window re-timed onto this one (wrong-content control)."""
    S, edges, alphas = SB.bins_of(m_prev, m1)
    al = torch.tensor(alphas, dtype=torch.float32, device=ctx.dev)
    _, uv, j = ctx.geo.trajectories(x0[None], x1[None], al, sd.betas, sd.K)
    crop = SB.crop_from(uv)
    hand_box = SB.crop_from(uv[:, -1:], margin=0)
    cap = SB.capsule_mask(j[0, -1], sd.K, ctx.geo.grid, mcp)
    if events is None:
        ev, ms = sd.events(m_prev, m1)
    else:
        ev, ms = events
    return SB.make_window(ev, ms, m_prev, m1, crop, ctx.geo.grid, cap, hand_box, ctx.dev)


def retime(ev_ms, src_prev: int, dst_prev: int, dst_m1: int):
    """Shift the events of another window onto (dst_prev, dst_m1] (wrong-content control)."""
    ev, ms = ev_ms
    ms2 = ms - src_prev + dst_prev
    keep = (ms2 > dst_prev) & (ms2 <= dst_m1)
    return ev[keep], ms2[keep]


# ------------------------------------------------------------------------------------------- XDI perturbations
PERTURB = {"clean": ("none", 0)}
for _s in (1, -1):
    PERTURB[f"z{50 * _s:+d}"] = ("t", (2, .05 * _s))
    PERTURB[f"x{20 * _s:+d}"] = ("t", (0, .02 * _s))
    PERTURB[f"y{20 * _s:+d}"] = ("t", (1, .02 * _s))
    for _a in range(3):
        PERTURB[f"r{'xyz'[_a]}{10 * _s:+d}"] = ("root", (_a, 10 * _s))
        PERTURB[f"r{'xyz'[_a]}{20 * _s:+d}"] = ("root", (_a, 20 * _s))
    PERTURB[f"f{20 * _s:+d}"] = ("finger", 20 * _s)
ROOT10 = [k for k in PERTURB if k.startswith("r") and k[2:] in ("+10", "-10")]


def so3_exp(w):
    from semkine.lie import so3_exp as E
    return E(w)


def so3_log(R):
    from semkine.lie import so3_log as L
    return L(R)


def perturb(ctx: Ctx, s: torch.Tensor, kind: str, val, joint: int, betas) -> torch.Tensor:
    """XDI's perturbations (tools/xdi/feasibility.py): translation in the camera frame; root left-multiplied in the
    camera frame with the translation compensated so that the FK wrist stays; finger = joint `joint`, local x
    axis, right-multiplied on the full local rotation."""
    s = s.clone().view(1, 51)
    if kind == "t":
        s[0, val[0]] += val[1]
    elif kind == "root":
        w = torch.zeros(3, device=s.device)
        w[val[0]] = math.radians(val[1])
        w0 = ctx.wrist(s, betas)
        s[0, 3:6] = so3_log(so3_exp(w)[None] @ so3_exp(s[:, 3:6]))[0]
        s[0, :3] += (w0 - ctx.wrist(s, betas))[0]
    elif kind == "finger":
        full = s[0, 6 + 3 * joint:9 + 3 * joint] + ctx.mean[joint]
        w = torch.tensor([math.radians(val), 0., 0.], device=s.device)
        s[0, 6 + 3 * joint:9 + 3 * joint] = so3_log(so3_exp(full)[None] @ so3_exp(w)[None])[0] - ctx.mean[joint]
    return s[0]


def geo_deg(r_a: torch.Tensor, r_b: torch.Tensor) -> float:
    R = so3_exp(r_a.view(1, 3))[0].T @ so3_exp(r_b.view(1, 3))[0]
    return float(torch.rad2deg(so3_log(R[None])[0].norm()))


# ------------------------------------------------------------------------------------------- packets
def bank():
    return json.loads(BANK.read_text())


def bank_rows(b, seq: str):
    return b["sequences"][seq]["rows"]


def donor_row(ctx: Ctx, sd: SeqData, rows, index: int, betas, min_mm: float = 40.0, min_deg: float = 15.0):
    """First row from (index + n/2) onward whose GT end differs from the target's by >= 40 mm (wrist) or
    >= 15 deg (root); None if none exists."""
    n = len(rows)
    tgt = sd.gt(rows[index]["end_ms"])
    wt = ctx.wrist(tgt, betas)[0]
    for k in range(n):
        r = rows[(index + n // 2 + k) % n]
        g = sd.gt(r["end_ms"])
        if float((ctx.wrist(g, betas)[0] - wt).norm()) * 1000 >= min_mm or geo_deg(tgt[3:6], g[3:6]) >= min_deg:
            return r
    return None


def endpoint_rows(sd: SeqData):
    """XCL endpoint grid: (run_id, end, previous_ms, is_first)."""
    return C.endpoint_grid(sd.runs)


def score_ra(ctx: Ctx, preds: np.ndarray, targets: np.ndarray, sd: SeqData):
    return C.score_ra(ctx.mano, preds, targets, sd.sq.betas, ctx.dev)


def finger_prior(dev, path: Path = OUT / "priors" / "finger_prior.json"):
    """Gaussian of the 45 finger parameters over the GT of the training base sequences (every 10th ms), shrunk
    10% to the diagonal; computed once and cached."""
    if not path.exists():
        X = []
        for s, d in EF.training_base_seqs():
            sq = EF.Seq(s, d)
            X.append(np.asarray(sq.pos51[::10, 6:], np.float64))
        X = np.concatenate(X)
        mu = X.mean(0)
        cov = np.cov(X.T)
        cov = 0.9 * cov + 0.1 * np.diag(np.diag(cov))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"mean": mu.tolist(), "cov": cov.tolist(), "n": int(len(X)),
                                    "source": "training base sequences, GT every 10 ms"}))
    d = json.loads(path.read_text())
    mu = torch.tensor(d["mean"], dtype=torch.float32, device=dev)
    prec = torch.linalg.inv(torch.tensor(d["cov"], dtype=torch.float64)).float().to(dev)
    return mu, prec
