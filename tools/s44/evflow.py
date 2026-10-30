#!/usr/bin/env python3
"""S44 EvTree phase 0: pose-agnostic event motion evidence read through the MANO Jacobian.

docs/EXPERIMENT_RECORDS.md#s44-prereg-20261019 section 1 defines every quantity here; this module is the one
implementation the phase-0 probes (tools/s44/p*.py) share. Nothing in the production model imports it.

    writer   SAE: per (pixel, polarity) the time (us, `ms * 1000 + tsub`) of the latest event of a window of the
             stream. It is a max over the events, so any chunking of the stream gives the same surface.
    fit      plane fit of the same-polarity SAE around a measurement pixel -> normal flow: direction n (unit,
             image x right / y down, pointing along the motion) and speed s (px/s); constraint n^T udot = s.
    owner    oracle: the XAG GT source (face, barycentric weights) of the pixel's latest event; deployable: the
             <= 4 nearest visible projected vertices of a state's mesh within 8 px, softmax weights and a dustbin.
    jacobian d(projected vertex)/d(51-D additive state), central differences in float64.
    solve    posterior mean of the state rate under the normal-flow constraints, robust (Huber IRLS), with a
             Gaussian prior on the rate (covariance of 50 ms GT increments on the training sequences).
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "model")):
    if p not in sys.path:
        sys.path.insert(0, p)

from mano_layer import ManoLayer                                     # noqa: E402
from pose_repr import decode_to_mano_inputs                          # noqa: E402
from semkine import eval_track as ET                                 # noqa: E402
from semkine.dataset import sequences_for_split                      # noqa: E402
from semkine.kinematic_gate import support_matrix                    # noqa: E402

H, W = 180, 240
STEP = 50
RENDER_SCALE = 0.375
DATA_ROOT = Path("/data1/lyq/code/mesh/EventHands/data/hand_data51")
MANIFEST = DATA_ROOT / "splits_semkine.json"
SOURCE_ROOT = REPO / "outputs" / "xag" / "sources"
MANO_NPZ = REPO / "assets" / "mano_right.npz"
ROW_BITS = 25                    # SAE key = (t_rel_us << ROW_BITS) | row_in_window
W_MEM = 300                      # ms of stream the SAE of a query holds (clipped to the segment start)
FINGER = slice(6, 51)

#: the fixed (non-selected) constants of section 1
FIT_MIN_NB, FIT_MAX_COND = 5, 1e4
SPEED_MIN, SPEED_MAX = 2.0, 2000.0
EPS_PX_GRID = (0.5, 1.0, 2.0)    # deviation 1: the spatial-residual cap is selected on the training subjects
OWN_K, OWN_R, OWN_SIGMA, VIS_TOL = 4, 8.0, 3.0, 0.010
SIGMA0, HUBER_C, IRLS_ITERS, FD_EPS = 2.0, 2.0, 3, 1e-4


# ------------------------------------------------------------------------------------------------ data
class Seq:
    """One recording: events (mmap), offsets, sub-ms times, aux, GT per ms, valid runs, betas, K."""

    def __init__(self, name: str, d: str, root: Path = DATA_ROOT):
        ev, off, aux, pos = ET.load_sequence(root, d, name)
        self.name, self.d = name, d
        self.events, self.offsets, self.pos51 = ev, off, pos
        self.tsub = np.load(root / d / f"{name}_tsub.npy", mmap_mode="r")
        self.runs = np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2)
        self.betas = np.asarray(aux["betas"], np.float64).reshape(-1)
        self.K = np.asarray(aux["camera_K"], np.float64).reshape(3, 3)

    def steps(self):
        """[(run_id, run_start, end)] of the protocol's evaluated steps (ends a+49, a+99, ... < b)."""
        out = []
        for rid, (a, b) in enumerate(self.runs):
            out += [(rid, int(a), int(e)) for e in np.arange(a + STEP - 1, b, STEP)]
        return out

    def intrinsics(self):
        """(fx, fy, cx, cy) of the 240 x 180 event / render grid (camera K scaled by 0.375, as `_intrinsics`)."""
        K = self.K * RENDER_SCALE
        return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def split_seqs(split: str):
    return sequences_for_split(DATA_ROOT, split, MANIFEST)


def training_base_seqs():
    """The 18 base sequences ({subj}_local, {subj}_global) of the nine training subjects."""
    return [(s, d) for s, d in split_seqs("train") if not s.split("_")[-1].startswith("v")]


# ------------------------------------------------------------------------------------------------ writer
def window_rows(seq: Seq, lo_ms: int, hi_ms: int):
    a0, a1 = int(seq.offsets[lo_ms]), int(seq.offsets[hi_ms + 1])
    return a0, a1


def event_times_us(seq: Seq, lo_ms: int, hi_ms: int) -> np.ndarray:
    """Absolute event times (us) of the events of ms [lo_ms, hi_ms], in row order."""
    a0, a1 = window_rows(seq, lo_ms, hi_ms)
    counts = np.diff(seq.offsets[lo_ms:hi_ms + 2]).astype(np.int64)
    ms = np.repeat(np.arange(lo_ms, hi_ms + 1, dtype=np.int64), counts)
    return ms * 1000 + np.asarray(seq.tsub[a0:a1]).astype(np.int64)


def build_sae(seq: Seq, lo_ms: int, hi_ms: int, device, perm: Optional[np.ndarray] = None,
              chunks: Optional[list] = None):
    """SAE of the events of ms [lo_ms, hi_ms]: `(T, R)` with T (2, H, W) int64 time (us, absolute) of the latest
    event per (polarity, pixel) or -1, R (2, H, W) int64 its global event row or -1. `perm` (optional): a
    permutation of the window's timestamps (the time-shuffle control: pixels, polarities and the multiset of times
    unchanged). `chunks` (optional, test only): split points of the window's rows; the surface is built by
    successive max-updates over the chunks (prefix equivalence)."""
    a0, a1 = window_rows(seq, lo_ms, hi_ms)
    key = torch.full((2 * H * W,), -1, dtype=torch.int64, device=device)
    n = a1 - a0
    if n > 0:
        if n >= (1 << ROW_BITS):
            raise ValueError("window holds too many events for the SAE key")
        ev = np.asarray(seq.events[a0:a1])
        t = event_times_us(seq, lo_ms, hi_ms)
        if perm is not None:
            t = t[perm]
        base = lo_ms * 1000
        x = torch.from_numpy(ev[:, 0].astype(np.int64)).to(device)
        y = torch.from_numpy(ev[:, 1].astype(np.int64)).to(device)
        p = torch.from_numpy(np.clip(ev[:, 2], 0, 1).astype(np.int64)).to(device)
        idx = (p * H + y) * W + x
        k = (torch.from_numpy(t - base).to(device) << ROW_BITS) | torch.arange(n, device=device)
        cuts = [0] + list(chunks or []) + [n]
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            if c1 > c0:
                key.scatter_reduce_(0, idx[c0:c1], k[c0:c1], reduce="amax", include_self=True)
    has = key >= 0
    T = torch.where(has, (key >> ROW_BITS) + lo_ms * 1000, torch.full_like(key, -1)).view(2, H, W)
    R = torch.where(has, (key & ((1 << ROW_BITS) - 1)) + a0, torch.full_like(key, -1)).view(2, H, W)
    return T, R


def measurement_pixels(T: torch.Tensor, end: int, hop: int = STEP):
    """(pol, y, x) of the (polarity, pixel) entries whose latest event falls in ms (end - hop, end]."""
    sel = T >= (end - hop + 1) * 1000
    pol, y, x = sel.nonzero(as_tuple=True)
    return pol, y, x


def plane_fit(T: torch.Tensor, pol, y, x, r: int, dfit_us: float):
    """Normal flow at the measurement pixels. Returns dict: valid (P,) bool (neighbours, conditioning, speed range;
    NOT the residual cap, which the caller applies to `eps_px`), n (P, 2) unit (x, y), s (P,) px/s, eps_px (P,),
    nb (P,) neighbour count."""
    dev = T.device
    o = torch.arange(-r, r + 1, device=dev)
    dy, dx = torch.meshgrid(o, o, indexing="ij")
    dy, dx = dy.reshape(-1), dx.reshape(-1)
    yy, xx = y[:, None] + dy[None], x[:, None] + dx[None]
    inb = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    Tn = T[pol[:, None].expand_as(yy), yy.clamp(0, H - 1), xx.clamp(0, W - 1)]
    Tc = T[pol, y, x]
    dt = (Tn - Tc[:, None]).to(torch.float64)
    m = (inb & (Tn >= 0) & (dt.abs() <= dfit_us)).to(torch.float64)
    fx, fy = dx.to(torch.float64)[None], dy.to(torch.float64)[None]
    N = m.sum(1)
    Sx, Sy = (m * fx).sum(1), (m * fy).sum(1)
    Sxx, Sxy, Syy = (m * fx * fx).sum(1), (m * fx * fy).sum(1), (m * fy * fy).sum(1)
    St, Stx, Sty = (m * dt).sum(1), (m * dt * fx).sum(1), (m * dt * fy).sum(1)
    Nc = N.clamp(min=1.0)
    cxx, cxy, cyy = Sxx / Nc - (Sx / Nc) ** 2, Sxy / Nc - (Sx / Nc) * (Sy / Nc), Syy / Nc - (Sy / Nc) ** 2
    tr, det = cxx + cyy, cxx * cyy - cxy * cxy
    disc = (tr * tr / 4 - det).clamp(min=0).sqrt()
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    ok = (N >= FIT_MIN_NB) & (l2 > 1e-9) & (l1 <= FIT_MAX_COND * l2.clamp(min=1e-12))
    M = torch.stack([torch.stack([Sxx, Sxy, Sx], -1), torch.stack([Sxy, Syy, Sy], -1),
                     torch.stack([Sx, Sy, N], -1)], -2)
    eye = torch.eye(3, dtype=torch.float64, device=dev).expand_as(M)
    M = torch.where(ok[:, None, None], M, eye)
    rhs = torch.stack([Stx, Sty, St], -1)
    sol = torch.linalg.solve(M, rhs)
    a, b, d = sol.unbind(-1)
    res = dt - (a[:, None] * fx + b[:, None] * fy + d[:, None])
    rms = ((m * res * res).sum(1) / Nc).sqrt()
    g = torch.stack([a, b], -1)                                            # us / px
    gn = g.norm(dim=-1).clamp(min=1e-12)
    s = 1e6 / gn                                                           # px / s
    eps_px = rms / gn
    valid = ok & (s >= SPEED_MIN) & (s <= SPEED_MAX)          # the residual cap is applied by the caller
    return {"valid": valid, "n": g / gn[:, None], "s": s, "eps_px": eps_px, "nb": N}


# ------------------------------------------------------------------------------------------------ MANO
class Mano64:
    """Float64 MANO with the evaluator's decoding; FK, projection and the projected-vertex Jacobian."""

    def __init__(self, device):
        self.device = device
        self.m = ManoLayer(str(MANO_NPZ), add_mean=False, dtype=torch.float64).to(device).eval()
        self.faces = self.m.f.to(device)
        self.support = support_matrix(self.m).to(device)                    # (778, 16): palm, finger subtrees
        self.dominant = self.m.weights.argmax(dim=1)

    def fk(self, params: torch.Tensor, betas: torch.Tensor):
        """params (B, 51) float64 -> verts (B, 778, 3), joints (B, 21, 3)."""
        dec = decode_to_mano_inputs(params, "mano_full_axis_angle", self.m.hands_components, self.m.hands_mean)
        return self.m(betas.expand(len(params), -1), dec["global_orient"], dec["local_full_aa"], dec["transl"])

    def fk_hand(self, params: torch.Tensor, betas: torch.Tensor):
        """Joints with translation and root rotation set to zero (the fingers' articulation only)."""
        p = params.clone()
        p[:, :6] = 0
        return self.fk(p, betas)[1]

    def fk_chunked(self, params: torch.Tensor, betas: torch.Tensor, chunk: int = 4096):
        vs, js = [], []
        for i in range(0, len(params), chunk):
            v, j = self.fk(params[i:i + chunk], betas)
            vs.append(v)
            js.append(j)
        return torch.cat(vs), torch.cat(js)

    @staticmethod
    def project(verts: torch.Tensor, intr):
        fx, fy, cx, cy = intr
        z = verts[..., 2].clamp(min=1e-6)
        return torch.stack([fx * verts[..., 0] / z + cx, fy * verts[..., 1] / z + cy], -1), verts[..., 2]

    def jacobian(self, params: torch.Tensor, betas: torch.Tensor, intr):
        """params (B, 51) -> (verts (B, 778, 3), uv (B, 778, 2), z (B, 778), J (B, 778, 2, 51)): d uv / d params."""
        B = params.shape[0]
        eye = torch.eye(51, dtype=torch.float64, device=params.device) * FD_EPS
        P = torch.cat([params[:, None], params[:, None] + eye[None], params[:, None] - eye[None]], 1)  # (B, 103, 51)
        v, _ = self.fk_chunked(P.reshape(-1, 51), betas)
        v = v.view(B, 103, 778, 3)
        v0 = v[:, 0]
        dX = (v[:, 1:52] - v[:, 52:103]) / (2 * FD_EPS)                       # (B, 51, 778, 3)
        fx, fy, _, _ = intr
        X, Y, Z = v0[..., 0], v0[..., 1], v0[..., 2].clamp(min=1e-6)
        du = fx / Z[:, None] * dX[..., 0] - fx * X[:, None] / Z[:, None] ** 2 * dX[..., 2]
        dv = fy / Z[:, None] * dX[..., 1] - fy * Y[:, None] / Z[:, None] ** 2 * dX[..., 2]
        J = torch.stack([du, dv], -1).permute(0, 2, 3, 1).contiguous()          # (B, 778, 2, 51)
        uv, z = self.project(v0, intr)
        return v0, uv, z, J


def visible(uv: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """(V,) bool: vertex depth within VIS_TOL of the 3 x 3 vertex-splat z-buffer at its pixel."""
    dev = uv.device
    ui, vi = uv[:, 0].round().long(), uv[:, 1].round().long()
    zb = torch.full((H * W,), 1e9, dtype=z.dtype, device=dev)
    for dyy in (-1, 0, 1):
        for dxx in (-1, 0, 1):
            uu, vv = ui + dxx, vi + dyy
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H) & (z > 0)
            zb.scatter_reduce_(0, (vv * W + uu)[ok], z[ok], reduce="amin", include_self=True)
    inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (z > 0)
    zz = zb[(vi.clamp(0, H - 1) * W + ui.clamp(0, W - 1))]
    return inb & (z <= zz + VIS_TOL)


def owners_deploy(px: torch.Tensor, uv: torch.Tensor, vis: torch.Tensor):
    """px (P, 2) pixel (x, y); uv (V, 2) projected vertices of one state; vis (V,) bool.
    Returns vertex ids (P, k) long and weights (P, k) (sum <= 1; the rest is the dustbin)."""
    vid = vis.nonzero().flatten()
    P = px.shape[0]
    if len(vid) == 0 or P == 0:
        return torch.zeros(P, OWN_K, dtype=torch.long, device=px.device), torch.zeros(P, OWN_K, dtype=uv.dtype,
                                                                                       device=px.device)
    d2 = torch.cdist(px.to(uv.dtype), uv[vid]).square()
    k = min(OWN_K, len(vid))
    dk, ik = d2.topk(k, dim=1, largest=False)
    logit = -dk / (2 * OWN_SIGMA ** 2)
    logit = torch.where(dk <= OWN_R ** 2, logit, torch.full_like(logit, -1e9))
    dust = torch.full((P, 1), -OWN_R ** 2 / (2 * OWN_SIGMA ** 2), dtype=uv.dtype, device=px.device)
    w = torch.softmax(torch.cat([logit, dust], 1), 1)[:, :k]
    ids = vid[ik]
    if k < OWN_K:
        pad = OWN_K - k
        ids = torch.cat([ids, ids[:, :1].expand(-1, pad)], 1)
        w = torch.cat([w, torch.zeros(P, pad, dtype=w.dtype, device=w.device)], 1)
    return ids, w


@lru_cache(maxsize=None)
def _source_arrays(seq: str):
    base = SOURCE_ROOT / seq
    return {k: np.load(base / f"{k}.npy", mmap_mode="r") for k in ("face", "bary", "part", "done_ms")}


def owners_oracle(seq: Seq, rows: np.ndarray, faces: torch.Tensor):
    """GT source labels of event rows -> vertex ids (P, 3) long and weights (P, 3) (0 rows: dustbin)."""
    a = _source_arrays(seq.name)
    ms = np.searchsorted(seq.offsets, rows, side="right") - 1
    if not np.all(a["done_ms"][ms]):
        raise RuntimeError(f"XAG source cache incomplete: {seq.name}")
    face = np.asarray(a["face"][rows]).astype(np.int64)
    bary = np.asarray(a["bary"][rows]).astype(np.float64)
    part = np.asarray(a["part"][rows]).astype(np.int64)
    ok = (face >= 0) & (part >= 0)
    face_t = torch.from_numpy(np.where(ok, face, 0)).to(faces.device)
    ids = faces[face_t]                                                       # (P, 3)
    w = torch.from_numpy(np.where(ok[:, None], bary, 0.0)).to(faces.device)
    return ids, w, torch.from_numpy(ok).to(faces.device)


# ------------------------------------------------------------------------------------------------ solve
def rows_A(J: torch.Tensor, n: torch.Tensor, ids: torch.Tensor, w: torch.Tensor):
    """Constraint rows A (P, 51) = n^T sum_v w_v J_v and the owner mass (P,). J (778, 2, 51) of one state."""
    Jv = J[ids]                                                               # (P, k, 2, 51)
    Jw = (w[..., None, None] * Jv).sum(1)                                     # (P, 2, 51)
    A = (n[:, :, None] * Jw).sum(1)                                           # (P, 51)
    return A, w.sum(1)


def solve_rate(A: torch.Tensor, s: torch.Tensor, mass: torch.Tensor, kappa: float, lam_p: float,
               prior_prec: torch.Tensor):
    """Posterior mean of the rate (51,) per second; the data information Lam (51, 51) at the final IRLS weights."""
    sig = SIGMA0 + kappa * s
    base = mass / sig ** 2
    wt = base
    for it in range(IRLS_ITERS + 1):
        Aw = A * wt[:, None]
        Lam = Aw.T @ A
        eta = Aw.T @ s
        th = torch.linalg.solve(Lam + lam_p * prior_prec, eta)
        if it == IRLS_ITERS:
            break
        rr = ((A @ th - s) / sig).abs()
        wt = base * torch.clamp(HUBER_C / rr.clamp(min=1e-12), max=1.0)
    return th, Lam


# ------------------------------------------------------------------------------------------------ prior
def rate_prior(device, path: Path = REPO / "outputs" / "s44" / "prior_rate51.json"):
    """Precision of the 51-D state rate (1/s units): inverse of the shrunk covariance of all 50 ms GT increments of
    the 72 training sequences (cached in `path`)."""
    if path.exists():
        d = json.loads(path.read_text())
        return torch.tensor(d["precision"], dtype=torch.float64, device=device), d
    incs = []
    for s, dd in split_seqs("train"):
        sq = Seq(s, dd)
        for a, b in sq.runs:
            e = np.arange(a + STEP, b, STEP)
            if len(e):
                incs.append((sq.pos51[e] - sq.pos51[e - STEP]).astype(np.float64) / (STEP * 1e-3))
    X = np.concatenate(incs)
    C = np.cov(X, rowvar=False)
    Cs = 0.9 * C + 0.1 * np.diag(np.diag(C))
    prec = np.linalg.inv(Cs)
    d = {"n_increments": int(len(X)), "shrink": 0.1, "rate_std": np.sqrt(np.diag(C)).tolist(),
         "covariance": Cs.tolist(), "precision": prec.tolist(), "sequences": len(split_seqs("train"))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d))
    return torch.tensor(prec, dtype=torch.float64, device=device), d


# ------------------------------------------------------------------------------------------------ one step
def step_measurements(seq: Seq, run_start: int, end: int, r: int, dfit_ms: float, device,
                      perm_rng: Optional[np.random.Generator] = None, mem_ms: int = W_MEM):
    """SAE of [max(end - mem_ms + 1, run_start), end] and the normal flow of this step's measurement pixels
    (`mem_ms` default W_MEM = 300, the preregistered memory; the P0b round also uses 600)."""
    lo = max(end - int(mem_ms) + 1, run_start)
    perm = None
    if perm_rng is not None:
        a0, a1 = window_rows(seq, lo, end)
        perm = perm_rng.permutation(a1 - a0)
    T, R = build_sae(seq, lo, end, device, perm=perm)
    pol, y, x = measurement_pixels(T, end)
    fit = plane_fit(T, pol, y, x, r, dfit_ms * 1000.0)
    v = fit["valid"]
    return {"px": torch.stack([x[v], y[v]], -1).to(torch.float64), "n": fit["n"][v], "s": fit["s"][v],
            "rows": R[pol[v], y[v], x[v]], "n_meas": int(len(pol)), "n_valid": int(v.sum()),
            "eps_px": fit["eps_px"][v]}
