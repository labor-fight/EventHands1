#!/usr/bin/env python3
"""M6 KSCS . GSCD: kinematic-subspace correlation spectrum and grid-sharing correlation decomposition
(diagnostic, CPU, no GPU arm; draft pre-registration `tools/mv2/prereg_m6.md`, plan section M6, R3 entries P5-E5-R1 and
P1-E7-R1).

Question: the KS2 residual correlation spectrum of the frozen K7 patch head (hop1 0.838 / hop2 0.598 / hop3 0.247 /
hop4 -0.117, `docs/EXPERIMENT_RECORDS.md#kstar-results-20261025`) -- is it the image of a kinematic (FK / LBS Jacobian)
error subspace, and how much of it is shared sampling windows? Nothing is solved, fused or whitened.

Three parts, one `decide()`:
  PE1  no data.  The MANO vertex Jacobian J = dv/dxi (Lie right tangent, `semkine/jacobian.py`) at FIT7 GT poses
       (labels, not effects), projected on the image-plane vertex normal e1 (the KS2 `R[..., :, 0]` frame): row_i =
       e1_i^T (d pi / d X)_i dX_i/dxi (px per unit tangent).  A joint-error prior delta ~ N(0, diag sigma^2) gives the
       residual r = J delta; the hop-h pooled Pearson of KS2's estimator has the closed form
       sum_k s_k^2 cov_h[k] / sqrt(sum_k s_k^2 vx_h[k] * sum_k s_k^2 vy_h[k]) (independent DOFs, zero mean), so no Monte
       Carlo noise.  Null models: graph-Laplacian heat-kernel low-pass fields (scalar, and a 2-D vector field projected on
       e1) with the hop-1 correlation matched to the ledger, and isotropic per-vertex noise.
  PE2  the real residuals of the frozen patch head (`outputs/xvm/s2/p4/A_m600/heads.pt`, `patch_r2`) on SELECT, recomputed
       read-only through `tools/kstar/head_probe.py`'s own classes (KS2 stored only moment accumulators, no per-vertex
       residual): reproduces the ledger spectrum (tool check), then the variance share inside span(J) (<= 51 dims at each
       sample's own anchor pose, rank cut), with a vertex-permuted chance null, a wrong-pose null and a simulated positive
       control.
  GSCD the head's patch window (16 x 16 rotated-square bilinear taps, half-size rho = f 0.02 m / z, on the 240 x 180
       grid, source-area gains of `semkine/sba.py::make_grid`) gives a shot-noise sharing s_ij = g-weighted cosine of the
       two windows' pixel-weight maps; within a hop, tertiles of s_ij; top-minus-bottom pooled residual correlation, a
       within-segment shuffle null that keeps the hop distribution, a within-(sample, hop) rank-tertile and a
       distance-matched variant, and a synthetic window-noise positive control.

    python tools/mv2/m6_kscs.py --smoke
    python tools/mv2/m6_kscs.py --out outputs/mv2/m6            # formal (needs the freeze)
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEED = 20261026
HOPS = (1, 2, 3, 4)
GATE_HOPS = (1, 2, 3)           # KSCS "hop1-3 max |delta|"
GSCD_HOPS = (1, 2, 3)
LEDGER_JSON = REPO / "outputs" / "kstar" / "s2_stale" / "result.json"
LEDGER_TEXT_LOCAL = (0.838, 0.598, 0.247, -0.117)       # docs/EXPERIMENT_RECORDS.md#kstar-results-20261025
LEDGER_N_SAMPLES = 10140
HEAD_R = 0.02                   # patch_r2: 2 cm half-window
PATCH = 16
CONTOUR_COS = 0.4               # semkine.xvm.CONTOUR_COS
RANK_CUT = 1e-2                 # relative singular-value cut of sigma-weighted J (span rank)
RANK_CUT_ALT = 1e-3
PE1_STRIDE = 2                  # FIT7: every 2nd P4-grid endpoint of each base sequence (~9k poses)
SEL_STRIDE = 4                  # SELECT: KS2's stride
N_PERM = 200
N_BOOT = 1000
NEAR_ID_SHIFT = 97              # wrong-pose partner: sample 97 places later in the same sequence
N_PERM_SPAN = 4
TERTILE_MIN_SEG = 9
DIST_BINS = 5
POS_NOISE_ENERGY = 0.1
# gate constants written in the plan / ledger entries (kept literal; FIT7-computed ones are derived in `decide`)
D_PASS, D_VOID, D_PERM_MAX = 0.15, 0.05, 0.05
SHARE_PASS, SHARE_KILL = 0.6, 0.3
CHANCE_VACUOUS = 0.5
SHARE_POS_MIN = 0.85
PERM_SHARE_TOL = 0.03
REPRO_R_TOL = 0.01
REPRO_N_REL = 1e-4
SIGMA_VARIANTS = ("J_main", "J_perdof", "J_equal_effect", "J_fingers")
NULL_MODELS = ("lap_scalar", "lap_vector", "iid")


# ======================================================================================================== deps / mesh
_D: dict = {}


def deps():
    """Heavy read-only imports, loaded on first use."""
    if not _D:
        from s44 import evflow as EF
        from xvm import common as XC
        from semkine import xvm as XV
        from semkine import jacobian as JA
        from semkine import lie
        from semkine.mrf import hop_matrix
        from semkine.xyz_mesh import vertex_normals
        from semkine.sba import make_grid
        _D.update(EF=EF, XC=XC, XV=XV, JA=JA, lie=lie, hop_matrix=hop_matrix, vertex_normals=vertex_normals,
                  make_grid=make_grid)
    return _D


_T: dict = {}


def tables():
    """MANO float64 layer, faces, hop matrix (<= 4), pair lists (i < j, hop 1-4), upper-triangular hop adjacency
    (scipy csr per hop), graph Laplacian, source-area gains."""
    if not _T:
        import scipy.sparse as sp
        d = deps()
        mano = d["EF"].Mano64(torch.device("cpu"))
        faces = mano.faces.cpu()
        D = d["hop_matrix"](faces, 778, 4)
        PI, PJ, PH = [], [], []
        A = {}
        for h in HOPS:
            i, j = torch.nonzero(torch.triu(D == h, diagonal=1), as_tuple=True)
            i, j = i.numpy(), j.numpy()
            PI.append(i), PJ.append(j), PH.append(np.full(len(i), h, np.int8))
            A[h] = sp.csr_matrix((np.ones(len(i)), (i, j)), shape=(778, 778))
        PI, PJ, PH = np.concatenate(PI), np.concatenate(PJ), np.concatenate(PH)
        A1 = (D == 1).double().numpy()
        L = np.diag(A1.sum(1)) - A1
        gain = d["make_grid"](torch.device("cpu")).gain.double().numpy().reshape(-1)
        _T.update(mano=mano, faces=faces, D=D.numpy().astype(np.int8), PI=PI, PJ=PJ, PH=PH, A=A, L=L, gain=gain)
    return _T


# ======================================================================================================== geometry / J
@torch.no_grad()
def geometry(mano, x51: torch.Tensor, betas1: torch.Tensor, K: torch.Tensor) -> dict:
    """`tools/xvm/p4_train.py::Geo.__call__` (arm A) geometry at anchor states `x51` (B, 51) f64 without any event:
    V0, uv0, z0, R (columns e1, e2), cos, vis, Jp.  `betas1` (1, 10), `K` (3, 3) already x 0.375."""
    d = deps()
    B = x51.shape[0]
    V0, _ = mano.fk_chunked(x51, betas1)
    fx = torch.full((B,), float(K[0, 0]), dtype=torch.float64)
    fy = torch.full((B,), float(K[1, 1]), dtype=torch.float64)
    cx, cy = float(K[0, 2]), float(K[1, 2])
    z0 = V0[..., 2]
    zc = z0.clamp(min=1e-6)
    uv0 = torch.stack([fx[:, None] * V0[..., 0] / zc + cx, fy[:, None] * V0[..., 1] / zc + cy], -1)
    normals = d["vertex_normals"](V0.float(), mano.faces).double()
    Jp = d["XV"].projection_jacobian(V0, fx, fy)
    R, cos = d["XV"].vertex_frames(V0, normals, Jp)
    vis = torch.stack([d["EF"].visible(uv0[b], z0[b]) for b in range(B)])
    return {"V0": V0, "uv0": uv0, "z0": z0, "R": R, "cos": cos, "vis": vis, "Jp": Jp}


@torch.no_grad()
def tangent_jacobian(mano, x51: torch.Tensor, betas1: torch.Tensor, bidx: torch.Tensor, vidx: torch.Tensor):
    """Lie-tangent vertex Jacobian (`semkine.jacobian.vertex_jacobian`) of states `x51` (B, 51) at the flat
    (state, vertex) lists: points X (n, 3) and dX/dxi (n, 3, 51), xi = [rho(3) | phi(3) | dphi(45)]."""
    d = deps()
    m = mano.m
    B = x51.shape[0]
    b = betas1.expand(B, -1)
    j0 = (m.J_regressor @ (m.v_template + torch.einsum("bl,mkl->bmk", b, m.shapedirs)))[:, 0]
    state = d["lie"].state_from_51d(x51, m.hands_mean, j0)
    fk = d["JA"].forward_kinematics(m, state, b)
    return d["JA"].vertex_jacobian(m, fk, bidx, vidx)


def proj_jac_flat(X: torch.Tensor, fx: float, fy: float) -> torch.Tensor:
    """d pi / d X (n, 2, 3) at camera-frame points X (n, 3)."""
    Z = X[:, 2].clamp(min=1e-6)
    J = X.new_zeros(X.shape[0], 2, 3)
    J[:, 0, 0] = fx / Z
    J[:, 0, 2] = -fx * X[:, 0] / Z ** 2
    J[:, 1, 1] = fy / Z
    J[:, 1, 2] = -fy * X[:, 1] / Z ** 2
    return J


def normal_rows(X: torch.Tensor, dX: torch.Tensor, e1: torch.Tensor, K) -> torch.Tensor:
    """e1^T (d pi / d X) dX/dxi: (n, 51) px per tangent unit.  X (n, 3), dX (n, 3, 51), e1 (n, 2)."""
    Jp = proj_jac_flat(X, float(K[0, 0]), float(K[1, 1]))
    Juv = Jp @ dX
    return (e1.unsqueeze(-1) * Juv).sum(1)


def part_index() -> np.ndarray:
    """(51,) part id: 0 translation, 1 root rotation, 2..16 the 15 finger joints (3 DOF each)."""
    idx = np.zeros(51, np.int64)
    idx[3:6] = 1
    for j in range(15):
        idx[6 + 3 * j: 9 + 3 * j] = 2 + j
    return idx


# ======================================================================================================== estimators
def moments(x, y) -> np.ndarray:
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    return np.array([len(x), x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()])


def r_from_moments(m) -> np.ndarray:
    """Pooled Pearson r of moment sums (..., 6) = [n, sx, sy, sxx, syy, sxy] (KS2's estimator, `Lin.stats`)."""
    m = np.asarray(m, np.float64)
    n = np.maximum(m[..., 0], 1.0)
    cov = m[..., 5] / n - m[..., 1] * m[..., 2] / n ** 2
    vx = np.clip(m[..., 3] / n - (m[..., 1] / n) ** 2, 1e-30, None)
    vy = np.clip(m[..., 4] / n - (m[..., 2] / n) ** 2, 1e-30, None)
    return cov / np.sqrt(vx * vy)


def binned_moments(x, y, code, ncode) -> np.ndarray:
    """(ncode, 6) moment sums of (x, y) grouped by integer `code`."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    out = np.empty((ncode, 6))
    out[:, 0] = np.bincount(code, minlength=ncode)
    out[:, 1] = np.bincount(code, weights=x, minlength=ncode)
    out[:, 2] = np.bincount(code, weights=y, minlength=ncode)
    out[:, 3] = np.bincount(code, weights=x * x, minlength=ncode)
    out[:, 4] = np.bincount(code, weights=y * y, minlength=ncode)
    out[:, 5] = np.bincount(code, weights=x * y, minlength=ncode)
    return out


def boot_weights(n_groups: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """(n_boot, G) multinomial group-resampling counts."""
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
                     for _ in range(n_boot)]).astype(np.float64)


def pct(v, q=(2.5, 97.5)):
    return [float(np.nanpercentile(v, q[0])), float(np.nanpercentile(v, q[1]))]


class HopAcc:
    """Pooled hop-h residual-pair moments per bootstrap group, KS2's estimator (pairs i < j, both vertices in the sample
    mask, x = res_i, y = res_j)."""

    def __init__(self, PI, PJ, PH):
        self.sel = {h: (PI[PH == h], PJ[PH == h]) for h in HOPS}
        self.g = {}

    def add(self, key, res_dense: np.ndarray, mask: np.ndarray, group: int):
        for h, (i, j) in self.sel.items():
            ok = mask[i] & mask[j]
            if ok.any():
                m = moments(res_dense[i[ok]], res_dense[j[ok]])
                d = self.g.setdefault((key, h), {})
                d[group] = d.get(group, 0) + m

    def result(self, key, n_boot=N_BOOT, seed=SEED) -> dict:
        r, n, ci = [], [], []
        groups = sorted({g for h in HOPS for g in self.g.get((key, h), {})})
        gi = {g: k for k, g in enumerate(groups)}
        W = boot_weights(max(len(groups), 1), n_boot, seed) if groups else None
        for h in HOPS:
            d = self.g.get((key, h), {})
            if not d:
                r.append(float("nan")), n.append(0), ci.append([float("nan")] * 2)
                continue
            a = np.zeros((len(groups), 6))
            for g, m in d.items():
                a[gi[g]] = m
            tot = a.sum(0)
            r.append(float(r_from_moments(tot))), n.append(int(tot[0]))
            ci.append(pct(r_from_moments(W @ a)))
        return {"r": r, "n": n, "ci": ci, "n_groups": len(groups)}


# ======================================================================================================== PE1 (no data)
def fit7_seqs():
    XC = deps()["XC"]
    return XC.subject_seqs(XC.TRAIN_SUBJ, base_only=True)


def domain_of(name: str) -> str:
    return "local" if "local" in name else "global"


def p4_grid(sq, stride: int, max_n: int = 0):
    """(end, prev, run_a, run_b) of the P4 sample grid (`tools/xvm/p4_data.py`): every stride-th endpoint of each run."""
    out, k = [], 0
    for rid, (a, b) in enumerate(sq.runs):
        for end in range(int(a) + 49, int(b), 50):
            if end - 50 < a:
                continue
            if k % stride == 0:
                out.append((end, end - 50, int(a), int(b)))
            k += 1
    return out[:max_n] if max_n else out


def smoothed_pair(sq, grid):
    """Anchor (smoothed GT at prev) and target (smoothed GT at end) states of the KS2 samples, (n, 51) f64 each."""
    XC = deps()["XC"]
    x0 = np.stack([XC.smoothed_gt(sq.pos51, p, a, b - 1) for (_, p, a, b) in grid]).astype(np.float64)
    xt = np.stack([XC.smoothed_gt(sq.pos51, e, a, b - 1) for (e, _, a, b) in grid]).astype(np.float64)
    return x0, xt


def new_pe1_acc(P):
    return {"cov": np.zeros((4, 51)), "vx": np.zeros((4, 51)), "vy": np.zeros((4, 51)), "Wc": np.zeros(P),
            "Wv": np.zeros(P), "en": np.zeros(51), "cnt": 0.0, "inc2": np.zeros(51), "ninc": 0, "n_poses": 0,
            "white_sum": 0.0, "white_n": 0, "reff_sum": 0.0}


def accumulate_jmodel(acc: dict, Jn: np.ndarray, m: np.ndarray, e1: np.ndarray, T: dict):
    """One pose: Jn (778, 51) normal-projected Jacobian, m (778,) bool mask, e1 (778, 2).  Adds the per-DOF pair sums
    that make the J-model hop correlation a closed form, and the null-model pair weights."""
    mf = m.astype(np.float64)
    Jm = Jn * mf[:, None]
    for k, h in enumerate(HOPS):
        A = T["A"][h]
        AJ = A @ Jm
        acc["cov"][k] += (AJ * Jm).sum(0)
        cx = A @ mf
        cy = A.T @ mf
        acc["vx"][k] += ((mf * cx)[:, None] * Jn ** 2).sum(0)
        acc["vy"][k] += ((mf * cy)[:, None] * Jn ** 2).sum(0)
    PI, PJ = T["PI"], T["PJ"]
    w = mf[PI] * mf[PJ]
    acc["Wc"] += w
    acc["Wv"] += w * (e1[PI] * e1[PJ]).sum(-1)
    acc["en"] += (mf[:, None] * Jn ** 2).sum(0)
    acc["cnt"] += mf.sum()
    acc["n_poses"] += 1


def pe1_sigma_pass(max_seq: int = 0, max_n: int = 0) -> dict:
    """FIT7 GT increments only (no FK): per-domain per-DOF second moments of the Lie increment between the smoothed GT at
    end - 50 and at end over the P4 grid -> J_main sigma by domain, before any Jacobian is read."""
    d = deps()
    T = tables()
    mano = T["mano"]
    acc = {}
    seqs = fit7_seqs()
    if max_seq:
        seqs = seqs[:max_seq]
    for name, dd in seqs:
        sq = d["EF"].Seq(name, dd)
        grid = p4_grid(sq, PE1_STRIDE, max_n)
        if not grid:
            continue
        x0, xt = smoothed_pair(sq, grid)
        betas1 = torch.as_tensor(sq.betas, dtype=torch.float64).view(1, -1)
        c = len(x0)
        j0 = (mano.m.J_regressor @ (mano.m.v_template + torch.einsum(
            "bl,mkl->bmk", betas1.expand(c, -1), mano.m.shapedirs)))[:, 0]
        inc = d["lie"].local_coordinates_51d(torch.as_tensor(x0), torch.as_tensor(xt), mano.m.hands_mean, j0).numpy()
        a = acc.setdefault(domain_of(name), [np.zeros(51), 0])
        a[0] += (inc ** 2).sum(0)
        a[1] += c
    return {dom: sigma_vectors(a[0], a[1], np.ones(51), 1.0)["J_main"] for dom, a in acc.items()}


def pe1_accumulate(sig_main: dict, chunk: int = 24, max_seq: int = 0, max_n: int = 0, log=None) -> dict:
    """FIT7 base sequences -> {(subject, domain): accumulator}.  Poses = KS2's anchor grid (smoothed GT at end - 50);
    mask = visible & contour (no events at PE1).  `sig_main[domain]` (from `pe1_sigma_pass`) sets the rank-cut scaling
    of the white-noise chance share r_eff / n."""
    d = deps()
    T = tables()
    mano = T["mano"]
    lie = d["lie"]
    out = {}
    seqs = fit7_seqs()
    if max_seq:
        seqs = seqs[:max_seq]
    for name, dd in seqs:
        t0 = time.time()
        sq = d["EF"].Seq(name, dd)
        grid = p4_grid(sq, PE1_STRIDE, max_n)
        if not grid:
            continue
        dom = domain_of(name)
        x0, xt = smoothed_pair(sq, grid)
        betas1 = torch.as_tensor(sq.betas, dtype=torch.float64).view(1, -1)
        K = torch.as_tensor(sq.K * d["EF"].RENDER_SCALE, dtype=torch.float64)
        key = (name.split("_")[0], dom)
        acc = out.setdefault(key, new_pe1_acc(len(T["PI"])))
        for c0 in range(0, len(x0), chunk):
            xa = torch.as_tensor(x0[c0:c0 + chunk])
            xb = torch.as_tensor(xt[c0:c0 + chunk])
            c = xa.shape[0]
            g = geometry(mano, xa, betas1, K)
            bi = torch.arange(c).repeat_interleave(778)
            vi = torch.arange(778).repeat(c)
            X, dX = tangent_jacobian(mano, xa, betas1, bi, vi)
            e1 = g["R"][..., :, 0].reshape(-1, 2)
            Jn = normal_rows(X, dX, e1, K).view(c, 778, 51).numpy()
            mask = (g["vis"] & (g["cos"] < CONTOUR_COS)).numpy()
            e1n = e1.view(c, 778, 2).numpy()
            for b in range(c):
                accumulate_jmodel(acc, Jn[b], mask[b], e1n[b], T)
                nb = int(mask[b].sum())
                if nb >= 3:
                    sv = np.linalg.svd(Jn[b][mask[b]] * sig_main[dom][None, :], compute_uv=False)
                    k = int((sv >= RANK_CUT * sv[0]).sum())
                    acc["white_sum"] += k / nb
                    acc["reff_sum"] += k
                    acc["white_n"] += 1
            j0 = (mano.m.J_regressor @ (mano.m.v_template + torch.einsum(
                "bl,mkl->bmk", betas1.expand(c, -1), mano.m.shapedirs)))[:, 0]
            inc = lie.local_coordinates_51d(xa, xb, mano.m.hands_mean, j0).numpy()
            acc["inc2"] += (inc ** 2).sum(0)
            acc["ninc"] += c
        if log:
            log(f"pe1 {name}: {len(x0)} poses {time.time() - t0:.1f}s")
    return out


def sigma_vectors(inc2: np.ndarray, ninc: int, en: np.ndarray, cnt: float) -> dict:
    """The four pre-declared joint-error priors (51 tangent-DOF standard deviations), from FIT7 GT increments over 50 ms
    (and FIT7 Jacobian energies for `equal_effect`):
      J_main          per-part isotropic increment RMS (translation, root rotation, 15 joints; independent parts)
      J_perdof        per-DOF increment RMS
      J_equal_effect  each DOF contributes the same mean-square pixel effect (sigma_k = 1 / RMS of its Jacobian column)
      J_fingers       J_main with translation and root rotation held (sigma = 0), the KS2 `art` target convention."""
    rms = np.sqrt(inc2 / max(ninc, 1))
    part = part_index()
    main = np.zeros(51)
    for p in range(17):
        main[part == p] = math.sqrt(float(np.mean(rms[part == p] ** 2)))
    colrms = np.sqrt(en / max(cnt, 1.0))
    eq = 1.0 / np.clip(colrms, 1e-12, None)
    eq = eq / np.sqrt(np.mean(eq ** 2)) * np.sqrt(np.mean(main ** 2))        # overall scale is irrelevant to r; keep O(1)
    fing = main.copy()
    fing[:6] = 0.0
    return {"J_main": main, "J_perdof": rms, "J_equal_effect": eq, "J_fingers": fing}


def jmodel_spectrum(cov, vx, vy, sigma) -> np.ndarray:
    """(4,) closed-form hop correlation of r = J delta, delta ~ N(0, diag sigma^2)."""
    s2 = np.asarray(sigma, np.float64) ** 2
    num = cov @ s2
    return num / np.sqrt(np.clip((vx @ s2) * (vy @ s2), 1e-300, None))


def heat_kernel(L_eig, tau: float) -> np.ndarray:
    w, V = L_eig
    return (V * np.exp(-tau * w)) @ V.T


def null_spectra(Wc, Wv, K, T) -> dict:
    """Hop spectra (4,) of the Laplacian-field nulls with covariance `K` (778, 778) under pair weights Wc (mask-pair
    counts) and Wv (mask-pair counts x e1_i . e1_j)."""
    PI, PJ, PH = T["PI"], T["PJ"], T["PH"]
    dk = np.diag(K)
    Kp, Ki, Kj = K[PI, PJ], dk[PI], dk[PJ]
    out = {"lap_scalar": [], "lap_vector": []}
    for h in HOPS:
        s = PH == h
        den = math.sqrt(max(float((Wc[s] * Ki[s]).sum()) * float((Wc[s] * Kj[s]).sum()), 1e-300))
        out["lap_scalar"].append(float((Wc[s] * Kp[s]).sum()) / den)
        out["lap_vector"].append(float((Wv[s] * Kp[s]).sum()) / den)
    return {k: np.array(v) for k, v in out.items()}


def calibrate_tau(Wc, Wv, target: float, T, kind: str, L_eig=None, lo=-4.0, hi=4.0, iters=40) -> dict:
    """Bisection on log10(tau) so that the null's hop-1 pooled correlation equals `target` (the ledger's hop 1)."""
    L_eig = L_eig or np.linalg.eigh(T["L"])

    def f(lt):
        return null_spectra(Wc, Wv, heat_kernel(L_eig, 10.0 ** lt), T)[kind][0] - target
    flo, fhi = f(lo), f(hi)
    if flo > 0 or fhi < 0:
        best = lo if abs(flo) < abs(fhi) else hi
        return {"tau": 10.0 ** best, "calibrated": False}
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return {"tau": 10.0 ** (0.5 * (lo + hi)), "calibrated": True}


def pe1_models(accs: dict, domain: str, ledger_h1: float, T: dict) -> dict:
    """Spectra of every model / variant per FIT7 subject and pooled, for one domain."""
    keys = sorted(k for k in accs if k[1] == domain)
    tot = new_pe1_acc(len(T["PI"]))
    for k in keys:
        for f in ("cov", "vx", "vy", "Wc", "Wv", "en", "inc2"):
            tot[f] = tot[f] + accs[k][f]
        for f in ("cnt", "ninc", "n_poses", "white_sum", "white_n", "reff_sum"):
            tot[f] += accs[k][f]
    sig = sigma_vectors(tot["inc2"], tot["ninc"], tot["en"], tot["cnt"])
    L_eig = np.linalg.eigh(T["L"])
    ws = {s: accs[(s, domain)]["white_sum"] / max(accs[(s, domain)]["white_n"], 1) for s, _ in keys}
    out = {"sigma": {k: v.tolist() for k, v in sig.items()}, "n_poses": {s: accs[(s, domain)]["n_poses"]
           for s, _ in keys}, "spectra": {}, "tau": {},
           "white_chance": {"per_subject": ws, "mean": float(np.mean(list(ws.values()))),
                            "sd": float(np.std(list(ws.values()), ddof=1)) if len(ws) > 1 else 0.0,
                            "pooled": tot["white_sum"] / max(tot["white_n"], 1),
                            "r_eff_mean": tot["reff_sum"] / max(tot["white_n"], 1)}}
    units = {"pooled": tot, **{s: accs[(s, domain)] for s, _ in keys}}
    for name in SIGMA_VARIANTS:
        out["spectra"][name] = {"pooled": None, "per_subject": {}}
        for u, a in units.items():
            sp = jmodel_spectrum(a["cov"], a["vx"], a["vy"], sig[name]).tolist()
            if u == "pooled":
                out["spectra"][name]["pooled"] = sp
            else:
                out["spectra"][name]["per_subject"][u] = sp
    for kind in ("lap_scalar", "lap_vector"):
        out["spectra"][kind] = {"pooled": None, "per_subject": {}}
        out["tau"][kind] = {}
        for u, a in units.items():
            cal = calibrate_tau(a["Wc"], a["Wv"], ledger_h1, T, kind, L_eig)
            sp = null_spectra(a["Wc"], a["Wv"], heat_kernel(L_eig, cal["tau"]), T)[kind].tolist()
            out["tau"][kind][u] = cal
            if u == "pooled":
                out["spectra"][kind]["pooled"] = sp
            else:
                out["spectra"][kind]["per_subject"][u] = sp
    out["spectra"]["iid"] = {"pooled": [0.0] * 4, "per_subject": {s: [0.0] * 4 for s, _ in keys}}
    # leave-one-FIT7-subject-out pooled J_main spectra (descriptive: the SD of these replicates is the strict reading of
    # "LOSO SD"; the gate uses the SD across the per-subject spectra, which is larger)
    out["J_main_loso"] = {s: jmodel_spectrum(tot["cov"] - accs[(s, domain)]["cov"], tot["vx"] - accs[(s, domain)]["vx"],
                                             tot["vy"] - accs[(s, domain)]["vy"], sig["J_main"]).tolist() for s, _ in keys}
    return out


# ======================================================================================================== ledger
def load_ledger() -> dict:
    """KS2 hop spectra of `outputs/kstar/s2_stale/result.json` (SELECT, all 16 ycy / ylf sequences, stride 4)."""
    import json
    rd = json.loads(LEDGER_JSON.read_text())
    out = {"n_samples": rd["meta"]["n_samples"], "stride": rd["meta"]["stride"]}
    for dom in ("local", "global"):
        keys = [f"all|{dom}|hop{h}|residual" for h in HOPS]
        out[dom] = {"r": [rd["readings"][k]["r"] for k in keys], "ci": [rd["readings"][k]["r_ci"] for k in keys],
                    "n": [rd["readings"][k]["n"] for k in keys]}
    return out


# ======================================================================================================== PE2 collect
def collect_select(stride: int = SEL_STRIDE, workers: int = 16, max_n: int = 0, log=None) -> dict:
    """The KS2 `stale` sample set (SELECT subjects, every sequence of `subject_seqs`, base and v2-v4, pairs of consecutive
    endpoints) through the frozen `patch_r2` head at R0 (as trained); stores the per-vertex residual
    res = prediction - `full` normal-displacement target on the KS2 mask (visible & observed & contour) with the geometry
    needed later.  Imports `tools/kstar/head_probe.py` read-only."""
    from kstar import common as KC
    from kstar import head_probe as HP
    from torch.utils.data import DataLoader
    d = deps()
    XC, XV = d["XC"], d["XV"]
    torch.set_num_threads(8)
    dev = torch.device("cpu")
    head, mem = HP.load_head(KC.HEADS.parent, HP.HEAD, dev)
    assert mem == 600
    geo = HP.T.Geo(dev)
    seqs = XC.subject_seqs(KC.SELECT_SUBJ)
    ds = HP.StaleData(seqs, stride, ("R0",))
    if max_n:                                    # engineering slice: evenly spaced rows, both domains
        ds.index = ds.index[::max(1, len(ds.index) // max_n)][:max_n]
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=workers, collate_fn=HP.collate)
    names = [sq.name for sq in ds.seqs]
    rows = {k: [] for k in ("seq", "end", "x0", "betas", "K", "nm", "vid", "res", "tfull", "uv0", "R", "z0", "cos")}
    rng = np.random.default_rng(0)
    t0 = time.time()
    with torch.no_grad():
        for bn, batch in enumerate(loader):
            g = geo(batch, "A", rng)
            e1 = g["R"].float()[..., :, 0]
            tfull = (e1 * g["t"].float()).sum(-1)
            mk = g["vis"] & g["obs2"] & (g["cos"] < XV.CONTOUR_COS)
            pred = HP.read_head(head, g, batch["sae_R0"].to(dev))
            res = pred - tfull
            for b in range(res.shape[0]):
                v = mk[b].nonzero().flatten()
                rows["seq"].append(int(batch["seq"][b])), rows["end"].append(int(batch["end"][b]))
                rows["x0"].append(g["x0"][b].numpy()), rows["betas"].append(batch["betas"][b].numpy())
                rows["K"].append(batch["K"][b].numpy()), rows["nm"].append(len(v))
                rows["vid"].append(v.numpy().astype(np.int16)), rows["res"].append(res[b][v].numpy().astype(np.float32))
                rows["tfull"].append(tfull[b][v].numpy().astype(np.float32))
                rows["uv0"].append(g["uv0"][b][v].numpy()), rows["R"].append(g["R"][b][v].numpy().astype(np.float32))
                rows["z0"].append(g["z0"][b][v].numpy()), rows["cos"].append(g["cos"][b][v].numpy().astype(np.float32))
            if log and bn % 100 == 0:
                log(f"collect batch {bn}/{len(loader)} {time.time() - t0:.0f}s")
    nm = np.asarray(rows["nm"], np.int64)
    S = {"names": np.asarray(names), "seq": np.asarray(rows["seq"], np.int64), "end": np.asarray(rows["end"], np.int64),
         "x0": np.stack(rows["x0"]), "betas": np.stack(rows["betas"]), "K": np.stack(rows["K"]), "nm": nm,
         "off": np.concatenate([[0], np.cumsum(nm)]),
         "n_dataset": len(ds), "wall_s": time.time() - t0}
    for k in ("vid", "res", "tfull", "uv0", "R", "z0", "cos"):
        S[k] = np.concatenate(rows[k]) if rows[k] else np.zeros(0)
    return S


def sample_tags(S: dict):
    """Per-sample subject, domain code (0 local, 1 global), base flag and KS2 bootstrap group id."""
    names = [str(n) for n in S["names"]]
    subj = np.asarray([names[s].split("_")[0] for s in S["seq"]])
    dom = np.asarray([0 if "local" in names[s] else 1 for s in S["seq"]], np.int64)
    base = np.asarray([names[s] in (f"{names[s].split('_')[0]}_local", f"{names[s].split('_')[0]}_global")
                       for s in S["seq"]])
    grp = S["seq"].astype(np.int64) * 1_000_000 + S["end"] // 1000
    return subj, dom, base, grp


# ======================================================================================================== PE2 analysis
def _span_share(Jw: np.ndarray, r: np.ndarray, cut: float):
    """Share of r (n,) inside the column space of Jw (n, 51) (singular values >= cut * s_max), rank, coefficient PR."""
    U, s, _ = np.linalg.svd(Jw, full_matrices=False)
    k = int((s >= cut * s[0]).sum()) if len(s) and s[0] > 0 else 0
    if k == 0:
        return 0.0, 0, 0.0, U[:, :0]
    Uk = U[:, :k]
    c = Uk.T @ r
    e = float(r @ r)
    ce = c * c
    return float(ce.sum() / max(e, 1e-300)), k, float(ce.sum() ** 2 / max((ce ** 2).sum(), 1e-300)), Uk


def chunk_rows(S: dict, idx: np.ndarray, x0: np.ndarray, betas1: torch.Tensor, Kmat, mano):
    """Normal-projected Jacobian rows (n_tot, 51) at the KS2-mask vertices of samples `idx` (one sequence), with the
    states `x0` (len(idx), 51) -- the samples' own anchors or, for the wrong-pose null, other poses -- plus the flat
    vertex ids and frames.  Row order: sample by sample, mask vertices ascending."""
    off = S["off"]
    nm = S["nm"][idx]
    vids = np.concatenate([S["vid"][off[i]:off[i + 1]] for i in idx]).astype(np.int64)
    Rm = np.concatenate([S["R"][off[i]:off[i + 1]] for i in idx]).astype(np.float64)
    e1 = torch.as_tensor(Rm[:, :, 0])
    bi = torch.repeat_interleave(torch.arange(len(idx)), torch.as_tensor(nm))
    X, dX = tangent_jacobian(mano, torch.as_tensor(np.asarray(x0, np.float64)), betas1, bi, torch.as_tensor(vids))
    return normal_rows(X, dX, e1, Kmat).numpy(), vids, Rm


def analyze_select(S: dict, sigma_by_dom: dict, chunk: int = 48, log=None) -> dict:
    """Per-sample span(J) shares, nulls and controls; KS2-estimator hop moments; analytic J-model pair sums on the
    real masks.  Returns per-sample arrays and per-domain accumulators."""
    T = tables()
    mano = T["mano"]
    N = len(S["seq"])
    off = S["off"]
    subj, dom, base, grp = sample_tags(S)
    names = [str(n) for n in S["names"]]
    keys = ("E", "EJ", "EJ_alt", "Eperm", "Ewrong", "Epos", "reff", "n", "pr", "gauss", "reff_alt", "Esmooth")
    Vlow = np.linalg.eigh(T["L"])[1][:, :51]          # 51 lowest mesh-Laplacian modes (generic smooth basis)
    A = {k: np.full(N, np.nan) for k in keys}
    hop = HopAcc(T["PI"], T["PJ"], T["PH"])
    jacc = {0: new_pe1_acc(len(T["PI"])), 1: new_pe1_acc(len(T["PI"]))}
    t0 = time.time()
    for sq_id in np.unique(S["seq"]):
        idx_all = np.nonzero(S["seq"] == sq_id)[0]
        betas1 = torch.as_tensor(S["betas"][idx_all[0]], dtype=torch.float64).view(1, -1)
        Kmat = S["K"][idx_all[0]]
        dm = int(dom[idx_all[0]])
        sig = sigma_by_dom["local" if dm == 0 else "global"]["J_main"]
        for p0 in range(0, len(idx_all), chunk):
            pos = np.arange(p0, min(p0 + chunk, len(idx_all)))
            idx = idx_all[pos]
            nm = S["nm"][idx]
            if int(nm.sum()) == 0:
                continue
            part = idx_all[(pos + NEAR_ID_SHIFT) % len(idx_all)]
            Jn, vids, Rm = chunk_rows(S, idx, S["x0"][idx], betas1, Kmat, mano)
            Jw_all = chunk_rows(S, idx, S["x0"][part], betas1, Kmat, mano)[0]
            o = np.concatenate([[0], np.cumsum(nm)])
            for q, i in enumerate(idx):
                n = int(nm[q])
                r = S["res"][off[i]:off[i + 1]].astype(np.float64)
                mask = np.zeros(778, bool)
                mask[S["vid"][off[i]:off[i + 1]]] = True
                dense = np.zeros(778)
                dense[S["vid"][off[i]:off[i + 1]]] = r
                dkey = "local" if dom[i] == 0 else "global"
                hop.add(dkey, dense.astype(np.float32).astype(np.float64), mask, int(grp[i]))
                if n < 3:
                    continue
                Ji = Jn[o[q]:o[q + 1]]
                Jd = np.zeros((778, 51))
                Jd[S["vid"][off[i]:off[i + 1]]] = Ji
                e1d = np.zeros((778, 2))
                e1d[S["vid"][off[i]:off[i + 1]]] = Rm[o[q]:o[q + 1], :, 0]
                accumulate_jmodel(jacc[int(dom[i])], Jd, mask, e1d, T)
                rng = np.random.default_rng(SEED + int(i))
                Jws = Ji * sig[None, :]
                sh, k, pr, Uk = _span_share(Jws, r, RANK_CUT)
                sh_alt, k_alt, _, _ = _span_share(Jws, r, RANK_CUT_ALT)
                e = float(r @ r)
                perm = np.mean([float(np.sum((Uk.T @ rng.permutation(r)) ** 2)) for _ in range(N_PERM_SPAN)]) / max(e, 1e-300)
                gs_draws = [rng.standard_normal(n) for _ in range(N_PERM_SPAN)]
                gs = float(np.mean([float(np.sum((Uk.T @ gd) ** 2)) / float(gd @ gd) for gd in gs_draws]))
                shw, _, _, _ = _span_share(Jw_all[o[q]:o[q + 1]] * sig[None, :], r, RANK_CUT)
                Qs = np.linalg.qr(Vlow[S["vid"][off[i]:off[i + 1]].astype(np.int64), :max(k, 1)])[0]
                sm_share = float(np.sum((Qs.T @ r) ** 2)) / max(e, 1e-300)
                sig_sim = Jws @ rng.standard_normal(51)
                noise = rng.standard_normal(n) * math.sqrt(POS_NOISE_ENERGY * float(sig_sim @ sig_sim) / n)
                rp = sig_sim + noise
                pos_share = float(np.sum((Uk.T @ rp) ** 2)) / max(float(rp @ rp), 1e-300)
                for kk, vv in (("E", e), ("EJ", sh * e), ("EJ_alt", sh_alt * e), ("Eperm", perm * e), ("Ewrong", shw * e),
                               ("Epos", pos_share), ("reff", k), ("n", n), ("pr", pr), ("gauss", gs), ("reff_alt", k_alt),
                               ("Esmooth", sm_share * e)):
                    A[kk][i] = vv
        if log:
            log(f"analyze seq {names[sq_id]}: {len(idx_all)} samples {time.time() - t0:.0f}s")
    return {"A": A, "hop": hop, "jacc": jacc, "tags": (subj, dom, base, grp)}


def synthetic_residuals(S: dict, sigma_by_dom: dict, kind: str, seed: int = SEED) -> dict:
    """Copy of S whose residuals are replaced by a synthetic field with known structure (tool controls, never data):
    kind 'J' = J delta + 10 % iid noise energy; 'smooth' = random combination of the 20 lowest mesh-Laplacian modes
    + the same noise."""
    T = tables()
    mano = T["mano"]
    Vlow = np.linalg.eigh(T["L"])[1]
    subj, dom, base, grp = sample_tags(S)
    rng = np.random.default_rng(seed)
    S2 = dict(S)
    res = np.zeros_like(S["res"])
    for sq in np.unique(S["seq"]):
        idx_all = np.nonzero(S["seq"] == sq)[0]
        if int(S["nm"][idx_all].sum()) == 0:
            continue
        betas1 = torch.as_tensor(S["betas"][idx_all[0]], dtype=torch.float64).view(1, -1)
        sg = sigma_by_dom["local" if dom[idx_all[0]] == 0 else "global"]["J_main"]
        Jn, vids, Rm = chunk_rows(S, idx_all, S["x0"][idx_all], betas1, S["K"][idx_all[0]], mano)
        o = np.concatenate([[0], np.cumsum(S["nm"][idx_all])])
        for q, i in enumerate(idx_all):
            a, b = int(S["off"][i]), int(S["off"][i + 1])
            n = b - a
            if n == 0:
                continue
            if kind == "J":
                sgn = (Jn[o[q]:o[q + 1]] * sg) @ rng.standard_normal(51)
            else:
                sgn = Vlow[S["vid"][a:b].astype(np.int64), :20] @ (rng.standard_normal(20) / (1.0 + np.arange(20)))
            res[a:b] = sgn + rng.standard_normal(n) * math.sqrt(POS_NOISE_ENERGY * float(sgn @ sgn) / n)
    S2["res"] = res.astype(np.float32)
    return S2


def share_summary(A: dict, tags, sel: np.ndarray, n_boot: int = N_BOOT) -> dict:
    """Pooled (energy-weighted) variance share with group-bootstrap CIs for the samples `sel`."""
    subj, dom, base, grp = tags
    ok = sel & np.isfinite(A["E"])
    if ok.sum() < 3:
        return {"n_samples": int(ok.sum())}
    g = grp[ok]
    ug, gi = np.unique(g, return_inverse=True)
    G = len(ug)

    def per_group(v):
        return np.bincount(gi, weights=v[ok], minlength=G)
    S_ = {k: per_group(A[k]) for k in ("E", "EJ", "EJ_alt", "Eperm", "Ewrong", "Esmooth")}
    cnt = np.bincount(gi, minlength=G).astype(np.float64)
    pos = per_group(A["Epos"])
    gauss = per_group(A["gauss"])
    expect = per_group(A["reff"] / A["n"])
    W = boot_weights(G, n_boot)

    def ratio(num, den, w):
        return (w @ num) / (w @ den)
    share, chance, wrong, alt = (S_["EJ"].sum() / S_["E"].sum(), S_["Eperm"].sum() / S_["E"].sum(),
                                 S_["Ewrong"].sum() / S_["E"].sum(), S_["EJ_alt"].sum() / S_["E"].sum())
    smooth = S_["Esmooth"].sum() / S_["E"].sum()
    smb = ratio(S_["Esmooth"], S_["E"], W)
    sb = ratio(S_["EJ"], S_["E"], W)
    cb = ratio(S_["Eperm"], S_["E"], W)
    wb = ratio(S_["Ewrong"], S_["E"], W)
    ex = sb - cb
    per_subj = {}
    for sj in sorted(set(subj[ok])):
        m = ok & (subj == sj)
        per_subj[sj] = {"share": float(A["EJ"][m].sum() / A["E"][m].sum()), "chance": float(A["Eperm"][m].sum() / A["E"][m].sum()),
                        "n_samples": int(m.sum())}
    return {"n_samples": int(ok.sum()), "n_groups": int(G), "share": float(share), "ci": pct(sb),
            "chance": float(chance), "chance_ci": pct(cb), "excess": float(share - chance), "excess_ci": pct(ex),
            "wrong_pose": float(wrong), "wrong_pose_ci": pct(wb), "share_rank1e3": float(alt),
            "smooth_basis": float(smooth), "smooth_basis_ci": pct(smb), "vs_smooth": float(share - smooth),
            "vs_smooth_ci": pct(sb - smb),
            "per_sample_mean_share": float(np.mean(A["EJ"][ok] / A["E"][ok])),
            "pos_control": float(pos.sum() / cnt.sum()), "gauss_control": float(gauss.sum() / cnt.sum()),
            "gauss_expected": float(expect.sum() / cnt.sum()),
            "r_eff_mean": float(np.mean(A["reff"][ok])), "r_eff_1e3_mean": float(np.mean(A["reff_alt"][ok])),
            "n_mask_mean": float(np.mean(A["n"][ok])), "pr_coef_mean": float(np.mean(A["pr"][ok])),
            "per_subject": per_subj}


# ======================================================================================================== GSCD
H_IMG, W_IMG = 180, 240


def window_weights(uv: np.ndarray, R: np.ndarray, rho: np.ndarray, P: int = PATCH):
    """Pixel-weight maps (csr, (n, 180 * 240)) of the head's patch windows: the P x P taps `uv + R loc rho`,
    loc in [-1, 1]^2, bilinear on the 240 x 180 grid with zero padding (`grid_sample`, align_corners=True, as
    `PatchHead.forward`); duplicate taps are summed."""
    import scipy.sparse as sp
    n = uv.shape[0]
    o = np.linspace(-1.0, 1.0, P)
    gy, gx = np.meshgrid(o, o, indexing="ij")
    loc = np.stack([gx, gy], -1).reshape(-1, 2)
    off = np.einsum("nab,pb->npa", R, loc) * rho[:, None, None]
    pix = uv[:, None, :] + off
    x, y = pix[..., 0], pix[..., 1]
    x0, y0 = np.floor(x), np.floor(y)
    wx, wy = x - x0, y - y0
    rows, cols, vals = [], [], []
    for dx, wxx in ((0, 1.0 - wx), (1, wx)):
        for dy, wyy in ((0, 1.0 - wy), (1, wy)):
            xi, yi = (x0 + dx).astype(np.int64), (y0 + dy).astype(np.int64)
            ok = (xi >= 0) & (xi < W_IMG) & (yi >= 0) & (yi < H_IMG)
            rr = np.broadcast_to(np.arange(n)[:, None], x.shape)
            rows.append(rr[ok]), cols.append((yi * W_IMG + xi)[ok]), vals.append((wxx * wyy)[ok])
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, H_IMG * W_IMG))


def window_sharing(A, gain: np.ndarray):
    """(n, n) source-area-weighted and unweighted cosine overlaps of the windows' pixel-weight maps.  With iid pixel
    noise of variance proportional to the pixel's source-pixel count (shot noise of the 9 / 6 / 4 pooled sensor
    pixels), the weighted cosine IS the correlation two linear window readouts inherit from shared pixels."""
    import scipy.sparse as sp
    Gw = (A @ sp.diags(gain) @ A.T).toarray()
    Gu = (A @ A.T).toarray()

    def cos(G):
        dg = np.sqrt(np.clip(np.diag(G), 1e-300, None))
        return G / dg[:, None] / dg[None, :]
    return cos(Gw), cos(Gu)


def cell_overlap(uv_a: np.ndarray, uv_b: np.ndarray, stride: int) -> np.ndarray:
    """Histogram intersection of the two vertices' bilinear cell supports on a stride-`stride` feature map
    (`tools/mv2/common.py::sample_map` geometry: cell m at input pixel stride * m)."""
    def cells(u):
        u = u / stride
        c0 = np.floor(u)
        w = u - c0
        keys, wts = [], []
        for dy in (0, 1):
            for dx in (0, 1):
                keys.append((c0[:, 1] + dy) * 8192 + (c0[:, 0] + dx))
                wts.append((1.0 - w[:, 0] if dx == 0 else w[:, 0]) * (1.0 - w[:, 1] if dy == 0 else w[:, 1]))
        return np.stack(keys, 1), np.stack(wts, 1)
    ka, wa = cells(uv_a)
    kb, wb = cells(uv_b)
    eq = ka[:, :, None] == kb[:, None, :]
    return (eq * np.minimum(wa[:, :, None], wb[:, None, :])).sum((1, 2))


_GS: dict = {}


def _worker_init():
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:                                    # noqa: BLE001
        pass
    torch.set_num_threads(1)


def gscd_one(i: int):
    """Hop <= 3 vertex pairs of sample i (both vertices in the KS2 mask): hop, window sharing (weighted / unweighted),
    layer-cell overlaps (strides 4, 8, 16), pair distance / window half-size, residuals, and the synthetic window-noise
    residuals of the positive control."""
    S, T = _GS["S"], _GS["T"]
    a, b = int(S["off"][i]), int(S["off"][i + 1])
    if b - a < 2:
        return None
    vid = S["vid"][a:b].astype(np.int64)
    res = S["res"][a:b].astype(np.float64)
    uv = S["uv0"][a:b].astype(np.float64)
    R = S["R"][a:b].astype(np.float64)
    z = S["z0"][a:b].astype(np.float64)
    K = S["K"][i]
    rho = math.sqrt(K[0, 0] * K[1, 1]) * HEAD_R / np.clip(z, 1e-3, None)
    Dsub = T["D"][np.ix_(vid, vid)]
    ia, ib = np.nonzero(np.triu((Dsub >= 1) & (Dsub <= max(GSCD_HOPS)), 1))
    if len(ia) == 0:
        return None
    A = window_weights(uv, R, rho)
    sw, su = window_sharing(A, T["gain"])
    rng = np.random.default_rng(SEED + 10 * int(i) + 7)
    syn = A @ (rng.standard_normal(H_IMG * W_IMG) * np.sqrt(T["gain"]))
    lap = _GS["H"] @ rng.standard_normal(778) if _GS.get("H") is not None else np.zeros(778)
    dn = np.linalg.norm(uv[ia] - uv[ib], axis=-1) / (0.5 * (rho[ia] + rho[ib]))
    return {"hop": Dsub[ia, ib].astype(np.int8), "sw": sw[ia, ib].astype(np.float32), "su": su[ia, ib].astype(np.float32),
            "sL1": cell_overlap(uv[ia], uv[ib], 4).astype(np.float32),
            "sL2": cell_overlap(uv[ia], uv[ib], 8).astype(np.float32),
            "sL3": cell_overlap(uv[ia], uv[ib], 16).astype(np.float32), "dn": dn.astype(np.float32),
            "x": res[ia].astype(np.float32), "y": res[ib].astype(np.float32),
            "xs": syn[ia].astype(np.float32), "ys": syn[ib].astype(np.float32),
            "xl": lap[vid[ia]].astype(np.float32), "yl": lap[vid[ib]].astype(np.float32),
            "samp": np.full(len(ia), i, np.int32)}


def gscd_pairs(S: dict, workers: int = 16, H_lap=None, log=None) -> dict:
    """All pair records of all samples, concatenated (order: sample, then pair).  `H_lap` (778, 778): square root of the
    Laplacian-field null covariance (geometry-only control: a graph-smooth field with no pixel mechanism)."""
    T = tables()
    _GS.update(S=S, T=T, H=H_lap)
    ids = list(range(len(S["seq"])))
    t0 = time.time()
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers, initializer=_worker_init) as pool:
            parts = [p for p in pool.imap(gscd_one, ids, chunksize=16) if p is not None]
    else:
        parts = [p for p in map(gscd_one, ids) if p is not None]
    if log:
        log(f"gscd pairs {time.time() - t0:.0f}s from {len(parts)} samples")
    keys = parts[0].keys() if parts else []
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


# ---------------------------------------------------------------------------------------------------- stratification
def thresholds(s, hop, hops=GSCD_HOPS) -> dict:
    return {h: tuple(np.quantile(s[hop == h], [1.0 / 3, 2.0 / 3])) for h in hops if (hop == h).any()}


def labels_from_thr(s, hop, thr) -> np.ndarray:
    lab = np.full(len(s), -1, np.int8)
    for h, (c1, c2) in thr.items():
        m = hop == h
        lab[m] = (s[m] > c1).astype(np.int8) + (s[m] > c2).astype(np.int8)
    return lab


def labels_within_segments(s, seg, min_seg: int = TERTILE_MIN_SEG) -> np.ndarray:
    """Rank-tertile label of s within each segment (segments must be contiguous and sorted); small segments -> -1."""
    order = np.lexsort((s, seg))
    seg_o = seg[order]
    start = np.searchsorted(seg_o, seg_o, side="left")
    end = np.searchsorted(seg_o, seg_o, side="right")
    frac = (np.arange(len(s)) - start + 0.5) / (end - start)
    lab_o = np.minimum((3 * frac).astype(np.int8), 2)
    lab_o[(end - start) < min_seg] = -1
    lab = np.empty(len(s), np.int8)
    lab[order] = lab_o
    return lab


def labels_distance_matched(s, hop, dn, hops=GSCD_HOPS, bins: int = DIST_BINS) -> np.ndarray:
    """Tertile of s within (hop, quantile bin of the pair distance / window size)."""
    lab = np.full(len(s), -1, np.int8)
    for h in hops:
        m = np.nonzero(hop == h)[0]
        if not len(m):
            continue
        edges = np.quantile(dn[m], np.linspace(0, 1, bins + 1)[1:-1])
        b = np.digitize(dn[m], edges)
        for k in range(bins):
            mm = m[b == k]
            if len(mm) < 3:
                continue
            c1, c2 = np.quantile(s[mm], [1.0 / 3, 2.0 / 3])
            lab[mm] = (s[mm] > c1).astype(np.int8) + (s[mm] > c2).astype(np.int8)
    return lab


def diff_from_labels(x, y, hop, lab, gcode, G, W=None, hops=GSCD_HOPS, min_n: int = 30):
    """Top-minus-bottom-tertile pooled correlation, averaged over hops.  With `W` (n_boot, G): bootstrap over groups."""
    ok = lab >= 0
    hi = np.asarray([hops.index(h) if h in hops else -1 for h in range(0, 8)])[hop[ok]]
    code = ((hi * 3 + lab[ok].astype(np.int64)) * G + gcode[ok])
    nh = len(hops)
    M = binned_moments(x[ok], y[ok], code, nh * 3 * G).reshape(nh, 3, G, 6)
    tot = M.sum(2)
    r = r_from_moments(tot)                                           # (hops, 3)
    n_ok = (tot[:, 0, 0] >= min_n) & (tot[:, 2, 0] >= min_n)
    d_h = r[:, 2] - r[:, 0]
    d_h = np.where(n_ok, d_h, np.nan)
    out = {"D": float(np.nanmean(d_h)) if n_ok.any() else float("nan"), "D_hop": d_h.tolist(),
           "r_tertile": r.tolist(), "n_tertile": tot[:, :, 0].tolist()}
    if W is not None and n_ok.any():
        tb = np.einsum("bg,hlgk->bhlk", W, M)
        rb = r_from_moments(tb)
        db = np.where(n_ok[None], rb[:, :, 2] - rb[:, :, 0], np.nan)
        Db = np.nanmean(db, axis=1)
        out["D_ci"] = pct(Db)
        out["D_boot_frac_pos"] = float(np.mean(Db > 0))
    return out


def shuffle_within(s, seg, rng) -> np.ndarray:
    """Random permutation of s inside each segment (`seg` non-decreasing, i.e. segments contiguous): keeps every
    segment's multiset, hence the hop distribution, and destroys which pair carries which s."""
    key = seg.astype(np.float64) + rng.random(len(s))
    return s[np.argsort(key, kind="stable")]


def perm_null(x, y, s, hop, seg, thr, n_perm: int = N_PERM, seed: int = SEED, hops=GSCD_HOPS) -> dict:
    rng = np.random.default_rng(seed)
    hidx = np.asarray([hops.index(h) if h in hops else 0 for h in range(0, 8)])[hop]
    xy = None
    Ds = []
    for _ in range(n_perm):
        sp = shuffle_within(s, seg, rng)
        lab = labels_from_thr(sp, hop, thr)
        ok = lab >= 0
        code = hidx[ok] * 3 + lab[ok].astype(np.int64)
        M = binned_moments(x[ok], y[ok], code, len(hops) * 3).reshape(len(hops), 3, 6)
        r = r_from_moments(M)
        Ds.append(float(np.nanmean(r[:, 2] - r[:, 0])))
    Ds = np.asarray(Ds)
    return {"mean": float(Ds.mean()), "sd": float(Ds.std()), "p975_abs": float(np.percentile(np.abs(Ds), 97.5)),
            "max_abs": float(np.abs(Ds).max()), "n": int(n_perm)}


def spearman_fast(a, b) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def gscd_domain(P: dict, grp_of_samp: np.ndarray, sel_samp: np.ndarray, n_boot: int = N_BOOT, n_perm: int = N_PERM,
                min_pairs: int = 1000) -> dict:
    """Full GSCD statistic set for the samples `sel_samp` (bool over samples)."""
    ok = sel_samp[P["samp"]]
    if ok.sum() < min_pairs:
        return {"n_pairs": int(ok.sum())}
    sub = {k: v[ok] for k, v in P.items()}
    ordr = np.argsort(sub["samp"].astype(np.int64) * 4 + sub["hop"], kind="stable")        # segments contiguous
    sub = {k: v[ordr] for k, v in sub.items()}
    hop = sub["hop"]
    g_all = grp_of_samp[sub["samp"]]
    ug, gcode = np.unique(g_all, return_inverse=True)
    G = len(ug)
    W = boot_weights(G, n_boot)
    seg = sub["samp"].astype(np.int64) * 4 + hop
    s = sub["sw"].astype(np.float64)
    thr = thresholds(s, hop)
    lab = labels_from_thr(s, hop, thr)
    prim = diff_from_labels(sub["x"], sub["y"], hop, lab, gcode, G, W)
    out = {"n_pairs": int(ok.sum()), "n_groups": int(G), "thresholds": {str(h): list(v) for h, v in thr.items()},
           "primary": prim, "per_hop_n": {str(h): int((hop == h).sum()) for h in GSCD_HOPS}}
    out["perm"] = perm_null(sub["x"], sub["y"], s, hop, seg, thr, n_perm)
    labw = labels_within_segments(s, seg)
    out["within"] = diff_from_labels(sub["x"], sub["y"], hop, labw, gcode, G, W)
    labd = labels_distance_matched(s, hop, sub["dn"].astype(np.float64))
    out["dist_matched"] = diff_from_labels(sub["x"], sub["y"], hop, labd, gcode, G, W)
    out["control_syn"] = diff_from_labels(sub["xs"], sub["ys"], hop, lab, gcode, G, W)
    out["control_lap"] = diff_from_labels(sub["xl"], sub["yl"], hop, lab, gcode, G, W)
    rng_i = np.random.default_rng(SEED + 11)
    out["control_iid"] = diff_from_labels(rng_i.standard_normal(len(hop)), rng_i.standard_normal(len(hop)), hop, lab,
                                          gcode, G, W)
    out["sens_unweighted"] = diff_from_labels(sub["x"], sub["y"], hop,
                                              labels_from_thr(sub["su"].astype(np.float64), hop,
                                                              thresholds(sub["su"].astype(np.float64), hop)), gcode, G, W)
    cells = {}
    rng = np.random.default_rng(SEED + 5)
    for L in ("sL1", "sL2", "sL3"):
        sl = sub[L].astype(np.float64) + 1e-9 * rng.random(len(hop))              # random tie-break
        labc = labels_from_thr(sl, hop, thresholds(sl, hop))
        cells[L] = diff_from_labels(sub["x"], sub["y"], hop, labc, gcode, G, W)
        cells[L]["frac_shared"] = {str(h): float(np.mean(sub[L][hop == h] > 0)) for h in GSCD_HOPS}
        cells[L]["mean_overlap"] = {str(h): float(np.mean(sub[L][hop == h])) for h in GSCD_HOPS}
    out["layer_cells"] = cells
    col = {}
    rng_c = np.random.default_rng(SEED + 6)
    for h in GSCD_HOPS:
        ih = np.nonzero(hop == h)[0]
        ih = rng_c.choice(ih, min(len(ih), 200000), replace=False)
        col[str(h)] = spearman_fast(s[ih], sub["dn"][ih].astype(np.float64))
    out["collinearity_spearman_s_dn"] = col
    out["s_quantiles"] = {str(h): np.quantile(s[hop == h], [0.05, 0.5, 0.95]).tolist() for h in GSCD_HOPS}
    return out


# ======================================================================================================== decide
def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)))


def decide(result: dict) -> dict:
    """Applies exactly the gate of `tools/mv2/prereg_m6.md`.  Verdicts: pass / kill / descriptive; any tool-validity
    failure is `tool_defect` (never pass).  Three verdicts: `kscs_spectrum` (PE1, local), `kscs_span` (PE2 share, local,
    all 16 SELECT sequences), `gscd` (local, head-window sharing)."""
    out = {"tool_defect": [], "notes": []}
    led = result["ledger"]
    # ------------------------------------------------------------------ tool validity (shared)
    bad_units = [k for k, v in result.get("unit_checks", {}).items() if not v]
    if bad_units or not result.get("unit_checks"):
        out["tool_defect"].append(f"unit_checks failed or absent: {bad_units}")
    if not np.allclose(np.round(led["local"]["r"], 3), LEDGER_TEXT_LOCAL, atol=5e-4):
        out["tool_defect"].append("ledger result.json does not match the ledger text spectrum")
    pe2 = result.get("pe2", {})
    rep = pe2.get("repro", {})
    for dom in ("local", "global"):
        if dom in rep:
            r, n = rep[dom]["r"], rep[dom]["n"]
            if (not np.all(np.abs(np.asarray(r) - np.asarray(led[dom]["r"])) <= REPRO_R_TOL)
                    or not np.all(np.abs(np.asarray(n) - np.asarray(led[dom]["n"])) <= REPRO_N_REL * np.asarray(led[dom]["n"]))):
                out["tool_defect"].append(f"PE2 does not reproduce the KS2 {dom} hop spectrum / pair counts")
    if pe2 and result["meta"].get("n_dataset") not in (None, led["n_samples"]):
        out["tool_defect"].append("PE2 sample count differs from KS2 n_samples")

    # ------------------------------------------------------------------ KSCS spectrum (PE1, local)
    sp = result["pe1"]["local"]["spectra"]
    L = np.asarray(led["local"]["r"])
    half = np.asarray([(c[1] - c[0]) / 2 for c in led["local"]["ci"]])
    J = np.asarray(sp["J_main"]["pooled"])
    per = np.asarray([sp["J_main"]["per_subject"][s] for s in sorted(sp["J_main"]["per_subject"])])
    sd = per.std(0, ddof=1) if len(per) > 1 else np.zeros(4)
    Th = np.maximum(2 * sd[:3], half[:3])
    T = float(Th.max())
    loso = np.asarray([v for v in result["pe1"]["local"].get("J_main_loso", {}).values()])
    sd_loso = loso.std(0, ddof=1).tolist() if len(loso) > 1 else None
    delta = np.abs(J[:3] - L[:3])
    null_names = [k for k in NULL_MODELS if k in sp]
    rm_null = {k: _rmse(sp[k]["pooled"], L) for k in null_names}
    best_null = min(rm_null.values())
    rm_J = _rmse(J, L)
    gaps = []
    for s in sorted(sp["J_main"]["per_subject"]):
        gaps.append(min(_rmse(sp[k]["per_subject"][s], L) for k in null_names) - _rmse(sp["J_main"]["per_subject"][s], L))
    T2 = float(max(2 * np.std(gaps, ddof=1) if len(gaps) > 1 else 0.0, np.sqrt(np.mean(half ** 2))))
    gap = best_null - rm_J
    pass_a = bool(delta.max() <= T and J[3] < 0)
    pass_b = bool(gap >= T2)
    sv = "pass" if (pass_a and pass_b) else ("kill" if not pass_b else "descriptive")
    cal_bad = [k for k in ("lap_scalar", "lap_vector")
               if not result["pe1"]["local"]["tau"][k]["pooled"]["calibrated"]]
    if cal_bad:
        out["notes"].append(f"Laplacian null could not reach the ledger hop-1 correlation: {cal_bad} (null weaker than designed)")
    out["kscs_spectrum"] = {"verdict": sv, "delta_hop123": delta.tolist(), "delta_max": float(delta.max()), "T": T,
                            "T_hop": Th.tolist(), "hop4_model": float(J[3]), "hop4_negative": bool(J[3] < 0),
                            "rmse_J": rm_J, "rmse_null": rm_null, "gap_vs_best_null": float(gap), "T2": T2,
                            "gap_per_subject": gaps, "pass_spectrum_match": pass_a, "pass_beats_null": pass_b,
                            "sd_between_subjects_hop1234": sd.tolist(), "sd_loso_replicates_hop1234": sd_loso,
                            "ledger_half_ci_hop1234": half.tolist(),
                            "variants_delta_max": {k: float(np.abs(np.asarray(sp[k]["pooled"])[:3] - L[:3]).max())
                                                   for k in SIGMA_VARIANTS if k in sp}}

    # ------------------------------------------------------------------ KSCS span share (PE2, local, all sequences)
    sh = pe2.get("share", {}).get("local", {}).get("all")
    if sh and "share" in sh:
        wc = result["pe1"]["local"]["white_chance"]
        white_hi = wc["mean"] + 2 * wc["sd"]
        if wc["mean"] >= CHANCE_VACUOUS:
            out["tool_defect"].append("share metric vacuous: FIT7 white-noise chance share >= 0.5")
        if sh["pos_control"] < SHARE_POS_MIN:
            out["tool_defect"].append("share positive control (simulated J delta + noise) below 0.85")
        if abs(sh["gauss_control"] - sh["gauss_expected"]) > PERM_SHARE_TOL:
            out["tool_defect"].append("share Gaussian negative control differs from r_eff / n")
        s, ci, exc = sh["share"], sh["ci"], sh["excess_ci"]
        if s < SHARE_KILL or exc[1] <= 0:
            vs = "kill"
        elif s >= SHARE_PASS and exc[0] > 0 and s > white_hi and sh["vs_smooth_ci"][0] > 0:
            vs = "pass"
        else:
            vs = "descriptive"
        out["kscs_span"] = {"verdict": vs, "share": s, "ci": ci, "chance_perm": sh["chance"], "excess_ci": exc,
                            "white_chance_fit7_hi": float(white_hi), "wrong_pose": sh["wrong_pose"],
                            "smooth_basis": sh["smooth_basis"], "vs_smooth_ci": sh["vs_smooth_ci"],
                            "r_eff_mean": sh["r_eff_mean"], "n_mask_mean": sh["n_mask_mean"],
                            "pos_control": sh["pos_control"]}
    else:
        out["kscs_span"] = {"verdict": "not_run"}

    # ------------------------------------------------------------------ GSCD (local, primary = head-window sharing)
    gs = result.get("gscd", {}).get("local")
    if gs and "primary" in gs:
        D = gs["primary"]["D"]
        lo, hi = gs["primary"].get("D_ci", [np.nan, np.nan])
        p975 = gs["perm"]["p975_abs"]
        wlo = gs["within"].get("D_ci", [np.nan, np.nan])[0]
        d_syn = gs["control_syn"]["D"]
        d_lap = gs["control_lap"]["D"]
        d_iid = gs["control_iid"]["D"]
        power = bool(np.isfinite(d_syn) and d_syn >= D_PASS)
        if p975 >= D_PERM_MAX:
            out["tool_defect"].append("GSCD shuffle null exceeds 0.05")
        if not (np.isfinite(d_iid) and abs(d_iid) < 0.01):
            out["tool_defect"].append("GSCD iid-noise control not ~0 (stratifier biased)")
        if D >= D_PASS and lo > 0 and p975 < D_PERM_MAX and power:
            if not (np.isfinite(wlo) and wlo > 0):
                gv, why = "descriptive", "grid-sharing difference present but not within-segment (scale / depth confounded)"
            elif np.isfinite(d_lap) and d_lap >= D_PASS:
                gv, why = "descriptive", "a graph-smooth geometry-only field (no pixel mechanism) separates the strata as much"
            else:
                gv, why = "pass", "pass"
        elif D < D_VOID and lo <= 0 <= hi and power:
            gv, why = "kill", "sharing strata do not separate the residual correlation: the GSCD replacement is void"
        elif not power:
            gv, why = "descriptive", "positive control (synthetic window noise) shows D < 0.15: tool underpowered, no verdict"
        else:
            gv, why = "descriptive", "between the pass and void gates"
        out["gscd"] = {"verdict": gv, "why": why, "D": D, "D_ci": [lo, hi], "perm_p975_abs": p975,
                       "within_D": gs["within"]["D"], "within_D_ci": gs["within"].get("D_ci"),
                       "dist_matched_D": gs["dist_matched"]["D"], "control_syn_D": d_syn, "control_lap_D": d_lap,
                       "control_iid_D": d_iid, "power": power}
    else:
        out["gscd"] = {"verdict": "not_run"}

    # ------------------------------------------------------------------ consequences (plan section M6, no GPU arm)
    cons = []
    if out["tool_defect"]:
        out["overall"] = "tool_defect"
        cons.append("a tool-validity condition failed: no verdict is read from this run")
    else:
        v1, v2, v3 = out["kscs_spectrum"]["verdict"], out["kscs_span"]["verdict"], out["gscd"]["verdict"]
        if v1 == "pass" and v2 == "pass":
            cons.append("(a) M1 paired arm: lifted features projected on span(J) (51 dims) before readout vs the "
                        "unprojected 778-vertex readout; CI containing 0 would downgrade the per-vertex claim to per-joint")
            cons.append("(c) n_eff of the vertex measurement is capped by the kinematic dimension; report n_eff with M1")
        if v1 == "kill" or v2 == "kill":
            cons.append("kinematic-subspace hypothesis dead for the KS2 head: no n_eff-cap claim; M6 keeps diagnostic identity")
        if v3 == "pass":
            cons.append("(b) M1 single-variable arm: per-cell de-duplicated pooling (each grid cell once); "
                        "subtract grid sharing before any correlated-precision talk; M6 no longer claims sensor-correlated noise")
        if v3 == "kill":
            cons.append("GSCD replacement void; M6 keeps diagnostic identity; no de-duplication arm")
        if not cons:
            cons.append("descriptive only: nothing triggered; M6 keeps diagnostic identity (no GPU arm in any case)")
        cons.append("(d) clone / refinement invariance and NIS / NEES calibration remain M1 / M3 acceptance items; "
                    "recording-level bias is a nuisance, not zero-mean noise")
        out["overall"] = f"{v1}|{v2}|{v3}"
    out["consequence"] = cons
    out["m6_identity"] = "diagnostic (no GPU arm; not a performance claim; the KS2 spectrum is the patch CNN head's, not S39's)"
    return out


# ======================================================================================================== unit checks
def _u_hop_known_spectrum() -> bool:
    T = tables()
    w, V = np.linalg.eigh(T["L"])
    tau = 0.9
    Hh = (V * np.exp(-0.5 * tau * w)) @ V.T
    rng = np.random.default_rng(SEED)
    acc = HopAcc(T["PI"], T["PJ"], T["PH"])
    mask = np.ones(778, bool)
    for k in range(250):
        acc.add("syn", Hh @ rng.standard_normal(778), mask, k % 25)
    est = np.asarray(acc.result("syn", n_boot=10)["r"])
    P = len(T["PI"])
    ana = null_spectra(np.ones(P), np.ones(P), heat_kernel((w, V), tau), T)["lap_scalar"]
    return bool(np.abs(est - ana).max() < 0.03)


def _u_jmodel_closed_form() -> bool:
    T = tables()
    w, V = np.linalg.eigh(T["L"])
    rng = np.random.default_rng(SEED + 1)
    Jn = ((V * np.exp(-0.3 * 0.6 * w)) @ V.T) @ rng.standard_normal((778, 51))
    mask = rng.random(778) < 0.6
    e1 = rng.standard_normal((778, 2))
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    sig = np.abs(rng.standard_normal(51)) + 0.2
    acc = new_pe1_acc(len(T["PI"]))
    accumulate_jmodel(acc, Jn, mask, e1, T)
    ana = jmodel_spectrum(acc["cov"], acc["vx"], acc["vy"], sig)
    hop = HopAcc(T["PI"], T["PJ"], T["PH"])
    for k in range(600):
        hop.add("mc", Jn @ (sig * rng.standard_normal(51)), mask, k % 20)
    est = np.asarray(hop.result("mc", n_boot=10)["r"])
    return bool(np.abs(est - ana).max() < 0.04)


def _u_jacobian_fd() -> bool:
    T = tables()
    mano = T["mano"]
    lie = deps()["lie"]
    rng = np.random.default_rng(SEED + 2)
    x = torch.zeros(1, 51, dtype=torch.float64)
    x[0, :3] = torch.tensor([0.02, -0.01, 0.5], dtype=torch.float64)
    x[0, 3:6] = torch.as_tensor(rng.normal(0, 0.3, 3))
    x[0, 6:] = torch.as_tensor(rng.normal(0, 0.3, 45))
    betas = torch.zeros(1, 10, dtype=torch.float64)
    vids = torch.as_tensor(rng.choice(778, 12, replace=False))
    X, dX = tangent_jacobian(mano, x, betas, torch.zeros(12, dtype=torch.long), vids)
    v0, _ = mano.fk(x, betas)
    ok = bool((X - v0[0, vids]).abs().max() < 1e-9)
    j0 = (mano.m.J_regressor @ (mano.m.v_template + torch.einsum("bl,mkl->bmk", betas, mano.m.shapedirs)))[:, 0]
    eps = 1e-6
    err, scale = 0.0, 0.0
    for k in range(51):
        dl = torch.zeros(1, 51, dtype=torch.float64)
        dl[0, k] = eps
        vp, _ = mano.fk(lie.retract_51d(x, dl, mano.m.hands_mean, j0), betas)
        vm, _ = mano.fk(lie.retract_51d(x, -dl, mano.m.hands_mean, j0), betas)
        fd = (vp[0, vids] - vm[0, vids]) / (2 * eps)
        err = max(err, float((fd - dX[:, :, k]).abs().max()))
        scale = max(scale, float(fd.abs().max()))
    return bool(ok and err < 1e-5 * max(scale, 1.0))


def _u_span_projection() -> bool:
    rng = np.random.default_rng(SEED + 3)
    n = 300
    Jw = rng.standard_normal((n, 51)) * (0.5 + rng.random(51))
    r_in = Jw @ rng.standard_normal(51)
    s_in, k, _, Uk = _span_share(Jw, r_in, RANK_CUT)
    r_perp = rng.standard_normal(n)
    r_perp -= Uk @ (Uk.T @ r_perp)
    s_perp, _, _, _ = _span_share(Jw, r_perp, RANK_CUT)
    g = [_span_share(Jw, rng.standard_normal(n), RANK_CUT)[0] for _ in range(300)]
    return bool(abs(s_in - 1) < 1e-9 and s_perp < 1e-9 and k == 51 and abs(np.mean(g) - k / n) < 0.015)


def _u_window_sharing() -> bool:
    T = tables()
    uv = np.array([[100.0, 90.0], [100.0, 90.0], [105.0, 90.0], [110.0, 90.0], [115.0, 90.0], [130.0, 90.0], [100.0, 90.0]])
    R = np.broadcast_to(np.eye(2), (7, 2, 2)).copy()
    R[6] = np.array([[0.0, -1.0], [1.0, 0.0]])
    rho = np.full(7, 10.0)
    A = window_weights(uv, R, rho)
    sw, su = window_sharing(A, T["gain"])
    mono = sw[0, 1] > sw[0, 2] > sw[0, 3] > sw[0, 4] > sw[0, 5]
    A1 = window_weights(uv, R, rho)
    swu, suu = window_sharing(A1, np.ones(H_IMG * W_IMG))
    return bool(sw[0, 1] > 0.9999 and mono and sw[0, 5] < 1e-9 and sw[0, 6] > 0.98 and np.abs(swu - suu).max() < 1e-12
                and (sw <= 1 + 1e-9).all() and (sw >= -1e-12).all())


def _u_window_taps_match_grid_sample() -> bool:
    import torch.nn.functional as F
    rng = np.random.default_rng(SEED + 4)
    img = rng.random((H_IMG, W_IMG))
    th = 0.7
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])[None]
    uv = np.array([[120.3, 85.7]])
    rho = np.array([13.0])
    A = window_weights(uv, R, rho)
    mine = float((A @ img.reshape(-1))[0])
    o = torch.linspace(-1, 1, PATCH, dtype=torch.float64)
    gy, gx = torch.meshgrid(o, o, indexing="ij")
    loc = torch.stack([gx, gy], -1).view(-1, 2)
    pix = torch.as_tensor(uv[0]) + (torch.as_tensor(R[0]) @ loc.T).T * 13.0
    norm = torch.stack([pix[:, 0] / (W_IMG - 1) * 2 - 1, pix[:, 1] / (H_IMG - 1) * 2 - 1], -1).view(1, PATCH, PATCH, 2)
    ref = F.grid_sample(torch.as_tensor(img)[None, None], norm, align_corners=True).sum().item()
    # window near the border: zero padding as grid_sample
    uv2 = np.array([[3.2, 177.0]])
    A2 = window_weights(uv2, R, rho)
    pix2 = torch.as_tensor(uv2[0]) + (torch.as_tensor(R[0]) @ loc.T).T * 13.0
    norm2 = torch.stack([pix2[:, 0] / (W_IMG - 1) * 2 - 1, pix2[:, 1] / (H_IMG - 1) * 2 - 1], -1).view(1, PATCH, PATCH, 2)
    ref2 = F.grid_sample(torch.as_tensor(img)[None, None], norm2, align_corners=True).sum().item()
    return bool(abs(mine - ref) < 1e-9 and abs(float((A2 @ img.reshape(-1))[0]) - ref2) < 1e-9)


def _u_shot_noise_cosine() -> bool:
    T = tables()
    uv = np.array([[100.0, 90.0], [106.0, 91.0], [114.0, 88.0]])
    R = np.broadcast_to(np.eye(2), (3, 2, 2)).copy()
    A = window_weights(uv, R, np.full(3, 9.0))
    sw, _ = window_sharing(A, T["gain"])
    rng = np.random.default_rng(SEED + 5)
    z = rng.standard_normal((H_IMG * W_IMG, 4000)) * np.sqrt(T["gain"])[:, None]
    r = A @ z
    c = np.corrcoef(r)
    return bool(np.abs(c - sw).max() < 0.06)


def _u_tertile_estimator() -> bool:
    rng = np.random.default_rng(SEED + 6)
    nseg, per = 20000, 20
    seg = np.repeat(np.arange(nseg), per)
    hop = (1 + seg % 3).astype(np.int8)
    n = nseg * per
    s = rng.random(n)
    rho = 0.2 + 0.5 * s
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    x = z1
    y = rho * z1 + np.sqrt(1 - rho ** 2) * z2
    gcode = (seg % 40).astype(np.int64)
    thr = thresholds(s, hop)
    lab = labels_from_thr(s, hop, thr)
    D = diff_from_labels(x, y, hop, lab, gcode, 40)["D"]
    s_ind = rng.random(n)
    D0 = diff_from_labels(x, y, hop, labels_from_thr(s_ind, hop, thresholds(s_ind, hop)), gcode, 40)["D"]
    pn = perm_null(x, y, s, hop, seg, thr, n_perm=12)
    labw = labels_within_segments(s, seg)
    Dw = diff_from_labels(x, y, hop, labw, gcode, 40)["D"]
    return bool(abs(D - 1.0 / 3.0) < 0.02 and abs(D0) < 0.02 and pn["max_abs"] < 0.03 and abs(Dw - 1.0 / 3.0) < 0.03)


def _u_shuffle_preserves_segments() -> bool:
    rng = np.random.default_rng(SEED + 7)
    seg = np.sort(rng.integers(0, 50, 5000))
    s = rng.random(5000)
    sp = shuffle_within(s, seg, rng)
    ok = all(np.allclose(np.sort(s[seg == k]), np.sort(sp[seg == k])) for k in range(50))
    return bool(ok and not np.allclose(s, sp))


def _u_estimator_equals_ks2() -> bool:
    from kstar import head_probe as HP
    rng = np.random.default_rng(SEED + 8)
    x = rng.standard_normal(5000).astype(np.float32)
    y = (0.4 * x + rng.standard_normal(5000)).astype(np.float32)
    acc = HP.Lin()
    for g, sl in enumerate(np.array_split(np.arange(5000), 7)):
        acc.add("k", torch.as_tensor(x[sl]), torch.as_tensor(y[sl]), group=g)
    ref = acc.result(n_boot=20)["k"]["r"]
    mine = float(r_from_moments(moments(x, y)))
    return bool(abs(ref - mine) < 1e-10)


def _u_hop_pairs_equal_ks2() -> bool:
    from kstar import head_probe as HP
    T = tables()
    ref = HP.hop_pairs(T["faces"])
    for h in HOPS:
        i, j = ref[h]
        if not (np.array_equal(i.numpy(), T["PI"][T["PH"] == h]) and np.array_equal(j.numpy(), T["PJ"][T["PH"] == h])):
            return False
    return True


def _u_calibration() -> bool:
    T = tables()
    P = len(T["PI"])
    a = calibrate_tau(np.ones(P), 0.9 * np.ones(P), 0.7, T, "lap_scalar")
    b = calibrate_tau(np.ones(P), 0.9 * np.ones(P), 0.6, T, "lap_vector")
    w, V = np.linalg.eigh(T["L"])
    ra = null_spectra(np.ones(P), 0.9 * np.ones(P), heat_kernel((w, V), a["tau"]), T)["lap_scalar"][0]
    rb = null_spectra(np.ones(P), 0.9 * np.ones(P), heat_kernel((w, V), b["tau"]), T)["lap_vector"][0]
    return bool(a["calibrated"] and b["calibrated"] and abs(ra - 0.7) < 1e-6 and abs(rb - 0.6) < 1e-6)


def _u_increment_zero() -> bool:
    T = tables()
    x = torch.as_tensor(np.random.default_rng(SEED + 9).normal(0, 0.2, (3, 51)))
    j0 = torch.zeros(3, 3, dtype=torch.float64)
    inc = deps()["lie"].local_coordinates_51d(x, x, T["mano"].m.hands_mean, j0)
    return bool(inc.abs().max() < 1e-9)


def _syn_result(J=(0.80, 0.58, 0.26, -0.10), lap_s=(0.838, 0.45, 0.12, 0.05), lap_v=(0.838, 0.40, 0.05, 0.0),
                share=0.75, share_ci=(0.7, 0.8), chance=0.30, chance_ci=(0.28, 0.32), D=0.22, D_ci=(0.18, 0.26),
                perm=0.01, within_lo=0.05, d_syn=0.5, pos=0.93, units=True, repro=True, white=(0.25, 0.01),
                smooth_ci=(0.05, 0.3)) -> dict:
    led = list(LEDGER_TEXT_LOCAL)
    subs = ["ch", "lfz", "lpc", "lr", "ly", "lyh", "lyq"]

    def mk(v):
        return {"pooled": list(v), "per_subject": {s: [x + 0.01 * (k - 3) * (1 if i % 2 else -1) for i, x in enumerate(v)]
                                                     for k, s in enumerate(subs)}}
    sp = {"J_main": mk(J), "J_perdof": mk(J), "J_equal_effect": mk(J), "J_fingers": mk(J), "lap_scalar": mk(lap_s),
          "lap_vector": mk(lap_v), "iid": {"pooled": [0.0] * 4, "per_subject": {s: [0.0] * 4 for s in subs}}}
    ledger = {"n_samples": LEDGER_N_SAMPLES, "local": {"r": led, "ci": [[r - 0.006, r + 0.006] for r in led], "n": [1176273, 1269334, 1371837, 1996191]},
              "global": {"r": [0.88, 0.73, 0.51, 0.10], "ci": [[0.8, 0.9]] * 4, "n": [1276671, 1276680, 1303861, 2045360]}}
    rep = {"local": {"r": led if repro else [0.5] * 4, "n": ledger["local"]["n"]},
           "global": {"r": ledger["global"]["r"], "n": ledger["global"]["n"]}}
    cal = {"pooled": {"tau": 1.0, "calibrated": True}}
    return {"meta": {"n_dataset": LEDGER_N_SAMPLES}, "ledger": ledger, "unit_checks": {"x": units},
            "pe1": {"local": {"spectra": sp, "tau": {"lap_scalar": cal, "lap_vector": cal},
                              "white_chance": {"mean": white[0], "sd": white[1]}}},
            "pe2": {"repro": rep, "share": {"local": {"all": {
                "share": share, "ci": list(share_ci), "chance": chance, "chance_ci": list(chance_ci),
                "excess_ci": [share_ci[0] - chance_ci[1], share_ci[1] - chance_ci[0]], "wrong_pose": 0.5,
                "smooth_basis": 0.5, "vs_smooth_ci": list(smooth_ci),
                "pos_control": pos, "gauss_control": 0.3, "gauss_expected": 0.3, "r_eff_mean": 30, "n_mask_mean": 150}}}},
            "gscd": {"local": {"primary": {"D": D, "D_ci": list(D_ci)}, "perm": {"p975_abs": perm},
                               "within": {"D": D, "D_ci": [within_lo, 0.3]}, "dist_matched": {"D": D / 2},
                               "control_syn": {"D": d_syn}, "control_lap": {"D": 0.04}, "control_iid": {"D": 0.0}}}}


def _u_decide_branches() -> bool:
    ok = []
    r = decide(_syn_result())
    ok += [r["kscs_spectrum"]["verdict"] == "pass", r["kscs_span"]["verdict"] == "pass", r["gscd"]["verdict"] == "pass",
           not r["tool_defect"]]
    r = decide(_syn_result(J=(0.838, 0.45, 0.12, 0.05)))
    ok += [r["kscs_spectrum"]["verdict"] == "kill"]
    r = decide(_syn_result(J=(0.76, 0.52, 0.18, -0.05)))
    ok += [r["kscs_spectrum"]["verdict"] == "descriptive"]
    r = decide(_syn_result(J=(0.80, 0.58, 0.26, 0.10)))
    ok += [r["kscs_spectrum"]["verdict"] == "descriptive"]
    r = decide(_syn_result(share=0.2, share_ci=(0.18, 0.22)))
    ok += [r["kscs_span"]["verdict"] == "kill"]
    r = decide(_syn_result(share=0.5, share_ci=(0.45, 0.55)))
    ok += [r["kscs_span"]["verdict"] == "descriptive"]
    r = decide(_syn_result(smooth_ci=(-0.05, 0.3)))
    ok += [r["kscs_span"]["verdict"] == "descriptive"]
    r = decide(_syn_result(share=0.7, share_ci=(0.6, 0.8), chance=0.7, chance_ci=(0.65, 0.75)))
    ok += [r["kscs_span"]["verdict"] == "descriptive"]
    r = decide(_syn_result(share=0.7, share_ci=(0.69, 0.71), chance=0.8, chance_ci=(0.79, 0.81)))
    ok += [r["kscs_span"]["verdict"] == "kill"]
    r = decide(_syn_result(D=0.03, D_ci=(-0.01, 0.07)))
    ok += [r["gscd"]["verdict"] == "kill"]
    r = decide(_syn_result(D=0.03, D_ci=(-0.01, 0.07), d_syn=0.05))
    ok += [r["gscd"]["verdict"] == "descriptive"]
    r = decide(_syn_result(within_lo=-0.02))
    ok += [r["gscd"]["verdict"] == "descriptive"]
    r = decide(_syn_result(D=0.10, D_ci=(0.06, 0.14)))
    ok += [r["gscd"]["verdict"] == "descriptive"]
    q = _syn_result()
    q["gscd"]["local"]["control_lap"]["D"] = 0.2
    ok += [decide(q)["gscd"]["verdict"] == "descriptive"]
    q = _syn_result()
    q["gscd"]["local"]["control_iid"]["D"] = 0.05
    ok += [bool(decide(q)["tool_defect"])]
    r = decide(_syn_result(units=False))
    ok += [bool(r["tool_defect"]) and r["overall"] == "tool_defect"]
    r = decide(_syn_result(repro=False))
    ok += [bool(r["tool_defect"])]
    r = decide(_syn_result(perm=0.08))
    ok += [bool(r["tool_defect"])]
    r = decide(_syn_result(pos=0.5))
    ok += [bool(r["tool_defect"])]
    r = decide(_syn_result(white=(0.6, 0.01)))
    ok += [bool(r["tool_defect"])]
    return all(ok)


def run_unit_checks(include_ks2_imports: bool = True) -> dict:
    checks = [("hop_estimator_recovers_known_spectrum", _u_hop_known_spectrum),
              ("jmodel_closed_form_equals_monte_carlo", _u_jmodel_closed_form),
              ("tangent_jacobian_matches_finite_difference", _u_jacobian_fd),
              ("span_projection_exact_and_chance_rank_over_n", _u_span_projection),
              ("window_sharing_identity_monotone_disjoint_rotation", _u_window_sharing),
              ("window_taps_equal_torch_grid_sample", _u_window_taps_match_grid_sample),
              ("shot_noise_correlation_equals_weighted_cosine", _u_shot_noise_cosine),
              ("tertile_estimator_recovers_known_dependence_and_null", _u_tertile_estimator),
              ("shuffle_preserves_segment_multiset", _u_shuffle_preserves_segments),
              ("heat_kernel_calibration_hits_target", _u_calibration),
              ("lie_increment_of_identical_states_is_zero", _u_increment_zero),
              ("decide_branches", _u_decide_branches)]
    if include_ks2_imports:
        checks += [("pooled_estimator_equals_ks2_Lin", _u_estimator_equals_ks2),
                   ("hop_pairs_equal_ks2_hop_pairs", _u_hop_pairs_equal_ks2)]
    out, err, tm = {}, {}, {}
    for name, fn in checks:
        t0 = time.time()
        try:
            out[name] = bool(fn())
        except Exception as e:                       # noqa: BLE001 -- a failing check is reported, never raised
            out[name], err[name] = False, f"{type(e).__name__}: {str(e)[:100]}"
        tm[name] = round(time.time() - t0, 2)
    return {"checks": out, "errors": err, "seconds": tm}


# ======================================================================================================== pipeline
def log_(msg: str):
    print(f"[m6 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_pe1(accs: dict, ledger: dict) -> dict:
    T = tables()
    return {dom: pe1_models(accs, dom, ledger[dom]["r"][0], T) for dom in ("local", "global") if any(k[1] == dom for k in accs)}


def run_pe2(S: dict, pe1: dict, ledger: dict, workers: int, n_boot: int, n_perm: int, min_pairs: int = 1000, log=None) -> dict:
    """PE2 + GSCD on a collected SELECT set."""
    T = tables()
    sig = {dom: {k: np.asarray(v) for k, v in pe1[dom]["sigma"].items()} for dom in pe1}
    ana = analyze_select(S, sig, log=log)
    subj, dom, base, grp = ana["tags"]
    A = ana["A"]
    out = {"n_samples": int(len(S["seq"])), "n_mask_vertices_total": int(S["nm"].sum()), "repro": {}, "share": {},
           "j_on_select_masks": {}}
    for dname, dcode in (("local", 0), ("global", 1)):
        if not (dom == dcode).any():
            continue
        rr = ana["hop"].result(dname, n_boot=n_boot)
        out["repro"][dname] = {**rr, "r_ledger": ledger[dname]["r"], "n_ledger": ledger[dname]["n"],
                               "max_abs_diff_r": float(np.nanmax(np.abs(np.asarray(rr["r"]) - np.asarray(ledger[dname]["r"]))))}
        out["share"][dname] = {"all": share_summary(A, ana["tags"], dom == dcode, n_boot),
                               "base": share_summary(A, ana["tags"], (dom == dcode) & base, n_boot)}
        ja = ana["jacc"][dcode]
        res_j = {k: jmodel_spectrum(ja["cov"], ja["vx"], ja["vy"], sig[dname][k]).tolist() for k in SIGMA_VARIANTS
                 if k in sig[dname]}
        L_eig = np.linalg.eigh(T["L"])
        for kind in ("lap_scalar", "lap_vector"):
            cal = calibrate_tau(ja["Wc"], ja["Wv"], ledger[dname]["r"][0], T, kind, L_eig)
            res_j[kind] = null_spectra(ja["Wc"], ja["Wv"], heat_kernel(L_eig, cal["tau"]), T)[kind].tolist()
            res_j[kind + "_calibrated"] = cal["calibrated"]
        out["j_on_select_masks"][dname] = res_j
    ja0 = ana["jacc"][0] if int(ana["jacc"][0]["n_poses"]) else ana["jacc"][1]
    cal = calibrate_tau(ja0["Wc"], ja0["Wv"], ledger["local"]["r"][0], T, "lap_scalar", np.linalg.eigh(T["L"]))
    w_, V_ = np.linalg.eigh(T["L"])
    H_lap = (V_ * np.exp(-0.5 * cal["tau"] * w_)) @ V_.T
    out["gscd_lap_control_tau"] = {"tau": cal["tau"], "calibrated": cal["calibrated"]}
    P = gscd_pairs(S, workers, H_lap=H_lap, log=log)
    out["n_pairs_hop123"] = int(len(P["hop"])) if P else 0
    gs = {}
    if P:
        for dname, dcode in (("local", 0), ("global", 1)):
            if (dom == dcode).any():                       # the gate statistic (local, all sequences) gets the full null
                gs[dname] = gscd_domain(P, grp, dom == dcode, n_boot, n_perm if dname == "local" else min(n_perm, 50), min_pairs)
                gs[dname + "_base"] = gscd_domain(P, grp, (dom == dcode) & base, n_boot, min(n_perm, 50), min_pairs)
    return {"pe2": out, "gscd": gs}


def assemble(unit: dict, ledger: dict, pe1: dict, pe2g: dict, S: dict, meta: dict) -> dict:
    meta = dict(meta, n_dataset=int(S["n_dataset"]), collect_wall_s=float(S["wall_s"]))
    return {"meta": meta, "ledger": ledger, "unit_checks": unit["checks"], "unit_check_seconds": unit["seconds"],
            "pe1": pe1, "pe2": pe2g["pe2"], "gscd": pe2g["gscd"]}


def formal(out: Path, workers: int, stage: str):
    import pickle
    from mv2 import common as MC
    MC.verify_protocol()
    out = Path(out)
    if (out / "result.json").exists():
        raise SystemExit(f"{out}/result.json exists; preserve it")
    (out / "cache").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)
    t0 = time.time()
    unit = run_unit_checks()
    log_(f"unit checks done {time.time() - t0:.0f}s: failed={[k for k, v in unit['checks'].items() if not v]}")
    ledger = load_ledger()
    c_pe1, c_sel = out / "cache" / "pe1.pkl", out / "cache" / "select.npz"
    if c_pe1.exists():
        accs = pickle.loads(c_pe1.read_bytes())
    else:
        sig_pass = pe1_sigma_pass()
        accs = pe1_accumulate(sig_pass, log=log_)
        c_pe1.write_bytes(pickle.dumps(accs))
    pe1 = build_pe1(accs, ledger)
    log_(f"PE1 done {time.time() - t0:.0f}s")
    if stage == "pe1":
        return
    if c_sel.exists():
        z = np.load(c_sel, allow_pickle=True)
        S = {k: z[k] for k in z.files}
        S["n_dataset"], S["wall_s"] = int(S["n_dataset"]), float(S["wall_s"])
    else:
        S = collect_select(SEL_STRIDE, workers, 0, log=log_)
        np.savez(c_sel, **S)
    log_(f"collect done {time.time() - t0:.0f}s")
    if stage == "collect":
        return
    pe2g = run_pe2(S, pe1, ledger, workers, N_BOOT, N_PERM, log=log_)
    meta = {"argv": sys.argv, "seed": SEED, "pe1_stride": PE1_STRIDE, "select_stride": SEL_STRIDE, "n_boot": N_BOOT,
            "n_perm": N_PERM, "rank_cut": RANK_CUT, "head": "patch_r2", "head_radius_m": HEAD_R, "patch": PATCH,
            "wall_s": time.time() - t0}
    result = assemble(unit, ledger, pe1, pe2g, S, meta)
    dec = decide(result)
    MC.save_json(out / "result.json", result)
    MC.save_json(out / "decision.json", dec)
    MC.receipt(out, {"stage": stage, "wall_s": time.time() - t0, "overall": dec["overall"]},
               scripts=(Path(__file__).resolve(), REPO / "tools/kstar/head_probe.py"))
    log_(f"wrote {out}/result.json decision.json overall={dec['overall']} ({time.time() - t0:.0f}s)")


def check_equals_run_stale(pe1: dict, n_rows: int = 32) -> bool:
    """Runs `tools/kstar/head_probe.py::run_stale` (read-only import, KS2's own code) on `n_rows` evenly spaced rows of the
    KS2 sample index and this tool's `collect_select` + `HopAcc` on the same rows; the pooled hop correlations and pair
    counts must agree to 1e-9.  Both are computed here and only the boolean leaves this function."""
    from kstar import common as KC
    from kstar import head_probe as HP
    from torch.utils.data import DataLoader
    d = deps()
    torch.set_num_threads(8)
    dev = torch.device("cpu")
    head, _ = HP.load_head(KC.HEADS.parent, HP.HEAD, dev)
    geo = HP.T.Geo(dev)
    ds = HP.StaleData(d["XC"].subject_seqs(KC.SELECT_SUBJ), SEL_STRIDE, HP.STALE_VARIANTS)
    ds.index = ds.index[::max(1, len(ds.index) // n_rows)][:n_rows]
    with torch.no_grad():
        ref = HP.run_stale(head, geo, DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=HP.collate), dev)
    S = collect_select(SEL_STRIDE, 0, max_n=n_rows)
    sig = {dm: {k: np.asarray(v) for k, v in pe1[dm]["sigma"].items()} for dm in pe1}
    ana = analyze_select(S, sig)
    ok, seen = True, 0
    for dname in ("local", "global"):
        mine = ana["hop"].result(dname, n_boot=5)
        for h in HOPS:
            key = f"all|{dname}|hop{h}|residual"
            if key in ref and mine["n"][h - 1] > 0:
                seen += 1
                ok &= bool(abs(ref[key]["r"] - mine["r"][h - 1]) < 1e-9 and ref[key]["n"] == mine["n"][h - 1])
    return bool(ok and seen >= 4)


def smoke(workers: int = 0):
    """Engineering smoke: the full code path on a tiny slice.  Prints only shapes, counts, timings and unit-check pass /
    fail -- no correlation, share or difference value is printed or saved."""
    import json
    torch.set_num_threads(16)
    rep = {"timings_s": {}, "shapes": {}, "checks": {}}
    t0 = time.time()

    def tick(name):
        rep["timings_s"][name] = round(time.time() - t0, 1)
        print(f"smoke {name}: {rep['timings_s'][name]} s", flush=True)
    T = tables()
    rep["shapes"]["pairs_hop1234"] = [int((T["PH"] == h).sum()) for h in HOPS]
    rep["shapes"]["gain"] = list(T["gain"].shape)
    tick("tables")
    unit = run_unit_checks()
    rep["checks"].update({f"unit:{k}": v for k, v in unit["checks"].items()})
    rep["unit_errors"] = unit["errors"]
    tick("unit_checks")
    ledger = load_ledger()
    rep["checks"]["ledger_matches_text"] = bool(np.allclose(np.round(ledger["local"]["r"], 3), LEDGER_TEXT_LOCAL, atol=5e-4))
    rep["shapes"]["ledger_n_samples"] = ledger["n_samples"]
    # ---- PE1 on a 2-sequence, 6-pose slice
    sig_pass = pe1_sigma_pass(max_seq=2, max_n=6)
    accs = pe1_accumulate(sig_pass, max_seq=2, max_n=6)
    pe1 = build_pe1(accs, ledger)
    rep["shapes"]["pe1_domains"] = sorted(pe1)
    rep["shapes"]["pe1_sigma_len"] = len(pe1["local"]["sigma"]["J_main"])
    rep["shapes"]["pe1_models"] = sorted(pe1["local"]["spectra"])
    rep["shapes"]["pe1_poses"] = {f"{k[0]}_{k[1]}": v["n_poses"] for k, v in accs.items()}
    rep["checks"]["pe1_spectra_finite"] = bool(all(np.isfinite(np.asarray(v["pooled"], float)).all()
                                                    for dom in pe1 for v in pe1[dom]["spectra"].values()))
    tick("pe1_slice")
    # ---- PE2 collection on 8 samples (reads SELECT events, no zgz)
    S = collect_select(SEL_STRIDE, 0, max_n=8)
    rep["shapes"]["collected"] = {"samples": int(len(S["seq"])), "mask_vertices": int(S["nm"].sum()),
                                  "res": list(S["res"].shape), "uv0": list(S["uv0"].shape), "R": list(S["R"].shape)}
    tick("collect_8")
    # geometry reproduces the head's own Geo and the Jacobian's points equal the FK vertices
    mano = T["mano"]
    ok_geo = True
    for i in range(len(S["seq"])):
        a, b = int(S["off"][i]), int(S["off"][i + 1])
        if b - a == 0:
            continue
        betas1 = torch.as_tensor(S["betas"][i], dtype=torch.float64).view(1, -1)
        g = geometry(mano, torch.as_tensor(S["x0"][i:i + 1]), betas1, S["K"][i])
        v = torch.as_tensor(S["vid"][a:b].astype(np.int64))
        ok_geo &= bool(np.abs(g["uv0"][0][v].numpy() - S["uv0"][a:b]).max() < 1e-6)
        ok_geo &= bool(np.abs(g["R"][0][v].numpy() - S["R"][a:b]).max() < 1e-5)
        ok_geo &= bool(np.abs(g["cos"][0][v].numpy() - S["cos"][a:b]).max() < 1e-5)
        X, _ = tangent_jacobian(mano, torch.as_tensor(S["x0"][i:i + 1]), betas1, torch.zeros(len(v), dtype=torch.long), v)
        ok_geo &= bool((X - g["V0"][0][v]).abs().max() < 1e-7)
    rep["checks"]["geometry_matches_Geo_and_J_points_equal_FK"] = bool(ok_geo)
    # ---- PE2 analysis + GSCD on the 8 samples (small bootstrap / permutation counts)
    pe2g = run_pe2(S, pe1, ledger, workers=1, n_boot=20, n_perm=5, min_pairs=50)
    rep["shapes"]["pe2_keys"] = sorted(pe2g["pe2"])
    rep["shapes"]["gscd_keys"] = {k: sorted(v) for k, v in pe2g["gscd"].items()}
    rep["shapes"]["n_pairs_hop123"] = pe2g["pe2"]["n_pairs_hop123"]
    rep["checks"]["gscd_ran"] = bool(pe2g["gscd"]) and all("primary" in v for k, v in pe2g["gscd"].items()
                                                           if v and not k.endswith("_base"))
    shs = [v["all"] for v in pe2g["pe2"]["share"].values() if "share" in v["all"]]
    rep["checks"]["span_controls_on_slice"] = bool(shs) and all(
        c["pos_control"] >= SHARE_POS_MIN and abs(c["gauss_control"] - c["gauss_expected"]) <= 0.05 for c in shs)
    tick("pe2_gscd_slice")
    # ---- exact equivalence with KS2's own accumulation on the same rows (values compared, never printed)
    rep["checks"]["recompute_equals_head_probe_run_stale_on_slice"] = check_equals_run_stale(pe1, n_rows=32)
    tick("equivalence_with_run_stale")
    # ---- known-structure controls through the real-geometry code path (synthetic residuals, 96 slice samples)
    S96 = collect_select(SEL_STRIDE, 8, max_n=96)
    sigd = {d: {k: np.asarray(v) for k, v in pe1[d]["sigma"].items()} for d in pe1}
    sub96, dom96, base96, grp96 = sample_tags(S96)
    okc = {}
    for kind in ("J", "smooth"):
        ana = analyze_select(synthetic_residuals(S96, sigd, kind), sigd)
        sh = share_summary(ana["A"], ana["tags"], np.ones(len(dom96), bool), n_boot=20)
        okc[kind] = sh
    rep["checks"]["span_J_synthetic_high_share_and_beats_smooth_basis"] = bool(
        okc["J"]["share"] >= SHARE_POS_MIN and okc["J"]["vs_smooth"] > 0 and okc["J"]["chance"] < 0.5)
    rep["checks"]["span_smooth_synthetic_loses_to_smooth_basis"] = bool(okc["smooth"]["vs_smooth"] < 0)
    P96 = gscd_pairs(S96, 8, H_lap=(lambda w, V: (V * np.exp(-0.5 * 1.0 * w)) @ V.T)(*np.linalg.eigh(T["L"])))
    gsl = gscd_domain(P96, grp96, np.ones(len(dom96), bool), 20, 5, 50)
    rep["checks"]["gscd_window_noise_control_powered"] = bool(gsl["control_syn"]["D"] >= D_PASS)
    rep["checks"]["gscd_iid_control_near_zero"] = bool(abs(gsl["control_iid"]["D"]) < 0.03)
    rep["checks"]["gscd_shuffle_null_small"] = bool(gsl["perm"]["max_abs"] < 0.03)
    rep["shapes"]["slice96"] = {"samples": int(len(S96["seq"])), "pairs_hop123": int(len(P96["hop"]))}
    tick("synthetic_controls")
    # ---- cache round trips (formal-mode IO) in the smoke directory
    import pickle
    import tempfile
    from mv2 import common as MC
    with tempfile.TemporaryDirectory(dir=str(REPO / "outputs" / "mv2" / "smoke" / "m6")) as td:
        np.savez(Path(td) / "select.npz", **S)
        z = np.load(Path(td) / "select.npz", allow_pickle=True)
        rep["checks"]["select_cache_roundtrip"] = bool(all(np.array_equal(z[k], np.asarray(S[k])) for k in S))
        pk = Path(td) / "pe1.pkl"
        pk.write_bytes(pickle.dumps(accs))
        back = pickle.loads(pk.read_bytes())
        rep["checks"]["pe1_cache_roundtrip"] = bool(all(np.array_equal(back[k]["cov"], accs[k]["cov"]) for k in accs))
    # ---- assemble + decide runs end to end (verdicts are NOT printed: the slice is not a measurement)
    result = assemble(unit, ledger, pe1, pe2g, S, {"smoke": True})
    import json as _json
    rep["checks"]["result_json_serializable"] = bool(len(_json.dumps(result, default=MC._np)) > 0)
    try:
        if "local" in result["gscd"] and "local" in result["pe1"]:
            dec = decide(result)
            rep["shapes"]["decide_keys"] = sorted(dec)
            rep["checks"]["decide_runs_on_slice"] = True
        else:
            rep["checks"]["decide_runs_on_slice"] = "local domain absent in the 8-sample slice (decide exercised by unit check)"
    except Exception as e:                                # noqa: BLE001
        rep["checks"]["decide_runs_on_slice"] = False
        rep["unit_errors"]["decide_slice"] = f"{type(e).__name__}"
    tick("decide")
    bad = [k for k, v in rep["checks"].items() if v is False]
    rep["all_pass"] = not bad
    rep["failed"] = bad
    outdir = REPO / "outputs" / "mv2" / "smoke" / "m6"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "smoke.json").write_text(json.dumps(rep, indent=1, sort_keys=True, default=str))
    print("SMOKE", "OK" if not bad else f"FAILED {bad}")
    for k, v in sorted(rep["checks"].items()):
        print(f"  {k}: {'PASS' if v is True else v}")
    print("shapes:", json.dumps(rep["shapes"], default=str))
    print(f"total {time.time() - t0:.1f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "outputs" / "mv2" / "m6"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stage", choices=("all", "pe1", "collect"), default="all",
                    help="pe1 / collect only fill the caches under <out>/cache; all also analyses and writes result.json")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.smoke:
        smoke()
        return
    formal(Path(a.out), a.workers, a.stage)


if __name__ == "__main__":
    main()
