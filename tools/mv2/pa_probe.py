#!/usr/bin/env python3
"""MV2 zero-training probes of M1 (surface-lifted readout) and M2 (ray-gauge quotient / metric-scale neighbourhood).

docs/EXPERIMENT_RECORDS.md#mv2-prereg-20261026 sections M1 / M2. Frozen S39 (3 seeds, step 6000) on the FIT7 and SELECT
base recordings, the evaluator's W200 single step (prev = GT(end - 50 ms), `tools/mv2/common.py`).

  pa1   ladder of closed-form readouts of the frozen trunk's layer1-3 maps for the target GT(end) - prev, with prev =
        GT(end - 50) + S39's own training prev-noise ("mixed", seeded per sample). Arms (all next to the GAP block):
        gap | rootanc21 | tile (registered grid around the FK wrist) | lift_mean | lift_vert | lift_win | lift_ring |
        lift_lbs; controls vshuffle / shift10 / tshuffle / noise on lift_lbs. Dual ridge fitted on FIT7, lambda by
        leave-one-FIT7-subject-out CV on root-aligned joint error, evaluated on SELECT as E3 RA of prev + prediction.
  pa2   position information: prev offsets (image-plane 5/10/14/19/25 mm, depth 45 mm, root 5 / 11 deg) re-run through
        S39; readouts gap / tile / lift_lbs fitted on FIT7 with offset augmentation (p 0.8); b_off = 1 - carry of the
        readout's corrected state, per condition, plus S39's own 1 - carry as the ceiling reference.
  pa3   descriptive: "motion-illuminated contour" proxy vs event count as predictors of whether S39's single step helps
        a subtree (AUC), SELECT only.
  rgq   M2 PA-RGQ: prev scaled about the wrist ray (sigma 0.94 .. 1.06; and a "small hand" betas shape), S39's
        articulation (root-aligned) response in mm and its t_z response.
  mscn  M2 PA5: events and prev render zoomed about the principal point (s 0.94 .. 1.06); drift of per-vertex features
        read with a fixed 6 px window / fixed layer / metric 10 mm continuous scale selection, over the A/A drift.

    python tools/mv2/pa_probe.py <probe> --smoke            # shapes and timings only
    python tools/mv2/pa_probe.py <probe> --out outputs/mv2/<probe> [--seeds 3407 3408 3409] [--workers 3]
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mv2 import common as C  # noqa: E402

RNG_SEED = 20261026
EVERY = 8
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)        # x trace(K) / n
N_BOOT = 2000
BLOCK_MS = 1000
TILE_N = 9                                             # TILE_N x TILE_N registered grid
WIN_MM = 10.0
PA2_CONDS = ("clean", "t5", "t10", "t14", "t19", "t25", "z45", "r5", "r11")
SIGMAS = (0.94, 0.97, 1.00, 1.03, 1.06)
ZOOMS = (0.94, 0.97, 1.03, 1.06)
PROJ_DIM = 64


# =============================================================================================== sample construction
def seq_hash(seq: str) -> int:
    return int.from_bytes(seq.encode()[:8].ljust(8, b"\0"), "little") % (2 ** 31)


def mixed_noise(rng: np.random.Generator, track: dict) -> np.ndarray:
    """`SemKineDataset._sample_prev_noise` for PREV_NOISE_MODE mixed (S39's training prev noise)."""
    st, sr, sp = (float(track[k]) for k in ("PREV_NOISE_T", "PREV_NOISE_R", "PREV_NOISE_POSE"))
    lt, lr, lp = (float(track[k]) for k in ("PREV_NOISE_LARGE_T", "PREV_NOISE_LARGE_R", "PREV_NOISE_LARGE_POSE"))
    ps, pl = float(track["PREV_NOISE_P_SMALL"]), float(track["PREV_NOISE_P_LARGE"])

    def fill(a, b, c):
        n = np.empty(51, np.float32)
        n[0:3] = rng.standard_normal(3) * a
        n[3:6] = rng.standard_normal(3) * b
        n[6:51] = rng.standard_normal(45) * c
        return n
    u = float(rng.random())
    if u < ps:
        return fill(st, sr, sp)
    if u < ps + pl:
        return fill(lt, lr, lp)

    def directed(n, mag):
        v = rng.standard_normal(n).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8) * np.float32(rng.uniform(0.0, mag))
    n = np.empty(51, np.float32)
    n[0:3], n[3:6], n[6:51] = directed(3, lt), directed(3, lr), directed(45, lp)
    return n


def aa_compose(rot_aa, aa):
    from scipy.spatial.transform import Rotation as Rr
    return (Rr.from_rotvec(rot_aa) * Rr.from_rotvec(aa)).as_rotvec().astype(np.float32)


def wrist_xyz(model, P: np.ndarray, betas: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        b = torch.tensor(betas, dtype=torch.float32).view(1, -1).expand(len(P), -1)
        _, j = model._fk(torch.from_numpy(np.asarray(P, np.float32)), b)
    return j[:, 0].numpy().astype(np.float64)


def perturb(model, P: np.ndarray, betas: np.ndarray, cond: str, seq: str, idx: np.ndarray) -> np.ndarray:
    """PA2 offsets of the prev states P (N, 51): seeded per (seq, sample, condition)."""
    if cond == "clean":
        return P.copy()
    out = P.copy()
    w = wrist_xyz(model, P, betas)
    ray = w / np.linalg.norm(w, axis=1, keepdims=True)
    cid = PA2_CONDS.index(cond)
    for k, i in enumerate(idx):
        rng = np.random.default_rng([RNG_SEED, seq_hash(seq), int(i), cid])
        if cond.startswith("t"):
            d = rng.standard_normal(3)
            d -= (d @ ray[k]) * ray[k]
            d /= np.linalg.norm(d)
            out[k, 0:3] += (d * float(cond[1:]) / 1000.0).astype(np.float32)
        elif cond == "z45":
            out[k, 0:3] += (ray[k] * 0.045 * (1 if rng.random() < 0.5 else -1)).astype(np.float32)
        elif cond.startswith("r"):
            ax = rng.standard_normal(3)
            ax /= np.linalg.norm(ax)
            out[k, 3:6] = aa_compose(ax * np.deg2rad(float(cond[1:])), P[k, 3:6])
    return out


def scale_about_wrist(model, P: np.ndarray, betas: np.ndarray, sigma: float) -> np.ndarray:
    """Move the hand rigidly along its wrist ray so the wrist sits at w / sigma (image size x ~sigma about the wrist)."""
    out = P.copy()
    if sigma == 1.0:
        return out
    w = wrist_xyz(model, P, betas)
    out[:, 0:3] += (w / sigma - w).astype(np.float32)
    return out


# =============================================================================================== features
class Featurizer:
    """Readout features of the frozen trunk's maps at the FK surface of a prev state (all CPU float32)."""

    def __init__(self, model, seed_proj: int = RNG_SEED):
        self.m = model
        mt = C.mesh_tables(model)
        self.lbs = mt["lbs"]                                       # (778, 16)
        self.ring_idx, self.ring_mask = mt["ring_idx"], mt["ring_mask"]
        g = torch.Generator().manual_seed(seed_proj)
        self.proj = {t: torch.linalg.qr(torch.randn(c, PROJ_DIM, generator=g))[0]
                     for t, c in (("layer1", 64), ("layer2", 128), ("layer3", 256))}

    @torch.no_grad()
    def surface(self, P, betas, K):
        return C.fk_surface(self.m, P, torch.tensor(betas), torch.tensor(K))

    @staticmethod
    def sample_all(taps, uv):
        """(B, N, 448) concatenated layer1-3 samples at render pixels uv (B, N, 2)."""
        return torch.cat([C.sample_map(taps[t].float(), uv, C.TAP_STRIDE[t]) for t in C.TAPS], -1)

    def inb(self, srf):
        uv, z = srf["uv"], srf["z"]
        return (z > 1e-6) & (uv[..., 0] >= -0.5) & (uv[..., 0] < C.W - 0.5) & (uv[..., 1] >= -0.5) & \
               (uv[..., 1] < C.H - 0.5)

    def window_uv(self, srf, mm: float = WIN_MM):
        fx = srf["intr"][0]
        rho = fx[:, None] * (mm / 1000.0) / srf["z"].clamp(min=1e-3)
        uv = srf["uv"]
        offs = [torch.zeros_like(uv)]
        for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            offs.append(torch.stack([rho * du, rho * dv], -1))
        return [uv + o for o in offs]

    def arms(self, taps, srf, gap, which, rng: np.random.Generator):
        """dict arm -> (B, D) float32 features (the arm's own block; GAP is concatenated by the caller)."""
        B = gap.shape[0]
        out = {}
        vis = srf["vis"] & self.inb(srf)
        a = vis.float()                                                      # (B, 778)
        uv = srf["uv"]
        need_vert = any(w in which for w in ("lift_mean", "lift_vert", "lift_ring", "lift_lbs", "lift_win",
                                              "ctl_vshuffle", "ctl_shift10", "ctl_tshuffle", "ctl_noise"))
        if "rootanc21" in which:
            j = srf["joints"]
            fx, fy, cx, cy = srf["intr"]
            zj = j[..., 2].clamp(min=1e-6)
            uvj = torch.stack([fx[:, None] * j[..., 0] / zj + cx[:, None], fy[:, None] * j[..., 1] / zj + cy[:, None]], -1)
            inbj = ((j[..., 2] > 1e-6) & (uvj[..., 0] >= -0.5) & (uvj[..., 0] < C.W - 0.5) & (uvj[..., 1] >= -0.5)
                    & (uvj[..., 1] < C.H - 0.5)).float()
            s2 = C.sample_map(taps["layer2"].float(), uvj, 8) * inbj.unsqueeze(-1)
            pos = torch.stack([2 * uvj[..., 0] / (C.W - 1) - 1, 2 * uvj[..., 1] / (C.H - 1) - 1], -1).clamp(-2, 2)
            out["rootanc21"] = torch.cat([s2, pos, inbj.unsqueeze(-1)], -1).reshape(B, -1)
        if "tile" in which:
            wr = uv.new_zeros(B, 2)
            j0 = srf["joints"][:, 0]
            fx, fy, cx, cy = srf["intr"]
            wr[:, 0] = fx * j0[:, 0] / j0[:, 2].clamp(min=1e-6) + cx
            wr[:, 1] = fy * j0[:, 1] / j0[:, 2].clamp(min=1e-6) + cy
            rad = ((uv - wr[:, None]).norm(dim=-1) * a).amax(1).clamp(min=8.0)   # hand radius in px (visible verts)
            g = torch.linspace(-1, 1, TILE_N)
            gy, gx = torch.meshgrid(g, g, indexing="ij")
            grid = torch.stack([gx, gy], -1).reshape(1, -1, 2) * rad[:, None, None] + wr[:, None]
            out["tile"] = self.sample_all(taps, grid).reshape(B, -1)
        if need_vert:
            v0 = self.sample_all(taps, uv) * a.unsqueeze(-1)                    # (B, 778, 448)
            if "lift_mean" in which:
                out["lift_mean"] = torch.cat([v0.sum(1) / a.sum(1, keepdim=True).clamp(min=1), a.mean(1, keepdim=True)],
                                             -1)
            if "lift_vert" in which:
                out["lift_vert"] = v0.reshape(B, -1)
            vw = torch.stack([self.sample_all(taps, u) for u in self.window_uv(srf)], 0).mean(0) * a.unsqueeze(-1)
            if "lift_win" in which:
                out["lift_win"] = vw.reshape(B, -1)
            if "lift_ring" in which:
                K = self.ring_idx.shape[1]
                nb = vw[:, self.ring_idx.reshape(-1)].view(B, 778, K, -1)
                wgt = (self.ring_mask.view(1, 778, K) * a[:, self.ring_idx.reshape(-1)].view(B, 778, K)).unsqueeze(-1)
                ring = (nb * wgt).sum(2) + vw
                out["lift_ring"] = (ring / (wgt.sum(2) + 1.0)).reshape(B, -1) * 1.0
            if "lift_lbs" in which:
                out["lift_lbs"] = self.lbs_pool(vw, a)
            if "ctl_vshuffle" in which:
                perm = torch.stack([torch.from_numpy(rng.permutation(778)) for _ in range(B)])
                vs = torch.gather(vw, 1, perm.unsqueeze(-1).expand(-1, -1, vw.shape[-1]))
                as_ = torch.gather(a, 1, perm)
                out["ctl_vshuffle"] = self.lbs_pool(vs, as_)
            if "ctl_shift10" in which:
                fx = srf["intr"][0]
                ang = torch.from_numpy(rng.uniform(0, 2 * math.pi, size=B)).float()
                sh = fx * 0.010 / srf["joints"][:, 0, 2].clamp(min=1e-3)
                d = torch.stack([torch.cos(ang), torch.sin(ang)], -1) * sh[:, None]
                srf2 = dict(srf)
                srf2["uv"] = srf["uv"] + d[:, None]
                vsh = torch.stack([self.sample_all(taps, u) for u in self.window_uv(srf2)], 0).mean(0) * a.unsqueeze(-1)
                out["ctl_shift10"] = self.lbs_pool(vsh, a)
            if "ctl_noise" in which:
                out["ctl_noise"] = torch.from_numpy(rng.standard_normal((B, 16 * 448)).astype(np.float32))
        return out

    def lbs_pool(self, v, a):
        wj = self.lbs.unsqueeze(0) * a.unsqueeze(-1)                          # (B, 778, 16)
        mass = wj.sum(1)
        ej = torch.einsum("bvj,bvd->bjd", wj, v) / mass.clamp(min=1e-6).unsqueeze(-1)
        cov = (mass / self.lbs.sum(0).clamp(min=1e-6)).unsqueeze(-1)
        return torch.cat([ej, cov], -1).reshape(v.shape[0], -1)


def run_s39(model, X, P, betas, K, chunk: int = 64, taps=C.TAPS):
    """S39 out51, taps (CPU float32), GAP (B, 256); betas / K set as the hand context."""
    model.set_hand_context(torch.tensor(betas).view(1, -1), torch.tensor(K).view(1, 3, 3))
    outs, tps, gaps = [], {t: [] for t in taps}, []
    for i in range(0, len(P), chunk):
        x = torch.from_numpy(X[i:i + chunk])
        p = torch.from_numpy(np.asarray(P[i:i + chunk], np.float32))
        o, got, _ = C.s39_forward_taps(model, x, p, taps)
        outs.append(o)
        for t in taps:
            tps[t].append(got[t])
        gaps.append(got["layer3"].float().mean((2, 3)))
    return torch.cat(outs), {t: torch.cat(v) for t, v in tps.items()}, torch.cat(gaps)


# =============================================================================================== ridge machinery
class KernelBank:
    """Per-arm linear kernels over all samples (train + test rows), built block-wise from features standardised with
    the FIT7 rows' statistics; the GAP block and the arm block each scaled to unit mean squared norm per row."""

    def __init__(self):
        self.parts = {}

    @staticmethod
    def standardise(Z: np.ndarray, train_rows: np.ndarray) -> np.ndarray:
        mu = Z[train_rows].mean(0)
        sd = Z[train_rows].std(0) + 1e-6
        Z = (Z - mu) / sd
        return Z / math.sqrt(Z.shape[1])

    @staticmethod
    def kernel(blocks) -> np.ndarray:
        K = None
        for Z in blocks:
            Zt = torch.from_numpy(np.ascontiguousarray(Z, dtype=np.float32))
            k = (Zt @ Zt.T).double().numpy()
            K = k if K is None else K + k
        return K


def ridge_kernel_predict(K: np.ndarray, Y: np.ndarray, tr: np.ndarray, te: np.ndarray, lam_rel: float) -> np.ndarray:
    Ktr = K[np.ix_(tr, tr)]
    n = len(tr)
    H = np.eye(n) - 1.0 / n
    Kc = H @ Ktr @ H
    lam = lam_rel * np.trace(Kc) / n
    my = Y[tr].mean(0)
    alpha = np.linalg.solve(Kc + lam * np.eye(n), Y[tr] - my)
    Kte = K[np.ix_(te, tr)]
    Kte_c = (Kte - Kte.mean(1, keepdims=True)) - (Ktr.mean(0, keepdims=True) - Ktr.mean())
    return Kte_c @ alpha + my


def eff_df(K: np.ndarray, tr: np.ndarray, lam_rel: float) -> float:
    Ktr = K[np.ix_(tr, tr)]
    n = len(tr)
    H = np.eye(n) - 1.0 / n
    s = np.clip(np.linalg.eigvalsh(H @ Ktr @ H), 0, None)
    lam = lam_rel * s.sum() / n
    return float((s / (s + lam)).sum())


def lam_for_df(K: np.ndarray, tr: np.ndarray, df_target: float) -> float:
    Ktr = K[np.ix_(tr, tr)]
    n = len(tr)
    H = np.eye(n) - 1.0 / n
    s = np.clip(np.linalg.eigvalsh(H @ Ktr @ H), 0, None)
    lo, hi = 1e-8, 1e4
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        df = (s / (s + mid * s.sum() / n)).sum()
        lo, hi = (mid, hi) if df > df_target else (lo, mid)
    return math.sqrt(lo * hi)


class RA:
    """Root-aligned joint error (mm) of 51-d states against GT, per sample, with each sample's betas."""

    def __init__(self, model):
        self.m = model

    def joints(self, P: np.ndarray, betas_rows: np.ndarray) -> np.ndarray:
        out = np.zeros((len(P), 21, 3), np.float32)
        with torch.no_grad():
            for b in np.unique(betas_rows, axis=0):
                rows = np.flatnonzero((betas_rows == b).all(1))
                for i in range(0, len(rows), 1024):
                    r = rows[i:i + 1024]
                    bb = torch.from_numpy(b.astype(np.float32)).view(1, -1).expand(len(r), -1)
                    _, j = self.m._fk(torch.from_numpy(np.asarray(P[r], np.float32)), bb)
                    out[r] = (j - j[:, :1]).numpy()
        return out

    def err(self, P, G, betas_rows):
        return np.linalg.norm(self.joints(P, betas_rows) - self.joints(G, betas_rows), axis=-1).mean(-1) * 1000.0


def subject_cv_lambda(K, Y, prev, gt, betas_rows, subj, tr, ra: RA):
    """lambda (relative) minimising leave-one-FIT7-subject-out mean RA of prev + prediction."""
    best, scores = None, {}
    for lam in LAMBDAS:
        errs = []
        for s in np.unique(subj[tr]):
            te_f = tr[subj[tr] == s]
            tr_f = tr[subj[tr] != s]
            if len(tr_f) < 2:
                continue
            pred = ridge_kernel_predict(K, Y, tr_f, te_f, lam)
            errs.append(ra.err(prev[te_f] + pred, gt[te_f], betas_rows[te_f]))
        scores[lam] = float(np.concatenate(errs).mean())
        if best is None or scores[lam] < scores[best]:
            best = lam
    return best, scores


def boot_mean_ci(v, groups, seed=RNG_SEED):
    lo, hi = C.block_bootstrap_ci(v, groups, N_BOOT, seed)
    rng = np.random.default_rng(seed + 1)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    bs = np.array([np.mean(np.concatenate([v[idx[g]] for g in rng.choice(ug, len(ug))])) for _ in range(N_BOOT)])
    return {"mean": float(v.mean()), "ci": [lo, hi], "p_ge0": float((bs >= 0).mean()), "n": int(len(v)),
            "n_groups": int(len(ug))}


# =============================================================================================== data gathering
def gather(seqs, every, max_n=None):
    out = []
    for s in seqs:
        ss = C.SeqSamples(s, every=every, max_n=max_n)
        if len(ss):
            out.append(ss)
    return out


def rows_meta(samples):
    seq, subj, dom, ends, betas = [], [], [], [], []
    for ss in samples:
        n = len(ss)
        seq += [ss.seq] * n
        subj += [C.subject_of(ss.seq)] * n
        dom += ["local" if ss.seq.endswith("_local") else "global"] * n
        ends.append(ss.ends)
        betas.append(np.repeat(ss.betas[None], n, 0))
    return (np.array(seq), np.array(subj), np.array(dom), np.concatenate(ends), np.concatenate(betas))


# =============================================================================================== PA1
PA1_ARMS = ("gap", "rootanc21", "tile", "lift_mean", "lift_vert", "lift_win", "lift_ring", "lift_lbs",
            "ctl_vshuffle", "ctl_shift10", "ctl_tshuffle", "ctl_noise")


def pa1_seed(seed: int, smoke: bool, threads: int):
    torch.set_num_threads(threads)
    model, cfg = C.load_s39(seed, torch.device("cpu"))
    fz = Featurizer(model)
    seqs = C.base_seqs(C.FIT7 + C.SELECT) if not smoke else ["ch_local", "lfz_local", "ycy_local"]
    samples = gather(seqs, EVERY, max_n=24 if smoke else None)
    seq, subj, dom, ends, betas = rows_meta(samples)
    track = cfg["TRACK"]
    feats = {a: [] for a in PA1_ARMS if a not in ("gap", "ctl_tshuffle")}
    gaps, prevs, gts, outs = [], [], [], []
    t0 = time.time()
    for ss in samples:
        idx = np.arange(len(ss))
        P = np.stack([ss.prev[i] + mixed_noise(np.random.default_rng([RNG_SEED, seq_hash(ss.seq), int(i), 99]), track)
                      for i in idx]).astype(np.float32)
        X = ss.lnes(idx)
        out, taps, gap = run_s39(model, X, P, ss.betas, ss.K)
        rng = np.random.default_rng([RNG_SEED, seq_hash(ss.seq), 7])
        for i0 in range(0, len(idx), 64):
            sl = slice(i0, i0 + 64)
            srf = fz.surface(torch.from_numpy(P[sl]), ss.betas, ss.K)
            tp = {t: v[sl] for t, v in taps.items()}
            fa = fz.arms(tp, srf, gap[sl], tuple(feats), rng)
            for k, v in fa.items():
                feats[k].append(v.numpy().astype(np.float16))
        gaps.append(gap.numpy())
        prevs.append(P)
        gts.append(ss.gt)
        outs.append(out.numpy())
    feats = {k: np.concatenate(v) for k, v in feats.items()}
    gap = np.concatenate(gaps)
    prev, gt, s39 = np.concatenate(prevs), np.concatenate(gts), np.concatenate(outs)
    t_feat = time.time() - t0
    tr = np.flatnonzero(np.isin(subj, C.FIT7))
    te = np.flatnonzero(np.isin(subj, C.SELECT))
    rng = np.random.default_rng([RNG_SEED, seed, 11])
    perm = np.arange(len(gap))
    for s in np.unique(seq):                                   # tshuffle: features of another sample of the same sequence
        r = np.flatnonzero(seq == s)
        perm[r] = rng.permutation(r)
    Y = (gt - prev).astype(np.float64)
    gap_z = KernelBank.standardise(gap.astype(np.float64), tr)
    Kg = KernelBank.kernel([gap_z])
    ra = RA(model)
    res = {"seed": seed, "n_train": int(len(tr)), "n_test": int(len(te)), "t_features_s": t_feat, "arms": {}}
    if smoke:
        res["shapes"] = {k: list(v.shape) for k, v in feats.items()}
    pred_rows = {}
    for arm in PA1_ARMS:
        if arm == "gap":
            K = Kg
        elif arm == "ctl_tshuffle":
            z = KernelBank.standardise(feats["lift_lbs"].astype(np.float32), tr)[perm]
            K = Kg + KernelBank.kernel([z])
        else:
            z = KernelBank.standardise(feats[arm].astype(np.float32), tr)
            K = Kg + KernelBank.kernel([z])
            del z
        lam, cv = subject_cv_lambda(K, Y, prev, gt, betas, subj, tr, ra)
        pred = ridge_kernel_predict(K, Y, tr, te, lam)
        e = ra.err(prev[te] + pred, gt[te], betas[te])
        res["arms"][arm] = {"lambda_rel": lam, "cv_ra": cv, "df": eff_df(K, tr, lam), "dim": int(feats[arm].shape[1])
                            if arm in feats else (int(gap.shape[1]) if arm == "gap" else int(feats["lift_lbs"].shape[1]))}
        pred_rows[arm] = {"err": e, "K": K if arm in ("tile", "lift_lbs") else None, "lam": lam}
        res["arms"][arm]["blocks_r2"] = {}
        for name, sl in (("transl", slice(0, 3)), ("root", slice(3, 6)), ("fingers", slice(6, 51))):
            ss_res = ((Y[te][:, sl] - pred[:, sl]) ** 2).sum()
            ss_tot = ((Y[te][:, sl] - Y[tr][:, sl].mean(0)) ** 2).sum()
            res["arms"][arm]["blocks_r2"][name] = float(1 - ss_res / max(ss_tot, 1e-12))
        if arm != "gap":
            del K
    # df-matched primary contrast: lift_lbs at the CV-optimal df of tile
    Kt, Kl = pred_rows["tile"]["K"], pred_rows["lift_lbs"]["K"]
    df_ref = eff_df(Kt, tr, pred_rows["tile"]["lam"])
    lam_l = lam_for_df(Kl, tr, df_ref)
    pred_l = ridge_kernel_predict(Kl, Y, tr, te, lam_l)
    err_l_df = ra.err(prev[te] + pred_l, gt[te], betas[te])
    hold = ra.err(prev[te], gt[te], betas[te])
    s39e = ra.err(s39[te], gt[te], betas[te])
    res["df_matched"] = {"df_ref_tile": df_ref, "lambda_rel_lift_lbs": lam_l, "df_lift_lbs": eff_df(Kl, tr, lam_l)}
    rows = {"seq": seq[te], "dom": dom[te], "end": ends[te], "hold": hold, "s39": s39e,
            "lift_lbs_dfmatched": err_l_df}
    rows.update({f"err_{a}": pred_rows[a]["err"] for a in PA1_ARMS})
    return res, rows


def pa1(args):
    seeds = args.seeds
    if args.smoke:
        res, rows = pa1_seed(seeds[0], True, args.threads)
        print("PA1 smoke:", json.dumps({"n_train": res["n_train"], "n_test": res["n_test"], "shapes": res["shapes"],
                                        "t_features_s": round(res["t_features_s"], 1),
                                        "arms_fitted": list(res["arms"])}))
        return
    C.verify_protocol()
    with mp.get_context("spawn").Pool(len(seeds)) as pool:
        outs = pool.starmap(pa1_seed, [(s, False, args.threads) for s in seeds])
    result = {"probe": "pa1", "seeds": {}, "contrast": {}}
    per_seed_rows = {}
    for (res, rows), s in zip(outs, seeds):
        result["seeds"][str(s)] = res
        per_seed_rows[s] = rows
        np.savez_compressed(Path(args.out) / f"rows_s{s}.npz", **rows)
    result["contrast"] = pa1_contrast(per_seed_rows)
    C.save_json(Path(args.out) / "result.json", result)
    C.receipt(args.out, {"probe": "pa1", "seeds": seeds}, scripts=[__file__, C.__file__])


def groups_of(rows):
    return np.array([f"{s}|{int(e) // BLOCK_MS}" for s, e in zip(rows["seq"], rows["end"])])


def pa1_contrast(per_seed_rows):
    """Primary: err(lift_lbs) - err(tile) per SELECT sample, averaged over seeds; per domain; Holm over 2 domains.
    Also every arm vs gap and vs tile (descriptive), per seed means, df-matched version."""
    seeds = sorted(per_seed_rows)
    r0 = per_seed_rows[seeds[0]]
    g = groups_of(r0)
    out = {}
    for dom in ("local", "global", "all"):
        m = np.ones(len(g), bool) if dom == "all" else (r0["dom"] == dom)
        d = {}
        for a in PA1_ARMS:
            if a == "gap":
                continue
            for ref in ("gap", "tile"):
                if a == ref:
                    continue
                diffs = np.stack([per_seed_rows[s][f"err_{a}"][m] - per_seed_rows[s][f"err_{ref}"][m] for s in seeds])
                d[f"{a}-{ref}"] = boot_mean_ci(diffs.mean(0), g[m]) | {"per_seed_mean": diffs.mean(1).tolist()}
        for a, ref in (("lift_lbs", "ctl_vshuffle"), ("lift_lbs", "ctl_shift10")):
            diffs = np.stack([per_seed_rows[s][f"err_{a}"][m] - per_seed_rows[s][f"err_{ref}"][m] for s in seeds])
            d[f"{a}-{ref}"] = boot_mean_ci(diffs.mean(0), g[m]) | {"per_seed_mean": diffs.mean(1).tolist()}
        diffs = np.stack([per_seed_rows[s]["lift_lbs_dfmatched"][m] - per_seed_rows[s]["err_tile"][m] for s in seeds])
        d["lift_lbs_dfmatched-tile"] = boot_mean_ci(diffs.mean(0), g[m]) | {"per_seed_mean": diffs.mean(1).tolist()}
        for name in ("hold", "s39") + tuple(f"err_{a}" for a in PA1_ARMS):
            d[f"mean_{name}"] = float(np.mean([per_seed_rows[s][name][m].mean() for s in seeds]))
        out[dom] = d
    return out


# =============================================================================================== PA2
PA2_ARMS = ("gap", "tile", "lift_lbs")


def pa2_seed(seed: int, smoke: bool, threads: int):
    torch.set_num_threads(threads)
    model, cfg = C.load_s39(seed, torch.device("cpu"))
    fz = Featurizer(model)
    seqs = C.base_seqs(C.FIT7 + C.SELECT) if not smoke else ["ch_local", "lfz_local", "ycy_local"]
    samples = gather(seqs, EVERY, max_n=16 if smoke else None)
    rows = {"seq": [], "subj": [], "dom": [], "end": [], "cond": [], "betas": [], "prev": [], "prev0": [], "gt": [],
            "s39": [], "s39_0": []}
    feats = {a: [] for a in PA2_ARMS}
    t0 = time.time()
    for ss in samples:
        subj = C.subject_of(ss.seq)
        idx = np.arange(len(ss))
        X = ss.lnes(idx)
        P0 = ss.prev.copy()
        if subj in C.FIT7:                                 # augmentation: each FIT7 sample gets one condition (p 0.8 offset)
            rng = np.random.default_rng([RNG_SEED, seq_hash(ss.seq), 5])
            conds = np.where(rng.random(len(idx)) < 0.8, rng.integers(1, len(PA2_CONDS), len(idx)), 0)
            plan = [(c, np.flatnonzero(conds == k)) for k, c in enumerate(PA2_CONDS)]
        else:
            plan = [(c, idx) for c in PA2_CONDS]
        out0, _, _ = run_s39(model, X, P0, ss.betas, ss.K, taps=("layer3",))
        for cond, sel in plan:
            if len(sel) == 0:
                continue
            Pc = perturb(model, P0[sel], ss.betas, cond, ss.seq, sel)
            out, taps, gap = run_s39(model, X[sel], Pc, ss.betas, ss.K)
            rng = np.random.default_rng([RNG_SEED, seq_hash(ss.seq), PA2_CONDS.index(cond), 3])
            for i0 in range(0, len(sel), 64):
                sl = slice(i0, i0 + 64)
                srf = fz.surface(torch.from_numpy(Pc[sl]), ss.betas, ss.K)
                fa = fz.arms({t: v[sl] for t, v in taps.items()}, srf, gap[sl], ("tile", "lift_lbs"), rng)
                feats["tile"].append(fa["tile"].numpy().astype(np.float16))
                feats["lift_lbs"].append(fa["lift_lbs"].numpy().astype(np.float16))
            feats["gap"].append(gap.numpy().astype(np.float16))
            n = len(sel)
            rows["seq"] += [ss.seq] * n
            rows["subj"] += [subj] * n
            rows["dom"] += ["local" if ss.seq.endswith("_local") else "global"] * n
            rows["end"].append(ss.ends[sel])
            rows["cond"] += [cond] * n
            rows["betas"].append(np.repeat(ss.betas[None], n, 0))
            rows["prev"].append(Pc)
            rows["prev0"].append(P0[sel])
            rows["gt"].append(ss.gt[sel])
            rows["s39"].append(out.numpy())
            rows["s39_0"].append(out0.numpy()[sel])
    for k in ("end", "betas", "prev", "prev0", "gt", "s39", "s39_0"):
        rows[k] = np.concatenate(rows[k])
    for k in ("seq", "subj", "dom", "cond"):
        rows[k] = np.array(rows[k])
    feats = {k: np.concatenate(v) for k, v in feats.items()}
    t_feat = time.time() - t0
    tr = np.flatnonzero(np.isin(rows["subj"], C.FIT7))
    te = np.flatnonzero(np.isin(rows["subj"], C.SELECT))
    Y = (rows["gt"] - rows["prev"]).astype(np.float64)
    ra = RA(model)
    gz = KernelBank.standardise(feats["gap"].astype(np.float64), tr)
    Kg = KernelBank.kernel([gz])
    res = {"seed": seed, "n_train": int(len(tr)), "n_test": int(len(te)), "t_features_s": t_feat, "arms": {}}
    preds = {}
    for arm in PA2_ARMS:
        K = Kg if arm == "gap" else Kg + KernelBank.kernel([KernelBank.standardise(feats[arm].astype(np.float32), tr)])
        lam, cv = subject_cv_lambda(K, Y, rows["prev"], rows["gt"], rows["betas"], rows["subj"], tr, ra)
        preds[arm] = ridge_kernel_predict(K, Y, tr, te, lam)
        res["arms"][arm] = {"lambda_rel": lam, "cv_ra": cv}
        del K
    if smoke:
        res["shapes"] = {k: list(v.shape) for k, v in feats.items()}
        return res, None
    out_rows = {k: rows[k][te] for k in ("seq", "dom", "end", "cond", "betas", "prev", "prev0", "gt", "s39", "s39_0")}
    for arm in PA2_ARMS:
        out_rows[f"pred_{arm}"] = preds[arm]
    return res, out_rows


def carry_stats(model, rows, out_key):
    """b_off = 1 - carry per condition: carry = <J(out_c) - J(out_0), J(prev_c) - J(prev_0)> / |J(prev_c) - J(prev_0)|^2,
    absolute joints (metres) for translation offsets, root-aligned joints for rotation offsets; per sample terms so a
    block bootstrap can resample them. out_0 is the same readout's output on the clean (cond 'clean') row of the sample."""
    ra = RA(model)
    out = {}
    key = np.array([f"{s}|{int(e)}" for s, e in zip(rows["seq"], rows["end"])])
    clean = {k: i for i, k in enumerate(key) if rows["cond"][i] == "clean"}
    for cond in PA2_CONDS[1:]:
        sel = np.flatnonzero(rows["cond"] == cond)
        sel = np.array([i for i in sel if key[i] in clean])
        c0 = np.array([clean[key[i]] for i in sel])
        o_c, o_0 = rows[out_key][sel], rows[out_key][c0]
        p_c, p_0 = rows["prev"][sel], rows["prev"][c0]
        b = rows["betas"][sel]
        absolute = cond.startswith("t") or cond == "z45"

        def J(P):
            j = ra.joints(P, b)
            if absolute:
                tr_ = P[:, 0:3]
                return j + 0.0 * tr_[:, None]          # root-aligned joints + translation added below
            return j
        jo_c, jo_0, jp_c, jp_0 = J(o_c), J(o_0), J(p_c), J(p_0)
        if absolute:
            jo_c += o_c[:, None, 0:3]
            jo_0 += o_0[:, None, 0:3]
            jp_c += p_c[:, None, 0:3]
            jp_0 += p_0[:, None, 0:3]
        din = (jp_c - jp_0).reshape(len(sel), -1)
        dout = (jo_c - jo_0).reshape(len(sel), -1)
        out[cond] = {"num": (din * dout).sum(1), "den": (din * din).sum(1), "sel": sel}
    return out


def pa2(args):
    seeds = args.seeds
    if args.smoke:
        res, _ = pa2_seed(seeds[0], True, args.threads)
        print("PA2 smoke:", json.dumps({"n_train": res["n_train"], "n_test": res["n_test"], "shapes": res["shapes"],
                                        "t_features_s": round(res["t_features_s"], 1)}))
        return
    C.verify_protocol()
    with mp.get_context("spawn").Pool(len(seeds)) as pool:
        outs = pool.starmap(pa2_seed, [(s, False, args.threads) for s in seeds])
    model, _ = C.load_s39(seeds[0], torch.device("cpu"))
    result = {"probe": "pa2", "seeds": {}, "b_off": {}}
    agg = {}
    for (res, rows), s in zip(outs, seeds):
        result["seeds"][str(s)] = res
        np.savez_compressed(Path(args.out) / f"rows_s{s}.npz", **rows)
        for arm in ("s39",) + tuple(f"pred_{a}" for a in PA2_ARMS):
            if arm.startswith("pred_"):
                rows[f"out_{arm}"] = rows["prev"] + rows[arm]
                key = f"out_{arm}"
            else:
                key = "s39"
            cs = carry_stats(model, rows, key)
            for cond, v in cs.items():
                agg.setdefault((arm, cond), []).append((v, rows))
    for (arm, cond), lst in agg.items():
        name = arm.replace("pred_", "")
        for dom in ("all", "local", "global"):
            per_seed, terms = [], []
            for v, rows in lst:
                sel = v["sel"]
                m = np.ones(len(sel), bool) if dom == "all" else (rows["dom"][sel] == dom)
                per_seed.append(1 - v["num"][m].sum() / v["den"][m].sum())
                terms.append((v["num"][m], v["den"][m], groups_of({"seq": rows["seq"][sel][m], "end": rows["end"][sel][m]})))
            result["b_off"][f"{name}|{cond}|{dom}"] = {"per_seed": per_seed, "mean": float(np.mean(per_seed))}
    result["contrast_t14"] = pa2_contrast(agg)
    C.save_json(Path(args.out) / "result.json", result)
    C.receipt(args.out, {"probe": "pa2", "seeds": seeds}, scripts=[__file__, C.__file__])


def pa2_contrast(agg):
    """b_off(lift_lbs) - max(b_off(gap), b_off(tile)) at t14 with a (sequence, 1 s) block bootstrap over samples,
    pooling seeds (each seed's per-sample terms resampled with the same block draw)."""
    out = {}
    for dom in ("all", "local", "global"):
        comps = {}
        for arm in ("pred_gap", "pred_tile", "pred_lift_lbs", "s39"):
            nums, dens, grps = [], [], []
            for v, rows in agg[(arm, "t14")]:
                sel = v["sel"]
                m = np.ones(len(sel), bool) if dom == "all" else (rows["dom"][sel] == dom)
                nums.append(v["num"][m])
                dens.append(v["den"][m])
                grps.append(groups_of({"seq": rows["seq"][sel][m], "end": rows["end"][sel][m]}))
            comps[arm] = (nums, dens, grps)
        g = comps["pred_gap"][2][0]
        if len(g) == 0:
            out[dom] = None
            continue
        ug = np.unique(g)
        idx = {u: np.flatnonzero(g == u) for u in ug}
        rng = np.random.default_rng(RNG_SEED + 7)

        def boff(arm, rows_idx):
            nums, dens, _ = comps[arm]
            vals = [1 - n[rows_idx].sum() / max(d[rows_idx].sum(), 1e-12) for n, d in zip(nums, dens)]
            return float(np.mean(vals)), vals
        full = np.arange(len(g))
        b = {a: boff(a, full) for a in comps}
        diff = b["pred_lift_lbs"][0] - max(b["pred_gap"][0], b["pred_tile"][0])
        bs = []
        for _ in range(N_BOOT):
            r = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
            bs.append(boff("pred_lift_lbs", r)[0] - max(boff("pred_gap", r)[0], boff("pred_tile", r)[0]))
        out[dom] = {"b_off": {a: b[a][0] for a in b}, "b_off_per_seed": {a: b[a][1] for a in b}, "diff": diff,
                    "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    "ceiling_ref_max_gap_tile": max(b["pred_gap"][0], b["pred_tile"][0]),
                    "s39_one_minus_carry": b["s39"][0]}
    return out


# =============================================================================================== PA3 (descriptive)
SUBTREES = {"index": (1, 2, 3), "middle": (4, 5, 6), "pinky": (7, 8, 9), "ring": (10, 11, 12), "thumb": (13, 14, 15)}


def pa3_seed(seed: int, smoke: bool, threads: int):
    torch.set_num_threads(threads)
    model, _ = C.load_s39(seed, torch.device("cpu"))
    seqs = C.base_seqs(C.SELECT) if not smoke else ["ycy_local"]
    samples = gather(seqs, EVERY, max_n=16 if smoke else None)
    mt = C.mesh_tables(model)
    part = mt["lbs"].argmax(1).numpy()
    ra = RA(model)
    recs = []
    for ss in samples:
        idx = np.arange(len(ss))
        X = ss.lnes(idx)
        P = ss.prev
        out, _, _ = run_s39(model, X, P, ss.betas, ss.K, taps=("layer3",))
        Pm = np.stack([np.asarray(ss.pos51[int(e) - 2 * C.STEP], np.float32) for e in ss.ends])   # causal: GT(end-100)
        srf = C.fk_surface(model, torch.from_numpy(P), torch.tensor(ss.betas), torch.tensor(ss.K))
        srf_m = C.fk_surface(model, torch.from_numpy(Pm), torch.tensor(ss.betas), torch.tensor(ss.K))
        vel = (srf["uv"] - srf_m["uv"]).numpy()                                      # px per 50 ms (causal)
        nrm = srf["normals"].numpy()[..., :2]
        nrm = nrm / (np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9)
        vis = srf["vis"].numpy()
        illum = np.abs((vel * nrm).sum(-1)) * vis                                   # (n, 778)
        uv = srf["uv"].numpy()
        jo, jp, jg = ra.joints(out.numpy(), np.repeat(ss.betas[None], len(P), 0)), ra.joints(P, np.repeat(
            ss.betas[None], len(P), 0)), ra.joints(ss.gt, np.repeat(ss.betas[None], len(P), 0))
        for k in range(len(P)):
            end = int(ss.ends[k])
            w = int(min(C.WINDOW, end + 1 - int(ss.run_a[k])))
            a0, a1 = int(ss._offsets[end - w + 1]), int(ss._offsets[end + 1])
            ev = np.asarray(ss._events[a0:a1])[:, :2].astype(np.float32)
            for name, js in SUBTREES.items():
                vm = np.isin(part, js)
                pts = uv[k, vm & vis[k]]
                cnt = 0
                if len(pts) and len(ev):
                    d2 = ((ev[:, None, :] - pts[None, :, :]) ** 2).sum(-1).min(1)
                    cnt = int((d2 <= 4.0).sum())
                jj = [j for j in range(21)]
                helps = float(np.linalg.norm(jo[k] - jg[k], axis=-1).mean() < np.linalg.norm(jp[k] - jg[k], axis=-1).mean())
                recs.append({"seq": ss.seq, "end": end, "subtree": name, "illum": float(illum[k, vm].sum()),
                             "count": cnt, "helps": helps})
    return recs


def auc(score, label):
    from scipy.stats import rankdata
    score, label = np.asarray(score, float), np.asarray(label, bool)
    if label.all() or (~label).all():
        return float("nan")
    r = rankdata(score)
    n1 = label.sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * (~label).sum()))


def pa3(args):
    if args.smoke:
        recs = pa3_seed(args.seeds[0], True, args.threads)
        print("PA3 smoke: records", len(recs), "fields", sorted(recs[0]))
        return
    C.verify_protocol()
    with mp.get_context("spawn").Pool(len(args.seeds)) as pool:
        outs = pool.starmap(pa3_seed, [(s, False, args.threads) for s in args.seeds])
    res = {"probe": "pa3", "descriptive": True, "per_seed": {}}
    for s, recs in zip(args.seeds, outs):
        d = {}
        for name in list(SUBTREES) + ["all"]:
            rr = [r for r in recs if name == "all" or r["subtree"] == name]
            d[name] = {"auc_illum": auc([r["illum"] for r in rr], [r["helps"] for r in rr]),
                       "auc_count": auc([r["count"] for r in rr], [r["helps"] for r in rr]), "n": len(rr)}
        res["per_seed"][str(s)] = d
    C.save_json(Path(args.out) / "result.json", res)
    C.receipt(args.out, {"probe": "pa3"}, scripts=[__file__, C.__file__])


# =============================================================================================== M2 PA-RGQ
def small_hand_betas(betas: np.ndarray, sigma: float) -> np.ndarray:
    """betas whose first shape coefficient changes hand length by ~(sigma - 1) (Stage 0: +1 SD ~ -4.7 % length)."""
    b = betas.copy()
    b[0] += (1.0 - sigma) / 0.047
    return b


def rgq_seed(seed: int, smoke: bool, threads: int):
    torch.set_num_threads(threads)
    model, _ = C.load_s39(seed, torch.device("cpu"))
    seqs = C.base_seqs(C.FIT7 + C.SELECT) if not smoke else ["ycy_local"]
    samples = gather(seqs, EVERY, max_n=16 if smoke else None)
    ra = RA(model)
    recs = {"seq": [], "subj": [], "dom": [], "end": []}
    vals = {}
    for ss in samples:
        idx = np.arange(len(ss))
        X = ss.lnes(idx)
        B = np.repeat(ss.betas[None], len(idx), 0)
        out1, _, _ = run_s39(model, X, ss.prev, ss.betas, ss.K, taps=("layer3",))
        j1 = ra.joints(out1.numpy(), B)
        w = wrist_xyz(model, ss.prev, ss.betas)
        ray = w / np.linalg.norm(w, axis=1, keepdims=True)
        for sg in SIGMAS:
            if sg == 1.0:
                continue
            Ps = scale_about_wrist(model, ss.prev, ss.betas, sg)
            o, _, _ = run_s39(model, X, Ps, ss.betas, ss.K, taps=("layer3",))
            o = o.numpy()
            art = np.linalg.norm(ra.joints(o, B) - j1, axis=-1).mean(-1) * 1000.0           # mm, root-aligned
            need = ((ss.prev[:, 0:3] - Ps[:, 0:3]) * ray).sum(1)                             # along-ray correction needed
            got = ((o[:, 0:3] - Ps[:, 0:3]) * ray).sum(1) - ((out1.numpy()[:, 0:3] - ss.prev[:, 0:3]) * ray).sum(1)
            vals.setdefault(f"wrist|{sg}|art_mm", []).append(art)
            vals.setdefault(f"wrist|{sg}|tz_need_m", []).append(need)
            vals.setdefault(f"wrist|{sg}|tz_got_m", []).append(got)
            # "small hand" shape: the prev rendered with a hand of length x sigma at the same pose
            bs = small_hand_betas(ss.betas, sg)
            o2, _, _ = run_s39(model, X, ss.prev, bs, ss.K, taps=("layer3",))
            art2 = np.linalg.norm(ra.joints(o2.numpy(), B) - j1, axis=-1).mean(-1) * 1000.0
            vals.setdefault(f"hand|{sg}|art_mm", []).append(art2)
        n = len(idx)
        recs["seq"] += [ss.seq] * n
        recs["subj"] += [C.subject_of(ss.seq)] * n
        recs["dom"] += ["local" if ss.seq.endswith("_local") else "global"] * n
        recs["end"].append(ss.ends)
    recs = {k: (np.concatenate(v) if k == "end" else np.array(v)) for k, v in recs.items()}
    recs.update({k: np.concatenate(v) for k, v in vals.items()})
    return recs


def rgq(args):
    if args.smoke:
        recs = rgq_seed(args.seeds[0], True, args.threads)
        print("RGQ smoke: n", len(recs["seq"]), "keys", sorted(k for k in recs if "|" in k)[:6], "...")
        return
    C.verify_protocol()
    with mp.get_context("spawn").Pool(len(args.seeds)) as pool:
        outs = pool.starmap(rgq_seed, [(s, False, args.threads) for s in args.seeds])
    res = {"probe": "rgq", "per_seed": {}, "summary": {}}
    for s, recs in zip(args.seeds, outs):
        np.savez_compressed(Path(args.out) / f"rows_s{s}.npz", **recs)
        d = {}
        for grp in ("SELECT", "FIT7"):
            m0 = np.isin(recs["subj"], C.SELECT if grp == "SELECT" else C.FIT7)
            for dom in ("all", "local", "global"):
                m = m0 & (np.ones_like(m0) if dom == "all" else (recs["dom"] == dom))
                g = groups_of({"seq": recs["seq"][m], "end": recs["end"][m]})
                for k in recs:
                    if "|" not in k or k.endswith("_need_m"):
                        continue
                    if k.endswith("tz_got_m"):
                        need = recs[k.replace("got", "need")][m]
                        d[f"{grp}|{dom}|{k.replace('tz_got_m', 'tz_ratio')}"] = float(
                            (recs[k][m] * need).sum() / max((need * need).sum(), 1e-12))
                    else:
                        d[f"{grp}|{dom}|{k}"] = boot_mean_ci(recs[k][m], g)
        res["per_seed"][str(s)] = d
    C.save_json(Path(args.out) / "result.json", res)
    C.receipt(args.out, {"probe": "rgq"}, scripts=[__file__, C.__file__])


# =============================================================================================== M2 PA5-MSCN
def zoom_events(s: float, cx: float, cy: float):
    def tf(x, y, p, ms):
        xs = np.rint(cx + s * (x - cx)).astype(np.int64)
        ys = np.rint(cy + s * (y - cy)).astype(np.int64)
        ok = (xs >= 0) & (xs < C.W) & (ys >= 0) & (ys < C.H)
        return xs[ok], ys[ok], p[ok], ms[ok]
    return tf


def shift_end(ss: "C.SeqSamples", idx, dt_ms: int):
    """A/A: the same sample's W200 image ending dt_ms earlier (same prev)."""
    out = []
    for i in idx:
        end = int(ss.ends[i]) - dt_ms
        w = int(min(C.WINDOW, end + 1 - int(ss.run_a[i])))
        out.append(C.ET.build_lnes(ss._events, ss._offsets, end, w, ss.channels))
    return np.stack(out).astype(np.float32)


def mscn_read(fz: Featurizer, taps, srf, mode: str, z_perm=None):
    """(B, 778, PROJ_DIM) per-vertex features: 'fixedwin' (layer2, +-6 px window), 'fixedlayer' (layer2 centre),
    'metric' (+-rho = fx 10 mm / z window, continuous scale selection over layer1-3 by log2(rho / 3 px))."""
    uv = srf["uv"]
    proj = {t: (taps[t].float().permute(0, 2, 3, 1) @ fz.proj[t]).permute(0, 3, 1, 2) for t in C.TAPS}
    if mode == "fixedlayer":
        return C.sample_map(proj["layer2"], uv, 8)
    if mode == "fixedwin":
        pts = [uv] + [uv + torch.tensor([du, dv], dtype=uv.dtype) * 6.0 for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        return torch.stack([C.sample_map(proj["layer2"], p, 8) for p in pts]).mean(0)
    z = srf["z"] if z_perm is None else srf["z"][:, z_perm]
    rho = srf["intr"][0][:, None] * 0.010 / z.clamp(min=1e-3)                    # (B, 778) px
    pts = [uv] + [uv + torch.stack([rho * du, rho * dv], -1) for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    lvl = torch.log2(rho / 3.0).clamp(0.0, 2.0)                                   # 0 -> layer1, 1 -> layer2, 2 -> layer3
    feats = {t: torch.stack([C.sample_map(proj[t], p, C.TAP_STRIDE[t]) for p in pts]).mean(0) for t in C.TAPS}
    w1 = (1 - lvl).clamp(0, 1)
    w3 = (lvl - 1).clamp(0, 1)
    w2 = 1 - w1 - w3
    return feats["layer1"] * w1[..., None] + feats["layer2"] * w2[..., None] + feats["layer3"] * w3[..., None]


def mscn_seed(seed: int, smoke: bool, threads: int):
    torch.set_num_threads(threads)
    model, _ = C.load_s39(seed, torch.device("cpu"))
    fz = Featurizer(model)
    seqs = C.base_seqs(C.FIT7 + C.SELECT) if not smoke else ["ch_local"]
    samples = gather(seqs, EVERY, max_n=8 if smoke else None)
    rng = np.random.default_rng([RNG_SEED, seed, 23])
    recs = {"seq": [], "subj": [], "end": []}
    vals = {}
    for ss in samples:
        idx = np.arange(len(ss))
        P = ss.prev
        Kr = ss.K.copy()
        cx, cy = Kr[0, 2] * 0.375, Kr[1, 2] * 0.375

        def feats_for(X, Kmat, zp=None):
            _, taps, _ = run_s39(model, X, P, ss.betas, Kmat)
            out = {}
            for i0 in range(0, len(P), 64):
                sl = slice(i0, i0 + 64)
                srf = fz.surface(torch.from_numpy(P[sl]), ss.betas, Kmat)
                tp = {t: v[sl] for t, v in taps.items()}
                vis = (srf["vis"] & fz.inb(srf)).float().unsqueeze(-1)
                for mode in ("fixedwin", "fixedlayer", "metric"):
                    out.setdefault(mode, []).append(mscn_read(fz, tp, srf, mode) * vis)
                if zp is not None:
                    out.setdefault("metric_zperm", []).append(mscn_read(fz, tp, srf, "metric", z_perm=zp) * vis)
            return {k: torch.cat(v) for k, v in out.items()}
        zp = torch.from_numpy(rng.permutation(778))
        base = feats_for(ss.lnes(idx), ss.K, zp)
        aa = feats_for(shift_end(ss, idx, 10), ss.K, zp)
        den = {k: (base[k] - aa[k]).flatten(1).norm(dim=1).numpy() for k in base}
        for s in ZOOMS:
            Kz = ss.K.copy()
            Kz[0, 0] *= s
            Kz[1, 1] *= s
            Xz = ss.lnes(idx, transform=zoom_events(s, cx, cy))
            fzm = feats_for(Xz, Kz, zp)
            for k in base:
                num = (fzm[k] - base[k]).flatten(1).norm(dim=1).numpy()
                vals.setdefault(f"{s}|{k}|num", []).append(num)
                vals.setdefault(f"{s}|{k}|den", []).append(den[k])
        recs["seq"] += [ss.seq] * len(idx)
        recs["subj"] += [C.subject_of(ss.seq)] * len(idx)
        recs["end"].append(ss.ends)
    recs = {"seq": np.array(recs["seq"]), "subj": np.array(recs["subj"]), "end": np.concatenate(recs["end"])}
    recs.update({k: np.concatenate(v) for k, v in vals.items()})
    return recs


def mscn(args):
    if args.smoke:
        recs = mscn_seed(args.seeds[0], True, args.threads)
        print("MSCN smoke: n", len(recs["seq"]), "keys", len([k for k in recs if "|" in k]))
        return
    C.verify_protocol()
    with mp.get_context("spawn").Pool(len(args.seeds)) as pool:
        outs = pool.starmap(mscn_seed, [(s, False, args.threads) for s in args.seeds])
    res = {"probe": "mscn", "per_seed": {}}
    for s, recs in zip(args.seeds, outs):
        np.savez_compressed(Path(args.out) / f"rows_s{s}.npz", **recs)
        d = {}
        for grp in ("FIT7", "SELECT"):
            m = np.isin(recs["subj"], C.FIT7 if grp == "FIT7" else C.SELECT)
            g = groups_of({"seq": recs["seq"][m], "end": recs["end"][m]})
            ug = np.unique(g)
            idx = {u: np.flatnonzero(g == u) for u in ug}
            rng = np.random.default_rng(RNG_SEED + 31)
            for z in ZOOMS:
                D = {k: (recs[f"{z}|{k}|num"][m], recs[f"{z}|{k}|den"][m])
                     for k in ("fixedwin", "fixedlayer", "metric", "metric_zperm")}

                def ratios(rows):
                    dd = {k: D[k][0][rows].sum() / max(D[k][1][rows].sum(), 1e-12) for k in D}
                    return dd, dd["metric"] / max(dd["fixedwin"], 1e-12), dd["metric_zperm"] / max(dd["fixedwin"], 1e-12)
                full = np.arange(m.sum())
                dd, R, Rz = ratios(full)
                bs = [ratios(np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])) for _ in range(N_BOOT)]
                d[f"{grp}|{z}"] = {"D": dd, "R": R, "R_ci": [float(np.percentile([b[1] for b in bs], 2.5)),
                                                             float(np.percentile([b[1] for b in bs], 97.5))],
                                   "R_zperm": Rz,
                                   "R_zperm_ci": [float(np.percentile([b[2] for b in bs], 2.5)),
                                                  float(np.percentile([b[2] for b in bs], 97.5))]}
        res["per_seed"][str(s)] = d
    C.save_json(Path(args.out) / "result.json", res)
    C.receipt(args.out, {"probe": "mscn"}, scripts=[__file__, C.__file__])


# =============================================================================================== decide
MDE_LOCAL, SD_LOCAL = 1.149, 0.379           # KS0 (#kstar-results-20261025), mm


def decide_m1(pa1_res: dict, pa2_res: dict) -> dict:
    """PA1 primary: lift_lbs - tile (CV-optimal and df-matched), per domain, Holm over (local, global) on the bootstrap
    one-sided p; pass in a domain = Holm-significant (mean < 0), CI upper < 0, the three seeds' means < 0, and the
    df-matched contrast mean < 0 too; and no domain significantly worse (CI lower > 0). Kill = no domain passes.
    PA2: diff = b_off(lift_lbs) - max(b_off(gap), b_off(tile)) at t14 (all SELECT); pass = diff >= 0.10 and CI lower > 0;
    'ceiling' (uninformative, does not gate) when max(b_off(gap), b_off(tile)) >= 0.90; kill otherwise.
    GPU release of M1 = PA1 pass and PA2 in {pass, ceiling}."""
    c = pa1_res["contrast"]
    doms = ("local", "global")
    p = {d: c[d]["lift_lbs-tile"]["p_ge0"] for d in doms}
    order = sorted(doms, key=lambda d: p[d])
    holm = {order[0]: p[order[0]] <= 0.025, order[1]: p[order[0]] <= 0.025 and p[order[1]] <= 0.05}
    dom_pass = {}
    for d in doms:
        e = c[d]["lift_lbs-tile"]
        dm = c[d]["lift_lbs_dfmatched-tile"]
        dom_pass[d] = bool(holm[d] and e["mean"] < 0 and e["ci"][1] < 0 and all(x < 0 for x in e["per_seed_mean"])
                           and dm["mean"] < 0)
    worse = any(c[d]["lift_lbs-tile"]["ci"][0] > 0 for d in doms)
    pa1_v = "pass" if any(dom_pass.values()) and not worse else "kill"
    t = pa2_res["contrast_t14"]["all"]
    ceil = t["ceiling_ref_max_gap_tile"]
    if ceil >= 0.90:
        pa2_v = "ceiling"
    elif t["diff"] >= 0.10 and t["ci"][0] > 0:
        pa2_v = "pass"
    else:
        pa2_v = "kill"
    tool_ok = all(c["all"][f"{k}-gap"]["mean"] >= -0.05 for k in ("ctl_tshuffle", "ctl_noise"))
    specific = c["all"]["lift_lbs-ctl_vshuffle"]["mean"] < 0 and c["all"]["lift_lbs-ctl_shift10"]["mean"] < 0
    if not tool_ok:
        pa1_v = "tool_defect"
    controls_ok = tool_ok and specific
    return {"pa1": pa1_v, "pa1_tool_ok": tool_ok, "pa1_mesh_specific": specific, "pa1_domains": dom_pass, "pa1_holm_p": p, "pa1_worse_domain": worse,
            "pa2": pa2_v, "pa2_diff": t["diff"], "pa2_ci": t["ci"],
            "pa2_ceiling_ref": ceil,
            "gpu_release_m1": bool(pa1_v == "pass" and pa2_v in ("pass", "ceiling") and controls_ok),
            "power_note": f"achievable gain must reach 1.5 x MDE_local = {1.5 * MDE_LOCAL:.2f} mm for a 3-seed "
                          "confirmation; PA1 passing only releases the exploratory single-seed debug"}


def decide_m2(rgq_res: dict, mscn_res: dict) -> dict:
    """RGQ: articulation response at sigma 0.94 (wrist-ray scaling), SELECT all: pass if all 3 seeds >= 1.149 mm;
    weak if all >= 0.379; kill otherwise. MSCN: FIT7 zoom 0.94 and 1.06: R = D(metric)/D(fixedwin) CI upper <= 0.8 in
    all seeds, D(fixedwin) >= 1.5 (sensitivity), z-perm control (the metric gain
    is not smoothing: the z-permuted metric reading's R CI contains 1); kill if any R >= 0.95; untestable if insensitive."""
    vals = [v["SELECT|all|wrist|0.94|art_mm"]["mean"] for v in rgq_res["per_seed"].values()]
    rgq_v = "pass" if all(x >= MDE_LOCAL for x in vals) else ("weak" if all(x >= SD_LOCAL for x in vals) else "kill")
    ok, sens, zctl, kill = True, True, True, False
    for v in mscn_res["per_seed"].values():
        for z in (0.94, 1.06):
            e = v[f"FIT7|{z}"]
            ok &= e["R_ci"][1] <= 0.80
            kill |= e["R"] >= 0.95
            sens &= e["D"]["fixedwin"] >= 1.5
            zctl &= e["R_zperm_ci"][0] <= 1.0 <= e["R_zperm_ci"][1]
    if not sens:
        mscn_v = "untestable"
    elif kill:
        mscn_v = "kill"
    elif ok and zctl:
        mscn_v = "pass"
    else:
        mscn_v = "descriptive"
    return {"rgq": rgq_v, "rgq_art_mm_sigma094_per_seed": vals, "mscn": mscn_v, "mscn_sensitive": sens,
            "mscn_zperm_ok": zctl}


# =============================================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=("pa1", "pa2", "pa3", "rgq", "mscn"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.SEEDS))
    ap.add_argument("--threads", type=int, default=40)
    a = ap.parse_args()
    if not a.smoke:
        if not a.out:
            raise SystemExit("--out required")
        Path(a.out).mkdir(parents=True, exist_ok=True)
        if (Path(a.out) / "result.json").exists():
            raise SystemExit(f"{a.out}/result.json exists; preserve it")
    {"pa1": pa1, "pa2": pa2, "pa3": pa3, "rgq": rgq, "mscn": mscn}[a.probe](a)


if __name__ == "__main__":
    main()
