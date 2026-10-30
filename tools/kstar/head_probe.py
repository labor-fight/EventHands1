#!/usr/bin/env python3
"""K-star KS2 (PB'), KS4 (PA-fit) and the descriptive capture curve: what the frozen XVM vertex-anchored patch head measures
(docs/EXPERIMENT_RECORDS.md#kstar-prereg-20261025).

Head: `outputs/xvm/s2/p4/A_m600/heads.pt`, `patch_r2` (trained on the seven FIT subjects at GT anchors, W_mem 600 ms,
+-25 ms box-smoothed GT targets; never retrained here). Data: the XVM P4 sample definition (`tools/xvm/p4_data.py`),
every `--stride`-th endpoint; gates read the SELECT subjects (ycy, ylf), local and global separately; zgz is scored once
with `--zgz` after the gates are decided. CPU only.

stale   (S2) time-surface variants of the same samples:
          R0 as trained ([end-599, end]); R1 stale (events older than 50 ms only); R2 recent (last 50 ms only);
          R3 timestamps permuted within the window; P1 per-pixel polarity swap (random half); P2 global polarity flip.
        Targets (normal component e1 of the vertex displacement, px): `full` (XVM), `art` (fingers only: root and
        translation held at the anchor), and the persistence predictor (the previous 50 ms displacement of the same
        vertex). Reads Pearson r_n, the partial correlation with persistence removed, the OLS calibration slope of the
        target on the prediction, and the lag-1 correlation of the residual between consecutive endpoints.
capture (S3) anchor perturbations (image-plane translation 5-25 mm in 8 directions, depth +45 mm, root 5 / 11 deg):
        the head reads at the perturbed anchor; scored against the differential target (the vertex's true motion,
        perturbed frame) and the innovation target (true end position minus the perturbed anchor), with the slope
        b_off of the prediction on the innovation offset.

    python tools/kstar/head_probe.py stale   --out outputs/kstar/s2_stale
    python tools/kstar/head_probe.py capture --out outputs/kstar/s4_capture
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kstar import common as KC  # noqa: E402
from xvm import common as XC  # noqa: E402
from xvm.common import EF  # noqa: E402
from xvm.p4_data import P4Data, collate  # noqa: E402
from xvm import p4_train as T  # noqa: E402
from xvm.p5_solve import load_head  # noqa: E402
from semkine import xvm as XV  # noqa: E402

HEAD = "patch_r2"
STALE_VARIANTS = ("R0", "R1", "R2", "R3", "P1", "P2")
TRANSL_MM = (5, 10, 14, 19, 25)
N_DIR = 8
DEPTH_MM = (45,)
ROOT_DEG = (5, 11)
VARIANT_SEED = 4242
POS_CORR_SEED = 4343
SUBJECT_KEYS = ("ycy", "ylf", "zgz", "all")


# ------------------------------------------------------------------------------------------------ data
class StaleData(P4Data):
    """P4 samples (W_mem 600) with the time-surface variants; adds the anchor of the previous endpoint for the
    persistence predictor and returns consecutive endpoint pairs (end - 50, end) for the residual lag-1."""

    def __init__(self, seqs, stride, variants):
        super().__init__(seqs, 600, stride=1, with_sae=False)
        self.variants = variants
        keep = []
        for k, (si, rid, end, prev) in enumerate(self.index):
            sq = self.seqs[si]
            a = int(sq.runs[rid][0])
            if end - 150 >= a:                     # lag-2 persistence anchors inside the run
                keep.append((si, rid, end, prev))
        # pairs of consecutive endpoints: sample every `stride`-th eligible endpoint and its predecessor
        idx = []
        for k in range(1, len(keep), stride):
            p, c = keep[k - 1], keep[k]
            if p[0] == c[0] and p[1] == c[1] and c[2] - p[2] == 50:
                idx += [p, c]
        self.index = idx

    def __getitem__(self, i):
        si, rid, end, prev = self.index[i]
        out = super().__getitem__(i)
        sq = self.seqs[si]
        a, b = (int(v) for v in sq.runs[rid])
        out["xm"] = XC.smoothed_gt(sq.pos51, end - 100, a, b - 1).astype(np.float64)   # anchor one step earlier
        out["xm2"] = XC.smoothed_gt(sq.pos51, end - 150, a, b - 1).astype(np.float64)  # two steps earlier
        lo = max(a, end - 599)
        a0, a1 = int(sq.offsets[lo]), int(sq.offsets[end + 1])
        ev = np.asarray(sq.events[a0:a1])
        t_us = EF.event_times_us(sq, lo, end)
        age = ((end + 1) * 1000.0 - t_us.astype(np.float64)) / 1000.0
        rng = np.random.default_rng(VARIANT_SEED + 7919 * i)
        for v in self.variants:
            x, y, p, ag = ev[:, 0].astype(np.int64), ev[:, 1].astype(np.int64), np.clip(ev[:, 2], 0, 1).astype(np.int64), age.copy()
            if v == "R1":
                m = ag >= 50.0
                x, y, p, ag = x[m], y[m], p[m], ag[m]
            elif v == "R2":
                m = ag < 50.0
                x, y, p, ag = x[m], y[m], p[m], ag[m]
            elif v == "R3":
                ag = ag[rng.permutation(len(ag))]
            elif v == "P1":
                swap = rng.random(EF.H * EF.W) < 0.5
                p = np.where(swap[y * EF.W + x], 1 - p, p)
            elif v == "P2":
                p = 1 - p
            out[f"sae_{v}"] = _sae(x, y, p, ag)
        return out


def _sae(x, y, p, age_ms):
    """Per-(polarity, pixel) age of the latest event (ms; 1e9 = none)."""
    H, W = EF.H, EF.W
    sae = np.full(2 * H * W, 1e9, np.float32)
    if len(x):
        key = p * (H * W) + y * W + x
        np.minimum.at(sae, key, age_ms.astype(np.float32))
    return sae.reshape(2, H, W)


# ------------------------------------------------------------------------------------------------ geometry
def project(V, g):
    z = V[..., 2].clamp(min=1e-6)
    return torch.stack([g["fx"][:, None] * V[..., 0] / z + g["cx"][:, None],
                        g["fy"][:, None] * V[..., 1] / z + g["cy"][:, None]], -1)


def verts(geo, X, betas):
    out = []
    for b in range(X.shape[0]):
        v, _ = geo.mano.fk(X[b:b + 1], betas[b:b + 1])
        out.append(v[0])
    return torch.stack(out)


def read_head(head, g, sae):
    m, L, rho, _ = head(g, {"sae_age": sae})
    return m[..., 0].float() * rho                                              # predicted normal displacement, px


# ------------------------------------------------------------------------------------------------ accumulators
class Lin:
    """Moment sums for Pearson r, partial r (one covariate) and the OLS slope of y on x, kept per bootstrap group
    (sequence, run, 1 s of endpoints) so that every statistic, and the ratio of two of them, gets a group-bootstrap CI
    from one shared set of resamples."""

    def __init__(self):
        self.g = {}

    def add(self, key, x, y, z=None, group=0):
        x, y = x.double().flatten(), y.double().flatten()
        z = torch.zeros_like(x) if z is None else z.double().flatten()
        v = torch.stack([x, y, z])
        m = np.concatenate([[len(x)], v.sum(1).numpy(), (v @ v.T).numpy().reshape(-1)])
        d = self.g.setdefault(key, {})
        d[group] = d.get(group, 0) + m

    @staticmethod
    def stats(m):
        n, s, S = m[..., 0], m[..., 1:4], m[..., 4:13].reshape(*m.shape[:-1], 3, 3)
        mu = s / n[..., None]
        C = S / n[..., None, None] - mu[..., :, None] * mu[..., None, :]
        sd = np.sqrt(np.clip(np.diagonal(C, axis1=-2, axis2=-1), 1e-30, None))
        R = C / (sd[..., :, None] * sd[..., None, :])
        r_xy, r_xz, r_yz = R[..., 0, 1], R[..., 0, 2], R[..., 1, 2]
        den = np.sqrt(np.clip((1 - r_xz ** 2) * (1 - r_yz ** 2), 1e-30, None))
        return {"r": r_xy, "partial_r": (r_xy - r_xz * r_yz) / den,
                "slope_y_on_x": C[..., 0, 1] / np.clip(C[..., 0, 0], 1e-30, None), "r_target_persistence": r_yz}

    def result(self, ratios=(), n_boot=1000, seed=KC.BOOT_SEED):
        groups = sorted({g for d in self.g.values() for g in d})
        gi = {g: i for i, g in enumerate(groups)}
        M = {}
        for k, d in self.g.items():
            a = np.zeros((len(groups), 13))
            for g, m in d.items():
                a[gi[g]] = m
            M[k] = a
        rng = np.random.default_rng(seed)
        W = np.stack([np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
                      for _ in range(n_boot)]).astype(np.float64)                # (n_boot, G)
        out, boot = {}, {}
        for k, a in M.items():
            tot = a.sum(0)
            if tot[0] < 3:
                continue
            st = {kk: float(v) for kk, v in self.stats(tot).items()}
            bs = self.stats(W @ a)
            boot[k] = bs
            out[k] = {"n": int(tot[0]), "n_groups": int((a[:, 0] > 0).sum()), **st,
                      **{f"{kk}_ci": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
                         for kk, v in bs.items() if kk in ("r", "partial_r", "slope_y_on_x")}}
        for name, num, den, stat in ratios:
            if num in out and den in out:
                rb = boot[num][stat] / boot[den][stat]
                out[name] = {"value": out[num][stat] / out[den][stat],
                             "ci": [float(np.nanpercentile(rb, 2.5)), float(np.nanpercentile(rb, 97.5))]}
        return out


def gid(batch, b):
    """Bootstrap group of a sample: its sequence and its 1 s block of endpoints."""
    return int(batch["seq"][b]) * 1_000_000 + int(batch["end"][b]) // 1000


# ------------------------------------------------------------------------------------------------ stale
MAX_HOP = 4


def hop_pairs(faces):
    """{h: (i, j)} vertex pairs at mesh-hop distance h = 1..MAX_HOP (i < j)."""
    from semkine.mrf import hop_matrix
    D = hop_matrix(faces.cpu(), 778, MAX_HOP)
    out = {}
    for h in range(1, MAX_HOP + 1):
        i, j = torch.nonzero(torch.triu(D == h, diagonal=1), as_tuple=True)
        out[h] = (i, j)
    return out


def who(loader, batch, b):
    """(subject, domain) of a sample."""
    name = loader.dataset.seqs[int(batch["seq"][b])].name
    return name.split("_")[0], ("local" if "local" in name else "global")


POS_CORR = 0.9          # positive control: the art-only target plus noise at this correlation
CONTROLS = ("PERS", "POS")


def run_stale(head, geo, loader, dev):
    """Gate covariate = lag-2 persistence: the displacement over [end-150, end-100], whose +-25 ms label boxes do not
    overlap the target's [end-75, end+25] (the lag-1 displacement shares the anchor box and its label noise)."""
    acc = Lin()
    rng = np.random.default_rng(0)
    nrng = np.random.default_rng(POS_CORR_SEED)
    pairs = hop_pairs(geo.faces)
    for batch in loader:
        g = geo(batch, "A", rng)
        betas = batch["betas"].to(dev)
        # art-only target: fingers of the target, root and translation of the anchor
        xa = g["x0"].clone()
        xa[:, 6:] = batch["xt"].to(dev)[:, 6:]
        uva = project(verts(geo, xa, betas), g)
        uvm = project(verts(geo, batch["xm"].to(dev), betas), g)
        uvm2 = project(verts(geo, batch["xm2"].to(dev), betas), g)
        e1 = g["R"].float()[..., :, 0]
        tg = {"full": (e1 * g["t"].float()).sum(-1), "art": (e1 * (uva - g["uv0"]).float()).sum(-1)}
        pers1 = (e1 * (g["uv0"] - uvm).float()).sum(-1)                      # lag-1 displacement (no events)
        pers2 = (e1 * (uvm - uvm2).float()).sum(-1)                          # lag-2 displacement (gate covariate)
        obs = g["vis"] & g["obs2"]
        con = g["cos"] < XV.CONTOUR_COS
        mk = obs & con
        preds = {v: read_head(head, g, batch[f"sae_{v}"].to(dev)) for v in STALE_VARIANTS}
        preds["PERS"] = pers1                                                 # placebo: event-free persistence
        sd = float(tg["art"][mk].std()) if mk.any() else 1.0
        noise = torch.as_tensor(nrng.standard_normal(tg["art"].shape), dtype=tg["art"].dtype)
        preds["POS"] = tg["art"] + noise * sd * math.sqrt(1.0 / POS_CORR ** 2 - 1.0)   # positive control
        for v, p in preds.items():
            for b in range(p.shape[0]):
                subj, dom = who(loader, batch, b)
                m = mk[b]
                if not m.any():
                    continue
                for s_ in (subj, "all"):
                    for t, y in tg.items():
                        acc.add(f"{s_}|{dom}|{v}|{t}", p[b][m], y[b][m], pers2[b][m], group=gid(batch, b))
        # spatial structure of the R0 residual (full target): correlation of residual pairs at mesh-hop h, both
        # vertices in the observed contour set of the same sample
        p0 = preds["R0"]
        for b in range(p0.shape[0]):
            subj, dom = who(loader, batch, b)
            res_b = p0[b] - tg["full"][b]
            m = mk[b]
            for h, (i, j) in pairs.items():
                ok = m[i] & m[j]
                if ok.any():
                    acc.add(f"all|{dom}|hop{h}|residual", res_b[i[ok]], res_b[j[ok]], group=gid(batch, b))
        # residual lag-1 on consecutive pairs (R0, full target): batch rows come in (prev, cur) order
        for b in range(0, p0.shape[0] - 1, 2):
            if int(batch["end"][b + 1]) - int(batch["end"][b]) != 50 or int(batch["seq"][b]) != int(batch["seq"][b + 1]):
                continue
            m = mk[b] & mk[b + 1]
            if not m.any():
                continue
            subj, dom = who(loader, batch, b)
            acc.add(f"all|{dom}|lag1|residual", (p0[b] - tg["full"][b])[m], (p0[b + 1] - tg["full"][b + 1])[m],
                    group=gid(batch, b))
    ratios = [(f"{s_}|{d}|rho_{v}|{t}|{st}", f"{s_}|{d}|{v}|{t}", f"{s_}|{d}|R0|{t}", st)
              for s_ in SUBJECT_KEYS for d in ("local", "global") for v in ("R1", "R3") for t in ("full", "art")
              for st in ("r", "partial_r")]
    return acc.result(ratios=ratios)


# ------------------------------------------------------------------------------------------------ capture
def perturbations():
    out = [("clean", None)]
    for d in TRANSL_MM:
        for k in range(N_DIR):
            th = 2 * math.pi * k / N_DIR
            out.append((f"t{d}", ("t", d * 1e-3 * math.cos(th), d * 1e-3 * math.sin(th))))
    for d in DEPTH_MM:
        out.append((f"z{d}", ("z", d * 1e-3)))
    for deg in ROOT_DEG:
        for k in range(N_DIR):
            ax = np.array([math.cos(2 * math.pi * k / N_DIR), math.sin(2 * math.pi * k / N_DIR), 0.3])
            ax = ax / np.linalg.norm(ax)
            out.append((f"r{deg}", ("r", *(ax * math.radians(deg)))))
    return out


def perturb(x0, spec):
    x = x0.clone()
    if spec is None:
        return x
    kind = spec[0]
    if kind == "t":
        x[:, 0] += spec[1]
        x[:, 1] += spec[2]
    elif kind == "z":
        x[:, 2] += spec[1]
    elif kind == "r":
        from semkine.lie import so3_exp, so3_log
        R = so3_exp(x[:, 3:6]) @ so3_exp(torch.tensor(spec[1:], dtype=x.dtype).expand(x.shape[0], 3))
        x[:, 3:6] = so3_log(R)
    return x


def run_capture(head, geo, loader, dev):
    acc = Lin()
    rng = np.random.default_rng(0)
    plist = perturbations()
    for batch in loader:
        betas = batch["betas"].to(dev)
        gc = geo(batch, "A", rng)                                            # clean anchor
        uvt = gc["uv0"] + gc["t"]                                            # true end position, px
        for name, spec in plist:
            bp = dict(batch)
            bp["x0"] = perturb(batch["x0"], spec)
            gp = geo(bp, "A", rng)
            e1 = gp["R"].float()[..., :, 0]
            t_diff = (e1 * gc["t"].float()).sum(-1)                         # true motion, perturbed frame
            t_innov = (e1 * (uvt - gp["uv0"]).float()).sum(-1)              # true end minus perturbed anchor
            off = (e1 * (gc["uv0"] - gp["uv0"]).float()).sum(-1)            # innovation offset
            p = read_head(head, gp, batch["sae_age"].to(dev))
            mk = gp["vis"] & gp["obs2"] & (gp["cos"] < XV.CONTOUR_COS)
            for b in range(p.shape[0]):
                subj, dom = who(loader, batch, b)
                m = mk[b]
                if not m.any():
                    continue
                for s_ in (subj, "all"):
                    acc.add(f"{s_}|{dom}|{name}|diff", p[b][m], t_diff[b][m], group=gid(batch, b))
                    acc.add(f"{s_}|{dom}|{name}|innov", p[b][m], t_innov[b][m], group=gid(batch, b))
                    acc.add(f"{s_}|{dom}|{name}|offset_slope", off[b][m], p[b][m], group=gid(batch, b))
    return acc.result()


# ------------------------------------------------------------------------------------------------ PA-fit
PAFIT_OFFSETS_MM = (0, 5, 10, 14, 19, 25)
PAFIT_LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
PAFIT_AUG_P = 0.8
PAFIT_FIT_STRIDE = 32


def patch_features(head, g, sae):
    """The frozen patch head's pooled CNN feature (64) and |n.d| for every visible vertex: (B, V, 65), mask (B, V)."""
    import torch.nn.functional as F
    dev = g["uv0"].device
    ts = torch.cat([torch.exp(-sae / t) for t in T.TAUS], 1)
    B = sae.shape[0]
    rho = (g["f"][:, None] * head.r / g["z0"].clamp(min=1e-3)).float()
    o = torch.linspace(-1, 1, T.PATCH, device=dev)
    gy, gx = torch.meshgrid(o, o, indexing="ij")
    loc = torch.stack([gx, gy], -1).view(1, 1, T.PATCH * T.PATCH, 2)
    R = g["R"].float()
    off = (R.unsqueeze(2) @ loc.unsqueeze(-1).expand(B, 778, -1, 2, 1)).squeeze(-1) * rho[..., None, None]
    pix = g["uv0"].float().unsqueeze(2) + off
    norm = torch.stack([pix[..., 0] / (EF.W - 1) * 2 - 1, pix[..., 1] / (EF.H - 1) * 2 - 1], -1)
    out = torch.zeros(B, 778, 65)
    for b in range(B):
        ids = g["vis"][b].nonzero().flatten()
        if not len(ids):
            continue
        smp = F.grid_sample(ts[b:b + 1], norm[b:b + 1, ids].view(1, len(ids) * T.PATCH, T.PATCH, 2), align_corners=True)
        h = head.cnn(smp.view(6, len(ids), T.PATCH, T.PATCH).permute(1, 0, 2, 3))
        out[b, ids] = torch.cat([h, g["cos"][b, ids].float().unsqueeze(-1)], -1)
    return out


def random_spec(rng):
    if rng.random() > PAFIT_AUG_P:
        return None
    k = rng.integers(3)
    if k == 0:
        d, th = rng.uniform(0, 25e-3), rng.uniform(0, 2 * math.pi)
        return ("t", d * math.cos(th), d * math.sin(th))
    if k == 1:
        return ("z", rng.uniform(-45e-3, 45e-3))
    ax = rng.standard_normal(3)
    return ("r", *(ax / np.linalg.norm(ax) * math.radians(rng.uniform(0, 11))))


def pafit_rows(head, geo, batch, spec_fn, dev, rng):
    """Per observed-contour vertex of the perturbed anchor: feature, differential target, innovation target and the
    innovation offset (normal component, px), plus the subject / domain / group of its sample."""
    gc = geo(batch, "A", rng)
    uvt = gc["uv0"] + gc["t"]
    bp = dict(batch)
    specs = [spec_fn(b) for b in range(batch["x0"].shape[0])]
    x0p = batch["x0"].clone()
    for b, sp in enumerate(specs):
        x0p[b:b + 1] = perturb(batch["x0"][b:b + 1], sp)
    bp["x0"] = x0p
    gp = geo(bp, "A", rng)
    e1 = gp["R"].float()[..., :, 0]
    feats = patch_features(head, gp, batch["sae_age"].to(dev))
    t_diff = (e1 * gc["t"].float()).sum(-1)
    t_innov = (e1 * (uvt - gp["uv0"]).float()).sum(-1)
    off = (e1 * (gc["uv0"] - gp["uv0"]).float()).sum(-1)
    mk = gp["vis"] & gp["obs2"] & (gp["cos"] < XV.CONTOUR_COS)
    rows = []
    for b in range(feats.shape[0]):
        m = mk[b]
        if m.any():
            rows.append((feats[b][m].numpy(), t_diff[b][m].numpy(), t_innov[b][m].numpy(), off[b][m].numpy(), b))
    return rows


def ridge(X, y, lam):
    X1 = np.concatenate([X, np.ones((len(X), 1))], 1)
    A = X1.T @ X1 + lam * len(X) * np.diag(np.r_[np.ones(X.shape[1]), 1e-6])
    return np.linalg.solve(A, X1.T @ y)


def apply_ridge(w, X):
    return np.concatenate([X, np.ones((len(X), 1))], 1) @ w


def run_pafit(head, geo, fit_loader, sel_loader, dev):
    rng = np.random.default_rng(0)
    srng = np.random.default_rng(5151)
    data = {"clean": [], "aug": []}
    subj_of = []
    for batch in fit_loader:
        for name, fn in (("clean", lambda b: None), ("aug", lambda b: random_spec(srng))):
            for f, td, ti, of, b in pafit_rows(head, geo, batch, fn, dev, rng):
                data[name].append((f, td, ti, int(batch["seq"][b])))
    # lambda by subject-held-out folds inside FIT (4 folds of sequences), then refit on all
    weights = {}
    for name, rows in data.items():
        X = np.concatenate([r[0] for r in rows]).astype(np.float64)
        Y = {"diff": np.concatenate([r[1] for r in rows]), "innov": np.concatenate([r[2] for r in rows])}
        sid = np.concatenate([np.full(len(r[0]), fit_loader.dataset.seqs[r[3]].name.split("_")[0]) for r in rows])
        subs = sorted(set(sid))
        fold = np.array([subs.index(x) % 4 for x in sid])
        for tname, y in Y.items():
            best = None
            usable = [f for f in range(4) if (fold == f).any() and (fold != f).any()]
            for lam in PAFIT_LAMBDAS:
                if len(usable) < 2:
                    break
                err = 0.0
                for f in usable:
                    w = ridge(X[fold != f], y[fold != f], lam)
                    err += float(((apply_ridge(w, X[fold == f]) - y[fold == f]) ** 2).sum())
                if best is None or err < best[1]:
                    best = (lam, err)
            if best is None:
                best = (PAFIT_LAMBDAS[len(PAFIT_LAMBDAS) // 2], None)
            weights[(name, tname)] = (ridge(X, y, best[0]), best[0])
    acc = Lin()
    plist = [("clean", None)] + [(f"t{d}", None) for d in PAFIT_OFFSETS_MM[1:]] + \
            [("z45", ("z", 45e-3)), ("r5", None), ("r11", None)]
    for batch in sel_loader:
        for pname, fixed in plist:
            def fn(b, pname=pname, fixed=fixed):
                if pname.startswith("t"):
                    d, th = float(pname[1:]) * 1e-3, srng.uniform(0, 2 * math.pi)
                    return ("t", d * math.cos(th), d * math.sin(th))
                if pname.startswith("r"):
                    ax = srng.standard_normal(3)
                    return ("r", *(ax / np.linalg.norm(ax) * math.radians(float(pname[1:]))))
                return fixed
            for f, td, ti, of, b in pafit_rows(head, geo, batch, fn, dev, rng):
                subj, dom = who(sel_loader, batch, b)
                g_ = gid(batch, b)
                for (name, tname), (w, lam) in weights.items():
                    pr = torch.as_tensor(apply_ridge(w, f.astype(np.float64)))
                    tgt = torch.as_tensor(td if tname == "diff" else ti)
                    for s_ in (subj, "all"):
                        acc.add(f"{s_}|{dom}|{pname}|{name}_{tname}", pr, tgt, group=g_)
                        if tname == "innov":
                            acc.add(f"{s_}|{dom}|{pname}|{name}_{tname}|offset_slope", torch.as_tensor(of), pr, group=g_)
    res = acc.result()
    res["lambdas"] = {f"{k[0]}_{k[1]}": v[1] for k, v in weights.items()}
    return res


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("stale", "capture", "pafit"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--zgz", action="store_true", help="score zgz once (only after the SELECT gates are decided)")
    ap.add_argument("--smoke", action="store_true", help="engineering smoke: 8 samples, no numbers printed or saved")
    a = ap.parse_args()
    torch.set_num_threads(8)
    dev = torch.device("cpu")
    out = Path(a.out)
    if not a.smoke:
        KC.verify_protocol()
        if (out / "result.json").exists():
            raise SystemExit(f"{out}/result.json exists; preserve it")
    head, mem = load_head(KC.HEADS.parent, HEAD, dev)
    assert mem == 600
    geo = T.Geo(dev)
    stride = a.stride or {"stale": 4, "capture": 8, "pafit": 8}[a.mode]
    seqs = list(XC.zgz_seqs()) if a.zgz else XC.subject_seqs(KC.SELECT_SUBJ)
    if a.mode == "stale":
        ds = StaleData(seqs, stride, STALE_VARIANTS)
    else:
        ds = P4Data(seqs, 600, stride=stride, with_sae=True)
    if a.smoke:
        ds.index = ds.index[:8]
    kw = dict(shuffle=False, num_workers=0 if a.smoke else a.workers, collate_fn=collate)
    loader = DataLoader(ds, batch_size=8 if a.mode == "stale" else 4, **kw)
    t0 = time.time()
    with torch.no_grad():
        if a.mode == "stale":
            res = run_stale(head, geo, loader, dev)
        elif a.mode == "capture":
            res = run_capture(head, geo, loader, dev)
        else:
            fit = P4Data(XC.subject_seqs(KC.FIT_SUBJ), 600, stride=PAFIT_FIT_STRIDE, with_sae=True)
            if a.smoke:
                fit.index = fit.index[:8]
            res = run_pafit(head, geo, DataLoader(fit, batch_size=4, **kw), loader, dev)
    meta = {"mode": a.mode, "head": HEAD, "heads_sha256": KC.sha(KC.HEADS), "stride": stride, "zgz": a.zgz,
            "subjects": "zgz" if a.zgz else list(KC.SELECT_SUBJ), "n_samples": len(ds), "wall_s": time.time() - t0}
    if a.smoke:
        print("SMOKE ok", a.mode, {k: v["n"] for k, v in list(res.items())[:3]}, f"{meta['wall_s']:.1f} s")
        return
    KC.save_json(out / "result.json", {"meta": meta, "readings": res})
    KC.receipt(out, meta, scripts=(__file__, XC.REPO / "tools/xvm/p4_data.py", XC.REPO / "tools/xvm/p4_train.py"),
               inputs=(KC.HEADS,))


if __name__ == "__main__":
    main()
