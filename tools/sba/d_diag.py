#!/usr/bin/env python3
"""SBA D-round (docs/EXPERIMENT_RECORDS.md#sba-d-prereg-20261022): why does the P0 likelihood put zgz_local's hand
~45 mm too far? Zero training; the P0 model and windows (profiled kappa), the same packets.

Hypothesis families, all scored on the packet's real events (profiled kappa unless stated):
    S   motion scale: root block (t, root) scaled by s_r in {0, 0.5, 1}, finger block by s_f in {0, 0.5, 1, 1.5},
        about the mid pose of the two GT knots          -> is the labelled motion inflated (and which block)?
    ZS  z-line (-60..+60 mm) with the root motion removed (s_r = 0)  -> does the depth bias need root motion?
    O   whole-trajectory image offset, du, dv in {-3..3} output px (camera-plane translation at the GT depth)
    G   geometric scale of the mesh about the wrist, sigma in {0.92, 0.94, ..., 1.08}, depth fixed
    T   label-event lag: knots GT(m_prev + tau), GT(m1 + tau), tau in {-30, -20, -10, 0, 10, 20, 30} ms
    K   z-line on +-25 ms box-smoothed GT knots
    E   z-line with (kappa, outlier fraction) jointly profiled
    C   z-line with the corrected forearm capsule (knuckles in OpenPose order; P0 used thumb/index/middle joints)
    PH  interior events (>= 2 px inside the end silhouette, covered at both knots): per pixel event count against
        the rotation of the surface normal |dn| (shading) and the image displacement |du| of the surface point (texture
        flow); polarity ON - OFF against sign(l . dn) with one light direction l per recording, fitted on even windows
        and scored on odd windows, against a pixel-shuffled null
    R   residual atlas at GT: events minus expectation summed over the window, in a hand frame (origin = projected
        wrist, x along wrist -> middle MCP, unit = projected wrist-to-middle-tip length), 64 x 64 over [-1.5, 1.5]^2

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python tools/sba/d_diag.py run --domain zgz_local --shard 0 --shards 2
    python tools/sba/d_diag.py summarize
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sba import common as CM  # noqa: E402
from sba.common import SB  # noqa: E402
from sba.p0_landscape import packets  # noqa: E402
from semkine.kssf import KSSF  # noqa: E402
from semkine.xyz_mesh import vertex_normals  # noqa: E402

OUT = CM.OUT / "d"
ZLINE = [k * 5 for k in range(-12, 13)]
S_ROOT, S_FING = (0.0, 0.5, 1.0), (0.0, 0.5, 1.0, 1.5)
OFFS = list(range(-3, 4))
SIGMAS = [0.92 + 0.02 * k for k in range(9)]
LAGS = [-30, -20, -10, 0, 10, 20, 30]
ATLAS = 64
ROOTB, FINGB = slice(0, 6), slice(6, 51)
MIDDLE_MCP, MIDDLE_TIP = 9, 12          # OpenPose order


def profile2(lam_h, U, B, N, iters=40):
    """Joint (k, eta) >= 0 maximising sum N log(k H + eta U + B) - k sum H - eta sum U (one hypothesis)."""
    H, U, B, N = lam_h.flatten().double(), U.flatten().double(), B.flatten().double(), N.flatten().double()
    th = torch.tensor([1.0, 0.3], dtype=torch.float64, device=H.device)
    X = torch.stack([H, U], 1)
    T = X.sum(0)
    for _ in range(iters):
        lam = (X @ th + B).clamp(min=1e-12)
        g = X.T @ (N / lam) - T
        Hm = -(X * (N / lam ** 2)[:, None]).T @ X - 1e-9 * torch.eye(2, dtype=torch.float64, device=H.device)
        th = (th - torch.linalg.solve(Hm, g)).clamp(min=1e-8)
    lam = (X @ th + B).clamp(min=1e-12)
    return float((N * lam.log()).sum() - lam.sum()), th.tolist()


class Scorer:
    def __init__(self, ctx, sd, win, model):
        self.ctx, self.sd, self.win, self.model = ctx, sd, win, model
        self.S = int(win.widths.numel())
        self.N = SB.counts(win, model)

    def areas(self, x0, x1, sigma=None):
        geo = self.ctx.geo
        v, uv, j = geo.trajectories(x0[None], x1[None], self.win.alphas, self.sd.betas, self.sd.K)
        if sigma is not None:
            w = j[:, :, :1]
            v = w + sigma * (v - w)
            uv = SB.project_src(v, self.sd.K)
        plan = SB.build_plan(v[0], uv[0], geo.topo, geo.grid, geo.groups, geo.n_groups)
        return SB.sweep_area(uv, plan, geo.grid, self.win.crop, self.S)

    def ll(self, A):
        k = SB.profile_kappa(A, self.win, self.model)
        m2 = replace(self.model, kappa_sil=self.model.kappa_sil * float(k[0]),
                     kappa_int=self.model.kappa_int * float(k[0]))
        return float(SB.loglik(SB.rates(A, self.win, m2), self.N)[0]), m2

    def ll_eta(self, A):
        m0 = replace(self.model, outlier=0.0)
        hand = SB.rates(A, self.win, m0) - SB.rates(torch.zeros_like(A), self.win, m0)
        return profile2(hand, self.win.uni[None].expand_as(hand), self.win.bkg[None].expand_as(hand), self.N)


def photometric(ctx, kssf, sd, prev, end, x0, x1, inner_px=2.0):
    """Per interior pixel of the end silhouette: events N, signed ON - OFF, normal change dn (3) and image displacement
    |du| (px) of the surface point between the two knots (output grid)."""
    v, _ = ctx.geo.fk(torch.stack([x0, x1]), sd.betas)
    K = sd.K[None].expand(2, 3, 3)
    f = kssf.rasterize(v, K)
    face, bary, sdf = f.face_id[1], f.bary[1], f.sdf[1]
    inside = (face >= 0) & (f.face_id[0] >= 0) & (sdf < -inner_px)
    if not bool(inside.any()):
        return None
    faces = ctx.geo.topo.faces
    nv = vertex_normals(v.float(), faces)                                    # (2, V, 3)
    fv = faces[face[inside]]                                                 # (P, 3)
    b = bary[inside][..., None]                                              # (P, 3, 1)
    n0 = (nv[0][fv] * b).sum(1)
    n1 = (nv[1][fv] * b).sum(1)
    n0 = n0 / n0.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    n1 = n1 / n1.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    p0 = (v[0][fv] * b).sum(1)
    p1 = (v[1][fv] * b).sum(1)
    du = (SB.project_src(p1, sd.K) - SB.project_src(p0, sd.K)).norm(dim=-1) * (SB.OUT_W / SB.SRC_W)
    ev, ms = sd.events(prev, end)
    cnt = torch.zeros(2, SB.OUT_H, SB.OUT_W, device=v.device)
    if len(ms):
        x = torch.from_numpy(ev[:, 0].astype(np.int64)).to(v.device)
        y = torch.from_numpy(ev[:, 1].astype(np.int64)).to(v.device)
        p = torch.from_numpy(np.clip(ev[:, 2], 0, 1).astype(np.int64)).to(v.device)
        cnt.view(-1).index_add_(0, (p * SB.OUT_H + y) * SB.OUT_W + x, torch.ones(len(ms), device=v.device))
    on, off = cnt[1][inside], cnt[0][inside]
    return {"N": (on + off).cpu().numpy().astype(np.float32), "S": (on - off).cpu().numpy().astype(np.float32),
            "dn": (n1 - n0).cpu().numpy().astype(np.float32), "du": du.cpu().numpy().astype(np.float32)}


def atlas(ctx, sd, win, sc, x0, x1, A):
    """Window residual (events - expectation) in the hand frame of the end knot, plus its event / model masses."""
    _, m2 = sc.ll(A)
    lam = SB.rates(A, win, m2)[0, 0].sum(0)                                  # (h, w)
    N = win.N.sum((0, 1)).float()
    res = N - lam
    _, uv, j = ctx.geo.trajectories(x1[None], x1[None], win.alphas[:2], sd.betas, sd.K)
    jo = SB.project_src(j[0, 0], sd.K)                                       # (21, 2) source px
    o, mcp, tip = jo[0], jo[MIDDLE_MCP], jo[MIDDLE_TIP]
    ax = mcp - o
    ax = ax / ax.norm().clamp(min=1e-6)
    L = (tip - o).norm().clamp(min=1e-6)
    g = ctx.geo.grid
    cy = g.cy_src[win.crop.y0:win.crop.y0 + win.crop.h][:, None].expand(win.crop.h, win.crop.w)
    cx = g.cx_src[win.crop.x0:win.crop.x0 + win.crop.w][None, :].expand(win.crop.h, win.crop.w)
    dx, dy = cx - o[0], cy - o[1]
    a = (dx * ax[0] + dy * ax[1]) / L
    b = (-dx * ax[1] + dy * ax[0]) / L
    ia = ((a + 1.5) / 3.0 * ATLAS).floor().long()
    ib = ((b + 1.5) / 3.0 * ATLAS).floor().long()
    ok = (ia >= 0) & (ia < ATLAS) & (ib >= 0) & (ib < ATLAS)
    idx = (ib * ATLAS + ia)[ok]
    out = []
    for f in (res, N, lam):
        h = torch.zeros(ATLAS * ATLAS, dtype=torch.float64, device=f.device)
        h.index_add_(0, idx, f[ok].double())
        out.append(h.view(ATLAS, ATLAS).cpu().numpy())
    return out


def run(domain, shard, shards):
    dev = CM.require_gpu()
    torch.set_grad_enabled(False)
    ctx = CM.Ctx(dev)
    model = replace(CM.load_model(), signed=False)
    items = packets(domain)[shard::shards]
    cache = {}
    recs = []
    at = {k: np.zeros((ATLAS, ATLAS)) for k in ("res", "N", "lam")}
    kssf = KSSF(ctx.mano, SB.OUT_H, SB.OUT_W, SB.OUT_W / SB.SRC_W).to(dev)
    phot = {k: [] for k in ("N", "S", "dn", "du", "win")}
    t0 = time.time()
    for seq, d, index, row, rows in items:
        if seq not in cache:
            cache = {seq: CM.SeqData(seq, d, dev)}
        sd = cache[seq]
        prev, end = int(row["previous_end_ms"]), int(row["end_ms"])
        x0, x1 = sd.gt(prev), sd.gt(end)
        win = CM.window(ctx, sd, prev, end, x0, x1, mcp=SB.MCP_JOINTS_LEGACY)       # P0's window
        rec = {"seq": seq, "index": index, "end": end, "n_events": win.n_events}
        if win.n_events < CM.MIN_EVENTS:
            recs.append(rec)
            continue
        sc = Scorer(ctx, sd, win, model)
        mid = 0.5 * (x0 + x1)

        def scaled(sr, sf):
            a, b = x0.clone(), x1.clone()
            for blk, s in ((ROOTB, sr), (FINGB, sf)):
                a[blk] = mid[blk] - s * (mid[blk] - x0[blk])
                b[blk] = mid[blk] + s * (x1[blk] - mid[blk])
            return a, b
        rec["S"] = {f"{sr:g},{sf:g}": sc.ll(sc.areas(*scaled(sr, sf)))[0] for sr in S_ROOT for sf in S_FING}
        a0, b0 = scaled(0.0, 1.0)
        rec["ZS"] = {}
        for z in ZLINE:
            a, b = a0.clone(), b0.clone()
            a[2] += z / 1000.0
            b[2] += z / 1000.0
            rec["ZS"][str(z)] = sc.ll(sc.areas(a, b))[0]
        f_out = float(sd.K[0, 0]) * SB.OUT_W / SB.SRC_W
        Zd = float(x1[2])
        rec["O"] = {}
        for du in OFFS:
            for dv in OFFS:
                a, b = x0.clone(), x1.clone()
                a[0] += du * Zd / f_out; b[0] += du * Zd / f_out
                a[1] += dv * Zd / f_out; b[1] += dv * Zd / f_out
                rec["O"][f"{du},{dv}"] = sc.ll(sc.areas(a, b))[0]
        rec["G"] = {f"{s:.2f}": sc.ll(sc.areas(x0, x1, sigma=s))[0] for s in SIGMAS}
        ra, rb = sd.run_bounds(end)
        rec["T"] = {}
        for tau in LAGS:
            p2, e2 = min(max(prev + tau, ra), rb), min(max(end + tau, ra), rb)
            rec["T"][str(tau)] = sc.ll(sc.areas(sd.gt(p2), sd.gt(e2)))[0]
        xs0 = torch.tensor(CM.C.smoothed_gt(sd.pos51, prev, ra, rb), dtype=torch.float32, device=dev)
        xs1 = torch.tensor(CM.C.smoothed_gt(sd.pos51, end, ra, rb), dtype=torch.float32, device=dev)
        rec["K"], rec["E"] = {}, {}
        for z in ZLINE:
            a, b = xs0.clone(), xs1.clone()
            a[2] += z / 1000.0
            b[2] += z / 1000.0
            rec["K"][str(z)] = sc.ll(sc.areas(a, b))[0]
            a, b = x0.clone(), x1.clone()
            a[2] += z / 1000.0
            b[2] += z / 1000.0
            rec["E"][str(z)] = sc.ll_eta(sc.areas(a, b))[0]
        wc = CM.window(ctx, sd, prev, end, x0, x1)
        scc = Scorer(ctx, sd, wc, model)
        rec["C"] = {}
        for z in ZLINE:
            a, b = x0.clone(), x1.clone()
            a[2] += z / 1000.0
            b[2] += z / 1000.0
            rec["C"][str(z)] = scc.ll(scc.areas(a, b))[0]
        ph = photometric(ctx, kssf, sd, prev, end, x0, x1)
        if ph is not None:
            for k, v in ph.items():
                phot[k].append(v)
            phot["win"].append(np.full(len(ph["N"]), len(recs), np.int32))
        A = sc.areas(x0, x1)
        r_, n_, l_ = atlas(ctx, sd, win, sc, x0, x1, A)
        at["res"] += r_; at["N"] += n_; at["lam"] += l_
        recs.append(rec)
        if len(recs) % 50 == 0:
            print(json.dumps({"domain": domain, "shard": shard, "done": len(recs), "of": len(items),
                              "s": round(time.time() - t0)}), flush=True)
    path = OUT / "raw" / f"{domain}__{shard:02d}of{shards:02d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": recs, "wall_s": time.time() - t0}))
    np.savez_compressed(path.with_suffix(".npz"), **at)
    if phot["N"]:
        seqs = np.array([r["seq"] for r in recs])
        np.savez_compressed(path.with_name(path.stem + "_phot.npz"),
                            **{k: np.concatenate(v) for k, v in phot.items()}, seqs=seqs)
    CM.receipt(OUT / "raw" / f"{domain}__{shard:02d}", {"probe": "D", "domain": domain, "shard": shard},
               scripts=(__file__,))


def _argmax(recs, fam, keys, conv=float):
    return [conv(keys[int(np.argmax([r[fam][k] for k in keys]))]) for r in recs if fam in r]


def summarize():
    res = {"domains": {}}
    zk = [str(z) for z in ZLINE]
    for domain in ("zgz_local", "zgz_global", "check_local", "check_global"):
        src = "check" if domain.startswith("check") else domain
        files = sorted((OUT / "raw").glob(f"{src}__*of*.json"))
        if not files:
            continue
        recs = [r for f in files for r in json.loads(f.read_text())["records"] if "S" in r]
        if domain.startswith("check"):
            recs = [r for r in recs if f"_{domain.split('_')[1]}" in r["seq"]]
        d = {"n": len(recs)}
        sk = [f"{sr:g},{sf:g}" for sr in S_ROOT for sf in S_FING]
        best = _argmax(recs, "S", sk, str)
        d["S_argmax_freq"] = {k: float(np.mean([b == k for b in best])) for k in sk}
        d["S_root_scale_median"] = float(np.median([float(b.split(",")[0]) for b in best]))
        d["S_finger_scale_median"] = float(np.median([float(b.split(",")[1]) for b in best]))
        for fam in ("ZS", "K", "E", "C"):
            z = np.array(_argmax(recs, fam, zk))
            d[f"{fam}_zstar_median_mm"] = float(np.median(z))
            d[f"{fam}_frac_abs_le10"] = float(np.mean(np.abs(z) <= 10))
        ok = [f"{du},{dv}" for du in OFFS for dv in OFFS]
        bo = [tuple(map(int, b.split(","))) for b in _argmax(recs, "O", ok, str)]
        d["O_offset_median_px"] = [float(np.median([b[0] for b in bo])), float(np.median([b[1] for b in bo]))]
        d["O_frac_zero"] = float(np.mean([b == (0, 0) for b in bo]))
        gk = [f"{s:.2f}" for s in SIGMAS]
        d["G_sigma_median"] = float(np.median(_argmax(recs, "G", gk)))
        tk = [str(t) for t in LAGS]
        d["T_lag_median_ms"] = float(np.median(_argmax(recs, "T", tk)))
        res["domains"][domain] = d
    for src in ("zgz_local", "zgz_global", "check"):
        files = sorted(f for f in (OUT / "raw").glob(f"{src}__*of*.npz") if not f.name.endswith("_phot.npz"))
        if not files:
            continue
        at = {k: sum(np.load(f)[k] for f in files) for k in ("res", "N", "lam")}
        c = (np.arange(ATLAS) + 0.5) / ATLAS * 3.0 - 1.5
        A_, B_ = np.meshgrid(c, c)
        pos = np.clip(at["res"], 0, None)
        tot = pos.sum()
        res.setdefault("atlas", {})[src] = {
            "positive_residual_fraction_of_events": float(tot / max(at["N"].sum(), 1)),
            "positive_residual_share_behind_wrist(a<0)": float(pos[A_ < 0].sum() / max(tot, 1e-9)),
            "positive_residual_share_far(r>1.1)": float(pos[np.hypot(A_, B_) > 1.1].sum() / max(tot, 1e-9)),
            "events_mean_a": float((at["N"] * A_).sum() / max(at["N"].sum(), 1)),
            "model_mean_a": float((at["lam"] * A_).sum() / max(at["lam"].sum(), 1)),
            "events_radius_rms": float(np.sqrt((at["N"] * (A_ ** 2 + B_ ** 2)).sum() / max(at["N"].sum(), 1))),
            "model_radius_rms": float(np.sqrt((at["lam"] * (A_ ** 2 + B_ ** 2)).sum() / max(at["lam"].sum(), 1)))}
    res["photometric"] = photometric_summary()
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def _spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def _partial_spearman(a, b, c):
    """Spearman of a and b after removing the rank-linear effect of c from both."""
    from scipy.stats import rankdata
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    X = np.stack([rc, np.ones_like(rc)], 1)
    ea = ra - X @ np.linalg.lstsq(X, ra, rcond=None)[0]
    eb = rb - X @ np.linalg.lstsq(X, rb, rcond=None)[0]
    return float(np.corrcoef(ea, eb)[0, 1])


def photometric_summary(rng_seed=0):
    out = {}
    rng = np.random.default_rng(rng_seed)
    for src in ("zgz_local", "zgz_global", "check"):
        files = sorted((OUT / "raw").glob(f"{src}__*of*_phot.npz"))
        if not files:
            continue
        parts = []
        for f in files:
            z = np.load(f)
            seqs = z["seqs"]
            parts.append({k: z[k] for k in ("N", "S", "dn", "du", "win")} | {"seq": seqs[z["win"]], "file": str(f)})
        groups = {}
        for q in parts:
            for sq in np.unique(q["seq"]):
                m = q["seq"] == sq
                key = sq if src == "check" else src
                groups.setdefault(key, []).append({k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in q.items()})
        stats = {"local": [], "global": []}
        corr = {"local": [], "global": []}
        for key, lst in groups.items():
            d = {k: np.concatenate([q[k] for q in lst]) for k in ("N", "S", "dn", "du")}
            w = np.concatenate([q["win"] + 100000 * i for i, q in enumerate(lst)])
            cat = "local" if "local" in key else "global"
            dn = np.linalg.norm(d["dn"], axis=1)
            corr[cat].append((len(d["N"]), _spearman(d["N"], dn), _spearman(d["N"], d["du"]),
                              _partial_spearman(d["N"], dn, d["du"]), _partial_spearman(d["N"], d["du"], dn)))
            m = d["S"] != 0
            fit, ev_ = m & (w % 2 == 0), m & (w % 2 == 1)
            if fit.sum() < 50 or ev_.sum() < 50:
                continue
            X = np.concatenate([d["dn"], np.ones((len(d["dn"]), 1))], 1)
            l = np.linalg.lstsq(X[fit], np.sign(d["S"][fit]), rcond=None)[0]
            acc = float(np.mean(np.sign(X[ev_] @ l) == np.sign(d["S"][ev_])))
            Xs = X[ev_].copy()
            Xs[:, :3] = Xs[rng.permutation(len(Xs)), :3]
            null = float(np.mean(np.sign(Xs @ l) == np.sign(d["S"][ev_])))
            base = float(max(np.mean(d["S"][ev_] > 0), np.mean(d["S"][ev_] < 0)))
            stats[cat].append((int(ev_.sum()), acc, null, base))
        res = {}
        for cat in ("local", "global"):
            if corr[cat]:
                c = np.array(corr[cat])
                wts = c[:, 0] / c[:, 0].sum()
                res[f"{cat}_spearman_N_dn"] = float((c[:, 1] * wts).sum())
                res[f"{cat}_spearman_N_du"] = float((c[:, 2] * wts).sum())
                res[f"{cat}_partial_N_dn|du"] = float((c[:, 3] * wts).sum())
                res[f"{cat}_partial_N_du|dn"] = float((c[:, 4] * wts).sum())
                res[f"{cat}_pixels"] = int(c[:, 0].sum())
            if stats[cat]:
                a = np.array(stats[cat])
                wts = a[:, 0] / a[:, 0].sum()
                res[f"{cat}_polarity_acc_heldout"] = float((a[:, 1] * wts).sum())
                res[f"{cat}_polarity_acc_shuffled"] = float((a[:, 2] * wts).sum())
                res[f"{cat}_polarity_majority"] = float((a[:, 3] * wts).sum())
                res[f"{cat}_recordings"] = len(a)
        out[src] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "summarize"))
    ap.add_argument("--domain", choices=("zgz_local", "zgz_global", "check"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.domain, a.shard, a.shards)
    else:
        summarize()


if __name__ == "__main__":
    main()
