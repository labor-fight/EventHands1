#!/usr/bin/env python3
"""XVM C.0 / C.1 information ceilings and oracle solves (docs/EXPERIMENT_RECORDS.md#xvm-prereg-20261021 §1-2).

    p0a   identity / constant velocity at the GT previous endpoint (E3), zgz + 18 training base sequences
    p0b   label-noise energy of h = 50/100/200 ms finger increments against a +-25 ms box-smoothed GT
    p0c   linear observability fraction OF on visible vertices and on the observed set O, sigma 0.25/0.5/1/2 px
    p1    GT 3-D displacement on O -> Gauss-Newton (E3 single step, E1 closed loop, all-778 self-check)
    p2    GT 2-D projected displacement on O -> GN (E3, E1)
    p3    GT contour-normal 1-D component on C -> GN (E3, E1)
    p3n   P3 with Gaussian noise at correlations c, r* = c where R_art crosses 0.85 (E3)

No network, no training. Writes outputs/xvm/s1/<probe>/{summary.json, steps.npz, receipt.json}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xvm import common as C  # noqa: E402
from xvm.common import EF  # noqa: E402

BATCH = 32
CORRS = (0.9, 0.8, 0.68, 0.5, 0.42, 0.3, 0.22)
SIGMAS = (0.25, 0.5, 1.0, 2.0)
E1_TRACE = C.REPO / "outputs/xcl/20261019/evaluations/xcl_a0foo_s3407/e1_step6000_repeat1_none/states.npz"


class Ctx:
    def __init__(self, dev):
        self.dev = dev
        self.mano = EF.Mano64(dev)
        self.m32 = C.mano32(dev)
        self.P = C.increment_precision(dev)
        self.faces = self.mano.faces


def seq_arrays(s, d):
    sq = EF.Seq(s, d)
    rows = C.endpoint_grid(sq.runs)
    return sq, rows


def t64(x, dev):
    return torch.as_tensor(np.asarray(x, np.float64), dtype=torch.float64, device=dev)


# ---------------------------------------------------------------------------------------------------- P0a
def p0a(ctx, out):
    res, steps = {}, {}
    seqs = list(C.zgz_seqs()) + EF.training_base_seqs()
    for s, d in seqs:
        sq, rows = seq_arrays(s, d)
        runs = sq.runs
        ends = np.array([r[1] for r in rows])
        prevs = np.array([r[2] for r in rows])
        starts = np.array([runs[r[0]][0] for r in rows])
        gt_end = sq.pos51[ends].astype(np.float64)
        gt_prev = sq.pos51[prevs].astype(np.float64)
        pp = prevs - C.STEP
        has = pp >= starts
        gt_pp = sq.pos51[np.where(has, pp, prevs)].astype(np.float64)
        cv = gt_prev + np.where(has[:, None], gt_prev - gt_pp, 0.0)
        betas = t64(sq.betas, ctx.dev)[None]
        ent = {"n": int(len(rows))}
        for name, pred in (("identity", gt_prev), ("const_vel", cv)):
            sc = C.score_ra(ctx.m32, pred, gt_end, sq.betas, ctx.dev)
            ea, ez, _, _ = C.r_art_terms(ctx.mano, t64(gt_prev, ctx.dev), t64(pred, ctx.dev), t64(gt_end, ctx.dev), betas)
            ent[name] = {"mpjpe_ra_mm": float(sc["mpjpe_ra_mm"].mean()), "mpvpe_ra_mm": float(sc["mpvpe_ra_mm"].mean()),
                         "wrist_mm": float(sc["wrist_mm"].mean()), "R_art": float(ea.sum() / ez.sum()),
                         "e_zero_mm": float(ez.mean())}
            steps[f"{s}|{name}_mpjpe_ra"] = sc["mpjpe_ra_mm"].astype(np.float32)
        # per-joint identity error (root aligned)
        beta = torch.tensor(sq.betas, device=ctx.dev, dtype=torch.float32).view(1, -1)
        pj, _ = C.ET.mano_fk(ctx.m32, gt_prev.astype(np.float32), beta, ctx.dev)
        gj, _ = C.ET.mano_fk(ctx.m32, gt_end.astype(np.float32), beta, ctx.dev)
        ent["identity"]["per_joint_mm"] = (np.linalg.norm(C.ET.root_align(pj) - C.ET.root_align(gj), axis=-1)
                                           .mean(0) * 1000).tolist()
        res[s] = ent
        print(s, json.dumps({k: ent[k]["mpjpe_ra_mm"] for k in ("identity", "const_vel")}), flush=True)
    train = [s for s, _ in EF.training_base_seqs()]
    res["training_base_mean"] = {k: {m: float(np.mean([res[s][k][m] for s in train])) for m in ("mpjpe_ra_mm", "R_art")}
                                 for k in ("identity", "const_vel")}
    return res, steps


# ---------------------------------------------------------------------------------------------------- P0b
def p0b(ctx, out):
    res, steps = {}, {}
    seqs = list(C.zgz_seqs()) + EF.training_base_seqs()
    for s, d in seqs:
        sq = EF.Seq(s, d)
        betas = t64(sq.betas, ctx.dev)[None]
        ent = {}
        for h in (50, 100, 200):
            raw_a, raw_b, sm_a, sm_b = [], [], [], []
            for a, b in sq.runs:
                for end in range(int(a) + 49, int(b), 50):
                    if end - h - 25 < a or end + 25 > b - 1:
                        continue
                    raw_a.append(sq.pos51[end - h]); raw_b.append(sq.pos51[end])
                    sm_a.append(C.smoothed_gt(sq.pos51, end - h, int(a), int(b) - 1))
                    sm_b.append(C.smoothed_gt(sq.pos51, end, int(a), int(b) - 1))
            if not raw_a:
                continue
            X = t64(np.concatenate([raw_a, raw_b, sm_a, sm_b]), ctx.dev)
            J = []
            for i in range(0, len(X), 4096):
                J.append(C.fk_h(ctx.mano, X[i:i + 4096], betas))
            J = torch.cat(J)
            n = len(raw_a)
            dr = J[n:2 * n] - J[:n]
            ds = J[3 * n:] - J[2 * n:3 * n]
            e = dr - ds
            rho = float((e ** 2).sum() / (dr ** 2).sum())
            floor = float(e.norm(dim=-1).mean(-1).sum() / dr.norm(dim=-1).mean(-1).sum())
            rho_j = ((e ** 2).sum(-1).sum(0) / (dr ** 2).sum(-1).sum(0).clamp(min=1e-18)).cpu().numpy()
            ent[f"h{h}"] = {"n": n, "rho_energy": rho, "R_art_floor": floor, "rho_per_joint": rho_j.tolist(),
                            "mean_raw_mm": float(dr.norm(dim=-1).mean() * 1000),
                            "mean_noise_mm": float(e.norm(dim=-1).mean() * 1000)}
        res[s] = ent
        print(s, json.dumps({k: (v["rho_energy"], v["R_art_floor"]) for k, v in ent.items()}), flush=True)
    train = [s for s, _ in EF.training_base_seqs()]
    res["training_base_mean"] = {h: {m: float(np.mean([res[s][h][m] for s in train if h in res[s]]))
                                     for m in ("rho_energy", "R_art_floor")} for h in ("h50", "h100", "h200")}
    return res, steps


# ---------------------------------------------------------------------------------------------------- P0c
def p0c(ctx, out):
    res, steps = {}, {}
    for s, d in C.zgz_seqs():
        sq, rows = seq_arrays(s, d)
        intr = sq.intrinsics()
        betas = t64(sq.betas, ctx.dev)[None]
        acc = {f"{S}_{sg}": [0.0, 0.0, 0.0, 0.0] for S in ("visible", "observed") for sg in SIGMAS}
        n_obs, n_vis = [], []
        for i0 in range(0, len(rows), BATCH):
            chunk = rows[i0:i0 + BATCH]
            x0 = t64(sq.pos51[[r[2] for r in chunk]], ctx.dev)
            ge = t64(sq.pos51[[r[1] for r in chunk]], ctx.dev)
            v0, dX, uv, z, J2, Jp = C.fd_jacobian(ctx.mano, x0, betas, intr)
            dstar = ge - x0
            for b, r in enumerate(chunk):
                vis = EF.visible(uv[b], z[b]).nonzero().flatten()
                vid, mass = C.observed(sq, r[1], ctx.faces)
                n_obs.append(len(vid)); n_vis.append(len(vis))
                for S, (ids, w) in (("visible", (vis, torch.ones(len(vis), dtype=torch.float64, device=ctx.dev))),
                                    ("observed", (vid, mass))):
                    Jb = J2[b, ids]                                           # (n,2,51)
                    H0 = torch.einsum("n,nka,nkb->ab", w, Jb, Jb)
                    for sg in SIGMAS:
                        H = H0 / sg ** 2
                        dhat = torch.linalg.solve(H + C.LAM_P * ctx.P, H @ dstar[b])
                        xh = x0[b:b + 1].clone()
                        xh[0, C.FINGER] += dhat[C.FINGER]
                        ea, ez, e, e0 = C.r_art_terms(ctx.mano, x0[b:b + 1], xh, ge[b:b + 1], betas)
                        a = acc[f"{S}_{sg}"]
                        a[0] += float((e ** 2).sum()); a[1] += float((e0 ** 2).sum())
                        a[2] += float(ea.sum()); a[3] += float(ez.sum())
        res[s] = {k: {"OF": 1 - v[0] / v[1], "R_art_lin": v[2] / v[3]} for k, v in acc.items()}
        res[s]["median_observed_vertices"] = float(np.median(n_obs))
        res[s]["median_visible_vertices"] = float(np.median(n_vis))
        print(s, json.dumps({k: round(v["OF"], 3) for k, v in res[s].items() if isinstance(v, dict)}), flush=True)
    return res, steps


# ---------------------------------------------------------------------------------------------------- GN probes
def run_gn(ctx, sq, betas, intr, x0, targets_v, targets_uv, kind, rows_ends, noise=None, sig=None, e1_fixed=None,
           vid_fixed=None):
    """Gauss-Newton for a batch of states. x0 (B,51); targets (B,778,3)/(B,778,2) absolute GT end geometry.
    kind in p1 / p1_all / p2 / p3 / p3n. Returns x (B,51) and the per-sample measured vertex count."""
    B = x0.shape[0]
    x = x0.clone()
    v0, dX0, uv0, z0, J20, Jp0 = C.fd_jacobian(ctx.mano, x0, betas, intr)
    specs = []
    for b in range(B):
        if vid_fixed is not None:
            vid, mass = vid_fixed[b]
        else:
            vid, mass = C.observed(sq, rows_ends[b], ctx.faces)
        if kind == "p1_all":
            vid = torch.arange(778, device=ctx.dev)
            mass = torch.ones(778, dtype=torch.float64, device=ctx.dev)
        e1 = None
        if kind in ("p3", "p3n"):
            _, cos, e1a = C.projected_normals(v0[b:b + 1], ctx.faces, Jp0[b:b + 1])
            keep = cos[0, vid] < C.CONTOUR_COS
            vid, mass = vid[keep], mass[keep]
            e1 = e1a[0, vid]
        specs.append((vid, mass, e1))
    for it in range(C.GN_ITERS):
        v, dX, uv, z, J2, Jp = (v0, dX0, uv0, z0, J20, Jp0) if it == 0 else C.fd_jacobian(ctx.mano, x, betas, intr)
        for b in range(B):
            vid, mass, e1 = specs[b]
            n = len(vid)
            if n == 0:
                continue
            if kind in ("p1", "p1_all"):
                A_y, dy = dX[b, vid], v[b, vid] - v0[b, vid]
                m = targets_v[b, vid] - v0[b, vid]
                Q = torch.eye(3, dtype=torch.float64, device=ctx.dev).expand(n, 3, 3)
                W = (mass / C.SIGMA_3D ** 2)[:, None, None] * torch.eye(3, dtype=torch.float64, device=ctx.dev)
            elif kind == "p2":
                A_y, dy = J2[b, vid], uv[b, vid] - uv0[b, vid]
                m = targets_uv[b, vid] - uv0[b, vid]
                Q = torch.eye(2, dtype=torch.float64, device=ctx.dev).expand(n, 2, 2)
                W = (mass / C.SIGMA_2D ** 2)[:, None, None] * torch.eye(2, dtype=torch.float64, device=ctx.dev)
            else:  # p3 / p3n: 1-D normal component, e1 fixed at x0
                A_y, dy = J2[b, vid], uv[b, vid] - uv0[b, vid]
                Q = e1[:, None, :]
                m = (e1 * (targets_uv[b, vid] - uv0[b, vid])).sum(-1, keepdim=True)
                s_ = C.SIGMA_2D
                if kind == "p3n":
                    m = m + noise[b][:, None]
                    s_ = max(float(sig), C.SIGMA_2D)
                W = (mass / s_ ** 2)[:, None, None]
            delta = C.gn_step(A_y, dy, Q, m, W, ctx.P, C.LAM_P, x[b] - x0[b])
            x[b] += delta
    return x, [len(sp[0]) for sp in specs], specs


def gt_geometry(ctx, sq, ends, betas, intr):
    ge = t64(sq.pos51[ends], ctx.dev)
    v, _ = ctx.mano.fk_chunked(ge, betas)
    uv, _ = EF.Mano64.project(v, intr)
    return ge, v, uv


def p_gn(ctx, out, kind, modes=("e3", "e1")):
    res, steps = {}, {}
    e1trace = np.load(E1_TRACE) if "e1" in modes else None
    for s, d in C.zgz_seqs():
        sq, rows = seq_arrays(s, d)
        intr = sq.intrinsics()
        betas = t64(sq.betas, ctx.dev)[None]
        ends = [r[1] for r in rows]
        ent = {}
        for mode in modes:
            preds, nmeas = [], []
            if mode == "e3":
                for i0 in range(0, len(rows), BATCH):
                    ch = rows[i0:i0 + BATCH]
                    x0 = t64(sq.pos51[[r[2] for r in ch]], ctx.dev)
                    ge, gv, guv = gt_geometry(ctx, sq, [r[1] for r in ch], betas, intr)
                    x, nm, _ = run_gn(ctx, sq, betas, intr, x0, gv, guv, kind, [r[1] for r in ch])
                    preds.append(x); nmeas += nm
                X = torch.cat(preds)
                X0 = t64(sq.pos51[[r[2] for r in rows]], ctx.dev)
            else:
                inc = e1trace[f"{s}/incoming"].astype(np.float64)
                X, X0l = [], []
                x = None
                for i, r in enumerate(rows):
                    x0 = t64(inc[i], ctx.dev)[None] if r[3] else x
                    ge, gv, guv = gt_geometry(ctx, sq, [r[1]], betas, intr)
                    x, nm, _ = run_gn(ctx, sq, betas, intr, x0, gv, guv, kind, [r[1]])
                    X.append(x); X0l.append(x0); nmeas += nm
                X, X0 = torch.cat(X), torch.cat(X0l)
            GE = t64(sq.pos51[ends], ctx.dev)
            sc = C.score_ra(ctx.m32, X.cpu().numpy(), GE.cpu().numpy(), sq.betas, ctx.dev)
            ea, ez, _, _ = C.r_art_terms(ctx.mano, X0, X, GE, betas)
            ent[mode] = {"mpjpe_ra_mm": float(sc["mpjpe_ra_mm"].mean()), "mpvpe_ra_mm": float(sc["mpvpe_ra_mm"].mean()),
                         "wrist_mm": float(sc["wrist_mm"].mean()), "R_art": float(ea.sum() / ez.sum()),
                         "median_measured_vertices": float(np.median(nmeas))}
            steps[f"{s}|{mode}_pred"] = X.cpu().numpy().astype(np.float32)
            steps[f"{s}|{mode}_mpjpe_ra"] = sc["mpjpe_ra_mm"].astype(np.float32)
            print(s, kind, mode, json.dumps(ent[mode]), flush=True)
        res[s] = ent
    return res, steps


def p3n(ctx, out):
    """Noisy contour-normal measurements: per sequence noise std from the true measurement std."""
    res, steps = {}, {}
    rng = np.random.default_rng(0)
    for s, d in C.zgz_seqs():
        sq, rows = seq_arrays(s, d)
        intr = sq.intrinsics()
        betas = t64(sq.betas, ctx.dev)[None]
        # pass 1: specs and true measurements
        cache = []
        allm = []
        for i0 in range(0, len(rows), BATCH):
            ch = rows[i0:i0 + BATCH]
            x0 = t64(sq.pos51[[r[2] for r in ch]], ctx.dev)
            ge, gv, guv = gt_geometry(ctx, sq, [r[1] for r in ch], betas, intr)
            v0, dX0, uv0, z0, J20, Jp0 = C.fd_jacobian(ctx.mano, x0, betas, intr)
            vf = []
            for b, r in enumerate(ch):
                vid, mass = C.observed(sq, r[1], ctx.faces)
                _, cos, e1a = C.projected_normals(v0[b:b + 1], ctx.faces, Jp0[b:b + 1])
                keep = cos[0, vid] < C.CONTOUR_COS
                vid, mass = vid[keep], mass[keep]
                mt = (e1a[0, vid] * (guv[b, vid] - uv0[b, vid])).sum(-1)
                allm.append(mt.cpu().numpy())
                vf.append((vid, mass))
            cache.append((ch, x0, gv, guv, vf))
        sd = float(np.concatenate(allm).std())
        ent = {"true_normal_disp_std_px": sd}
        X0 = t64(sq.pos51[[r[2] for r in rows]], ctx.dev)
        GE = t64(sq.pos51[[r[1] for r in rows]], ctx.dev)
        for c in CORRS:
            sig = sd * np.sqrt(1.0 / c ** 2 - 1.0)
            preds = []
            for ch, x0, gv, guv, vf in cache:
                noise = [torch.as_tensor(rng.standard_normal(len(v[0])) * sig, dtype=torch.float64, device=ctx.dev)
                         for v in vf]
                x, _, _ = run_gn(ctx, sq, betas, intr, x0, gv, guv, "p3n", [r[1] for r in ch], noise=noise, sig=sig,
                                 vid_fixed=vf)
                preds.append(x)
            X = torch.cat(preds)
            ea, ez, _, _ = C.r_art_terms(ctx.mano, X0, X, GE, betas)
            sc = C.score_ra(ctx.m32, X.cpu().numpy(), GE.cpu().numpy(), sq.betas, ctx.dev)
            ent[f"c{c}"] = {"noise_std_px": sig, "R_art": float(ea.sum() / ez.sum()),
                            "mpjpe_ra_mm": float(sc["mpjpe_ra_mm"].mean())}
            print(s, "p3n", c, json.dumps(ent[f"c{c}"]), flush=True)
        cs = np.array(CORRS)[::-1]                    # ascending correlation
        ra = np.array([ent[f"c{c}"]["R_art"] for c in cs])
        rstar = None
        if ra[-1] > 0.85:
            rstar = ">0.9"
        elif ra[0] <= 0.85:
            rstar = "<=0.22"
        else:
            k = int(np.argmax(ra <= 0.85))
            c0, c1, r0, r1 = cs[k - 1], cs[k], ra[k - 1], ra[k]
            rstar = float(c0 + (0.85 - r0) * (c1 - c0) / (r1 - r0))
        ent["r_star"] = rstar
        res[s] = ent
    return res, steps


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, choices=("p0a", "p0b", "p0c", "p1", "p1_all", "p2", "p3", "p3n"))
    a = ap.parse_args()
    dev = C.require_gpu()
    ctx = Ctx(dev)
    out = C.OUT / "s1" / a.probe
    if out.exists() and (out / "summary.json").exists():
        raise SystemExit(f"{out} already holds a result; preserve it")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.probe in ("p0a", "p0b", "p0c"):
        res, steps = {"p0a": p0a, "p0b": p0b, "p0c": p0c}[a.probe](ctx, out)
    elif a.probe == "p3n":
        res, steps = p3n(ctx, out)
    elif a.probe == "p1_all":
        res, steps = p_gn(ctx, out, "p1_all", modes=("e3",))
    else:
        res, steps = p_gn(ctx, out, a.probe)
    res["_meta"] = {"wall_s": time.time() - t0, "sigma_3d_m": C.SIGMA_3D, "sigma_2d_px": C.SIGMA_2D,
                    "lam_p": C.LAM_P, "gn_iters": C.GN_ITERS, "contour_cos": C.CONTOUR_COS}
    if steps:
        np.savez_compressed(out / "steps.npz", **steps)
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    C.receipt(out, {"probe": a.probe, "prereg": "docs/EXPERIMENT_RECORDS.md#xvm-prereg-20261021"}, scripts=(__file__,))
    print("DONE", a.probe, f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
