#!/usr/bin/env python3
r"""[XYZ-ASSOC] align probe (zero training, read-only on repository and checkpoint).

Question: how much of the closed-loop association collapse (prev error -> wrong ray/mesh assignment) is won back by
aligning the *whole* previous mesh with a 2D similarity (translation, rotation, scale) before `associate_rays`, and can
that transform be estimated from the events alone?

Stages (run separately; `--part a` needs the GPU, `--part b,report` are CPU only):

    a       closed-loop replay of `eval_track.track_sequence` (rng(0), 50 ms, val_core = zgz_global + zgz_local,
            noise scale 1) of xa_c1_s3407 step 6000 with a `state_hook` that records the model's own prev of every step,
            its prediction, the window-start / window-end ground truth and the step index
            (`closedloop_prev_<seq>.npz`). The hook returns `pred` untouched. RA must equal the selection grid (22.934).
    b       per step, on the <= 2048 events `stride_sample` hands to the model (same sampling and source labels as P2),
            run the *current* `associate_rays` (hit_first, band 16 px, front 3 px, z_tol 0.01, k 16) under several prev
            meshes and count with P2's `count_step`:
              cl               the closed-loop prev as is
              sim2d_oracle     2D similarity (least squares over all 778 projected vertices, prev -> window-end GT
                               projection) applied to prev's projection, back-projected at prev's own vertex depths
              rigid3d_oracle   prev with root translation + root rotation (prev[0:6]) replaced by the window-end GT
              sim2d_est        events only: robust ICP (events -> nearest silhouette-contour pixel of prev's rasterised
                               mesh, Tukey weights with c = 16 px, closed-form similarity, <= 10 iterations, identity
                               start; identity if < 100 events), applied like sim2d_oracle
              gt               window-start GT (the teacher-forced reference of P2-b)
            supplementary: trans2d_oracle / trans2d_est (translation only), sim2d_est_cf (same ICP, Tukey scale
            annealed 64 -> 16 px).
    report  tables -> align_probe.json, REPORT.md

Metrics are P2's (`probe_xa_assoc.count_step` / `rates`): exact part agreement = association-mass-weighted hard-joint
argmax == src_part over events with src_part >= 0 (unassociated events count as disagreement, `agree_exact_allhand`);
M = share of finger-contour foreground events (src_kind in {1,2}, src_part >= 1) whose association puts > 50 % of the mass
on joint 0; hit = share of sampled events whose ray hits the mesh surface.

Usage (server):
    CUDA_VISIBLE_DEVICES=2 python tools/xa/probe_align.py --part a
    CUDA_VISIBLE_DEVICES= python tools/xa/probe_align.py --part b,report --workers 64
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO), str(REPO / "model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                         # noqa: E402
import probe_xa_assoc as PX                          # noqa: E402  (P2: Labels / SeqCtx / count_step / rates / hook)
from scipy.spatial import cKDTree                    # noqa: E402
from scipy.spatial.transform import Rotation         # noqa: E402

ET = PX.ET
H, W = 180, 240
ST = PX.ST
RAD2DEG = PX.RAD2DEG
KEYS = list(PX.COUNT_KEYS)
CONDS = ("cl", "sim2d_oracle", "rigid3d_oracle", "sim2d_est", "gt", "trans2d_oracle", "trans2d_est", "sim2d_est_cf",
         "sim2d_est_it30", "sim2d_est_tight", "sim2d_est_p2l", "gt_end")
MAIN = CONDS[:5]
SUPP = CONDS[5:]
#: estimator variants: (degrees of freedom, Tukey scale per iteration [px]; the length is the iteration cap)
EST_SPEC = {"sim2d_est": ("sim", [16.0] * 10),
            "trans2d_est": ("trans", [16.0] * 10),
            "sim2d_est_cf": ("sim", [64.0, 48.0, 32.0, 24.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0]),
            "sim2d_est_it30": ("sim", [16.0] * 30),
            "sim2d_est_tight": ("sim", [16.0] * 10 + [4.0] * 10),
            "sim2d_est_p2l": ("sim_p2l", [16.0] * 10)}
#: synthetic recovery test: known similarity perturbations (about the GT hand centroid) of the window-end GT mesh
SYN_ITEMS = ([("t_px", m) for m in (0.0, 2.0, 4.0, 8.0, 16.0, 32.0)] + [("rot_deg", d) for d in (5.0, 10.0, 20.0)] +
             [("scale", f) for f in (0.85, 0.92, 1.08, 1.15)])
MIN_EV_EST = 100                                     # fewer sampled events than this: no estimate (identity)
MIN_CONTOUR = 20
OFFSET_BINS = ((0.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, float("inf")))
OFFSET_BINS_FINE = ((8.0, 16.0), (16.0, 32.0), (32.0, float("inf")))
log = PX.log


# --------------------------------------------------------------------------- 2D similarity algebra
def rot2(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


def apply_sim(P, T):
    s, th, t = T
    return s * (P @ rot2(th).T) + t


def fit_sim(P, Q, w=None, dof="sim"):
    """Weighted least squares min sum w |s R p + t - q|^2 (2D Umeyama, closed form). Returns (s, theta, t) or None
    when degenerate. dof='trans' fixes s = 1, R = I."""
    if w is None:
        w = np.ones(len(P))
    sw = float(w.sum())
    if sw <= 1e-9:
        return None
    pb = (w[:, None] * P).sum(0) / sw
    qb = (w[:, None] * Q).sum(0) / sw
    if dof == "trans":
        return 1.0, 0.0, qb - pb
    P0, Q0 = P - pb, Q - qb
    a = float((w * (P0 * Q0).sum(1)).sum())
    b = float((w * (P0[:, 0] * Q0[:, 1] - P0[:, 1] * Q0[:, 0])).sum())
    den = float((w * (P0 * P0).sum(1)).sum())
    if den < 1e-9 or a * a + b * b < 1e-18:
        return None
    th = math.atan2(b, a)
    s = math.hypot(a, b) / den
    return s, th, qb - s * (rot2(th) @ pb)


def wrap_deg(x):
    return (x + 180.0) % 360.0 - 180.0


def tukey_w(r, c):
    x = r / c
    return np.where(x < 1.0, (1.0 - x * x) ** 2, 0.0)


def icp(C, E, dof, cs):
    """Robust ICP. Model = silhouette-contour pixels C (n, 2) of prev's projection, observations = event pixels E (m, 2).
    Each iteration: every event takes its nearest (currently transformed) contour point, Tukey weight of that distance
    (scale cs[it] px), then the closed-form similarity from the ORIGINAL contour points to the events. Identity start,
    at most len(cs) iterations.
    Returns (T, n_iter, n_inliers)."""
    T = (1.0, 0.0, np.zeros(2))
    cbar = C.mean(0)
    n_iter, n_inl = 0, 0
    for it in range(len(cs)):
        c = cs[min(it, len(cs) - 1)]
        d, j = cKDTree(apply_sim(C, T)).query(E)
        w = tukey_w(d, c)
        k = int((w > 0).sum())
        if k < 10:
            break
        sol = fit_sim(C[j], E, w, dof)
        if sol is None:
            break
        n_inl = k
        n_iter = it + 1
        moved = np.linalg.norm(apply_sim(cbar[None], sol) - apply_sim(cbar[None], T))
        done = moved < 0.01 and abs(sol[1] - T[1]) < 1e-4 and abs(sol[0] - T[0]) < 1e-4
        T = sol
        if done:
            break
    return T, n_iter, n_inl


def contour_normals(C, k=7):
    """Unit normals of an (n, 2) pixel contour: smallest-eigenvalue direction of the local covariance of the k nearest
    contour points (sign irrelevant for point-to-line residuals)."""
    k = min(k, len(C))
    _, nn = cKDTree(C).query(C, k=k)
    P = C[nn]
    P = P - P.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", P, P)
    _, v = np.linalg.eigh(cov)
    return v[:, :, 0]


def icp_p2l(C, N, E, cs):
    """Point-to-line variant of `icp` (supplementary): same nearest-point matching and Tukey weights, but each
    iteration takes one Gauss-Newton step on the weighted point-to-line residuals n . (e - T c) over (t, theta, s),
    which removes the aperture ('sliding along the contour') degeneracy of point-to-point ICP."""
    T = (1.0, 0.0, np.zeros(2))
    cbar = C.mean(0)
    n_iter, n_inl = 0, 0
    for it in range(len(cs)):
        s_, th, t = T
        R = rot2(th)
        Cp = s_ * (C @ R.T) + t
        d, j = cKDTree(Cp).query(E)
        w = tukey_w(d, cs[min(it, len(cs) - 1)])
        k = int((w > 0).sum())
        if k < 10:
            break
        Cj = C[j]
        Nj = N[j] @ R.T
        r = ((E - Cp[j]) * Nj).sum(1)
        c_, s_th = math.cos(th), math.sin(th)
        dR = np.array([[-s_th, -c_], [c_, -s_th]])
        a = np.stack([Nj[:, 0], Nj[:, 1], (Nj * (s_ * (Cj @ dR.T))).sum(1), (Nj * (Cj @ R.T)).sum(1)], 1)
        A = a.T @ (w[:, None] * a) + 1e-6 * np.eye(4)
        try:
            dl = np.linalg.solve(A, a.T @ (w * r))
        except np.linalg.LinAlgError:
            break
        sol = (max(s_ + dl[3], 0.2), th + dl[2], t + dl[:2])
        n_inl, n_iter = k, it + 1
        moved = np.linalg.norm(apply_sim(cbar[None], sol) - apply_sim(cbar[None], T))
        done = moved < 0.01 and abs(sol[1] - T[1]) < 1e-4 and abs(sol[0] - T[0]) < 1e-4
        T = sol
        if done:
            break
    return T, n_iter, n_inl


# --------------------------------------------------------------------------- projection, silhouette
def project_np(verts, intr):
    fx, fy, cx, cy = intr
    z = np.maximum(verts[:, 2], 1e-6)
    return np.stack([fx * verts[:, 0] / z + cx, fy * verts[:, 1] / z + cy], 1), z


def backproject_np(uv, z, intr):
    fx, fy, cx, cy = intr
    return np.stack([(uv[:, 0] - cx) / fx * z, (uv[:, 1] - cy) / fy * z, z], 1)


def silhouette_contour(uv, faces, eps=1e-5):
    """Rasterise the union of all projected triangles (pixel centres at integer coordinates, the labels' convention)
    plus the rounded vertices, and return the boundary pixels (mask pixels with a 4-neighbour outside the mask; the
    frame border does not count as outside) as an (n, 2) float64 array of (x, y), and the mask."""
    fu, fv = uv[faces, 0], uv[faces, 1]
    x0, x1, x2 = fu[:, 0], fu[:, 1], fu[:, 2]
    y0, y1, y2 = fv[:, 0], fv[:, 1], fv[:, 2]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    ok = (np.abs(area) > 1e-9) & (fu.max(1) >= 0) & (fu.min(1) <= W - 1) & (fv.max(1) >= 0) & (fv.min(1) <= H - 1)
    xl = np.clip(np.floor(fu.min(1)), 0, W - 1).astype(np.int64)
    xh = np.clip(np.ceil(fu.max(1)), 0, W - 1).astype(np.int64)
    yl = np.clip(np.floor(fv.min(1)), 0, H - 1).astype(np.int64)
    yh = np.clip(np.ceil(fv.max(1)), 0, H - 1).astype(np.int64)
    nw, nh = xh - xl + 1, yh - yl + 1
    n = np.where(ok, nw * nh, 0)
    total = int(n.sum())
    mask = np.zeros(H * W, bool)
    if 0 < total < 40_000_000:
        fid = np.repeat(np.arange(len(faces)), n)
        k = np.arange(total) - np.repeat(np.cumsum(n) - n, n)
        px = xl[fid] + k % nw[fid]
        py = yl[fid] + k // nw[fid]
        A = area[fid]
        dxp, dyp = px - x0[fid], py - y0[fid]
        b1 = (dxp * (y2 - y0)[fid] - (x2 - x0)[fid] * dyp) / A
        b2 = ((x1 - x0)[fid] * dyp - dxp * (y1 - y0)[fid]) / A
        b0 = 1.0 - b1 - b2
        inside = (b0 >= -eps) & (b1 >= -eps) & (b2 >= -eps)
        mask[(py * W + px)[inside]] = True
    ui, vi = np.round(uv[:, 0]).astype(np.int64), np.round(uv[:, 1]).astype(np.int64)
    okp = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    mask[(vi * W + ui)[okp]] = True
    m = mask.reshape(H, W)
    p = np.pad(m, 1, constant_values=True)
    inner = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    ys, xs = np.nonzero(m & ~inner)
    return np.stack([xs, ys], 1).astype(np.float64), m


# --------------------------------------------------------------------------- worker (CPU)
_S = {}


def _init_worker(cfgd):
    torch.set_num_threads(1)
    os.chdir(cfgd["repo"])
    from mano_layer import ManoLayer
    from pose_repr import decode_to_mano_inputs
    from semkine.xyz_mesh import associate_rays, vertex_face_table
    mano = ManoLayer(cfgd["mano_npz"], add_mean=False).eval()
    faces = mano.f
    vert_faces = vertex_face_table(faces, mano.weights.shape[0])
    lbs_hard = mano.weights.detach().argmax(1)

    def fk(params51, betas):
        p = torch.from_numpy(np.ascontiguousarray(params51, dtype=np.float32)).view(1, -1)
        dec = decode_to_mano_inputs(p, "mano_full_axis_angle", mano.hands_components, mano.hands_mean)
        return mano(betas, dec["global_orient"], dec["local_full_aa"], dec["transl"])

    def assoc(verts, xy, part, kind, intr_t):
        """Current associate_rays on the sampled events, counted with P2's count_step (same code path as
        `_xyz_mesh_forward`: render-frame intrinsics, d_z = 1 rays, band / front_tol in ray units)."""
        n = xy.shape[0]
        if n == 0:
            return np.zeros(len(KEYS))
        fx, fy, cx, cy = intr_t
        px = torch.from_numpy(np.ascontiguousarray(xy[:, 0])).view(1, -1)
        py = torch.from_numpy(np.ascontiguousarray(xy[:, 1])).view(1, -1)
        d = torch.stack([(px - cx[:, None]) / fx[:, None], (py - cy[:, None]) / fy[:, None], torch.ones_like(px)], -1)
        mask = torch.ones(1, n, dtype=torch.bool)
        f_px = (fx * fy).sqrt()
        a = associate_rays(d, mask, verts, faces, vert_faces, band=cfgd["band_px"] / f_px,
                           front_tol=cfgd["front_px"] / f_px, z_tol=cfgd["z_tol"], k=cfgd["k"])
        ev = PX.event_majority(a, lbs_hard)
        cnt, _, _ = PX.count_step(part, kind, ev)
        return np.array([float(cnt[k]) for k in KEYS])

    _S.update(fk=fk, assoc=assoc, faces=faces.numpy(), render_scale=cfgd["render_scale"])


def _to_t(V):
    return torch.from_numpy(np.ascontiguousarray(V, dtype=np.float32)).view(1, -1, 3)


def _est_record(T, To, cbar, uv_p, uv_g, status, n_iter, n_inl):
    """Estimated-vs-oracle transform error: centroid displacement, rotation, scale; vertex RMS after the estimate."""
    s_e, th_e, _ = T
    s_o, th_o, _ = To
    ce, co = apply_sim(cbar[None], T)[0], apply_sim(cbar[None], To)[0]
    return {"status": status, "n_iter": n_iter, "n_inl": n_inl, "s": s_e, "th_deg": th_e * RAD2DEG,
            "err_c_px": float(np.linalg.norm(ce - co)), "err_th_deg": wrap_deg((th_e - th_o) * RAD2DEG),
            "s_ratio": float(s_e / s_o), "rms_px": float(np.sqrt(((apply_sim(uv_p, T) - uv_g) ** 2).sum(1).mean()))}


def process_step(task):
    t0 = time.time()
    fk, assoc, faces = _S["fk"], _S["assoc"], _S["faces"]
    betas = torch.from_numpy(task["betas"]).float().view(1, -1)
    K = torch.from_numpy(task["K"]).float().view(1, 3, 3)
    sc = _S["render_scale"]
    intr_t = (K[:, 0, 0] * sc, K[:, 1, 1] * sc, K[:, 0, 2] * sc, K[:, 1, 2] * sc)
    intr = tuple(float(x) for x in intr_t)
    prev, ge, gs = task["prev"], task["gt_end"], task["gt_start"]
    xy, part, kind = task["xy"], task["part"], task["kind"]
    n = xy.shape[0]
    with torch.no_grad():
        Vp, Jp = fk(prev, betas)
        Vg, Jg = fk(ge, betas)
        Vs, Js = fk(gs, betas)
        pr = prev.copy()
        pr[:6] = ge[:6]
        Vr, _ = fk(pr, betas)
        vp, vg, vs = (v[0].double().numpy() for v in (Vp, Vg, Vs))
        uv_p, z_p = project_np(vp, intr)
        uv_g, _ = project_np(vg, intr)
        uv_s, _ = project_np(vs, intr)
        cbar = uv_p.mean(0)
        # ---- oracle transforms
        To = fit_sim(uv_p, uv_g)
        Tt = fit_sim(uv_p, uv_g, dof="trans")
        # ---- events-only estimates (sampled events)
        ests = {}
        E = xy.astype(np.float64)
        contour = None
        tm = {"contour_ms": float("nan"), **{f"icp_{k}_ms": float("nan") for k in EST_SPEC}}      # CPU wall time of the estimator
        cnorm = None
        if n > 0:
            tc = time.perf_counter()
            contour, _ = silhouette_contour(uv_p, faces)
            tm["contour_ms"] = (time.perf_counter() - tc) * 1e3
            if len(contour) >= MIN_CONTOUR:
                cnorm = contour_normals(contour)
        for name, (dof, cs) in EST_SPEC.items():
            if n < MIN_EV_EST:
                ests[name] = ((1.0, 0.0, np.zeros(2)), -1, 0, 0)
            elif len(contour) < MIN_CONTOUR:
                ests[name] = ((1.0, 0.0, np.zeros(2)), -2, 0, 0)
            else:
                ti = time.perf_counter()
                if dof == "sim_p2l":
                    T, ni, nin = icp_p2l(contour, cnorm, E, cs)
                else:
                    T, ni, nin = icp(contour, E, dof, cs)
                tm[f"icp_{name}_ms"] = (time.perf_counter() - ti) * 1e3
                ests[name] = (T, 1 if ni > 0 else 0, ni, nin)
        # ---- event support of the contour (prev's and the window-end GT's silhouette) and of the parts
        cov_prev3 = cov_gt3 = float("nan")
        if n > 0:
            tree_e = cKDTree(E)
            if len(contour):
                cov_prev3 = float((tree_e.query(contour)[0] <= 3.0).mean())
            cg_cont, _ = silhouette_contour(uv_g, faces)
            if len(cg_cont):
                cov_gt3 = float((tree_e.query(cg_cont)[0] <= 3.0).mean())
        part_hist = np.bincount(part.astype(np.int64) + 1, minlength=17).astype(np.float64)
        # ---- synthetic recovery: known similarity perturbation of the window-end GT mesh, same events, same ICP
        syn = {name: np.full((len(SYN_ITEMS), 9), np.nan) for name in ("sim2d_est", "sim2d_est_p2l")}
        if n >= MIN_EV_EST:
            cg = uv_g.mean(0)
            rng = np.random.default_rng(1000003 * (task["j"] + 1))
            for i, (kind_, val) in enumerate(SYN_ITEMS):
                sp, thp, tp = 1.0, 0.0, np.zeros(2)
                if kind_ == "t_px":
                    a = rng.uniform(0.0, 2.0 * math.pi)
                    tp = val * np.array([math.cos(a), math.sin(a)])
                elif kind_ == "rot_deg":
                    thp = math.radians(val) * float(rng.choice([-1.0, 1.0]))
                else:
                    sp = val
                Tp = (sp, thp, cg - sp * (rot2(thp) @ cg) + tp)
                uv_pert = apply_sim(uv_g, Tp)
                cont_p, _ = silhouette_contour(uv_pert, faces)
                if len(cont_p) < MIN_CONTOUR:
                    continue
                for name in syn:
                    if name == "sim2d_est":
                        Te, _, _ = icp(cont_p, E, "sim", EST_SPEC["sim2d_est"][1])
                    else:
                        Te, _, _ = icp_p2l(cont_p, contour_normals(cont_p), E, EST_SPEC["sim2d_est_p2l"][1])
                    uv_tot = apply_sim(uv_pert, Te)
                    dc = uv_tot.mean(0) - cg
                    dth = wrap_deg((Te[1] + thp) * RAD2DEG)
                    ds = Te[0] * sp - 1.0
                    syn[name][i] = [np.sqrt(((uv_pert - uv_g) ** 2).sum(1).mean()), np.sqrt(((uv_tot - uv_g) ** 2).sum(1).mean()),
                                    np.linalg.norm(dc), abs(dth), abs(ds), dc[0], dc[1], dth, ds]
        # ---- association under every prev mesh
        verts = {"cl": Vp, "gt": Vs, "gt_end": Vg, "rigid3d_oracle": Vr,
                 "sim2d_oracle": _to_t(backproject_np(apply_sim(uv_p, To), z_p, intr)),
                 "trans2d_oracle": _to_t(backproject_np(apply_sim(uv_p, Tt), z_p, intr))}
        for name, (T, st, _, _) in ests.items():
            verts[name] = _to_t(backproject_np(apply_sim(uv_p, T), z_p, intr))
        cnt = {}
        for c in CONDS:
            if c in ests and ests[c][1] <= 0:
                cnt[c] = None                                   # identity transform: reuse cl below
            else:
                cnt[c] = assoc(verts[c], xy, part, kind, intr_t)
        for c in CONDS:
            if cnt[c] is None:
                cnt[c] = cnt["cl"].copy()
    # ---- offsets of the closed-loop prev relative to the ground truth
    Rp = Rotation.from_rotvec(prev[3:6].astype(np.float64))
    Rg = Rotation.from_rotvec(ge[3:6].astype(np.float64))
    Rs = Rotation.from_rotvec(gs[3:6].astype(np.float64))
    geo = {
        "c_off": float(np.linalg.norm(uv_g.mean(0) - cbar)),                 # vs window-end GT (what the oracle uses)
        "c_off0": float(np.linalg.norm(uv_s.mean(0) - cbar)),                # vs window-start GT (prev's own target)
        "c_off_lag": float(np.linalg.norm(uv_g.mean(0) - uv_s.mean(0))),     # GT motion over the window
        "orc_th_deg": To[1] * RAD2DEG, "orc_s": To[0],
        "rms_before": float(np.sqrt(((uv_p - uv_g) ** 2).sum(1).mean())),
        "rms_orc": float(np.sqrt(((apply_sim(uv_p, To) - uv_g) ** 2).sum(1).mean())),
        "rms_orc_t": float(np.sqrt(((apply_sim(uv_p, Tt) - uv_g) ** 2).sum(1).mean())),
        "err_depth_mm": float(abs(Jp[0, 0, 2] - Jg[0, 0, 2]) * 1000.0),
        "err_t_mm": float(np.linalg.norm(prev[0:3].astype(np.float64) - ge[0:3]) * 1000.0),
        "err_rot_deg": float((Rp.inv() * Rg).magnitude() * RAD2DEG),
        "err_fing_rad": float(np.linalg.norm(prev[6:51].astype(np.float64) - ge[6:51])),
        "err0_depth_mm": float(abs(Jp[0, 0, 2] - Js[0, 0, 2]) * 1000.0),
        "err0_t_mm": float(np.linalg.norm(prev[0:3].astype(np.float64) - gs[0:3]) * 1000.0),
        "err0_rot_deg": float((Rp.inv() * Rs).magnitude() * RAD2DEG),
        "err0_fing_rad": float(np.linalg.norm(prev[6:51].astype(np.float64) - gs[6:51])),
        "lag_rot_deg": float((Rs.inv() * Rg).magnitude() * RAD2DEG),
        "lag_fing_rad": float(np.linalg.norm(gs[6:51].astype(np.float64) - ge[6:51])),
        "lag_t_mm": float(np.linalg.norm(gs[0:3].astype(np.float64) - ge[0:3]) * 1000.0),
        "n_contour": 0 if contour is None else int(len(contour)),
        "cov_prev3": cov_prev3, "cov_gt3": cov_gt3,
    }
    est = {name: _est_record(T, To, cbar, uv_p, uv_g, st, ni, nin) for name, (T, st, ni, nin) in ests.items()}
    est["trans2d_oracle"] = _est_record(Tt, To, cbar, uv_p, uv_g, 2, 0, 0)
    return {"j": task["j"], "cnt": np.stack([cnt[c] for c in CONDS]), "geo": geo, "est": est, "part_hist": part_hist, "tm": tm,
            "syn": syn["sim2d_est"], "syn_p2l": syn["sim2d_est_p2l"], "oracle": {"s": To[0], "th_deg": To[1] * RAD2DEG, "t": To[2].tolist()}, "sec": time.time() - t0}


# --------------------------------------------------------------------------- setup shared by stages
def load_cfg_seqs(args):
    from config import load_config
    from semkine.dataset import sequences_for_split, splits_manifest
    run = REPO / args.run_dir
    cfg_path = args.config or str(next(iter(sorted(run.glob("*.yaml"))), "")) or \
        json.loads((run / "training_metadata.json").read_text())["config_path"]
    cfg = load_config(cfg_path)
    root = Path(cfg["DATA"]["ROOT"])
    seqs = sequences_for_split(root, "val_core", splits_manifest(cfg))
    return cfg, cfg_path, root, seqs


# --------------------------------------------------------------------------- stage A
class AlignHook(PX.ClosedLoopHook):
    """P2-a's hook (labels, association counts from `xyz_last`) that also records the model's prev and pred."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.prev_rec, self.pred_rec = [], []

    def __call__(self, kind, prev_t, pred):
        if kind != "step":
            return pred
        self.prev_rec.append(prev_t.detach().cpu().numpy()[0].copy())
        out = super().__call__(kind, prev_t, pred)
        self.pred_rec.append(pred.detach().cpu().numpy()[0].copy())
        return out


def run_stage_a(args, out_dir):
    with PX.limited_runs(args.max_steps):
        cfg, cfg_path, device, mano, root, seqs, model, ckpt = PX.load_everything(args)
        log(f"device={device} ckpt={ckpt} cfg={cfg_path} seqs={[s for s, _ in seqs]}")
        cl = PX.build_ctx_labels(root, seqs, REPO / args.labels_dir)
        lbs_hard = model.xyz.lbs_hard.to(device)
        model.xyz_keep_aux = True
        rng = np.random.default_rng(0)                       # shared across the sequences, as select_checkpoint
        res, hooks, t0 = {}, {}, time.time()
        for seq, d in seqs:
            ctx, lab = cl[seq]
            hook = AlignHook(model, ctx, lab, lbs_hard)
            ts = time.time()
            res[seq] = ET.track_sequence(model, mano, cfg, root, d, seq, ST, device, rng, 1.0, 1.0, None, state_hook=hook)
            assert hook.i == len(ctx.steps) == res[seq]["n_frames"], (hook.i, len(ctx.steps), res[seq]["n_frames"])
            hooks[seq] = hook
            log(f"A {seq}: {hook.i} steps RA={res[seq]['mpjpe_ra_mm']:.4f} ({time.time() - ts:.0f}s)")
        model.xyz_keep_aux = False
    n_tot = sum(r["n_frames"] for r in res.values())
    ra = {"overall": sum(r["mpjpe_ra_mm"] * r["n_frames"] for r in res.values()) / max(n_tot, 1),
          "per_seq": {s: r["mpjpe_ra_mm"] for s, r in res.items()}, "n_frames": {s: r["n_frames"] for s, r in res.items()}}
    full = not (args.max_steps or args.random_init)
    ref = PX.find_ra_reference(REPO / args.run_dir) if full else None
    ra_check = PX.compute_ra_check(ra, ref) if ref else {"reference": None}
    p2a_path = REPO / "outputs/xa/step2/p2a_closedloop.json"
    p2a = json.loads(p2a_path.read_text()) if (p2a_path.exists() and full) else None
    if p2a:
        ra_check["p2a_overall"] = p2a["ra"]["overall"]
        ra_check["abs_diff_vs_p2a"] = abs(ra["overall"] - p2a["ra"]["overall"])
    log("RA", ra, "check:", {k: v for k, v in ra_check.items() if k != "reference"})
    step_check = {}
    for seq, hook in hooks.items():
        ctx, lab = cl[seq]
        S = len(ctx.steps)
        cols = {k: np.asarray(v) for k, v in hook.cols.items()}
        np.savez(out_dir / f"closedloop_prev_{seq}.npz",
                 prev=np.stack(hook.prev_rec).astype(np.float32), pred=np.stack(hook.pred_rec).astype(np.float32),
                 gt_end=np.stack([ctx.pos51[s.end] for s in ctx.steps]).astype(np.float32),
                 gt_start=np.stack([ctx.pos51[s.prev_idx] for s in ctx.steps]).astype(np.float32),
                 step_j=np.arange(S), end_ms=np.array([s.end for s in ctx.steps]), prev_ms=np.array([s.prev_idx for s in ctx.steps]),
                 run_id=np.array([s.run_id for s in ctx.steps]), n_ev=np.array([s.n_ev for s in ctx.steps]),
                 fit=np.array([bool(lab.fit[s.j]) for s in ctx.steps]),
                 run_first=np.array([s.prev_idx == int(np.asarray(ctx.aux["valid_runs_ms"]).reshape(-1, 2)[s.run_id, 0]) for s in ctx.steps]),
                 betas=np.asarray(ctx.aux["betas"], np.float32), camera_K=np.asarray(ctx.aux["camera_K"], np.float32),
                 ra_mpjpe_mm=res[seq]["mpjpe_ra_mm"],
                 **{"ref_" + k: cols[k] for k in KEYS})
        log("wrote", out_dir / f"closedloop_prev_{seq}.npz")
        if p2a and full:                                      # per-step counts must equal P2-a's, exactly
            pc = p2a["sequences"][seq]["columns"]
            step_check[seq] = {k: int(np.sum(np.asarray(pc[k], np.float64) != cols[k].astype(np.float64))) for k in KEYS}
    meta = PX.meta(args, cfg_path, ckpt, device, {"script": "tools/xa/probe_align.py", "seconds": time.time() - t0, "stage": "a",
                                                  "protocol": "rng(0) shared across sequences, 50 ms, val_core, noise 1.0"})
    PX.dump(out_dir / "stage_a.json", {"meta": meta, "ra": ra, "ra_check": ra_check,
                                       "p2a_step_count_mismatch": step_check})
    if p2a and full:
        bad = {s: {k: v for k, v in d.items() if v} for s, d in step_check.items()}
        log("per-step count mismatches vs P2-a:", bad)


# --------------------------------------------------------------------------- stage B
def make_cfgd(cfg):
    m = cfg["MODEL"]
    cfgd = {"repo": str(REPO), "mano_npz": cfg["MANO"]["NPZ"], "render_scale": float(m.get("RENDER_SCALE", 240.0 / 640.0)),
            "band_px": float(m.get("XYZ_BAND_PX", 16.0)), "front_px": float(m.get("XYZ_FRONT_PX", 3.0)),
            "z_tol": float(m.get("XYZ_Z_TOL", 0.01)), "k": int(m.get("XYZ_K", 16))}
    assert str(m.get("XYZ_ASSOC_MODE", "hit_first")) == "hit_first" and str(m.get("XYZ_ASSOC_PREV", "self")) == "self"
    return cfgd


def build_tasks(ctx, lab, z, steps):
    """Per-step worker inputs: closed-loop prev, window-start / window-end GT, the model's sampled events (pixel
    coordinates) with their P1 source labels (alignment with the label file is asserted by `Labels.step`)."""
    tasks = []
    for st in steps:
        assert int(z["end_ms"][st.j]) == st.end and int(z["prev_ms"][st.j]) == st.prev_idx
        ev5 = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, st.end, ST)
        part_s, kind_s, fit, idx = lab.step(st, ev5)
        assert bool(z["fit"][st.j]) == fit
        xy = ev5[idx][:, 1:3].astype(np.float32) if st.n_ev else np.zeros((0, 2), np.float32)
        tasks.append({"j": st.j, "prev": z["prev"][st.j], "gt_end": z["gt_end"][st.j], "gt_start": z["gt_start"][st.j],
                      "betas": z["betas"], "K": z["camera_K"], "xy": xy, "part": part_s, "kind": kind_s})
    return tasks


def run_timing(args, out_dir):
    """Sequential single-process, single-thread CPU timing of the events-only estimator (silhouette rasterisation +
    ICP) on evenly spaced packets, free of the 64-way contention of stage B."""
    cfg, cfg_path, root, seqs = load_cfg_seqs(args)
    cfgd = make_cfgd(cfg)
    cl = PX.build_ctx_labels(root, seqs, REPO / args.labels_dir)
    _init_worker(cfgd)
    res = {}
    for seq, d in seqs:
        z = np.load(out_dir / f"closedloop_prev_{seq}.npz")
        ctx, lab = cl[seq]
        pick = np.unique(np.linspace(0, len(z["prev"]) - 1, args.timing_n).astype(int))
        tasks = build_tasks(ctx, lab, z, [ctx.steps[j] for j in pick])
        process_step(tasks[0])                                       # warm-up
        rows = [process_step(t) for t in tasks]
        ctm = np.array([r["tm"]["contour_ms"] for r in rows])
        itm = np.array([r["tm"]["icp_sim2d_est_ms"] for r in rows])
        ptm = np.array([r["tm"]["icp_sim2d_est_p2l_ms"] for r in rows])
        ok = np.isfinite(itm)
        q = lambda x: {"mean": float(np.mean(x)), "p50": float(np.median(x)), "p90": float(np.quantile(x, 0.9)), "max": float(np.max(x))}   # noqa: E731
        res[seq] = {"n_timed": int(ok.sum()), "n_packets_sampled": int(len(rows)), "contour_ms": q(ctm[ok]), "icp_sim2d_est_ms": q(itm[ok]),
                    "icp_sim2d_est_p2l_ms": q(ptm[ok]), "n_iter_mean_p2l": float(np.mean([r["est"]["sim2d_est_p2l"]["n_iter"] for r, o in zip(rows, ok) if o])),
                    "estimator_total_ms": q(ctm[ok] + itm[ok]), "n_iter_mean": float(np.mean([r["est"]["sim2d_est"]["n_iter"] for r, o in zip(rows, ok) if o])),
                    "n_events_sampled_mean": float(np.mean([len(t["xy"]) for t, o in zip(tasks, ok) if o]))}
        log(f"timing {seq}:", res[seq])
    import platform
    PX.dump(out_dir / "timing.json", {"note": "single process, torch/OMP threads = 1, evenly spaced packets, sim2d_est only (contour rasterisation + ICP, 10 iterations max)",
                                       "host": platform.node(), "cpu_threads_used": 1, "date": time.strftime("%F %T"), "sequences": res})


def run_stage_b(args, out_dir):
    cfg, cfg_path, root, seqs = load_cfg_seqs(args)
    cfgd = make_cfgd(cfg)
    log("assoc params", cfgd)
    if args.seqs:
        seqs = [(s, d) for s, d in seqs if s in args.seqs.split(",")]
    cl = PX.build_ctx_labels(root, seqs, REPO / args.labels_dir)
    p2b_path = REPO / "outputs/xa/step2/p2b_tf_corr.json"
    p2b = json.loads(p2b_path.read_text()) if p2b_path.exists() else None
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"                                       # workers inherit: one thread each
    ctx_pool = mp.get_context("spawn")
    t0 = time.time()
    with ctx_pool.Pool(args.workers, initializer=_init_worker, initargs=(cfgd,)) as pool:
        log(f"pool of {args.workers} workers up ({time.time() - t0:.0f}s)")
        for seq, d in seqs:
            z = np.load(out_dir / f"closedloop_prev_{seq}.npz")
            ctx, lab = cl[seq]
            S = len(z["prev"])
            if args.limit:
                S = min(S, args.limit)
            ts = time.time()
            tasks = build_tasks(ctx, lab, z, ctx.steps[:S])
            log(f"B {seq}: {len(tasks)} tasks built ({time.time() - ts:.0f}s)")
            out = list(pool.imap(process_step, tasks, chunksize=2))
            assert [o["j"] for o in out] == list(range(S))
            log(f"B {seq}: done ({time.time() - ts:.0f}s, worker-sec {sum(o['sec'] for o in out):.0f})")
            cnt = np.stack([o["cnt"] for o in out])                    # (S, n_cond, n_keys)
            cols = {"cnt": cnt, "cond_names": np.array(CONDS), "keys": np.array(KEYS), "fit": z["fit"][:S],
                    "run_id": z["run_id"][:S], "end_ms": z["end_ms"][:S], "n_ev": z["n_ev"][:S], "run_first": z["run_first"][:S]}
            for k in out[0]["geo"]:
                cols["geo_" + k] = np.array([o["geo"][k] for o in out], np.float64)
            for e in out[0]["est"]:
                for k in out[0]["est"][e]:
                    cols[f"est_{e}_{k}"] = np.array([o["est"][e][k] for o in out], np.float64)
            for k in out[0]["tm"]:
                cols["tm_" + k] = np.array([o["tm"][k] for o in out], np.float64)
            cols["part_hist"] = np.stack([o["part_hist"] for o in out])
            cols["syn"] = np.stack([o["syn"] for o in out])
            cols["syn_p2l"] = np.stack([o["syn_p2l"] for o in out])
            cols["syn_items"] = np.array([f"{k}:{v}" for k, v in SYN_ITEMS])
            cols["orc_tx"] = np.array([o["oracle"]["t"][0] for o in out])
            cols["orc_ty"] = np.array([o["oracle"]["t"][1] for o in out])
            # ---- reproduction checks: cl vs the GPU closed-loop counts of stage A, gt vs P2-b teacher-forced counts
            chk = {"n_steps": S}
            ci, gi = CONDS.index("cl"), CONDS.index("gt")
            chk["cl_vs_stage_a_gpu_mismatch"] = {k: int(np.sum(cnt[:, ci, i] != z["ref_" + k][:S])) for i, k in enumerate(KEYS)}
            if p2b and seq in p2b["sequences"]:
                pc = p2b["sequences"][seq]["columns"]
                chk["gt_vs_p2b_tf_mismatch"] = {k: int(np.sum(cnt[:, gi, i] != np.asarray(pc[k], np.float64)[:S])) for i, k in enumerate(KEYS)}
            log(f"B {seq}: check", {k: ({a: b for a, b in v.items() if b} if isinstance(v, dict) else v) for k, v in chk.items()})
            cols["checks_json"] = np.array(json.dumps(chk))
            np.savez(out_dir / f"per_step_{seq}.npz", **cols)
            log("wrote", out_dir / f"per_step_{seq}.npz")


# --------------------------------------------------------------------------- report
SEQS = ("zgz_global", "zgz_local")
STRATA = (("全部", None), ("fingertip_in_front=真", True), ("fingertip_in_front=假", False))
CL = {"cl": "cl", "sim2d_oracle": "cl+sim2d_oracle", "rigid3d_oracle": "cl+rigid3d_oracle", "sim2d_est": "cl+sim2d_est",
      "gt": "gt（窗首真值）", "trans2d_oracle": "cl+trans2d_oracle（仅平移）", "trans2d_est": "cl+trans2d_est（仅平移 ICP）",
      "sim2d_est_cf": "cl+sim2d_est_cf（ICP 尺度 64→16 px）", "sim2d_est_it30": "cl+sim2d_est_it30（ICP 30 次迭代）",
      "sim2d_est_tight": "cl+sim2d_est_tight（ICP 16 px×10 后 4 px×10）", "sim2d_est_p2l": "cl+sim2d_est_p2l（点到线 Gauss-Newton ICP）",
      "gt_end": "gt_end（窗末真值网格，参考）"}
BOOT_B, BOOT_SEED = 1000, 0


def _pct(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


def _num(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _qs(x, qs=(0.5, 0.9)):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, q)) for q in qs] if len(x) else [float("nan")] * len(qs)


def _sel(Zs, key):
    fit = Zs["fit"].astype(bool)
    return np.ones(len(fit), bool) if key is None else (fit if key else ~fit)


def _boot(cnt, sel, B=BOOT_B, seed=BOOT_SEED):
    """Moving-block bootstrap (block 20 steps, P2's `block_idx`) of the pooled (agree, M, hit) of every condition."""
    idx = np.nonzero(sel)[0]
    X = cnt[idx]
    kk = [KEYS.index(k) for k in ("n_agree", "n_hand", "n_mis", "n_cf", "n_hit", "n_live")]
    rng = np.random.default_rng(seed)
    out = np.zeros((B, cnt.shape[1], 3))
    for b in range(B):
        s = X[PX.block_idx(len(idx), rng)].sum(0)
        out[b, :, 0] = s[:, kk[0]] / np.maximum(s[:, kk[1]], 1)
        out[b, :, 1] = s[:, kk[2]] / np.maximum(s[:, kk[3]], 1)
        out[b, :, 2] = s[:, kk[4]] / np.maximum(s[:, kk[5]], 1)
    return out


def _ci(v):
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def _pooled(cnt, sel):
    return {c: PX.rates({k: float(cnt[sel, ci, i].sum()) for i, k in enumerate(KEYS)}) for ci, c in enumerate(CONDS)}


def compute_all(Z):
    R = {"main": {}, "bins": {}, "bins0": {}, "offsets": {}, "est": {}, "est_by_bin": {}, "syn": {}, "support": {}, "steplevel": {},
         "corr": {}, "evbins": {}, "timing": {}}
    gi = CONDS.index("gt")
    for seq in SEQS:
        Zs = Z[seq]
        cnt = Zs["cnt"]
        R["main"][seq] = {}
        for sname, skey in STRATA:
            sel = _sel(Zs, skey)
            if not sel.any():
                continue
            bt = _boot(cnt, sel)
            pr = _pooled(cnt, sel)
            res = {}
            for ci, c in enumerate(CONDS):
                r = pr[c]
                sums = {k: float(cnt[sel, ci, i].sum()) for i, k in enumerate(KEYS)}
                d = {"n_steps": int(sel.sum()), "n_events": sums["n_live"], "n_hand_events": sums["n_hand"],
                     "agree": r["agree_exact_allhand"], "agree_ci": _ci(bt[:, ci, 0]),
                     "agree_has": r["agree_exact"], "M": r["M_all"], "M_ci": _ci(bt[:, ci, 1]), "M_has": r["M_has"],
                     "hit": r["hit_frac"], "hit_ci": _ci(bt[:, ci, 2]), "valid": r["valid_frac"], "counts": sums}
                if c != "cl":
                    dd = bt[:, ci, 0] - bt[:, 0, 0]
                    d["delta_agree_pp"] = 100 * (r["agree_exact_allhand"] - pr["cl"]["agree_exact_allhand"])
                    d["delta_agree_pp_ci"] = [100 * x for x in _ci(dd)]
                    d["delta_M_pp"] = 100 * (r["M_all"] - pr["cl"]["M_all"])
                    d["delta_hit_pp"] = 100 * (r["hit_frac"] - pr["cl"]["hit_frac"])
                    gap = pr["gt"]["agree_exact_allhand"] - pr["cl"]["agree_exact_allhand"]
                    d["gap_recovery"] = (r["agree_exact_allhand"] - pr["cl"]["agree_exact_allhand"]) / gap
                    den = bt[:, gi, 0] - bt[:, 0, 0]
                    ok = den > 1e-3
                    d["gap_recovery_ci"] = _ci(dd[ok] / den[ok]) if ok.sum() > 50 else [float("nan")] * 2
                res[c] = d
            R["main"][seq][sname] = res
        # ---- offset bins (per stratum)
        for key, tgt, offk in (("bins", "bins", "geo_c_off"), ("bins0", "bins0", "geo_c_off0")):
            co = Zs[offk]
            R[tgt][seq] = {}
            for sname, skey in STRATA:
                ssel = _sel(Zs, skey)
                if not ssel.any():
                    continue
                rows = []
                for lo, hi in OFFSET_BINS[:3] + ((8.0, float("inf")),) + OFFSET_BINS_FINE:
                    sel = ssel & (co >= lo) & (co < hi)
                    if not sel.any():
                        continue
                    pr = _pooled(cnt, sel)
                    rows.append({"lo": lo, "hi": hi, "fine": (lo, hi) in OFFSET_BINS_FINE, "n_steps": int(sel.sum()),
                                 "n_events": float(cnt[sel, 0, KEYS.index("n_live")].sum()),
                                 "rates": {c: {"agree": pr[c]["agree_exact_allhand"], "M": pr[c]["M_all"], "hit": pr[c]["hit_frac"]} for c in CONDS}})
                R[tgt][seq][sname] = rows
        # ---- sim2d_est by number of events per window
        ran = Zs["est_sim2d_est_status"] > 0
        n_raw = Zs["n_ev"].astype(np.float64)
        inf_ = float("inf")
        schemes = (("raw", "窗内原始事件数", ((0, 1000), (1000, 3000), (3000, inf_)) if seq == "zgz_local" else ((0, 1000), (1000, 10000), (10000, inf_)), n_raw),
                   ("sampled", "模型看到的抽样事件数（上限 2048）", ((0, 100), (100, 300), (300, 1000), (1000, 2048), (2048, inf_)), np.minimum(n_raw, 2048.0)))
        R["evbins"][seq] = []
        for key, desc, edges, x in schemes:
            for lo, hi in edges:
                sel = (x >= lo) & (x < hi)
                if not sel.any():
                    continue
                selr = sel & ran
                pr = _pooled(cnt, sel)
                R["evbins"][seq].append({
                    "scheme": key, "desc": desc, "lo": lo, "hi": hi, "n_steps": int(sel.sum()), "n_ran": int(selr.sum()),
                    "id_c": _qs(Zs["geo_c_off"][selr]), "est_c": _qs(Zs["est_sim2d_est_err_c_px"][selr]),
                    "id_th": _qs(np.abs(Zs["geo_orc_th_deg"][selr])), "est_th": _qs(np.abs(Zs["est_sim2d_est_err_th_deg"][selr])),
                    "id_s": _qs(np.abs(Zs["geo_orc_s"][selr] - 1)), "est_s": _qs(np.abs(Zs["est_sim2d_est_s_ratio"][selr] - 1)),
                    "frac_rms_improved": float((Zs["est_sim2d_est_rms_px"][selr] < Zs["geo_rms_before"][selr]).mean()) if selr.any() else float("nan"),
                    "agree": {c: pr[c]["agree_exact_allhand"] for c in ("cl", "sim2d_est", "sim2d_oracle", "gt")},
                    "M": {c: pr[c]["M_all"] for c in ("cl", "sim2d_est", "sim2d_oracle", "gt")},
                    "hit": {c: pr[c]["hit_frac"] for c in ("cl", "sim2d_est", "sim2d_oracle", "gt")},
                    "delta_est_pp": 100 * (pr["sim2d_est"]["agree_exact_allhand"] - pr["cl"]["agree_exact_allhand"])})
        # ---- ICP iterations and CPU time (stage B, 64 processes in parallel)
        R["timing"][seq] = {"n_ran": int(ran.sum()),
                            "n_iter_mean": {e: float(Zs[f"est_{e}_n_iter"][Zs[f"est_{e}_status"] > 0].mean()) for e in EST_SPEC},
                            "frac_full_cap": {e: float((Zs[f"est_{e}_n_iter"][Zs[f"est_{e}_status"] > 0] >= len(EST_SPEC[e][1])).mean()) for e in EST_SPEC},
                            "contour_ms": {"mean": float(np.nanmean(Zs["tm_contour_ms"][ran])), "p50": _qs(Zs["tm_contour_ms"][ran], (0.5,))[0], "p90": _qs(Zs["tm_contour_ms"][ran], (0.9,))[0]},
                            "icp_ms": {e: {"mean": float(np.nanmean(Zs[f"tm_icp_{e}_ms"][ran])), "p50": _qs(Zs[f"tm_icp_{e}_ms"][ran], (0.5,))[0],
                                           "p90": _qs(Zs[f"tm_icp_{e}_ms"][ran], (0.9,))[0]} for e in EST_SPEC}}
        # ---- offset descriptives and correlations
        R["offsets"][seq] = {}
        for k in ("geo_c_off", "geo_c_off0", "geo_c_off_lag", "geo_err_t_mm", "geo_err_depth_mm", "geo_err_rot_deg", "geo_err_fing_rad",
                  "geo_err0_t_mm", "geo_err0_depth_mm", "geo_err0_rot_deg", "geo_err0_fing_rad", "geo_lag_t_mm", "geo_lag_rot_deg",
                  "geo_lag_fing_rad", "geo_orc_th_deg", "geo_orc_s", "geo_rms_before", "geo_rms_orc", "geo_rms_orc_t"):
            R["offsets"][seq][k] = dict(zip(("p10", "p50", "p90"), _qs(Zs[k], (0.1, 0.5, 0.9))))
        R["offsets"][seq]["abs_orc_th_deg"] = dict(zip(("p10", "p50", "p90"), _qs(np.abs(Zs["geo_orc_th_deg"]), (0.1, 0.5, 0.9))))
        R["offsets"][seq]["abs_orc_s_rel"] = dict(zip(("p10", "p50", "p90"), _qs(np.abs(Zs["geo_orc_s"] - 1), (0.1, 0.5, 0.9))))
        ci0 = CONDS.index("cl")
        nh = cnt[:, ci0, KEYS.index("n_hand")]
        ag = cnt[:, ci0, KEYS.index("n_agree")] / np.maximum(nh, 1)
        okc = nh >= 50
        R["corr"][seq] = {"n_steps": int(okc.sum())}
        for k in ("geo_c_off", "geo_c_off0", "geo_rms_before", "geo_err_t_mm", "geo_err_depth_mm", "geo_err_rot_deg", "geo_err_fing_rad",
                  "geo_orc_th_deg", "geo_orc_s"):
            x = np.abs(Zs[k]) if k == "geo_orc_th_deg" else (np.abs(Zs[k] - 1) if k == "geo_orc_s" else Zs[k])
            R["corr"][seq][k] = PX.spearman(x[okc], ag[okc])
        # ---- estimator error
        R["est"][seq] = {}
        for e in list(EST_SPEC) + ["trans2d_oracle"]:
            st = Zs[f"est_{e}_status"]
            ok = st > 0 if e != "trans2d_oracle" else np.ones(len(st), bool)
            if e != "trans2d_oracle":
                ok_ref = Zs["est_sim2d_est_status"] > 0
            else:
                ok_ref = Zs["est_sim2d_est_status"] > 0
                ok = ok_ref
            d = {"n_total": int(len(st)), "n_ran": int(ok.sum()), "n_lt100_events": int((Zs["est_sim2d_est_status"] == -1).sum()),
                 "n_no_contour": int((Zs["est_sim2d_est_status"] == -2).sum()), "n_iter_mean": float(Zs[f"est_{e}_n_iter"][ok].mean()) if ok.any() else float("nan")}
            a = {"err_c_px": Zs[f"est_{e}_err_c_px"][ok], "err_th_deg": np.abs(Zs[f"est_{e}_err_th_deg"][ok]),
                 "err_s_rel": np.abs(Zs[f"est_{e}_s_ratio"][ok] - 1), "rms_px": Zs[f"est_{e}_rms_px"][ok]}
            for k, v in a.items():
                d[k] = dict(zip(("p50", "p90"), _qs(v)))
            d["frac_rms_improved"] = float((a["rms_px"] < Zs["geo_rms_before"][ok]).mean()) if ok.any() else float("nan")
            R["est"][seq][e] = d
        ok = Zs["est_sim2d_est_status"] > 0
        R["est"][seq]["identity"] = {"n_ran": int(ok.sum()), "err_c_px": dict(zip(("p50", "p90"), _qs(Zs["geo_c_off"][ok]))),
                                     "err_th_deg": dict(zip(("p50", "p90"), _qs(np.abs(Zs["geo_orc_th_deg"][ok])))),
                                     "err_s_rel": dict(zip(("p50", "p90"), _qs(np.abs(Zs["geo_orc_s"][ok] - 1)))),
                                     "rms_px": dict(zip(("p50", "p90"), _qs(Zs["geo_rms_before"][ok])))}
        R["est"][seq]["oracle_floor"] = {"rms_px": dict(zip(("p50", "p90"), _qs(Zs["geo_rms_orc"][ok]))),
                                         "rms_px_trans_only": dict(zip(("p50", "p90"), _qs(Zs["geo_rms_orc_t"][ok])))}
        # ---- estimator error by oracle displacement bin (sim2d_est)
        rows = []
        for lo, hi in OFFSET_BINS[:3] + ((8.0, 16.0), (16.0, float("inf"))):
            sel = ok & (Zs["geo_c_off"] >= lo) & (Zs["geo_c_off"] < hi)
            if not sel.any():
                continue
            rows.append({"lo": lo, "hi": hi, "n_steps": int(sel.sum()),
                         "id_c": _qs(Zs["geo_c_off"][sel])[0], "est_c": _qs(Zs["est_sim2d_est_err_c_px"][sel], (0.5, 0.9)),
                         "id_th": _qs(np.abs(Zs["geo_orc_th_deg"][sel]))[0], "est_th": _qs(np.abs(Zs["est_sim2d_est_err_th_deg"][sel]), (0.5, 0.9)),
                         "id_s": _qs(np.abs(Zs["geo_orc_s"][sel] - 1))[0], "est_s": _qs(np.abs(Zs["est_sim2d_est_s_ratio"][sel] - 1), (0.5, 0.9)),
                         "frac_rms_improved": float((Zs["est_sim2d_est_rms_px"][sel] < Zs["geo_rms_before"][sel]).mean())})
        R["est_by_bin"][seq] = rows
        # ---- synthetic recovery
        syn, items = Zs["syn"], [str(x) for x in Zs["syn_items"]]
        R["syn_p2l"] = R.get("syn_p2l", {})
        R["syn_p2l"][seq] = []
        for i, it in enumerate(items):
            v = Zs["syn_p2l"][:, i, :]
            okv = np.isfinite(v[:, 0])
            R["syn_p2l"][seq].append({"item": it, "n": int(okv.sum()), "rms_id": _qs(v[okv, 0]), "rms_after": _qs(v[okv, 1]),
                                      "err_c": _qs(v[okv, 2]), "err_th": _qs(v[okv, 3]), "err_s": _qs(v[okv, 4]),
                                      "bias": {"dx_px": _qs(v[okv, 5])[0], "dy_px": _qs(v[okv, 6])[0], "dth_deg": _qs(v[okv, 7])[0], "ds": _qs(v[okv, 8])[0]}})
        R["syn"][seq] = []
        for i, it in enumerate(items):
            v = syn[:, i, :]
            okv = np.isfinite(v[:, 0])
            R["syn"][seq].append({"item": it, "n": int(okv.sum()), "rms_id": _qs(v[okv, 0]), "rms_after": _qs(v[okv, 1]),
                                  "err_c": _qs(v[okv, 2]), "err_th": _qs(v[okv, 3]), "err_s": _qs(v[okv, 4]),
                                  "bias": {"dx_px": _qs(v[okv, 5])[0], "dy_px": _qs(v[okv, 6])[0], "dth_deg": _qs(v[okv, 7])[0],
                                           "ds": _qs(v[okv, 8])[0]}})
        # ---- event support
        ph = Zs["part_hist"].sum(0)
        hand = ph[1:].sum()
        R["support"][seq] = {"events": float(ph.sum()), "share_nonhand": float(ph[0] / ph.sum()), "share_palm_of_hand": float(ph[1] / hand),
                             "share_by_group_of_hand": {g: float(ph[1:][np.array(PX.GROUP) == gi_].sum() / hand) for gi_, g in
                                                        enumerate(["掌", "食指", "中指", "小指", "无名指", "拇指"])},
                             "cov_gt3": dict(zip(("p10", "p50", "p90"), _qs(Zs["geo_cov_gt3"], (0.1, 0.5, 0.9)))),
                             "cov_prev3": dict(zip(("p10", "p50", "p90"), _qs(Zs["geo_cov_prev3"], (0.1, 0.5, 0.9)))),
                             "n_ev_window": dict(zip(("p10", "p50", "p90"), _qs(Zs["n_ev"], (0.1, 0.5, 0.9))))}
        # ---- step level: change of exact agreement relative to cl
        okn = nh > 0
        R["steplevel"][seq] = {}
        for ci, c in enumerate(CONDS):
            if c == "cl":
                continue
            dd = (cnt[:, ci, KEYS.index("n_agree")] / np.maximum(cnt[:, ci, KEYS.index("n_hand")], 1) - ag)[okn]
            R["steplevel"][seq][c] = {"n": int(okn.sum()), "mean_pp": float(100 * dd.mean()), "frac_up2": float((dd > 0.02).mean()),
                                      "frac_down2": float((dd < -0.02).mean())}
    return R


def _row_main(seq, sname, c, d):
    ci = d["agree_ci"]
    mci = d["M_ci"]
    delta = "—" if c == "cl" else f"{d['delta_agree_pp']:+.2f} [{d['delta_agree_pp_ci'][0]:+.2f}, {d['delta_agree_pp_ci'][1]:+.2f}]"
    rec = "—" if c in ("cl", "gt") else f"{100 * d['gap_recovery']:.1f}% [{100 * d['gap_recovery_ci'][0]:.0f}, {100 * d['gap_recovery_ci'][1]:.0f}]"
    return (f"| {seq} | {sname} | {d['n_steps']} | {CL[c]} | {_pct(d['agree'])} [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}] | "
            f"{_pct(d['M'])} [{100 * mci[0]:.1f}, {100 * mci[1]:.1f}] | {_pct(d['hit'])} | {delta} | {rec} |")


def make_report(args, out_dir):
    Z = {s: {k: v for k, v in np.load(out_dir / f"per_step_{s}.npz").items()} for s in SEQS}
    A = json.loads((out_dir / "stage_a.json").read_text())
    R = compute_all(Z)
    checks = {s: json.loads(str(Z[s]["checks_json"])) for s in SEQS}
    TM = json.loads((out_dir / "timing.json").read_text()) if (out_dir / "timing.json").exists() else None
    P = REPO / "outputs/xa/step2"
    p2a = json.loads((P / "p2a_closedloop.json").read_text()) if (P / "p2a_closedloop.json").exists() else None
    out = {"meta": {**A["meta"], "stage": "report", "script": "tools/xa/probe_align.py", "date_report": time.strftime("%F %T"),
                    "boot": {"B": BOOT_B, "block": PX.BLOCK, "seed": BOOT_SEED}, "conditions": list(CONDS),
                    "icp": {"weights": "Tukey biweight, c = 16 px (cf: 64,48,32,24,16..)", "iterations": 10, "min_events": MIN_EV_EST,
                            "min_contour_px": MIN_CONTOUR, "init": "identity", "variants": {k: {"dof": v[0], "scales": v[1]} for k, v in EST_SPEC.items()}}},
           "reproduction": {"ra": A["ra"], "ra_check": A["ra_check"], "stage_a_vs_p2a_step_count_mismatch": A["p2a_step_count_mismatch"],
                            "stage_b_checks": checks},
           "results": R, "timing_sequential": TM}
    PX.dump(out_dir / "align_probe.json", out)
    L = []
    w = L.append
    nS = {s: len(Z[s]["fit"]) for s in SEQS}
    w("# [XYZ-ASSOC] 整手 2D 相似对齐可行性探针（xa_c1_s3407 step 6000，零训练）\n")
    w("- 脚本：`tools/xa/probe_align.py`；产物：`outputs/xa/align/`（`closedloop_prev_<seq>.npz` 闭环前态，`per_step_<seq>.npz` 逐步计数与偏移量，`align_probe.json`，本文件）。只读检查点与仓库，未改任何已有文件。")
    w(f"- 检查点：`{A['meta']['ckpt']}`；配置：`{A['meta']['config']}`；闭环重放设备：{A['meta']['device']}（{A['meta']['gpu_name']}，CUDA_VISIBLE_DEVICES={A['meta']['cuda_visible_devices']}）；关联与 ICP 在 CPU 多进程上完成。")
    w(f"- 协议：`track_sequence`，`np.random.default_rng(0)` 跨序列共用，50 ms，val_core = zgz_global（{nS['zgz_global']} 包）+ zgz_local（{nS['zgz_local']} 包）；每包 = 一个 50 ms 评估步，关联对象 = 模型实际看到的 `stride_sample(ptr, 2048)` 事件；来源标签 = P1 `labels_zgz_*.npz`（逐包校验事件像素、抽样索引一致）。")
    w("- 指标口径与 P2-a 相同（直接复用 `probe_xa_assoc.count_step / rates`）：**精确部位一致率** = 关联质量加权硬关节 argmax == src_part，分母为 src_part≥0 的全部事件（未关联者算不一致，即 P2 的 `agree_exact_allhand`）；**M** = 来源为手指轮廓前景（src_kind∈{1,2} 且 src_part≥1）的事件中 >50% 质量落在关节 0 的比例；**hit** = 抽样事件中射线命中网格表面的比例。池化 = 各包事件数求和后相除；区间 = 滑动块 bootstrap（块长 20 包，1000 次，按步重采样，配对于同一抽样）。")
    w("- 偏移量与 oracle 都相对**窗末真值** `pos51[end]`；`gt` 条件的 prev = 窗首真值（`pos51[prev_idx]`，与 P2-b TF 相同）。\n")
    # ---- summary (placed first, generated from the numbers)
    def _st(seq, sname="全部"):
        return R["main"][seq][sname]
    g_, l_ = _st("zgz_global"), _st("zgz_local")
    ok_cl = (abs(100 * l_["cl"]["agree"] - 41.85) < 0.006 and abs(100 * l_["cl"]["M"] - 12.63) < 0.006 and
             abs(100 * g_["cl"]["agree"] - 78.86) < 0.006 and abs(100 * g_["cl"]["M"] - 2.44) < 0.006)
    ok_gt = (abs(100 * l_["gt"]["agree"] - 80.56) < 0.006 and abs(100 * l_["gt"]["M"] - 6.31) < 0.006)
    summ = [f"- **复现**：闭环 RA 总体 {A['ra']['overall']:.3f} mm（global {A['ra']['per_seq']['zgz_global']:.3f} / local {A['ra']['per_seq']['zgz_local']:.3f}），"
            f"与选点网格 / P2-a 的差 {A['ra_check']['abs_diff_overall']:.1e}；`cl` 的 local {_pct(l_['cl']['agree'])} / M {_pct(l_['cl']['M'])}、global {_pct(g_['cl']['agree'])} / M {_pct(g_['cl']['M'])} "
            f"{'与 P2-a 一致' if ok_cl else '**与 P2-a 不一致**'}；`gt` 的 local {_pct(l_['gt']['agree'])} / M {_pct(l_['gt']['M'])} {'与 P2-b TF 一致' if ok_gt else '**与 P2-b TF 不一致**'}。"]
    for sq, M in (("zgz_local", l_), ("zgz_global", g_)):
        n_ = M["cl"]["n_steps"]
        summ.append(f"- **{sq}（全部 {n_} 包）整手 oracle**：精确一致率 cl {_pct(M['cl']['agree'])} → sim2d_oracle {_pct(M['sim2d_oracle']['agree'])}（{M['sim2d_oracle']['delta_agree_pp']:+.1f} pp，回收 cl→gt 差距的 {100 * M['sim2d_oracle']['gap_recovery']:.0f}%）、"
                    f"rigid3d_oracle {_pct(M['rigid3d_oracle']['agree'])}（{100 * M['rigid3d_oracle']['gap_recovery']:.0f}%）、仅平移 oracle {_pct(M['trans2d_oracle']['agree'])}（{100 * M['trans2d_oracle']['gap_recovery']:.0f}%）、gt {_pct(M['gt']['agree'])}；"
                    f"M {_pct(M['cl']['M'])} → {_pct(M['sim2d_oracle']['M'])} / {_pct(M['rigid3d_oracle']['M'])} / {_pct(M['trans2d_oracle']['M'])}（gt {_pct(M['gt']['M'])}）；hit {_pct(M['cl']['hit'])} → {_pct(M['sim2d_oracle']['hit'])} / {_pct(M['rigid3d_oracle']['hit'])} / {_pct(M['trans2d_oracle']['hit'])}（gt {_pct(M['gt']['hit'])}）。")
    for sname in ("fingertip_in_front=真", "fingertip_in_front=假"):
        M = R["main"]["zgz_local"].get(sname)
        if M:
            summ.append(f"- zgz_local，{sname}（{M['cl']['n_steps']} 包）：一致率 cl {_pct(M['cl']['agree'])} → sim2d_oracle {_pct(M['sim2d_oracle']['agree'])}（回收 {100 * M['sim2d_oracle']['gap_recovery']:.0f}%）/ rigid3d_oracle {_pct(M['rigid3d_oracle']['agree'])} / sim2d_est {_pct(M['sim2d_est']['agree'])}（回收 {100 * M['sim2d_est']['gap_recovery']:.0f}%）/ gt {_pct(M['gt']['agree'])}；M {_pct(M['cl']['M'])} → {_pct(M['sim2d_oracle']['M'])} / {_pct(M['rigid3d_oracle']['M'])} / {_pct(M['sim2d_est']['M'])}（gt {_pct(M['gt']['M'])}）。")
    for sq in ("zgz_local", "zgz_global"):
        O = R["offsets"][sq]
        b0 = R["bins"][sq]["全部"][0]
        summ.append(f"- **{sq} 偏移量（{nS[sq]} 包，p50 [p90]）**：投影质心偏移 {O['geo_c_off']['p50']:.1f} [{O['geo_c_off']['p90']:.1f}] px；oracle 相似拟合 |旋转| {O['abs_orc_th_deg']['p50']:.1f} [{O['abs_orc_th_deg']['p90']:.1f}]°、尺度 {O['geo_orc_s']['p50']:.3f}；"
                    f"根深度误差 {O['geo_err_depth_mm']['p50']:.0f} [{O['geo_err_depth_mm']['p90']:.0f}] mm、根旋转误差 {O['geo_err_rot_deg']['p50']:.1f} [{O['geo_err_rot_deg']['p90']:.1f}]°、手指角误差范数 {O['geo_err_fing_rad']['p50']:.2f} [{O['geo_err_fing_rad']['p90']:.2f}] rad；"
                    f"顶点 2D RMS {O['geo_rms_before']['p50']:.1f} → oracle 后 {O['geo_rms_orc']['p50']:.1f} px。质心偏移 <2 px 的 {b0['n_steps']} 包上 cl {_pct(b0['rates']['cl']['agree'], 1)} / sim2d_oracle {_pct(b0['rates']['sim2d_oracle']['agree'], 1)} / gt {_pct(b0['rates']['gt']['agree'], 1)}（见表 2）。")
    for sq in ("zgz_local", "zgz_global"):
        M, E = _st(sq), R["est"][sq]
        e_, i_ = E["sim2d_est"], E["identity"]
        sl = R["steplevel"][sq]["sim2d_est"]
        summ.append(f"- **{sq} 仅事件 ICP（sim2d_est，{e_['n_ran']}/{nS[sq]} 包运行，其余事件<100 取恒等）**：一致率 {_pct(M['sim2d_est']['agree'])}（{M['sim2d_est']['delta_agree_pp']:+.1f} pp [{M['sim2d_est']['delta_agree_pp_ci'][0]:+.1f}, {M['sim2d_est']['delta_agree_pp_ci'][1]:+.1f}]，回收 {100 * M['sim2d_est']['gap_recovery']:.0f}%，占 sim2d_oracle 增益的 {100 * M['sim2d_est']['delta_agree_pp'] / M['sim2d_oracle']['delta_agree_pp']:.0f}%）；"
                    f"与 oracle 变换之差（p50 / p90）：质心 {e_['err_c_px']['p50']:.1f} / {e_['err_c_px']['p90']:.1f} px（不对齐 {i_['err_c_px']['p50']:.1f} / {i_['err_c_px']['p90']:.1f}）、旋转 {e_['err_th_deg']['p50']:.1f} / {e_['err_th_deg']['p90']:.1f}°（不对齐 {i_['err_th_deg']['p50']:.1f} / {i_['err_th_deg']['p90']:.1f}）、"
                    f"尺度 {100 * e_['err_s_rel']['p50']:.1f} / {100 * e_['err_s_rel']['p90']:.1f}%（不对齐 {100 * i_['err_s_rel']['p50']:.1f} / {100 * i_['err_s_rel']['p90']:.1f}）；顶点 RMS 变小的包 {_pct(e_['frac_rms_improved'], 0)}；逐包一致率提高 >2 pp 的 {_pct(sl['frac_up2'], 0)}、降低 >2 pp 的 {_pct(sl['frac_down2'], 0)}。"
                    f"变体（tight / 30 次迭代 / 尺度退火 / 仅平移 / 点到线）的回收：{100 * M['sim2d_est_tight']['gap_recovery']:.0f}% / {100 * M['sim2d_est_it30']['gap_recovery']:.0f}% / {100 * M['sim2d_est_cf']['gap_recovery']:.0f}% / {100 * M['trans2d_est']['gap_recovery']:.0f}% / {100 * M['sim2d_est_p2l']['gap_recovery']:.0f}%（点到线变体与 oracle 变换之差 p50：质心 {E['sim2d_est_p2l']['err_c_px']['p50']:.1f} px、旋转 {E['sim2d_est_p2l']['err_th_deg']['p50']:.1f}°、尺度 {100 * E['sim2d_est_p2l']['err_s_rel']['p50']:.1f}%）。")
    sy = {sq: {r["item"]: r for r in R["syn"][sq]} for sq in SEQS}
    summ.append("- **估计器自检（真值网格 + 已知相似扰动，p50 质心差 px / 旋转差 ° / 尺度差 %）**：" + "；".join(
        f"{sq}：零扰动 {sy[sq]['t_px:0.0']['err_c'][0]:.1f} / {sy[sq]['t_px:0.0']['err_th'][0]:.1f} / {100 * sy[sq]['t_px:0.0']['err_s'][0]:.1f}"
        f"（尺度带符号偏置 {100 * sy[sq]['t_px:0.0']['bias']['ds']:+.1f}%，dy {sy[sq]['t_px:0.0']['bias']['dy_px']:+.1f} px），"
        f"平移 8 / 16 / 32 px 后 {sy[sq]['t_px:8.0']['err_c'][0]:.1f} / {sy[sq]['t_px:16.0']['err_c'][0]:.1f} / {sy[sq]['t_px:32.0']['err_c'][0]:.1f} px，"
        f"旋转 ±10° / ±20° 后 {sy[sq]['rot_deg:10.0']['err_th'][0]:.1f} / {sy[sq]['rot_deg:20.0']['err_th'][0]:.1f}°" for sq in ("zgz_global", "zgz_local")) + "。")
    for sq in ("zgz_local", "zgz_global"):
        rb = [r for r in R["evbins"][sq] if r["scheme"] == "raw"]
        summ.append(f"- **{sq} 按窗内原始事件数分箱的 sim2d_est**：" + "；".join(
            f"{'<' + format(r['hi'], 'g') if r['lo'] == 0 else ('>' + format(r['lo'], 'g') if not np.isfinite(r['hi']) else format(r['lo'], 'g') + '–' + format(r['hi'], 'g'))} 事件：{r['n_steps']} 包，"
            f"Δ一致率 {r['delta_est_pp']:+.1f} pp（cl {100 * r['agree']['cl']:.1f} → est {100 * r['agree']['sim2d_est']:.1f} / oracle {100 * r['agree']['sim2d_oracle']:.1f} / gt {100 * r['agree']['gt']:.1f}），"
            + (f"质心差 {r['id_c'][0]:.1f} → {r['est_c'][0]:.1f} px、旋转差 {r['id_th'][0]:.1f} → {r['est_th'][0]:.1f}°、尺度差 {100 * r['id_s'][0]:.1f} → {100 * r['est_s'][0]:.1f}%（p50，恒等 → 估计）" if r["n_ran"] else "未运行 ICP")
            for r in rb) + "。")
    tq_ = (TM or {}).get("sequences", {})
    summ.append("- **ICP 迭代与 CPU 耗时**：" + "；".join(
        f"{sq}：平均迭代 {R['timing'][sq]['n_iter_mean']['sim2d_est']:.2f}/10 次（{_pct(R['timing'][sq]['frac_full_cap']['sim2d_est'], 0)} 的包跑满）、" +
        (f"每包剪影光栅化 + ICP 顺序单线程均值 {tq_[sq]['estimator_total_ms']['mean']:.1f} ms（p90 {tq_[sq]['estimator_total_ms']['p90']:.1f}；n = {tq_[sq]['n_timed']}）"
         if sq in tq_ else "顺序计时 n/a") + f"，64 进程并行下 ICP 均值 {R['timing'][sq]['icp_ms']['sim2d_est']['mean']:.1f} ms" for sq in ("zgz_global", "zgz_local")) + "。")
    w("## 摘要（事实性读数，由下列表中数字自动生成，不含架构结论）\n")
    for t_ in summ:
        w(t_)
    w("")
    # ---- 0. reproduction
    w("## 0. 复现核对\n")
    w("| 序列 | 包数 | 本次闭环 RA (mm) | 选点网格 step 6000 RA | P2-a RA | 差 |")
    w("|---|---|---|---|---|---|")
    ref = A["ra_check"].get("reference") or {}
    for s in SEQS:
        w(f"| {s} | {A['ra']['n_frames'][s]} | {A['ra']['per_seq'][s]:.4f} | {_num((ref.get('per_seq') or {}).get(s), 4)} | "
          f"{_num(p2a['ra']['per_seq'][s], 4) if p2a else 'n/a'} | {_num(A['ra_check']['abs_diff_per_seq'].get(s), 5)} |")
    w(f"| 全体（帧加权） | {sum(A['ra']['n_frames'].values())} | {A['ra']['overall']:.4f} | {_num(ref.get('overall'), 4)} | {_num(A['ra_check'].get('p2a_overall'), 4)} | {_num(A['ra_check']['abs_diff_overall'], 5)} |\n")
    bad_a = {s: {k: v for k, v in d.items() if v} for s, d in A["p2a_step_count_mismatch"].items()}
    w(f"- 阶段 A（GPU）逐包事件计数与 `outputs/xa/step2/p2a_closedloop.json` 对比，25 个计数键不一致的包数：{json.dumps(bad_a, ensure_ascii=False) if any(bad_a.values()) else '全部为 0'}。`state_hook` 原样返回 pred，RA 与选点网格逐位相同。")
    for s in SEQS:
        c = checks[s]
        b1 = {k: v for k, v in c["cl_vs_stage_a_gpu_mismatch"].items() if v}
        b2 = {k: v for k, v in c.get("gt_vs_p2b_tf_mismatch", {}).items() if v}
        w(f"- {s}：阶段 B（CPU）重算的 `cl` 与阶段 A（GPU，模型自己的 `xyz_last`）逐包计数不一致的包数（仅列非零键）：{json.dumps(b1) if b1 else '无'}；`gt` 与 P2-b TF 计数不一致的包数：{json.dumps(b2) if b2 else '无'}（`q_mis` 为浮点累加量，其余为整数计数）。")
    w("")
    # ---- 1. main table
    w("## 1. 五个条件：精确部位一致率 / M / hit\n")
    w("`Δ一致率` = 相对 `cl` 的百分点变化；`回收` = (条件 − cl) / (gt − cl)，即回收了闭环到窗首真值前态之间一致率差距的比例（方括号 = 95% 区间）。\n")
    w("| 序列 | 层 | 包数 | 条件 | 精确一致率 [95% CI] | M [95% CI] | hit | Δ一致率 pp [95% CI] | 回收 cl→gt 差距 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for s in SEQS:
        for sname, _ in STRATA:
            if sname in R["main"][s]:
                for c in MAIN:
                    w(_row_main(s, sname, c, R["main"][s][sname][c]))
    w("")
    w("补充条件（同口径；`trans2d_*` 只估平移；`sim2d_est_cf` 把 Tukey 尺度从 64 px 退火到 16 px；`sim2d_est_it30` 为 30 次迭代以检验收敛；`sim2d_est_tight` 在 10 次 16 px 迭代之后再用 4 px 尺度迭代 10 次以收紧内点；`sim2d_est_p2l` 把点对点闭式解换成点到线（轮廓法向）的 Gauss-Newton，其余（最近点匹配、Tukey 16 px、10 次、恒等初值）不变；`gt_end` = 用窗末真值网格做关联的参考，不是协议里的 prev）：\n")
    w("| 序列 | 层 | 包数 | 条件 | 精确一致率 [95% CI] | M [95% CI] | hit | Δ一致率 pp [95% CI] | 回收 cl→gt 差距 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for s in SEQS:
        for sname, _ in STRATA:
            if sname in R["main"][s] and (sname == "全部" or s == "zgz_local"):
                for c in SUPP:
                    w(_row_main(s, sname, c, R["main"][s][sname][c]))
    w("zgz_global 没有 fingertip_in_front=真 的包（0/1386），该层为空，故只有“全部”（= 假）层。\n")
    # ---- 2. bins
    for key, ttl, offname in (("bins", "质心偏移分箱（偏移 = 闭环 prev 的 778 个投影顶点质心 − 窗末真值投影质心，px）", "窗末真值"),
                              ("bins0", "补充：同样分箱，但偏移取相对窗首真值（prev 自己的目标；扣除 50 ms 内真值自身的运动）", "窗首真值")):
        w(f"## {'2' if key == 'bins' else '2b'}. {ttl}\n")
        w("单元格 = 精确一致率 % / M %（池化，该箱内全部包）。`cl` / `cl+sim2d_oracle` / `cl+rigid3d_oracle` 为要求的三个条件，`sim2d_est`、`gt` 作参照。" +
          ("local 另按 fingertip_in_front 分层（global 没有“真”的包，其“假”层与“全部”相同，省略）。" if key == "bins" else "") + "\n")
        w(f"| 序列 | 层 | 偏移（相对{offname}）px | 包数 | 事件数 | cl | cl+sim2d_oracle | cl+rigid3d_oracle | cl+sim2d_est | gt |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for s in SEQS:
            for sname, _ in STRATA:
                if sname not in R[key][s] or (key == "bins0" and sname != "全部") or (s == "zgz_global" and sname != "全部"):
                    continue
                for r in R[key][s][sname]:
                    hi = "∞" if not np.isfinite(r["hi"]) else f"{r['hi']:g}"
                    name = (f"  其中 {r['lo']:g}–{hi}" if r["fine"] else (f">{r['lo']:g}" if not np.isfinite(r["hi"]) and r["lo"] == 8.0 else f"{r['lo']:g}–{hi}"))
                    cells = " | ".join(f"{100 * r['rates'][c]['agree']:.1f} / {100 * r['rates'][c]['M']:.1f}" for c in MAIN)
                    w(f"| {s} | {sname} | {name} | {r['n_steps']} | {int(r['n_events'])} | {cells} |")
        w("")
    # ---- 3. offsets
    w("## 3. 闭环 prev 相对真值的整手偏移量（逐包分布，p10 / p50 / p90）\n")
    w("| 序列（包数） | 层 | 量 | p10 | p50 | p90 |")
    w("|---|---|---|---|---|---|")
    names = [("geo_c_off", "投影质心偏移 px（相对窗末真值）"), ("geo_c_off0", "投影质心偏移 px（相对窗首真值）"),
             ("geo_c_off_lag", "真值自身 50 ms 质心位移 px（窗首→窗末）"), ("geo_rms_before", "778 顶点 2D 均方根误差 px（未对齐）"),
             ("geo_rms_orc", "同上，2D 相似 oracle 对齐后（下限）"), ("geo_orc_th_deg", "oracle 相似拟合旋转 °（带符号）"),
             ("abs_orc_th_deg", "oracle 相似拟合旋转 °（绝对值）"), ("geo_orc_s", "oracle 相似拟合尺度（prev→真值）"),
             ("abs_orc_s_rel", "oracle 尺度偏离 |s−1|"), ("geo_err_depth_mm", "根深度误差 mm（相对窗末真值）"),
             ("geo_err_t_mm", "根平移误差 mm（3D，相对窗末真值）"), ("geo_err_rot_deg", "根旋转误差 °（测地线）"),
             ("geo_err_fing_rad", "手指角误差范数 rad（45 维轴角）"), ("geo_err0_t_mm", "根平移误差 mm（相对窗首真值）"),
             ("geo_err0_fing_rad", "手指角误差范数 rad（相对窗首真值）")]
    for s in SEQS:
        for k, nm in names:
            q = R["offsets"][s][k]
            w(f"| {s}（{nS[s]}） | 全部 | {nm} | {_num(q['p10'], 3)} | {_num(q['p50'], 3)} | {_num(q['p90'], 3)} |")
    w("\n逐包 Spearman 相关：闭环 `cl` 的精确一致率 与 各偏移量（仅 hand 事件数 ≥50 的包；描述性，量之间相关）：\n")
    w("| 序列（包数；层 = 全部） | 质心偏移(窗末) | 质心偏移(窗首) | 顶点 2D RMS | 根平移误差 | 根深度误差 | 根旋转误差 | 手指角误差 | \\|oracle 旋转\\| | \\|oracle 尺度−1\\| |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for s in SEQS:
        c = R["corr"][s]
        w(f"| {s}（{c['n_steps']}） | " + " | ".join(_num(c[k], 3) for k in ("geo_c_off", "geo_c_off0", "geo_rms_before", "geo_err_t_mm", "geo_err_depth_mm",
                                                                         "geo_err_rot_deg", "geo_err_fing_rad", "geo_orc_th_deg", "geo_orc_s")) + " |")
    w("")
    # ---- 4. estimate error
    w("## 4. 事件估计的相似变换 vs oracle 变换（误差分布，中位 / p90）\n")
    w("误差定义：质心位移差 = ‖T_est(c) − T_oracle(c)‖（c = prev 投影质心；T_oracle(c) 即真值质心），旋转差 = |θ_est − θ_oracle|，尺度差 = |s_est/s_oracle − 1|，顶点 RMS = T 作用后 778 顶点与窗末真值投影的 2D 均方根误差。“恒等”行 = 不对齐时的同一误差（即 oracle 位移本身）。统计范围 = ICP 实际运行的包（抽样事件 ≥ 100）；其余包 `sim2d_est` 取恒等。\n")
    w("| 序列 | 层 | 运行包数/总包数 | 估计器 | 质心差 px | 旋转差 ° | 尺度差 % | 顶点 RMS px | RMS 比不对齐小的包占比 |")
    w("|---|---|---|---|---|---|---|---|---|")
    f2 = lambda q, d=2, sc=1.0: f"{sc * q['p50']:.{d}f} / {sc * q['p90']:.{d}f}"          # noqa: E731
    for s in SEQS:
        E = R["est"][s]
        idn = E["identity"]
        w(f"| {s} | 全部 | {idn['n_ran']}/{nS[s]} | 恒等（不对齐） | {f2(idn['err_c_px'])} | {f2(idn['err_th_deg'], 1)} | {f2(idn['err_s_rel'], 1, 100)} | {f2(idn['rms_px'])} | — |")
        for e in ("sim2d_est", "trans2d_est", "sim2d_est_cf", "sim2d_est_it30", "sim2d_est_tight", "sim2d_est_p2l"):
            d = E[e]
            w(f"| {s} | 全部 | {d['n_ran']}/{nS[s]} | {CL[e].replace('cl+', '')} | {f2(d['err_c_px'])} | {f2(d['err_th_deg'], 1)} | {f2(d['err_s_rel'], 1, 100)} | {f2(d['rms_px'])} | {_pct(d['frac_rms_improved'], 1)} |")
        d = E["trans2d_oracle"]
        w(f"| {s} | 全部 | {d['n_ran']}/{nS[s]} | trans2d_oracle（仅平移 oracle） | {f2(d['err_c_px'])} | {f2(d['err_th_deg'], 1)} | {f2(d['err_s_rel'], 1, 100)} | {f2(d['rms_px'])} | — |")
        fl = E["oracle_floor"]
        w(f"| {s} | 全部 | {idn['n_ran']}/{nS[s]} | sim2d_oracle（下限） | 0 / 0 | 0 / 0 | 0 / 0 | {f2(fl['rms_px'])} | — |")
    w("")
    w("`sim2d_est` 误差按 oracle 位移（= 质心偏移）分箱（质心差/旋转差/尺度差：中位 [p90]；“恒等”列 = 不对齐的中位）：\n")
    w("| 序列 | 层 | oracle 质心偏移 px | 包数 | 质心差 恒等 → 估计 px | 旋转差 恒等 → 估计 ° | 尺度差 恒等 → 估计 % | RMS 变小的包占比 |")
    w("|---|---|---|---|---|---|---|---|")
    for s in SEQS:
        for r in R["est_by_bin"][s]:
            hi = "∞" if not np.isfinite(r["hi"]) else f"{r['hi']:g}"
            w(f"| {s} | 全部 | {r['lo']:g}–{hi} | {r['n_steps']} | {r['id_c']:.2f} → {r['est_c'][0]:.2f} [{r['est_c'][1]:.2f}] | {r['id_th']:.1f} → {r['est_th'][0]:.1f} [{r['est_th'][1]:.1f}] | "
              f"{100 * r['id_s']:.1f} → {100 * r['est_s'][0]:.1f} [{100 * r['est_s'][1]:.1f}] | {_pct(r['frac_rms_improved'], 1)} |")
    w("")
    w("逐包一致率变化（相对 `cl`，hand 事件数>0 的包；±2 pp 为阈值）：\n")
    w("| 序列 | 层 | 条件 | 包数 | 平均变化 pp | 提高>2pp 的包占比 | 降低>2pp 的包占比 |")
    w("|---|---|---|---|---|---|---|")
    for s in SEQS:
        for c in CONDS:
            if c == "cl":
                continue
            d = R["steplevel"][s][c]
            w(f"| {s} | 全部 | {CL[c]} | {d['n']} | {d['mean_pp']:+.2f} | {_pct(d['frac_up2'], 1)} | {_pct(d['frac_down2'], 1)} |")
    w("")
    # ---- 4c. event-count bins and 4d. iterations / CPU time
    w("### 4c. `sim2d_est` 按每窗事件数分箱：估计误差与应用后的一致率\n")
    w("两种分箱依据：①窗内原始事件数（local：<1000 / 1000–3000 / >3000；global 几乎全部超过抽样上限 2048，故按 <1000 / 1k–10k / >10k）；②模型实际看到的抽样事件数（≤2048，ICP 至少需要 100 个）。"
      "误差列 = 不对齐（恒等）→ ICP 估计，中位 [p90]，只统计 ICP 实际运行的包；一致率为池化精确一致率 %（层 = 全部）。\n")
    w("| 序列 | 层 | 分箱依据 | 事件数 | 包数（ICP 运行） | 质心差 px 恒等 → 估计 | 旋转差 ° 恒等 → 估计 | 尺度差 % 恒等 → 估计 | 一致率 cl | sim2d_est | sim2d_oracle | gt | Δ(est−cl) pp |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in SEQS:
        for r in R["evbins"][s]:
            hi = "∞" if not np.isfinite(r["hi"]) else f"{r['hi']:g}"
            nm = (f">{r['lo']:g}" if not np.isfinite(r["hi"]) and r["scheme"] == "raw" else (f"={r['lo']:g}（达上限）" if r["scheme"] == "sampled" and r["lo"] == 2048 else f"{r['lo']:g}–{hi}"))
            if r["n_ran"]:
                err = (f"{r['id_c'][0]:.2f} → {r['est_c'][0]:.2f} [{r['est_c'][1]:.2f}] | {r['id_th'][0]:.1f} → {r['est_th'][0]:.1f} [{r['est_th'][1]:.1f}] | "
                       f"{100 * r['id_s'][0]:.1f} → {100 * r['est_s'][0]:.1f} [{100 * r['est_s'][1]:.1f}]")
            else:
                err = "— | — | —"
            a = r["agree"]
            w(f"| {s} | 全部 | {'窗内原始' if r['scheme'] == 'raw' else '抽样'} | {nm} | {r['n_steps']}（{r['n_ran']}） | {err} | {100 * a['cl']:.1f} | {100 * a['sim2d_est']:.1f} | {100 * a['sim2d_oracle']:.1f} | {100 * a['gt']:.1f} | {r['delta_est_pp']:+.2f} |")
    w("")
    w("### 4d. ICP 迭代次数与每包 CPU 耗时\n")
    w("迭代次数 = 实际执行的迭代数（上限 = 变体的迭代预算；收敛阈为质心位移 <0.01 px 且 Δθ、Δs <1e-4）。耗时 = 墙钟，单线程；“并行”列取自阶段 B（64 个进程同时跑，含互相争用），“顺序”列取自 `--part timing`（单进程、独立于其他负载，均匀抽样包）。耗时只含估计器本身（剪影光栅化 + ICP），不含 FK 与关联。\n")
    w("| 序列 | 层 | 估计器 | ICP 运行包数 | 平均迭代次数 | 跑满迭代预算的包占比 | 并行：ICP ms 均值 / p50 / p90 |")
    w("|---|---|---|---|---|---|---|")
    for s in SEQS:
        Tm = R["timing"][s]
        for e in EST_SPEC:
            ic = Tm["icp_ms"][e]
            w(f"| {s} | 全部 | {e}（预算 {len(EST_SPEC[e][1])} 次） | {Tm['n_ran']} | {Tm['n_iter_mean'][e]:.2f} | {_pct(Tm['frac_full_cap'][e], 1)} | {ic['mean']:.1f} / {ic['p50']:.1f} / {ic['p90']:.1f} |")
    w("")
    w("| 序列 | 层 | 剪影光栅化 ms 并行（均值 / p50 / p90） | 顺序计时：包数 | 光栅化 ms 均值 | ICP(sim2d_est) ms 均值 / p50 / p90 | 光栅化 + ICP ms 均值 / p50 / p90 | 顺序计时的平均迭代数 | 参与 ICP 的平均抽样事件数 | 点到线变体 ICP ms 均值（平均迭代数） |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for s in SEQS:
        cm = R["timing"][s]["contour_ms"]
        tq = (TM or {}).get("sequences", {}).get(s)
        if tq:
            w(f"| {s} | 全部 | {cm['mean']:.1f} / {cm['p50']:.1f} / {cm['p90']:.1f} | {tq['n_timed']} | {tq['contour_ms']['mean']:.1f} | {tq['icp_sim2d_est_ms']['mean']:.1f} / {tq['icp_sim2d_est_ms']['p50']:.1f} / {tq['icp_sim2d_est_ms']['p90']:.1f} | "
              f"{tq['estimator_total_ms']['mean']:.1f} / {tq['estimator_total_ms']['p50']:.1f} / {tq['estimator_total_ms']['p90']:.1f} | {tq['n_iter_mean']:.2f} | {tq['n_events_sampled_mean']:.0f} | "
              f"{tq['icp_sim2d_est_p2l_ms']['mean']:.1f}（{tq['n_iter_mean_p2l']:.2f}） |")
        else:
            w(f"| {s} | 全部 | {cm['mean']:.1f} / {cm['p50']:.1f} / {cm['p90']:.1f} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    w("")
    # ---- 5. synthetic and support
    w("## 5. 估计器自检：对窗末真值网格施加已知相似扰动后的恢复（同一批事件、同一 ICP）\n")
    w("模型点集 = 被扰动的真值网格的剪影轮廓（形状完全正确，没有关节误差），扰动绕真值质心；表中为 ICP 后残余（中位 / p90）：质心差 px、旋转差 °、尺度差 %、顶点 RMS px（“扰动”列 = 未对齐的顶点 RMS）。平移方向 / 旋转符号随机（按包号定种子）。事件 < 100 的包不计。“平移 0 px”行是该估计器在**正确形状、零偏移**时的噪声底；它含窗内真值运动的贡献（事件分布在 50 ms 内，目标取窗末真值；真值质心位移 p50："
      + f"global {R['offsets']['zgz_global']['geo_c_off_lag']['p50']:.1f} px，local {R['offsets']['zgz_local']['geo_c_off_lag']['p50']:.1f} px）。\n")
    w("| 序列 | 层 | 扰动 | 包数 | 扰动顶点 RMS px | 对齐后顶点 RMS px | 质心差 px | 旋转差 ° | 尺度差 % | 带符号中位偏置 (dx, dy px / dθ ° / ds %) |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for s in SEQS:
        for r in R["syn"][s]:
            k, v = r["item"].split(":")
            nm = {"t_px": f"平移 {float(v):g} px", "rot_deg": f"旋转 ±{float(v):g}°", "scale": f"尺度 ×{float(v):g}"}[k]
            w(f"| {s} | 全部 | {nm} | {r['n']} | {r['rms_id'][0]:.2f} / {r['rms_id'][1]:.2f} | {r['rms_after'][0]:.2f} / {r['rms_after'][1]:.2f} | "
              f"{r['err_c'][0]:.2f} / {r['err_c'][1]:.2f} | {r['err_th'][0]:.1f} / {r['err_th'][1]:.1f} | {100 * r['err_s'][0]:.1f} / {100 * r['err_s'][1]:.1f} | "
              f"({r['bias']['dx_px']:+.2f}, {r['bias']['dy_px']:+.2f} / {r['bias']['dth_deg']:+.2f} / {100 * r['bias']['ds']:+.1f}) |")
    w("")
    w("点到线变体 `sim2d_est_p2l` 的同一自检（子集；同列定义）：\n")
    w("| 序列 | 层 | 扰动 | 包数 | 扰动顶点 RMS px | 对齐后顶点 RMS px | 质心差 px | 旋转差 ° | 尺度差 % | 带符号中位偏置 (dx, dy px / dθ ° / ds %) |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for s in SEQS:
        for r in R["syn_p2l"][s]:
            if r["item"] not in ("t_px:0.0", "t_px:8.0", "t_px:16.0", "t_px:32.0", "rot_deg:10.0", "rot_deg:20.0", "scale:0.85", "scale:1.15"):
                continue
            k, v = r["item"].split(":")
            nm = {"t_px": f"平移 {float(v):g} px", "rot_deg": f"旋转 ±{float(v):g}°", "scale": f"尺度 ×{float(v):g}"}[k]
            w(f"| {s} | 全部 | {nm} | {r['n']} | {r['rms_id'][0]:.2f} / {r['rms_id'][1]:.2f} | {r['rms_after'][0]:.2f} / {r['rms_after'][1]:.2f} | "
              f"{r['err_c'][0]:.2f} / {r['err_c'][1]:.2f} | {r['err_th'][0]:.1f} / {r['err_th'][1]:.1f} | {100 * r['err_s'][0]:.1f} / {100 * r['err_s'][1]:.1f} | "
              f"({r['bias']['dx_px']:+.2f}, {r['bias']['dy_px']:+.2f} / {r['bias']['dth_deg']:+.2f} / {100 * r['bias']['ds']:+.1f}) |")
    w("")
    w("## 6. 事件对整手变换的支撑（描述性）\n")
    w("| 序列（层 = 全部） | 抽样事件数 | 非手事件占比 | 掌(关节0)占手部事件 | 食指 | 中指 | 小指 | 无名指 | 拇指 | 窗内事件数 p10/p50/p90 | 真值剪影轮廓 3 px 内有事件的像素占比 p10/p50/p90 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in SEQS:
        d = R["support"][s]
        g = d["share_by_group_of_hand"]
        nv, cv = d["n_ev_window"], d["cov_gt3"]
        w(f"| {s} | {int(d['events'])} | {_pct(d['share_nonhand'], 1)} | {_pct(g['掌'], 1)} | {_pct(g['食指'], 1)} | {_pct(g['中指'], 1)} | {_pct(g['小指'], 1)} | {_pct(g['无名指'], 1)} | {_pct(g['拇指'], 1)} | "
          f"{nv['p10']:.0f}/{nv['p50']:.0f}/{nv['p90']:.0f} | {_pct(cv['p10'], 0)}/{_pct(cv['p50'], 0)}/{_pct(cv['p90'], 0)} |")
    w("")
    w("## 7. 实现与口径说明\n")
    w("- **oracle 2D 相似**：对 prev 的 778 个投影顶点与窗末真值的 778 个投影顶点做无权最小二乘相似拟合（闭式 Umeyama，平移 + 旋转 + 尺度，全部顶点含不可见者）；作用于 prev 投影后，按 prev 各顶点**原深度**反投影 V' = z·K⁻¹[u', v', 1]（K 为渲染帧内参 0.375·K），再调用现行 `associate_rays`（hit_first、band 16/f_px、front_tol 3/f_px、z_tol 0.01、k 16；与 `_xyz_mesh_forward` 同一调用，射线由抽样事件像素算出）。")
    w("- **rigid3d_oracle**：prev[0:6]（根平移、根旋转）换成窗末真值，手指 prev[6:51] 保持闭环值，FK 后关联。")
    w("- **sim2d_est**：模型点集 = prev 的 778 顶点投影网格的**光栅化剪影掩膜边界**（全部三角形并集 + 取整顶点；掩膜像素有 4 邻域非掩膜者；画面边界不算轮廓）；观测点集 = 模型实际看到的抽样事件像素（≤2048）。每次迭代：每个事件取当前变换后轮廓点中的最近点，权重 Tukey 双权 (1−(r/16)²)²（r<16 px，否则 0），用**原始**轮廓点到事件的加权闭式相似变换（不累积复合）；初值恒等，最多 10 次（收敛阈：质心位移 <0.01 px 且 Δθ、Δs <1e-4 才提前停，实际几乎总是跑满 10 次）；抽样事件 <100 个或有效权重事件 <10 个时保持恒等（该包的 sim2d_est 条件与 cl 同）。除此之外没有任何先验、范围钳制或后处理。")
    w("- **补充条件**：`trans2d_*` 只解平移（加权质心差；`trans2d_oracle` = 两个投影质心之差）；`sim2d_est_cf` 的 Tukey 尺度依次 64,48,32,24 后固定 16 px；`sim2d_est_it30` 为 30 次迭代；`sim2d_est_tight` 为 10 次 16 px 后再 10 次 4 px；`sim2d_est_p2l` 用轮廓法向（局部 7 点协方差最小特征向量）的点到线残差做 Gauss-Newton（参数 t、θ、s，阻尼 1e-6，尺度下限 0.2），其余同 `sim2d_est`；`gt_end` = 窗末真值网格直接关联（不是协议里的 prev，只作参照）。这些变体都是在看到 `sim2d_est` 的结果之后加的敏感性对照，不替代题设的 `sim2d_est`。")
    w("- **估计器的方向与结构性特点**：`sim2d_est` 的匹配方向为事件→最近轮廓点（题设）；内部（手内部）事件会被吸附到最近的外轮廓，点对点解沿轮廓“滑动”收敛慢（对干净椭圆做的单元检查：10 次点对点迭代只恢复了 5.7° 中的 1.4°，点到线 5 次恢复全部）。oracle 用了 778 个顶点的对应关系，估计器只有轮廓形状与事件，二者的信息量不同。")
    w("- **自检的读法**：第 5 节把窗末真值网格按已知相似扰动后交给同一估计器，形状完全正确，所以它给出的是估计器 + 这批事件的噪声底与捕获范围；零扰动行的 dy ≈ +1 px、尺度 −4% 到 −6% 是带符号的系统偏置（点对点与点到线两种解一致），本探针没有对它做校正。")
    w("- **样本范围**：单个检查点（xa_c1_s3407 step 6000）、两条序列（zgz_global 1386 包、zgz_local 1204 包，同属 val_core）、一次闭环重放（rng(0)）；区间只反映包间相关性下的重采样不确定度，不含训练种子差异（xa_c1 与 xa_c1r 的末步 RA 差 3.1 mm 说明训练非确定性远大于此处的区间）。")
    mx = 100 * max(abs(d["M"] - d["M_has"]) for sq in SEQS for d in R["main"][sq]["全部"].values())
    Ml = R["main"]["zgz_local"]["全部"]
    w(f"- **M 的读法**：M 的分母含未关联事件、分子要求有关联；本数据里手指轮廓前景事件几乎都有关联（各条件 `M_has` 与 M 的最大差 {mx:.3f} 个百分点），所以 M 的变化是分配的变化，不是关联丢失造成的机械下降。M 只看手指轮廓事件是否被判给关节 0，与精确一致率度量不同的东西：例如 zgz_local 的 `sim2d_est` M = {_pct(Ml['sim2d_est']['M'])}、低于 `sim2d_oracle` 的 {_pct(Ml['sim2d_oracle']['M'])}，而它的一致率增益更小（{Ml['sim2d_est']['delta_agree_pp']:+.2f} 对 {Ml['sim2d_oracle']['delta_agree_pp']:+.2f} pp），两者不要互相推断。")
    w("- **偏移量**：质心 = 778 个投影顶点的均值；根深度误差取 MANO 腕关节（OpenPose 关节 0）的 z 差；根旋转误差 = R(prev[3:6])⁻¹R(gt[3:6]) 的测地线角；手指角误差 = ‖prev[6:51] − gt[6:51]‖₂（45 维轴角，rad）。")
    w("- **起点包**：每个 run 的第一包 prev = 真值 + 初始噪声（不是模型输出），与 P2-a 口径一致地计入，未剔除（逐包标志 `run_first` 在 `per_step_*.npz`）。")
    w(f"- 运行：阶段 A {A['meta']['seconds']:.0f} s（闭环重放，GPU 2）；阶段 B（CPU，64 个进程）每条序列 20–40 s，日志见 `stage_b.log`；顺序计时见 `timing.log` / `timing.json`。命令：`CUDA_VISIBLE_DEVICES=2 python tools/xa/probe_align.py --part a`；`CUDA_VISIBLE_DEVICES= python tools/xa/probe_align.py --part b --workers 64`；`... --part timing --timing-n 120`；`... --part report`。")
    (out_dir / "REPORT.md").write_text("\n".join(L) + "\n")
    log("wrote", out_dir / "REPORT.md")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="a,b,report")
    ap.add_argument("--run-dir", default=PX.RUN_DEFAULT)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--labels-dir", default=PX.LABELS_DEFAULT)
    ap.add_argument("--out-dir", default="outputs/xa/align")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=0, help="self-test: first N evaluation steps of each sequence (stage a)")
    ap.add_argument("--limit", type=int, default=0, help="self-test: first N steps in stage b")
    ap.add_argument("--seqs", default="", help="stage b: comma separated subset")
    ap.add_argument("--timing-n", type=int, default=300, help="timing stage: packets per sequence")
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    os.chdir(REPO)
    out_dir = Path(args.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = set(args.part.split(","))
    if "a" in parts:
        run_stage_a(args, out_dir)
    if "b" in parts:
        run_stage_b(args, out_dir)
    if "timing" in parts:
        run_timing(args, out_dir)
    if "report" in parts:
        make_report(args, out_dir)


if __name__ == "__main__":
    main()
