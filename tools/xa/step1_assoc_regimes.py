#!/usr/bin/env python3
"""[XYZ-ASSOC] P1 step 2: current vs alternative ray/mesh association under four prev regimes, scored against
the GT-mesh source labels of step1_source_labels.py (CPU only).

Pre-registration: docs/EXPERIMENT_RECORDS.md, XYZ-ASSOC-20261009, section 2.

Regimes (prev grid for the association; the event rays and sampling are the evaluator's):
  R3   each event uses the GT mesh of its own millisecond (only the rule's error is left)
  R0   the window-start GT (evaluator ideal prev: pos51[end-50], first step of a run pos51[a]) -> rule + 50 ms lag
  R1s  R0 + Gaussian noise, sigma (0.005, 0.05, 0.05) on t[0:3] / rot[3:6] / pose[6:51], per dimension
  R1l  R0 + Gaussian noise, sigma (0.05, 0.3, 0.3)
Noise: rng = np.random.default_rng(0), one standard-normal (steps x 51) draw per sequence in step order,
scaled by sigma, fresh per step (not chained); R1s and R1l share the standard normals.

Rules:
  cur    semkine.xyz_mesh.associate_rays (the library function, unchanged)
  alt    XYZ_ASSOC_MODE=contour of the pre-registration: hit as cur; near layer always computed with vertex-normal
         back-face filter (n.d > 0 removed); hit present, filtered near layer non-empty and zfront <= t_hit - 0.01
         -> two hypotheses (hit 3 vertices (1-a) A_hit, foreground <= 4 vertices a A_near, a = q_near, r = 0, ctr = 1);
         otherwise exactly cur (also for no-hit events: the *unfiltered* near layer).
  altf   sensitivity variant of alt: identical decision, but the no-hit fallback uses the back-face-filtered near layer.
The reimplementation of cur inside `associate_dual` is compared with the library output for every call.

Event sets: `sampled` = stride_sample(ptr, 2048) of each window (what the model sees; primary), `all` = every event
(secondary).  Row independence of the association lets `all` runs derive `sampled` by indexing.
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step1_common as C                                           # noqa: E402
from step1_common import (H, W, CAP, OUT_DIR, BAND_PX, FRONT_PX, Z_TOL, KNN, JOINT_GROUP, get_mano, sequences,
                          enumerate_steps, fk_batch, render_intrinsics_f32, face_part_table, load_seq_bundle,
                          _window_events)                          # noqa: E402
import numpy as np                                                 # noqa: E402
import torch                                                       # noqa: E402
from semkine.xyz_mesh import associate_rays, stride_sample, vertex_face_table, ray_triangle, R_MAX   # noqa: E402

REGIMES = ["R3", "R0", "R1s", "R1l"]
RULES = ["cur", "alt", "altf"]
SIGMA_S = np.array([0.005] * 3 + [0.05] * 3 + [0.05] * 45, np.float32)
SIGMA_L = np.array([0.05] * 3 + [0.3] * 3 + [0.3] * 45, np.float32)
GAP = 0.01
CHUNK = 8192
# aggregation table geometry
NFIP, NKIND, NGRP, NAT, NCTR = 2, 4, 7, 3, 2
NCELL = NFIP * NKIND * NGRP * NAT * NCTR
CNT = ["n", "valid", "hit", "resid", "agree_exact", "agree_group", "maj0", "plur0", "m0frac_sum"]
NBIN, BIN_MM, ZRANGE = 2400, 0.25, 300.0          # z-error histograms: +-300 mm at 0.25 mm
HA, HB, HC = 2 * NKIND * 2 * NFIP, 2 * NGRP, 2 * NKIND * 2
NHROW = HA + HB + HC


# ------------------------------------------------------------------ association: library cur + alternatives
def vertex_normals(verts, faces):
    """Area-weighted vertex normals (sum of unnormalised face normals), normalised. verts (B, V, 3).
    MANO face winding gives outward normals (front faces have n.d < 0, verified on the rasteriser)."""
    f = faces.long()
    v0, v1, v2 = verts[:, f[:, 0]], verts[:, f[:, 1]], verts[:, f[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    vn = torch.zeros_like(verts)
    for c in range(3):
        vn.index_add_(1, f[:, c], fn)
    return vn / vn.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def associate_dual(d, mask, verts, faces, vert_faces, vn, band, front_tol, z_tol, k=16, gap=GAP, chunk=CHUNK):
    """One pass that returns `cur` (a copy of associate_rays' computation), `alt` and `altf` (see module doc).
    cur: dict vid/A (B,N,4), q/z/valid/hit (B,N).  alt/altf: vid/w/zr (B,N,8), r/valid/hit/ctr (B,N)."""
    B, N, _ = d.shape
    V = verts.shape[1]
    dev = d.device
    z0 = lambda *s: torch.zeros(*s, dtype=torch.float32, device=dev)
    cur = {"vid": torch.zeros(B, N, R_MAX, dtype=torch.long, device=dev), "A": z0(B, N, R_MAX), "q": z0(B, N),
           "z": z0(B, N), "valid": torch.zeros(B, N, dtype=torch.bool, device=dev),
           "hit": torch.zeros(B, N, dtype=torch.bool, device=dev)}
    mk = lambda: {"vid": torch.zeros(B, N, 2 * R_MAX, dtype=torch.long, device=dev), "w": z0(B, N, 2 * R_MAX),
                  "zr": z0(B, N, 2 * R_MAX), "r": z0(B, N), "valid": torch.zeros(B, N, dtype=torch.bool, device=dev),
                  "hit": torch.zeros(B, N, dtype=torch.bool, device=dev),
                  "ctr": torch.zeros(B, N, dtype=torch.bool, device=dev)}
    alt, altf = mk(), mk()
    outs = (cur, alt, altf)
    flat = mask.reshape(-1).nonzero().squeeze(1)
    if flat.numel() == 0:
        return cur, alt, altf
    Vf = verts.float()
    zv = Vf[..., 2]
    bad = zv <= 1e-6
    vu = (Vf[..., 0] / zv.clamp_min(1e-6)).masked_fill(bad, 1e6)
    vv = (Vf[..., 1] / zv.clamp_min(1e-6)).masked_fill(bad, 1e6)
    faces = faces.long()
    kk = max(1, min(int(k), V))
    col = torch.arange(kk, device=dev)
    parts = [{} for _ in outs]
    for i0 in range(0, int(flat.numel()), int(chunk)):
        rows = flat[i0:i0 + chunk]
        b = rows // N
        dd = d.reshape(-1, 3)[rows].float()
        m = rows.numel()
        d2 = (vu[b] - dd[:, 0:1]).square() + (vv[b] - dd[:, 1:2]).square()
        near = d2.topk(kk, dim=-1, largest=False)
        nid = near.indices
        nang = near.values.clamp_min(0.0).sqrt()
        cf = vert_faces[nid].reshape(m, -1)
        cfs = cf.clamp_min(0)
        tri = faces[cfs]
        P = Vf[b.view(m, 1, 1), tri]
        t, u, v = ray_triangle(dd.view(m, 1, 3), P[:, :, 0], P[:, :, 1], P[:, :, 2])
        t = t.masked_fill(cf < 0, float("inf"))
        tmin, fsel = t.min(dim=-1)
        is_hit = torch.isfinite(tmin)
        fh = cfs.gather(1, fsel[:, None]).squeeze(1)
        uh = u.gather(1, fsel[:, None]).squeeze(1)
        vh = v.gather(1, fsel[:, None]).squeeze(1)
        wb = torch.stack([(1.0 - uh - vh).clamp_min(0), uh.clamp_min(0), vh.clamp_min(0),
                          torch.zeros_like(uh)], dim=-1)
        A_hit = wb / wb.sum(-1, keepdim=True).clamp_min(1e-12)
        fv = faces[fh]
        vid_hit = torch.cat([fv, fv[:, :1]], dim=1)
        bnd = band[b].float().unsqueeze(1)
        ftl = front_tol[b].float().unsqueeze(1)
        Vn = Vf[b.view(m, 1), nid]
        zj = (Vn * dd.unsqueeze(1)).sum(-1) / (dd * dd).sum(-1, keepdim=True)
        sig = bnd / 2.0
        back = (vn[b.view(m, 1), nid] * dd.unsqueeze(1)).sum(-1) > 0           # (m, kk) back-facing candidates

        def layer_of(allowed):
            inband = (nang <= bnd) & (zj > 1e-4) & allowed
            ref = nang.masked_fill(~allowed, float("inf")).min(dim=-1, keepdim=True).values
            same = inband & (nang <= ref + ftl)
            zfront = torch.where(same, zj, torch.full_like(zj, float("inf"))).min(-1).values
            layer = inband & (zj <= zfront.unsqueeze(1) + float(z_tol))
            rank = torch.where(layer, col.unsqueeze(0), col.unsqueeze(0) + kk)
            cols = rank.argsort(dim=-1)[:, :R_MAX]
            kept = layer.gather(1, cols)
            ang = nang.gather(1, cols)
            kw = torch.exp(-0.5 * (ang / sig).square()) * kept
            A_n = kw / kw.sum(-1, keepdim=True).clamp_min(1e-12)
            q_n = torch.exp(-0.5 * (ang[:, 0] / sig[:, 0]).square())
            z_n = (A_n * zj.gather(1, cols)).sum(-1)
            return dict(nonempty=kept.any(-1), A=A_n, q=q_n, z=z_n, vid=nid.gather(1, cols), zfront=zfront)

        nc = layer_of(torch.ones_like(back))                                   # == associate_rays' layer
        nf = layer_of(~back)
        # ---- cur (same expressions as associate_rays)
        is_near = (~is_hit) & nc["nonempty"]
        ok = is_hit | is_near
        c_vid = torch.where(is_hit[:, None], vid_hit, nc["vid"]) * ok[:, None]
        c_A = torch.where(is_hit[:, None], A_hit, nc["A"]) * ok[:, None]
        c_q = torch.where(is_hit, torch.ones_like(nc["q"]), nc["q"]) * ok
        c_z = torch.where(is_hit, tmin, nc["z"]) * ok
        parts[0].setdefault("vid", []).append(c_vid); parts[0].setdefault("A", []).append(c_A)
        parts[0].setdefault("q", []).append(c_q); parts[0].setdefault("z", []).append(c_z)
        parts[0].setdefault("valid", []).append(ok); parts[0].setdefault("hit", []).append(is_hit)
        # ---- decision of the contour rule
        trig = is_hit & nf["nonempty"] & (nf["zfront"] <= tmin - float(gap))
        alpha = nf["q"]
        w_trig = torch.cat([(1.0 - alpha).unsqueeze(1) * A_hit, alpha.unsqueeze(1) * nf["A"]], dim=1)
        vid_trig = torch.cat([vid_hit, nf["vid"]], dim=1)
        z_trig = torch.cat([tmin.unsqueeze(1).expand(m, R_MAX), nf["z"].unsqueeze(1).expand(m, R_MAX)], dim=1)
        zero4 = torch.zeros(m, R_MAX, device=dev)
        zero4l = torch.zeros(m, R_MAX, dtype=torch.long, device=dev)
        # alt: fallback = cur;  altf: fallback = filtered near for the no-hit events
        is_near_f = (~is_hit) & nf["nonempty"]
        ok_f = is_hit | is_near_f
        f_vid = torch.where(is_hit[:, None], vid_hit, nf["vid"]) * ok_f[:, None]
        f_A = torch.where(is_hit[:, None], A_hit, nf["A"]) * ok_f[:, None]
        f_q = torch.where(is_hit, torch.ones_like(nf["q"]), nf["q"]) * ok_f
        f_z = torch.where(is_hit, tmin, nf["z"]) * ok_f
        for j, (vv_, AA, qq, zz, okk) in ((1, (c_vid, c_A, c_q, c_z, ok)), (2, (f_vid, f_A, f_q, f_z, ok_f))):
            w_fb = torch.cat([qq.unsqueeze(1) * AA, zero4], 1)
            vid_fb = torch.cat([vv_, zero4l], 1)
            z_fb = torch.cat([zz.unsqueeze(1).expand(m, R_MAX), zero4], 1)
            pj = parts[j]
            pj.setdefault("vid", []).append(torch.where(trig[:, None], vid_trig, vid_fb))
            pj.setdefault("w", []).append(torch.where(trig[:, None], w_trig, w_fb))
            pj.setdefault("zr", []).append(torch.where(trig[:, None], z_trig, z_fb))
            pj.setdefault("r", []).append(torch.where(trig, torch.zeros_like(qq), 1.0 - qq))
            pj.setdefault("valid", []).append(okk | trig)
            pj.setdefault("hit", []).append(is_hit)
            pj.setdefault("ctr", []).append(trig)
    for o, pr in zip(outs, parts):
        for key, dst in o.items():
            dst.view(-1, *dst.shape[2:])[flat] = torch.cat(pr[key]).to(dst.dtype)
    return cur, alt, altf


# ------------------------------------------------------------------ per-event summary and accumulation
def summarize(vid, w, zr, r, valid, hit, ctr, hard):
    """Per-event quantities from association records. vid (n,R) long, w (n,R) mass, zr (n,R) record depth."""
    n = vid.shape[0]
    jm = torch.zeros(n, 16, dtype=torch.float32).scatter_add_(1, hard[vid], w.float())
    tot = jm.sum(1)
    has = tot > 0
    pred = torch.where(has, jm.argmax(1), torch.full((n,), -1, dtype=torch.long))
    m0 = jm[:, 0]
    maj0 = has & (m0 > 0.5 * tot)
    m0frac = torch.where(has, m0 / tot.clamp_min(1e-12), torch.full_like(tot, float("nan")))
    R = w.shape[1]
    zhit = zr[:, 0]
    if R > R_MAX:
        mass_hit, mass_front = w[:, :R_MAX].sum(1), w[:, R_MAX:].sum(1)
        zfront = zr[:, R_MAX]
        zmaj = torch.where(ctr & (mass_front > mass_hit), zfront, zhit)
    else:
        zfront, zmaj = zhit, zhit
    atype = torch.where(hit, torch.full((n,), 2), torch.where(valid, torch.full((n,), 1), torch.zeros(n, dtype=torch.long)))
    return {k: v.numpy() for k, v in dict(pred=pred, maj0=maj0, m0frac=m0frac, tot=tot, zmaj=zmaj, zfront=zfront,
                                          zhit=zhit, atype=atype, resid=r.float(), valid=valid, hit=hit,
                                          ctr=ctr).items()}


class Table:
    """Sufficient statistics of one (event set, regime, rule): counters per stratum cell, z-error histograms,
    a source-part x associated-joint confusion table."""

    def __init__(self):
        self.cnt = np.zeros((len(CNT), NCELL), np.float64)
        self.hz = np.zeros((NHROW, NBIN), np.int64)
        self.conf = np.zeros(NKIND * 17 * 17, np.int64)

    def add(self, kind, part, depth, fip, s):
        kind, part = kind.astype(np.int64), part.astype(np.int64)
        pred = s["pred"].astype(np.int64)
        hand = part >= 0
        grp = np.where(hand, 1 + JOINT_GROUP[np.maximum(part, 0)], 0)
        atype, ctr = s["atype"].astype(np.int64), s["ctr"].astype(np.int64)
        cell = (((int(fip) * NKIND + (kind + 1)) * NGRP + grp) * NAT + atype) * NCTR + ctr
        ag_e = hand & (pred == part)
        ag_g = hand & (pred >= 0) & (JOINT_GROUP[np.maximum(pred, 0)] == JOINT_GROUP[np.maximum(part, 0)])
        m0f = np.nan_to_num(s["m0frac"].astype(np.float64), nan=0.0)
        vals = [np.ones(len(kind)), s["valid"], s["hit"], s["resid"], ag_e, ag_g, s["maj0"], pred == 0, m0f]
        for i, v in enumerate(vals):
            self.cnt[i] += np.bincount(cell, weights=np.asarray(v, np.float64), minlength=NCELL)
        mism = (s["maj0"] & (kind >= 1) & (part >= 1)).astype(np.int64)
        fin = hand & s["valid"] & np.isfinite(depth)
        def binof(err_m):
            return np.clip(np.floor((err_m * 1000.0 + ZRANGE) / BIN_MM), 0, NBIN - 1).astype(np.int64)
        if fin.any():
            e = binof(s["zmaj"][fin].astype(np.float64) - depth[fin].astype(np.float64))
            at, kd, mm, gr = atype[fin], kind[fin], mism[fin], grp[fin]
            ra = (((at - 1) * NKIND + (kd + 1)) * 2 + mm) * NFIP + int(fip)
            rb = HA + (at - 1) * NGRP + gr
            self.hz += np.bincount(np.concatenate([ra, rb]) * NBIN + np.concatenate([e, e]),
                                   minlength=NHROW * NBIN).reshape(NHROW, NBIN)[:NHROW]
        cm = fin & (ctr == 1)
        if cm.any():
            kd, mm = kind[cm], mism[cm]
            ef = binof(s["zfront"][cm].astype(np.float64) - depth[cm].astype(np.float64))
            eh = binof(s["zhit"][cm].astype(np.float64) - depth[cm].astype(np.float64))
            rows = np.concatenate([HA + HB + (0 * NKIND + kd + 1) * 2 + mm, HA + HB + (1 * NKIND + kd + 1) * 2 + mm])
            self.hz += np.bincount(rows * NBIN + np.concatenate([ef, eh]), minlength=NHROW * NBIN).reshape(NHROW, NBIN)
        self.conf += np.bincount((kind + 1) * 289 + (part + 1) * 17 + (pred + 1), minlength=NKIND * 289)

    def merge(self, o):
        self.cnt += o.cnt; self.hz += o.hz; self.conf += o.conf


# ------------------------------------------------------------------ worker
_G = {}
FIELDS = ["pred", "maj0", "m0frac", "zmaj", "atype", "ctr", "resid"]
FDT = dict(pred=np.int8, maj0=bool, m0frac=np.float16, zmaj=np.float32, atype=np.uint8, ctr=bool, resid=np.float16)


def _init(seq, legacy_dir, mode, lab_path, noise):
    torch.set_num_threads(1)
    ev, off, tsub, aux, pos51, betas, K = load_seq_bundle(seq, legacy_dir)
    mano = get_mano()
    hard, fpart, _ = face_part_table(mano)
    fx, fy, cx, cy = render_intrinsics_f32(K)
    f_px = (fx * fy).sqrt()
    _G.update(events=ev, offsets=off, tsub=tsub, pos51=pos51, betas=betas, mano=mano, faces=mano.f,
              vert_faces=vertex_face_table(mano.f, mano.weights.shape[0]), hard=torch.from_numpy(hard),
              fx=fx, fy=fy, cx=cx, cy=cy, band=BAND_PX / f_px, front=FRONT_PX / f_px, mode=mode, lab=lab_path,
              noise=noise, seq=seq)
    _, ends, prev = enumerate_steps(aux)
    _G.update(ends=ends, prev=prev)


def _regime_grids(s):
    """FK grids of one step: 50 per-ms GT grids, then R0, R1s, R1l prev grids."""
    g = _G
    end, prev_ms = int(g["ends"][s]), int(g["prev"][s])
    p0 = g["pos51"][prev_ms].astype(np.float32)
    z = g["noise"][s]
    params = np.concatenate([g["pos51"][end - 49:end + 1], p0[None], (p0 + z * SIGMA_S)[None],
                             (p0 + z * SIGMA_L)[None]]).astype(np.float32)
    verts, _ = fk_batch(g["mano"], g["betas"], params)
    return verts


def _batch(idx, ms_rel, d_all, regime, verts, vn):
    """Association inputs for event subset `idx` (sorted) under `regime`. Returns d, mask, verts, vn, band, front,
    and the function that flattens an (B, N, ...) output back to event order."""
    g = _G
    m = len(idx)
    if regime == "R3":
        ml = ms_rel[idx]
        counts = np.bincount(ml, minlength=50)
        N = max(int(counts.max()), 1)
        starts = np.cumsum(counts) - counts
        colp = np.arange(m) - starts[ml]
        dp = torch.zeros(50, N, 3)
        dp[torch.from_numpy(ml), torch.from_numpy(colp)] = d_all[idx]
        mask = torch.zeros(50, N, dtype=torch.bool)
        mask[torch.from_numpy(ml), torch.from_numpy(colp)] = True
        gi = 0
        ml_t, col_t = torch.from_numpy(ml), torch.from_numpy(colp)
        take = lambda x: x[ml_t, col_t]
        vb, nb = verts[:50], vn[:50]
        B = 50
    else:
        gi = {"R0": 50, "R1s": 51, "R1l": 52}[regime]
        dp = d_all[idx].unsqueeze(0)
        mask = torch.ones(1, m, dtype=torch.bool)
        take = lambda x: x[0, :m]
        vb, nb = verts[gi:gi + 1], vn[gi:gi + 1]
        B = 1
    return dp, mask, vb, nb, g["band"].expand(B).contiguous(), g["front"].expand(B).contiguous(), take


def run_task(task):
    j, steps = task
    g = _G
    lab = g["lab"]
    mode = g["mode"]
    sets = ["sampled"] if mode == "sampled" else ["sampled", "all"]
    tabs = {(st, rg, ru): Table() for st in sets for rg in REGIMES for ru in RULES}
    chk = {}
    bump = lambda k, v=1: chk.__setitem__(k, chk.get(k, 0) + int(v))
    per_event = {}                                              # sampled-set compact outputs (per step lists)
    t0 = time.time()
    for ii, s in enumerate(steps):
        end = int(g["ends"][s]); start = end - 49
        off = g["offsets"]
        ev5 = _window_events(g["events"], off, g["tsub"], end, 50)
        n = len(ev5)
        e0, e1 = int(lab["step_ev_start"][s]), int(lab["step_ev_start"][s + 1])
        assert e1 - e0 == n, (s, n, e1 - e0)
        x, y = ev5[:, 1], ev5[:, 2]
        counts = np.diff(off[start:end + 2]).astype(np.int64)
        ms_abs = np.repeat(np.arange(start, end + 1, dtype=np.int64), counts)
        assert (lab["ev_x"][e0:e1] == x.astype(np.int16)).all() and (lab["ev_y"][e0:e1] == y.astype(np.int16)).all()
        assert (lab["ms"][e0:e1] == ms_abs).all()
        ms_rel = (ms_abs - start)
        src, mk = stride_sample(torch.tensor([0, n], dtype=torch.int64), CAP)
        samp = src[0][mk[0]].numpy()
        assert (lab["samp_local"][lab["samp_start"][s]:lab["samp_start"][s + 1]] == samp).all()
        kind_a, part_a, depth_a = lab["src_kind"][e0:e1], lab["src_part"][e0:e1], lab["src_depth"][e0:e1]
        fip = bool(lab["fingertip_in_front"][s])
        px, py = torch.from_numpy(x.copy()).float(), torch.from_numpy(y.copy()).float()
        d_all = torch.stack([(px - g["cx"]) / g["fx"], (py - g["cy"]) / g["fy"], torch.ones_like(px)], dim=-1)   # (n, 3)
        verts = _regime_grids(s)
        vn = vertex_normals(verts, g["faces"])
        idx = samp if mode == "sampled" else np.arange(n)
        st = "sampled" if mode == "sampled" else "all"          # the set the association is actually run on
        if len(idx) == 0:
            continue
        res_by_reg = {}
        for rg in REGIMES:
            dp, mask, vb, nb, band, front, take = _batch(idx, ms_rel, d_all, rg, verts, vn)
            lib = associate_rays(dp, mask, vb, g["faces"], g["vert_faces"], band=band, front_tol=front,
                                 z_tol=Z_TOL, k=KNN, chunk=CHUNK)
            cur, alt, altf = associate_dual(dp, mask, vb, g["faces"], g["vert_faces"], nb, band, front, Z_TOL,
                                            KNN, GAP, CHUNK)
            lib = {k_: take(v_) for k_, v_ in lib.items()}
            cur = {k_: take(v_) for k_, v_ in cur.items()}
            alt = {k_: take(v_) for k_, v_ in alt.items()}
            altf = {k_: take(v_) for k_, v_ in altf.items()}
            # ---- consistency self-checks (every call)
            m = len(idx)
            bump(f"{st}/{rg}/n_events", m)
            for key in ("vid", "A", "q", "z", "valid", "hit"):
                ne = (lib[key] != cur[key]).reshape(m, -1).any(1)
                bump(f"{st}/{rg}/lib_vs_cur_mismatch_{key}", ne.sum())
            nt = ~alt["ctr"]
            bump(f"{st}/{rg}/alt_n_ctr", alt["ctr"].sum())
            bump(f"{st}/{rg}/alt_n_noctr", nt.sum())
            bump(f"{st}/{rg}/altf_n_ctr", altf["ctr"].sum())
            wc = lib["q"].unsqueeze(1) * lib["A"]
            zc = lib["z"].unsqueeze(1).expand(m, R_MAX)
            bad = ((alt["vid"][:, :R_MAX] != lib["vid"]).any(1) | (alt["w"][:, :R_MAX] != wc).any(1) |
                   (alt["zr"][:, :R_MAX] != zc).any(1) | (alt["w"][:, R_MAX:] != 0).any(1) |
                   (alt["r"] != (1.0 - lib["q"])) | (alt["valid"] != lib["valid"]) | (alt["hit"] != lib["hit"]))
            bump(f"{st}/{rg}/alt_vs_cur_mismatch_on_noctr", (bad & nt).sum())
            bump(f"{st}/{rg}/altf_vs_alt_differ", ((altf["w"] != alt["w"]).any(1) | (altf["vid"] != alt["vid"]).any(1)
                                                     | (altf["valid"] != alt["valid"])).sum())
            bump(f"{st}/{rg}/altf_ctr_differs_from_alt", (altf["ctr"] != alt["ctr"]).sum())
            bump(f"{st}/{rg}/alt_ctr_mass_err_gt1e-5", (alt["ctr"] & ((alt["w"].sum(1) - 1.0).abs() > 1e-5)).sum())
            summ = {
                "cur": summarize(lib["vid"], wc, zc, 1.0 - lib["q"], lib["valid"], lib["hit"],
                                 torch.zeros(m, dtype=torch.bool), g["hard"]),
                "alt": summarize(alt["vid"], alt["w"], alt["zr"], alt["r"], alt["valid"], alt["hit"], alt["ctr"], g["hard"]),
                "altf": summarize(altf["vid"], altf["w"], altf["zr"], altf["r"], altf["valid"], altf["hit"], altf["ctr"], g["hard"]),
            }
            res_by_reg[rg] = summ
            kd, pt, dp_ = kind_a[idx], part_a[idx], depth_a[idx]
            if rg == "R3":          # end-to-end geometry check: interior events must hit the surface the label saw
                ki = (kd == 0)
                sc = summ["cur"]
                bump(f"{st}/R3/interior_n", ki.sum())
                bump(f"{st}/R3/interior_hit", (ki & sc["hit"]).sum())
                bump(f"{st}/R3/interior_hit_and_z_within_0p1mm", (ki & sc["hit"] & (np.abs(sc["zhit"] - dp_) < 1e-4)).sum())
            for ru in RULES:
                tabs[(st, rg, ru)].add(kd, pt, dp_, fip, summ[ru])
                if mode == "both":      # the sampled set is a subset of the all-event run (rows are independent)
                    sub = {k_: v_[samp] for k_, v_ in summ[ru].items()}
                    tabs[("sampled", rg, ru)].add(kind_a[samp], part_a[samp], depth_a[samp], fip, sub)
        sel = np.arange(len(idx)) if mode == "sampled" else samp
        for rg in REGIMES:
            for ru in RULES:
                for f in FIELDS:
                    per_event.setdefault(f"{rg}_{ru}_{f}", []).append(
                        np.asarray(res_by_reg[rg][ru][f])[sel].astype(FDT[f]))
        if j == 0 and (ii + 1) % max(1, len(steps) // 10) == 0:
            print(f"  worker0: {ii + 1}/{len(steps)} steps  {time.time() - t0:.0f}s", flush=True)
    per_event = {k: np.concatenate(v) for k, v in per_event.items()}
    return dict(steps=np.asarray(steps), tabs=tabs, chk=chk, per_event=per_event)


# ------------------------------------------------------------------ driver
def run_seq(seq, legacy_dir, a):
    lab_path = OUT_DIR / f"labels_{seq}.npz"
    z = np.load(lab_path)
    lab = {k: z[k] for k in z.files}
    ev, off, tsub, aux, pos51, betas, K = load_seq_bundle(seq, legacy_dir)
    run_id, ends, prev = enumerate_steps(aux)
    S_all = len(ends)
    assert S_all == len(lab["step_end_ms"]) and (lab["step_end_ms"] == ends).all(), "labels do not match the evaluator steps"
    noise = np.random.default_rng(0).standard_normal((S_all, 51)).astype(np.float32)
    steps = np.arange(S_all) if a.max_steps is None else np.arange(min(a.max_steps, S_all))
    # interleaved static partition (balances heavy windows)
    nproc = min(a.procs, len(steps))
    tasks = [(j, steps[j::nproc].tolist()) for j in range(nproc)]
    print(f"[{seq}] steps={len(steps)}/{S_all} events(all)={int(lab['step_ev_start'][len(steps)])} mode={a.mode} procs={nproc}",
          flush=True)
    t0 = time.time()
    ctx = mp.get_context("fork")
    _init(seq, legacy_dir, a.mode, lab, noise)            # in the parent: workers inherit by fork
    results = []
    with ctx.Pool(nproc, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        for r in pool.imap_unordered(run_task, tasks):
            results.append(r)
            print(f"  collected {len(results)}/{nproc}  {time.time() - t0:.0f}s", flush=True)
    # ---- merge
    tabs, chk = {}, {}
    for r in results:
        for k, t in r["tabs"].items():
            if k not in tabs:
                tabs[k] = Table()
            tabs[k].merge(t)
        for k, v in r["chk"].items():
            chk[k] = chk.get(k, 0) + v
    # per-event compact arrays, back in step order
    order_steps = np.concatenate([r["steps"] for r in results])
    n_s = np.diff(lab["samp_start"])[order_steps] if len(order_steps) else np.zeros(0, int)
    per_event = {}
    if results[0]["per_event"]:
        # each task's arrays are concatenated over its own steps in the order of r["steps"]
        keys = results[0]["per_event"].keys()
        segs = []                                                  # (step, task, start, stop)
        for ti, r in enumerate(results):
            ns = np.diff(lab["samp_start"])[r["steps"]]
            c = np.concatenate([[0], np.cumsum(ns)])
            for si, s_ in enumerate(r["steps"]):
                segs.append((int(s_), ti, int(c[si]), int(c[si + 1])))
        segs.sort()
        for k in keys:
            per_event[k] = np.concatenate([results[ti]["per_event"][k][a_:b_] for _, ti, a_, b_ in segs])
        per_event["samp_step"] = np.concatenate([np.full(b_ - a_, s_, np.int32) for s_, _, a_, b_ in segs])
        per_event["samp_lab_pos"] = np.concatenate(
            [lab["step_ev_start"][s_] + lab["samp_local"][lab["samp_start"][s_]:lab["samp_start"][s_ + 1]]
             for s_, _, _, _ in segs]).astype(np.int64)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = {}
    for (st, rg, ru), t in tabs.items():
        flat[f"{st}|{rg}|{ru}|cnt"] = t.cnt
        flat[f"{st}|{rg}|{ru}|hz"] = t.hz
        flat[f"{st}|{rg}|{ru}|conf"] = t.conf
    np.savez(out_dir / f"assoc_tables_{seq}.npz", **flat)
    (out_dir / f"assoc_checks_{seq}.json").write_text(json.dumps(
        dict(seq=seq, mode=a.mode, n_steps=int(len(steps)), checks=chk, cnt_names=CNT, shape=dict(
            NFIP=NFIP, NKIND=NKIND, NGRP=NGRP, NAT=NAT, NCTR=NCTR, NBIN=NBIN, BIN_MM=BIN_MM, ZRANGE=ZRANGE,
            HA=HA, HB=HB, HC=HC)), indent=1, sort_keys=True))
    if per_event and not a.no_events:
        np.savez(out_dir / f"assoc_events_{seq}.npz", **per_event)
    print(f"[{seq}] done in {time.time() - t0:.0f}s  -> {out_dir}", flush=True)
    return chk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default=None)
    ap.add_argument("--mode", choices=["sampled", "both"], default="sampled")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--procs", type=int, default=100)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--no-events", action="store_true")
    a = ap.parse_args()
    for seq, legacy_dir in sequences():
        if a.seq in (None, seq):
            chk = run_seq(seq, legacy_dir, a)
            bad = {k: v for k, v in chk.items() if "mismatch" in k and v}
            print(f"[{seq}] consistency: nonzero mismatch counters: {bad if bad else 'none'}", flush=True)


if __name__ == "__main__":
    main()
