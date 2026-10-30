#!/usr/bin/env python3
"""AURA Stage B: the association as an operator, zero training, CPU only (docs/EXPERIMENT_RECORDS.md#aura-prereg-20261024).

For every packet (evaluator window, the model's own <= 2048 stride sample) and every anchor regime, every association
rule is run on the anchor mesh and scored against per-event source labels rendered from the GT mesh of each event's
millisecond (the XYZ-ASSOC P1 rules, `tools/xcl/sources/step1_source_labels.py`: interior / silhouette / depth
discontinuity / non-hand, part, depth) and the XAG source faces (material points). Statistics are accumulated as sums and
fixed-bin histograms, merged over processes, and summarised by `summarize.py b`.

    sets      zgz (all 2590 endpoints), sel (ycy / ylf every 4th), fit (fit subjects every 20th)
    regimes   R3 (each event's own-ms GT mesh), E3 (GT at the previous endpoint), noise (C1 mixed, rng 777 in the C
              evaluation order), E2 (C1 strict trace incoming, zgz), R3s / E3s (meshes scaled about the wrist by the
              SBA sigma*, zgz), sweeps from E3 (zgz, sel): camera x / y / z translation 5 / 10 / 20 / 40 mm,
              in-plane rotation about the wrist 0.05 / 0.1 / 0.2 rad, scale about the wrist 0.94 / 1.06, finger pose
              noise 0.05 / 0.1 / 0.2 rad
    rules     c1 surf b8 b32 mM m10 m20 aura10 aura20 p2dvis p2dh vis ctr A1

    python tools/aura/b_operator.py --set zgz --procs 48 --out outputs/aura/b
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model"), str(ROOT / "tools"), str(ROOT / "tools" / "xcl" / "sources")]

RULES_B = ("c1", "surf", "b8", "b32", "mM", "m10", "m20", "aura10", "aura20", "p2dvis", "p2dh", "vis", "ctr", "A1")
SWEEPS = ([("tx%d" % m, "t", 0, m) for m in (5, 10, 20, 40)] + [("ty%d" % m, "t", 1, m) for m in (5, 10, 20, 40)]
          + [("tz%d" % m, "t", 2, m) for m in (5, 10, 20, 40)]
          + [("rot%s" % a, "rot", None, float(a) / 100) for a in ("5", "10", "20")]
          + [("sc94", "scale", None, 0.94), ("sc106", "scale", None, 1.06)]
          + [("fp%s" % a, "finger", None, float(a) / 100) for a in ("5", "10", "20")])
REGIMES = {"zgz": ["R3", "E3", "noise", "E2", "R3s", "E3s"] + [s[0] for s in SWEEPS],
           "sel": ["R3", "E3", "noise"] + [s[0] for s in SWEEPS],
           "fit": ["R3", "E3", "noise"]}
SIGMA_STAR = {"zgz_local": 0.94, "zgz_global": 0.96}                                  # SBA D, outputs/sba/d/result.json
DOMS = ("local", "global")
# histogram layouts
QB, DB, DMAX = 200, 256, 64.0                                                       # q bins, distance bins (px)
EB, EMAX = 300, 150.0                                                               # depth error bins (mm), +-150
MB, MMAX = 200, 100.0                                                               # material-point error (mm)
SB = 101                                                                            # direction score bins in [-1, 1]
UBINS = (1.0, 2.0, 4.0, 8.0, np.inf)                                               # |normal displacement| strata (px)
DBINS = (0, 2, 4, 8, 16, 32, np.inf)                                               # detection distance strata (px)
LOCAL_BINS = 128                                                                    # shuffle partner distance (px)
FINGER = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
CONTOUR_COS = 0.4

COUNT_KEYS = ("live", "hand", "hand_valid", "hand_q", "non", "non_valid", "non_q", "mesh_mass", "mesh_mass_non",
              "agree_exact", "agree_finger", "fc", "fc_mism", "near", "near_xtalk", "O_sum", "packets",
              "own", "own_cov", "dir_elig", "dir_cov", "matpt_n", "depth_n")


def new_stats():
    n = len(DOMS)
    return {"cnt": {k: np.zeros(n) for k in COUNT_KEYS},
            "kind_valid": np.zeros((n, 4, 2)), "kind_n": np.zeros((n, 4)), "kind_agree": np.zeros((n, 4)),
            "dist_valid": np.zeros((n, 2, len(DBINS) - 1, 2)),
            "q_hist": np.zeros((n, 2, QB + 1)), "dist_hist": np.zeros((n, 2, DB + 1)),
            "depth_hist": np.zeros((n, EB + 2)), "bind_hist": np.zeros((n, EB + 2)), "matpt_hist": np.zeros((n, MB + 1)),
            "dir_hist": np.zeros((n, len(UBINS) - 1, 2, SB))}


def merge(a, b):
    for k, v in b.items():
        if k == "cnt":
            for kk, vv in v.items():
                a["cnt"][kk] += vv
        else:
            a[k] += v
    return a


# ------------------------------------------------------------------------------------------------ geometry helpers
def scale2d(verts, wrist, s):
    """Scale the projection about the wrist's projection by s, depth unchanged (SBA family G)."""
    z = verts[..., 2:3]
    xn = verts[..., :2] / z
    xw = (wrist[..., :2] / wrist[..., 2:3]).unsqueeze(-2)
    return torch.cat([(xw + s * (xn - xw)) * z, z], -1)


def rotz(verts, wrist, a):
    """Rotate about the camera z axis through the wrist (an in-plane rotation of the hand)."""
    c, s = float(np.cos(a)), float(np.sin(a))
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=verts.dtype)
    w = wrist.unsqueeze(-2)
    return (verts - w) @ R.T + w


def hist_idx(x, lo, hi, nb):
    return np.clip(np.floor((x - lo) / (hi - lo) * nb), 0, nb).astype(np.int64)


# ------------------------------------------------------------------------------------------------ worker
_G = {}


def _init(set_name):
    torch.set_num_threads(1)
    import step1_source_labels as L                                                  # noqa: F401 (CPU only)
    from step1_common import get_mano, face_part_table
    from semkine.xyz_mesh import vertex_face_table
    mano = get_mano()
    hard, fpart, _ = face_part_table(mano)
    _G.update(set=set_name, mano=mano, faces=mano.f.long(), faces_np=mano.f.numpy().astype(np.int64),
              fpart=fpart, hard=torch.as_tensor(hard), lbs_w=mano.weights.float(),
              vert_faces=vertex_face_table(mano.f.long(), 778), seqs={})


def _seq(name, d):
    from tools.agg778 import common as C
    if name not in _G["seqs"]:
        _G["seqs"][name] = C.Seq(name, d)
    return _G["seqs"][name]


def labels_for(sq, end, ev5, sigma=None):
    """P1 labels (part, kind, depth) of the window's events from each event's own-ms GT mesh; also returns the 50
    meshes (optionally scaled about the wrist by sigma) and the wrist of each ms."""
    import step1_source_labels as L
    from step1_common import fk_batch
    start = end - 49
    off = sq.offsets
    counts = np.diff(off[start:end + 2]).astype(np.int64)
    verts, joints = fk_batch(_G["mano"], torch.tensor(sq.betas).view(1, -1), sq.pos51[start:end + 1])
    if sigma is not None:
        verts = scale2d(verts, joints[:, 0], sigma)
    ex, ey = ev5[:, 1].astype(np.int64), ev5[:, 2].astype(np.int64)
    n = len(ev5)
    part, kind = np.full(n, -1, np.int8), np.full(n, -1, np.int8)
    depth = np.full(n, np.nan, np.float32)
    cs = np.concatenate([[0], np.cumsum(counts)])
    fxyc = tuple(float(x) for x in _G["intr_of"](sq))
    for i in range(50):
        if counts[i] == 0:
            continue
        Z, P, _ = L.rasterize(verts[i].numpy().astype(np.float64), _G["faces_np"], _G["fpart"], fxyc)
        lp, lk, ld, _ = L.label_maps(Z, P)
        sl = slice(cs[i], cs[i + 1])
        part[sl], kind[sl], depth[sl] = lp[ey[sl], ex[sl]], lk[ey[sl], ex[sl]], ld[ey[sl], ex[sl]]
    ms_rel = np.repeat(np.arange(50), counts)
    return part, kind, depth, verts, joints[:, 0], ms_rel


def intr_of(sq):
    s = 0.375
    K = torch.tensor(sq.K, dtype=torch.float32)
    return K[0, 0] * s, K[1, 1] * s, K[0, 2] * s, K[1, 2] * s


def anchor_meshes(sq, it, regime, noise_vec, idx):
    """(verts (1, V, 3), wrist (1, 3)) of one anchor regime (except R3 / R3s)."""
    from step1_common import fk_batch
    beta = torch.tensor(sq.betas).view(1, -1)
    p0 = it["gt_prev"].astype(np.float32).copy()
    if regime == "noise":
        p = p0 + noise_vec
    elif regime == "E2":
        p = _G["trace"][it["seq"]][it["k"]].astype(np.float32)
    else:
        p = p0.copy()
    sw = next((s for s in SWEEPS if s[0] == regime), None)
    rng = np.random.default_rng(4242 + idx)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    if sw is not None and sw[1] == "t":
        p[sw[2]] += sign * sw[3] * 1e-3
    if sw is not None and sw[1] == "finger":
        p[6:51] += np.random.default_rng(5151 + idx).standard_normal(45).astype(np.float32) * sw[3]
    v, j = fk_batch(_G["mano"], beta, p[None])
    w = j[:, 0]
    if regime == "E3s":
        v = scale2d(v, w, SIGMA_STAR[it["seq"]])
    if sw is not None and sw[1] == "rot":
        v = rotz(v, w, sign * sw[3])
    if sw is not None and sw[1] == "scale":
        v = scale2d(v, w, sw[3])
    return v.float(), w


def run_rules(d, mask, verts, intr, src, part_t):
    """All association rules on one batch (B packets / ms groups) of sampled events."""
    from semkine import aura as AU
    from semkine.xag import route_association
    from semkine.xyz_mesh import associate_rays, vertex_normals
    faces, vf = _G["faces"], _G["vert_faces"]
    f_px = (intr[0] * intr[1]).sqrt()
    out = {}
    vis = AU.visible(verts, intr)
    c1 = AU.associate("c1", d, mask, verts, faces, vf, intr)
    for rule in RULES_B:
        if rule == "c1":
            out[rule] = c1
        elif rule == "ctr":
            out[rule] = associate_rays(d, mask, verts, faces, vf, band=16.0 / f_px, front_tol=3.0 / f_px, z_tol=0.01,
                                       k=16, mode="contour", normals=vertex_normals(verts, faces), gap=0.01)
        elif rule == "A1":
            out[rule], _ = route_association("part_route", c1, d, mask, verts, faces, vf, _G["lbs_w"], 16.0 / f_px,
                                             3.0 / f_px, 0.01, source_part=src["part"], source_face=src["face"],
                                             source_bary=src["bary"], source_valid=src["valid"])
        else:
            out[rule] = AU.associate(rule, d, mask, verts, faces, vf, intr, c1=c1, vis=vis)
    return out, vis


def score(st, dom, a, d, mask, verts, intr, lab, src_pt, vis, owner, u_norm, contour):
    """Accumulate one batch of one rule into stats `st` (domain index `dom`). lab: part / kind / depth (B, N);
    src_pt (B, N, 3) GT material point on the anchor mesh (nan where unknown); owner (B, V) bool, u_norm (B, V) px."""
    c = st["cnt"]
    hard = _G["hard"]
    B, N = mask.shape
    R = a["vid"].shape[-1]
    w = a["q"].unsqueeze(-1) * a["A"] * mask.float().unsqueeze(-1)                   # (B, N, R)
    part, kind, depth = lab
    live = mask
    hand = live & (part >= 0)
    non = live & (part < 0)
    valid = a["valid"] & live
    q = a["q"]
    c["live"][dom] += float(live.sum())
    c["hand"][dom] += float(hand.sum())
    c["hand_valid"][dom] += float((hand & valid).sum())
    c["hand_q"][dom] += float(q[hand].sum())
    c["non"][dom] += float(non.sum())
    c["non_valid"][dom] += float((non & valid).sum())
    c["non_q"][dom] += float(q[non].sum())
    c["mesh_mass"][dom] += float(w.sum())
    c["mesh_mass_non"][dom] += float(w[non].sum())
    # P1 part prediction: mass per hard joint, argmax; palm majority
    jm = torch.zeros(B, N, 16).scatter_add_(2, hard[a["vid"]], w.float())
    tot = jm.sum(-1)
    has = tot > 0
    pred = torch.where(has, jm.argmax(-1), torch.full_like(tot, -1, dtype=torch.long))
    maj0 = has & (jm[..., 0] > 0.5 * tot)
    pl = part.long()
    ag = hand & (pred == pl)
    agf = hand & (pred >= 0) & (torch.as_tensor(FINGER)[pred.clamp_min(0)] == torch.as_tensor(FINGER)[pl.clamp_min(0)])
    c["agree_exact"][dom] += float(ag.sum())
    c["agree_finger"][dom] += float(agf.sum())
    fc = live & (kind >= 1) & (part >= 1)
    c["fc"][dom] += float(fc.sum())
    c["fc_mism"][dom] += float((fc & maj0).sum())
    for k in range(-1, 3):
        sel = live & (kind == k)
        st["kind_n"][dom, k + 1] += float(sel.sum())
        st["kind_valid"][dom, k + 1, 0] += float((sel & valid).sum())
        st["kind_agree"][dom, k + 1] += float((sel & ag).sum())
    # crosstalk: near events whose records span >= 2 finger groups
    near = valid & ~a["hit"]
    fg = torch.as_tensor(FINGER)[hard[a["vid"]]]
    used = w > 0
    fmin = torch.where(used, fg, torch.full_like(fg, 99)).min(-1).values
    fmax = torch.where(used, fg, torch.full_like(fg, -1)).max(-1).values
    c["near"][dom] += float(near.sum())
    c["near_xtalk"][dom] += float((near & (fmax > fmin)).sum())
    # detection histograms: q and distance to the nearest projected anchor vertex
    uv = _G["AU"].project(verts, intr)
    px = torch.stack([d[..., 0] * intr[0][:, None] + intr[2][:, None], d[..., 1] * intr[1][:, None] + intr[3][:, None]], -1)
    dist = torch.cdist(px, uv).min(-1).values
    for ci, sel in enumerate((hand, non)):
        qs = q[sel].numpy()
        st["q_hist"][dom, ci] += np.bincount(hist_idx(qs, 0.0, 1.0, QB), minlength=QB + 1)[:QB + 1]
        ds_ = dist[sel].numpy()
        st["dist_hist"][dom, ci] += np.bincount(hist_idx(ds_, 0.0, DMAX, DB), minlength=DB + 1)[:DB + 1]
        bins = np.digitize(ds_, DBINS[1:-1])
        vv = valid[sel].numpy()
        st["dist_valid"][dom, ci, :, 0] += np.bincount(bins, minlength=len(DBINS) - 1)
        st["dist_valid"][dom, ci, :, 1] += np.bincount(bins, weights=vv.astype(np.float64), minlength=len(DBINS) - 1)
    # depth: the dominant hypothesis's z (contour: front records when they carry more mass)
    z = a["z"]
    if z.dim() == 3:
        mh, mf = w[..., :R // 2].sum(-1), w[..., R // 2:].sum(-1)
        z = torch.where(a.get("ctr", torch.zeros_like(w, dtype=torch.bool)).any(-1) & (mf > mh), z[..., R // 2], z[..., 0])
    fin = hand & valid & torch.isfinite(depth)
    e = (z[fin] - depth[fin]).numpy() * 1000.0
    st["depth_hist"][dom] += np.bincount(hist_idx(e, -EMAX, EMAX, EB), minlength=EB + 2)[:EB + 2]
    c["depth_n"][dom] += float(fin.sum())
    # material point on the anchor mesh vs the GT source's material point there
    An = a["A"] * (w > 0).float()
    An = An / An.sum(-1, keepdim=True).clamp_min(1e-12)
    Vr = verts[torch.arange(B).view(B, 1, 1), a["vid"]]
    X = (An.unsqueeze(-1) * Vr).sum(2)
    okm = hand & valid & torch.isfinite(src_pt).all(-1)
    em = (X[okm] - src_pt[okm]).norm(dim=-1).numpy() * 1000.0
    st["matpt_hist"][dom] += np.bincount(hist_idx(em, 0.0, MMAX, MB), minlength=MB + 1)[:MB + 1]
    c["matpt_n"][dom] += float(okm.sum())
    # binding depth: z against the depth of the GT material point's closest point on the event ray
    zb = (z[okm] - (src_pt[okm] * d[okm]).sum(-1) / (d[okm] * d[okm]).sum(-1)).numpy() * 1000.0
    st["bind_hist"][dom] += np.bincount(hist_idx(zb, -EMAX, EMAX, EB), minlength=EB + 2)[:EB + 2]
    # coverage and direction on GT-owning visible contour vertices
    V = verts.shape[1]
    m_hit = torch.zeros(B * V).index_add(0, (torch.arange(B).view(B, 1, 1) * V + a["vid"]).reshape(-1),
                                         (w * a["hit"].unsqueeze(-1).float()).reshape(-1)).view(B, V)
    m_all = torch.zeros(B * V).index_add(0, (torch.arange(B).view(B, 1, 1) * V + a["vid"]).reshape(-1),
                                         w.reshape(-1)).view(B, V)
    m_near = m_all - m_hit
    c["O_sum"][dom] += float((m_all > 0).sum())
    c["packets"][dom] += float(B)
    own = owner & vis & contour
    c["own"][dom] += float(own.sum())
    c["own_cov"][dom] += float((own & (m_all > 0)).sum())
    elig = own & (u_norm.abs() > UBINS[0])
    cov = elig & (m_all > 0)
    c["dir_elig"][dom] += float(elig.sum())
    c["dir_cov"][dom] += float(cov.sum())
    s = ((m_near - m_hit) / (m_near + m_hit).clamp_min(1e-12))[cov].numpy()
    u = u_norm[cov].numpy()
    ub = np.digitize(np.abs(u), UBINS[1:-1])
    sb = hist_idx(s, -1.0, 1.0, SB - 1)
    for sg, sel in ((1, u > 0), (0, u <= 0)):
        np.add.at(st["dir_hist"][dom], (ub[sel], sg, sb[sel]), 1.0)


def run_task(task):
    from semkine import aura as AU
    from semkine.xyz_mesh import stride_sample, vertex_normals
    from tools.agg778 import common as C
    _G["AU"] = AU
    _G["intr_of"] = intr_of
    set_name, idxs, noise, seq_dirs = task
    if "trace" not in _G and set_name == "zgz":
        _G["trace"] = C.strict_trace()
    stats = {}
    loc = {"G": {"dist": np.zeros(LOCAL_BINS + 1), "same_part": 0.0, "same_finger": 0.0, "self": 0.0, "n": 0.0},
           "P": {"dist": np.zeros(LOCAL_BINS + 1), "same_part": 0.0, "same_finger": 0.0, "self": 0.0, "n": 0.0}}
    chk = {"p1_fc": np.zeros(2), "p1_mism": np.zeros(2), "p1_hand": np.zeros(2), "p1_agree": np.zeros(2)}
    ds = _G["ds"]
    t0 = time.time()
    for ii, idx in enumerate(idxs):
        it = ds[idx]
        sq = _seq(it["seq"], seq_dirs[it["seq"]])
        dom = 0 if "_local" in it["seq"] else 1
        ev5 = it["ev5"]
        n = len(ev5)
        if n == 0:
            continue
        src_i, mk = stride_sample(torch.tensor([0, n], dtype=torch.int64), 2048)
        samp = src_i[0][mk[0]].numpy()
        part, kind, depth, verts_ms, wrist_ms, ms_rel = labels_for(sq, it["end"], ev5)
        if set_name == "zgz":                                    # labels of the sigma*-scaled GT meshes (R3s / E3s)
            part_s, kind_s, depth_s, verts_ms_s, _, _ = labels_for(sq, it["end"], ev5, sigma=SIGMA_STAR[it["seq"]])
        fx, fy, cx, cy = intr_of(sq)
        px = torch.as_tensor(ev5[samp, 1], dtype=torch.float32)
        py = torch.as_tensor(ev5[samp, 2], dtype=torch.float32)
        dsamp = torch.stack([(px - cx) / fx, (py - cy) / fy, torch.ones_like(px)], -1)
        m = len(samp)
        lab = tuple(torch.as_tensor(x[samp]).view(1, m) for x in (part, kind, depth))
        lab_s = tuple(torch.as_tensor(x[samp]).view(1, m) for x in (part_s, kind_s, depth_s)) if set_name == "zgz" \
            else None
        s = it["src"]
        src = {"part": torch.as_tensor(s["xag_part"][samp]).long().view(1, m),
               "face": torch.as_tensor(s["xag_face"][samp]).long().view(1, m),
               "bary": torch.as_tensor(s["xag_bary"][samp]).float().view(1, m, 3),
               "valid": torch.as_tensor(s["xag_valid"][samp]).bool().view(1, m)}
        from step1_common import fk_batch
        vgt_end, _ = fk_batch(_G["mano"], torch.tensor(sq.betas).view(1, -1), it["gt_end"][None].astype(np.float32))
        regimes = REGIMES[set_name]
        for regime in regimes:
            labr = lab_s if regime in ("R3s", "E3s") else lab
            if regime in ("R3", "R3s"):
                vb = verts_ms_s if regime == "R3s" else verts_ms
                ml = ms_rel[samp]
                cnts = np.bincount(ml, minlength=50)
                Nm = max(int(cnts.max()), 1)
                colp = np.arange(m) - (np.cumsum(cnts) - cnts)[ml]
                d_b = torch.zeros(50, Nm, 3)
                d_b[torch.as_tensor(ml), torch.as_tensor(colp)] = dsamp
                mask_b = torch.zeros(50, Nm, dtype=torch.bool)
                mask_b[torch.as_tensor(ml), torch.as_tensor(colp)] = True
                intr_b = tuple(x.reshape(1).expand(50).contiguous() for x in (fx, fy, cx, cy))
                rearr = lambda x, fill: _scatter_ms(x, ml, colp, Nm, fill)              # noqa: E731
                lab_b = (rearr(labr[0][0], -1), rearr(labr[1][0], -1), rearr(labr[2][0], float("nan")))
                src_b = {k: rearr(v[0], 0 if k != "valid" else False) for k, v in src.items()}
                verts_r = vb.float()
            else:
                verts_r, _ = anchor_meshes(sq, it, regime, noise.get(idx), idx)
                d_b, mask_b = dsamp.view(1, m, 3), torch.ones(1, m, dtype=torch.bool)
                intr_b = tuple(x.reshape(1) for x in (fx, fy, cx, cy))
                lab_b, src_b = labr, src
            res, vis = run_rules(d_b, mask_b, verts_r, intr_b, src_b, lab_b[0])
            # GT material point on the anchor mesh, owners, normal displacement, contour vertices
            Bn = verts_r.shape[0]
            fsrc = _G["faces"][src_b["face"].clamp_min(0)]                         # (B, N, 3)
            Vg = verts_r[torch.arange(Bn).view(Bn, 1, 1), fsrc]
            src_pt = (src_b["bary"].unsqueeze(-1) * Vg).sum(2)
            src_pt = torch.where((src_b["valid"] & mask_b).unsqueeze(-1), src_pt, torch.full_like(src_pt, float("nan")))
            owner = torch.zeros(Bn, 778, dtype=torch.bool)
            okf = src_b["valid"] & mask_b & (lab_b[0] >= 0)
            bidx = torch.arange(Bn).view(Bn, 1, 1).expand_as(fsrc)[okf]
            owner[bidx.reshape(-1), fsrc[okf].reshape(-1)] = True
            nrm = vertex_normals(verts_r, _G["faces"])
            dhat = verts_r / verts_r.norm(dim=-1, keepdim=True)
            contour = (nrm * dhat).sum(-1).abs() < CONTOUR_COS
            X, Y, Z = verts_r[..., 0], verts_r[..., 1], verts_r[..., 2].clamp_min(1e-6)
            exn = intr_b[0][:, None] * (nrm[..., 0] / Z - nrm[..., 2] * X / Z ** 2)
            eyn = intr_b[1][:, None] * (nrm[..., 1] / Z - nrm[..., 2] * Y / Z ** 2)
            e1 = torch.stack([exn, eyn], -1)
            e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            duv = AU.project(vgt_end.expand(Bn, -1, -1), intr_b) - AU.project(verts_r, intr_b)
            u_norm = (e1 * duv).sum(-1)
            for rule, a in res.items():
                key = (regime, rule)
                if key not in stats:
                    stats[key] = new_stats()
                score(stats[key], dom, a, d_b, mask_b, verts_r, intr_b, lab_b, src_pt, vis, owner, u_norm, contour)
            if regime == "E3":
                # P1 toolchain check counters (c1) and shuffle locality (zgz)
                a = res["c1"]
                w = a["q"].unsqueeze(-1) * a["A"]
                jm = torch.zeros(1, m, 16).scatter_add_(2, _G["hard"][a["vid"]], w.float())
                tot = jm.sum(-1)
                has = tot > 0
                pred = torch.where(has, jm.argmax(-1), torch.full_like(tot, -1, dtype=torch.long))
                fcm = (lab[1] >= 1) & (lab[0] >= 1)
                chk["p1_fc"][dom] += float(fcm.sum())
                chk["p1_mism"][dom] += float((fcm & has & (jm[..., 0] > 0.5 * tot)).sum())
                hd = lab[0] >= 0
                chk["p1_hand"][dom] += float(hd.sum())
                chk["p1_agree"][dom] += float((hd & (pred == lab[0].long())).sum())
                seeds = torch.tensor([int(it["si"]) * 1000003 + int(it["k"])])
                for nm, part_of in (("G", None), ("P", _G["hard"])):
                    sh, info = AU.shuffle_assoc(a, d_b, mask_b, verts_r, seeds, part_of)
                    L_ = AU.shuffle_locality(a, info, d_b, intr_b, _G["hard"], torch.as_tensor(FINGER))
                    loc[nm]["dist"] += np.bincount(hist_idx(L_["dist_px"].numpy(), 0.0, 128.0, LOCAL_BINS),
                                                   minlength=LOCAL_BINS + 1)[:LOCAL_BINS + 1]
                    for k_ in ("same_part", "same_finger", "self"):
                        loc[nm][k_] += float(L_[k_].sum())
                    loc[nm]["n"] += float(L_["dist_px"].numel())
        if ii % 50 == 0:
            print(set_name, "task progress", ii, len(idxs), f"{time.time() - t0:.0f} s", flush=True)
    return stats, loc, chk


def _scatter_ms(x, ml, colp, Nm, fill):
    out = torch.full((50, Nm, *x.shape[1:]), fill, dtype=x.dtype)
    out[torch.as_tensor(ml), torch.as_tensor(colp)] = x
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=tuple(REGIMES), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    from tools.agg778 import common as C
    from tools.aura import common as AC
    if not a.smoke:
        AC.verify_protocol("B")
    seqs = {"zgz": C.zgz_seqs, "sel": lambda: C.subject_seqs(C.SELECT_SUBJ),
            "fit": lambda: C.subject_seqs(C.FIT_SUBJ)}[a.set]()
    stride = {"zgz": 1, "sel": 4, "fit": 20}[a.set]
    ds = C.Endpoints(seqs, stride=stride)
    if a.smoke:
        ds.index = ds.index[:2] + ds.index[-2:]
    # the noise anchors of the C evaluation order (rng 777 over the set), so B and C see the same noisy anchors
    from config import load_config
    cfg = load_config(str(C.REPO / C.MODELS["A0"]["config"]))
    draw = C.noise_sampler(cfg)
    rng = np.random.default_rng(777)
    noise_all = {i: draw(rng)[0] for i in range(len(ds))}
    seq_dirs = dict(seqs)
    _G["ds"] = ds
    n = len(ds)
    chunks = [list(range(i, n, a.procs * 4)) for i in range(min(n, a.procs * 4))]
    tasks = [(a.set, ch, {i: noise_all[i] for i in ch}, seq_dirs) for ch in chunks if ch]
    t0 = time.time()
    ctx = mp.get_context("fork")
    stats, loc, chk = {}, None, None
    with ctx.Pool(a.procs, initializer=_init, initargs=(a.set,)) as pool:
        for k, (st, lc, ck) in enumerate(pool.imap_unordered(run_task, tasks)):
            for key, v in st.items():
                stats[key] = merge(stats[key], v) if key in stats else v
            if loc is None:
                loc, chk = lc, ck
            else:
                for nm in loc:
                    for k_ in loc[nm]:
                        loc[nm][k_] = loc[nm][k_] + lc[nm][k_]
                for k_ in chk:
                    chk[k_] += ck[k_]
            print(f"{k + 1}/{len(tasks)} tasks {time.time() - t0:.0f} s", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    flat = {}
    for (regime, rule), st in stats.items():
        for k, v in st.items():
            if k == "cnt":
                for kk, vv in v.items():
                    flat[f"{regime}|{rule}|cnt|{kk}"] = vv
            else:
                flat[f"{regime}|{rule}|{k}"] = v
    for nm in loc:
        for k_, v in loc[nm].items():
            flat[f"shuffle|{nm}|{k_}"] = np.asarray(v)
    for k_, v in chk.items():
        flat[f"p1check|{k_}"] = v
    np.savez_compressed(out / f"stats_{a.set}.npz", **flat)
    C.save_json(out / f"result_{a.set}.json", {"set": a.set, "n_packets": n, "regimes": REGIMES[a.set],
                                               "rules": list(RULES_B), "wall_s": time.time() - t0,
                                               "smoke": bool(a.smoke)}, exclusive=False)
    AC.receipt(out, {"stage": "B", "set": a.set}, scripts=(__file__, C.REPO / "semkine" / "aura.py"))
    print("COMPLETE", a.set, flush=True)


if __name__ == "__main__":
    main()
