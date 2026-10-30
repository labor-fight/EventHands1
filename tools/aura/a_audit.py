#!/usr/bin/env python3
"""AURA Stage A: implementation correctness of the C1 association (docs/EXPERIMENT_RECORDS.md#aura-prereg-20261024),
CPU only, zero training. The association under test is the frozen C1 module's own (`forward_capture`, i.e.
`associate_rays` as `_xyz_mesh_forward` calls it) on evaluator packets:

    zgz_e3     zgz every 10th endpoint, anchor = GT at the previous endpoint
    zgz_e2     zgz every 10th endpoint, anchor = C1's strict closed-loop incoming state
    sel_e3     selection subjects every 40th endpoint, GT anchor
    fit_noise  fit subjects every 200th endpoint, GT + C1's mixed prev noise (rng 777)

A1  against the specification (surface): exhaustive nearest positive hit over all 1538 faces with the same
    `ray_triangle`; every disagreement must be a miss of the 16-candidate approximation (the exhaustive nearest face is
    not among the candidate faces) or an equal-depth tie; missed / nearer-face events <= 1 % of exhaustive hits.
A2  against the specification (silhouette): the rasterised z-buffer of the anchor mesh (`step1_source_labels.rasterize`)
    at each event pixel; covered <=> exhaustive hit except at triangle-edge slack (<= 0.1 % of events); production
    hit <=> covered except A1-explained misses and that slack.
A3  against the code (rule): an independent float64 NumPy rewrite of hit_first (the pinned code facts); vid / valid /
    hit equal and A, q, z within float32 tolerance, every disagreement explained by a float32-vs-float64 tie.
A4  semantics: mass conservation, hit => q = 1, near => 0 < q <= 1, none => r = 1, z > 0, hit point = z d (1e-5 m).
A5  invariance: event order, batch composition, padding, chunk size (bitwise, CPU).
A6  `pytest tests/test_aura.py` (run separately; recorded by `combine`).
A7  descriptive: near records on back-facing (n.d > 0) / z-buffer-occluded vertices; first hits on back faces;
    association statistics.

    python tools/aura/a_audit.py --set zgz_e3 --out outputs/aura/a      (one process per set)
    python tools/aura/a_audit.py --combine --out outputs/aura/a
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.agg778 import common as C  # noqa: E402
from tools.aura import common as AC  # noqa: E402
from semkine import aura as AU  # noqa: E402
from semkine.xyz_mesh import BARY_EPS, R_MAX, associate_rays, ray_triangle, vertex_normals  # noqa: E402

SETS = {"zgz_e3": ("zgz", 10, "e3"), "zgz_e2": ("zgz", 10, "e2"), "sel_e3": ("sel", 40, "e3"),
        "fit_noise": ("fit", 200, "noise")}
BATCH = 16
TIE_REL = 1e-5


# ------------------------------------------------------------------------------------------------ helpers
def packets(set_name, smoke):
    src, stride, regime = SETS[set_name]
    seqs = {"zgz": C.zgz_seqs, "sel": lambda: C.subject_seqs(C.SELECT_SUBJ),
            "fit": lambda: C.subject_seqs(C.FIT_SUBJ)}[src]()
    ds = C.Endpoints(seqs, stride=stride)
    if smoke:
        ds.index = ds.index[:2 * BATCH]
    return ds, regime


def anchors_for(items, regime, trace, draw, rng):
    if regime == "e3":
        return np.stack([x["gt_prev"] for x in items])
    if regime == "e2":
        return np.stack([trace[x["seq"]][x["k"]] for x in items])
    return np.stack([x["gt_prev"] + draw(rng)[0] for x in items])


@torch.no_grad()
def exhaustive(d, mask, verts, faces):
    """(B, N) nearest positive hit depth and face over all faces (inf / -1 where none), with the library's
    `ray_triangle` (float32, same arithmetic per face)."""
    B, N, _ = d.shape
    t_all = torch.full((B, N), float("inf"))
    f_all = torch.full((B, N), -1, dtype=torch.long)
    tri = verts[:, faces.long()]                                                     # (B, F, 3, 3)
    for b in range(B):
        idx = mask[b].nonzero().squeeze(1)
        for i0 in range(0, len(idx), 2048):
            r = idx[i0:i0 + 2048]
            t, _, _ = ray_triangle(d[b, r].float().view(-1, 1, 3), tri[b, None, :, 0], tri[b, None, :, 1],
                                   tri[b, None, :, 2])
            tm, fm = t.min(1)
            t_all[b, r] = tm
            f_all[b, r] = torch.where(torch.isfinite(tm), fm, torch.full_like(fm, -1))
    return t_all, f_all


def candidate_faces(d, verts, vert_faces, b, n):
    """The 16-candidate face set of one event, as `associate_rays` builds it (float32)."""
    V = verts[b].float()
    zv = V[:, 2]
    vu, vv = V[:, 0] / zv.clamp_min(1e-6), V[:, 1] / zv.clamp_min(1e-6)
    dd = d[b, n].float()
    d2 = (vu - dd[0]).square() + (vv - dd[1]).square()
    nid = d2.topk(16, largest=False).indices
    cf = vert_faces[nid].reshape(-1)
    return set(cf[cf >= 0].tolist())


def mt64(dd, P0, P1, P2, eps=BARY_EPS):
    """Moeller-Trumbore in float64 (dd (m, 3), P* (m, C, 3)): t (inf where no hit), u, v, relative det."""
    e1, e2 = P1 - P0, P2 - P0
    D = np.broadcast_to(dd[:, None, :], e1.shape)
    pvec = np.cross(D, e2)
    det = (e1 * pvec).sum(-1)
    ok = np.abs(det) > 1e-14
    inv = 1.0 / np.where(ok, det, 1.0)
    tvec = -P0
    u = (tvec * pvec).sum(-1) * inv
    qvec = np.cross(tvec, e1)
    v = (D * qvec).sum(-1) * inv
    t = (e2 * qvec).sum(-1) * inv
    good = ok & (u >= -eps) & (v >= -eps) & (u + v <= 1.0 + eps) & (t > 1e-4)
    rel = np.abs(det) / (np.linalg.norm(e1, axis=-1) * np.linalg.norm(e2, axis=-1) * np.linalg.norm(D, axis=-1)
                         + 1e-30)
    return np.where(good, t, np.inf), u, v, rel


def reference_rule(dd, V, faces, vert_faces, band, ftl, z_tol=0.01, k=16):
    """Independent float64 hit_first for the events of one packet (dd (m, 3), V (778, 3)): (vid, A, q, z, valid,
    hit) and per-event tie flags (float32-vs-float64 decision margins)."""
    m = len(dd)
    vu, vv = V[:, 0] / V[:, 2], V[:, 1] / V[:, 2]
    d2 = (vu[None] - dd[:, 0:1]) ** 2 + (vv[None] - dd[:, 1:2]) ** 2                 # (m, V)
    part = np.argpartition(d2, k, axis=1)[:, :k + 1]
    pd = np.take_along_axis(d2, part, 1)
    o = np.argsort(pd, axis=1, kind="stable")
    part, pd = np.take_along_axis(part, o, 1), np.take_along_axis(pd, o, 1)
    nid, nd2 = part[:, :k], pd[:, :k]
    tie = (pd[:, k] - pd[:, k - 1]) <= TIE_REL * np.maximum(pd[:, k - 1], 1e-12)       # candidate-set boundary
    # ordering ties inside the 16 candidates (amendment 2026-10-24: they reorder the near records and fix q)
    tie |= ((pd[:, 1:k] - pd[:, :k - 1]) <= TIE_REL * np.maximum(pd[:, :k - 1], 1e-12)).any(1)
    nang = np.sqrt(np.maximum(nd2, 0.0))
    cf = vert_faces[nid].reshape(m, -1)
    cfs = np.maximum(cf, 0)
    P = V[faces[cfs]]                                                               # (m, C, 3, 3)
    t, u, v, rel = mt64(dd, P[:, :, 0], P[:, :, 1], P[:, :, 2])
    t = np.where(cf < 0, np.inf, t)
    eps = BARY_EPS
    near_edge = (np.minimum.reduce([np.abs(u + eps), np.abs(v + eps), np.abs(1 + eps - u - v)]) <= 1e-5) & (cf >= 0)
    tie |= near_edge.any(1)
    fsel = t.argmin(1)
    tmin = t[np.arange(m), fsel]
    is_hit = np.isfinite(tmin)
    ts = np.sort(t, 1)
    with np.errstate(invalid="ignore"):
        gap = ts[:, 1] - ts[:, 0]
    tie |= is_hit & (gap <= 1e-6) & (cfs[np.arange(m), fsel] != cfs[np.arange(m), t.argsort(1)[:, 1]])
    fh = cfs[np.arange(m), fsel]
    uh, vh = u[np.arange(m), fsel], v[np.arange(m), fsel]
    wb = np.stack([np.maximum(1 - uh - vh, 0), np.maximum(uh, 0), np.maximum(vh, 0), np.zeros(m)], 1)
    A_hit = wb / np.maximum(wb.sum(1, keepdims=True), 1e-12)
    fv = faces[fh]
    vid_hit = np.concatenate([fv, fv[:, :1]], 1)
    hit_rel = rel[np.arange(m), fsel]
    Vn = V[nid]                                                                      # (m, k, 3)
    zj = (Vn * dd[:, None]).sum(-1) / (dd * dd).sum(-1, keepdims=True)
    ref = nang[:, :1]
    inband = (nang <= band) & (zj > 1e-4)
    same = inband & (nang <= ref + ftl)
    zfront = np.where(same, zj, np.inf).min(1)
    layer = inband & (zj <= zfront[:, None] + z_tol)
    tie |= (np.abs(nang - band) <= TIE_REL * band).any(1) | (np.abs(nang - (ref + ftl)) <= TIE_REL * band).any(1)
    tie |= (np.isfinite(zfront)[:, None] & (np.abs(zj - (zfront[:, None] + z_tol)) <= 1e-6)).any(1)
    col = np.arange(k)
    rank = np.where(layer, col[None], col[None] + k)
    cols = np.argsort(rank, 1, kind="stable")[:, :R_MAX]
    kept = np.take_along_axis(layer, cols, 1)
    ang = np.take_along_axis(nang, cols, 1)
    sig = band / 2.0
    kw = np.exp(-0.5 * (ang / sig) ** 2) * kept
    is_near = ~is_hit & kept.any(1)
    A_near = kw / np.maximum(kw.sum(1, keepdims=True), 1e-12)
    q_near = np.exp(-0.5 * (ang[:, 0] / sig) ** 2)
    z_near = (A_near * np.take_along_axis(zj, cols, 1)).sum(1)
    vid_near = np.take_along_axis(nid, cols, 1)
    ok = is_hit | is_near
    vid = np.where(is_hit[:, None], vid_hit, vid_near) * ok[:, None]
    A = np.where(is_hit[:, None], A_hit, A_near) * ok[:, None]
    q = np.where(is_hit, 1.0, q_near) * ok
    z = np.where(is_hit, tmin, z_near) * ok
    return dict(vid=vid, A=A, q=q, z=z, valid=ok, hit=is_hit), tie, hit_rel


# ------------------------------------------------------------------------------------------------ one set
def run_set(set_name, out_dir, smoke=False, threads=8):
    torch.set_num_threads(threads)
    sys.path.insert(0, str(C.REPO / "tools" / "xcl" / "sources"))
    import step1_source_labels as L                                                  # noqa: E402  (CPU only)
    from step1_common import face_part_table                                        # noqa: E402
    dev = torch.device("cpu")
    frozen, cfg = C.load_frozen("A0", dev)
    faces = frozen.mano.f.long()
    faces_np = faces.numpy()
    vert_faces = frozen.xyz.vert_faces
    vf_np = vert_faces.numpy()
    _, fpart, _ = face_part_table(frozen.mano)
    ds, regime = packets(set_name, smoke)
    trace = C.strict_trace() if regime == "e2" else None
    draw = C.noise_sampler(cfg)
    rng = np.random.default_rng(777)
    S = {k: 0 for k in ("events", "exh_hit", "a1_agree", "a1_tie", "a1_miss_explained", "a1_nearer_explained",
                        "a1_unexplained", "a2_slack", "a2_prod_vs_raster", "a3_events", "a3_ties", "a3_unexplained",
                        "a3_tol_violations", "a4_violations", "near_records", "near_backface", "near_occluded",
                        "hits", "hit_backface", "valid", "a5_checked", "a5_mismatch")}
    maxdiff = {"A": 0.0, "q": 0.0, "z": 0.0}
    examples = []
    t0 = time.time()
    for i0 in range(0, len(ds), BATCH):
        items = [ds[i] for i in range(i0, min(i0 + BATCH, len(ds)))]
        anchors = anchors_for(items, regime, trace, draw, rng)
        batch = C.make_batch(items, anchors, dev, with_sources=False)
        cap = C.forward_capture(frozen, batch)
        a, d, mask, verts = cap["assoc"], cap["d"], cap["mask"], cap["verts"]
        intr = C.intrinsics(frozen, batch)
        f_px = (intr[0] * intr[1]).sqrt()
        B = d.shape[0]
        # ---- A4 semantics
        w = a["q"].unsqueeze(-1) * a["A"] * mask.float().unsqueeze(-1)
        tot = w.sum(-1) + (1 - a["q"]) * mask.float()
        Vr = verts[torch.arange(B).view(B, 1, 1), a["vid"]]
        Xh = (a["A"].unsqueeze(-1) * Vr).sum(2)
        bad = ((tot - 1).abs() > 1e-5) & mask
        bad |= a["hit"] & (a["q"] != 1)
        bad |= a["valid"] & ~a["hit"] & ((a["q"] <= 0) | (a["q"] > 1))
        bad |= ~a["valid"] & mask & (a["q"] != 0)
        bad |= a["valid"] & (a["z"] <= 0)
        bad |= a["hit"] & ((Xh - a["z"].unsqueeze(-1) * d).norm(dim=-1) > 1e-5)
        S["a4_violations"] += int(bad.sum())
        S["events"] += int(mask.sum())
        S["valid"] += int(a["valid"].sum())
        S["hits"] += int(a["hit"].sum())
        # ---- A1 exhaustive
        t_ex, f_ex = exhaustive(d, mask, verts, faces)
        S["exh_hit"] += int((torch.isfinite(t_ex) & mask).sum())
        prod_hit = a["hit"]
        eh_all = torch.isfinite(t_ex)
        same_face = (a["vid"][..., :3].sort(-1).values == faces[f_ex.clamp_min(0)].sort(-1).values).all(-1)
        agree = mask & ((~prod_hit & ~eh_all) | (prod_hit & eh_all & same_face & (a["z"] == t_ex)))
        tie = mask & ~agree & prod_hit & eh_all & ((a["z"] - t_ex).abs() <= 1e-6)
        S["a1_agree"] += int(agree.sum())
        S["a1_tie"] += int(tie.sum())
        for b, n in (mask & ~agree & ~tie).nonzero().tolist():
            ph, eh = bool(prod_hit[b, n]), bool(eh_all[b, n])
            cand = candidate_faces(d, verts, vert_faces, b, n)
            if eh and int(f_ex[b, n]) not in cand and (not ph or float(t_ex[b, n]) < float(a["z"][b, n])):
                S["a1_miss_explained" if not ph else "a1_nearer_explained"] += 1
            else:
                S["a1_unexplained"] += 1
                if len(examples) < 20:
                    examples.append({"set": set_name, "b": b, "n": n, "prod_hit": ph, "exh_hit": eh,
                                     "z_prod": float(a["z"][b, n]), "t_exh": float(t_ex[b, n])})
        # ---- A2 raster silhouette, A7 back faces / occlusion
        vis = AU.visible(verts, intr)
        nrm = vertex_normals(verts.float(), faces)
        for b in range(B):
            fx, fy, cx, cy = (float(x[b]) for x in intr)
            _, _, FID = L.rasterize(verts[b].double().numpy(), faces_np, fpart, (fx, fy, cx, cy))
            live = mask[b].nonzero().squeeze(1)
            px = (d[b, live, 0] * fx + cx).round().long().clamp(0, 239)
            py = (d[b, live, 1] * fy + cy).round().long().clamp(0, 179)
            cov = torch.from_numpy(FID[py.numpy(), px.numpy()] >= 0)
            eh = torch.isfinite(t_ex[b, live])
            S["a2_slack"] += int((cov != eh).sum())
            S["a2_prod_vs_raster"] += int((prod_hit[b, live] != cov).sum())             # = A1 misses + slack
            near = a["valid"][b] & ~a["hit"][b]
            if near.any():
                vids = a["vid"][b][near]
                wn = (a["A"][b][near] > 0)
                dn = d[b][near].unsqueeze(1).expand(-1, R_MAX, -1)
                back = (nrm[b][vids] * dn).sum(-1) > 0
                S["near_records"] += int(wn.sum())
                S["near_backface"] += int((back & wn).sum())
                S["near_occluded"] += int((~vis[b][vids] & wn).sum())
            hh = a["hit"][b]
            if hh.any():
                fv = a["vid"][b][hh][:, :3]
                P = verts[b][fv]
                fn = torch.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], dim=-1)
                S["hit_backface"] += int(((fn * d[b][hh]).sum(-1) > 0).sum())
        # ---- A3 float64 rule
        for b in range(B):
            live = mask[b].nonzero().squeeze(1)
            if len(live) == 0:
                continue
            dd = d[b, live].double().numpy()
            ref, tie, hrel = reference_rule(dd, verts[b].double().numpy(), faces_np, vf_np, 16.0 / float(f_px[b]),
                                            3.0 / float(f_px[b]))
            pv, pA, pq, pz = (a[k_][b, live].numpy() for k_ in ("vid", "A", "q", "z"))
            pval, phit = a["valid"][b, live].numpy(), a["hit"][b, live].numpy()
            dis = (pval != ref["valid"]) | (phit != ref["hit"]) | (pv != ref["vid"]).any(1)
            cond = np.maximum(np.where(ref["hit"], hrel, 1.0), 1e-3)                  # grazing hits: small det
            tolz, tolA = 2e-6 / cond, 1e-4 / cond                                     # float32 MT cancellation
            num = (np.abs(pA - ref["A"]).max(1) > tolA) | (np.abs(pq - ref["q"]) > 1e-5) | (np.abs(pz - ref["z"]) > tolz)
            S["a3_events"] += len(live)
            S["a3_ties"] += int(tie.sum())
            S["a3_unexplained"] += int((dis & ~tie).sum())
            S["a3_tol_violations"] += int((num & ~dis & ~tie).sum())
            agree = ~dis & ~tie
            if agree.any():
                maxdiff["A"] = max(maxdiff["A"], float(np.abs(pA - ref["A"])[agree].max()))
                maxdiff["q"] = max(maxdiff["q"], float(np.abs(pq - ref["q"])[agree].max()))
                maxdiff["z"] = max(maxdiff["z"], float(np.abs(pz - ref["z"])[agree].max()))
            for j in np.nonzero(dis & ~tie)[0][:3]:
                if len(examples) < 40:
                    examples.append({"set": set_name, "a3": True, "b": b, "n": int(live[j]),
                                     "prod": {"valid": bool(pval[j]), "hit": bool(phit[j]), "vid": pv[j].tolist()},
                                     "ref": {"valid": bool(ref["valid"][j]), "hit": bool(ref["hit"][j]),
                                             "vid": ref["vid"][j].tolist()}})
        # ---- A5 invariance (library associate_rays, CPU): order, batch composition, padding, chunk size
        band, front = 16.0 / f_px, 3.0 / f_px
        base = associate_rays(d, mask, verts, faces, vert_faces, band=band, front_tol=front, z_tol=0.01, k=16)
        perm = torch.arange(d.shape[1] - 1, -1, -1)
        rev = associate_rays(d[:, perm], mask[:, perm], verts, faces, vert_faces, band=band, front_tol=front,
                             z_tol=0.01, k=16)
        pad = 37
        dpad = torch.cat([d, torch.ones(B, pad, 3)], 1)
        mpad = torch.cat([mask, torch.zeros(B, pad, dtype=torch.bool)], 1)
        padded = associate_rays(dpad, mpad, verts, faces, vert_faces, band=band, front_tol=front, z_tol=0.01, k=16)
        chunked = associate_rays(d, mask, verts, faces, vert_faces, band=band, front_tol=front, z_tol=0.01, k=16,
                                 chunk=1000)
        single = [associate_rays(d[b:b + 1], mask[b:b + 1], verts[b:b + 1], faces, vert_faces, band=band[b:b + 1],
                                 front_tol=front[b:b + 1], z_tol=0.01, k=16) for b in range(B)]
        for k_ in base:
            S["a5_checked"] += 1
            ok = torch.equal(base[k_], a[k_]) and torch.equal(rev[k_][:, perm], base[k_])
            ok &= torch.equal(padded[k_][:, :d.shape[1]], base[k_]) and torch.equal(chunked[k_], base[k_])
            ok &= all(torch.equal(single[b][k_][0], base[k_][b]) for b in range(B))
            S["a5_mismatch"] += int(not ok)
        if (i0 // BATCH) % 10 == 0:
            print(set_name, i0, len(ds), f"{time.time() - t0:.0f} s", json.dumps({k: S[k] for k in (
                "events", "a1_unexplained", "a3_unexplained", "a5_mismatch")}), flush=True)
    res = {"set": set_name, "n_packets": len(ds), "counts": S, "a3_max_abs_diff_agreeing": maxdiff,
           "examples": examples, "wall_s": time.time() - t0, "smoke": bool(smoke)}
    C.save_json(Path(out_dir) / f"set_{set_name}.json", res, exclusive=False)
    return res


def combine(out_dir):
    out_dir = Path(out_dir)
    sets = {s: json.loads((out_dir / f"set_{s}.json").read_text()) for s in SETS}
    tot = {k: sum(r["counts"][k] for r in sets.values()) for k in next(iter(sets.values()))["counts"]}
    z = sets["zgz_e3"]["counts"]
    tests = subprocess.run([sys.executable, "-m", "pytest", "tests/test_aura.py", "-q", "-p", "no:cacheprovider"],
                           cwd=C.REPO, capture_output=True, text=True)
    gates = {
        "A1_no_unexplained": tot["a1_unexplained"] == 0,
        "A1_miss_budget_zgz_e3": (z["a1_miss_explained"] + z["a1_nearer_explained"]) <= 0.01 * max(z["exh_hit"], 1),
        "A2_raster_slack": tot["a2_slack"] <= 0.001 * max(tot["events"], 1),
        "A3_no_unexplained": tot["a3_unexplained"] == 0 and tot["a3_tol_violations"] == 0,
        "A4_semantics": tot["a4_violations"] == 0,
        "A5_invariance": tot["a5_mismatch"] == 0,
        "A6_tests": tests.returncode == 0,
    }
    desc = {s: {"valid": r["counts"]["valid"] / max(r["counts"]["events"], 1),
                "hit": r["counts"]["hits"] / max(r["counts"]["events"], 1),
                "candidate_miss_of_exhaustive_hits": (r["counts"]["a1_miss_explained"] + r["counts"][
                    "a1_nearer_explained"]) / max(r["counts"]["exh_hit"], 1),
                "near_records_on_backfacing": r["counts"]["near_backface"] / max(r["counts"]["near_records"], 1),
                "near_records_on_occluded": r["counts"]["near_occluded"] / max(r["counts"]["near_records"], 1),
                "first_hit_on_backface": r["counts"]["hit_backface"] / max(r["counts"]["hits"], 1),
                "production_vs_raster": r["counts"]["a2_prod_vs_raster"] / max(r["counts"]["events"], 1),
                "a3_ties": r["counts"]["a3_ties"] / max(r["counts"]["a3_events"], 1)} for s, r in sets.items()}
    res = {"pass": all(gates.values()), "gates": gates, "totals": tot, "descriptive": desc,
           "a3_max_abs_diff_agreeing": {s: r["a3_max_abs_diff_agreeing"] for s, r in sets.items()},
           "tests_tail": tests.stdout[-400:], "r_match_m": AU.R_MATCH}
    C.save_json(out_dir / "result.json", res, exclusive=False)
    AC.receipt(out_dir, {"stage": "A"}, scripts=(__file__, C.REPO / "semkine" / "aura.py"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=tuple(SETS))
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    if not a.smoke:
        AC.verify_protocol("A")
    if a.combine:
        r = combine(a.out)
        print(json.dumps({k: r[k] for k in ("pass", "gates")}, indent=1))
        print("COMPLETE", a.out, flush=True)
        return
    run_set(a.set, a.out, a.smoke, a.threads)
    print("COMPLETE", a.set, flush=True)


if __name__ == "__main__":
    main()
