#!/usr/bin/env python3
"""[XYZ-ASSOC] P1 step 1: per-event *source* labels from the ground-truth mesh grid (CPU only).

Pre-registration: docs/EXPERIMENT_RECORDS.md, XYZ-ASSOC-20261009, section 2 ("来源标签").
For every event of every evaluated 50 ms window of zgz_global / zgz_local (all events, not only the
2048 the model sees), the ground-truth 51D pose of the event's own millisecond is run through MANO FK,
the mesh is z-buffered at 240x180 with intrinsics s*K, and the event is labelled by what the rendered
GT shows at (x, y):

  src_kind 1 / 2 : a contour pixel lies within 2 px (1 silhouette: hand pixel with a non-hand 4-neighbour;
                   2 depth discontinuity: hand pixel with a hand 4-neighbour more than 10 mm apart).
                   src_part / src_depth = part / depth of the *foreground* side of the nearest contour pixel.
  src_kind 0     : no contour within 2 px, event on a hand pixel: part and depth of that pixel.
  src_kind -1    : otherwise (noise / background / pseudo-label error): src_part -1, src_depth nan.

The "source" is defined by the GT mesh grid; the events are v2e-simulated and the GT is a pseudo label, so
a non-hand label is not necessarily noise.  Event order is exactly that of
semkine.eval_track._window_events.

Usage:  step1_source_labels.py --seq zgz_local --max-steps 20 --out-dir /tmp/x   (dry run + self-check)
        step1_source_labels.py                                                 (all sequences, full run)
"""
import argparse
import json
import multiprocessing as mp
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from step1_common import *                                   # noqa: E402,F401,F403
from step1_common import (H, W, RENDER_SCALE, CAP, OUT_DIR, TIPS21, get_mano, sequences, enumerate_steps,
                          fk_batch, render_intrinsics_f32, face_part_table, load_seq_bundle, _window_events)
from semkine.xyz_mesh import stride_sample, ray_triangle   # noqa: E402
import numpy as np                                         # noqa: E402
import torch                                               # noqa: E402
from scipy.spatial import ConvexHull                       # noqa: E402

DEPTH_JUMP = 0.010          # m, depth-discontinuity threshold
CONTOUR_R = 2.0             # px, search radius for a contour pixel
BARY_EPS = 1e-5

# offsets within CONTOUR_R, nearest first
_OFFS = sorted([(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if dx * dx + dy * dy <= CONTOUR_R ** 2],
               key=lambda o: (o[0] ** 2 + o[1] ** 2, o))


# ------------------------------------------------------------------ rasteriser
def rasterize(verts, faces, face_part, fxyc, eps=BARY_EPS):
    """Software z-buffer. verts (778, 3) float64 camera frame, faces (F, 3) int, pixel (x, y) = integer
    coordinate (the ray through it is ((x-cx)/fx, (y-cy)/fy, 1), the same convention as associate_rays).
    Depth is the exact ray/plane depth (perspective-correct barycentrics), all faces front and back.
    Returns Z (H, W) float64 (inf = empty), PART (H, W) int8 (-1 = empty), FID (H, W) int32 (-1)."""
    fx, fy, cx, cy = fxyc
    z = verts[:, 2]
    okv = z > 1e-6
    zs = np.where(okv, z, 1.0)
    u = fx * verts[:, 0] / zs + cx
    v = fy * verts[:, 1] / zs + cy
    fu, fv, fz = u[faces], v[faces], z[faces]
    x0, x1, x2 = fu[:, 0], fu[:, 1], fu[:, 2]
    y0, y1, y2 = fv[:, 0], fv[:, 1], fv[:, 2]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    ok = okv[faces].all(1) & (np.abs(area) > 1e-9)
    ok &= (fu.max(1) >= 0) & (fu.min(1) <= W - 1) & (fv.max(1) >= 0) & (fv.min(1) <= H - 1)
    xl = np.clip(np.floor(fu.min(1)), 0, W - 1).astype(np.int64)
    xh = np.clip(np.ceil(fu.max(1)), 0, W - 1).astype(np.int64)
    yl = np.clip(np.floor(fv.min(1)), 0, H - 1).astype(np.int64)
    yh = np.clip(np.ceil(fv.max(1)), 0, H - 1).astype(np.int64)
    nw, nh = xh - xl + 1, yh - yl + 1
    n = np.where(ok, nw * nh, 0)
    total = int(n.sum())
    Z = np.full(H * W, np.inf)
    FID = np.full(H * W, -1, np.int32)
    if total == 0:
        return Z.reshape(H, W), np.full((H, W), -1, np.int8), FID.reshape(H, W)
    assert total < 30_000_000, total
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
    fid, px, py, b0, b1, b2 = fid[inside], px[inside], py[inside], b0[inside], b1[inside], b2[inside]
    zz = 1.0 / (b0 / fz[fid, 0] + b1 / fz[fid, 1] + b2 / fz[fid, 2])
    pix = py * W + px
    key = pix.astype(np.float64) + zz / 16.0                      # integer part = pixel, fraction = depth (< 16 m)
    order = np.argsort(key, kind="stable")
    ps = pix[order]
    first = np.ones(len(ps), bool)
    first[1:] = ps[1:] != ps[:-1]
    sel = order[first]
    Z[pix[sel]] = zz[sel]
    FID[pix[sel]] = fid[sel]
    FIDm = FID.reshape(H, W)
    PART = np.where(FIDm >= 0, face_part[np.maximum(FIDm, 0)], -1).astype(np.int8)
    return Z.reshape(H, W), PART, FIDm


# ------------------------------------------------------------------ labels
def _shift(a, dx, dy, fill):
    """s[y, x] = a[y+dy, x+dx], `fill` outside the image."""
    out = np.full_like(a, fill)
    ys0, ys1 = max(0, -dy), min(H, H - dy)
    xs0, xs1 = max(0, -dx), min(W, W - dx)
    out[ys0:ys1, xs0:xs1] = a[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
    return out


def label_maps(Z, P):
    """Per-pixel event labels (part int8, kind int8, depth float32) plus diagnostics.

    Contour pixel: hand pixel with a non-hand 4-neighbour inside the image (silhouette; pixels outside the
    image are ignored, a hand cut by the frame border is not a silhouette) or with a hand 4-neighbour whose
    depth differs by more than 10 mm (depth discontinuity).  Foreground of a contour pixel c: c itself,
    unless a hand 4-neighbour is nearer by more than 10 mm, then the nearest such neighbour.  kind = 1 if c
    has a non-hand 4-neighbour else 2.  An event pixel takes the nearest contour pixel within 2 px (Euclid
    distance, ties -> smaller foreground depth)."""
    hand = np.isfinite(Z)
    Zp = np.pad(Z, 1, constant_values=np.nan)
    Pp = np.pad(P, 1, constant_values=-1)
    sil = np.zeros((H, W), bool)
    disc = np.zeros((H, W), bool)
    fgZ = Z.copy()
    fgP = P.copy()
    with np.errstate(invalid="ignore"):
        for sl in ((slice(0, -2), slice(1, -1)), (slice(2, None), slice(1, -1)),
                   (slice(1, -1), slice(0, -2)), (slice(1, -1), slice(2, None))):
            Zn, Pn = Zp[sl], Pp[sl]
            inimg = ~np.isnan(Zn)
            nhand = np.isfinite(Zn)
            sil |= hand & inimg & ~nhand
            far = hand & nhand & ((Z - Zn) > DEPTH_JUMP)            # the neighbour is nearer: c is the far side
            near = hand & nhand & ((Zn - Z) > DEPTH_JUMP)           # c is the near side
            disc |= far | near
            upd = far & (Zn < fgZ)
            fgZ = np.where(upd, Zn, fgZ)
            fgP = np.where(upd, Pn, fgP)
    contour = sil | disc
    ckind = np.where(sil, 1, 2).astype(np.int8)
    bestd = np.full((H, W), np.inf)
    bestz = np.full((H, W), np.inf)
    bpart = np.full((H, W), -1, np.int8)
    bkind = np.full((H, W), -1, np.int8)
    for dx, dy in _OFFS:
        d = float(np.hypot(dx, dy))
        c = _shift(contour, dx, dy, False)
        z_ = _shift(np.where(contour, fgZ, np.inf), dx, dy, np.inf)
        better = c & ((d < bestd) | ((d == bestd) & (z_ < bestz)))
        bestd = np.where(better, d, bestd)
        bestz = np.where(better, z_, bestz)
        bpart = np.where(better, _shift(fgP, dx, dy, -1), bpart)
        bkind = np.where(better, _shift(ckind, dx, dy, -1), bkind)
    has_c = np.isfinite(bestd)
    kind = np.where(has_c, bkind, np.where(hand, 0, -1)).astype(np.int8)
    part = np.where(has_c, bpart, np.where(hand, P, -1)).astype(np.int8)
    depth = np.where(has_c, bestz, np.where(hand, Z, np.nan)).astype(np.float32)
    return part, kind, depth, {"hand": hand, "contour": contour, "sil": sil, "disc": disc}


def fingertip_flags(verts, joints21, palm_ids, fxyc):
    """Window-end GT: any of the five fingertips projects into the palm region (convex hull of the projected
    palm vertices, hard LBS joint 0) and is nearer to the camera than the palm.  Primary = the pre-registration
    task wording (tip depth < min depth of all palm vertices); `local` = tip depth < min depth of the 5 palm
    vertices image-nearest to the tip (sensitivity variant).  Returns (primary, local, n_tips_in_hull)."""
    fx, fy, cx, cy = fxyc
    pv = verts[palm_ids].astype(np.float64)
    pu = np.stack([fx * pv[:, 0] / pv[:, 2] + cx, fy * pv[:, 1] / pv[:, 2] + cy], 1)
    pz = pv[:, 2]
    tj = joints21[TIPS21].astype(np.float64)
    tu = np.stack([fx * tj[:, 0] / tj[:, 2] + cx, fy * tj[:, 1] / tj[:, 2] + cy], 1)
    tz = tj[:, 2]
    try:
        hull = ConvexHull(pu)
    except Exception:
        return False, False, 0
    eq = hull.equations
    inside = np.array([bool(np.all(eq[:, :2] @ p + eq[:, 2] <= 0)) for p in tu])
    prim = inside & (tz < pz.min())
    dist = np.linalg.norm(pu[None] - tu[:, None], axis=-1)
    nn = np.argsort(dist, axis=1)[:, :5]
    surf = np.take_along_axis(np.broadcast_to(pz, dist.shape), nn, 1).min(1)
    loc = inside & (tz < surf)
    return bool(prim.any()), bool(loc.any()), int(inside.sum())


# ------------------------------------------------------------------ worker
_G = {}


def _init_worker(seq, legacy_dir):
    torch.set_num_threads(1)
    ev, off, tsub, aux, pos51, betas, K = load_seq_bundle(seq, legacy_dir)
    mano = get_mano()
    hard, fpart, ntie = face_part_table(mano)
    fx, fy, cx, cy = [float(t) for t in render_intrinsics_f32(K)]
    _G.update(events=ev, offsets=off, tsub=tsub, pos51=pos51, betas=betas, mano=mano,
              faces=mano.f.numpy().astype(np.int64), face_part=fpart, palm_ids=np.where(hard == 0)[0],
              fxyc=(fx, fy, cx, cy))


def process_step(task):
    s, end = task
    g = _G
    start = end - 49
    off = g["offsets"]
    ev5 = _window_events(g["events"], off, g["tsub"], int(end), 50)       # the evaluator's event order
    n = len(ev5)
    counts = np.diff(off[start:end + 2]).astype(np.int64)
    assert int(counts.sum()) == n
    ms_of = np.repeat(np.arange(start, end + 1, dtype=np.int64), counts)
    ex, ey = ev5[:, 1].astype(np.int64), ev5[:, 2].astype(np.int64)
    assert n == 0 or (ex.min() >= 0 and ex.max() < W and ey.min() >= 0 and ey.max() < H)
    ms_idx = np.arange(start, end + 1)
    verts, joints = fk_batch(g["mano"], g["betas"], g["pos51"][ms_idx])
    verts = verts.numpy().astype(np.float64)
    part = np.full(n, -1, np.int8)
    kind = np.full(n, -1, np.int8)
    depth = np.full(n, np.nan, np.float32)
    diag = np.zeros((50, 6))        # hand px, contour px, sil px, disc px, ... per frame
    cs = np.concatenate([[0], np.cumsum(counts)])
    for i in range(50):
        Z, P, _ = rasterize(verts[i], g["faces"], g["face_part"], g["fxyc"])
        lp, lk, ld, m = label_maps(Z, P)
        diag[i] = [m["hand"].sum(), m["contour"].sum(), m["sil"].sum(), m["disc"].sum(), counts[i],
                   (m["hand"] & ~m["contour"]).sum()]
        if counts[i]:
            sl = slice(cs[i], cs[i + 1])
            part[sl], kind[sl], depth[sl] = lp[ey[sl], ex[sl]], lk[ey[sl], ex[sl]], ld[ey[sl], ex[sl]]
    prim, loc, nin = fingertip_flags(verts[-1], joints[-1].numpy(), g["palm_ids"], g["fxyc"])
    ptr = torch.tensor([0, n], dtype=torch.int64)
    src, mask = stride_sample(ptr, CAP)
    samp = src[0][mask[0]].numpy().astype(np.int32)
    return dict(s=s, part=part, kind=kind, depth=depth, ms=ms_of.astype(np.int32), x=ex.astype(np.int16),
                y=ey.astype(np.int16), a0=int(off[start]), prim=prim, loc=loc, nin=nin, samp=samp, diag=diag)


# ------------------------------------------------------------------ driver
def verify_render(seq, legacy_dir, steps, n_frames=3, n_pix=3000, seed=0):
    """Independent check of the rasteriser: brute-force Moeller-Trumbore (the function associate_rays uses)
    from the camera origin against all 1538 faces at random pixels around the hand."""
    _init_worker(seq, legacy_dir)
    g = _G
    rng = np.random.default_rng(seed)
    out = []
    ends = steps[1]
    for e in rng.choice(ends, n_frames, replace=False):
        v, _ = fk_batch(g["mano"], g["betas"], g["pos51"][[int(e)]])
        v64 = v[0].numpy().astype(np.float64)
        Z, P, FID = rasterize(v64, g["faces"], g["face_part"], g["fxyc"])
        ys, xs = np.where(np.isfinite(Z))
        x0, x1, y0, y1 = max(xs.min() - 6, 0), min(xs.max() + 6, W - 1), max(ys.min() - 6, 0), min(ys.max() + 6, H - 1)
        px = rng.integers(x0, x1 + 1, n_pix); py = rng.integers(y0, y1 + 1, n_pix)
        fx, fy, cx, cy = g["fxyc"]
        d = torch.tensor(np.stack([(px - cx) / fx, (py - cy) / fy, np.ones(n_pix)], 1), dtype=torch.float32)
        tri = v[0][torch.from_numpy(g["faces"])]
        t, _, _ = ray_triangle(d[:, None, :], tri[None, :, 0], tri[None, :, 1], tri[None, :, 2])
        tmin, fsel = t.min(1)
        zr = Z[py, px]
        bh = torch.isfinite(tmin).numpy()
        rh = np.isfinite(zr)
        both = bh & rh
        # winding: front-most face normal . ray direction (< 0 if the MANO face winding is outward)
        fidv = FID[py, px][both]
        fl = tri[fidv]
        nrm = torch.cross(fl[:, 1] - fl[:, 0], fl[:, 2] - fl[:, 0], dim=-1)
        dn = (nrm * d[both]).sum(-1)
        out.append(dict(frame=int(e), n_pix=n_pix, both_hit=int(both.sum()), raster_only=int((rh & ~bh).sum()),
                        brute_only=int((bh & ~rh).sum()), neither=int((~rh & ~bh).sum()),
                        max_abs_dz_mm=float(np.abs(zr[both] - tmin.numpy()[both]).max() * 1000) if both.any() else None,
                        frac_dz_gt_0p1mm=float((np.abs(zr[both] - tmin.numpy()[both]) > 1e-4).mean()) if both.any() else None,
                        front_face_n_dot_d_negative=float((dn < 0).float().mean()) if both.any() else None))
    return out


def run_seq(seq, legacy_dir, procs, max_steps, out_dir):
    ev, off, tsub, aux, pos51, betas, K = load_seq_bundle(seq, legacy_dir)
    run_id, ends, prev_end = enumerate_steps(aux)
    S = len(ends) if max_steps is None else min(max_steps, len(ends))
    tasks = [(s, int(ends[s])) for s in range(S)]
    print(f"[{seq}] steps {S}/{len(ends)} runs={len(np.unique(run_id))} n_ms={len(off) - 1} pos51={pos51.shape} "
          f"procs={procs}", flush=True)
    t0 = time.time()
    res = [None] * S
    ctx = mp.get_context("fork")
    with ctx.Pool(procs, initializer=_init_worker, initargs=(seq, legacy_dir)) as pool:
        for k, r in enumerate(pool.imap_unordered(process_step, tasks, chunksize=1)):
            res[r["s"]] = r
            if (k + 1) % max(1, S // 10) == 0:
                print(f"  {k + 1}/{S} steps  {time.time() - t0:.0f}s", flush=True)
    n_ev = np.array([len(r["part"]) for r in res], np.int64)
    ev_start = np.concatenate([[0], np.cumsum(n_ev)])
    samp_n = np.array([len(r["samp"]) for r in res], np.int64)
    samp_start = np.concatenate([[0], np.cumsum(samp_n)])
    cat = lambda key: np.concatenate([r[key] for r in res])
    out = dict(
        src_part=cat("part"), src_kind=cat("kind"), src_depth=cat("depth"),
        step_idx=np.repeat(np.arange(S, dtype=np.int32), n_ev), ms=cat("ms"), ev_x=cat("x"), ev_y=cat("y"),
        step_ev_start=ev_start, step_a0=np.array([r["a0"] for r in res], np.int64),
        step_end_ms=ends[:S].astype(np.int32), step_prev_ms=prev_end[:S].astype(np.int32),
        run_id=run_id[:S].astype(np.int32),
        fingertip_in_front=np.array([r["prim"] for r in res], bool),
        fingertip_in_front_local=np.array([r["loc"] for r in res], bool),
        tips_in_hull=np.array([r["nin"] for r in res], np.int8),
        samp_local=cat("samp").astype(np.int32), samp_start=samp_start,
    )
    diag = np.stack([r["diag"] for r in res])                          # (S, 50, 6)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"labels_{seq}.npz"
    np.savez(path, **out)
    print(f"[{seq}] wrote {path}  events={len(out['src_part'])}  {time.time() - t0:.0f}s", flush=True)
    return out, diag


def selfcheck(seq, out, diag):
    n = len(out["src_part"])
    k = out["src_kind"]
    tot_px = diag.shape[0] * diag.shape[1] * H * W
    hand, cont, sil, disc = diag[..., 0].sum(), diag[..., 1].sum(), diag[..., 2].sum(), diag[..., 3].sum()
    print(f"\n==== self-check {seq} ({diag.shape[0]} steps, {diag.shape[0] * 50} frames, {n} events) ====")
    print(f"hand pixels / image pixels      : {hand / tot_px:.4f}")
    print(f"contour pixels / image pixels   : {cont / tot_px:.4f}   (silhouette {sil / tot_px:.4f}, depth-disc. {disc / tot_px:.4f})")
    print(f"contour pixels / hand pixels    : {cont / max(hand, 1):.4f}")
    for kk, name in ((0, "interior"), (1, "silhouette"), (2, "depth-disc."), (-1, "non-hand")):
        print(f"events src_kind {kk:2d} ({name:12s}): {np.mean(k == kk):.4f}  (n={int((k == kk).sum())})")
    hp = out["src_part"][out["src_part"] >= 0]
    print("src_part histogram (0..15):", np.bincount(hp, minlength=16).tolist())
    fm = (k >= 1) & (out["src_part"] >= 1)
    print(f"finger-contour-foreground events (the M denominator): {fm.mean():.4f} of all events, n={int(fm.sum())}")
    print(f"depth: finite for hand events {np.isfinite(out['src_depth'][k >= 0]).all()}, nan for non-hand {np.isnan(out['src_depth'][k < 0]).all()}")
    print(f"fingertip_in_front (steps): primary {out['fingertip_in_front'].mean():.4f}, local {out['fingertip_in_front_local'].mean():.4f}; "
          f"any tip in hull {np.mean(out['tips_in_hull'] > 0):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default=None, help="one of the eval sequences (default: all)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--procs", type=int, default=100)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--verify", action="store_true", help="also run the brute-force render check")
    a = ap.parse_args()
    out_dir = __import__("pathlib").Path(a.out_dir)
    seqs = [(s, d) for s, d in sequences() if a.seq in (None, s)]
    summary = {}
    for seq, legacy_dir in seqs:
        out, diag = run_seq(seq, legacy_dir, a.procs, a.max_steps, out_dir)
        selfcheck(seq, out, diag)
        if a.verify:
            ev, off, tsub, aux, pos51, betas, K = load_seq_bundle(seq, legacy_dir)
            st = enumerate_steps(aux)
            v = verify_render(seq, legacy_dir, st)
            print("render verification vs brute-force ray/triangle:")
            for r in v:
                print("  ", json.dumps(r))
            summary[seq] = {"render_verify": v}
    if summary:
        (out_dir / "labels_render_verify.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
