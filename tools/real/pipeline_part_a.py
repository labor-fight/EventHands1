#!/usr/bin/env python3
"""Panel (a) of the paper's pipeline figure: event-to-surface association, drawn from real outputs.

One update interval k of a real DAVIS346 recording (ch_global, normal light, the undistorted 346 x 260 stream the
networks read; interval k = ms (35357, 35407] of the converted time base, the 50 ms in front of the event-input figure's
stream). Everything shown is computed, not sketched:

  data    (torch; EventHandsTrain env) the real-data ESMNet (r_s2dx2, outputs/real/20261110/routers/chosen.json) on every
          event of the interval: on-hand probability, part, template point -> face + barycentrics; route-first selection
          (on-hand at tau 0.5, plus off-hand events within 16 px of an on-hand event; semkine/rfs.py); the previous state
          x_{k-1} = the label at the interval's start (the figure's stand-in for the tracker's state); the routed points
          P_i on that MANO mesh, rays d_i, depths z_i, residuals delta_i = z_i d_i - P_i (semkine/s45sm.py PFA).
  render  (numpy, cv2, matplotlib; EventHands env) the event stream (tools/real/event_stream_figure.py), the routed
          events, the previous-state mesh and the delta_i inset, composed with the module boxes into one panel.

    /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/real/pipeline_part_a.py data
    /data1/lyq/miniconda3/envs/EventHands/bin/python tools/real/pipeline_part_a.py render [--preview]

CPU only. ch is one of the real data's training subjects: the panel illustrates the pipeline, it is not a result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEQ, SPLIT = "ch_global", "train"
DATA = Path("/data1/lyq/code/mesh/EventHands/data/hand_data_real_n346")
CKPT = ROOT / "outputs/real/20261110/routers/r_s2dx2/r_s2d_final.ckpt"
END_MS, WINDOW = 35407, 50                       # interval k = ms (END_MS - WINDOW, END_MS]
STREAM_MS = (34237, 35407)                       # the stream shown before it (the event-input figure's span)
TAU, NEAR = 0.5, 16                              # route-first rule of the real data (#real-rfs-20261112)
W, H = 346, 260
WORK = ROOT / "outputs/real/20261110/analysis/figure_pipeline_a"
OUT = ROOT / "docs/assets/pipeline_part_a_20261113"
N_PARTS = 16
#: MANO joint order of the 16 parts: 0 palm, 1-3 index, 4-6 middle, 7-9 pinky, 10-12 ring, 13-15 thumb
FINGER_OF_PART = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
FINGER_NAMES = ("palm", "index", "middle", "pinky", "ring", "thumb")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ================================================================================================================ data
def stage_data():
    import torch
    import torch.nn.functional as F
    from tools.real import evht as X
    X.enter()                                                   # router modules on the 346 x 260 canvas
    from semkine import pfo_router as PR
    from semkine import rfs as RFS
    from semkine import s45sm as SM
    from semkine import s45sm_fast as FAST
    from mano_layer import ManoLayer
    from tools.erh.convert import read_meta
    assert (SM.WIDTH, SM.HEIGHT) == (W, H)

    d = DATA / SPLIT
    ev = np.load(d / f"{SEQ}_events.npy", mmap_mode="r")
    off = np.load(d / f"{SEQ}_offsets.npy")
    tsub = np.load(d / f"{SEQ}_tsub.npy", mmap_mode="r")
    aux = np.load(d / f"{SEQ}_aux.npz")
    meta = read_meta(d / f"{SEQ}.meta")
    assert aux["valid_ms"][END_MS - WINDOW] and aux["valid_ms"][END_MS]

    # ---- the interval's packet, as semkine/dataset.py builds it: ms [start, end], t = microseconds from start
    start = END_MS - WINDOW + 1
    a0, a1 = int(off[start]), int(off[END_MS + 1])
    e = np.asarray(ev[a0:a1]).astype(np.int64)
    us = np.repeat(np.arange(WINDOW, dtype=np.int64), np.diff(off[start:END_MS + 2]).astype(np.int64)) * 1000 \
        + np.asarray(tsub[a0:a1]).astype(np.int64)
    x, y, p = e[:, 0], e[:, 1], np.clip(e[:, 2], 0, 1)
    n = len(x)
    pk = torch.zeros(n, 5)
    pk[:, 1], pk[:, 2] = torch.from_numpy(x).float(), torch.from_numpy(y).float()
    pk[:, 3], pk[:, 4] = torch.from_numpy(us / 1e6).float(), torch.from_numpy(p).float()
    ptr, dt = torch.tensor([0, n]), torch.tensor([WINDOW / 1000.0])

    # ---- ESMNet on every event of the interval (route first)
    net, obj = PR.load(CKPT)
    net = net.float().eval()
    with torch.no_grad():
        out = FAST.router_forward(net, pk, ptr, dt, torch.arange(n).view(1, n), torch.ones(1, n, dtype=torch.bool))
        prob = torch.sigmoid(out["onhand"].float())[0]
        on = prob >= TAU
        r = net.route(out, on.view(1, n))
        face, bary = r[0][0], r[1][0]
        part = out["part"].float()[0].argmax(-1)
        coord = out["coord"].float()[0]                             # rest-template point, cm
        img = torch.zeros(1, 1, H, W)
        img[0, 0, pk[on, 2].long(), pk[on, 1].long()] = 1.0
        k = 2 * NEAR + 1
        img = F.max_pool2d(img, (1, k), stride=1, padding=(0, NEAR))
        img = F.max_pool2d(img, (k, 1), stride=1, padding=(NEAR, 0))
        elig = on | (img[0, 0, pk[:, 2].long(), pk[:, 1].long()] > 0)
        src, smask = RFS.stride_over(elig, ptr, 2048)                 # the packet form's nodes (illustration)
    nodes = src[0][smask[0]].numpy()

    # ---- previous state x_{k-1} (label at the interval's start) and the current label, MANO in camera coordinates
    mano = ManoLayer(ROOT / "assets/mano_right.npz", add_mean=True, dtype=torch.float64)
    prm = torch.tensor(meta[[END_MS - WINDOW, END_MS]], dtype=torch.float64)
    betas = torch.tensor(aux["betas"], dtype=torch.float64).view(1, -1).expand(2, -1)
    with torch.no_grad():
        V, J = mano(betas, prm[:, 3:6], prm[:, 6:51], prm[:, 0:3])
    V, J = V.numpy(), J.numpy()
    faces = mano.f.numpy()
    lbs = mano.weights.numpy()
    face_part = SM.face_part_table(lbs, faces)
    K = aux["camera_K"].astype(np.float64).copy()
    K[:2] *= W / 640.0                                                # the canvas's render scale (346 / 640)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # ---- surface association (PFA): routed point on the previous mesh, its depth, the event ray, the residual
    on_n = on.numpy()
    fv = faces[face.numpy()]                                          # (n, 3)
    b = bary.numpy().astype(np.float64)
    b = b / np.clip(b.sum(-1, keepdims=True), 1e-12, None)
    P = (b[..., None] * V[0][fv]).sum(1)
    z = np.maximum(P[:, 2], 0.05)
    dr = np.stack([(x - cx) / fx, (y - cy) / fy, np.ones(n)], -1)
    delta = z[:, None] * dr - P
    Puv = np.stack([fx * P[:, 0] / P[:, 2] + cx, fy * P[:, 1] / P[:, 2] + cy], -1)
    for arr in (P, z, delta, Puv):
        arr[~on_n] = np.nan

    # ---- the stream before it, for the space-time panel (absolute microseconds of the converted time base)
    s0, s1 = STREAM_MS[0] - 120, STREAM_MS[1] + 60
    q0, q1 = int(off[s0]), int(off[s1 + 1])
    se = np.asarray(ev[q0:q1])
    sms = np.repeat(np.arange(s0, s1 + 1, dtype=np.int64), np.diff(off[s0:s1 + 2]).astype(np.int64))
    st = sms * 1000 + np.asarray(tsub[q0:q1]).astype(np.int64)

    WORK.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        WORK / "data.npz",
        x=x.astype(np.int16), y=y.astype(np.int16), p=p.astype(np.uint8), t_us=us.astype(np.int64),
        prob=prob.numpy().astype(np.float32), on=on_n, eligible=elig.numpy(), part=part.numpy().astype(np.int8),
        coord_cm=coord.numpy().astype(np.float32), face=face.numpy().astype(np.int32), bary=b.astype(np.float32),
        nodes=nodes.astype(np.int64), P=P, z=z, d=dr, delta=delta, Puv=Puv,
        V_prev=V[0], V_cur=V[1], J_prev=J[0], J_cur=J[1], faces=faces.astype(np.int32), lbs=lbs.astype(np.float32),
        face_part=face_part.astype(np.int8), tmpl=net.tmpl.numpy().astype(np.float64), K=K,
        params_prev=meta[END_MS - WINDOW], params_cur=meta[END_MS], betas=aux["betas"],
        sx=se[:, 0].astype(np.int16), sy=se[:, 1].astype(np.int16), sp=np.clip(se[:, 2], 0, 1).astype(np.uint8),
        st_us=st)
    info = {
        "sequence": SEQ, "split": SPLIT, "subject_role": "train (illustration only)",
        "interval_ms": [END_MS - WINDOW, END_MS], "stream_ms": list(STREAM_MS),
        "router": str(CKPT.relative_to(ROOT)), "router_sha256": sha256(CKPT), "router_design": obj.get("design"),
        "tau": TAU, "near_px": NEAR,
        "events": n, "onhand": int(on_n.sum()), "eligible": int(elig.sum()), "nodes": int(len(nodes)),
        "part_hist_onhand": np.bincount(part.numpy()[on_n], minlength=N_PARTS).tolist(),
        "delta_px_median": float(np.nanmedian(np.hypot(x - Puv[:, 0], y - Puv[:, 1]))),
        "delta_mm_median": float(np.nanmedian(np.linalg.norm(delta, axis=-1)) * 1000),
        "label_motion_px_mean": float(np.linalg.norm(
            (V[1] @ K.T)[:, :2] / (V[1] @ K.T)[:, 2:] - (V[0] @ K.T)[:, :2] / (V[0] @ K.T)[:, 2:], axis=-1).mean()),
        "stream_events": int(len(st)),
    }
    (WORK / "data.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))


# ============================================================================================================== render
#: hand-part palette: palm tan; one hue per finger (no pure red / blue: those mean polarity), three shades from
#: the proximal (darker) to the distal (lighter) segment
FINGER_RGB = {0: (204, 172, 128), 1: (222, 172, 32), 2: (58, 160, 100), 3: (214, 96, 160), 4: (138, 108, 204),
              5: (236, 132, 40)}
PANE_FILL = (0.955, 0.962, 0.98)
DROP_RGB, NEAR_RGB = (0.72, 0.74, 0.79), (0.38, 0.40, 0.45)
SLATE = (64 / 255, 68 / 255, 82 / 255)
INSET = dict(cx=209.0, cy=124.0, r=16.0)        # canvas pixels: the thumb and the base of the index finger
ROUTED_CROP = (112, 70, 262, 183)               # canvas pixels x0, y0, x1, y1 of the routed pane (4:3, the hand)
INSET_FINGERS = (1, 5)                          # arrows from index / thumb events
MESH_VIEW = dict(yaw=-14.0, pitch=8.0)          # degrees, about the hand's centre (camera axes: x right, y down)
DPI = 600


def part_rgb():
    out = np.zeros((N_PARTS, 3))
    for p in range(N_PARTS):
        base = np.array(FINGER_RGB[FINGER_OF_PART[p]], float) / 255
        out[p] = base if p == 0 else [base * 0.86, base, base + (1 - base) * 0.32][(p - 1) % 3]
    return out


def straight(premult):
    """(4, h, w) premultiplied -> (h, w, 4) straight-alpha float."""
    a = premult[3]
    rgb = np.where(a[None] > 1e-6, premult[:3] / np.maximum(a[None], 1e-6), 0.0)
    return np.clip(np.concatenate([rgb, a[None]], 0), 0, 1).transpose(1, 2, 0)


def save_png(rgba, path):
    import cv2
    out = (np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA))


def resize(rgba, w, h):
    """Straight-alpha (h, w, 4) resized through premultiplied space (no dark fringes)."""
    import cv2
    pm = rgba.copy()
    pm[..., :3] *= pm[..., 3:4]
    big = pm.shape[1] > w
    pm = cv2.resize(pm.astype(np.float32), (int(w), int(h)), interpolation=cv2.INTER_AREA if big else cv2.INTER_CUBIC)
    pm = np.clip(pm, 0, 1)
    a = pm[..., 3:4]
    return np.concatenate([np.where(a > 1e-6, pm[..., :3] / np.maximum(a, 1e-6), 0), a], -1)


# ---------------------------------------------------------------------------------------------------- event stream
def render_stream(z, ESF, width):
    """The interval's space-time stream (the event-input figure's `events` view) on the undistorted canvas: the
    front pane holds exactly the interval's 50 ms. Returns straight RGBA and anchors in its pixels."""
    t, x, y, p = z["st_us"], z["sx"].astype(np.int64), z["sy"].astype(np.int64), z["sp"]
    t0 = int(t[0])
    s_end = ((END_MS + 1) * 1000 - t0) / 1e6
    cfg = {**ESF.DEFAULT, **ESF.LAYOUTS["events"]}
    cfg.update(t_start=(STREAM_MS[0] * 1000 - t0) / 1e6, t_end=s_end, slices=[s_end], slice_ms=WINDOW,
               xy_labels=False, legend=False, axis_label="", pad=(0, 0, 0, 0), width=width, lens=None)
    canvas, overlays, stats = ESF.render(cfg, (t, x, y, p))
    ESF.over(canvas, overlays["axis"])
    ss = cfg["ss"]
    img = ESF.downsample(canvas, ss)
    y0, y1, x0, x1 = ESF.crop_box(img[3], int(10 * width / 1400), cfg["crop_alpha"])
    img = img[:, y0:y1, x0:x1]
    cam, zt, L, _, _, _ = ESF.frame(cfg)
    zf = zt(s_end)

    def at(P):
        u, v, _ = cam.project(np.atleast_2d(P))
        return [float(u[0] / ss - x0), float(v[0] / ss - y0)]
    ax_, ay_ = cfg["axis_xy"]
    anchors = {"pane_tl": at((-W / 2, H / 2, zf)), "pane_tr": at((W / 2, H / 2, zf)),
               "pane_br": at((W / 2, -H / 2, zf)), "pane_bl": at((-W / 2, -H / 2, zf)),
               "axis_tip": at((ax_, ay_, L / 2 + cfg["axis_z1"] + 0.55 * cfg["axis_pad"])),
               "size": [int(x1 - x0), int(y1 - y0)]}
    return straight(img), anchors, {"events_drawn": stats["events_drawn"], "pane_events": stats["slices"]}


# ---------------------------------------------------------------------------------------------------- 2-D event panes
def event_layer(ESF, tex, x, y, cls, part, ox, oy, S, prt, size=1.0):
    """Discs of one interval's events on a texture (S texels per pixel, window origin (ox, oy)), one disc per pixel
    and class, drawn dropped -> near -> on-hand. cls: 0 dropped, 1 near (kept, q = 0), 2 on-hand (colour = part)."""
    for c, rad, alpha in ((0, 0.55, 0.9), (1, 0.62, 0.95), (2, 0.78, 1.0)):
        m = np.flatnonzero(cls == c)
        if not len(m):
            continue
        key = y[m] * W + x[m]
        _, last = np.unique(key[::-1], return_index=True)              # the latest event of each pixel
        m = m[::-1][last]
        col = (np.tile(DROP_RGB, (len(m), 1)) if c == 0 else np.tile(NEAR_RGB, (len(m), 1)) if c == 1
               else prt[part[m]])
        ESF.draw_discs(tex, (x[m] + 0.5 - ox) * S, (y[m] + 0.5 - oy) * S, np.full(len(m), rad * size * S),
                       col.astype(np.float32), np.full(len(m), alpha))


def render_routed(z, ESF, S, crop=ROUTED_CROP):
    """Route-first selection on the interval's pane (window `crop` of it): on-hand events in part colours, the near
    band dark grey, the rest light grey."""
    import cv2
    x0, y0, x1, y1 = crop
    cw, ch = x1 - x0, y1 - y0
    x, y = z["x"].astype(np.int64), z["y"].astype(np.int64)
    cls = np.where(z["on"], 2, np.where(z["eligible"], 1, 0))
    m = (x >= x0 - 1) & (x <= x1) & (y >= y0 - 1) & (y <= y1)
    pad = int(S)
    tex = np.zeros((4, ch * S + 2 * pad, cw * S + 2 * pad), np.float32)
    tex[:, pad:pad + ch * S, pad:pad + cw * S] = np.array(list(PANE_FILL) + [1.0], np.float32)[:, None, None]
    lay = np.zeros_like(tex)
    event_layer(ESF, lay, x[m], y[m], cls[m], z["part"].astype(np.int64)[m], x0 - pad / S, y0 - pad / S, S,
                part_rgb())
    inner = np.zeros(tex.shape[1:], np.float32)
    inner[pad:pad + ch * S, pad:pad + cw * S] = 1.0
    ESF.over(tex, lay * inner[None])
    b = np.zeros(tex.shape[1:], np.float32)
    cv2.rectangle(b, (pad, pad), (pad + cw * S - 1, pad + ch * S - 1), 1.0, max(1, int(round(0.7 * S))), cv2.LINE_AA)
    ESF.over(tex, ESF.tinted(b, (64, 68, 82), 0.95))
    return straight(tex), pad


def event_map_icon(z, S=2):
    """ESMNet's input of the interval as a tiny picture: ON counts red, OFF counts blue."""
    x, y, p = z["x"].astype(np.int64), z["y"].astype(np.int64), z["p"].astype(np.int64)
    img = np.ones((H, W, 3))
    for pol, col in ((1, (222, 45, 38)), (0, (33, 88, 214))):
        c = np.zeros((H, W))
        np.add.at(c, (y[p == pol], x[p == pol]), 1)
        a = (1 - np.exp(-c / 1.2))[..., None] * 0.9
        img = img * (1 - a) + np.array(col) / 255 * a
    import cv2
    img = cv2.resize(img.astype(np.float32), (W * S, H * S), interpolation=cv2.INTER_NEAREST)
    return np.concatenate([img, np.ones_like(img[..., :1])], -1)


# ---------------------------------------------------------------------------------------------------- meshes
def raster(uv, zz, faces, h, w):
    """Z-buffer triangle rasteriser: depth, face id and barycentrics per pixel (pixel centres at +0.5)."""
    zbuf = np.full((h, w), np.inf)
    fid = np.full((h, w), -1, np.int64)
    bc = np.zeros((h, w, 3))
    for i, (a, b, c) in enumerate(faces):
        pa, pb, pc = uv[a], uv[b], uv[c]
        x0, x1 = max(int(np.floor(min(pa[0], pb[0], pc[0]))), 0), min(int(np.ceil(max(pa[0], pb[0], pc[0]))), w - 1)
        y0, y1 = max(int(np.floor(min(pa[1], pb[1], pc[1]))), 0), min(int(np.ceil(max(pa[1], pb[1], pc[1]))), h - 1)
        if x1 < x0 or y1 < y0:
            continue
        det = (pb[1] - pc[1]) * (pa[0] - pc[0]) + (pc[0] - pb[0]) * (pa[1] - pc[1])
        if abs(det) < 1e-12:
            continue
        X, Y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        l1 = ((pb[1] - pc[1]) * (X - pc[0]) + (pc[0] - pb[0]) * (Y - pc[1])) / det
        l2 = ((pc[1] - pa[1]) * (X - pc[0]) + (pa[0] - pc[0]) * (Y - pc[1])) / det
        l3 = 1 - l1 - l2
        ins = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        if not ins.any():
            continue
        zp = l1 * zz[a] + l2 * zz[b] + l3 * zz[c]
        sub = zbuf[y0:y1 + 1, x0:x1 + 1]
        upd = ins & (zp < sub)
        sub[upd] = zp[upd]
        fid[y0:y1 + 1, x0:x1 + 1][upd] = i
        bc[y0:y1 + 1, x0:x1 + 1][upd] = np.stack([l1, l2, l3], -1)[upd]
    return zbuf, fid, bc


def vertex_normals(V, faces):
    fn = np.cross(V[faces[:, 1]] - V[faces[:, 0]], V[faces[:, 2]] - V[faces[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def rot(yaw, pitch):
    a, b = np.deg2rad(yaw), np.deg2rad(pitch)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return Rx @ Ry


def pixel_parts(fid, bc, faces, lbs):
    """Part of every covered pixel = argmax of the barycentric blend of its face's skinning weights (smooth part
    borders instead of the faces' staircase)."""
    ok = fid >= 0
    w = (bc[ok][..., None] * lbs[faces[fid[ok]]]).sum(1)
    return ok, w.argmax(-1)


def render_mesh(V, faces, lbs, size_px, ss=3, yaw=0.0, pitch=0.0, margin=0.06):
    """Shaded orthographic render of the posed mesh about its centre, coloured by hand part. Returns straight RGBA
    (h, w, 4) and project(P) -> (u, v, visible) in output pixels."""
    from scipy import ndimage
    c = V.mean(0)
    R = rot(yaw, pitch)
    Vv = (V - c) @ R.T
    lo, hi = Vv[:, :2].min(0), Vv[:, :2].max(0)
    span = (hi - lo).max() * (1 + 2 * margin)
    scale = size_px * ss / span
    w = int(np.ceil((hi[0] - lo[0]) * scale + 2 * margin * span * scale))
    h = int(np.ceil((hi[1] - lo[1]) * scale + 2 * margin * span * scale))
    w, h = w + (-w) % ss, h + (-h) % ss
    off = np.array([w, h]) / 2 - (lo + hi) / 2 * scale
    uv = Vv[:, :2] * scale + off
    zbuf, fid, bc = raster(uv, Vv[:, 2], faces, h, w)
    N = vertex_normals(Vv, faces)
    ok = fid >= 0
    n = (bc[ok][..., None] * N[faces[fid[ok]]]).sum(1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    n[n[:, 2] > 0] *= -1                                               # two-sided: face the viewer
    L1 = np.array([-0.45, -0.65, -0.62]); L1 /= np.linalg.norm(L1)
    L2 = np.array([0.70, 0.15, -0.70]); L2 /= np.linalg.norm(L2)
    Hh = L1 + np.array([0, 0, -1.0]); Hh /= np.linalg.norm(Hh)
    d1, d2 = np.clip(n @ L1, 0, 1), np.clip(n @ L2, 0, 1)
    sp = np.clip(n @ Hh, 0, 1) ** 28
    _, pp = pixel_parts(fid, bc, faces, lbs)
    base = part_rgb()[pp] * 0.88 + 0.12
    rgb = np.zeros((h, w, 3))
    rgb[ok] = np.clip(base * (0.50 + 0.42 * d1 + 0.16 * d2)[:, None] + 0.10 * sp[:, None], 0, 1)
    alpha = ok.astype(np.float64)
    edge = alpha - ndimage.binary_erosion(ok, iterations=max(1, int(round(0.9 * ss)))).astype(np.float64)
    rgb = rgb * (1 - 0.55 * edge[..., None]) + np.array(SLATE) * 0.55 * edge[..., None]
    pm = np.concatenate([rgb * alpha[..., None], alpha[..., None]], -1)
    pm = pm.reshape(h // ss, ss, w // ss, ss, 4).mean((1, 3))
    a = pm[..., 3:4]
    img = np.concatenate([np.where(a > 1e-6, pm[..., :3] / np.maximum(a, 1e-6), 0), a], -1)

    def project(P):
        q = (np.atleast_2d(P) - c) @ R.T
        u, v = q[:, 0] * scale + off[0], q[:, 1] * scale + off[1]
        ui = np.clip(u.astype(int), 0, w - 1)
        vi = np.clip(v.astype(int), 0, h - 1)
        vis = q[:, 2] <= zbuf[vi, ui] + 0.004
        return u / ss, v / ss, vis
    return img, project


def render_inset_bg(z, S):
    """The inset's background: the previous-state mesh projected through the camera (light part tint, outline) and
    the interval's events of the window. Window = INSET (centre, half size) in canvas pixels."""
    from scipy import ndimage
    import sys as _s
    _s.path.insert(0, str(ROOT / "tools/real"))
    import event_stream_figure as ESF
    cx0, cy0, r = INSET["cx"], INSET["cy"], INSET["r"]
    ox, oy = cx0 - r, cy0 - r
    n = int(round(2 * r * S))
    tex = np.zeros((4, n, n), np.float32)
    tex[:] = np.array(list(PANE_FILL) + [1.0], np.float32)[:, None, None]
    V, F, K = z["V_prev"], z["faces"].astype(np.int64), z["K"]
    q = V @ K.T
    uv = (q[:, :2] / q[:, 2:] + 0.5 - [ox, oy]) * S                    # event pixel i sits at i + 0.5
    zbuf, fid, bc = raster(uv, V[:, 2], F, n, n)
    ok, pp = pixel_parts(fid, bc, F, z["lbs"])
    prt = part_rgb()
    x, y = z["x"].astype(np.int64), z["y"].astype(np.int64)
    m = (x >= ox - 1) & (x < ox + 2 * r + 1) & (y >= oy - 1) & (y < oy + 2 * r + 1)
    m &= z["on"]                                                       # the inset shows the on-hand events only
    cls = np.full(len(x), 2)
    tint = np.zeros((4, n, n), np.float32)                             # previous-state surface, lightly tinted
    tint[:3, ok] = (prt[pp] * 0.26).T
    tint[3, ok] = 0.26
    ESF.over(tex, tint)
    ev = np.zeros_like(tex)
    event_layer(ESF, ev, x[m], y[m], cls[m], z["part"].astype(np.int64)[m], ox, oy, S, prt, size=0.58)
    ev *= 0.85                                                         # events a little lighter: the arrows lead
    ESF.over(tex, ev)
    edge = ok & ~ndimage.binary_erosion(ok, iterations=max(1, int(round(0.07 * S))))
    ESF.over(tex, ESF.tinted(edge.astype(np.float32), (64, 68, 82), 0.75))
    return straight(tex)


# ---------------------------------------------------------------------------------------------------- selections
def pick_correspondences(z, project):
    """One event per finger (its distal part) and one on the palm: the event whose routed point P_i is nearest the
    median of that part's visible routed points."""
    on, part, P = z["on"], z["part"].astype(np.int64), z["P"]
    out = []
    for f, p in ((0, 0), (1, 3), (2, 6), (3, 9), (4, 12), (5, 15)):
        m = np.flatnonzero(on & (part == p) & np.isfinite(P[:, 0]))
        if not len(m):
            continue
        _, _, vis = project(P[m])
        m = m[vis]
        if not len(m):
            continue
        med = np.median(P[m], 0)
        out.append(int(m[np.argmin(np.linalg.norm(P[m] - med, axis=1))]))
    return out


def pick_arrows(z, k=9, min_sep=4.0):
    """Residual arrows of the inset: on-hand events of INSET_FINGERS in the window, |delta| 1.5-6 px, spread out."""
    cx0, cy0, r = INSET["cx"], INSET["cy"], INSET["r"]
    x, y, Puv = z["x"].astype(float) + 0.5, z["y"].astype(float) + 0.5, z["Puv"] + 0.5
    fop = np.array(FINGER_OF_PART)[z["part"].astype(np.int64)]
    d = np.hypot(x - Puv[:, 0], y - Puv[:, 1])
    inside = (np.hypot(x - cx0, y - cy0) < 0.78 * r) & (np.hypot(Puv[:, 0] - cx0, Puv[:, 1] - cy0) < 0.78 * r)
    m = np.flatnonzero(z["on"] & np.isin(fop, INSET_FINGERS) & inside & (d > 1.5) & (d < 6.0))
    if not len(m):
        return []
    pts = np.stack([x[m], y[m]], 1)
    sel = [int(np.argmin(np.hypot(*(pts - [cx0, cy0]).T)))]
    dist = np.hypot(*(pts - pts[sel[0]]).T)
    while len(sel) < k:
        j = int(np.argmax(dist))
        if dist[j] < min_sep:
            break
        sel.append(j)
        dist = np.minimum(dist, np.hypot(*(pts - pts[j]).T))
    return [int(m[j]) for j in sel]


# ---------------------------------------------------------------------------------------------------- composition
def snowflake(ax, cx, cy, r, color, lw):
    for k in range(3):
        a = np.deg2rad(90 + 60 * k)
        dx, dy = np.cos(a) * r, np.sin(a) * r
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=color, lw=lw, solid_capstyle="round", zorder=12)
        for s in (-1, 1):
            tx, ty = cx + s * dx * 0.62, cy + s * dy * 0.62
            for b in (-1, 1):
                bb = a + b * np.deg2rad(40) + (np.pi if s < 0 else 0)
                ax.plot([tx, tx + np.cos(bb) * r * 0.32], [ty, ty + np.sin(bb) * r * 0.32], color=color, lw=lw * 0.9,
                        solid_capstyle="round", zorder=12)


def compose(parts, z, sel, out_base, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
    sys.path.insert(0, str(ROOT / "tools/real"))
    import event_stream_figure as ESF

    plt.rcParams.update({"font.family": "Liberation Sans", "mathtext.fontset": "stix", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "font.size": 7, "axes.unicode_minus": False})
    Wmm, Hmm = 140.0, 78.0
    fig = plt.figure(figsize=(Wmm / 25.4, Hmm / 25.4), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, Wmm)
    ax.set_ylim(Hmm, 0)
    ax.axis("off")
    ink, soft = SLATE, (0.43, 0.45, 0.51)
    green = (0.33, 0.58, 0.41)
    prt = part_rgb()
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]

    def place(rgba, x0, y0, w=None, h=None, zo=3):
        hh, ww = rgba.shape[:2]
        if w is None:
            w = h * ww / hh
        if h is None:
            h = w * hh / ww
        px_w, px_h = max(2, int(round(w / 25.4 * dpi))), max(2, int(round(h / 25.4 * dpi)))
        im = ax.imshow(resize(rgba, px_w, px_h), extent=(x0, x0 + w, y0 + h, y0), interpolation="none", zorder=zo)
        return im, (x0, y0, w, h)

    def arrow(p0, p1, dashed=False, lw=0.7, col=ink, zo=6, head=5.0, clip=None):
        a = FancyArrowPatch(p0, p1, arrowstyle="-|>,head_length=0.55,head_width=0.28", mutation_scale=head, lw=lw,
                            color=col, shrinkA=0, shrinkB=0, zorder=zo, joinstyle="miter", capstyle="butt",
                            linestyle=(0, (2.4, 1.6)) if dashed else "-")
        ax.add_patch(a)
        if clip is not None:
            a.set_clip_path(clip)
        return a

    def box(x0, y0, w, h, fc, ec, dashed=False, lw=0.6, r=1.1, zo=4):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                    linestyle=(0, (2.2, 1.4)) if dashed else "-", zorder=zo))

    def text(x, y, s, size=7, col=ink, ha="center", va="center", zo=14, **kw):
        return ax.text(x, y, s, fontsize=size, color=col, ha=ha, va=va, zorder=zo, **kw)

    # ---- panel frame, title, row labels
    ax.add_patch(FancyBboxPatch((0.6, 0.6), Wmm - 1.2, Hmm - 1.2, boxstyle="round,pad=0,rounding_size=2.2",
                                fc=(0.972, 0.976, 0.986), ec=(0.82, 0.84, 0.88), lw=0.6, zorder=0))
    text(3.2, 5.0, "(a)  Event-to-surface association", 8.2, ha="left", fontweight="bold")
    text(3.2, 10.2, "Where on the hand?  — from the events alone", 6.4, soft, ha="left", style="italic")
    text(3.2, 49.4, "Where in space?  — from the previous state", 6.4, soft, ha="left", style="italic")

    # ---- row 1: event stream
    _, st_box = place(parts["stream"], 2.0, 12.4, h=29.6)
    sa = parts["stream_anchors"]
    smm = lambda q: (st_box[0] + q[0] / sa["size"][0] * st_box[2], st_box[1] + q[1] / sa["size"][1] * st_box[3])
    tip = smm(sa["axis_tip"])
    text(tip[0] + 0.9, tip[1] + 0.2, "$t$", 8.0, ha="left")
    ptl, ptr_, pbr = smm(sa["pane_tl"]), smm(sa["pane_tr"]), smm(sa["pane_br"])
    text((ptl[0] + ptr_[0]) / 2, (ptl[1] + ptr_[1]) / 2 - 1.5, "interval $k$ (50 ms)", 5.4, soft, style="italic",
         path_effects=halo)
    text(st_box[0] + st_box[2] / 2, st_box[1] + st_box[3] + 2.5, r"events  $e_i=(u_i,\,v_i,\,t_i,\,p_i)$", 6.0)
    pr_mid = ((ptr_[0] + pbr[0]) / 2, (ptr_[1] + pbr[1]) / 2)

    # ---- ESMNet (frozen, run once per interval on the interval's event map, one output per event)
    ex0, ey0, ew, eh = 38.4, 17.6, 18.6, 17.6
    box(ex0, ey0, ew, eh, (0.905, 0.928, 0.958), (0.50, 0.56, 0.66))
    text(ex0 + ew / 2 - 1.5, ey0 + 3.1, "ESMNet", 7.4, fontweight="bold")
    snowflake(ax, ex0 + ew - 2.3, ey0 + 3.0, 1.2, (0.30, 0.45, 0.66), 0.5)
    iy = ey0 + 11.0
    ax.add_patch(FancyBboxPatch((ex0 + 2.0, iy - 2.9), 4.4, 3.3, boxstyle="square,pad=0", fc=(0.86, 0.88, 0.92),
                                ec=(0.55, 0.58, 0.65), lw=0.35, zorder=5))
    place(parts["map_icon"], ex0 + 1.3, iy - 2.2, w=4.4, h=3.3, zo=6)
    ax.add_patch(FancyBboxPatch((ex0 + 1.3, iy - 2.2), 4.4, 3.3, boxstyle="square,pad=0", fc="none",
                                ec=(0.55, 0.58, 0.65), lw=0.35, zorder=7))
    tx0, tx1 = ex0 + 7.3, ex0 + 12.5
    ax.add_patch(Polygon([(tx0, iy - 3.5), (tx1, iy - 1.9), (tx1, iy + 1.9), (tx0, iy + 3.5)], closed=True,
                         fc=(0.77, 0.82, 0.89), ec=(0.45, 0.52, 0.63), lw=0.45, zorder=5))
    text((tx0 + tx1) / 2, iy, "CNN", 4.9)
    arrow((ex0 + 5.9, iy), (tx0 - 0.15, iy), lw=0.45, head=3.6)
    hx0 = ex0 + 13.7
    box(hx0, iy - 2.2, 3.6, 4.4, (0.77, 0.82, 0.89), (0.45, 0.52, 0.63), lw=0.45, r=0.6, zo=5)
    text(hx0 + 1.8, iy, "MLP", 4.5)
    arrow((tx1 + 0.1, iy), (hx0 - 0.1, iy), lw=0.45, head=3.6)
    text(ex0 + ew / 2, ey0 + eh + 2.5, r"$o_i$,  part,  $\mathbf{c}_i$  per event", 5.8, soft)
    yin = pr_mid[1] if ey0 + 2.5 < pr_mid[1] < ey0 + eh - 2.5 else ey0 + eh / 2
    arrow((pr_mid[0] + 0.7, yin), (ex0 - 0.2, yin))

    # ---- routed events: the interval's pane after route-first selection
    _, rp_box = place(parts["routed"], 65.0, 12.2, w=41.0)
    rpad, S_r = parts["routed_pad"], parts["routed_S"]
    cx0_, cy0_, cx1_, cy1_ = ROUTED_CROP
    tot_w, tot_h = (cx1_ - cx0_) * S_r + 2 * rpad, (cy1_ - cy0_) * S_r + 2 * rpad

    def pane_mm(u, v):
        return (rp_box[0] + ((u - cx0_) * S_r + rpad) / tot_w * rp_box[2],
                rp_box[1] + ((v - cy0_) * S_r + rpad) / tot_h * rp_box[3])
    import cv2
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]),
                                     np.float32([ptl, ptr_, pbr, smm(sa["pane_bl"])]))
    q = cv2.perspectiveTransform(np.float32([[[cx0_, cy0_], [cx1_, cy0_], [cx1_, cy1_], [cx0_, cy1_]]]), Hm)[0]
    ax.add_patch(Polygon(q, closed=True, fc="none", ec=ink, lw=0.5, linestyle=(0, (1.6, 1.0)), zorder=8))
    ym = ey0 + eh / 2
    arrow((ex0 + ew + 0.3, ym), (rp_box[0] + 0.25, ym))
    gx = (ex0 + ew + rp_box[0]) / 2
    text(gx, ym - 2.3, "route", 5.6, soft, style="italic")
    text(gx, ym + 2.3, "first", 5.6, soft, style="italic")
    lx = rp_box[0] + rp_box[2] + 2.4                                   # legend, right of the pane
    for j, p in enumerate((3, 6, 12, 9, 15, 0)):
        ax.add_patch(Circle((lx + (j % 3) * 0.95, 13.4 + (j // 3) * 0.95), 0.38, fc=prt[p], ec="none", zorder=12))
    text(lx + 3.2, 13.9, "on hand, colour = part", 5.4, ha="left")
    text(lx + 3.2, 16.3, "($q_i{=}1$)", 5.4, soft, ha="left")
    ax.add_patch(Circle((lx + 0.95, 19.6), 0.40, fc=NEAR_RGB, ec="none", zorder=12))
    text(lx + 3.2, 19.6, r"near, $\leq$16 px ($q_i{=}0$)", 5.4, ha="left")
    ax.add_patch(Circle((lx + 0.95, 22.6), 0.40, fc=DROP_RGB, ec="none", zorder=12))
    text(lx + 3.2, 22.6, "dropped", 5.4, ha="left")

    # ---- row 2: previous state -> MANO -> previous-state mesh
    yb = 62.4
    box(4.6, yb - 3.9, 11.4, 7.8, (0.885, 0.945, 0.905), (0.37, 0.62, 0.45))
    text(10.3, yb, r"$\mathbf{x}_{k-1}$", 8.4)
    arrow((10.3, Hmm - 1.8), (10.3, yb + 4.0), dashed=True, col=green)
    text(11.6, Hmm - 3.8, "from step $k\\!-\\!1$", 5.0, (0.28, 0.50, 0.35), ha="left", style="italic")
    box(21.0, yb - 3.9, 12.6, 7.8, "white", (0.48, 0.50, 0.56), dashed=True)
    text(27.3, yb - 0.9, "MANO", 7.2)
    text(27.3, yb + 2.1, "no parameters", 4.6, soft)
    arrow((16.2, yb), (20.8, yb))
    hand_c = pane_mm(*np.nanmedian(np.stack([z["x"][z["on"]], z["y"][z["on"]]], 1).astype(float), 0) + 0.5)
    mh = 24.6
    mw = mh * parts["mesh"].shape[1] / parts["mesh"].shape[0]
    _, me_box = place(parts["mesh"], hand_c[0] - mw / 2, 51.0, h=mh)
    arrow((33.8, yb), (me_box[0] - 0.5, yb))
    text((33.8 + me_box[0]) / 2, yb - 1.9, "posed mesh", 5.4, soft, style="italic")
    text(me_box[0] + me_box[2] + 0.6, me_box[1] + me_box[3] - 2.4, "previous-state\nmesh", 5.4, soft, ha="left",
         linespacing=1.1)

    # ---- surface association: each on-hand event's surface address on the previous-state mesh
    proj = parts["mesh_project"]
    msz = parts["mesh"].shape
    xs = []
    for i in sel["corr"]:
        pa = pane_mm(z["x"][i] + 0.5, z["y"][i] + 0.5)
        u, v, _ = proj(z["P"][i])
        pb = (me_box[0] + u[0] / msz[1] * me_box[2], me_box[1] + v[0] / msz[0] * me_box[3])
        xs += [pa[0], pb[0]]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=prt[z["part"][i]] * 0.80, lw=0.55, alpha=0.85, zorder=9,
                solid_capstyle="round")
        for q in (pa, pb):
            ax.add_patch(Circle(q, 0.45, fc=prt[z["part"][i]], ec="white", lw=0.4, zorder=10))
    lx2 = min(xs) - 1.4
    text(lx2, 46.6, "surface association", 6.6, ha="right", fontweight="bold")
    text(lx2, 49.6, r"$(f_i,\mathbf{a}_i)\ \mapsto\ \mathbf{P}_i$  on the mesh", 6.0, ha="right")

    # ---- inset: the residual delta_i on the image plane (window around the thumb)
    C, Rr = (122.4, 46.0), 11.6
    circ = Circle(C, Rr, transform=ax.transData)
    im, _ = place(parts["inset"], C[0] - Rr, C[1] - Rr, w=2 * Rr, h=2 * Rr, zo=7)
    im.set_clip_path(circ)
    ax.add_patch(Circle(C, Rr, fc="none", ec=ink, lw=0.8, zorder=11))
    cx0, cy0, r = INSET["cx"], INSET["cy"], INSET["r"]
    ins_mm = lambda u, v: (C[0] + (u - cx0) / r * Rr, C[1] + (v - cy0) / r * Rr)
    for i in sel["arrows"]:
        p0 = ins_mm(*(z["Puv"][i] + 0.5))
        p1 = ins_mm(z["x"][i] + 0.5, z["y"][i] + 0.5)
        arrow(p0, p1, lw=0.65, head=4.4, zo=12, clip=circ)
        ax.add_patch(Circle(p0, 0.42, fc="white", ec=ink, lw=0.45, zorder=13))
    if sel["arrows"]:
        ends = [(np.array(ins_mm(*(z["Puv"][i] + 0.5))), np.array(ins_mm(z["x"][i] + 0.5, z["y"][i] + 0.5)))
                for i in sel["arrows"]]
        best, room = 0, -1.0
        for j, (a0, a1) in enumerate(ends):
            if np.hypot(*(a0 - C)) > 0.62 * Rr or np.hypot(*(a1 - C)) > 0.62 * Rr:
                continue
            others = [q for k, e in enumerate(ends) if k != j for q in e]
            r_ = min(min(np.hypot(*(a0 - q)), np.hypot(*(a1 - q))) for q in others) if others else 9.0
            if r_ > room:
                best, room = j, r_
        a0, a1 = ends[best]
        d = (a1 - a0) / max(np.hypot(*(a1 - a0)), 1e-9)
        nrm = np.array([-d[1], d[0]])
        nrm = nrm if nrm[0] < 0 else -nrm                              # labels on the arrow's left side
        lp, le = a0 - d * 1.5 + nrm * 1.3, a1 + d * 1.3 + nrm * 1.0
        text(lp[0], lp[1], r"$\mathbf{P}_i$", 6.4, path_effects=halo)
        text(le[0], le[1], r"$e_i$", 6.4, path_effects=halo)
    text(C[0], C[1] + Rr + 2.6, r"$\boldsymbol{\delta}_i=z_i\,\mathbf{d}_i-\mathbf{P}_i$", 7.4)
    text(C[0], C[1] + Rr + 5.5, "residual, no parameters", 4.9, soft)
    zc = pane_mm(cx0, cy0)                                             # zoom cone from the pane
    zr = r * S_r / tot_w * rp_box[2]
    ax.add_patch(Circle(zc, zr, fc="none", ec=ink, lw=0.55, zorder=11))
    for a, b in ESF.tangents(zc, zr, C, Rr):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=ink, lw=0.4, alpha=0.7, zorder=10)
    arrow((C[0] + Rr + 0.4, C[1]), (Wmm - 1.2, C[1]), lw=0.7)
    text((C[0] + Rr + Wmm) / 2 - 0.2, C[1] - 2.3, "to (b)", 5.2, soft)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base) + ".png", dpi=dpi, facecolor="white")
    fig.savefig(str(out_base) + ".pdf", facecolor="none")
    fig.savefig(str(out_base) + ".svg", facecolor="none")
    plt.close(fig)


def stage_render(preview=False):
    sys.path.insert(0, str(ROOT / "tools/real"))
    import event_stream_figure as ESF
    z = dict(np.load(WORK / "data.npz"))
    out = (WORK / "preview") if preview else OUT
    (out / "assets").mkdir(parents=True, exist_ok=True)
    S_r = 6 if preview else 12
    stream, anchors, sstat = render_stream(z, ESF, 1000 if preview else 2600)
    routed, rpad = render_routed(z, ESF, S_r)
    mesh, project = render_mesh(z["V_prev"], z["faces"].astype(np.int64), z["lbs"].astype(np.float64),
                                700 if preview else 1500, ss=2 if preview else 3, **MESH_VIEW)
    inset = render_inset_bg(z, 22 if preview else 48)
    sel = {"corr": pick_correspondences(z, project), "arrows": pick_arrows(z)}
    parts = {"stream": stream, "stream_anchors": anchors, "routed": routed, "routed_pad": rpad, "routed_S": S_r,
             "mesh": mesh, "mesh_project": project, "inset": inset, "map_icon": event_map_icon(z)}
    for k in ("stream", "routed", "mesh", "inset"):
        save_png(parts[k], out / "assets" / f"{k}.png")
    compose(parts, z, sel, out / "pipeline_part_a", 300 if preview else DPI)
    info = json.loads((WORK / "data.json").read_text())
    info.update(stream=sstat, inset_window_px=INSET, mesh_view_deg=MESH_VIEW,
                correspondences=[{"event": i, "part": int(z["part"][i]), "u": int(z["x"][i]), "v": int(z["y"][i])}
                                 for i in sel["corr"]],
                residual_arrows=[{"event": i, "part": int(z["part"][i]),
                                  "delta_px": [float(z["x"][i] - z["Puv"][i][0]), float(z["y"][i] - z["Puv"][i][1])]}
                                 for i in sel["arrows"]])
    (out / "pipeline_part_a.json").write_text(json.dumps(info, indent=1) + "\n")
    print("wrote", out, "correspondences", len(sel["corr"]), "arrows", len(sel["arrows"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["data", "render"])
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    stage_data() if a.stage == "data" else stage_render(a.preview)
