"""EVF virtual frame (docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104): the previous state's MANO mesh rendered into the
240x180 event grid, and what an event node reads from it at its own pixel.

Everything here is a deterministic function of the previous state (no parameters, no gradient):

  rasterize     exact z-buffered coverage of the pixel centres (integer coordinates, the convention of the event grid and
                of S43's vertex splat) by the projected MANO triangles, front and back faces alike (the wrist opening is
                visible); per-face candidate windows of 4 / 8 / 16 / 32 pixels by the face's bounding box, a face wider
                than 32 pixels is clipped and counted. Depth is perspective-correct.
  contour seeds the silhouette boundary (silhouette pixels with a 4-neighbour outside) and the occluding contours (also
                silhouette pixels in front of a 4-neighbour more than CONTOUR_DZ deeper: a finger over the palm)
  jfa           jump flooding (steps 32 .. 1, then 1 again): the nearest seed of every pixel within FAR_PX; farther pixels
                have no seed
  sample        for event-node pixels: inside / signed distance to the silhouette boundary (positive outside) and its
                outward direction; distance to the nearest occluding contour and the direction from it; at the surface
                point under the pixel (or, outside, at the nearest silhouette pixel) and at the nearest contour pixel:
                depth relative to the root joint and template coordinates (rest pose, wrist-centred), optionally the
                LBS weights there
  moments       mean and covariance of the silhouette pixels relative to the projected root, in units of the hand scale
                s = f * 0.1 m / z_root (pixels per 10 cm at the root depth)

The previous state only ever enters as these readable features: no event is assigned to anything by it.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

WINDOWS = (4, 8, 16, 32)
JFA_STEPS = (32, 16, 8, 4, 2, 1, 1)
FAR_PX = 64.0
CONTOUR_DZ = 0.008          # metres: depth step that makes the nearer pixel an occluding-contour pixel
Z_MIN = 1e-3                # metres: faces with a vertex nearer than this are not drawn
FACE_BITS = 12
ZQ = 1e5                    # depth quantisation of the z-buffer key (10 micrometres)
SENTINEL = torch.iinfo(torch.int64).max
BARY_EPS = 1e-6
HAND_SCALE_M = 0.1          # the hand scale s is the pixel size of HAND_SCALE_M at the root depth


@torch.no_grad()
def project(verts: torch.Tensor, fx, fy, cx, cy) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pinhole projection into the event grid; `verts (B, V, 3)` metres, intrinsics `(B,)`."""
    z = verts[..., 2]
    zs = z.clamp_min(1e-6)
    u = fx[:, None] * verts[..., 0] / zs + cx[:, None]
    v = fy[:, None] * verts[..., 1] / zs + cy[:, None]
    return u, v, z


@torch.no_grad()
def rasterize(u: torch.Tensor, v: torch.Tensor, z: torch.Tensor, faces: torch.Tensor, H: int, W: int):
    """`(face (B, H, W) long, -1 = background; clipped faces (B,) long)`."""
    B = u.shape[0]
    dev = u.device
    f = faces.long()
    nf = f.shape[0]
    assert nf < (1 << FACE_BITS)
    pu, pv, pz = u[:, f], v[:, f], z[:, f]                                         # (B, F, 3)
    umin, umax = pu.amin(-1).ceil(), pu.amax(-1).floor()
    vmin, vmax = pv.amin(-1).ceil(), pv.amax(-1).floor()
    area = (pu[..., 1] - pu[..., 0]) * (pv[..., 2] - pv[..., 0]) - (pv[..., 1] - pv[..., 0]) * (pu[..., 2] - pu[..., 0])
    ok = (pz > Z_MIN).all(-1) & (umax >= umin) & (vmax >= vmin) & (umax >= 0) & (umin <= W - 1) & \
        (vmax >= 0) & (vmin <= H - 1) & (area.abs() > 1e-12)
    span = torch.maximum(umax - umin, vmax - vmin) + 1
    clipped = (ok & (span > WINDOWS[-1])).sum(1)
    keys = torch.full((B * H * W,), SENTINEL, dtype=torch.int64, device=dev)
    lo = 0
    for win in WINDOWS:
        sel = ok & (span > lo) & ((span <= win) if win != WINDOWS[-1] else torch.ones_like(ok))
        lo = win
        idx = sel.nonzero()
        if idx.numel() == 0:
            continue
        b, fi = idx[:, 0], idx[:, 1]
        offs = torch.arange(win, device=dev, dtype=pu.dtype)
        px = umin[b, fi][:, None, None] + offs[None, None, :]                       # (M, 1, win)
        py = vmin[b, fi][:, None, None] + offs[None, :, None]                       # (M, win, 1)
        px, py = px.expand(-1, win, win), py.expand(-1, win, win)
        inb = (px <= umax[b, fi][:, None, None]) & (py <= vmax[b, fi][:, None, None]) & \
            (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)
        u0, u1, u2 = (pu[b, fi, k][:, None, None] for k in range(3))
        v0, v1, v2 = (pv[b, fi, k][:, None, None] for k in range(3))
        a = area[b, fi][:, None, None]
        b0 = ((u2 - u1) * (py - v1) - (v2 - v1) * (px - u1)) / a
        b1 = ((u0 - u2) * (py - v2) - (v0 - v2) * (px - u2)) / a
        b2 = 1.0 - b0 - b1
        inside = inb & (b0 >= -BARY_EPS) & (b1 >= -BARY_EPS) & (b2 >= -BARY_EPS)
        z0, z1, z2 = (pz[b, fi, k][:, None, None] for k in range(3))
        zp = 1.0 / (b0 / z0 + b1 / z1 + b2 / z2)
        key = ((zp * ZQ).round().to(torch.int64) << FACE_BITS) | fi[:, None, None]
        pix = (b[:, None, None] * (H * W) + py.long() * W + px.long())
        keys.scatter_reduce_(0, pix[inside], key[inside], reduce="amin")
    face = torch.where(keys == SENTINEL, torch.full_like(keys, -1), keys & ((1 << FACE_BITS) - 1))
    return face.view(B, H, W), clipped


@torch.no_grad()
def surface_at(u, v, z, faces, b: torch.Tensor, face: torch.Tensor, px: torch.Tensor, py: torch.Tensor):
    """Perspective-correct barycentrics `(M, 3)` and depth `(M,)` of face `face` at pixel centres (px, py) of samples
    `b`; rows with face < 0 return zeros."""
    f = faces.long()[face.clamp_min(0)]                                             # (M, 3)
    pu, pv, pz = u[b[:, None], f], v[b[:, None], f], z[b[:, None], f]               # (M, 3)
    a = (pu[:, 1] - pu[:, 0]) * (pv[:, 2] - pv[:, 0]) - (pv[:, 1] - pv[:, 0]) * (pu[:, 2] - pu[:, 0])
    a = torch.where(a.abs() > 1e-12, a, torch.ones_like(a))
    b0 = ((pu[:, 2] - pu[:, 1]) * (py - pv[:, 1]) - (pv[:, 2] - pv[:, 1]) * (px - pu[:, 1])) / a
    b1 = ((pu[:, 0] - pu[:, 2]) * (py - pv[:, 2]) - (pv[:, 0] - pv[:, 2]) * (px - pu[:, 2])) / a
    bs = torch.stack([b0, b1, 1.0 - b0 - b1], -1).clamp_min(0.0)
    bs = bs / bs.sum(-1, keepdim=True).clamp_min(1e-12)
    w = bs / pz.clamp_min(1e-6)
    depth = 1.0 / w.sum(-1).clamp_min(1e-12)
    bary = w * depth[:, None]
    ok = (face >= 0).to(bary.dtype)
    return bary * ok[:, None], depth * ok


def _shift(t: torch.Tensor, dy: int, dx: int, fill: int) -> torch.Tensor:
    """out[..., y, x] = t[..., y + dy, x + dx], `fill` outside."""
    k = max(abs(dy), abs(dx))
    if k == 0:
        return t
    H, W = t.shape[-2:]
    p = F.pad(t, (k, k, k, k), value=fill)
    return p[..., k + dy:k + dy + H, k + dx:k + dx + W]


BIG = torch.iinfo(torch.int32).max

try:                                                  # the fused GPU pass; the torch pass below is its reference
    import triton
    import triton.language as tl

    @triton.jit
    def _jfa_pass_kernel(sy_in, sx_in, best_in, sy_out, sx_out, best_out, H, W, k, total, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offs < total
        hw = H * W
        b = offs // hw
        rem = offs % hw
        y = rem // W
        x = rem % W
        best = tl.load(best_in + offs, mask=valid, other=2147483647)
        by = tl.load(sy_in + offs, mask=valid, other=-1)
        bx = tl.load(sx_in + offs, mask=valid, other=-1)
        for i in tl.static_range(3):
            for j in tl.static_range(3):
                if i * 3 + j != 4:
                    yy = y + (i - 1) * k
                    xx = x + (j - 1) * k
                    ok = valid & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
                    idx = b * hw + yy * W + xx
                    cy = tl.load(sy_in + idx, mask=ok, other=-1)
                    cx = tl.load(sx_in + idx, mask=ok, other=-1)
                    d = (cy - y) * (cy - y) + (cx - x) * (cx - x)
                    d = tl.where(cy >= 0, d, 2147483647)
                    better = d < best
                    best = tl.where(better, d, best)
                    by = tl.where(better, cy, by)
                    bx = tl.where(better, cx, bx)
        tl.store(sy_out + offs, by, mask=valid)
        tl.store(sx_out + offs, bx, mask=valid)
        tl.store(best_out + offs, best, mask=valid)
except Exception:                                     # no Triton: the torch pass everywhere
    triton = None


def _jfa_pass_torch(sy, sx, best, k, yy, xx):
    """One jump-flooding pass: every candidate is read from the pass's input maps; the candidates of a pixel are
    taken in the order (dy, dx) = (-k, -k), (-k, 0), ... (k, k), the first strictly nearer one wins."""
    ny, nx, nb = sy, sx, best
    for dy in (-k, 0, k):
        for dx in (-k, 0, k):
            if dy == 0 and dx == 0:
                continue
            cy, cx = _shift(sy, dy, dx, -1), _shift(sx, dy, dx, -1)
            d = (cy - yy).square() + (cx - xx).square()
            d = torch.where(cy >= 0, d, torch.full_like(d, BIG))
            better = d < nb
            ny, nx, nb = torch.where(better, cy, ny), torch.where(better, cx, nx), torch.where(better, d, nb)
    return ny, nx, nb


@torch.no_grad()
def jfa(seed: torch.Tensor, steps=JFA_STEPS, fused=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nearest seed `(sy, sx)` (int32, -1 = none within reach) and squared distance (int32) of every pixel. The fused
    Triton pass (CUDA) and the torch pass give the same integers."""
    B, H, W = seed.shape
    dev = seed.device
    yy = torch.arange(H, device=dev, dtype=torch.int32).view(1, H, 1).expand(B, H, W)
    xx = torch.arange(W, device=dev, dtype=torch.int32).view(1, 1, W).expand(B, H, W)
    sy = torch.where(seed, yy, torch.full_like(yy, -1)).contiguous()
    sx = torch.where(seed, xx, torch.full_like(xx, -1)).contiguous()
    best = torch.where(seed, torch.zeros_like(yy), torch.full_like(yy, BIG)).contiguous()
    fused = (triton is not None and dev.type == "cuda") if fused is None else fused
    if fused:
        total = B * H * W
        block = 512
        grid = ((total + block - 1) // block,)
        for k in steps:
            ny, nx, nb = torch.empty_like(sy), torch.empty_like(sx), torch.empty_like(best)
            _jfa_pass_kernel[grid](sy, sx, best, ny, nx, nb, H, W, int(k), total, BLOCK=block)
            sy, sx, best = ny, nx, nb
    else:
        for k in steps:
            sy, sx, best = _jfa_pass_torch(sy, sx, best, int(k), yy, xx)
    far = best.float() > FAR_PX ** 2
    sy, sx = torch.where(far, torch.full_like(sy, -1), sy), torch.where(far, torch.full_like(sx, -1), sx)
    return sy, sx, torch.where(far, torch.full_like(best, BIG), best)


@torch.no_grad()
def contour_seeds(face: torch.Tensor, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(silhouette boundary, occluding contour) pixel masks `(B, H, W)`; the second contains the first."""
    sil = face >= 0
    out_nb = torch.zeros_like(sil)
    occ_nb = torch.zeros_like(sil)
    big = torch.finfo(depth.dtype).max
    d = torch.where(sil, depth, torch.full_like(depth, big))
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        n_sil = _shift(sil, dy, dx, False)
        out_nb |= ~n_sil
        n_d = _shift(d, dy, dx, 0.0)
        occ_nb |= n_sil & (n_d > d + CONTOUR_DZ)
    boundary = sil & out_nb
    return boundary, boundary | (sil & occ_nb)


class Frame:
    """The rendered previous state of one batch: per-pixel face index, JFA maps, mesh projection, root and scale."""

    def __init__(self, verts, joints, faces, fx, fy, cx, cy, H, W):
        with torch.autocast(device_type=verts.device.type, enabled=False):
            verts, joints = verts.float(), joints.float()
            self.H, self.W = int(H), int(W)
            self.faces = faces
            self.u, self.v, self.z = project(verts, fx, fy, cx, cy)
            self.face, self.clipped = rasterize(self.u, self.v, self.z, faces, self.H, self.W)
            B = verts.shape[0]
            sil = self.face >= 0
            # depth image of the silhouette pixels (for the contour test)
            dev = verts.device
            yy, xx = torch.meshgrid(torch.arange(self.H, device=dev), torch.arange(self.W, device=dev), indexing="ij")
            bi, py, px = sil.nonzero(as_tuple=True)
            depth = torch.zeros(B, self.H, self.W, device=dev)
            if bi.numel():
                _, dz = surface_at(self.u, self.v, self.z, faces, bi, self.face[bi, py, px], px.float(), py.float())
                depth[bi, py, px] = dz
            self.boundary, self.contour = contour_seeds(self.face, depth)
            self.s1 = jfa(self.boundary)
            self.s2 = jfa(self.contour)
            ru, rv, rz = project(joints[:, :1], fx, fy, cx, cy)
            self.root_u, self.root_v = ru[:, 0], rv[:, 0]
            self.root_z = rz[:, 0].clamp_min(0.05)
            self.scale = (fx * fy).sqrt() * HAND_SCALE_M / self.root_z                     # pixels per 10 cm
            # silhouette moments in hand-scale units relative to the projected root
            m = sil.float()
            n = m.sum((1, 2)).clamp_min(1.0)
            xr = (xx.float()[None] - self.root_u[:, None, None]) / self.scale[:, None, None]
            yr = (yy.float()[None] - self.root_v[:, None, None]) / self.scale[:, None, None]
            mx, my = (m * xr).sum((1, 2)) / n, (m * yr).sum((1, 2)) / n
            cxx = (m * (xr - mx[:, None, None]).square()).sum((1, 2)) / n
            cyy = (m * (yr - my[:, None, None]).square()).sum((1, 2)) / n
            cxy = (m * (xr - mx[:, None, None]) * (yr - my[:, None, None])).sum((1, 2)) / n
            self.sil_area = m.sum((1, 2))
            self.sil_moments = torch.stack([mx, my, cxx, cyy, cxy], -1)


def _seed_lookup(seed_maps, b, py, px):
    sy, sx, best = seed_maps
    ys, xs, d2 = sy[b, py, px], sx[b, py, px], best[b, py, px]
    ok = ys >= 0
    d = torch.where(ok, d2.float().sqrt(), torch.full_like(d2, 0, dtype=torch.float32) + FAR_PX)
    return ys.long(), xs.long(), d, ok


@torch.no_grad()
def sample(frame: Frame, b: torch.Tensor, px: torch.Tensor, py: torch.Tensor, template: torch.Tensor,
           joints_root_z: torch.Tensor, lbs: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    """Virtual-frame features at integer pixels (px, py) of samples `b` (all `(M,)` long)."""
    with torch.autocast(device_type=px.device.type, enabled=False):
        fr = frame
        pxf, pyf = px.float(), py.float()
        face_here = fr.face[b, py, px]
        inside = face_here >= 0
        y1, x1, d1, ok1 = _seed_lookup(fr.s1, b, py, px)
        y2, x2, d2, ok2 = _seed_lookup(fr.s2, b, py, px)
        sgn = torch.where(inside, -torch.ones_like(d1), torch.ones_like(d1))
        ds = torch.where(ok1, sgn * d1, torch.full_like(d1, FAR_PX))
        vx1, vy1 = pxf - x1.float(), pyf - y1.float()                       # seed -> pixel
        n1 = d1.clamp_min(1e-6)
        dir1 = torch.stack([vx1, vy1], -1) / n1[:, None] * sgn[:, None]        # outward normal of the silhouette
        dir1 = torch.where((ok1 & (d1 > 0))[:, None], dir1, torch.zeros_like(dir1))
        dc = torch.where(ok2, d2, torch.full_like(d2, FAR_PX))
        dir2 = torch.stack([pxf - x2.float(), pyf - y2.float()], -1) / d2.clamp_min(1e-6)[:, None]
        dir2 = torch.where((ok2 & (d2 > 0))[:, None], dir2, torch.zeros_like(dir2))
        # surface point: under the pixel when inside, else the nearest silhouette (boundary) pixel
        sy_ = torch.where(inside, py, y1.clamp_min(0))
        sx_ = torch.where(inside, px, x1.clamp_min(0))
        f_s = torch.where(inside | ok1, fr.face[b, sy_, sx_], torch.full_like(face_here, -1))
        f_c = torch.where(ok2, fr.face[b, y2.clamp_min(0), x2.clamp_min(0)], torch.full_like(face_here, -1))
        out = {"inside": inside, "ds": ds, "dir_s": dir1, "dc": dc, "dir_c": dir2, "far_s": ~ok1, "far_c": ~ok2}
        for tag, fsel, qy, qx in (("s", f_s, sy_, sx_), ("c", f_c, y2.clamp_min(0), x2.clamp_min(0))):
            bary, depth = surface_at(fr.u, fr.v, fr.z, fr.faces, b, fsel, qx.float(), qy.float())
            fv = fr.faces.long()[fsel.clamp_min(0)]                               # (M, 3)
            valid = (fsel >= 0).float()
            out[f"tmpl_{tag}"] = (bary[..., None] * template[fv]).sum(1) * valid[:, None]
            out[f"dz_{tag}"] = (depth - joints_root_z[b]) * valid
            out[f"valid_{tag}"] = fsel >= 0
            if lbs is not None:
                out[f"lbs_{tag}"] = (bary[..., None] * lbs[fv]).sum(1) * valid[:, None]
        return out


__all__ = ["CONTOUR_DZ", "FAR_PX", "Frame", "HAND_SCALE_M", "JFA_STEPS", "WINDOWS", "contour_seeds", "jfa", "project",
           "rasterize", "sample", "surface_at"]
