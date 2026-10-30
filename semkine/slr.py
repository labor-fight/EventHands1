"""MV2 M1 / M2: the surface-lifted readout (SLR) of the dense S39 tracker (docs/EXPERIMENT_RECORDS.md#mv2-prereg-20261026).

`SLRModel` is `MNISTModel` plus one opt-in readout, `MODEL.SLR: true`. Without the key it is `MNISTModel` exactly (same
modules, same random draws, same function), so S39 checkpoints load and evaluate through it bit for bit.

With `MODEL.SLR: true` the trunk's logits (the global-average-pool readout, `rn.fc`, kept as the global bypass) get an
additive 51-d correction read off the MANO surface of the previous state:

  1. FK(prev) with the sample's own betas and render intrinsics K' (DomRand's K' included) gives the 778 vertices, their
     render-pixel projections, depths z, unit normals and visibility (`visible_static`: `semkine.aura.visible`'s 3 x 3
     vertex-splat z-buffer with 1 cm tolerance, bit for bit, without boolean indexing); no gradient flows into the
     previous state (it is an input).
  2. Each tapped ResNet stage (layer1 / layer2 / layer3, strides 4 / 8 / 16) is projected by a 1 x 1 conv to
     `SLR_PROJ` channels and bilinearly sampled at every vertex and at four offsets +-rho along u and v, with
     rho = fx * SLR_RADIUS_MM / z (a metric radius on the surface, so the window covers the same millimetres at any
     depth); the stem geometry is `MNISTModel._anchor_features`' (cell m of a stride-s map sits at pixel s m).
  3. A per-vertex MLP turns [samples, normalised u / v, metric scale 1 / z, normal, visibility, in-bounds] into an
     `SLR_HIDDEN` vector; `SLR_CONV` rounds of face-1-ring mean aggregation (GraphSAGE form, self + neighbour weights)
     mix it on the fixed MANO topology.
  4. LBS pooling: joint j's evidence is the mean of the vertex vectors weighted by w_ij x a_i, w the MANO skinning
     weights and a_i the vertex's soft visibility (1 if visible, sigmoid(g) with a learnt g, initialised at -2.2, if
     occluded; 0 out of the image). A shared joint MLP with a joint embedding reads the 15 finger joints' 3 axis-angle
     corrections; the root rotation and the translation are read from [joint 0 evidence, the visible-vertex mean].
     The last linear layers start at zero, so the model computes S39's function at initialisation.

`MODEL.SLR_SCALE: metric | fixed` (M2's MSCN factor): `fixed` uses a constant SLR_FIXED_PX window instead of the metric
radius. The readout runs in float32 with autocast off. Its modules are built after the trunk inside a forked RNG, so the
trunk, prev_mlp and the training data stream see exactly the random draws of the model without the readout.
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import resnet_forward_taps
from model import MNISTModel

SLR_KEYS = frozenset({"SLR", "SLR_TAPS", "SLR_PROJ", "SLR_HIDDEN", "SLR_CONV", "SLR_RADIUS_MM", "SLR_SCALE",
                      "SLR_FIXED_PX", "SLR_SEED"})
STRIDE = {"layer1": 4, "layer2": 8, "layer3": 16}
CHANNELS = {"layer1": 64, "layer2": 128, "layer3": 256}
N_OFF = 5            # centre, +-rho along u, +-rho along v


@torch.no_grad()
def visible_static(verts: torch.Tensor, intr) -> torch.Tensor:
    """`semkine.aura.visible` bit for bit (3 x 3 vertex-splat z-buffer at the rounded pixel, 1 cm tolerance, 240 x 180,
    z > 0) without boolean-mask indexing: out-of-image or z <= 0 splats go to one dummy slot past the image, so every
    tensor shape is static and nothing syncs with the host (CUDA-graph capturable)."""
    from semkine.aura import IMG_H, IMG_W, VIS_TOL
    fx, fy, cx, cy = intr
    z = verts[..., 2].float()
    zs = z.clamp(min=1e-6)
    u = fx[:, None] * verts[..., 0].float() / zs + cx[:, None]
    v = fy[:, None] * verts[..., 1].float() / zs + cy[:, None]
    B, V = z.shape
    ui, vi = u.round().long(), v.round().long()
    base = torch.arange(B, device=z.device).view(B, 1) * (IMG_H * IMG_W)
    dummy = B * IMG_H * IMG_W
    zb = torch.full((dummy + 1,), 1e9, dtype=z.dtype, device=z.device)
    for dyy in (-1, 0, 1):
        for dxx in (-1, 0, 1):
            uu, vv = ui + dxx, vi + dyy
            ok = (uu >= 0) & (uu < IMG_W) & (vv >= 0) & (vv < IMG_H) & (z > 0)
            idx = torch.where(ok, base + vv * IMG_W + uu, torch.full_like(uu, dummy))
            zb.scatter_reduce_(0, idx.reshape(-1), torch.where(ok, z, torch.full_like(z, 1e9)).reshape(-1),
                               reduce="amin", include_self=True)
    inb = (ui >= 0) & (ui < IMG_W) & (vi >= 0) & (vi < IMG_H) & (z > 0)
    zz = zb[(base + vi.clamp(0, IMG_H - 1) * IMG_W + ui.clamp(0, IMG_W - 1)).reshape(-1)].view(B, V)
    return inb & (z <= zz + VIS_TOL)


class SurfaceLiftedReadout(nn.Module):
    def __init__(self, lbs: torch.Tensor, faces: torch.Tensor, ring_idx: torch.Tensor, ring_mask: torch.Tensor,
                 taps: Sequence[str] = ("layer1", "layer2", "layer3"), proj: int = 32, hidden: int = 64,
                 n_conv: int = 1, radius_mm: float = 10.0, scale: str = "metric", fixed_px: float = 3.0,
                 joint_hidden: int = 128):
        super().__init__()
        if scale not in ("metric", "fixed"):
            raise ValueError(f"MODEL.SLR_SCALE must be metric or fixed, got {scale!r}")
        self.taps = tuple(taps)
        self.radius_m = float(radius_mm) / 1000.0
        self.scale, self.fixed_px = scale, float(fixed_px)
        self.register_buffer("lbs", lbs.float(), persistent=False)                    # (778, 16)
        self.register_buffer("faces", faces.long(), persistent=False)
        self.register_buffer("ring_idx", ring_idx.long(), persistent=False)          # (778, K)
        self.register_buffer("ring_mask", ring_mask.float(), persistent=False)       # (778, K)
        self.proj = nn.ModuleDict({t: nn.Conv2d(CHANNELS[t], proj, 1) for t in self.taps})
        d_in = len(self.taps) * proj * N_OFF + 8      # + uv (2), metric scale, normal (3), visible, in-bounds
        self.vert = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.conv_self = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_conv)])
        self.conv_nbr = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(n_conv)])
        self.occluded_logit = nn.Parameter(torch.tensor(-2.2))
        self.joint_emb = nn.Embedding(16, 16)
        self.finger = nn.Sequential(nn.Linear(hidden + 1 + 16, joint_hidden), nn.ReLU(), nn.Linear(joint_hidden, 3))
        self.rootT = nn.Sequential(nn.Linear(2 * hidden + 2, joint_hidden), nn.ReLU(), nn.Linear(joint_hidden, 6))
        for last in (self.finger[-1], self.rootT[-1]):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, maps: Dict[str, torch.Tensor], geo: Dict[str, torch.Tensor], h: int, w: int) -> torch.Tensor:
        uv, z, n, vis = geo["uv"], geo["z"], geo["normals"], geo["vis"]
        fx = geo["fx"]
        B, V = z.shape
        inb = (z > 1e-6) & (uv[..., 0] >= -0.5) & (uv[..., 0] < w - 0.5) & (uv[..., 1] >= -0.5) & (uv[..., 1] < h - 0.5)
        if self.scale == "metric":
            rho = fx[:, None] * self.radius_m / z.clamp(min=1e-3)                      # (B, V) render pixels
        else:
            rho = torch.full_like(z, self.fixed_px)
        offs = torch.stack([torch.zeros_like(rho), rho, -rho, torch.zeros_like(rho), torch.zeros_like(rho)], -1)
        offs = torch.stack([offs, torch.stack([torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho),
                                               rho, -rho], -1)], -1)                   # (B, V, 5, 2)
        pts = (uv.unsqueeze(2) + offs).reshape(B, V * N_OFF, 2)
        feats = []
        for t in self.taps:
            f = self.proj[t](maps[t].float())
            _, C, hh, ww = f.shape
            s = float(STRIDE[t])
            gx = 2.0 * (pts[..., 0] / s) / (ww - 1) - 1.0
            gy = 2.0 * (pts[..., 1] / s) / (hh - 1) - 1.0
            grid = torch.stack([gx, gy], -1).unsqueeze(2)
            smp = F.grid_sample(f, grid, mode="bilinear", padding_mode="border", align_corners=True)
            feats.append(smp.squeeze(-1).transpose(1, 2).reshape(B, V, N_OFF * C))
        pos = torch.stack([2.0 * uv[..., 0] / (w - 1) - 1.0, 2.0 * uv[..., 1] / (h - 1) - 1.0], -1).clamp(-2.0, 2.0)
        scale = (fx[:, None] * 0.01 / z.clamp(min=1e-3) / 10.0).unsqueeze(-1)       # pixels per cm / 10
        x = torch.cat(feats + [pos, scale, n, vis.float().unsqueeze(-1), inb.float().unsqueeze(-1)], -1)
        x = x * inb.float().unsqueeze(-1)
        hv = self.vert(x)                                                             # (B, V, hidden)
        a = torch.where(vis, torch.ones_like(z), torch.sigmoid(self.occluded_logit).expand_as(z)) * inb.float()
        K = self.ring_idx.shape[1]
        for ws, wn in zip(self.conv_self, self.conv_nbr):
            nb = hv[:, self.ring_idx.reshape(-1)].view(B, V, K, -1)
            wgt = (self.ring_mask.view(1, V, K) * a[:, self.ring_idx.reshape(-1)].view(B, V, K)).unsqueeze(-1)
            mean = (nb * wgt).sum(2) / wgt.sum(2).clamp(min=1e-6)
            hv = F.relu(ws(hv) + wn(mean))
        wj = self.lbs.unsqueeze(0) * a.unsqueeze(-1)                                  # (B, V, 16)
        mass = wj.sum(1)                                                              # (B, 16)
        ej = torch.einsum("bvj,bvd->bjd", wj, hv) / mass.clamp(min=1e-6).unsqueeze(-1)
        cov = (mass / self.lbs.sum(0).clamp(min=1e-6).unsqueeze(0)).unsqueeze(-1)     # (B, 16, 1)
        emb = self.joint_emb.weight.unsqueeze(0).expand(B, -1, -1)
        fing = self.finger(torch.cat([ej[:, 1:], cov[:, 1:], emb[:, 1:]], -1)).reshape(B, 45)
        gmean = (hv * a.unsqueeze(-1)).sum(1) / a.sum(1, keepdim=True).clamp(min=1e-6)
        rt = self.rootT(torch.cat([ej[:, 0], gmean, cov[:, 0], (a.sum(1, keepdim=True) / V)], -1))
        return torch.cat([rt[:, 3:6], rt[:, 0:3], fing], -1)                         # [transl, root, fingers]


class SLRModel(MNISTModel):
    MODEL_KEYS = MNISTModel.MODEL_KEYS | SLR_KEYS

    def __init__(self, cfg=None, num_classes=None):
        super().__init__(cfg=cfg, num_classes=num_classes)
        mc = self.cfg.get("MODEL", {}) or {}
        self.slr = bool(mc.get("SLR", False))
        stray = sorted(k for k in mc if k in SLR_KEYS and k != "SLR")
        if not self.slr:
            if stray:
                raise ValueError(f"{stray} without MODEL.SLR: true would silently do nothing")
            return
        if not (self.prev_render and self.predict_delta and self.root_head_kind == "none" and not self.encoder_name
                and not self.glimpse and not self.sparse_trunk and not self.active_head and self.output_dim == 51
                and self.pose_repr == "mano_full_axis_angle"):
            raise ValueError("MODEL.SLR reads the dense 51-d tracker's trunk: it needs PREV_RENDER, PREDICT_DELTA, a "
                             "dense ResNet trunk and no ROOT_HEAD / GLIMPSE / ENCODER / ACTIVE_HEAD")
        from semkine.xyz_mesh import mesh_ring
        faces = self.mano.f.detach().cpu().long()
        ring_idx, ring_mask = mesh_ring(faces, int(self.mano.weights.shape[0]))
        taps = mc.get("SLR_TAPS", ["layer1", "layer2", "layer3"])
        for t in taps:
            self._stage_channels(self.rn, t)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(mc.get("SLR_SEED", 20261026)))
            self.slr_readout = SurfaceLiftedReadout(
                self.mano.weights.detach().cpu(), faces, ring_idx, ring_mask, taps=taps,
                proj=int(mc.get("SLR_PROJ", 32)), hidden=int(mc.get("SLR_HIDDEN", 64)),
                n_conv=int(mc.get("SLR_CONV", 1)), radius_mm=float(mc.get("SLR_RADIUS_MM", 10.0)),
                scale=str(mc.get("SLR_SCALE", "metric")), fixed_px=float(mc.get("SLR_FIXED_PX", 3.0)))
        # route `forward` through `_trunk_with_root_head` (betas_f / k_f are resolved there because PREV_RENDER is on)
        self.root_head_kind = "slr"

    @torch.no_grad()
    def slr_geometry(self, prevpos, betas_f, k_f) -> Dict[str, torch.Tensor]:
        from semkine.xyz_mesh import vertex_normals
        verts, _ = self._fk(prevpos.float(), betas_f.float())
        intr = self._intrinsics(k_f.float())
        fx, fy, cx, cy = intr
        z = verts[..., 2]
        zs = z.clamp(min=1e-6)
        uv = torch.stack([fx[:, None] * verts[..., 0] / zs + cx[:, None], fy[:, None] * verts[..., 1] / zs + cy[:, None]],
                         -1)
        return {"uv": uv, "z": z, "normals": vertex_normals(verts, self.mano.f), "vis": visible_static(verts, intr), "fx": fx}

    def _trunk_with_root_head(self, z, prevpos, betas_f, k_f):
        if not self.slr:
            return super()._trunk_with_root_head(z, prevpos, betas_f, k_f)
        out, got = resnet_forward_taps(self.rn, z, self.slr_readout.taps)
        with torch.autocast(device_type=z.device.type, enabled=False):
            geo = self.slr_geometry(prevpos, betas_f, k_f)
            corr = self.slr_readout(got, geo, self.render_h, self.render_w)
        return out + corr.to(out.dtype)
