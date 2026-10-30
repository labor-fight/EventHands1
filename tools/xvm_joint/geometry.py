"""Target-free XJ support construction and fixed-reference geometry.

Supports and their image normals are computed from the current 50 ms count
planes before MANO is consulted.  MANO supplies conditional candidate addresses,
metric crop scales and physical-parameter pixel Jacobians.  These addresses are
geometry proxies, not recovered material/contour event identities.

The separate ``targets`` method belongs to supervision and must be called only
after the caller has completed its prediction.  No labels or target shape are
accepted by ``prepare``.  Geometry is intentionally detached: the local head and
linear solve train through their own output, not a hidden geometry optimizer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class GeometryConfig:
    max_supports: int = 24
    candidates_per_support: int = 3
    cell_size: int = 16
    patch_size: int = 16
    radius_m: float = .02
    near_m: float = .03
    jacobian_chunk: int = 64
    translation_epsilon_m: float = 1e-4
    rotation_epsilon_rad: float = 1e-3
    height: int = 180
    width: int = 240

    def __post_init__(self):
        if self.max_supports < 1 or not 1 <= self.candidates_per_support <= 16:
            raise ValueError('positive support budget and <=16 distinct LBS candidates required')
        if self.patch_size != 16 or min(self.height, self.width, self.cell_size, self.jacobian_chunk) < 1:
            raise ValueError('positive image/grid/chunk sizes and registered 16x16 patch required')
        if min(self.radius_m, self.near_m, self.translation_epsilon_m, self.rotation_epsilon_rad) <= 0:
            raise ValueError('positive physical scales and finite-difference steps required')


@dataclass
class PreparedGeometry:
    patches: Tensor
    rho: Tensor
    observed: Tensor
    B0: Tensor
    B1: Tensor
    normal: Tensor
    nominal_delta: Tensor
    source_ids: Tensor
    candidate_mask: Tensor
    spatial_logits: Tensor
    vertex_ids: Tensor
    support_xy: Tensor
    uv0: Tensor
    uv1: Tensor
    q0_reference: Tensor
    q1_reference: Tensor
    K_event: Tensor
    metadata: dict

    @property
    def model_kwargs(self):
        names = ('patches', 'rho', 'observed', 'B0', 'B1', 'normal',
                 'nominal_delta', 'source_ids', 'candidate_mask', 'spatial_logits')
        return {name: getattr(self, name) for name in names}


def sensor_supports(field: Tensor, config: Optional[GeometryConfig] = None) -> Dict[str, Tensor]:
    """Bounded current-count supports, independent of every pose hypothesis.

    One occupied spatial cell supplies at most one normal-bearing point.  The
    highest-count cells are used, with deterministic raster-order tie breaks.
    Within a cell select the strongest Sobel location that actually has events;
    a count centroid is used when there is no reliable gradient, then marked
    normal-invalid.  Latest-age channels never create current observations.
    """
    c = config or GeometryConfig()
    if field.ndim != 4 or field.shape[1:] != (16, c.height, c.width):
        raise ValueError('F16[B,16,H,W] required')
    if not bool(torch.isfinite(field).all()) or bool((field < 0).any()):
        raise ValueError('finite nonnegative public field required')
    with torch.no_grad(), torch.autocast(device_type=field.device.type, enabled=False):
        counts = field[:, 6:].float().sum(1)
        image = torch.log1p(counts).unsqueeze(1)
        kx = field.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32) / 8
        ky = kx.T
        grad = F.conv2d(image, torch.stack((kx, ky))[:, None], padding=1)
        xy = field.new_zeros((len(field), c.max_supports, 2), dtype=torch.float32)
        normal = torch.zeros_like(xy)
        valid = torch.zeros((len(field), c.max_supports), device=field.device, dtype=torch.bool)
        source = torch.full(valid.shape, -1, device=field.device, dtype=torch.long)
        mass = torch.zeros_like(valid, dtype=torch.float32)
        ncol = (c.width + c.cell_size - 1) // c.cell_size
        for b in range(len(field)):
            choices = []
            for top in range(0, c.height, c.cell_size):
                for left in range(0, c.width, c.cell_size):
                    tile = counts[b, top:top+c.cell_size, left:left+c.cell_size]
                    weight = float(tile.sum())
                    if weight > 0:
                        sid = (top // c.cell_size) * ncol + left // c.cell_size
                        choices.append((-weight, sid, top, left))
            choices.sort()
            for slot, (negative_mass, sid, top, left) in enumerate(choices[:c.max_supports]):
                tile = counts[b, top:top+c.cell_size, left:left+c.cell_size]
                g = grad[b, :, top:top+tile.shape[0], left:left+tile.shape[1]]
                magnitude = g.square().sum(0).masked_fill(tile <= 0, -1.)
                peak = int(magnitude.reshape(-1).argmax())
                yy, xx = divmod(peak, tile.shape[1])
                edge = g[:, yy, xx]
                good = bool(edge.square().sum() > 1e-12)
                if good:
                    point = field.new_tensor([left+xx, top+yy], dtype=torch.float32)
                    direction = edge / edge.norm()
                else:
                    gy, gx = torch.meshgrid(
                        torch.arange(tile.shape[0], device=field.device),
                        torch.arange(tile.shape[1], device=field.device), indexing='ij')
                    point = torch.stack(((tile*gx).sum()/tile.sum()+left,
                                         (tile*gy).sum()/tile.sum()+top))
                    direction = torch.zeros(2, device=field.device)
                xy[b, slot] = point
                normal[b, slot] = direction
                source[b, slot] = sid
                mass[b, slot] = -negative_mass
                valid[b, slot] = good
        return dict(xy=xy, normal=normal, normal_valid=valid, source_ids=source, count=mass)


def _project(vertices: Tensor, K_event: Tensor, near: float):
    z = vertices[..., 2]
    safe_z = torch.where(z > near, z, torch.ones_like(z))
    u = K_event[:, None, 0, 0] * vertices[..., 0] / safe_z + K_event[:, None, 0, 2]
    v = K_event[:, None, 1, 1] * vertices[..., 1] / safe_z + K_event[:, None, 1, 2]
    uv = torch.stack((u, v), -1)
    valid = (z > near) & torch.isfinite(uv).all(-1)
    return uv, z, valid


def _gather_vertices(values: Tensor, ids: Tensor):
    tail = values.shape[2:]
    index = ids.reshape(ids.shape + (1,) * len(tail)).expand(ids.shape + tail)
    return values.gather(1, index)


def triangle_silhouette(vertices: np.ndarray, faces: np.ndarray, K_event: np.ndarray,
                        image_size=(180, 240), *, near=.03) -> np.ndarray:
    """Pure-NumPy two-sided silhouette at integer pixel centres.

    This is the union of the same closed projected triangles used by the
    repository's constant-albedo perspective rasterizer.  When every vertex is
    in front of ``near``, every covered pixel has a positive-depth hit, and
    z-buffering cannot change a binary silhouette.  No model package import,
    state, dataset, texture, or intensity likelihood is involved here.
    """
    vertices = np.asarray(vertices, np.float64)
    faces = np.asarray(faces, np.int64)
    camera = np.asarray(K_event, np.float64)
    height, width = map(int, image_size)
    if vertices.shape != (778, 3) or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('MANO778 vertices and triangular faces required')
    if camera.shape != (3, 3) or min(height, width) < 1:
        raise ValueError('3x3 camera and positive image dimensions required')
    if not np.isfinite(vertices).all() or not np.isfinite(camera).all():
        raise ValueError('finite rendering geometry and camera required')
    if np.min(vertices[:, 2]) <= float(near):
        raise ValueError('unsupported near-plane intersection')
    homogeneous = vertices @ camera.T
    xy = homogeneous[:, :2]/homogeneous[:, 2:3]
    silhouette = np.zeros((height, width), bool)
    for ids in faces:
        triangle = xy[ids]
        left = max(0, int(np.ceil(triangle[:, 0].min())))
        right = min(width-1, int(np.floor(triangle[:, 0].max())))
        top = max(0, int(np.ceil(triangle[:, 1].min())))
        bottom = min(height-1, int(np.floor(triangle[:, 1].max())))
        if left > right or top > bottom:
            continue
        x0, y0 = triangle[0];x1, y1 = triangle[1];x2, y2 = triangle[2]
        denominator = (y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(denominator) < 1e-14:
            continue
        yy, xx = np.mgrid[top:bottom+1, left:right+1]
        w0 = ((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/denominator
        w1 = ((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/denominator
        w2 = 1.-w0-w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        silhouette[top:bottom+1, left:right+1] |= inside
    return silhouette


class JointGeometry:
    def __init__(self, base, config: Optional[GeometryConfig] = None):
        self.base = base
        self.config = config or GeometryConfig()
        if not hasattr(base, '_fk') or not hasattr(base, '_intrinsics') or not hasattr(base, 'mano'):
            raise ValueError('canonical differentiable MANO base required')
        self.faces = base.mano.f.detach().cpu().numpy().copy()
        self.part = base.mano.weights.detach().argmax(-1)
        if self.part.shape != (778,):
            raise ValueError('canonical778 MANO LBS weights required')

    def event_intrinsics(self, K_source: Tensor) -> Tensor:
        """Use the legacy render-space camera exactly once, with no recrop."""
        if K_source.ndim == 2:
            K_source = K_source.unsqueeze(0)
        if K_source.ndim != 3 or K_source.shape[1:] != (3, 3):
            raise ValueError('source camera[B,3,3] required')
        fx, fy, cx, cy = self.base._intrinsics(K_source)
        K = torch.zeros_like(K_source)
        K[:, 0, 0], K[:, 1, 1] = fx, fy
        K[:, 0, 2], K[:, 1, 2] = cx, cy
        K[:, 2, 2] = 1
        if not bool(torch.isfinite(K).all()) or bool((fx <= 0).any()) or bool((fy <= 0).any()):
            raise ValueError('finite positive camera focal lengths required')
        return K

    @torch.no_grad()
    def _fk(self, q):
        beta = q.new_zeros((len(q), 10))
        with torch.autocast(device_type=q.device.type, enabled=False):
            vertices, joints = self.base._fk(q.float(), beta.float())
        return vertices.float(), joints.float()

    @torch.no_grad()
    def projected_jacobian(self, q: Tensor, K_event: Tensor, vertex_ids: Tensor, *, return_valid=False):
        """Central differences versus physical original 51D additive AA state.

        No state-scale D is applied here.  The linear operator applies it once.
        The 102 displaced states per lane are evaluated in bounded FK chunks.
        """
        c = self.config
        B, records = vertex_ids.shape
        eps = q.new_tensor([c.translation_epsilon_m]*3 + [c.rotation_epsilon_rad]*48)
        perturb = torch.eye(51, device=q.device, dtype=q.dtype) * eps[:, None]
        displaced = torch.stack((q[:, None] + perturb[None], q[:, None] - perturb[None]), 2)
        flat = displaced.reshape(-1, 51)
        flat_k = K_event[:, None, None].expand(B, 51, 2, 3, 3).reshape(-1, 3, 3)
        flat_ids = vertex_ids[:, None, None].expand(B, 51, 2, records).reshape(-1, records)
        projected, validity = [], []
        for first in range(0, len(flat), c.jacobian_chunk):
            verts, _ = self._fk(flat[first:first+c.jacobian_chunk])
            uv, _, valid = _project(verts, flat_k[first:first+c.jacobian_chunk], c.near_m)
            selected = _gather_vertices(uv, flat_ids[first:first+c.jacobian_chunk])
            selected_valid = valid.gather(1, flat_ids[first:first+c.jacobian_chunk])
            projected.append(torch.where(selected_valid[..., None], selected, torch.zeros_like(selected)))
            validity.append(selected_valid)
        uv = torch.cat(projected).reshape(B, 51, 2, records, 2)
        jac = (uv[:, :, 0]-uv[:, :, 1])/(2*eps[None, :, None, None])
        fd_valid = torch.cat(validity).reshape(B, 51, 2, records).all(2).all(1)
        jac = jac.permute(0, 2, 3, 1).contiguous()
        # A near-plane crossing is not a derivative of the projection map.
        # Reject the complete record rather than compare a finite pixel with0.
        jac = torch.where(fd_valid[..., None, None], jac, torch.zeros_like(jac))
        return (jac, fd_valid) if return_valid else jac

    @torch.no_grad()
    def silhouette_distance(self, vertices: Tensor, K_event: Tensor):
        """True triangle-raster SDF, positive outside, negative inside, in px."""
        from scipy.ndimage import distance_transform_edt
        c = self.config
        result, valid = [], []
        for v, k in zip(vertices.detach().cpu().numpy(), K_event.detach().cpu().numpy()):
            good = bool(np.isfinite(v).all() and np.min(v[:, 2]) > c.near_m)
            if good:
                silhouette = triangle_silhouette(v, self.faces, k, (c.height, c.width), near=c.near_m)
                good = bool(silhouette.any())
                if good:
                    sdf = distance_transform_edt(~silhouette)-distance_transform_edt(silhouette)
                else:
                    sdf = np.zeros((c.height, c.width), np.float32)
            else:
                sdf = np.zeros((c.height, c.width), np.float32)
            result.append(np.asarray(sdf, np.float32));valid.append(good)
        return (torch.as_tensor(np.stack(result), device=vertices.device),
                torch.tensor(valid, device=vertices.device, dtype=torch.bool))

    @torch.no_grad()
    def prepare(self, field0: Tensor, field1: Tensor, q0: Tensor, q1: Tensor,
                K_source: Tensor, both_exposed: Optional[Tensor] = None) -> PreparedGeometry:
        c = self.config
        if field0.shape != field1.shape or field0.ndim != 4 or field0.shape[1:] != (16, c.height, c.width):
            raise ValueError('matching dual-time F16[B,16,H,W] required')
        B = len(field0)
        if q0.shape != (B, 51) or q1.shape != (B, 51):
            raise ValueError('reference states[B,51] required')
        if not bool(torch.isfinite(q0).all()) or not bool(torch.isfinite(q1).all()):
            raise ValueError('nonfinite reference state')
        if K_source.ndim == 2:
            K_source = K_source.unsqueeze(0).expand(B, -1, -1)
        if len(K_source) != B:
            raise ValueError('camera batch mismatch')
        K_event = self.event_intrinsics(K_source.to(q0).float())
        support = sensor_supports(field1, c)
        v0, _ = self._fk(q0);v1, _ = self._fk(q1)
        uv0_all, z0, geo0 = _project(v0, K_event, c.near_m)
        uv1_all, _, geo1 = _project(v1, K_event, c.near_m)
        S, A = c.max_supports, c.candidates_per_support
        records = S*A
        vertex_ids = torch.zeros((B, S, A), device=q0.device, dtype=torch.long)
        candidate_mask = torch.zeros_like(vertex_ids, dtype=torch.bool)
        part = self.part.to(q0.device)
        fx, fy = K_event[:, 0, 0], K_event[:, 1, 1]
        cx, cy = K_event[:, 0, 2], K_event[:, 1, 2]
        rho_all = torch.sqrt(fx*fy)[:, None]*c.radius_m/z0.clamp_min(c.near_m)
        inview0 = geo0 & (uv0_all[..., 0] >= 0) & (uv0_all[..., 0] <= c.width-1) & (uv0_all[..., 1] >= 0) & (uv0_all[..., 1] <= c.height-1)
        inview1 = geo1 & (uv1_all[..., 0] >= 0) & (uv1_all[..., 0] <= c.width-1) & (uv1_all[..., 1] >= 0) & (uv1_all[..., 1] <= c.height-1)
        for b in range(B):
            angular_vertex = torch.stack(((uv0_all[b, :, 0]-cx[b])/fx[b], (uv0_all[b, :, 1]-cy[b])/fy[b]), -1)
            for s in range(S):
                if not bool(support['normal_valid'][b, s]):
                    continue
                point = support['xy'][b, s]
                ray = torch.stack(((point[0]-cx[b])/fx[b], (point[1]-cy[b])/fy[b]))
                angular_dist = (angular_vertex-ray).square().sum(-1)
                pixel_dist = (uv0_all[b]-point).square().sum(-1)
                eligible = inview0[b] & inview1[b] & (pixel_dist <= rho_all[b].square())
                candidates = []
                for body_part in range(16):
                    distances = angular_dist.masked_fill(~(eligible & (part == body_part)), torch.inf)
                    distance, vid = distances.min(0)
                    if bool(torch.isfinite(distance)):
                        candidates.append((float(distance), int(vid)))
                candidates.sort()
                for a, (_, vid) in enumerate(candidates[:A]):
                    vertex_ids[b, s, a] = vid
                    candidate_mask[b, s, a] = True
        vertex_ids = vertex_ids.reshape(B, records)
        candidate_mask = candidate_mask.reshape(B, records)
        uv0 = _gather_vertices(uv0_all, vertex_ids)
        uv1 = _gather_vertices(uv1_all, vertex_ids)
        rho = rho_all.gather(1, vertex_ids)
        rho = torch.where(candidate_mask, rho, torch.ones_like(rho))
        normal = support['normal'][:, :, None].expand(B, S, A, 2).reshape(B, records, 2)
        source_xy = support['xy'][:, :, None].expand(B, S, A, 2).reshape(B, records, 2)
        source_ids = support['source_ids'][:, :, None].expand(B, S, A).reshape(B, records)
        if both_exposed is None:
            both_exposed = torch.ones(B, device=q0.device, dtype=torch.bool)
        else:
            both_exposed = torch.as_tensor(both_exposed, device=q0.device, dtype=torch.bool).reshape(B)
        sdf0, raster0 = self.silhouette_distance(v0, K_event)
        sdf1, raster1 = self.silhouette_distance(v1, K_event)
        geometry_valid = raster0 & raster1 & both_exposed
        candidate_mask = candidate_mask & geometry_valid[:, None]
        observed = candidate_mask.clone()
        loc = torch.linspace(-1, 1, c.patch_size, device=q0.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(loc, loc, indexing='ij')
        local = torch.stack((xx, yy), -1)
        tangent = torch.stack((-normal[..., 1], normal[..., 0]), -1)
        # Padding records use a finite identity frame, never a zero-grid NaN.
        nrm = normal.norm(dim=-1, keepdim=True)
        n = torch.where(nrm > 1e-6, normal, normal.new_tensor([1., 0.]))
        tangent = torch.where(nrm > 1e-6, tangent, tangent.new_tensor([0., 1.]))
        frame = torch.stack((n, tangent), -1)
        offset = torch.einsum('bvij,hwj->bvhwi', frame, local)*rho[..., None, None, None]
        pixels = uv0[..., None, None, :]+offset
        grid = torch.stack((pixels[..., 0]/(c.width-1)*2-1, pixels[..., 1]/(c.height-1)*2-1), -1)
        combined = torch.cat((field0.float(), field1.float(), sdf0[:, None], sdf1[:, None]), 1)
        # Only the model patch view gets log1p. Cached raw count planes remain unchanged.
        combined = combined.clone()
        combined[:, 6:16] = torch.log1p(combined[:, 6:16])
        combined[:, 22:32] = torch.log1p(combined[:, 22:32])
        sampled = F.grid_sample(combined, grid.reshape(B, records*c.patch_size, c.patch_size, 2),
                                align_corners=True, padding_mode='zeros')
        sampled = sampled.reshape(B, 34, records, c.patch_size, c.patch_size).permute(0, 2, 1, 3, 4)
        sampled[:, :, 32:34] /= rho[..., None, None, None]
        relative = (pixels-source_xy[..., None, None, :])/rho[..., None, None, None]
        coordinates = torch.einsum('bvij,bvhwi->bvhwj', frame, relative).permute(0, 1, 4, 2, 3)
        pixel_valid = ((pixels[..., 0] >= 0) & (pixels[..., 0] <= c.width-1) &
                       (pixels[..., 1] >= 0) & (pixels[..., 1] <= c.height-1) & geometry_valid[:, None, None, None])
        patches = torch.cat((sampled, coordinates, pixel_valid[:, :, None].float()), 2)
        patches = patches*observed[..., None, None, None]
        B0, fd0 = self.projected_jacobian(q0.detach().float(), K_event, vertex_ids, return_valid=True)
        B1, fd1 = self.projected_jacobian(q1.detach().float(), K_event, vertex_ids, return_valid=True)
        observed &= fd0 & fd1
        candidate_mask &= fd0 & fd1
        patches *= observed[..., None, None, None]
        B0 *= observed[..., None, None];B1 *= observed[..., None, None]
        delta = (uv1-uv0)*observed[..., None]
        spatial_logits = -.5*((uv0-source_xy).square().sum(-1)/rho.square())
        spatial_logits = torch.where(observed, spatial_logits, torch.zeros_like(spatial_logits))
        return PreparedGeometry(patches=patches.contiguous(), rho=rho, observed=observed,
            B0=B0, B1=B1, normal=normal, nominal_delta=delta, source_ids=source_ids,
            candidate_mask=candidate_mask, spatial_logits=spatial_logits, vertex_ids=vertex_ids,
            support_xy=source_xy, uv0=uv0, uv1=uv1, q0_reference=q0.detach().clone(),
            q1_reference=q1.detach().clone(), K_event=K_event,
            metadata=dict(config=asdict(c), K_scaling='base._intrinsics exactly once from source K',
                          support='current count Sobel, sensor-side and pose independent',
                          jacobian='physical51 additive AA central difference, no D applied',
                          raster_valid=(raster0 & raster1).cpu().tolist(),
                          finite_difference_valid_records=int((fd0 & fd1).sum()),
                          active_records=int(observed.sum()), GT_input=False,
                          semantics='beta0 canonical-vertex geometric proxies; no recovered event material identity'))

    @torch.no_grad()
    def targets(self, prepared: PreparedGeometry, gt0: Tensor, gt1: Tensor):
        """Supervision-only normal geometric proxies, never event identities."""
        if gt0.shape != prepared.q0_reference.shape or gt1.shape != gt0.shape:
            raise ValueError('matching supervised endpoint states[B,51] required')
        vertices0, _ = self._fk(gt0);vertices1, _ = self._fk(gt1)
        uv0, _, valid0 = _project(vertices0, prepared.K_event, self.config.near_m)
        uv1, _, valid1 = _project(vertices1, prepared.K_event, self.config.near_m)
        p0 = _gather_vertices(uv0, prepared.vertex_ids)
        p1 = _gather_vertices(uv1, prepared.vertex_ids)
        valid = prepared.observed & valid0.gather(1, prepared.vertex_ids) & valid1.gather(1, prepared.vertex_ids)
        a = ((p0-prepared.uv0)*prepared.normal).sum(-1)
        d = ((p1-p0)*prepared.normal).sum(-1)
        a = torch.where(valid, a, torch.zeros_like(a))
        d = torch.where(valid, d, torch.zeros_like(d))
        return dict(target_a_px=a, target_d_px=d, target_px=torch.stack((a, d), -1),
                    supervised=valid, validity_target=valid.float(),
                    target_kind='beta0 same-vertex geometry proxy; not actual event correspondence')


__all__ = ['GeometryConfig', 'PreparedGeometry', 'JointGeometry', 'sensor_supports', 'triangle_silhouette']
