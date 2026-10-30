"""Evaluation-only vectorization of the frozen XJ geometry contract.

No targets or recordings enter these functions.  The frozen projector, finite
differences, raster/SDF, patch sampling and target implementation are inherited
unchanged.  Only the scalar support/candidate selection loops are vectorized.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from .geometry import (GeometryConfig, JointGeometry, PreparedGeometry,
                       _gather_vertices, _project, sensor_supports)


@torch.no_grad()
def fast_sensor_supports(field: Tensor, config: Optional[GeometryConfig] = None):
    """Exact integer-count fast path, with the original general-value fallback.

    The sum of nonnegative integer counts is bounded by 2**24 per lane before
    changing reduction layout.  Every tile mass is therefore exact in FP32.
    Float counts or unusually large packets use the original implementation.
    The rare no-gradient centroid retains the original tile reductions.
    """
    c = config or GeometryConfig()
    if field.ndim != 4 or field.shape[1:] != (16, c.height, c.width):
        raise ValueError('F16[B,16,H,W] required')
    if not bool(torch.isfinite(field).all()) or bool((field < 0).any()):
        raise ValueError('finite nonnegative public field required')
    with torch.autocast(device_type=field.device.type, enabled=False):
        raw = field[:, 6:].float()
        if (bool((raw > 2**24).any()) or bool((raw != raw.trunc()).any())
                or bool((raw.to(torch.int64).flatten(1).sum(-1) > 2**24).any())):
            return sensor_supports(field, c)
        counts = raw.sum(1)
        image = torch.log1p(counts).unsqueeze(1)
        kx = field.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32) / 8
        grad = F.conv2d(image, torch.stack((kx, kx.T))[:, None], padding=1)
        batch, size = len(field), c.cell_size
        nrow = (c.height + size-1)//size
        ncol = (c.width + size-1)//size
        pad = (0, ncol*size-c.width, 0, nrow*size-c.height)
        tiles = F.pad(counts, pad).reshape(batch, nrow, size, ncol, size)
        tiles = tiles.permute(0, 1, 3, 2, 4).reshape(batch, nrow*ncol, size*size)
        gradients = F.pad(grad, pad).reshape(batch, 2, nrow, size, ncol, size)
        gradients = gradients.permute(0, 2, 4, 3, 5, 1).reshape(batch, nrow*ncol, size*size, 2)
        masses = tiles.sum(-1)
        used = min(c.max_supports, nrow*ncol)
        # Stable descending mass preserves the original sid/raster tie break.
        chosen = torch.argsort(masses, dim=-1, descending=True, stable=True)[:, :used]
        selected_mass = masses.gather(1, chosen)
        selected_tile = tiles.gather(1, chosen[..., None].expand(-1, -1, size*size))
        selected_grad = gradients.gather(1, chosen[..., None, None].expand(-1, -1, size*size, 2))
        magnitude = selected_grad.square().sum(-1).masked_fill(selected_tile <= 0, -1.)
        peak = magnitude.argmax(-1)
        edge = selected_grad.gather(2, peak[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
        occupied = selected_mass > 0
        good = occupied & (edge.square().sum(-1) > 1e-12)
        left = (chosen % ncol)*size
        top = (chosen // ncol)*size
        point = torch.stack((left+peak % size, top+peak // size), -1).float()
        direction = edge/edge.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        # Match the frozen centroid's exact unpadded shape/reduction order.
        for b, slot in (occupied & ~good).nonzero().cpu().tolist():
            t, l = int(top[b, slot]), int(left[b, slot])
            tile = counts[b, t:t+size, l:l+size]
            gy, gx = torch.meshgrid(torch.arange(tile.shape[0], device=field.device),
                                    torch.arange(tile.shape[1], device=field.device), indexing='ij')
            point[b, slot] = torch.stack(((tile*gx).sum()/tile.sum()+l,
                                          (tile*gy).sum()/tile.sum()+t))
        xy = field.new_zeros((batch, c.max_supports, 2), dtype=torch.float32)
        normal = torch.zeros_like(xy)
        valid = torch.zeros((batch, c.max_supports), device=field.device, dtype=torch.bool)
        source = torch.full(valid.shape, -1, device=field.device, dtype=torch.long)
        mass = torch.zeros_like(valid, dtype=torch.float32)
        xy[:, :used] = torch.where(occupied[..., None], point, torch.zeros_like(point))
        normal[:, :used] = torch.where(good[..., None], direction, torch.zeros_like(direction))
        valid[:, :used] = good
        source[:, :used] = torch.where(occupied, chosen, torch.full_like(chosen, -1))
        mass[:, :used] = selected_mass
        return dict(xy=xy, normal=normal, normal_valid=valid, source_ids=source, count=mass)


@torch.no_grad()
def fast_vertex_candidates(support, uv0, uv1, geo0, geo1, rho_all, part, K_event,
                           config: GeometryConfig):
    """Per-part nearest vertex and stable (angular distance, vertex id) order."""
    c = config
    batch, supports = support['normal_valid'].shape
    fx, fy = K_event[:, 0, 0], K_event[:, 1, 1]
    cx, cy = K_event[:, 0, 2], K_event[:, 1, 2]
    angular_vertex = torch.stack(((uv0[..., 0]-cx[:, None])/fx[:, None],
                                  (uv0[..., 1]-cy[:, None])/fy[:, None]), -1)
    point = support['xy']
    ray = torch.stack(((point[..., 0]-cx[:, None])/fx[:, None],
                        (point[..., 1]-cy[:, None])/fy[:, None]), -1)
    angular_dist = (angular_vertex[:, None]-ray[:, :, None]).square().sum(-1)
    pixel_dist = (uv0[:, None]-point[:, :, None]).square().sum(-1)
    inview0 = geo0 & (uv0[..., 0] >= 0) & (uv0[..., 0] <= c.width-1) & (uv0[..., 1] >= 0) & (uv0[..., 1] <= c.height-1)
    inview1 = geo1 & (uv1[..., 0] >= 0) & (uv1[..., 0] <= c.width-1) & (uv1[..., 1] >= 0) & (uv1[..., 1] <= c.height-1)
    eligible = (inview0[:, None] & inview1[:, None] &
                (pixel_dist <= rho_all[:, None].square()) & support['normal_valid'][..., None])
    body_part = torch.arange(16, device=uv0.device)
    distances = angular_dist[:, :, None].expand(-1, -1, 16, -1)
    distances = distances.masked_fill(~(eligible[:, :, None] &
                                       (part[None, None, None] == body_part[None, None, :, None])), torch.inf)
    distance, vid = distances.min(-1)  # first vertex on equal distance, as frozen
    # Python tuples sort by distance and then vertex id, not by body-part id.
    by_vid = torch.argsort(vid, dim=-1, stable=True)
    by_distance = torch.argsort(distance.gather(-1, by_vid), dim=-1, stable=True)
    order = by_vid.gather(-1, by_distance)[..., :c.candidates_per_support]
    valid = torch.isfinite(distance.gather(-1, order))
    ids = torch.where(valid, vid.gather(-1, order), torch.zeros_like(order))
    return ids.reshape(batch, supports*c.candidates_per_support), valid.reshape(batch, supports*c.candidates_per_support)


class FastJointGeometry(JointGeometry):
    """Evaluation-only geometry; no change to the registered measurement math."""

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
        support = fast_sensor_supports(field1, c)
        v0, _ = self._fk(q0);v1, _ = self._fk(q1)
        uv0_all, z0, geo0 = _project(v0, K_event, c.near_m)
        uv1_all, _, geo1 = _project(v1, K_event, c.near_m)
        S, A = c.max_supports, c.candidates_per_support
        records = S*A
        part = self.part.to(q0.device)
        fx, fy = K_event[:, 0, 0], K_event[:, 1, 1]
        rho_all = torch.sqrt(fx*fy)[:, None]*c.radius_m/z0.clamp_min(c.near_m)
        vertex_ids, candidate_mask = fast_vertex_candidates(
            support, uv0_all, uv1_all, geo0, geo1, rho_all, part, K_event, c)
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


__all__ = ['FastJointGeometry', 'fast_sensor_supports', 'fast_vertex_candidates']
