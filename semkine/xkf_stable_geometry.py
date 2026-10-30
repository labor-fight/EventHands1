"""Opt-in canonical single-sample geometry for frozen XKF inference.

No rounding or changed association thresholds. ``per_sample_geometry`` runs the
original FP32 FK/project/ray rule separately for every row. ``per_sample_i``
also canonicalizes the neural I path, preventing batch-shaped GEMM roundoff from
feeding the next frame's discrete geometry. Original source and weights remain
unchanged; the default is off and these modes are inference-only.
"""
from __future__ import annotations

import torch

from semkine.encoder import event_tokens
from semkine.events import EV_X, EV_Y
from semkine.xyz_mesh import associate_rays, stride_sample
from semkine.xyz_observation import project_state_joints
from semkine.xyz_recovery_study import select_packet_rows
from semkine.xkf_centroid import XKFCentroidModel

MODES = ('off', 'per_sample_geometry', 'per_sample_i', 'per_sample_model')


class XKFStableGeometryModel(XKFCentroidModel):
    def __init__(self, cfg=None, num_classes=None, fusion_spec=None, stability_mode='off'):
        super().__init__(cfg=cfg, num_classes=num_classes, fusion_spec=fusion_spec)
        self.configure_stability(stability_mode)

    def configure_stability(self, mode='off'):
        if mode not in MODES:
            raise ValueError(f'stability mode must be one of {MODES}')
        self.stability_mode = mode
        self.stability_last_rows = None
        return self

    def incremental_prediction(self, batch):
        if self.stability_mode == 'off':
            return super().incremental_prediction(batch)
        if self.training:
            raise ValueError('stability candidate is inference-only; no implicit training change')
        if self.stability_mode not in ('per_sample_i', 'per_sample_model') or batch.batch_size <= 1:
            return super().incremental_prediction(batch)
        result, traces = [], []
        for row in range(batch.batch_size):
            selected = select_packet_rows(batch, torch.tensor([row], device=batch.prev_state.device))
            result.append(super().incremental_prediction(selected))
            if getattr(self, 'xyz_keep_aux', False):
                traces.append(self.xyz_last if int(selected.counts[0]) else None)
        self.stability_last_rows = traces or None
        return torch.cat(result, 0)

    def forward_packet(self, batch):
        # This stronger optional mode canonicalizes D and fusion too. It calls
        # the genuine deployed forward for every row, without a D cache.
        if self.stability_mode != 'per_sample_model' or batch.batch_size <= 1:
            return super().forward_packet(batch)
        if self.training:
            raise ValueError('canonical full model is inference-only')
        predictions = []
        for row in range(batch.batch_size):
            selected = select_packet_rows(batch, torch.tensor([row], device=batch.prev_state.device))
            predictions.append(super().forward_packet(selected))
        return torch.cat(predictions, 0)

    @torch.no_grad()
    def canonical_geometry(self, batch, prev, betas, camera, src, mask):
        """Original per-row FP32 geometry, with the same rays and tie rules."""
        inverse = torch.argsort(self.mano.mano16_tips_to_openpose21)
        collected = {key: [] for key in ('verts', 'joints', 'uv', 'valid', 'raw', 'outside', 'd')}
        associations = []
        with torch.autocast(device_type=prev.device.type, enabled=False):
            for row in range(len(prev)):
                vertices, joints = self._fk(prev[row:row + 1].float(), betas[row:row + 1].float())
                if not bool(torch.isfinite(vertices).all() and torch.isfinite(joints).all()):
                    raise FloatingPointError('nonfinite MANO FK')
                intrinsics = self._intrinsics(camera[row:row + 1].float())
                uv, valid, raw, outside = project_state_joints(joints[:, inverse[:16]], intrinsics,
                                                               self.render_h, self.render_w)
                fx, fy, cx, cy = intrinsics
                events = batch.events[src[row]].unsqueeze(0)
                px, py = events[..., EV_X].float(), events[..., EV_Y].float()
                rays = torch.stack([(px - cx[:, None]) / fx[:, None],
                                    (py - cy[:, None]) / fy[:, None], torch.ones_like(px)], -1)
                fpx = (fx * fy).sqrt()
                assoc = associate_rays(rays, mask[row:row + 1], vertices, self.mano.f, self.xyz.vert_faces,
                    band=self.xyz_band_px / fpx, front_tol=self.xyz_front_px / fpx,
                    z_tol=self.xyz_z_tol, k=self.xyz_k)
                for key, value in zip(collected, (vertices, joints, uv, valid, raw, outside, rays)):
                    collected[key].append(value)
                associations.append(assoc)
        return {**{key: torch.cat(value, 0) for key, value in collected.items()},
                'assoc': {key: torch.cat([value[key] for value in associations], 0) for key in associations[0]}}

    def _xyz_mesh_forward(self, batch, prev):
        if self.stability_mode != 'per_sample_geometry':
            return super()._xyz_mesh_forward(batch, prev)
        if self.training:
            raise ValueError('canonical geometry is inference-only')
        events, ptr = batch.events, batch.ptr
        if events.shape[0] == 0:
            return super()._xyz_mesh_forward(batch, prev)
        if self.route_prev_override is not None:
            raise ValueError('stable geometry preserves the no-prior-override contract')
        betas, camera = self._resolve_betas_K(prev, batch.betas, batch.camera_K)
        src, mask = stride_sample(ptr, self.xyz_max_nodes)
        tok = event_tokens(events, ptr, batch.delta_t_s, self.render_h, self.render_w)
        tok = tok[src.reshape(-1)].view(batch.batch_size, -1, tok.shape[-1]) * mask[..., None]
        geometry = self.canonical_geometry(batch, prev, betas, camera, src, mask)
        vertices, rays, assoc = geometry['verts'], geometry['d'], geometry['assoc']
        uv, valid = geometry['uv'], geometry['valid']
        with torch.no_grad(), torch.autocast(device_type=prev.device.type, enabled=False):
            state = torch.cat([prev[:, :3].float() / .1, prev[:, 3:].float(),
                (uv / uv.new_tensor([self.render_w, self.render_h])).flatten(1), valid.float()], -1)
            if not self.xdl_state_t:
                state = state[:, 3:]
        observation = self.observation_bank.encode(events, ptr, batch.delta_t_s)
        if not torch.equal(src, observation['src']) or not torch.equal(mask, observation['mask']):
            raise RuntimeError('bank support differs from original XYZ support')
        bank, weights = self.observation_bank.query(observation, uv, self.xdl_query)
        self._recovery_inputs = bank, state
        try:
            delta, stats, aux = self.xyz(tok, rays, mask, vertices, assoc, prev[:, 3:51], ablate=self.ablate_evidence)
        finally:
            self.reset_recovery_context()
        self.route_stats = stats
        if getattr(self, 'xyz_keep_aux', False):
            self.xyz_last = {'assoc': assoc, 'd': rays, 'mask': mask, 'verts': vertices, 'tok': tok,
                'delta': delta.detach(), **{key: value.detach() for key, value in aux.items()},
                'xdl_uv': uv, 'xdl_raw_uv': geometry['raw'], 'xdl_projection_valid': valid,
                'xdl_outside': geometry['outside'], 'xdl_weights': weights.detach(),
                'xdl_bank': bank.detach(), 'xdl_state': state, 'xdl_src': src}
        return delta


__all__ = ['MODES', 'XKFStableGeometryModel']
