"""Opt-in, untrained field frontend for the original full S43 tracker.

No LNES input or base.forward hook exists here. The loaded S43 RN, MANO,
original loss and additive/left-SO3 root update are retained. This is a new
engineering prototype, independent of earlier field-reader decisions.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .field_query import FieldQueryBuilder, FieldQueryConfig, sample_query_patches
from tools.s43_integration.model import DEFAULT_CKPT, DEFAULT_CFG, load_legacy

ARMS = ('global_only', 'hand_box', 'static')
BIN_EDGES = (0, 7, 13, 19, 25, 32, 38, 44, 50)


def public_recency_weights():
    """Fixed positive weights of the eight native-stamp means; sum is one.

    This is only a public initialization heuristic, not an LNES equivalence.
    +1 makes the oldest native bin positive without consulting observations.
    """
    means = torch.tensor([(a+b-1)/2 for a, b in zip(BIN_EDGES[:-1], BIN_EDGES[1:])])
    return (means+1)/(means+1).sum()


def initialize_field_stem(old: nn.Conv2d):
    """Replace two old event channels by F16; C8/Q16 initially zero.

    Render slices and bias are copied exactly. Q/C weights remain trainable.
    """
    if old.in_channels != 4 or old.out_channels != 3 or old.kernel_size != (3, 3):
        raise ValueError('prototype requires original S43 4->3 k3 [two event | sil,inv] stem')
    stem = nn.Conv2d(42, 3, 3, padding=1, bias=old.bias is not None).to(old.weight)
    weights = public_recency_weights().to(old.weight)
    with torch.no_grad():
        stem.weight.zero_()
        for b in range(8):
            stem.weight[:, 2*b:2*b+2].copy_(old.weight[:, :2]*weights[b])
        stem.weight[:, 40:].copy_(old.weight[:, 2:])
        if stem.bias is not None:
            stem.bias.copy_(old.bias)
    return stem


def observed_activity(field40, previous_render):
    """Real F/Q positive count-support union previous-render support.

    F/Q contain log1p(nonnegative count density), whose >0 support equals
    raw counts. C8 coverage and learned stem features never define activity.
    """
    fine = field40[:, :16].detach().amax(1, keepdim=True) > 0
    coarse = field40[:, 24:40].detach().amax(1, keepdim=True) > 0
    render = previous_render.detach().abs().amax(1, keepdim=True) > 0
    return (fine | coarse | render).to(field40.dtype)


def _batched(value, B, shape, device, dtype, name):
    out = torch.as_tensor(value, device=device, dtype=dtype)
    if tuple(out.shape) == tuple(shape):
        out = out.unsqueeze(0).expand(B, *shape)
    elif tuple(out.shape) == (1, *shape):
        out = out.expand(B, *shape)
    if tuple(out.shape) != (B, *shape) or not torch.isfinite(out).all():
        raise ValueError(f'finite {name}[B,{shape}] required')
    return out


def _times(value, B, name):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    a = np.asarray(value)
    if a.ndim == 0:
        a = np.repeat(a, B)
    if a.shape != (B,) or not np.isfinite(a).all() or np.any(a != a.astype(np.int64)):
        raise ValueError(f'integer native-ms {name} scalar or [B] required')
    return a.astype(np.int64)


class S43FieldModel(nn.Module):
    """forward(field40, prev51, *, beta_input, K_source, ref_ms, cutoff_ms).

    field40: original log-count field F16+C8+Q16, [B,40,180,240].
    beta_input defaults to public zero shape. K_source is the 640x480 camera
    matrix, scaled exactly once by the original render_scale for queries.
    ref_ms is the timestamp of the supplied deployed/public prev51; histories
    are per-lane committed knots, never target states. Geometry is detached.
    No target is accepted by forward. loss(..., beta_target=...) is separate.
    """
    def __init__(self, base, arm='static', *, use_prev_mlp=True):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('global_only, hand_box or static required')
        required = dict(output_dim=51, pose_repr='mano_full_axis_angle', sparse_trunk=True,
                        predict_delta=True, prev_render=True, active_head=False, encoder_name='')
        if any(getattr(base, k, None) != v for k, v in required.items()):
            raise ValueError('requires the original MANO51 sparse render S43 model')
        if tuple(base.render_channels) != ('sil', 'inv') or base.root_compose not in ('add', 'so3'):
            raise ValueError('requires original sil/inv render and add or original left-SO3 root update')
        if getattr(base, 'sparse_noise', 0):
            raise ValueError('old LNES sparse-noise augmentation has no field semantics')
        self.base, self.arm, self.use_prev_mlp = base, arm, bool(use_prev_mlp)
        self.use_local = arm != 'global_only'
        self.base.conv1 = initialize_field_stem(base.conv1)
        self.local_encoder = nn.Sequential(nn.Conv2d(5, 8, 3, padding=1), nn.SiLU(),
            nn.Conv2d(8, 8, 3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d(1))
        self.local_head = nn.Sequential(nn.Linear(8*90*8, 128), nn.SiLU(), nn.Linear(128, 51))
        nn.init.zeros_(self.local_head[-1].weight)
        nn.init.zeros_(self.local_head[-1].bias)
        for p in self.parameters():
            p.requires_grad_(True)
        self.initialization = dict(recency_weights=public_recency_weights().tolist(),
            F16='copy old two-polarity weights times public normalized native-bin recency',
            C8='zero initialized, trainable', Q16='zero initialized, trainable',
            render='copy old sil/inv weights and old bias exactly',
            no_LNES_equivalence=True)

    def _local_patches(self, field, prev, beta, K, refs, cutoffs, histories):
        # Recover actual 2px cells from the field's nearest-upsampled payload.
        B = len(field)
        fine = field[:, :16].reshape(B, 8, 2, 180, 240)[:, :, :, ::2, ::2]
        coverage = field[:, 16:24, ::2, ::2].unsqueeze(2)
        cell = torch.cat((fine, coverage), dim=2)
        patches, metadata = [], []
        for lane in range(B):
            k_event = K[lane].detach().double().cpu().numpy().copy()
            k_event[:2] *= self.base.render_scale
            query = FieldQueryBuilder(prev[lane].detach(), beta[lane].detach(), k_event,
                int(refs[lane]), history=histories[lane], config=FieldQueryConfig(field_resolution=2)
                ).build(int(cutoffs[lane]), mode=self.arm)
            # Shared frozen helper returns [5,3,3,2,8,7,7,3]. Keep the full
            # 8 time x90 ordered anatomical/box slots through the local head.
            sampled = sample_query_patches(cell[lane], query).reshape(90, 8, 7, 7, 3)
            sampled = sampled.permute(1, 0, 4, 2, 3).contiguous()
            flags = torch.stack((query.in_view, query.full_interp_support), dim=2
                ).reshape(8, 90, 2, 7, 7).to(field)
            patches.append(torch.cat((sampled, flags), dim=2))
            metadata.append(query.meta)
        return torch.stack(patches), metadata

    def _compose(self, delta, prev, feats):
        out = delta + prev.to(delta.dtype)
        if self.base.root_compose == 'so3':
            out = self.base._compose_root_so3(out, delta, prev)
        if self.base.transl_head is not None:
            p_t = prev[:, :3].to(delta.dtype)
            t = p_t + self.base.transl_head(torch.cat((feats.detach().to(delta.dtype), p_t), 1))
            out = torch.cat((t, out[:, 3:]), dim=1)
        return out

    def forward(self, field40, prev51, *, K_source, ref_ms, cutoff_ms,
                beta_input=None, histories=None, return_aux=False):
        if not torch.is_tensor(field40) or not field40.is_floating_point() or field40.shape[1:] != (40, 180, 240):
            raise ValueError('floating observed field[B,40,180,240] required')
        if field40.ndim != 4 or not torch.isfinite(field40).all() or torch.any(field40 < 0):
            raise ValueError('finite nonnegative log-count/coverage field required')
        B = len(field40)
        prev = _batched(prev51, B, (51,), field40.device, torch.float32, 'prev51')
        beta = _batched(torch.zeros(10) if beta_input is None else beta_input, B, (10,),
                        field40.device, torch.float32, 'beta_input')
        K = _batched(K_source, B, (3, 3), field40.device, torch.float32, 'K_source')
        refs, cutoffs = _times(ref_ms, B, 'ref_ms'), _times(cutoff_ms, B, 'cutoff_ms')
        if np.any(refs > cutoffs):
            raise ValueError('deployed anchor reference cannot be in query future')
        histories = [()] * B if histories is None else histories
        if len(histories) != B:
            raise ValueError('one deployed history per batch lane required')
        with torch.no_grad():
            render = self.base._render_prev(prev.detach(), beta.detach(), K.detach()
                ).permute(0, 3, 1, 2).contiguous().to(field40)
        activity = observed_activity(field40, render)
        stem = self.base.conv1(torch.cat((field40, render), dim=1)) * activity
        feats, masks = self.base.rn.pooled(stem, activity)
        out = self.base.rn.fc(feats)
        local_delta, local, query_meta = torch.zeros_like(out), None, []
        if self.use_local:
            local, query_meta = self._local_patches(field40, prev, beta, K, refs, cutoffs, histories)
            encoded = self.local_encoder(local.reshape(-1, 5, 7, 7)).reshape(B, 8*90*8)
            local_delta = self.local_head(encoded)
            out = out + local_delta
        if self.use_prev_mlp and self.base.prevpos_embed:
            pm = self.base.prev_mlp(prev.to(out.dtype))
            if not self.base.prev_mlp_transl:
                t = self.base.slices.transl
                pm = torch.cat((pm[:, :t.start], torch.zeros_like(pm[:, t]), pm[:, t.stop:]), dim=1)
            out = out + pm
        prediction = self._compose(out, prev, feats)
        if not torch.isfinite(prediction).all():
            raise ArithmeticError('nonfinite field prediction; no hold/clamp fallback')
        if not return_aux:
            return prediction
        return prediction, dict(activity=activity, masks=masks, render=render,
            local=local, local_delta=local_delta, delta=out, query_metadata=query_meta,
            compose=self.base.root_compose, zero_event_gate=False, LNES_used=False)

    def loss(self, prediction, target51, *, beta_target):
        """Original objective, explicit supervision-only shape; no geometry call."""
        if beta_target is None:
            raise ValueError('explicit beta_target required; pass zeros for mean-shape supervision')
        return self.base._compute_loss(prediction, target51, beta_target)

    def _fk(self, params, betas):
        return self.base._fk(params, betas)

    def counts(self):
        n = lambda m: sum(p.numel() for p in m.parameters())
        local = n(self.local_encoder) + n(self.local_head)
        prev = n(self.base.prev_mlp) if self.base.prevpos_embed else 0
        return dict(total=n(self), trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
            structurally_active=n(self)-(0 if self.use_local else local)-(0 if self.use_prev_mlp else prev),
            field_stem=n(self.base.conv1), local_encoder=n(self.local_encoder), local_head=n(self.local_head),
            local_active=self.use_local, prev_mlp=prev, prev_mlp_active=self.use_prev_mlp and self.base.prevpos_embed,
            base_and_new_required_parameters_trainable=True,
            inactive_parameters_still_registered=True, BN_follows_train_mode=True)

    def metadata(self):
        return dict(schema='s43_field_frontend_untrained_prototype_v1', arm=self.arm,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            input='field40 F16+C8+Q16 log1p once, no added FIT normalization or clipping',
            activity='positive actual F16/Q16 count support union detached previous sil/inv render; C8 excluded',
            K_space='source640x480; original render scales internally; query scales once by render_scale',
            beta_input='external or public zeros; never taken from beta_target or base context',
            local_shape=[8, 90, 5, 7, 7], local_geometry='detached deployed/public prev and history',
            local_channels='F2,finecoverage,in_view,full_interp_support',
            root_compose=self.base.root_compose, root_and_AA_update='original S43 convention; not Stage4 right-action decoder',
            loss='unchanged base._compute_loss with separate explicit beta_target',
            old_empty_event_gate_used=False, new_hold_gate=False, LNES_used=False,
            original_backbone=True, original_FK=True, original_loss=True,
            use_prev_mlp=self.use_prev_mlp, initialization=self.initialization, counts=self.counts(),
            trained=False, prior_gate_status='no inherited scientific pass; opt-in engineering only')


def load_s43_field_model(*, arm='static', use_prev_mlp=True, device='cpu',
                         base_checkpoint=DEFAULT_CKPT, base_config=DEFAULT_CFG, checkpoint=None):
    base, cfg = load_legacy(base_checkpoint, base_config, device)
    model = S43FieldModel(base, arm, use_prev_mlp=use_prev_mlp).to(device)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location=device)
        if (payload.get('schema') != 's43_field_frontend_checkpoint_v1' or payload['arm'] != arm
                or payload['use_prev_mlp'] != use_prev_mlp):
            raise ValueError('explicit field frontend checkpoint identity required')
        model.load_state_dict(payload['model_state_dict'], strict=True)
    return model, copy.deepcopy(cfg)
