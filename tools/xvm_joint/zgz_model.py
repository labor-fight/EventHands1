"""Opt-in XJ adapter for the existing raw-packet recursive evaluator.

The adapter opens no data or labels.  It converts the evaluator's relative
float32-second event columns back to integer microseconds and reuses the
frozen causal F16 builder.  Every caller must reset at each valid-run boundary.
The original S43 is a sparse-trunk *image-input* model: its canonical call is
``base(lnes, previous, beta0, K)`` even though the surrounding adapter accepts
raw event packets.  No canonical model or training module is modified here.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import numpy as np
import torch
from torch import nn

from .data import HEIGHT, WIDTH, advance_field16
from .geometry import GeometryConfig
from .evaluation_geometry import FastJointGeometry
from .model import JointLocalOperator, fixed_reference_joint_solve
from .train import camera_valid


VARIANTS = ('primary', 'spatial', 'global', 'derived')


def relative_packet_events(batch):
    """Recover exact integer-us ``[x,y,relative_us,p]`` for a batch-1 packet.

    Official packets have columns ``[batch,x,y,t_rel_seconds,stored_p]``.
    Float32 seconds over at most 200ms recover the original integer-us stamp
    by rounding, rather than flooring (which can move an event into an earlier
    millisecond).  Integer coordinates and stored polarity are preserved.
    """
    if int(batch.ptr.numel()) != 2 or int(batch.ptr[0]) != 0:
        raise ValueError('XJ formal packet adapter requires batch size one')
    e = batch.events.detach().cpu().numpy()
    if e.ndim != 2 or e.shape[1] != 5 or int(batch.ptr[1]) != len(e):
        raise ValueError('Expected [batch,x,y,relative_seconds,stored_p] events')
    duration = int((batch.t_end_us - batch.t_start_us).reshape(-1)[0])
    if duration < 50000 or duration > 300000 or duration % 1000:
        raise ValueError('A 50--300ms integer-ms evidence packet is required')
    if not np.isfinite(e).all() or (len(e) and np.any(e[:, 0] != 0)):
        raise ValueError('Finite single-packet event columns required')
    out = np.empty((len(e), 4), np.int64)
    if len(e):
        xy = np.rint(e[:, 1:3]).astype(np.int64)
        polarity = np.rint(e[:, 4]).astype(np.int64)
        us = np.rint(e[:, 3].astype(np.float64) * 1e6).astype(np.int64)
        if (np.any(xy != e[:, 1:3]) or np.any(polarity != e[:, 4])
                or np.any(xy[:, 0] < 0) or np.any(xy[:, 0] >= WIDTH)
                or np.any(xy[:, 1] < 0) or np.any(xy[:, 1] >= HEIGHT)
                or np.any((polarity != 0) & (polarity != 1))
                or np.any(us < 0) or np.any(us >= duration)):
            raise ValueError('Event coordinate, stored polarity, or time is out of range')
        # This is an engineering invariant of ET._window_events, not a claim
        # that a generic float32 stream carries exact submicrosecond times.
        reconstructed = us.astype(np.float32) * np.float32(1e-6)
        if not np.array_equal(reconstructed, e[:, 3].astype(np.float32)):
            raise ValueError('Packet timestamps are not the declared integer-us ET encoding')
        out[:, :2], out[:, 2], out[:, 3] = xy, us, polarity
    return out, duration // 1000


def packet_lnes(events_relative_us, window_ms):
    """The full evidence window's original millisecond last-event surface."""
    e = np.asarray(events_relative_us)
    image = np.zeros((HEIGHT, WIDTH, 2), np.float32)
    if len(e):
        np.maximum.at(image, (e[:, 1], e[:, 0], e[:, 3]),
                      (e[:, 2] // 1000).astype(np.float32) / np.float32(window_ms))
    return image


class PacketFieldState:
    """Run-local causal F16 cache; overlapping evidence is ingested only once."""
    def __init__(self):
        self.reset()

    def reset(self, start_ms=0, sequence_id=None):
        self.run_start_ms = self.cursor_ms = int(start_ms)
        self.sequence_id = sequence_id
        self.queries = 0
        self.latest_us = np.full((2, HEIGHT, WIDTH), -1, np.int64)
        self.field = np.zeros((16, HEIGHT, WIDTH), np.float32)

    def advance(self, events_relative_us, window_ms):
        e = np.asarray(events_relative_us, np.int64)
        window_ms = int(window_ms)
        if window_ms < 50 or window_ms > 300 or window_ms % 50:
            raise ValueError('XJ field adapter supports 50ms-stepped evidence windows only')
        # A generic first long window can prime the causal field in 50ms
        # segments. Formal evalx + clip_run_start begins at W50 instead.
        cutoff = self.cursor_ms + (window_ms if self.queries == 0 else 50)
        window_start = cutoff - window_ms
        new_start = self.cursor_ms
        both_exposed = self.queries > 0
        previous_field = self.field
        last_new = np.zeros((0, 4), np.int64)
        for start in range(new_start, cutoff, 50):
            relative_start_us = (start - window_start) * 1000
            selected = (e[:, 2] >= relative_start_us) & (e[:, 2] < relative_start_us + 50000)
            native = e[selected].copy()
            native[:, 2] += window_start * 1000
            previous_field = self.field
            self.field, self.latest_us, _ = advance_field16(
                self.latest_us, native, start_ms=start, cutoff_ms=start + 50,
                run_start_ms=self.run_start_ms, history_ms=600, log_counts=False)
            last_new = native
        self.cursor_ms = cutoff
        self.queries += 1
        return dict(field0=previous_field, field1=self.field, current_events=last_new,
                    start_ms=cutoff-50, cutoff_ms=cutoff, both_exposed=both_exposed,
                    event_count=len(last_new), window_ms=window_ms)

    def snapshot(self):
        return copy.deepcopy(self.__dict__)

    def restore(self, state):
        self.__dict__ = copy.deepcopy(state)


@torch.no_grad()
def _selected_projection(base, states, K_event, vertex_ids):
    vertices, _ = base._fk(states.float(), states.new_zeros((len(states), 10)))
    if not bool(torch.isfinite(vertices).all()) or bool((vertices[..., 2] <= 1e-4).any()):
        raise ArithmeticError('Previous absolute proposal is camera-invalid')
    z = vertices[..., 2]
    uv = torch.stack((K_event[:, None, 0, 0] * vertices[..., 0] / z + K_event[:, None, 0, 2],
                      K_event[:, None, 1, 1] * vertices[..., 1] / z + K_event[:, None, 1, 2]), -1)
    return uv.gather(1, vertex_ids[..., None].expand(-1, -1, 2))


class XJPacketModel(nn.Module):
    """Fixed trained local operator around the canonical frozen S43 proposal.

    The evaluator owns the emitted previous pose (including any external
    S39-stack filtering). This class owns event history only and never reads
    ``batch.target`` or ``batch.betas``. ``reset_state`` must be called for
    every valid run. No ``get_state`` method is exposed: evalx would retain a
    multimegabyte field snapshot for every scored frame unnecessarily.
    """
    encoder_name = 'xvm_joint_packet'
    xyz_assoc_prev = 'self'

    def __init__(self, base, local=None, geometry=None, *, variant='primary'):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError('Unknown XJ evaluation variant')
        if variant != 'global' and local is None:
            raise ValueError('A trained local operator is required')
        self.base = base
        self.variant = variant
        if variant == 'spatial':
            self.local = JointLocalOperator(replace(local.config, motion_conditioned_assignment=False))
            self.local.to(next(local.parameters()).device)
            self.local.load_state_dict(local.state_dict(), strict=True)
        else:
            self.local = local
        self.geometry = geometry or FastJointGeometry(base, GeometryConfig())
        if getattr(base, 'encoder_name', ''):
            raise ValueError('XJ stage1 requires the canonical image-input S43')
        if getattr(base, 'root_compose', 'add') != 'add':
            raise ValueError('XJ requires the frozen additive physical51 chart')
        self.field_state = PacketFieldState()
        self.previous_absolute = None
        self.last_audit = {}
        self.last_lnes = None
        self.last_current_count = 0
        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad_(False)
        if self.local is not None:
            self.local.eval()
            for p in self.local.parameters():
                p.requires_grad_(False)

    def train(self, mode=True):
        super().train(False)
        return self

    def reset_run(self, start_ms=0, sequence_id=None):
        self.field_state.reset(start_ms=start_ms, sequence_id=sequence_id)
        self.previous_absolute = None
        self.last_audit = {}
        self.last_lnes = None
        self.last_current_count = 0

    def reset_state(self, prev=None):
        # Formal run initialisation is evaluator-owned; no target-derived
        # value is inspected or incorporated into the observation cache.
        self.reset_run()

    def snapshot_state(self):
        return dict(field=self.field_state.snapshot(),
                    previous_absolute=None if self.previous_absolute is None else self.previous_absolute.detach().clone(),
                    last_audit=copy.deepcopy(self.last_audit),
                    last_lnes=None if self.last_lnes is None else self.last_lnes.detach().clone(),
                    last_current_count=self.last_current_count)

    def restore_state(self, state):
        self.field_state.restore(state['field'])
        self.previous_absolute = (None if state['previous_absolute'] is None
                                  else state['previous_absolute'].detach().clone())
        self.last_audit = copy.deepcopy(state['last_audit'])
        self.last_lnes = None if state['last_lnes'] is None else state['last_lnes'].detach().clone()
        self.last_current_count = int(state['last_current_count'])

    def forward(self, batch):
        return self.forward_packet(batch)

    @torch.no_grad()
    def forward_packet(self, batch):
        previous = batch.prev_state
        if previous.shape != (1, 51):
            raise ValueError('One physical51 previous state is required')
        relative, window = relative_packet_events(batch)
        fields = self.field_state.advance(relative, window)
        device = previous.device
        K = batch.camera_K.to(device=device, dtype=torch.float32)
        beta0 = previous.new_zeros((1, 10))
        image = torch.from_numpy(packet_lnes(relative, window)).to(device)[None]
        self.last_lnes = image.detach()
        self.last_current_count = fields['event_count']
        absolute_raw = self.base(image, previous, betas=beta0, camera_K=K).float()
        absolute_valid = camera_valid(self.base, absolute_raw, beta0)
        absolute = torch.where(absolute_valid[:, None], absolute_raw, previous)
        delta = torch.zeros_like(absolute)
        aux = None
        if self.variant != 'global':
            f0 = torch.from_numpy(fields['field0'].copy()).to(device)[None]
            f1 = torch.from_numpy(fields['field1'].copy()).to(device)[None]
            prepared = self.geometry.prepare(f0, f1, previous, absolute, K,
                both_exposed=torch.tensor([fields['both_exposed']], dtype=torch.bool, device=device))
            _, delta, aux = self.local(**prepared.model_kwargs)
            if self.variant == 'derived':
                saved = previous if self.previous_absolute is None else self.previous_absolute
                uv = _selected_projection(self.base, saved, prepared.K_event, prepared.vertex_ids)
                a = ((uv-prepared.uv0)*aux['normal']).sum(-1)
                d = ((prepared.uv1-uv)*aux['normal']).sum(-1)
                _, delta, _ = fixed_reference_joint_solve(
                    torch.stack((a, d), -1), aux['covariance_px'], prepared.B0, prepared.B1,
                    aux['normal'], prepared.nominal_delta, aux['weights'], self.local.state_scale,
                    start_damping=self.local.config.start_damping, end_damping=self.local.config.end_damping)
        proposed = absolute + delta
        proposed_valid = camera_valid(self.base, proposed, beta0)
        emit = proposed_valid & (fields['event_count'] > 0)
        prediction = torch.where(emit[:, None], proposed, previous)
        if not bool(torch.isfinite(prediction).all()):
            raise ArithmeticError('Nonfinite XJ emitted state')
        self.previous_absolute = absolute.detach().clone()
        self.last_audit = dict(variant=self.variant, run_sequence_id=self.field_state.sequence_id,
            start_ms=fields['start_ms'], cutoff_ms=fields['cutoff_ms'], query=self.field_state.queries,
            window_ms=window, new_native_events=fields['event_count'], both_snapshots_valid=fields['both_exposed'],
            active_records=0 if aux is None else int(aux['active'].sum()),
            absolute_camera_invalid=int(not absolute_valid[0]), local_camera_invalid=int(not proposed_valid[0]),
            empty_fallback=int(fields['event_count']==0),
            raw_update_translation_mm=float(delta[0, :3].norm()*1000),
            raw_update_root_rad=float(delta[0, 3:6].norm()),
            raw_update_fingers_rad=float(delta[0, 6:].norm()),
            beta_input='zeros10', target_accessed=False, actual_beta_accessed=False,
            global_input='original millisecond LNES, canonical sparse-trunk image forward',
            local_history='causal F16 history600ms; last50ms raw count bins; overlapping events ingested once')
        return prediction
