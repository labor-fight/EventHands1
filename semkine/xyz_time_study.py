"""Isolated four-arm XYZ geometry/time study; no changes to the default model.

``TimeXYZModel`` retains the original XYZ graph and its 17 slots.  Every arm
adds the same 51 motion inputs to fc1; ``XYZ_GEOM`` retains its original grouped
meaning.  ``XYZ_CAUSAL_TIME_MODE`` chooses ``static`` or ``extrapolate4``.
Only past states determine motion.  Current packet targets are never read here.

Training pairs use a detached, static, zero-history leading prediction.  The
main packet keeps its original conditioning state.  Evaluation history is
explicit: packet-local timestamps/sequence IDs cannot identify a stream/run.
The caller resets at each run and supplies the previous incoming state and its
actual elapsed interval.  This module never advances history on its own.
"""
from __future__ import annotations

import contextlib
import copy
from typing import Optional

import torch
from torch import nn

from model import MNISTModel
from semkine.encoder import event_tokens
from semkine.events import EV_T, EV_X, EV_Y
from semkine.lie import so3_exp, so3_log
from semkine.xyz_mesh import L_REL, Z_REF, Z_SCALE, associate_rays, stride_sample


TIME_MODES = ("static", "extrapolate4")
SLICE_CENTERS = (0.125, 0.375, 0.625, 0.875)


def _batch_scalar(value, reference, name):
    value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if value.numel() == 1:
        value = value.reshape(1).expand(reference.shape[0])
    if value.shape != (reference.shape[0],):
        raise ValueError(f"{name} must be scalar or shape [B]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def _full_rotations(params, hands_mean):
    """Match pose_repr.decode_to_mano_inputs: local45 is residual + mean."""
    full = torch.cat([params[:, 3:6], params[:, 6:] + hands_mean.reshape(1, 45).to(params)], -1)
    return so3_exp(full.reshape(-1, 16, 3))


@torch.no_grad()
def physical_motion_delta(older, newer, hands_mean):
    """Past translation change and LEFT SO(3) changes, not axis-angle differences."""
    if older.shape != newer.shape or older.ndim != 2 or older.shape[1] != 51:
        raise ValueError("history states must have the same shape [B,51]")
    dtype = torch.float64 if older.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=older.device.type, enabled=False):
        old, new = older.to(dtype), newer.to(device=older.device, dtype=dtype)
        r_old, r_new = _full_rotations(old, hands_mean), _full_rotations(new, hands_mean)
        rotations = so3_log(r_new @ r_old.transpose(-1, -2)).reshape(-1, 48)
        delta = torch.cat([new[:, :3] - old[:, :3], rotations], -1)
        # Identical raw states are an exact zero, even at a principal-AA seam.
        same = (new == old).all(-1)
        return torch.where(same[:, None], torch.zeros_like(delta), delta)


@torch.no_grad()
def extrapolate_state(anchor, velocity, elapsed_s, hands_mean):
    """Extrapolate from the current conditioning state using past physical velocity.

    Translation stays in the repository's transl coordinates.  Root and local
    rotations use LEFT multiplication; local rotations add/subtract MANO mean.
    Zero-speed samples return the original coordinates bit for bit.
    """
    if anchor.ndim != 2 or anchor.shape[1] != 51 or velocity.shape != anchor.shape:
        raise ValueError("anchor and velocity must have shape [B,51]")
    dtype = torch.float64 if anchor.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=anchor.device.type, enabled=False):
        a, v = anchor.to(dtype), velocity.to(device=anchor.device, dtype=dtype)
        elapsed = _batch_scalar(elapsed_s, a, "elapsed_s")
        increment = v * elapsed[:, None]
        rotations = so3_exp(increment[:, 3:].reshape(-1, 16, 3)) @ _full_rotations(a, hands_mean)
        full = so3_log(rotations).reshape(-1, 48)
        result = torch.cat([a[:, :3] + increment[:, :3], full[:, :3],
                            full[:, 3:] - hands_mean.reshape(1, 45).to(a)], -1)
        keep = (v == 0).all(-1) | (elapsed == 0)
        return torch.where(keep[:, None], a, result)


@torch.no_grad()
def associate_time_slices(d, mask, verts_by_time, time_bin, faces, vert_faces, **kwargs):
    """Associate four time slices, returning each record in its original packet slot."""
    if verts_by_time.ndim != 4 or verts_by_time.shape[1] != 4:
        raise ValueError("verts_by_time must have shape [B,4,V,3]")
    if time_bin.shape != mask.shape or d.shape[:2] != mask.shape:
        raise ValueError("ray/mask/time-bin shapes differ")
    if bool(((time_bin < 0) | (time_bin > 3)).any()):
        raise ValueError("time bins must be in [0,3]")
    merged = None
    for index in range(4):
        selected = mask & (time_bin == index)
        part = associate_rays(d, selected, verts_by_time[:, index], faces, vert_faces, **kwargs)
        if merged is None:
            merged = {key: torch.zeros_like(value) for key, value in part.items()}
        for key, value in part.items():
            support = selected
            while support.ndim < value.ndim:
                support = support.unsqueeze(-1)
            merged[key] = torch.where(support, value, merged[key])
    return merged


class TimeXYZModel(MNISTModel):
    """Opt-in subclass with explicit history and unchanged default MNISTModel source.

    Public evaluation contract::

        model.reset_time_history()  # every valid-run start
        model.set_time_history(previous_incoming_state, past_interval_s,
                               query_start_offset_s=0.001)
        predicted = model.forward_packet(packet_with_current_incoming_state)

    First packet: no history, offset zero.  With the original 50-ms evaluator,
    the first past interval is 49 ms and subsequent intervals are 50 ms; the
    runner supplies actual state timestamp differences.  Random training
    minibatches never inherit a previous minibatch's history.
    """

    MODEL_KEYS = MNISTModel.MODEL_KEYS | frozenset({"XYZ_CAUSAL_STUDY", "XYZ_CAUSAL_TIME_MODE"})

    def __init__(self, cfg=None, num_classes=None, *, time_mode=None, geom: Optional[bool] = None):
        cfg = copy.deepcopy(cfg or {})
        model_cfg = cfg.setdefault("MODEL", {})
        if model_cfg.get("XYZ_CAUSAL_STUDY", True) is not True:
            raise ValueError("TimeXYZModel requires XYZ_CAUSAL_STUDY=true")
        mode = time_mode or model_cfg.get("XYZ_CAUSAL_TIME_MODE", "static")
        mode = "extrapolate4" if mode == "causal4" else mode
        if mode not in TIME_MODES:
            raise ValueError(f"XYZ_CAUSAL_TIME_MODE must be one of {TIME_MODES}")
        model_cfg["XYZ_CAUSAL_STUDY"] = True
        model_cfg["XYZ_CAUSAL_TIME_MODE"] = mode
        if geom is not None:
            model_cfg["XYZ_GEOM"] = bool(geom)
        super().__init__(cfg=cfg, num_classes=num_classes)
        if not self.xyz_mesh or self.pose_repr != "mano_full_axis_angle":
            raise ValueError("time study requires ENCODER=xyz_mesh and the 51D residual layout")
        if self.xyz.residual_mode != "legacy" or self.xyz_assoc_mode != "hit_first" or self.xyz_assoc_prev != "self":
            raise ValueError("time study requires legacy residual and deployable hit_first association")
        if self.unroll_p != 0 or self.rollout_p != 0 or self.self_p != 0:
            raise ValueError("time study preserves main.prev_state; other conditioning curricula must be disabled")
        self.time_mode = mode
        # Runtime-only OOD diagnostic.  No fifth training condition or token
        # change: +2 assigns events to geometry a half-window away.
        self.query_bin_shift = 0
        old = self.xyz.fc1
        expanded = nn.Linear(old.in_features + 51, old.out_features, bias=old.bias is not None)
        expanded = expanded.to(device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            expanded.weight[:, :old.in_features].copy_(old.weight)
            expanded.weight[:, old.in_features:].zero_()
            if old.bias is not None:
                expanded.bias.copy_(old.bias)
        self.xyz.fc1 = expanded
        self.xyz.fc1.register_forward_pre_hook(self._append_motion_delta)
        self.xyz.rel.register_forward_pre_hook(self._record_time_geometry)
        self.reset_time_history()

    def reset_time_history(self):
        self._time_history = None
        self._query_start_offset_s = 0.0
        self._time_mode_override = None
        self._motion_delta_context = None
        self._relation_geometry_context = None

    def set_time_history(self, previous_state, interval_s, query_start_offset_s=0.0, *, current_state=None):
        """Supply past endpoints; optional newer endpoint is used only by training lead.

        With ``current_state=None`` the next packet's incoming prev is the
        newer endpoint.  Supplying lead.pred keeps main.prev independent.
        Zero-length history has zero velocity; negative/nonfinite intervals fail.
        """
        if previous_state.ndim != 2 or previous_state.shape[1] != 51:
            raise ValueError("previous_state must have shape [B,51]")
        if current_state is not None and current_state.shape != previous_state.shape:
            raise ValueError("current_state history shape differs from previous_state")
        interval = _batch_scalar(interval_s, previous_state.float(), "interval_s")
        offset = _batch_scalar(query_start_offset_s, previous_state.float(), "query_start_offset_s")
        if bool((interval < 0).any()) or bool((offset < 0).any()):
            raise ValueError("history interval and query offset must be nonnegative")
        self._time_history = (previous_state.detach().clone(),
                              None if current_state is None else current_state.detach().clone(),
                              interval.detach().clone())
        self._query_start_offset_s = offset.detach().clone()

    @torch.no_grad()
    def prepare_history(self, current_prev):
        """Return detached physical velocity [B,51] without advancing stream history."""
        if self._time_history is None:
            return current_prev.new_zeros(current_prev.shape, dtype=torch.float32)
        old, newer, interval = self._time_history
        if old.shape != current_prev.shape:
            raise ValueError("history batch shape differs; reset between streams/batch sizes")
        old = old.to(device=current_prev.device, dtype=torch.float32)
        newer = current_prev if newer is None else newer.to(device=current_prev.device, dtype=torch.float32)
        delta = physical_motion_delta(old, newer, self.mano.hands_mean)
        interval = interval.to(delta)
        velocity = delta / interval.clamp_min(1e-12)[:, None]
        return torch.where((interval > 0)[:, None], velocity, torch.zeros_like(velocity)).detach()

    @contextlib.contextmanager
    def _preserve_history(self):
        names = ("_time_history", "_query_start_offset_s", "_time_mode_override",
                 "_motion_delta_context", "_relation_geometry_context")
        saved = {name: getattr(self, name) for name in names}
        try:
            yield
        finally:
            for name, value in saved.items():
                setattr(self, name, value)

    def _predict_batch(self, batch):
        if isinstance(batch, (tuple, list)) and len(batch) == 2 and all(hasattr(p, "events") for p in batch):
            lead, main = batch
            with self._preserve_history():
                self.reset_time_history()
                self._time_mode_override = "static"
                device_type = lead.prev_state.device.type
                amp_enabled = (torch.is_autocast_cpu_enabled() if device_type == "cpu"
                               else torch.is_autocast_enabled())
                amp_dtype = (torch.get_autocast_cpu_dtype() if device_type == "cpu"
                             else torch.get_autocast_gpu_dtype())
                # A no_grad leading forward must not cache detached low-precision
                # parameter copies in the caller's autocast region: the main
                # forward would reuse them and lose all parameter gradients.
                # Preserve the caller's precision; disable only leading caching.
                with torch.autocast(device_type=device_type, enabled=amp_enabled,
                                    dtype=amp_dtype, cache_enabled=False), \
                        torch.no_grad(), self._frozen_bn_stats():
                    prediction = self.forward_packet(lead).detach()
                self._time_mode_override = None
                # Only the leading prediction is used.  lead.target is not read.
                interval = (lead.t_end_us - lead.t_start_us).to(torch.float32) * 1e-6
                self.set_time_history(lead.prev_state, interval, current_state=prediction)
                return super()._predict_batch(main)
        # Ordinary random/validation packets do not acquire cross-batch history.
        with self._preserve_history():
            self.reset_time_history()
            return super()._predict_batch(batch)

    def _append_motion_delta(self, module, inputs):
        feature = inputs[0]
        motion = self._motion_delta_context
        if motion is None:
            raise RuntimeError("time-study fc1 used outside a prepared forward context")
        if feature.shape[-1] + 51 != module.in_features or motion.shape != (feature.shape[0], 51):
            raise RuntimeError("common 51D motion input shape mismatch")
        return (torch.cat([feature, motion.to(feature)], -1),) + inputs[1:]

    def _record_time_geometry(self, module, inputs):
        geometry = self._relation_geometry_context
        if geometry is None:
            return None
        # The original encoder next sees residual records.  Consume only its
        # first, matched-record call; residual geometry remains exactly zero.
        self._relation_geometry_context = None
        feature = inputs[0]
        if geometry.shape != (feature.shape[0], 4):
            raise RuntimeError("time geometry and matched-record order differ")
        if not self.xyz.geom:
            return None
        feature = feature.clone()
        feature[:, 7:11] = geometry.to(feature)
        return (feature,) + inputs[1:]

    def _xyz_mesh_forward(self, batch, prev):
        events, ptr = batch.events, batch.ptr
        batch_size = int(ptr.numel() - 1)
        if events.shape[0] == 0:
            self.route_stats = {}
            return prev.new_zeros(batch_size, 51)
        if self.route_prev_override is not None:
            raise ValueError("time study forbids route_prev_override; history and query must share the declared prior")
        velocity = self.prepare_history(prev)
        dt = batch.delta_t_s.to(velocity)
        offset = _batch_scalar(self._query_start_offset_s, velocity, "query_start_offset_s")
        # Target labels are at end_ms; t_end_us is the exclusive end_ms+1.
        horizon = (dt - 0.001 + offset).clamp_min(0.0)
        motion_delta = velocity * horizon[:, None]
        mode = self._time_mode_override or self.time_mode
        betas, camera = self._resolve_betas_K(prev, batch.betas, batch.camera_K)
        src, mask = stride_sample(ptr, self.xyz_max_nodes)
        tok = event_tokens(events, ptr, batch.delta_t_s, self.render_h, self.render_w)
        tok = tok[src.reshape(-1)].view(batch_size, -1, 7) * mask.unsqueeze(-1)
        with torch.no_grad(), torch.autocast(device_type=prev.device.type, enabled=False):
            verts, _ = self._fk(prev.float(), betas.float())
            fx, fy, cx, cy = self._intrinsics(camera.float())
            selected = events[src.reshape(-1)].view(batch_size, -1, events.shape[-1])
            d = torch.stack([(selected[..., EV_X].float() - cx[:, None]) / fx[:, None],
                             (selected[..., EV_Y].float() - cy[:, None]) / fy[:, None],
                             torch.ones_like(selected[..., EV_X], dtype=torch.float32)], -1)
            focal = (fx * fy).sqrt()
            kwargs = dict(band=self.xyz_band_px / focal, front_tol=self.xyz_front_px / focal,
                          z_tol=self.xyz_z_tol, k=self.xyz_k)
            time_bin = (selected[..., EV_T].float() / dt.clamp_min(1e-12)[:, None] * 4).floor().long().clamp(0, 3)
            if isinstance(self.query_bin_shift, bool) or not isinstance(self.query_bin_shift, int):
                raise ValueError("query_bin_shift must be an integer")
            relation_geometry = None
            verts_by_time = None
            if mode == "static" or not bool((velocity != 0).any()):
                assoc = associate_rays(d, mask, verts, self.mano.f, self.xyz.vert_faces, **kwargs)
            else:
                time_bin = (time_bin + self.query_bin_shift) % 4
                states = [extrapolate_state(prev, velocity, offset + center * dt, self.mano.hands_mean)
                          for center in SLICE_CENTERS]
                temporal = torch.stack(states, 1)
                flat_beta = betas[:, None].expand(-1, 4, -1).reshape(-1, betas.shape[-1])
                temporal_verts, _ = self._fk(temporal.reshape(-1, 51), flat_beta.float())
                verts_by_time = temporal_verts.reshape(batch_size, 4, verts.shape[1], 3)
                stationary = (velocity == 0).all(-1)
                verts_by_time = torch.where(stationary[:, None, None, None],
                                            verts[:, None].expand(-1, 4, -1, -1), verts_by_time)
                assoc = associate_time_slices(d, mask, verts_by_time, time_bin,
                                              self.mano.f, self.xyz.vert_faces, **kwargs)
                weights = assoc["q"].unsqueeze(-1) * assoc["A"] * mask.float().unsqueeze(-1)
                if self.ablate_evidence:
                    weights = torch.zeros_like(weights)
                rb, rn, rr = (weights > 0).nonzero(as_tuple=True)
                record_vertex = verts_by_time[rb, time_bin[rb, rn], assoc["vid"][rb, rn, rr]]
                z = assoc["z"][rb, rn]
                dxyz = (z[:, None] * d[rb, rn] - record_vertex) / L_REL
                relation_geometry = torch.cat([dxyz, ((z - Z_REF) / Z_SCALE)[:, None]], -1)
        saved_motion, saved_relation = self._motion_delta_context, self._relation_geometry_context
        self._motion_delta_context, self._relation_geometry_context = motion_delta, relation_geometry
        try:
            delta, stats, aux = self.xyz(tok, d, mask, verts, assoc, prev[:, 3:51], ablate=self.ablate_evidence)
        finally:
            self._motion_delta_context, self._relation_geometry_context = saved_motion, saved_relation
        self.route_stats = stats
        if getattr(self, "xyz_keep_aux", False):
            self.xyz_last = {"assoc": assoc, "d": d, "mask": mask, "verts": verts, "tok": tok,
                             "delta": delta.detach(), "motion_delta": motion_delta.detach(),
                             "physical_velocity": velocity.detach(), "time_bin": time_bin,
                             "query_bin_shift": self.query_bin_shift,
                             "query_start_offset_s": offset, "query_horizon_s": horizon,
                             **{key: value.detach() for key, value in aux.items()}}
            if verts_by_time is not None:
                self.xyz_last["verts_by_time"] = verts_by_time
                self.xyz_last["record_geometry"] = relation_geometry
        return delta


__all__ = ["TimeXYZModel", "TIME_MODES", "SLICE_CENTERS", "physical_motion_delta",
           "extrapolate_state", "associate_time_slices"]
