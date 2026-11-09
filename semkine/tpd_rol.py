r"""TPD stage ROL: closed-loop rollout training of B2 (docs/EXPERIMENT_RECORDS.md#tpd-rol-prereg-20261105; TRAIN.md section 12).

B2 (= PFO2 `fgs`: F + gshare, r_s2d + PFA) is trained one packet at a time from a GT + noise previous state; its loop
error comes from the amplification of its own errors, which no B2-line training ever penalised with gradient through the
previous state. This module trains B2 on K consecutive 50 ms packets of one valid run:

    x_0 = GT(start_0) + B2's mixed prior noise
    for k in 0..K-1:
        y_k = x_k + gate(f(events_k, x_k))      B2's forward; FK, PFA depth and edge geometry differentiable in x_k
        L  += w_k so3_trans_fk(y_k, GT(end_k))
        x_{k+1} = y_k                           (GRAD: true) or y_k.detach() (GRAD: false, DAgger-like exposure only)
    loss = log10(L / sum_k w_k)

Opt-in pieces (nothing here is imported by B2's own path):
  RolloutDataset   B2's training dataset (same augmentation contract) re-indexed by the last packet's end ms; one sample
                   = K packets of PACKET_MS sharing one domain-randomisation realisation (roll / scale / shift / keep
                   fraction), one polarity flip and one per-pixel swap mask; the prior noise is drawn once, for packet 0
  RolloutBatch     K collated `EventPacketBatch`es (`collate_any` keeps plain packets for the validation loader)
  pfa_association_grad / edge_geometry_grad
                   semkine/s45sm.py `pfa_association` and `XYZMeshNet.edge_geometry` with the same arithmetic, the depth
                   of the routed point and the 1-ring edge vectors differentiable in the previous-state vertices; the
                   router's hard (face, barycentric) assignment stays non-differentiable
  rol_model_class  `MNISTModel` subclass: `_xyz_mesh_forward` with the FK outside `no_grad` and optional precomputed
                   routes, the rollout `training_step` (per-packet gradient checkpointing for K > 1), cosine to 0
At evaluation nothing of this is used: a ROL checkpoint is B2's network (same state-dict keys), read by PFO's engine.
The outputs y_k are formed in float32 (B2's `forward_packet` adds delta and prev in the autocast dtype, bf16 in
training); the delta itself equals B2's bit for bit (checked at the first training step of every run).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List

import numpy as np
import torch

from . import domrand as DR
from .dataset import SemKineDataset
from .events import EventPacket, EventPacketBatch, collate_packets

#: per-sample stream salt of the rollout samples (B2's samples use (seed * 1_000_003 + idx) & 0x7FFFFFFF)
ROL_SALT = 20261105
PACKET_MS = 50
#: PFA depth floor, as semkine/s45sm.py
Z_MIN = 0.05


# --------------------------------------------------------------------------- data
@dataclass
class RolloutItem:
    packets: List[EventPacket]


@dataclass
class RolloutBatch:
    packets: List[EventPacketBatch]

    @property
    def batch_size(self) -> int:
        return self.packets[0].batch_size

    def to(self, device) -> "RolloutBatch":
        return RolloutBatch([p.to(device) for p in self.packets])


def collate_any(items):
    """K collated packet batches for rollout items; B2's collate for plain packets (the validation loader)."""
    if isinstance(items[0], RolloutItem):
        k = len(items[0].packets)
        if any(len(it.packets) != k for it in items):
            raise ValueError("rollout items of different lengths in one batch")
        return RolloutBatch([collate_packets([it.packets[i] for it in items]) for i in range(k)])
    return collate_packets(items)


class RolloutDataset(SemKineDataset):
    """B2's training dataset re-indexed for K consecutive packets (built by `upgrade`, never constructed directly)."""

    def _setup_rollout(self, k: int, packet_ms: int = PACKET_MS, salt: int = ROL_SALT):
        if k < 1 or packet_ms < 1:
            raise ValueError("K >= 1 and PACKET_MS >= 1")
        if not self.train or self.input_mode != "raw_packed" or self.xag_source_root or self.prev_lag_ms is not None:
            raise ValueError("rollout samples are built from B2's raw-packet training dataset only")
        self.k, self.packet_ms, self.salt = int(k), int(packet_ms), int(salt)
        span = self.k * self.packet_ms
        runs = np.unique(self.index[:, [0, 2, 3]], axis=0)                     # (sequence, run start, run end)
        rows = [np.stack([np.full(b - (a + span - 1), si), np.arange(a + span - 1, b), np.full(b - (a + span - 1), a),
                          np.full(b - (a + span - 1), b)], 1) for si, a, b in runs.tolist() if b - (a + span - 1) > 0]
        self.base_len = int(len(self.index))
        self.index = np.concatenate(rows).astype(np.int64) if rows else np.zeros((0, 4), np.int64)
        return self

    def ends(self, idx: int):
        si, last, run_a, run_b = (int(v) for v in self.index[idx])
        P, K = self.packet_ms, self.k
        return si, [last - P * (K - 1 - i) for i in range(K)], run_a, run_b

    def __getitem__(self, idx: int) -> RolloutItem:
        rng = np.random.default_rng([self.seed, int(idx), self.salt])
        si, ends, run_a, run_b = self.ends(idx)
        P = self.packet_ms
        h = self._handle(si)
        starts = [e - P + 1 for e in ends]
        camera_K = h.camera_K.copy()
        states = np.stack([self._to_target(h.pos51[starts[0]])] + [self._to_target(h.pos51[e]) for e in ends])
        dr = DR.sample_params(self.domrand, rng, self.height * self.width * 2 * P)
        raw = []
        for e in ends:
            xs, ys, ps, us = self._raw_events(h, e, P)
            if self.domrand.enabled:
                xs, ys, ps, us = self._domrand_events(xs, ys, ps, us, P, dr, camera_K, rng)
            raw.append((xs, ys, ps, us))
        if self.domrand.enabled:
            states, camera_K = DR.transform_labels(states, dr, camera_K, h.j0, self.render_scale)
        prev0 = states[0].copy()
        noise = self._sample_prev_noise(rng)
        if noise is not None:
            prev0 = prev0 + noise
        flip = self.polarity_flip and int(rng.integers(0, 2)) == 1
        swap_mask = None
        if self.pixel_polarity_swap and self.train:
            swap_mask = rng.integers(0, 2, size=(self.height, self.width)).astype(bool)
        packets = []
        for i, (xs, ys, ps, us) in enumerate(raw):
            p_eff = self._polarity(xs, ys, ps, flip, swap_mask)
            packets.append(self._pack(
                si, h, xs, ys, p_eff, us, P, t_start_us=int(starts[i]) * 1000, t_end_us=(int(ends[i]) + 1) * 1000,
                target=states[i + 1], prev=prev0 if i == 0 else states[i], camera_K=camera_K,
                is_start=bool(starts[i] == run_a), is_end=bool(ends[i] == run_b - 1),
                meta={"window_ms": P, "end_idx": int(ends[i]), "seq": h.seq, "category": h.category, "domrand": dr,
                      "rollout_index": int(idx), "rollout_step": i}))
        return RolloutItem(packets)


def upgrade(ds: SemKineDataset, k: int, packet_ms: int = PACKET_MS, salt: int = ROL_SALT) -> RolloutDataset:
    """Turn B2's training dataset (`semkine.dataset.build_dataset(cfg, "train", ...)`) into its rollout form, keeping
    every augmentation attribute of the instance."""
    if type(ds) is not SemKineDataset:
        raise TypeError("upgrade expects semkine.dataset.SemKineDataset")
    ds.__class__ = RolloutDataset
    return ds._setup_rollout(k, packet_ms, salt)


# --------------------------------------------------------------------------- differentiable geometry
def pfa_association_grad(mode, assoc, d, mask, verts, faces, vert_faces, lbs_weights, band, front_tol, z_tol, *,
                         source_part, source_face, source_bary, source_valid, source_depth=None, k=16, chunk=4096):
    """semkine/s45sm.py `pfa_association` (same records, same arithmetic) with `z` differentiable in `verts`."""
    if mode != "face_route":
        raise ValueError("PFA is installed under MODEL.XAG_ASSOC: face_route")
    from .xyz_mesh import R_MAX
    if assoc["vid"].shape != (*mask.shape, R_MAX) or assoc["z"].shape != mask.shape:
        raise ValueError("PFA expects the four-record hit_first association shapes")
    with torch.no_grad():
        known = source_valid & (source_face >= 0) & (source_face < len(faces))
        known = known & torch.isfinite(source_bary).all(-1) & (source_bary >= 0).all(-1) & (source_bary.sum(-1) > 0)
        eligible = known & mask
        out = {name: torch.zeros_like(assoc[name]) for name in ("vid", "A", "q", "valid", "hit")}
        b, n = eligible.nonzero(as_tuple=True)
        fv = faces.long()[source_face[b, n]]
        bary = source_bary[b, n].float()
        bary = bary / bary.sum(-1, keepdim=True)
        out["vid"][b, n] = torch.cat([fv, fv[:, :1]], -1)
        out["A"][b, n] = torch.cat([bary, torch.zeros_like(bary[:, :1])], -1).to(out["A"].dtype)
        out["q"][b, n] = 1
        out["valid"][b, n] = True
        out["hit"][b, n] = True
    point = (bary.unsqueeze(-1) * verts.float()[b.unsqueeze(1), fv]).sum(1)
    z0 = torch.zeros_like(assoc["z"])
    out["z"] = z0.index_put((b, n), point[:, 2].clamp_min(Z_MIN).to(z0.dtype))
    return out, {"xag_known": known & mask, "xag_eligible": eligible, "xag_replaced": eligible,
                 "xag_fallback": mask & ~eligible, "xag_recalled": eligible, "xag_none": mask & ~eligible}


def edge_geometry_grad(self, verts):
    """`XYZMeshNet.edge_geometry` without `no_grad`: `(B, V, K, 4)` = [(V_l - V_j) / L_EDGE, |V_l - V_j| / L_EDGE]."""
    from .xyz_mesh import L_EDGE
    B, V, _ = verts.shape
    K = self.ring_idx.shape[1]
    vf = verts.float()
    dv = vf[:, self.ring_idx.reshape(-1)].view(B, V, K, 3) - vf.unsqueeze(2)
    g = torch.cat([dv, dv.norm(dim=-1, keepdim=True)], dim=-1) / L_EDGE
    return g * self.ring_mask.view(1, V, K, 1)


_GEOM = {"saved": None}


def install_grad_geometry():
    """Patch the native `XYZMeshNet.edge_geometry` (the class PFOMeshNet4 calls explicitly) with the differentiable copy."""
    from . import pfo_trunk4 as T4
    cls = T4._NATIVE["cls"]
    if _GEOM["saved"] is not None:
        raise RuntimeError("grad geometry already installed")
    _GEOM["saved"] = cls.__dict__["edge_geometry"]
    cls.edge_geometry = edge_geometry_grad
    return "edge_geometry_grad"


def uninstall_grad_geometry():
    from . import pfo_trunk4 as T4
    if _GEOM["saved"] is not None:
        T4._NATIVE["cls"].edge_geometry = _GEOM["saved"]
        _GEOM["saved"] = None


# --------------------------------------------------------------------------- the model
_CLASS = {}


def rol_model_class():
    """`ROLModel(MNISTModel)`, built on first use (model/ must be importable as `model`)."""
    if "cls" in _CLASS:
        return _CLASS["cls"]
    from model import MNISTModel

    class ROLModel(MNISTModel):
        LR_FLOOR = 0.0                                      # cosine to zero (#tpd-rol-prereg-20261105)

        def __init__(self, cfg=None, num_classes=None):
            super().__init__(cfg, num_classes)
            r = self.cfg["ROL"]
            self.rol_k = int(r["K"])
            self.rol_grad = bool(r["GRAD"])
            self.rol_ckpt = bool(r["CHECKPOINT"])
            self.rol_w = [float(w) for w in r["WEIGHTS"]]
            if len(self.rol_w) != self.rol_k or min(self.rol_w) <= 0:
                raise ValueError("ROL.WEIGHTS: K positive weights")
            if not self.xyz_mesh or self.xag_assoc_mode != "face_route" or self.loss_type != "so3_trans_fk":
                raise ValueError("ROL trains B2's network (xyz_mesh, PFA under face_route, so3_trans_fk)")
            self._rol_checked = False

        # ---------------------------------------------------------------- forward
        def _xyz_mesh_forward(self, batch, prev, sources=None):
            """B2's forward (model/model.py `_xyz_mesh_forward` with PFA installed), differentiable in `prev`."""
            from .encoder import event_tokens
            from .events import EV_X, EV_Y
            from . import xag as XAG
            from .xyz_mesh import associate_rays, stride_sample
            events, ptr = batch.events, batch.ptr
            if events.shape[0] == 0:
                return MNISTModel._xyz_mesh_forward(self, batch, prev)
            B = int(ptr.numel() - 1)
            betas_f, k_f = self._resolve_betas_K(prev, batch.betas, batch.camera_K)
            src, mask = stride_sample(ptr, self.xyz_max_nodes)
            tok = event_tokens(events, ptr, batch.delta_t_s, self.render_h, self.render_w)
            tok = tok[src.reshape(-1)].view(B, -1, tok.shape[-1]) * mask.unsqueeze(-1)
            verts, _ = self._fk(prev.float(), betas_f)
            with torch.no_grad():
                fx, fy, cx, cy = self._intrinsics(k_f)
                ev = events[src.reshape(-1)].view(B, -1, events.shape[1])
                px, py = ev[..., EV_X].float(), ev[..., EV_Y].float()
                d = torch.stack([(px - cx[:, None]) / fx[:, None], (py - cy[:, None]) / fy[:, None],
                                 torch.ones_like(px)], dim=-1)
                f_px = (fx * fy).sqrt()
                assoc = associate_rays(d, mask, verts.detach(), self.mano.f, self.xyz.vert_faces,
                                       band=self.xyz_band_px / f_px, front_tol=self.xyz_front_px / f_px,
                                       z_tol=self.xyz_z_tol, k=self.xyz_k)
                if sources is None:
                    sources = XAG.sample_sources(batch, src, mask)
            assoc, _ = pfa_association_grad(
                self.xag_assoc_mode, assoc, d, mask, verts, self.mano.f, self.xyz.vert_faces, self.xyz.lbs_w,
                self.xyz_band_px / f_px, self.xyz_front_px / f_px, self.xyz_z_tol, source_part=sources["xag_part"],
                source_face=sources["xag_face"], source_bary=sources["xag_bary"], source_valid=sources["xag_valid"],
                k=self.xyz_k)
            delta, stats, _ = self.xyz(tok, d, mask, verts, assoc, prev[:, 3:51], ablate=self.ablate_evidence,
                                       capture=False)
            self.route_stats = stats
            return delta

        def _rol_step(self, batch, prev, sources):
            delta = self._xyz_mesh_forward(batch, prev, sources)
            if self.zero_event_gate:
                empty = (batch.counts <= 0).unsqueeze(-1)
                delta = torch.where(empty, torch.zeros_like(delta), delta)
            return prev.float() + delta.float()

        @staticmethod
        def _routes(batch, max_nodes):
            from . import xag as XAG
            from .xyz_mesh import stride_sample
            src, mask = stride_sample(batch.ptr, max_nodes)
            with torch.no_grad():
                return XAG.sample_sources(batch, src, mask)

        # ---------------------------------------------------------------- training
        def training_step(self, batch, batch_nb):
            if not isinstance(batch, RolloutBatch):
                raise TypeError("ROL trains on RolloutBatch")
            packets = batch.packets
            if len(packets) != self.rol_k:
                raise ValueError("rollout length differs from ROL.K")
            if not self._rol_checked:
                self._start_check(packets[0])
            B = packets[0].batch_size
            x = packets[0].prev_state
            total = 0.0
            for k, pk in enumerate(packets):
                sources = self._routes(pk, self.xyz_max_nodes)
                if self.rol_ckpt:
                    from torch.utils.checkpoint import checkpoint
                    y = checkpoint(self._rol_step, pk, x, sources, use_reentrant=False)
                else:
                    y = self._rol_step(pk, x, sources)
                loss_k, parts = self._compute_loss(y, pk.target, pk.betas)
                total = total + self.rol_w[k] * loss_k
                self.log(f"train_loss_k{k}", loss_k, sync_dist=True, batch_size=B)
                self.log(f"train_joint_error_mm_k{k}", parts["joint_error_mm"], sync_dist=True, batch_size=B)
                self.log(f"train_rotation_error_deg_k{k}", parts["rotation_error_deg"], sync_dist=True, batch_size=B)
                if k == 0:
                    for key in ("loss_rot", "loss_trans", "loss_fk", "rotation_error_deg", "joint_error_mm"):
                        self.log(f"train_{key}", parts[key], sync_dist=True, batch_size=B)
                x = y if self.rol_grad else y.detach()
            loss = total / sum(self.rol_w)
            self.log("train_loss", loss, prog_bar=True, sync_dist=True, batch_size=B)
            self._log_route_stats("train")
            return loss.float().log10()

        def _start_check(self, pk):
            """First step of every run: the weights equal the INIT_FROM checkpoint, and this forward's delta equals B2's
            (model/model.py `_xyz_mesh_forward`) bit for bit on the first packet batch, under the training autocast."""
            self._rol_checked = True
            with torch.no_grad():
                sources = self._routes(pk, self.xyz_max_nodes)
                mine = self._xyz_mesh_forward(pk, pk.prev_state, sources)
                base = MNISTModel._xyz_mesh_forward(self, pk, pk.prev_state)
            # autocast caches the low-precision casts of the weights for the whole autocast region, also when they are made
            # under no_grad: without clearing, the training forward that follows would reuse these grad-less casts
            torch.clear_autocast_cache()
            rec = {"global_step": int(self.global_step), "rank": int(self.global_rank),
                   "delta_bitwise_equal": bool(torch.equal(mine, base)),
                   "delta_max_abs_diff": float((mine.float() - base.float()).abs().max()),
                   "delta_dtype": str(mine.dtype),
                   "autocast": bool(torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()),
                   "samples": int(pk.batch_size), "events": int(pk.events.shape[0])}
            init = self.cfg["MODEL"].get("INIT_FROM")
            if init:
                sd = torch.load(init, map_location="cpu")["state_dict"]
                own = self.state_dict()
                rec["init_keys"] = len(sd)
                rec["init_equal"] = bool(set(sd) == set(own) and all(torch.equal(own[k].detach().cpu(), sd[k]) for k in sd))
            out = Path(self.cfg["TRAIN"]["OUTPUT_DIR"]) / f"rol_start_check_rank{self.global_rank}.json"
            out.write_text(json.dumps(rec, indent=2) + "\n")
            if not rec["delta_bitwise_equal"] or (init and not rec["init_equal"]):
                raise RuntimeError("ROL start check failed: " + json.dumps(rec))

    _CLASS["cls"] = ROLModel
    return ROLModel


__all__ = ["PACKET_MS", "ROL_SALT", "RolloutBatch", "RolloutDataset", "RolloutItem", "collate_any",
           "edge_geometry_grad", "install_grad_geometry", "pfa_association_grad", "rol_model_class",
           "uninstall_grad_geometry", "upgrade"]
