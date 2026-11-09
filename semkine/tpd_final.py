r"""TPD final configuration pieces: training in the causal (asynchronous) deployment form (docs/EXPERIMENT_RECORDS.md
#tpd-causal-prereg-20261105 decided TRAIN_CAUSAL; the final stage's own registration names the rest).

The closed-loop trunk takes its nodes by the causal stride and its tokens with the causal "never fired" fill
(semkine/tpd_causal.py); training must see the same inputs, so every training packet carries `n_prev`, the number of
events the sensor delivers (under the sample's augmentation) in the interval of the packet's length just before it:
  rollout samples (semkine/tpd_rol.py, K packets of 50 ms)  packet k >= 1: packet k-1's own (augmented) event count;
                                                            packet 0: the 50 ms before it, under the same geometric
                                                            map, its own keep draw and hot pixels at the dataset's rate
  plain samples (B2's variable 30-300 ms windows, K = 1)    the window-length interval before the window, likewise
  evaluation packets (fixed 50 ms, no augmentation)         the raw count of the previous 50 ms
(clipped at the recording start). Rollout samples may carry the XAG pseudo-sources (label_part training routes, IDP L1's
recipe); their K packets are otherwise ROL's exactly (same draws in the same order; the counts use a separate stream).

`final_model_class()` is semkine/tpd_rol.py's ROLModel with the trunk's nodes and tokens chosen by FINAL.CAUSAL, the
routes taken at the trunk's nodes (`semkine.xag.sample_sources`: the label_part mix of tools/pfo/nets.py or
semkine/s45sm_fast.py's deployed routes, both evaluated at the given nodes), the rollout step on RolloutBatch and B2's
single-packet step on plain batches, and B2's learning-rate floor (cosine to 0.02 of the peak).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import domrand as DR
from . import tpd_causal as TC
from . import tpd_rol as R
from .dataset import SemKineDataset
from .events import EventPacketBatch, collate_packets

FINAL_SALT = 61105


# --------------------------------------------------------------------------- batches
@dataclass
class CausalPacketBatch(EventPacketBatch):
    n_prev: Optional[torch.Tensor] = None          # (B,) int64

    def to(self, device) -> "CausalPacketBatch":
        def mv(t):
            return t.to(device, non_blocking=True) if torch.is_tensor(t) else t
        return CausalPacketBatch(**{k: mv(v) for k, v in self.__dict__.items()})


def causal_batch(items):
    b = collate_packets(items)
    return CausalPacketBatch(**b.__dict__, n_prev=torch.tensor([int(it.meta["n_prev"]) for it in items], dtype=torch.int64))


def collate_final(items):
    if isinstance(items[0], R.RolloutItem):
        k = len(items[0].packets)
        if any(len(it.packets) != k for it in items):
            raise ValueError("rollout items of different lengths in one batch")
        return R.RolloutBatch([causal_batch([it.packets[i] for it in items]) for i in range(k)])
    return causal_batch(items)


# --------------------------------------------------------------------------- counts
def augmented_count(ds, h, a, b, dr, camera_K, rng):
    """Events of [a, b] ms (a clipped at 0) the sensor delivers under the sample's geometric map, with their own keep draw
    and hot pixels at the dataset's rate; the raw count without augmentation."""
    a = max(int(a), 0)
    b = int(b)
    if b < a:
        return 0
    if not ds.domrand.enabled:
        return int(h.offsets[b + 1]) - int(h.offsets[a])
    xs, ys, _, _ = ds._raw_events(h, b, b - a + 1)
    n = 0
    if len(xs):
        keep = DR.transform_events(xs.copy(), ys.copy(), dr, camera_K, ds.render_scale, ds.width, ds.height)
        m = DR.keep_mask(len(xs), dr.keep, rng)
        if m is not None:
            keep = keep & m
        n = int(keep.sum())
    if ds.domrand.hot_pixel_rate > 0:
        n += int(rng.poisson(ds.domrand.hot_pixel_rate * ds.height * ds.width * 2 * (b - a + 1)))
    return n


class FinalRolloutDataset(R.RolloutDataset):
    """ROL's rollout samples with `n_prev` per packet and, when the dataset reads them, the XAG pseudo-sources."""

    def _setup_rollout(self, k, packet_ms=R.PACKET_MS, salt=R.ROL_SALT):
        xag, self.xag_source_root = self.xag_source_root, None          # ROL's check refuses sources; restore after
        try:
            return super()._setup_rollout(k, packet_ms, salt)
        finally:
            self.xag_source_root = xag

    def __getitem__(self, idx: int):
        rng = np.random.default_rng([self.seed, int(idx), self.salt])
        si, ends, run_a, run_b = self.ends(idx)
        P = self.packet_ms
        h = self._handle(si)
        starts = [e - P + 1 for e in ends]
        camera_K = h.camera_K.copy()
        states = np.stack([self._to_target(h.pos51[starts[0]])] + [self._to_target(h.pos51[e]) for e in ends])
        dr = DR.sample_params(self.domrand, rng, self.height * self.width * 2 * P)
        raw = []
        for s, e in zip(starts, ends):
            xs, ys, ps, us = self._raw_events(h, e, P)
            rows = np.arange(int(h.offsets[s]), int(h.offsets[e + 1]), dtype=np.int64) if self.xag_source_root else None
            if self.domrand.enabled:
                if rows is None:
                    xs, ys, ps, us = self._domrand_events(xs, ys, ps, us, P, dr, camera_K, rng)
                else:
                    xs, ys, ps, us, rows = self._domrand_events(xs, ys, ps, us, P, dr, camera_K, rng, source_rows=rows)
            raw.append((xs, ys, ps, us, rows))
        if self.domrand.enabled:
            states, camera_K_new = DR.transform_labels(states, dr, camera_K, h.j0, self.render_scale)
        else:
            camera_K_new = camera_K
        prev0 = states[0].copy()
        noise = self._sample_prev_noise(rng)
        if noise is not None:
            prev0 = prev0 + noise
        flip = self.polarity_flip and int(rng.integers(0, 2)) == 1
        swap_mask = None
        if self.pixel_polarity_swap and self.train:
            swap_mask = rng.integers(0, 2, size=(self.height, self.width)).astype(bool)
        crng = np.random.default_rng([self.seed, int(idx), FINAL_SALT])
        n0 = augmented_count(self, h, starts[0] - P, starts[0] - 1, dr, camera_K, crng)
        packets = []
        for i, (xs, ys, ps, us, rows) in enumerate(raw):
            p_eff = self._polarity(xs, ys, ps, flip, swap_mask)
            packets.append(self._pack(
                si, h, xs, ys, p_eff, us, P, t_start_us=int(starts[i]) * 1000, t_end_us=(int(ends[i]) + 1) * 1000,
                target=states[i + 1], prev=prev0 if i == 0 else states[i], camera_K=camera_K_new,
                is_start=bool(starts[i] == run_a), is_end=bool(ends[i] == run_b - 1),
                meta={"window_ms": P, "end_idx": int(ends[i]), "seq": h.seq, "category": h.category, "domrand": dr,
                      "rollout_index": int(idx), "rollout_step": i, "n_prev": n0 if i == 0 else len(raw[i - 1][0])},
                source_rows=rows))
        return R.RolloutItem(packets)


class FinalPlainDataset(SemKineDataset):
    """B2's samples (training: variable windows; evaluation: fixed windows, no augmentation) with `n_prev`."""

    def __getitem__(self, idx: int):
        packet = SemKineDataset.__getitem__(self, idx)
        if not isinstance(packet, R.EventPacket):
            raise RuntimeError("the final configuration reads raw packets")
        si = int(self.index[idx][0])
        h = self._handle(si)
        end, window = int(packet.meta["end_idx"]), int(packet.meta["window_ms"])
        start = end - window + 1
        crng = np.random.default_rng([self.seed, int(idx), FINAL_SALT])
        packet.meta["n_prev"] = augmented_count(self, h, start - window, start - 1, packet.meta["domrand"], h.camera_K, crng)
        return packet


def upgrade_rollout(ds, k, packet_ms=R.PACKET_MS):
    if type(ds) is not SemKineDataset or not ds.train or ds.input_mode != "raw_packed" or ds.prev_lag_ms is not None:
        raise TypeError("upgrade_rollout expects B2's raw-packet training dataset")
    ds.__class__ = FinalRolloutDataset
    return ds._setup_rollout(k, packet_ms)


def upgrade_plain(ds):
    if type(ds) is not SemKineDataset or ds.input_mode != "raw_packed":
        raise TypeError("upgrade_plain expects a raw-packet SemKineDataset")
    ds.__class__ = FinalPlainDataset
    return ds


# --------------------------------------------------------------------------- the model
_CLASS = {}


def final_model_class():
    if "cls" in _CLASS:
        return _CLASS["cls"]
    from model import MNISTModel
    base = R.rol_model_class()

    class FinalModel(base):
        LR_FLOOR = MNISTModel.LR_FLOOR                     # B2's recipe: cosine to 0.02 of the peak

        def __init__(self, cfg=None, num_classes=None):
            super().__init__(cfg, num_classes)
            self.causal = bool(self.cfg["FINAL"]["CAUSAL"])

        def _nodes(self, batch):
            if self.causal:
                n_prev = getattr(batch, "n_prev", None)
                if n_prev is None:
                    raise RuntimeError("causal nodes need the batch's n_prev")
                return TC.causal_stride(batch.ptr, TC.k_from_prev(n_prev).to(batch.ptr.device), cap=self.xyz_max_nodes)
            from .xyz_mesh import stride_sample
            return stride_sample(batch.ptr, self.xyz_max_nodes)

        def _tokens(self, batch):
            fn = TC.event_tokens_causal if self.causal else TC._STATE["tokens"]       # B2's tokens: the native builder
            return fn(batch.events, batch.ptr, batch.delta_t_s, self.render_h, self.render_w)

        def _routes(self, batch, max_nodes=None):
            from . import xag as XAG
            nodes = self._nodes(batch)
            with torch.no_grad():
                return XAG.sample_sources(batch, *nodes), nodes

        def _xyz_mesh_forward(self, batch, prev, sources=None, nodes=None):
            from .events import EV_X, EV_Y
            from . import xag as XAG
            from .xyz_mesh import associate_rays
            events, ptr = batch.events, batch.ptr
            if events.shape[0] == 0:
                return MNISTModel._xyz_mesh_forward(self, batch, prev)
            B = int(ptr.numel() - 1)
            betas_f, k_f = self._resolve_betas_K(prev, batch.betas, batch.camera_K)
            src, mask = nodes if nodes is not None else self._nodes(batch)
            tok = self._tokens(batch)
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
            assoc, _ = R.pfa_association_grad(
                self.xag_assoc_mode, assoc, d, mask, verts, self.mano.f, self.xyz.vert_faces, self.xyz.lbs_w,
                self.xyz_band_px / f_px, self.xyz_front_px / f_px, self.xyz_z_tol, source_part=sources["xag_part"],
                source_face=sources["xag_face"], source_bary=sources["xag_bary"], source_valid=sources["xag_valid"],
                k=self.xyz_k)
            delta, stats, _ = self.xyz(tok, d, mask, verts, assoc, prev[:, 3:51], ablate=self.ablate_evidence,
                                       capture=False)
            self.route_stats = stats
            return delta

        def _final_step(self, batch, prev, sources, nodes):
            delta = self._xyz_mesh_forward(batch, prev, sources, nodes)
            if self.zero_event_gate:
                empty = (batch.counts <= 0).unsqueeze(-1)
                delta = torch.where(empty, torch.zeros_like(delta), delta)
            return prev.float() + delta.float()

        def training_step(self, batch, batch_nb):
            if not isinstance(batch, R.RolloutBatch):
                return MNISTModel.training_step(self, batch, batch_nb)
            packets = batch.packets
            if len(packets) != self.rol_k:
                raise ValueError("rollout length differs from ROL.K")
            if not self._rol_checked:
                self._start_check(packets[0])
            B = packets[0].batch_size
            x = packets[0].prev_state
            total = 0.0
            for k, pk in enumerate(packets):
                sources, nodes = self._routes(pk)
                if self.rol_ckpt:
                    from torch.utils.checkpoint import checkpoint
                    y = checkpoint(self._final_step, pk, x, sources, nodes, use_reentrant=False)
                else:
                    y = self._final_step(pk, x, sources, nodes)
                loss_k, parts = self._compute_loss(y, pk.target, pk.betas)
                total = total + self.rol_w[k] * loss_k
                self.log(f"train_loss_k{k}", loss_k, sync_dist=True, batch_size=B)
                self.log(f"train_joint_error_mm_k{k}", parts["joint_error_mm"], sync_dist=True, batch_size=B)
                if k == 0:
                    for key in ("loss_rot", "loss_trans", "loss_fk", "rotation_error_deg", "joint_error_mm"):
                        self.log(f"train_{key}", parts[key], sync_dist=True, batch_size=B)
                x = y if self.rol_grad else y.detach()
            loss = total / sum(self.rol_w)
            self.log("train_loss", loss, prog_bar=True, sync_dist=True, batch_size=B)
            self._log_route_stats("train")
            return loss.float().log10()

        def _start_check(self, pk):
            """First rollout step: the INIT_FROM weights when there are any, and (non-causal form only) the delta equals
            B2's forward bit for bit; the causal form records its node and token settings."""
            self._rol_checked = True
            rec = {"global_step": int(self.global_step), "rank": int(self.global_rank), "causal": self.causal,
                   "autocast": bool(torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled())}
            if not self.causal:
                with torch.no_grad():
                    sources, nodes = self._routes(pk)
                    mine = self._xyz_mesh_forward(pk, pk.prev_state, sources, nodes)
                    ref = MNISTModel._xyz_mesh_forward(self, pk, pk.prev_state)
                torch.clear_autocast_cache()
                rec["delta_bitwise_equal"] = bool(torch.equal(mine, ref))
            else:
                n_prev = pk.n_prev
                rec.update(k_mean=float(TC.k_from_prev(n_prev).float().mean()), n_prev_mean=float(n_prev.float().mean()))
            init = self.cfg["MODEL"].get("INIT_FROM")
            if init:
                sd = torch.load(init, map_location="cpu")["state_dict"]
                own = self.state_dict()
                rec["init_equal"] = bool(set(sd) == set(own) and all(torch.equal(own[k].detach().cpu(), sd[k]) for k in sd))
            out = Path(self.cfg["TRAIN"]["OUTPUT_DIR"]) / f"final_start_check_rank{self.global_rank}.json"
            out.write_text(json.dumps(rec, indent=2) + "\n")
            if not rec.get("delta_bitwise_equal", True) or not rec.get("init_equal", True):
                raise RuntimeError("final start check failed: " + json.dumps(rec))

    _CLASS["cls"] = FinalModel
    return FinalModel


__all__ = ["CausalPacketBatch", "FINAL_SALT", "FinalPlainDataset", "FinalRolloutDataset", "augmented_count",
           "causal_batch", "collate_final", "final_model_class", "upgrade_plain", "upgrade_rollout"]
