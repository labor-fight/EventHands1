r"""ACU line B, B1 consistency training (docs/EXPERIMENT_RECORDS.md#acu-s1b-prereg-20261104).

B2 always trains with the prior at the window start (GT(start) + the noise contract) and the target at the window end,
so its update can learn "apply the motion this packet shows" instead of "move by what the evidence says the prior is
off by". B1 moves the prior's time inside the evidence window and tells the trunk where it is:

  prior time   t_p = start (p = 1/2, B2's sample), uniform in (start, end) (p = 1/4), or end (p = 1/4: the prior is the
               target itself up to the noise contract; nothing has to move but the noise), drawn from a separate
               per-sample stream default_rng([seed, idx, TAG_B1]); B2's own stream (window, events, augmentation, noise,
               polarity) is never advanced differently, so every other field of the sample is B2's
  prior        GT(t_p) through the same domrand label realisation and the same noise draw as B2's GT(start)
  trunk tokens the time column of the trunk's event tokens is measured from t_p: (t - (t_p - start)) / dt, negative for
               events before the prior (`semkine.encoder.event_tokens` is wrapped; routers keep their own tokens)
  age          the readout adds age * w_age to fc1's output (before its ReLU), age = (end - t_p) / 50 ms, w_age a new
               zero-initialised vector built in an isolated RNG scope: the initial function and every other initial
               value equal B2's

At evaluation (prior = the previous output at the window start) the offset is 0 and the age is the window length, so the
trunk sees B2's inputs plus the age term. Opt-in: `install_dataset()`, `install_collate(train_module)`,
`install_model(MNISTModel)`, `install_trunk(trunk_cls)`; `uninstall()` restores everything.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn

from . import dataset as D
from . import domrand as DR
from . import encoder as ENC
from .events import EventPacket, EventPacketBatch, collate_packets

TAG_B1 = 0xB1B1
P_START, P_UNIFORM = 0.5, 0.25          # p(prior at the window start), p(uniform inside); the rest: prior at the end
AGE_UNIT_S = 0.05
_ORIGINAL_GETITEM = D.SemKineDataset.__getitem__
_STATE = {"dataset": False, "collate": None, "tokens": None, "model": None, "offset": None, "age": None, "trunk": None}


# --------------------------------------------------------------------------- data
def draw_prior_ms(seed: int, idx: int, start: int, end: int):
    """(prior_ms, branch) from the B1 stream: branch 0 = start, 1 = uniform in (start, end), 2 = end."""
    rng = np.random.default_rng([int(seed), int(idx), TAG_B1])
    u = float(rng.random())
    if u < P_START or end <= start:
        return int(start), 0
    if u < P_START + P_UNIFORM and end - start >= 2:
        return int(rng.integers(start + 1, end)), 1
    return int(end), 2


def _getitem(self, idx: int):
    if not self.train:
        return _ORIGINAL_GETITEM(self, idx)
    if (self.input_mode != "raw_packed" or self._fast_on or self.prev_lag_ms is not None or self.unroll_pair
            or self.clip_k or self.triplet or self.fixed_window is not None):
        raise RuntimeError("B1 is defined on the raw-packet training path only")
    # ---- SemKineDataset.__getitem__ (semkine/dataset.py), raw-packet training path, verbatim except the B1 lines
    rng = np.random.default_rng((self.seed * 1_000_003 + idx) & 0x7FFFFFFF)
    si, end, window, run_a = self._window(idx, rng)
    h = self._handle(si)
    start = end - window + 1

    xs, ys, ps, us = self._raw_events(h, end, window)
    source_rows = np.arange(int(h.offsets[start]), int(h.offsets[end + 1]), dtype=np.int64) \
        if self.xag_source_root else None
    target = self._to_target(h.pos51[end])
    prior_ms, branch = draw_prior_ms(self.seed, idx, int(start), int(end))          # B1
    prev = self._to_target(h.pos51[prior_ms])                                       # B1: GT(t_p) for GT(start)
    camera_K = h.camera_K.copy()

    n_slots = self.height * self.width * 2 * window
    dr = DR.sample_params(self.domrand, rng, n_slots)
    if self.domrand.enabled:
        if source_rows is None:
            xs, ys, ps, us = self._domrand_events(xs, ys, ps, us, window, dr, camera_K, rng)
        else:
            xs, ys, ps, us, source_rows = self._domrand_events(
                xs, ys, ps, us, window, dr, camera_K, rng, source_rows=source_rows)
        stacked, camera_K = DR.transform_labels(
            np.stack([prev, target]), dr, camera_K, h.j0, self.render_scale
        )
        prev, target = stacked[0], stacked[1]

    noise = self._sample_prev_noise(rng)
    if noise is not None:
        prev = prev + noise

    flip = self.polarity_flip and int(rng.integers(0, 2)) == 1
    swap_mask = None
    if self.pixel_polarity_swap and self.train:
        swap_mask = rng.integers(0, 2, size=(self.height, self.width)).astype(bool)
    p_eff = self._polarity(xs, ys, ps, flip, swap_mask)

    packet = self._pack(
        si, h, xs, ys, p_eff, us, window,
        t_start_us=int(start) * 1000, t_end_us=(int(end) + 1) * 1000,
        target=target, prev=prev, camera_K=camera_K,
        is_start=bool(start == int(self.index[idx][2])),
        is_end=bool(end == int(self.index[idx][3]) - 1),
        meta={"window_ms": window, "end_idx": end, "seq": h.seq,
              "category": h.category, "domrand": dr},
        source_rows=source_rows,
    )
    packet.meta["acu_prior_offset_s"] = float(prior_ms - start) * 1e-3                 # B1
    packet.meta["acu_prior_branch"] = int(branch)
    return packet


@dataclass
class B1Batch(EventPacketBatch):
    """EventPacketBatch plus the per-sample prior time measured from the window start (s)."""

    prior_offset_s: Optional[torch.Tensor] = None

    def to(self, device) -> "B1Batch":
        def mv(t):
            return t.to(device, non_blocking=True) if torch.is_tensor(t) else t
        return B1Batch(**{k: mv(v) for k, v in self.__dict__.items()})


def collate_b1(items):
    base = collate_packets(items)
    off = torch.tensor([float(it.meta.get("acu_prior_offset_s", 0.0)) for it in items], dtype=torch.float32)
    return B1Batch(**base.__dict__, prior_offset_s=off)


# --------------------------------------------------------------------------- model
def _tokens(events, ptr, delta_t_s, height, width, eps=ENC.EPS_DT):
    tok = _STATE["tokens"](events, ptr, delta_t_s, height, width, eps)
    off = _STATE["offset"]
    if off is None or tok.shape[0] == 0 or not bool((off != 0).any()):
        return tok
    b = events[:, 0].long()
    dt = delta_t_s.clamp_min(ENC.EPS_DT)[b]
    tok = tok.clone()
    tok[:, 3] = tok[:, 3] - off.to(tok.dtype)[b] / dt.to(tok.dtype)
    return tok


def install_dataset():
    D.SemKineDataset.__getitem__ = _getitem
    _STATE["dataset"] = True


def install_collate(train_module):
    """semkine/train.py's loader collate calls the module-global `collate_packets` at call time."""
    _STATE["collate"] = (train_module, train_module.collate_packets)
    train_module.collate_packets = collate_b1


def install_model(model_cls):
    """forward_packet records the batch's prior offset and age; the trunk's tokens and fc1 read them."""
    native = model_cls.forward_packet

    def forward_packet(self, batch):
        B = int(batch.ptr.numel() - 1)
        off = getattr(batch, "prior_offset_s", None)
        if off is None:
            off = torch.zeros(B, device=batch.delta_t_s.device, dtype=torch.float32)
        off = off.to(batch.delta_t_s.device).float()
        _STATE["offset"] = off
        _STATE["age"] = ((batch.delta_t_s.float() - off) / AGE_UNIT_S).clamp_min(0.0)
        try:
            return native(self, batch)
        finally:
            _STATE["offset"], _STATE["age"] = None, None
    _STATE["model"] = (model_cls, native)
    model_cls.forward_packet = forward_packet
    _STATE["tokens"] = ENC.event_tokens
    ENC.event_tokens = _tokens


class AgeTerm(nn.Module):
    """fc1's output + age * w (w zero-initialised)."""

    def __init__(self, width):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(width))

    def hook(self, module, inputs, output):
        age = _STATE["age"]
        if age is None:
            raise RuntimeError("B1 trunk called outside forward_packet")
        return output + age.to(output.dtype).unsqueeze(-1) * self.w.to(output.dtype)


def b1_trunk(trunk_cls):
    """A subclass of `trunk_cls` whose readout adds the zero-initialised age term (isolated RNG scope)."""

    class B1Trunk(trunk_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            with torch.random.fork_rng(devices=[]):
                self.age = AgeTerm(self.fc1.out_features)
            self.fc1.register_forward_hook(self.age.hook)
    B1Trunk.__name__ = f"B1_{trunk_cls.__name__}"
    return B1Trunk


def install_trunk(xm_module):
    """Wrap the installed trunk class (semkine.xyz_mesh.XYZMeshNet, already a B2 / CA variant) with the age term."""
    _STATE["trunk"] = (xm_module, xm_module.XYZMeshNet)
    xm_module.XYZMeshNet = b1_trunk(xm_module.XYZMeshNet)
    return xm_module.XYZMeshNet.__name__


def uninstall():
    D.SemKineDataset.__getitem__ = _ORIGINAL_GETITEM
    if _STATE["collate"] is not None:
        mod, native = _STATE["collate"]
        mod.collate_packets = native
    if _STATE["model"] is not None:
        cls, native = _STATE["model"]
        cls.forward_packet = native
        ENC.event_tokens = _STATE["tokens"]
    if _STATE["trunk"] is not None:
        xm, native = _STATE["trunk"]
        xm.XYZMeshNet = native
    _STATE.update(dataset=False, collate=None, tokens=None, model=None, offset=None, age=None, trunk=None)


__all__ = ["AGE_UNIT_S", "AgeTerm", "B1Batch", "P_START", "P_UNIFORM", "TAG_B1", "b1_trunk", "collate_b1",
           "draw_prior_ms", "install_collate", "install_dataset", "install_model", "install_trunk", "uninstall"]
