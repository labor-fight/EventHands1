"""MV2 M2 (RGQ factor arm): ray-gauge quotient readout on top of the surface-lifted readout (docs/EXPERIMENT_RECORDS.md
#mv2-prereg-20261026, M2 GPU arm (b); configuration registered as an amendment after PA-RGQ = weak, before training).

`RGQModel` is `SLRModel` plus one opt-in key, `MODEL.RGQ: true`. Without it the class is `SLRModel` exactly (and so
`MNISTModel` exactly without `MODEL.SLR`), so every earlier checkpoint loads and evaluates through it bit for bit.

With `MODEL.RGQ: true` (requires `MODEL.SLR: true`):

  1. Gauge-invariant block / t_z head. The camera-z translation increment is read only by a separate small-gain head
     `rgq_tz` (a zero-initialised Linear on the trunk's global-average-pool features, learning rate x `RGQ_TZ_LR_MULT`).
     The t_z column of the trunk logits (`rn.fc`) + surface-lifted correction is replaced by it, and the t_z column of
     the event-blind previous-state MLP (`prev_mlp`) is zeroed by a forward hook. Image-plane x / y, the root rotation and
     the fingers keep S39 + SLR's readout unchanged.
  2. Previous-state gauge perturbation (training only). Each training sample's previous state is moved rigidly along
     its wrist ray so that the wrist sits at w / sigma, sigma ~ U[RGQ_SIGMA_LO, RGQ_SIGMA_HI] (the recording-level
     event-to-label scale range of Stage 0, 0.93 .. 1.01; image size about x sigma around the wrist); the label is
     unchanged. Only the translation changes: t' = t + w (1 / sigma - 1), w = FK(prev)'s wrist (OpenPose joint 0) in
     camera coordinates with the sample's betas. sigma is drawn from a dedicated CPU generator seeded with `RGQ_SEED`,
     so the global RNG, the model's draws and the training data stream are those of the SLR arm. Evaluation never
     perturbs.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import optim

from backbones import resnet_forward_taps
from semkine.slr import SLRModel

RGQ_KEYS = frozenset({"RGQ", "RGQ_SIGMA_LO", "RGQ_SIGMA_HI", "RGQ_TZ_LR_MULT", "RGQ_SEED"})


class RGQModel(SLRModel):
    MODEL_KEYS = SLRModel.MODEL_KEYS | RGQ_KEYS

    def __init__(self, cfg=None, num_classes=None):
        super().__init__(cfg=cfg, num_classes=num_classes)
        mc = self.cfg.get("MODEL", {}) or {}
        self.rgq = bool(mc.get("RGQ", False))
        stray = sorted(k for k in mc if k in RGQ_KEYS and k != "RGQ")
        if not self.rgq:
            if stray:
                raise ValueError(f"{stray} without MODEL.RGQ: true would silently do nothing")
            return
        if not self.slr or "layer3" not in self.slr_readout.taps:
            raise ValueError("MODEL.RGQ is a factor of the surface-lifted readout: it needs MODEL.SLR: true with layer3 tapped")
        if self.weight_decay > 0.0:
            raise ValueError("MODEL.RGQ builds its own Adam param groups; TRAIN.WEIGHT_DECAY is not supported")
        self.rgq_lo = float(mc.get("RGQ_SIGMA_LO", 0.93))
        self.rgq_hi = float(mc.get("RGQ_SIGMA_HI", 1.01))
        self.rgq_lr_mult = float(mc.get("RGQ_TZ_LR_MULT", 0.1))
        if not (0.0 < self.rgq_lo <= self.rgq_hi) or not self.rgq_lr_mult > 0.0:
            raise ValueError("RGQ_SIGMA_LO <= RGQ_SIGMA_HI must be positive and RGQ_TZ_LR_MULT > 0")
        self.rgq_tz_index = int(self.slices.transl.start) + 2
        with torch.random.fork_rng(devices=[]):
            self.rgq_tz = nn.Linear(int(self.rn.fc.in_features), 1)
        nn.init.zeros_(self.rgq_tz.weight)
        nn.init.zeros_(self.rgq_tz.bias)
        g = torch.Generator(device="cpu")
        g.manual_seed(int(mc.get("RGQ_SEED", 20261026)))
        self._rgq_gen_state = g.get_state()          # a plain tensor (deep-copyable, not in the state dict)
        if self.prevpos_embed:
            self.prev_mlp.register_forward_hook(self._rgq_mask_prev_mlp_tz)

    def _rgq_mask_prev_mlp_tz(self, module, inputs, out):
        i = self.rgq_tz_index
        return torch.cat([out[:, :i], torch.zeros_like(out[:, i:i + 1]), out[:, i + 1:]], dim=1)

    # ------------------------------------------------------------------------------------------- readout
    def _trunk_with_root_head(self, z, prevpos, betas_f, k_f):
        if not getattr(self, "rgq", False):
            return super()._trunk_with_root_head(z, prevpos, betas_f, k_f)
        out, got = resnet_forward_taps(self.rn, z, self.slr_readout.taps)
        with torch.autocast(device_type=z.device.type, enabled=False):
            geo = self.slr_geometry(prevpos, betas_f, k_f)
            corr = self.slr_readout(got, geo, self.render_h, self.render_w)
            gap = torch.flatten(self.rn.avgpool(got["layer3"]), 1).float()
            tz = self.rgq_tz(gap)
        out = out + corr.to(out.dtype)
        i = self.rgq_tz_index
        return torch.cat([out[:, :i], tz.to(out.dtype), out[:, i + 1:]], dim=1)

    # ------------------------------------------------------------------------------------------- training perturbation
    @torch.no_grad()
    def rgq_perturb_prev(self, prevpos, betas, camera_K, sigma=None):
        """The previous state moved rigidly along its wrist ray to w / sigma (label untouched). Returns (prev', sigma)."""
        B = prevpos.shape[0]
        if sigma is None:
            g = torch.Generator(device="cpu")
            g.set_state(self._rgq_gen_state)
            u = torch.rand(B, generator=g, dtype=torch.float64)
            self._rgq_gen_state = g.get_state()
            sigma = (self.rgq_lo + (self.rgq_hi - self.rgq_lo) * u).to(device=prevpos.device, dtype=torch.float32)
        betas_f, _ = self._resolve_betas_K(prevpos, betas, camera_K)
        with torch.autocast(device_type=prevpos.device.type, enabled=False):
            _, joints = self._fk(prevpos.float(), betas_f.float())
        w = joints[:, 0].float()                                                       # (B, 3) wrist, camera coordinates
        t = self.slices.transl
        new = prevpos.clone()
        new[:, t] = (prevpos[:, t].float() + w * (1.0 / sigma - 1.0).unsqueeze(1)).to(prevpos.dtype)
        return new, sigma

    def _unpack_batch(self, batch):
        packed = super()._unpack_batch(batch)
        if not (getattr(self, "rgq", False) and self.training) or hasattr(packed, "events"):
            return packed
        x, prevpos, y, betas, camera_K = packed
        prevpos, _ = self.rgq_perturb_prev(prevpos, betas, camera_K)
        return x, prevpos, y, betas, camera_K

    # ------------------------------------------------------------------------------------------- optimiser
    def configure_optimizers(self):
        if not getattr(self, "rgq", False):
            return super().configure_optimizers()
        tcfg = self.cfg.get("TRAIN", {})
        lr = float(tcfg.get("LR", 1e-3))
        tz_ids = {id(p) for p in self.rgq_tz.parameters()}
        main = [p for p in self.parameters() if p.requires_grad and id(p) not in tz_ids]
        tzp = [p for p in self.rgq_tz.parameters() if p.requires_grad]
        opt = optim.Adam([{"params": main, "lr": lr}, {"params": tzp, "lr": lr * self.rgq_lr_mult}], lr=lr)
        self._register_ema_hook(opt)
        warmup = int(tcfg.get("WARMUP_STEPS", 0))
        schedule = str(tcfg.get("LR_SCHEDULE", "constant"))
        if schedule not in self.LR_SCHEDULES:
            raise ValueError(f"TRAIN.LR_SCHEDULE must be one of {list(self.LR_SCHEDULES)}, got {schedule!r}")
        total = int(tcfg.get("MAX_STEPS", 0))
        if schedule == "cosine" and total <= warmup:
            raise ValueError("TRAIN.LR_SCHEDULE=cosine needs MAX_STEPS > WARMUP_STEPS")
        if warmup <= 0 and schedule == "constant":
            return opt
        floor = self.LR_FLOOR

        def lr_lambda(step):                                   # the parent's schedule, applied to both groups
            if warmup > 0 and step < warmup:
                return float(step + 1) / float(warmup)
            if schedule == "constant":
                return 1.0
            p = min(max((step - warmup) / float(total - warmup), 0.0), 1.0)
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * p))

        sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
