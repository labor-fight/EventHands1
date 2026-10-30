"""Opt-in S43 residual inputs, preserving its MANO decoder and loss.

The shared LNES/render branch remains present in every arm.  New sensor
channels enter the *three-channel conv1 feature map*, rather than a synthetic
two-channel LNES.  The typed relation package enters the existing readout.
All original network parameters are trainable; no production file is patched.
"""
from __future__ import annotations

import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from tools.s43_integration.model import DEFAULT_CKPT, DEFAULT_CFG, load_legacy

ARMS = ("s43", "enhanced", "no_order", "full")
SENSOR_CHANNELS = 20


def _tensor(value, reference):
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


class TypedSupportAdapter(nn.Module):
    """Read f and the (a, lambda) factors without materialising a Gram matrix.

This learned summary is a readout factor, not a calibrated confidence model.
Zero evidence is hard gated, including every learned bias.
"""

    def __init__(self, h=24, hidden=32):
        super().__init__()
        self.h = int(h)
        self.direction = nn.Sequential(nn.Linear(51 * h, hidden), nn.SiLU())
        self.support = nn.Sequential(nn.Linear(51 + h, hidden), nn.SiLU())
        self.output = nn.Linear(2 * hidden + 3, 51)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, f, a, lam, conditions):
        if f.ndim != 3 or f.shape[1:] != (51, self.h):
            raise ValueError(f"expected f[B,51,{self.h}]")
        if a.ndim != 3 or a.shape[0] != len(f) or a.shape[2] != 51:
            raise ValueError("expected a[B,N,51]")
        if lam.shape != (len(f), a.shape[1], self.h) or torch.any(lam < 0):
            raise ValueError("expected nonnegative lambda[B,N,H]")
        if not all(bool(torch.isfinite(x).all()) for x in (f, a, lam, conditions)):
            raise ValueError("nonfinite typed relation input")
        # Fixed analytic transforms, identical in training and online inference.
        ff = torch.sign(f) * torch.log1p(f.abs())
        ll = torch.log1p(lam)
        rows = (lam.sum(-1) > 0).to(a.dtype)
        point = self.support(torch.cat((torch.sign(a) * torch.log1p(a.abs()), ll), -1))
        pooled = (point * rows[..., None]).sum(1) / rows.sum(1, keepdim=True).clamp_min(1)
        features = torch.cat((self.direction(ff.flatten(1)), pooled, conditions), -1)
        live = (rows.sum(1, keepdim=True) > 0).to(features.dtype)
        return self.output(features) * live


class S43RelationModel(nn.Module):
    """Common initialized, fully trainable S43 plus explicitly counted deltas.

The old sparse active mask and ZERO_EVENT_GATE stay in force.  Sensor history
therefore affects conv1 only at the original active sites; geometry has its
own readout path.  This limitation is part of the first implementation.
    """

    def __init__(self, base, arm="full", sensor_channels=SENSOR_CHANNELS, h=24, support_reader="points"):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        if not getattr(base, "sparse_trunk", False):
            raise ValueError("this experiment requires the unchanged sparse S43 trunk")
        if getattr(base, "active_head", False) or getattr(base, "transl_head", None) is not None:
            raise ValueError("the canonical S43 readout is required")
        self.base, self.arm = base, arm
        self.sensor_channels, self.h = int(sensor_channels), int(h)
        if support_reader not in ("points", "joint_gram"):
            raise ValueError("SUPPORT_READER must be points or joint_gram")
        self.reader = support_reader
        self.sensor_adapter = nn.Conv2d(self.sensor_channels, base.conv1.out_channels, 3, padding=1, bias=False)
        nn.init.zeros_(self.sensor_adapter.weight)
        if self.reader == "joint_gram":
            from .joint_reader import JointGramSupportAdapter
            self.typed_adapter = JointGramSupportAdapter(self.h)
        else:
            self.typed_adapter = TypedSupportAdapter(self.h)
        for parameter in self.base.parameters():
            parameter.requires_grad_(True)
        # Disabled branches are explicit, not trainable bias-only pose heads.
        for parameter in self.sensor_adapter.parameters():
            parameter.requires_grad_(arm != "s43")
        for parameter in self.typed_adapter.parameters():
            parameter.requires_grad_(arm in ("no_order", "full"))
        self._active_sensor = None
        self._active_typed = None
        self._conv_handle = self.base.conv1.register_forward_hook(self._add_sensor)
        self._head_handle = self.base.rn.fc.register_forward_hook(self._add_typed)

    def _add_sensor(self, module, inputs, output):
        if self._active_sensor is None:
            return output
        sensor = F.interpolate(self._active_sensor, size=output.shape[-2:], mode="nearest")
        return output + self.sensor_adapter(sensor).to(output.dtype)

    def _add_typed(self, module, inputs, output):
        return output if self._active_typed is None else output + self._active_typed.to(output.dtype)

    def forward(self, representations, prev, betas=None, camera_K=None):
        if self._active_sensor is not None or self._active_typed is not None:
            raise RuntimeError("a model instance cannot serve concurrent forwards")
        if len(representations) != len(prev):
            raise ValueError("representation/previous-state batch mismatch")
        image = _tensor(np.stack([r["lnes"] for r in representations]), prev)
        try:
            if self.arm != "s43":
                sensor = _tensor(np.stack([r["sensor"] for r in representations]), prev)
                if sensor.ndim != 4 or sensor.shape[-1] != self.sensor_channels:
                    raise ValueError("sensor input must retain its H,W,channel grid")
                if not bool(torch.isfinite(sensor).all()):
                    raise ValueError("nonfinite sensor input")
                self._active_sensor = sensor.permute(0, 3, 1, 2).contiguous()
            if self.arm in ("no_order", "full"):
                values = [_tensor(np.stack([r[k] for r in representations]), prev)
                          for k in ("f", "a", "lam")]
                conditions = _tensor(np.asarray([[r.get("delta_commit_ms", 50.) / 50.,
                    r.get("output_lag_ms", 1.), float(r.get("missing", False))]
                    for r in representations], np.float32), prev)
                self._active_typed = self.typed_adapter(*values, conditions)
            return self.base(image, prev, betas=betas, camera_K=camera_K)
        finally:
            self._active_sensor = self._active_typed = None

    def loss(self, pred, target, betas=None):
        return self.base._compute_loss(pred, target, betas)

    def counts(self):
        def n(module, active=False):
            return sum(p.numel() for p in module.parameters() if not active or p.requires_grad)
        return {"total": n(self), "trainable": n(self, True), "base": n(self.base),
                "SUPPORT_READER": self.reader,
                "sensor_increment": n(self.sensor_adapter), "typed_increment": n(self.typed_adapter),
                "sensor_structurally_active_parameters": 540 if self.arm == "enhanced" else 162 if self.arm in ("full", "no_order") else 0,
                "typed_structurally_active_parameters": n(self.typed_adapter) if self.arm in ("full", "no_order") else 0,
                "sensor_active": self.arm != "s43", "typed_active": self.arm in ("no_order", "full"),
                "base_fully_trainable": True, "same_original_loss": True,
                "common_LNES_render_path": True, "sparse_mask": "unchanged current LNES + rendered previous mesh",
                "new_loss": False}


def load_relation_model(config, checkpoint=None, device="cpu"):
    import yaml
    cfg = yaml.safe_load(Path(config).read_text()) if not isinstance(config, dict) else copy.deepcopy(config)
    rel = cfg["REL"]
    # Seed independently of arm, so every original and additional shared weight matches.
    torch.manual_seed(int(rel.get("SEED", 3407)))
    base, _ = load_legacy(rel.get("BASE_CHECKPOINT", DEFAULT_CKPT),
                          rel.get("BASE_CONFIG", DEFAULT_CFG), device)
    model = S43RelationModel(base, rel["ARM"], rel.get("SENSOR_CHANNELS", 20), rel.get("H", 24),
                             rel.get("SUPPORT_READER", "points")).to(device)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location=device)
        if payload.get("schema") != "s43_relation_training_v1" or payload["arm"] != model.arm:
            raise ValueError("relation checkpoint schema/arm mismatch")
        # Original R0 checkpoints predate this metadata and mean points.
        # Identical parameter shapes must not silently reinterpret that head.
        if payload.get("reader", "points") != model.reader:
            raise ValueError("relation checkpoint SUPPORT_READER mismatch")
        model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, cfg
