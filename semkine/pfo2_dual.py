r"""PFO2 arm A2, double-forward correction supervision (docs/EXPERIMENT_RECORDS.md#pfo2-s1-prereg-20261103): opt-in, PFO2
training processes only; deployment and evaluation are unchanged.

`install(model_cls, lam_max=0.5, start=2000, end=6000)` replaces `training_step`: the base step's forward and loss
(L_clean, the batch's previous state with the training prior-noise contract) plus, from step `start`, a second forward of
the same packets from an injected previous state and its loss L_pert against the same target:
    loss = L_clean + lam(step) * L_pert,  lam = lam_max * clamp((step - start) / (end - start), 0, 1)
The injection per sample is R15 (the previous root rotation left-multiplied by a 15 degree rotation about a uniform
axis) or F15 (finger axis-angles + N(0, 0.15^2) per component), equally likely, from a generator seeded by
(SEED, global step) on the batch's previous state (the batch carries no GT previous state). Before `start` the step is
the base's exactly (no second forward), so an A2 run equals its base bitwise up to step `start`.
"""
from __future__ import annotations

import math

import torch

_NATIVE = {"step": None}


def axis_angle_to_matrix(aa):
    th = aa.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = aa / th
    K = torch.zeros(aa.shape[0], 3, 3, dtype=aa.dtype, device=aa.device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    s, c = torch.sin(th).unsqueeze(-1), torch.cos(th).unsqueeze(-1)
    return torch.eye(3, dtype=aa.dtype, device=aa.device) + s * K + (1 - c) * (K @ K)


def matrix_to_axis_angle(R):
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    th = torch.acos(cos)
    w = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
    return w * (th / (2 * torch.sin(th))).unsqueeze(-1)


def inject(prev, gen, root_deg=15.0, finger_rad=0.15):
    """R15 or F15 on each row of `prev` (B, 51); returns the injected copy and the kind (0 = R15, 1 = F15)."""
    B = prev.shape[0]
    p = prev.detach().double().cpu()
    kind = torch.randint(0, 2, (B,), generator=gen)
    u = torch.randn(B, 3, generator=gen, dtype=torch.float64)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    noise = torch.randn(B, 45, generator=gen, dtype=torch.float64) * finger_rad
    out = p.clone()
    R = axis_angle_to_matrix(u * math.radians(root_deg)) @ axis_angle_to_matrix(p[:, 3:6])
    out[:, 3:6] = torch.where((kind == 0).unsqueeze(-1), matrix_to_axis_angle(R), p[:, 3:6])
    out[:, 6:51] = torch.where((kind == 1).unsqueeze(-1), p[:, 6:51] + noise, p[:, 6:51])
    return out.to(prev.dtype).to(prev.device), kind


def install(model_cls, lam_max=0.5, start=2000, end=6000):
    if _NATIVE["step"] is not None:
        raise RuntimeError("A2 already installed")
    native = model_cls.training_step
    _NATIVE["step"] = (model_cls, native)

    def training_step(self, batch, batch_nb):
        step = int(self.global_step)
        lam = lam_max * min(max((step - start) / float(end - start), 0.0), 1.0)
        if lam <= 0.0:
            return native(self, batch, batch_nb)
        pred, y, betas, packed = self._predict_batch(batch)
        loss, parts = self._compute_loss(pred, y, betas)
        gen = torch.Generator().manual_seed(int(self.cfg["SEED"]) * 1000003 + step)
        prev2, kind = inject(packed.prev_state, gen)
        import dataclasses
        packed2 = dataclasses.replace(packed, prev_state=prev2)
        pred2 = self.forward_packet(packed2)
        loss2, _ = self._compute_loss(pred2, y, betas)
        total = loss + lam * loss2
        self.log("train_loss", total, prog_bar=True, sync_dist=True)
        self.log("train_loss_clean", loss.detach(), sync_dist=True)
        self.log("train_loss_pert", loss2.detach(), sync_dist=True)
        self.log("train_dual_lambda", lam, sync_dist=True)
        self.log("train_mano_loss", parts["mano_loss"], sync_dist=True)
        self.log("train_pos_loss", parts["pos_loss"], sync_dist=True)
        self.log("train_rot_loss", parts["rot_loss"], sync_dist=True)
        self._log_so3_fk_parts("train", parts)
        self._log_route_stats("train")
        total_f = total.float()
        return total_f.log10() if self.log10_loss else total_f
    model_cls.training_step = training_step


def uninstall():
    if _NATIVE["step"] is not None:
        cls, native = _NATIVE["step"]
        cls.training_step = native
        _NATIVE["step"] = None


__all__ = ["inject", "install", "uninstall"]
