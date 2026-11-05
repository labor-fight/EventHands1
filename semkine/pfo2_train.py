r"""PFO2 stage-1 training-side changes (docs/EXPERIMENT_RECORDS.md#pfo2-s1-prereg-20261103), opt-in, PFO2 processes only.

  install_disp_loss(model_cls, weight)
      the displacement head's auxiliary loss (semkine/pfo2_trunk.py `disp`): weight x mean over samples and all 778
      vertices of |u_v - R_prev^T (V_v(GT) - V_v(prev))| (metres, previous hand frame), added to the training loss
      inside `_compute_loss` (before the log10 of training_step); logged as train_loss_disp. Validation is unchanged.
"""
from __future__ import annotations

import torch

_NATIVE = {"compute": None}


def install_disp_loss(model_cls, weight):
    if _NATIVE["compute"] is not None:
        raise RuntimeError("PFO2 loss already installed")
    native = model_cls._compute_loss
    _NATIVE["compute"] = (model_cls, native)
    weight = float(weight)

    def compute(self, pred, y, betas=None):
        loss, parts = native(self, pred, y, betas)
        aux = getattr(getattr(self, "xyz", None), "pfo2_aux", None) or {}
        if not self.training or "u" not in aux:
            return loss, parts
        with torch.no_grad():
            b = betas.to(device=y.device, dtype=torch.float32) if betas is not None else \
                torch.zeros(y.shape[0], 10, device=y.device)
            Vgt, _ = self._fk(y.float(), b)
            target = torch.einsum("bik,bvi->bvk", aux["R"], Vgt.float() - aux["verts"].float())
        l_disp = (aux["u"] - target).norm(dim=-1).mean()
        self.log("train_loss_disp", l_disp.detach(), sync_dist=True)
        return loss + weight * l_disp, {**parts, "loss_disp": l_disp.detach()}
    model_cls._compute_loss = compute


def uninstall():
    if _NATIVE["compute"] is not None:
        cls, native = _NATIVE["compute"]
        cls._compute_loss = native
        _NATIVE["compute"] = None


__all__ = ["install_disp_loss", "uninstall"]
