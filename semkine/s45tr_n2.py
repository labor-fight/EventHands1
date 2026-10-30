r"""S45-TR arm N2: same-packet fixed-point consistency (docs/EXPERIMENT_RECORDS.md#s45tr-prereg-20261030).

S45-DEEP measured that S45 is not a fixed point at the truth: feeding the same correct events again and again moves
the state from the ground truth toward the training mean pose (RA 4.9 -> 21 mm, shrinkage k 0.10 -> 0.50). N2 adds
one term to the training objective and changes nothing else:

    pred1 = f(events, prior)                 the S45 forward and loss, unchanged
    pred2 = f(events, stopgrad(pred1))       the same packet (same events, same augmentation), prior = pred1
    loss  = L(pred1, GT) + lambda * L(pred2, GT),   lambda = 1,   then log10 as S45

`L` is S45's `so3_trans_fk` loss. Evaluation is untouched. The logged route statistics and the journalled
prediction stay those of the first pass, so the S45-TR audit records the same quantities as for every arm.

Opt-in only: `install(lambda)` replaces `MNISTModel.training_step` for the process (both DDP ranks run the wrapper).
"""
from __future__ import annotations

import dataclasses

_STATE = {"lambda": None, "original": None}


def _require_s45_path(model, batch):
    if (model.distill_weight > 0.0 or model.self_p > 0.0 or model.rollout_p > 0.0 or model.accel_weight > 0.0
            or model.gain_reg_w > 0.0 or model.loss_type != "so3_trans_fk" or not hasattr(batch, "events")):
        raise RuntimeError("N2 is defined on the S45 recipe only (packet batches, so3_trans_fk, no other terms)")


def training_step(self, batch, batch_nb):
    lam = _STATE["lambda"]
    _require_s45_path(self, batch)
    pred, y, betas, packed = self._predict_batch(batch)
    loss1, parts = self._compute_loss(pred, y, betas)
    first_stats = self.route_stats
    first_prediction = getattr(self, "_repro_last_prediction", None)
    second = dataclasses.replace(packed, prev_state=pred.detach().to(packed.prev_state.dtype))
    pred2 = self.forward_packet(second)
    loss2, _ = self._compute_loss(pred2, y, betas)
    self.route_stats = first_stats
    if first_prediction is not None:
        self._repro_last_prediction = first_prediction
    loss = loss1 + lam * loss2
    self.log("train_loss", loss, prog_bar=True, sync_dist=True)
    self.log("train_loss_first", loss1, sync_dist=True)
    self.log("train_loss_second", loss2, sync_dist=True)
    if getattr(self, "xyz_mesh", False) and int(self.global_step) % 100 == 0:
        self._agent_last_loss = {"loss": float(loss.detach()), "mano": float(parts["mano_loss"].detach()),
                                 "rot": float(parts["rot_loss"].detach()), "pos": float(parts["pos_loss"].detach())}
    self.log("train_mano_loss", parts["mano_loss"], sync_dist=True)
    self.log("train_pos_loss", parts["pos_loss"], sync_dist=True)
    self.log("train_rot_loss", parts["rot_loss"], sync_dist=True)
    self._log_so3_fk_parts("train", parts)
    self._log_route_stats("train")
    loss_f = loss.float()
    return loss_f.log10() if self.log10_loss else loss_f


def install(lam: float = 1.0) -> None:
    import semkine.train as T
    cls = T.MNISTModel
    if _STATE["original"] is None:
        _STATE["original"] = cls.training_step
    _STATE["lambda"] = float(lam)
    cls.training_step = training_step


def uninstall() -> None:
    import semkine.train as T
    if _STATE["original"] is not None:
        T.MNISTModel.training_step = _STATE["original"]
    _STATE["lambda"] = None


def installed():
    return _STATE["lambda"]


__all__ = ["install", "installed", "training_step", "uninstall"]
