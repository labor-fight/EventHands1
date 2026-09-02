#!/usr/bin/env python3
"""EventHands absolute-pose models (config-driven 12D / 51D)."""
from __future__ import annotations

import contextlib
import dataclasses
import gc
import math
from typing import Any, Dict, Optional

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl

from semkine import events as EV

from pose_repr import (
    axis_angle_to_quaternion,
    decode_to_mano_inputs,
    get_slices,
    split_losses,
)


class BaseModel(pl.LightningModule):
    """Original EventHands loss layout preserved for 12D; generalized via slices.

    LOSS.TYPE picks the training objective:

      mse_51d      (default) the historical weighted elementwise MSE
      so3_trans_fk L_rot + L_trans + 2 L_FK, each term on the manifold of the
                   quantity it measures -- see :meth:`_so3_fk_loss`
    """

    LOSS_TYPES = ("mse_51d", "so3_trans_fk")
    #: cosine decays to this fraction of the peak rather than to zero, so the last
    #: steps still move and the grid's final points are not all the same checkpoint.
    LR_FLOOR = 0.02
    #: wrist index in the 21-joint OpenPose ordering ManoLayer returns
    FK_ROOT_JOINT = 0

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.cfg = cfg or {
            "MODEL": {"POSE_REPR": "mano_pca6", "OUTPUT_DIM": 12},
            "LOSS": {
                "LAMBDA_POSE": 60.0,
                "LAMBDA_T": 30000.0,
                "LAMBDA_R": 60.0,
                "NORMALIZER": 12,
                "LOG10": True,
            },
            "TRAIN": {"LR": 1e-3, "WARMUP_STEPS": 0},
        }
        self.save_hyperparameters(self.cfg)
        self.pose_repr = self.cfg["MODEL"]["POSE_REPR"]
        self.output_dim = int(self.cfg["MODEL"]["OUTPUT_DIM"])
        self.slices = get_slices(self.pose_repr, self.output_dim)
        loss_cfg = self.cfg["LOSS"]
        self.lambda_pose = float(loss_cfg["LAMBDA_POSE"])
        self.lambda_t = float(loss_cfg["LAMBDA_T"])
        self.lambda_r = float(loss_cfg["LAMBDA_R"])
        self.normalizer = float(loss_cfg["NORMALIZER"])
        self.log10_loss = bool(loss_cfg.get("LOG10", True))
        self.loss_type = str(loss_cfg.get("TYPE", "mse_51d"))
        if self.loss_type not in self.LOSS_TYPES:
            raise ValueError(
                f"LOSS.TYPE must be one of {list(self.LOSS_TYPES)}, got {self.loss_type!r}"
            )
        self.rot_weight = float(loss_cfg.get("ROT_WEIGHT", 1.0))
        self.trans_weight = float(loss_cfg.get("TRANS_WEIGHT", 1.0))
        self.fk_weight = float(loss_cfg.get("FK_WEIGHT", 2.0))
        # S4. `so3_trans_fk` as first written leaves absolute translation almost unconstrained:
        # `L_FK` is root-relative by design, and `L_trans` is a SmoothL1 whose default transition
        # at 1 m puts every realistic error (centimetres) deep in the quadratic region, where the
        # gradient is proportional to the error and therefore negligible. Measured consequence:
        # non-aligned MPJPE degraded 63.7 -> 83.6 mm while root-relative MPJPE improved. Two
        # independently switchable remedies, both defaulting to the legacy behaviour so the
        # existing arms stay bit-identical:
        #   ABS_FK_WEIGHT  a non-root-relative FK term, in metres, which is the quantity the
        #                  non-aligned metric actually measures
        #   TRANS_BETA     the SmoothL1 transition; setting it near the error scale (~0.01 m)
        #                  restores a constant gradient instead of a vanishing one
        self.abs_fk_weight = float(loss_cfg.get("ABS_FK_WEIGHT", 0.0))
        self.trans_beta = float(loss_cfg.get("TRANS_BETA", 1.0))

    def forward(self, x, prevpos):
        raise NotImplementedError()

    def _compute_loss(self, pred, y, betas=None):
        if self.loss_type == "so3_trans_fk":
            return self._so3_fk_loss(pred, y, betas)
        return self._mse_51d_loss(pred, y, betas)

    def _so3_fk_loss(self, pred, y, betas):
        """L_rot + L_trans + 2 L_FK, all three read off the same MANO decode.

        The elementwise 51D MSE compares axis-angles and metres in one sum, so its
        lambdas (60 / 60 / 30000) are doing unit conversion rather than expressing a
        trade-off. Here each term already lives in its own natural scale:

          L_rot   1 - cos^2 of the quaternion angle, per joint. Dimensionless and
                  antipodally invariant, since q and -q are the same rotation.
          L_trans SmoothL1 on the root translation, in metres (the dataset's unit).
          L_FK    mean Euclidean distance between root-relative MANO joints, in
                  metres. Root-relative on purpose: translation is already L_trans's
                  job, so this term only has to explain articulation.

        The GT branch decodes through the very same MANO, so the two skeletons are
        identical by construction and the loss is zero at the ground truth.
        """
        if not hasattr(self, "mano"):
            raise RuntimeError("LOSS.TYPE=so3_trans_fk needs MODEL.PREV_RENDER for MANO")
        n_joint_aa = 1 + (self.slices.local.stop - self.slices.local.start) // 3
        # The transform chain is precision sensitive and the trunk may run in bf16.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred = pred.float()
            y = y.float()
            betas, _ = self._resolve_betas_K(pred, betas, None)

            dec_p = decode_to_mano_inputs(
                pred, self.pose_repr, self.mano.hands_components, self.mano.hands_mean
            )
            with torch.no_grad():
                dec_g = decode_to_mano_inputs(
                    y, self.pose_repr, self.mano.hands_components, self.mano.hands_mean
                )

            def quats(dec):
                aa = torch.cat([dec["global_orient"], dec["local_full_aa"]], dim=-1)
                return axis_angle_to_quaternion(aa.view(-1, n_joint_aa, 3))

            dot = (quats(dec_p) * quats(dec_g)).sum(dim=-1)
            # Clamped so a perfect prediction cannot round to a negative loss, which
            # training_step's log10 would turn into NaN.
            loss_rot = (1.0 - dot.square().clamp(max=1.0)).mean()
            loss_trans = F.smooth_l1_loss(dec_p["transl"], dec_g["transl"],
                                          beta=self.trans_beta)

            _, joints_p = self._fk(pred, betas)
            with torch.no_grad():
                _, joints_g = self._fk(y, betas)
            root = self.FK_ROOT_JOINT
            rel_p = joints_p - joints_p[:, root : root + 1]
            rel_g = joints_g - joints_g[:, root : root + 1]
            joint_dist = torch.norm(rel_p - rel_g, dim=-1)
            loss_fk = joint_dist.mean()
            abs_dist = torch.norm(joints_p - joints_g, dim=-1)
            loss_abs_fk = abs_dist.mean()

            loss = (
                self.rot_weight * loss_rot
                + self.trans_weight * loss_trans
                + self.fk_weight * loss_fk
                + self.abs_fk_weight * loss_abs_fk
            )
            with torch.no_grad():
                legacy, _ = self._mse_51d_loss(pred, y)
                angle_deg = torch.rad2deg(2.0 * dot.abs().clamp(max=1.0).acos()).mean()
        return loss, {
            "loss_rot": loss_rot,
            "loss_trans": loss_trans,
            "loss_fk": loss_fk,
            "loss_abs_fk": loss_abs_fk,
            "abs_joint_error_mm": abs_dist.mean().detach() * 1000.0,
            "legacy_loss_51d": legacy,
            "rotation_error_deg": angle_deg,
            "joint_error_mm": joint_dist.mean().detach() * 1000.0,
            # Aliases so the existing train/val logging keeps working unchanged.
            "mano_loss": loss_rot,
            "pos_loss": loss_trans,
            "rot_loss": loss_fk,
        }

    def _mse_51d_loss(self, pred, y, betas=None):
        # Match original EventHands weighted MSE for 12D:
        # (mano*6/0.1 + pos*3/0.0001 + rot*3/0.05)/12
        # which equals (60*mano + 30000*pos + 60*rot)/12 when using mean MSE.
        # For 51D: same lambdas, NORMALIZER=51.
        if self.pose_repr == "mano_pca6" and self.output_dim == 12:
            # Exact original slicing for bit-compatibility of loss semantics
            mano_loss = F.mse_loss(pred[:, :6], y[:, :6])
            pos_loss = F.mse_loss(pred[:, 6:9], y[:, 6:9])
            rot_loss = F.mse_loss(pred[:, 9:], y[:, 9:])
            z_loss = F.mse_loss(pred[:, 8:9], y[:, 8:9])
            loss = (
                mano_loss * 6 / 0.1 + pos_loss * 3 / 0.0001 + rot_loss * 3 / 0.05
            ) / 12
            return loss, {
                "mano_loss": mano_loss,
                "pos_loss": pos_loss,
                "rot_loss": rot_loss,
                "z_loss": z_loss,
            }

        l_local, l_root, l_t = split_losses(pred, y, self.pose_repr)
        loss = (
            self.lambda_pose * l_local
            + self.lambda_r * l_root
            + self.lambda_t * l_t
        ) / self.normalizer
        parts = {
            "mano_loss": l_local,
            "pos_loss": l_t,
            "rot_loss": l_root,
            "z_loss": F.mse_loss(pred[:, self.slices.transl][:, 2:3], y[:, self.slices.transl][:, 2:3]),
        }
        # S42. The one piece of S39 that measured well, grafted onto the loss that wins
        # recursive RA: `so3_trans_fk` lost RA (+1.39, fingers) but improved absolute MPJPE
        # by 2.5 mm, and the abs gain is attributable to the metre-scale absolute-FK term.
        # Added under a weight so `mse_51d` keeps its finger-favouring implicit weighting;
        # weight 0 is bit-identical to every existing mse arm. Guarded by loss_type so the
        # legacy-logging call inside `_so3_fk_loss` never pays a second FK.
        if self.abs_fk_weight > 0 and self.loss_type == "mse_51d":
            if not hasattr(self, "mano"):
                raise RuntimeError("LOSS.ABS_FK_WEIGHT on mse_51d needs MANO "
                                   "(MODEL.PREV_FK_DIRECT or PREV_RENDER)")
            with torch.autocast(device_type=pred.device.type, enabled=False):
                pred32, y32 = pred.float(), y.float()
                betas_r, _ = self._resolve_betas_K(pred32, betas, None)
                _, joints_p = self._fk(pred32, betas_r)
                with torch.no_grad():
                    _, joints_g = self._fk(y32, betas_r)
                abs_dist = torch.norm(joints_p - joints_g, dim=-1)
                loss_abs_fk = abs_dist.mean()
            loss = loss + self.abs_fk_weight * loss_abs_fk
            parts["loss_abs_fk"] = loss_abs_fk
            parts["abs_joint_error_mm"] = abs_dist.mean().detach() * 1000.0
        return loss, parts

    def _unpack_batch(self, batch):
        if hasattr(batch, "events"):
            return batch
        if len(batch) >= 5:
            return batch[0], batch[1], batch[2], batch[3], batch[4]
        if len(batch) == 4:
            return batch[0], batch[1], batch[2], batch[3], None
        return batch[0], batch[1], batch[2], None, None

    def unroll_p_now(self) -> float:
        """`UNROLL_P` after the ramp, at the current optimiser step."""
        if self.unroll_p <= 0.0:
            return 0.0
        s0, s1 = self.unroll_ramp
        if s1 <= s0:
            return self.unroll_p
        frac = min(max((int(self.global_step) - s0) / float(s1 - s0), 0.0), 1.0)
        return self.unroll_p * frac

    def _maybe_unroll(self, batch):
        """Replace a paired sample's conditioning state with the lead window's prediction.

        The measured defect this addresses: inside its own recursion the network's per-step update
        collapses to ~11% of the needed motion, while under teacher forcing with independent noise
        it moves 0.77-0.93 of it. The conditioning distribution training samples (ground truth plus
        independent noise) never contains self-generated, time-correlated drift, which is exposure
        bias in its standard form (Ross & Bagnell, AISTATS'11; Bengio et al., NeurIPS'15). The pair
        is how that distribution enters training.

        The lead forward is under `no_grad` and its result is detached: this changes what the
        network is conditioned *on*, not how gradient reaches it, so a step costs one extra forward
        and no extra backward graph.

        Two forms, measured (probe 2026-08-25 21:40, `closed_loop_sensitivity_unroll_50ms.json`):

        - **replacement** (UNROLL_RESIDUAL false): conditioning becomes the lead prediction
          itself. Because that prediction is near-clean (~1-2 mm off GT), this *narrows* the error
          curriculum: TF improved to 10.3-10.9 (vs 12.1 without unroll) but the recursive echo
          excess rose from x1.85-1.95 back to x2.22-2.26 and recursive RA worsened 27.8 -> 31-34.
          The network learned "prev is trustworthy", the opposite of what the closed loop needs.
        - **residual** (UNROLL_RESIDUAL true, E5.5c): conditioning becomes
          `(GT + curriculum noise) + (lead prediction - lead target)` -- the curriculum keeps its
          large-error coverage and the self-error contributes only its time-correlated structure.
        """
        if not (isinstance(batch, (tuple, list)) and len(batch) == 2
                and hasattr(batch[0], "events") and hasattr(batch[1], "events")):
            return batch
        lead, main = batch
        p = self.unroll_p_now()
        if p <= 0.0:
            return main
        with torch.no_grad(), self._frozen_bn_stats():
            # `no_grad` stops the gradient but not the BatchNorm buffer writes, so without the
            # second guard the lead window updates all 20 running statistics an extra time per
            # step. Measured on the penalty's identical second forward: checksum 5120.95 ->
            # 5409.75 in one step, and recursive RA 16.64 -> 18.6-20.5 mm. The s22/s23 arms were
            # trained with that leak, so their numbers confound unroll against BN contamination.
            est = self.forward_packet(lead).detach()
        take = torch.rand(main.prev_state.shape[0], device=main.prev_state.device) < p
        cond = est.to(main.prev_state.dtype)
        if self.unroll_residual:
            cond = main.prev_state + (est - lead.target).to(main.prev_state.dtype)
        main.prev_state = torch.where(take[:, None], cond, main.prev_state)
        return main

    def _sample_cond_delta(self, ref):
        """A conditioning-state perturbation drawn from the curriculum's own noise distribution.

        The penalty is only meaningful if it measures sensitivity over the same directions the
        tracker actually meets, so the per-block scales come from the same `PREV_NOISE_*` entries
        the dataset and the evaluation protocol use, and the layout matches
        `semkine.eval_track.sample_init_noise`: translation, global rotation, then pose.
        """
        track = self.cfg.get("TRACK", {}) or {}
        s = torch.empty(51, device=ref.device, dtype=ref.dtype)
        s[0:3] = float(track.get("PREV_NOISE_T", 0.0))
        s[3:6] = float(track.get("PREV_NOISE_R", 0.0))
        s[6:51] = float(track.get("PREV_NOISE_POSE", 0.0))
        return torch.randn_like(ref) * s * self.gain_reg_scale

    @contextlib.contextmanager
    def _frozen_bn_stats(self):
        """Run a forward pass without letting it touch the BatchNorm running statistics.

        The trunk has 20 BatchNorm layers, so a second train-mode forward updates every one of them
        a second time, using a batch whose conditioning state was deliberately corrupted. Inference
        then normalises with statistics that are part real and part perturbation. Measured: the
        running-stat checksum moved 5120.95 -> 5409.75 across the penalty forward at step 0, and
        recursive RA jumped from the baseline's 16.64 mm to 18.8-19.7 mm at *every* lambda, largest
        at the smallest lambda -- the damage tracked the extra forward, not the penalty weight.

        `track_running_stats = False` in train mode makes `F.batch_norm` receive `None` for both
        buffers while still normalising with batch statistics, so the penalty sees exactly the
        normalisation the task forward saw and the buffers are left alone. Switching the modules to
        eval mode instead would normalise with running statistics, which changes what the penalty
        measures.
        """
        bn = [m for m in self.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
        was = [m.track_running_stats for m in bn]
        for m in bn:
            m.track_running_stats = False
        try:
            yield
        finally:
            for m, w in zip(bn, was):
                m.track_running_stats = w

    def _ra_joints(self, params, betas):
        """Root-aligned MANO joints, the space the probe's gain is measured in."""
        _, j = self._fk(params, betas)
        return j - j[:, self.FK_ROOT_JOINT : self.FK_ROOT_JOINT + 1]

    def _gain_penalty(self, packed, pred):
        """Squared retention, measured in root-aligned joint space.

        The metric is not a detail, it is the whole term. The probe's `gain_rand` -- the quantity
        that correlates +0.955 with recursive error -- is a ratio of *root-aligned joint distances
        in millimetres* (`outputs/semkine/echo_gain_50ms.json`). Taking the same ratio in the raw
        51-D parameter space instead is a different objective, because that norm mixes metres with
        radians and is ~97% dominated by the 45-D pose block (sqrt(45)*0.05 rad against
        sqrt(3)*0.005 m), while root alignment discards global translation outright. Measured: a
        parameter-space penalty was monotone in lambda yet drove joint-space retention *above* the
        unregularised baseline at matched step 2000 (0.457/0.439/0.335/0.221 against 0.242), i.e.
        it optimised a direction nearly orthogonal to the one that predicts drift. Note that the
        magnitudes agreed (0.182 against the probe's 0.236) even while the metrics did not, so
        agreeing magnitudes are not evidence that two sensitivities are the same quantity.

        Both output branches keep their graph, so a step costs one extra forward and one extra
        backward. Detaching `f(x)` would turn this into a one-sided pull of the perturbed branch
        toward the current unperturbed output, which is a different objective. The denominator is
        under `no_grad` because it does not depend on the parameters.
        """
        if not (self.prev_render or self.prev_fk):
            # `self.mano` only exists on the conditioning paths, and silently falling back to a
            # parameter-space ratio is the exact failure this docstring documents.
            raise ValueError("TRACK.GAIN_REG_W needs MODEL.PREV_RENDER or PREV_FK_DIRECT "
                             "for the MANO forward pass")
        if hasattr(packed, "events"):
            prev, betas = packed.prev_state, packed.betas
            d = self._sample_cond_delta(prev)
            with self._frozen_bn_stats():
                out = self.forward_packet(dataclasses.replace(packed, prev_state=prev + d))
        else:
            x, prev, _y, betas, camera_K = packed
            d = self._sample_cond_delta(prev)
            with self._frozen_bn_stats():
                out = self(x, prev + d, betas=betas, camera_K=camera_K)
        betas, _ = self._resolve_betas_K(prev, betas, None)
        d_out = (self._ra_joints(out.float(), betas)
                 - self._ra_joints(pred.float(), betas)).norm(dim=-1).mean(-1)
        with torch.no_grad():
            d_in = (self._ra_joints((prev + d).float(), betas)
                    - self._ra_joints(prev.float(), betas)).norm(dim=-1).mean(-1)
        return (d_out / d_in.clamp_min(1e-9)).pow(2).mean()

    def _bn_checksum(self):
        """Sum of every BatchNorm running statistic, as a single number."""
        t = 0.0
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if m.running_mean is not None:
                    t += float(m.running_mean.double().abs().sum())
                if m.running_var is not None:
                    t += float(m.running_var.double().abs().sum())
        return t

    def _predict_batch(self, batch):
        """One forward that accepts either the legacy 5-tuple or an `EventPacketBatch`.

        Returns the unpacked batch as well, so a second forward can reuse the exact conditioning
        state this one saw: `_maybe_unroll` and the dataset's noise are both random, and
        re-unpacking would silently compare two different conditioning states.
        """
        packed = self._unpack_batch(self._maybe_unroll(batch))
        if hasattr(packed, "events"):
            pred = self.forward_packet(packed)
            return pred, packed.target, packed.betas, packed
        x, prevpos, y, betas, camera_K = packed
        return self(x, prevpos, betas=betas, camera_K=camera_K), y, betas, packed

    def _maybe_distill(self, pred, batch, loss, parts):
        if self.distill_weight <= 0.0 or self.teacher is None:
            return loss, parts
        if isinstance(batch, (tuple, list)) and len(batch) == 2 and hasattr(batch[1], "events"):
            batch = batch[1]          # a paired sample distils on its main window
        packed = batch if hasattr(batch, "events") else None
        if packed is None or packed.lnes is None:
            return loss, parts
        with torch.no_grad():
            teacher_out = self.teacher(
                packed.lnes.to(pred.device), packed.prev_state.to(pred.device),
                betas=packed.betas.to(pred.device), camera_K=packed.camera_K.to(pred.device),
            )
        parts["loss_distill"] = F.mse_loss(pred, teacher_out)
        return loss + self.distill_weight * parts["loss_distill"], parts

    #: prefix of the frozen distillation teacher's parameters
    TEACHER_PREFIX = "teacher."

    def on_save_checkpoint(self, checkpoint):
        """Leave the frozen teacher out of the student's checkpoint.

        The teacher is a submodule so that Lightning moves it to the right device, which also puts
        its 11.2 M frozen parameters into `state_dict`. Keeping them would triple the file and,
        worse, make every downstream loader (`select_checkpoint.py`, `eval_track.py`, the S9/S17
        tools) fail on unexpected keys unless it happened to rebuild a teacher too.
        """
        sd = checkpoint.get("state_dict")
        if sd is not None:
            for k in [k for k in sd if k.startswith(self.TEACHER_PREFIX)]:
                del sd[k]

    def on_load_checkpoint(self, checkpoint):
        """Accept checkpoints written before `on_save_checkpoint` existed."""
        sd = checkpoint.get("state_dict")
        if sd is not None:
            for k in [k for k in sd if k.startswith(self.TEACHER_PREFIX)]:
                del sd[k]
            # S27 briefly carried a `gain_u` power-iteration buffer. The arm it belonged to was
            # falsified and the buffer is gone, but the S27/S28 grids were written while it
            # existed, and Lightning loads strictly. Dropping the key keeps those checkpoints
            # loadable; nothing reads it.
            sd.pop("gain_u", None)

    def training_step(self, batch, batch_nb):
        pred, y, betas, packed = self._predict_batch(batch)
        loss, parts = self._compute_loss(pred, y, betas)
        loss, parts = self._maybe_distill(pred, batch, loss, parts)
        if self.gain_reg_w > 0.0:
            g2 = self._gain_penalty(packed, pred)
            loss = loss + self.gain_reg_w * g2
            # Log the retention itself, not its square: the pre-registered mechanism gate is that
            # it falls monotonically with the penalty weight, and that has to be readable here
            # rather than only in a post-hoc probe.
            self.log("train_gain_rand", g2.detach().clamp_min(0).sqrt(), prog_bar=True,
                     sync_dist=True)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_mano_loss", parts["mano_loss"], sync_dist=True)
        self.log("train_pos_loss", parts["pos_loss"], sync_dist=True)
        self.log("train_rot_loss", parts["rot_loss"], sync_dist=True)
        if "loss_distill" in parts:
            self.log("train_loss_distill", parts["loss_distill"], sync_dist=True)
        self._log_so3_fk_parts("train", parts)
        # Under AMP/bf16, log10 must run in fp32 to avoid illegal CUDA engines.
        loss_f = loss.float()
        out = loss_f.log10() if self.log10_loss else loss_f
        return out

    def validation_step(self, batch, batch_nb):
        pred, y, betas, packed = self._predict_batch(batch)
        loss, parts = self._compute_loss(pred, y, betas)
        if self.gain_reg_w > 0.0:
            self.log("val_gain_rand",
                     self._gain_penalty(packed, pred).detach().clamp_min(0).sqrt(),
                     sync_dist=True, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("val_mano_loss", parts["mano_loss"], sync_dist=True, on_epoch=True)
        self.log("val_pos_loss", parts["pos_loss"], sync_dist=True, on_epoch=True)
        self.log("val_rot_loss", parts["rot_loss"], sync_dist=True, on_epoch=True)
        self._log_so3_fk_parts("val", parts, on_epoch=True)
        return loss

    def _log_so3_fk_parts(self, stage, parts, on_epoch=False):
        """Named logs for the SO(3)+FK terms; the aliases above are too opaque to read."""
        if self.loss_type != "so3_trans_fk":
            # S42: the mse arm can carry the absolute-FK regulariser; its trajectory is
            # the readable evidence for whether the term is doing anything.
            if "loss_abs_fk" in parts:
                self.log(f"{stage}_loss_abs_fk", parts["loss_abs_fk"],
                         sync_dist=True, on_epoch=on_epoch)
                self.log(f"{stage}_abs_joint_error_mm", parts["abs_joint_error_mm"],
                         sync_dist=True, on_epoch=on_epoch)
            return
        for key in ("loss_rot", "loss_trans", "loss_fk", "legacy_loss_51d",
                    "rotation_error_deg", "joint_error_mm"):
            self.log(f"{stage}_{key}", parts[key], sync_dist=True, on_epoch=on_epoch)

    def on_validation_epoch_end(self):
        gc.collect()

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    #: `TRAIN.LR_SCHEDULE` values. `constant` is the historical warmup-then-flat path.
    LR_SCHEDULES = ("constant", "cosine")

    def configure_optimizers(self):
        tcfg = self.cfg.get("TRAIN", {})
        lr = float(tcfg.get("LR", 1e-3))
        opt = optim.Adam(self.parameters(), lr=lr)
        warmup = int(tcfg.get("WARMUP_STEPS", 0))
        schedule = str(tcfg.get("LR_SCHEDULE", "constant"))
        if schedule not in self.LR_SCHEDULES:
            raise ValueError(f"TRAIN.LR_SCHEDULE must be one of {list(self.LR_SCHEDULES)}, "
                             f"got {schedule!r}")
        total = int(tcfg.get("MAX_STEPS", 0))
        if schedule == "cosine" and total <= warmup:
            raise ValueError("TRAIN.LR_SCHEDULE=cosine needs MAX_STEPS > WARMUP_STEPS")
        if warmup <= 0 and schedule == "constant":
            return opt

        # The decay exists because the checkpoint grid says the run never settles. Held at
        # 4e-3 for all 6000 steps, the control recipe's own grid reads
        # 17.55 / 17.96 / 18.80 / 21.16 / 90.93 / 18.26 / 17.80 / 132.62 / 17.66 / ... --
        # two outright divergences and a 1.1 mm spread among the survivors, so the reported number
        # is the minimum of a noisy trajectory rather than where training converged. Four runs of
        # that recipe give 16.638 / 17.744 / 17.496 / 17.657 (sd 0.51 mm), which is the resolution
        # any claim has to clear.
        def lr_lambda(step):
            if warmup > 0 and step < warmup:
                return float(step + 1) / float(warmup)
            if schedule == "constant":
                return 1.0
            p = min(max((step - warmup) / float(total - warmup), 0.0), 1.0)
            return self.LR_FLOOR + (1.0 - self.LR_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * p))

        sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }


class ReturnPrevposModel(BaseModel):
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__(cfg)
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, prevpos):
        return prevpos + 0 * self.dummy


class MNISTModel(BaseModel):
    """ResNet18 absolute-pose regressor (name kept for checkpoint compatibility).

    Tracking mode (MODEL.PREDICT_DELTA): network output is a delta on top of
    prevpos (state at the LNES window start); forward returns prevpos + delta,
    so the output layout/loss stay identical to the absolute mode.

    Render-and-compare (MODEL.PREV_RENDER): prev pose is rasterized and the
    resulting faces are concatenated with LNES. MODEL.RENDER_CHANNELS picks which
    faces to rasterize (default `[sil, inv]`, i.e. the published 4-channel input):

      sil    (1ch) binary silhouette
      inv    (1ch) normalized inverse depth, defined inside the silhouette
      semsil (1ch) the silhouette with kinematic semantics written into its value
                   instead of a flat 1: see :meth:`_register_vertex_codes`. A
                   drop-in replacement for `sil` at the same channel count.

    Render-free conditioning (MODEL.PREV_FK_DIRECT, S37, raw-event line only):
    the previous state still goes through the MANO forward pass, but instead of
    rasterizing it into images and sampling those at event pixels, each event
    node gets analytic distances to the *projected* FK output directly -- see
    :meth:`_fk_extra`. Mutually exclusive with PREV_RENDER by construction so an
    arm is attributable to exactly one conditioning path.

    The whole hand-model path is confined to the rasterizer: the trunk stays a
    plain conv1 adapter followed by an unmodified resnet18.
    """

    #: rasterizable faces and their channel width
    RENDER_FACES = {"sil": 1, "inv": 1, "semsil": 1}
    #: every MODEL key the model or the training script understands
    MODEL_KEYS = frozenset(
        {
            "BACKBONE", "POSE_REPR", "OUTPUT_DIM", "MANO_NCOMPS", "PREDICT_BETA",
            "PREDICT_DELTA", "PREVPOS_EMBED", "PREV_RENDER", "PREV_FK_DIRECT",
            "FK_DIRECTIONAL", "ZERO_EVENT_GATE",
            "RENDER_CHANNELS", "RENDER_H", "RENDER_W", "RENDER_SCALE",
            "RENDER_CHUNK", "RENDER_DEPTH_TOL", "INIT_FROM",
            "ACTIVE_HEAD", "ACTIVE_FEAT_DIM", "ACTIVE_HIDDEN",
            "ENCODER", "ENCODER_HIDDEN", "ENCODER_FEAT", "ENCODER_CELL", "ENCODER_LAYERS",
            "ENCODER_K", "ENCODER_MAX_NODES", "ENCODER_WINDOW", "ENCODER_T_SCALE",
            "ENCODER_JOINT_READOUT", "ENCODER_ATTN_POOL",
            "DISTILL_WEIGHT", "DISTILL_CKPT",
        }
    )
    #: every TRACK key the model or the dataset understands. Whitelisted for the same reason
    #: MODEL keys are: `UNROLL_PAIR` / `UNROLL_P` / `UNROLL_RAMP` were once read by nothing, and
    #: two arms trained to completion as silent copies of the arm they were meant to differ from.
    TRACK_KEYS = frozenset(
        {
            "PREV_NOISE_T", "PREV_NOISE_R", "PREV_NOISE_POSE", "PREV_NOISE_MODE",
            "PREV_NOISE_LARGE_T", "PREV_NOISE_LARGE_R", "PREV_NOISE_LARGE_POSE",
            "PREV_NOISE_P_SMALL", "PREV_NOISE_P_LARGE", "PREV_NOISE_P_CORR",
            "UNROLL_PAIR", "UNROLL_P", "UNROLL_RAMP", "UNROLL_RESIDUAL",
            "GAIN_REG_W", "GAIN_REG_SCALE",
        }
    )
    #: floor of the semsil value inside the mask, so the mask stays readable as a
    #: binary support (the background gap is this large, vs a ~2e-3 bf16 step)
    SEMSIL_FLOOR = 0.35
    #: S37 render-free conditioning. Decay lengths of the analytic proximity
    #: channels, in render pixels. 8 px against a ~2.5 px projected vertex
    #: spacing makes `g_surf` read ~1 inside the silhouette and fall to noise a
    #: couple of vertex spacings outside it, which is the soft version of the
    #: binary `sil` face; joints sit ~15 px apart so their kernel is twice as wide.
    FK_TAU_SURF = 8.0
    FK_TAU_SKEL = 16.0
    #: the depth channel takes the min z over this many nearest projected
    #: vertices: a 2D-nearest vertex may belong to the *back* surface, and the
    #: min over a small neighbourhood is the z-buffer's front-surface answer
    #: without a z-buffer.
    FK_DEPTH_K = 8
    #: chunk of sampled nodes per (nodes x 778) distance block, bounding the
    #: transient at ~400 MB where the full 1M-node batch would need ~13 GB
    FK_CHUNK = 32768

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, num_classes: Optional[int] = None):
        super().__init__(cfg)
        if num_classes is not None:
            self.output_dim = int(num_classes)
        model_cfg = self.cfg.get("MODEL", {})
        unknown = sorted(set(model_cfg) - self.MODEL_KEYS)
        if unknown:
            # A silently ignored MODEL key means a config-only "ablation" that never
            # reached the network, so refuse to build instead of running the default.
            raise ValueError(f"unknown MODEL keys: {unknown}")
        track_cfg = self.cfg.get("TRACK", {}) or {}
        unknown = sorted(set(track_cfg) - self.TRACK_KEYS)
        if unknown:
            raise ValueError(f"unknown TRACK keys: {unknown}")
        # S22. Probability that a paired sample's main window is re-conditioned on the model's own
        # prediction for the window before it, annealed over `UNROLL_RAMP = [hold, full]`. The ramp
        # is not cosmetic: at a constant 0.5 from step 0 half the conditioning states come from an
        # untrained network, and the measured optimum is to stop reading the conditioning state at
        # all (teacher-forced RA 42.5 mm against 12.1 without the mixing).
        self.unroll_p = float(track_cfg.get("UNROLL_P", 0.0))
        ramp = track_cfg.get("UNROLL_RAMP") or (0, 0)
        self.unroll_ramp = (int(ramp[0]), int(ramp[1]))
        # E5.5c: add the lead prediction's *error* on top of the noised state instead of replacing
        # the state with the prediction. Both replacement variants failed with one root cause: the
        # lead prediction is near-clean (1-2 mm), so replacement teaches "prev is trustworthy".
        # Constant-P collapsed outright (TF 42.5 mm); annealed kept TF but pushed the recursive
        # echo excess back from x1.85-1.95 to x2.22-2.26 (probe 2026-08-25 21:40) because half the
        # samples lost the large-error curriculum. The residual form keeps GT + curriculum noise
        # and injects the self-error's correlated *structure* on top.
        self.unroll_residual = bool(track_cfg.get("UNROLL_RESIDUAL", False))
        # Retention penalty. Measured across 11 checkpoints and two seeds, the recursive error is
        # ranked by how much of a conditioning-state error survives one step (Spearman +0.93/+1.00)
        # and *anti*-ranked by single-step accuracy (-0.61/-1.00): `s22`'s unroll arm has the best
        # teacher-forced error in the set and the worst tracking. Retention spans 0.215-0.693 across
        # recipes and reproduces to within 0.05 across seeds, so it is a stable property that no
        # recipe has ever optimised on purpose. `GAIN_REG_W` optimises it directly.
        self.gain_reg_w = float(track_cfg.get("GAIN_REG_W", 0.0))
        self.gain_reg_scale = float(track_cfg.get("GAIN_REG_SCALE", 1.0))
        self.predict_delta = bool(model_cfg.get("PREDICT_DELTA", False))
        self.prevpos_embed = bool(model_cfg.get("PREVPOS_EMBED", False))
        self.prev_render = bool(model_cfg.get("PREV_RENDER", False))
        self.prev_fk = bool(model_cfg.get("PREV_FK_DIRECT", False))
        if self.prev_fk and self.prev_render:
            # One conditioning path per arm, or a difference is not attributable.
            raise ValueError("PREV_FK_DIRECT and PREV_RENDER are mutually exclusive")
        # S38. The pooled vector stays for the root head; the fifteen joint heads read a
        # per-joint spatial-kernel readout centred on the previous state's projected LBS
        # joints instead. The queries reuse the PREV_FK_DIRECT forward pass, so that path
        # is a prerequisite rather than an independent switch.
        self.joint_readout = bool(model_cfg.get("ENCODER_JOINT_READOUT", False))
        if self.joint_readout and not self.prev_fk:
            raise ValueError("ENCODER_JOINT_READOUT needs MODEL.PREV_FK_DIRECT "
                             "(the queries are the projected FK of the previous state)")
        # S40. Two more FK-direct channels: the unit direction from the event to the nearest
        # projected vertex, gated by g_surf. `g_surf` is the magnitude of the distance field's
        # local description; this is its direction -- the raster `sil` had neither.
        self.fk_directional = bool(model_cfg.get("FK_DIRECTIONAL", False))
        if self.fk_directional and not self.prev_fk:
            raise ValueError("FK_DIRECTIONAL extends the PREV_FK_DIRECT channels")
        # S41. Content-based attention pooling in the frontend readout. State-free by
        # construction (the queries are learned constants), so it is on the right side of
        # the S38/KEG law; incompatible with the retired state-centred joint readout so
        # every arm has exactly one readout mechanism.
        self.attn_pool = int(model_cfg.get("ENCODER_ATTN_POOL", 0))
        if self.attn_pool and self.joint_readout:
            raise ValueError("ENCODER_ATTN_POOL and ENCODER_JOINT_READOUT are exclusive")
        self.zero_event_gate = bool(model_cfg.get("ZERO_EVENT_GATE", self.prev_render))
        self.render_channels = self._parse_render_channels(model_cfg)
        # Two planes per event face. `DATA.EVENT_CHANNELS` is unset in every existing config, which
        # resolves to `("last",)` -- plain LNES, two planes, the historical width.
        self.event_channels = EV.event_channels(self.cfg)
        in_ch = (2 * len(self.event_channels)
                 + sum(self.RENDER_FACES[c] for c in self.render_channels))
        # Keep parameter names conv1 / rn for loading original 12D checkpoints.
        self.conv1 = nn.Conv2d(in_ch, 3, kernel_size=3, padding=1)

        # S10. The active head, and specifically the one design the repository's own history says
        # is required. A zero-initialised additive per-joint bypass alongside a direct 51D
        # regression was measured to be lazy: the direct path takes the gradient and the bypass
        # contributes 0.16 mm, indistinguishable from noise. So here the trunk does not emit a pose
        # at all. It emits a feature vector; a small head reads root translation and rotation off
        # it; and each of the fifteen joints has its own decoder which is the *sole* pathway to that
        # joint's local pose. There is no direct route for the trunk to write a finger angle, so the
        # per-joint decoders cannot be ignored, and zeroing them at inference on the same checkpoint
        # is a decisive non-laziness check rather than a suggestive one.
        self.active_head = bool(model_cfg.get("ACTIVE_HEAD", False))
        #: set at inference to zero every joint decoder, for the same-checkpoint ablation
        self.ablate_joint_heads = False
        self.encoder_name = str(model_cfg.get("ENCODER") or "").lower()
        if self.encoder_name in ("", "none", "lnes"):
            self.encoder_name = ""
        self.distill_weight = float(model_cfg.get("DISTILL_WEIGHT", 0.0))
        self.teacher = None
        hid = int(model_cfg.get("ACTIVE_HIDDEN", 64))
        if self.encoder_name:
            # S2 / S16. The dense conv trunk is not built: a raw-event encoder that still
            # allocated a resnet would be LNES with extra copies, and INIT_FROM would silently
            # load the wrong weights.
            from semkine.frontends import build_frontend
            feat = int(model_cfg.get("ENCODER_FEAT", model_cfg.get("ACTIVE_FEAT_DIM", 256)))
            # The previous state reaches the frontend the way it does in the arm that actually
            # tracks: two rendered values sampled at each event's own pixel. Comparative, not
            # indexing -- the event's position in the representation does not move with the pose.
            # S37 keeps that contract but computes four analytic values instead of sampling a
            # rasterized image; see `_fk_extra`.
            extra = 2 if self.prev_render else (
                (6 if self.fk_directional else 4) if self.prev_fk else 0)
            self.event_encoder = build_frontend(
                self.encoder_name,
                height=int(self.cfg.get("DATA", {}).get("HEIGHT", 180)),
                width=int(self.cfg.get("DATA", {}).get("WIDTH", 240)),
                hidden=int(model_cfg.get("ENCODER_HIDDEN", 128)),
                feat_dim=feat,
                extra_channels=extra,
                cell=int(model_cfg.get("ENCODER_CELL", 16)),
                n_layers=int(model_cfg.get("ENCODER_LAYERS", 2)),
                # S36 `event_gnn`. The events are the nodes, so the cost knobs are how many of them
                # survive per packet, how far back the causal neighbour search looks, and how many
                # edges each node keeps. `max_nodes` is what holds memory flat as the event rate
                # rises -- the property the retired `aegnn_lite` did not have.
                k=int(model_cfg.get("ENCODER_K", 8)),
                max_nodes=int(model_cfg.get("ENCODER_MAX_NODES", 768)),
                window=int(model_cfg.get("ENCODER_WINDOW", 32)),
                t_scale=float(model_cfg.get("ENCODER_T_SCALE", 1.0)),
                joint_queries=16 if self.joint_readout else 0,
                attn_pool=self.attn_pool,
            )
            self.conv1 = None
            self.rn = None
            if self.active_head:
                assert self.output_dim == 51, "the active head is defined for the 51D layout"
                self.root_head = nn.Linear(feat, 6)
                # S38: with the joint readout each decoder reads its own joint's local
                # feature (`hidden`-wide), and that is its *only* evidence -- the S10/KSGN
                # law that a bypass which can be ignored will be ignored, applied to the
                # readout. Without it, every decoder reads the shared pooled vector.
                head_in = (int(model_cfg.get("ENCODER_HIDDEN", 128))
                           if self.joint_readout else feat)
                self.joint_heads = nn.ModuleList([
                    nn.Sequential(nn.Linear(head_in + 3, hid), nn.ReLU(inplace=True),
                                  nn.Linear(hid, 3))
                    for _ in range(15)
                ])
            else:
                self.pose_head = nn.Linear(feat, self.output_dim)
        elif self.active_head:
            assert self.output_dim == 51, "the active head is defined for the 51D layout"
            feat = int(model_cfg.get("ACTIVE_FEAT_DIM", 256))
            self.rn = models.resnet18(num_classes=feat)
            self.root_head = nn.Linear(feat, 6)
            # One decoder per joint, each conditioned on the shared feature and on that joint's own
            # previous angle. Separate parameters rather than one grouped layer: the point is that a
            # joint's update can be produced, and later suppressed, independently of the others.
            self.joint_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(feat + 3, hid), nn.ReLU(inplace=True),
                              nn.Linear(hid, 3))
                for _ in range(15)
            ])
        else:
            self.rn = models.resnet18(num_classes=self.output_dim)
        if self.prevpos_embed:
            self.prev_mlp = nn.Sequential(
                nn.Linear(self.output_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.output_dim),
            )
            # Zero-init last layer: at start the delta comes purely from events.
            nn.init.zeros_(self.prev_mlp[2].weight)
            nn.init.zeros_(self.prev_mlp[2].bias)

        if self.prev_fk and not self.encoder_name:
            raise ValueError("PREV_FK_DIRECT conditions event nodes; it needs MODEL.ENCODER")
        self._ctx_betas = None
        self._ctx_K = None
        if self.prev_render or self.prev_fk:
            from mano_layer import ManoLayer

            mano_npz = self.cfg.get("MANO", {}).get("NPZ", "assets/mano_right.npz")
            self.mano = ManoLayer(mano_npz, add_mean=False)
            self.mano.eval()
            self.render_h = int(model_cfg.get("RENDER_H", self.cfg.get("DATA", {}).get("HEIGHT", 180)))
            self.render_w = int(model_cfg.get("RENDER_W", self.cfg.get("DATA", {}).get("WIDTH", 240)))
            self.render_scale = float(model_cfg.get("RENDER_SCALE", 240.0 / 640.0))
            self.render_chunk = int(model_cfg.get("RENDER_CHUNK", 256))
            self.render_depth_tol = float(model_cfg.get("RENDER_DEPTH_TOL", 5.0e-3))
        if self.prev_render:
            self._register_vertex_codes()
        if self.joint_readout:
            # Column j of the MANO skinning matrix says which vertices joint j drives; its
            # normalised transpose turns posed vertices into per-joint surface centroids, in
            # exactly the order the fifteen decoders and `prev[6+3k : 9+3k]` already use.
            w = self.mano.weights  # (778, 16)
            self.register_buffer(
                "lbs_wn", (w / w.sum(0, keepdim=True).clamp(min=1e-8)).T.contiguous(),
                persistent=False)

    def _register_vertex_codes(self):
        """Per-vertex semantic constant the rasterizer can splat into the mask value.

        MANO skins vertex i as v_i = sum_j W_ij G_j(theta) v_i, so row i of the
        778x16 weight matrix is the complete answer to "which internal joints drive
        this vertex". Contracting it with the rest-pose joints gives a canonical
        3-vector per vertex; taking its distance to the wrist collapses that to one
        continuous scalar -- a radial skeleton coordinate saying how far down the
        kinematic chain the vertex sits. One scalar is what fits in the mask channel,
        and unlike an argmax part ID it stays continuous across part boundaries
        because the skinning weights themselves blend. Rest-pose joints (betas = 0)
        are deliberate: this is a canonical label and must not move with hand shape.
        """
        with torch.no_grad():
            w = self.mano.weights  # (778, 16)
            j_rest = self.mano.J_regressor @ self.mano.v_template  # (16, 3)
            anchored = w @ j_rest  # (778, 3) LBS-weighted canonical joint position
            radial = (anchored - j_rest[0:1]).norm(dim=-1, keepdim=True)  # (778, 1)
            lo, hi = radial.amin(), radial.amax()
            code = (radial - lo) / (hi - lo).clamp(min=1e-8)
        self.register_buffer("vert_semcode", code.clone(), persistent=False)

    @classmethod
    def _parse_render_channels(cls, model_cfg) -> tuple:
        if not bool(model_cfg.get("PREV_RENDER", False)):
            return ()
        names = model_cfg.get("RENDER_CHANNELS", ["sil", "inv"])
        if isinstance(names, str):
            names = [names]
        names = tuple(str(n) for n in names)
        bad = [n for n in names if n not in cls.RENDER_FACES]
        if bad or len(set(names)) != len(names) or not names:
            raise ValueError(
                f"RENDER_CHANNELS must be a non-empty set of "
                f"{sorted(cls.RENDER_FACES)}, got {list(names)}"
            )
        return names

    def set_hand_context(self, betas, camera_K):
        """Store per-sequence betas / K for recursive eval (batch size 1)."""
        if betas is None:
            self._ctx_betas = None
        else:
            t = betas if torch.is_tensor(betas) else torch.as_tensor(betas, dtype=torch.float32)
            self._ctx_betas = t.detach().view(1, -1)
        if camera_K is None:
            self._ctx_K = None
        else:
            t = camera_K if torch.is_tensor(camera_K) else torch.as_tensor(camera_K, dtype=torch.float32)
            self._ctx_K = t.detach().view(1, 3, 3)

    def init_from_abs_checkpoint(self, ckpt_path: str) -> int:
        """Warm-start from an absolute-pose ckpt; extra conv1 channels stay zero."""
        state = torch.load(ckpt_path, map_location="cpu")
        return self.init_from_abs_checkpoint_state(state.get("state_dict", state))

    def init_from_abs_checkpoint_state(self, sd) -> int:
        own = self.state_dict()
        loaded = 0
        for k, v in sd.items():
            if k not in own:
                continue
            if k == "conv1.weight" and own[k].shape != v.shape:
                if own[k].shape[0] == v.shape[0] and own[k].shape[2:] == v.shape[2:] and own[k].shape[1] >= v.shape[1]:
                    own[k].zero_()
                    own[k][:, : v.shape[1]].copy_(v)
                    loaded += 1
                continue
            if own[k].shape == v.shape:
                own[k].copy_(v)
                loaded += 1
        self.load_state_dict(own)
        return loaded

    def _resolve_betas_K(self, prevpos, betas, camera_K):
        B = prevpos.shape[0]
        device = prevpos.device
        if betas is None:
            if self._ctx_betas is None:
                betas = torch.zeros(B, 10, device=device, dtype=torch.float32)
            else:
                betas = self._ctx_betas.to(device=device, dtype=torch.float32).expand(B, -1)
        else:
            betas = betas.to(device=device, dtype=torch.float32)
            if betas.dim() == 1:
                betas = betas.view(1, -1).expand(B, -1)
        if camera_K is None:
            if self._ctx_K is None:
                camera_K = torch.tensor(
                    [[603.4507, 0.0, 325.09183], [0.0, 602.95654, 242.09796], [0.0, 0.0, 1.0]],
                    device=device,
                    dtype=torch.float32,
                ).expand(B, -1, -1)
            else:
                camera_K = self._ctx_K.to(device=device, dtype=torch.float32).expand(B, -1, -1)
        else:
            camera_K = camera_K.to(device=device, dtype=torch.float32)
            if camera_K.dim() == 2:
                camera_K = camera_K.unsqueeze(0).expand(B, -1, -1)
        return betas, camera_K

    def _intrinsics(self, camera_K):
        """Render-space pinhole intrinsics (shared by verts and joints)."""
        s = self.render_scale
        return (
            camera_K[:, 0, 0] * s,
            camera_K[:, 1, 1] * s,
            camera_K[:, 0, 2] * s,
            camera_K[:, 1, 2] * s,
        )

    def _fk(self, params, betas):
        """Differentiable MANO forward: params -> (verts, 21 OpenPose joints)."""
        dec = decode_to_mano_inputs(
            params, self.pose_repr, self.mano.hands_components, self.mano.hands_mean
        )
        return self.mano(betas, dec["global_orient"], dec["local_full_aa"], dec["transl"])

    def _render_chunk(self, prevpos, betas, camera_K):
        """Rasterize the previous state into the requested faces; (B, H, W, C)."""
        verts, _ = self._fk(prevpos, betas)
        fx, fy, cx, cy = self._intrinsics(camera_K)
        x, y, z = verts.unbind(-1)
        z_safe = z.clamp(min=1e-6)
        u = fx[:, None] * x / z_safe + cx[:, None]
        v = fy[:, None] * y / z_safe + cy[:, None]
        ui = u.round().long()
        vi = v.round().long()
        B, n_v = ui.shape
        h, w = self.render_h, self.render_w
        valid = (z > 1e-6) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        b_idx = torch.arange(B, device=verts.device)[:, None].expand(B, n_v)
        pix = (b_idx * (h * w) + vi * w + ui)[valid]
        z_valid = z_safe[valid]
        depth_flat = torch.full((B * h * w,), 1.0e6, device=verts.device, dtype=torch.float32)
        sil_flat = torch.zeros((B * h * w,), device=verts.device, dtype=torch.float32)
        if pix.numel() > 0:
            depth_flat.scatter_reduce_(0, pix, z_valid, reduce="amin", include_self=True)
            sil_flat.scatter_(0, pix, torch.ones_like(z_valid))
        sil = sil_flat.view(B, h, w).clamp(0, 1)

        faces = {}
        if "sil" in self.render_channels:
            faces["sil"] = sil.unsqueeze(-1)
        if "inv" in self.render_channels:
            faces["inv"] = (
                ((1.0 / depth_flat.view(B, h, w)).clamp(0.0, 5.0) / 5.0) * sil
            ).unsqueeze(-1)
        if "semsil" in self.render_channels:
            # Average the per-vertex code over the vertices that survive the same z
            # test the depth buffer already computed, so each pixel carries the
            # semantics of the surface actually facing the camera. The floor keeps the
            # occupancy readable: any masked pixel is >= SEMSIL_FLOOR, so a threshold
            # recovers exactly `sil` while the value adds the kinematic coordinate.
            keep = z_valid <= depth_flat[pix] + self.render_depth_tol
            acc = torch.zeros((B * h * w, 1), device=verts.device, dtype=torch.float32)
            cnt = torch.zeros((B * h * w, 1), device=verts.device, dtype=torch.float32)
            if keep.any():
                pk = pix[keep]
                vk = self.vert_semcode.expand(B, -1, -1)[valid][keep]
                acc.index_add_(0, pk, vk)
                cnt.index_add_(0, pk, torch.ones_like(pk, dtype=torch.float32).unsqueeze(-1))
            code_img = (acc / cnt.clamp(min=1.0)).view(B, h, w)
            faces["semsil"] = (
                (self.SEMSIL_FLOOR + (1.0 - self.SEMSIL_FLOOR) * code_img) * sil
            ).unsqueeze(-1)

        return torch.cat([faces[c] for c in self.render_channels], dim=-1)

    def _render_prev(self, prevpos, betas, camera_K):
        B = prevpos.shape[0]
        chunk = max(int(self.render_chunk), 1)
        out = [
            self._render_chunk(
                prevpos[i0 : i0 + chunk], betas[i0 : i0 + chunk], camera_K[i0 : i0 + chunk]
            )
            for i0 in range(0, B, chunk)
        ]
        return torch.cat(out, dim=0)

    def _joint_queries(self, verts, camera_K):
        """S38. Pixel positions of the previous state's 16 LBS joints; (B, 16, 2).

        Surface centroids under the skinning weights rather than the 21 OpenPose joints the
        MANO layer returns: the centroids are indexed by the same 16 LBS columns the decoders
        are ordered by, so joint head `k` and query `k + 1` agree by construction instead of
        by a hand-written joint-order mapping.
        """
        cent = torch.einsum("jv,bvc->bjc", self.lbs_wn, verts.float())
        fx, fy, cx, cy = self._intrinsics(camera_K)
        x, y, z = cent.unbind(-1)
        z = z.clamp(min=1e-6)
        return torch.stack([fx[:, None] * x / z + cx[:, None],
                            fy[:, None] * y / z + cy[:, None]], dim=-1)

    def _fk_extra(self, events, ptr, verts, joints, camera_K):
        """S37. The previous state's MANO FK corresponds to the events directly, no raster.

        The render path answers two questions at each event's pixel -- "is the predicted hand
        here" (`sil`) and "how deep is its front surface here" (`inv`) -- by drawing 778
        projected vertices into an image and reading the image back at N pixels. This method
        answers the same questions by measuring each event against the projected FK output
        itself, which removes the z-buffer scatter and the (B, H, W, C) materialisation:

          g_surf  exp(-d_vert / FK_TAU_SURF): distance to the nearest projected vertex, as a
                  soft occupancy. Inside the silhouette a vertex projects within ~2.5 px of
                  every pixel, so this reads ~1 there and decays outside -- `sil`, made soft,
                  plus the gradient direction `sil` never had.
          z_surf  min z over the FK_DEPTH_K nearest vertices, normalised like `inv` and gated
                  by `g_surf`. The min stands in for the z-buffer: the 2D-nearest vertex may
                  be on the back of the hand, the min over a small 2D neighbourhood is the
                  front surface.
          g_skel / z_skel  the same pair against the 21 projected joints, a coarser skeleton
                  proximity the raster never provided.

        Contract notes, both load-bearing:
          * State enters as *values* on the nodes, never as structure -- same design law the
            render channels obey (KEG's state-routed graph is the measured counterexample).
          * Features are only computed for the <= `max_nodes` events the encoder will actually
            gather (`_sample` is deterministic, so calling it here and inside the encoder gives
            the same rows). Computing against all events would be an (N x 778) matrix with N in
            the tens of millions per training batch. Rows the encoder never reads stay zero,
            and the sampled indices are pairwise distinct, so the scatter is deterministic.

        Takes the FK products rather than the raw state so a caller that also needs the
        S38 queries pays for one MANO forward, not two.

        With `MODEL.FK_DIRECTIONAL` (S40) two more channels follow: the unit direction from
        the event to its nearest projected vertex, gated by `g_surf`. `g_surf` is the local
        magnitude of the distance field; the pair is its direction, so together they are the
        field's first-order description at the event -- which way the predicted surface lies,
        not only how far. Divisor clamped at 1 px: inside the silhouette the direction is
        noise, and the clamp fades it to zero smoothly instead of normalising the noise up.

        Returns `(N, 4)` float32 aligned with `events`, `(N, 6)` under FK_DIRECTIONAL.
        """
        n_ev = int(events.shape[0])
        width = 6 if self.fk_directional else 4
        out = torch.zeros(n_ev, width, device=events.device, dtype=torch.float32)
        if n_ev == 0:
            return out
        if not hasattr(self.event_encoder, "_sample"):
            raise RuntimeError("PREV_FK_DIRECT needs a frontend with node subsampling")
        src, mask = self.event_encoder._sample(events, ptr)
        keep = mask.reshape(-1)
        flat = src.reshape(-1)[keep]
        if flat.numel() == 0:
            return out
        B = int(ptr.numel() - 1)
        b_idx = (torch.arange(B, device=events.device)
                 .unsqueeze(1).expand_as(src).reshape(-1)[keep])

        fx, fy, cx, cy = self._intrinsics(camera_K)

        def project(pts):
            x, y, z = pts.float().unbind(-1)
            z = z.clamp(min=1e-6)
            return (fx[:, None] * x / z + cx[:, None],
                    fy[:, None] * y / z + cy[:, None], z)

        vu, vv, vz = project(verts)
        ju, jv, jz = project(joints)
        eu = events[flat, EV.EV_X].float()
        ev_y = events[flat, EV.EV_Y].float()

        chunks = []
        for i0 in range(0, int(flat.numel()), self.FK_CHUNK):
            sl = slice(i0, i0 + self.FK_CHUNK)
            b = b_idx[sl]
            d2v = ((vu[b] - eu[sl, None]).square()
                   + (vv[b] - ev_y[sl, None]).square())          # (m, 778)
            near = d2v.topk(self.FK_DEPTH_K, dim=-1, largest=False)
            d_surf = near.values[:, 0].clamp(min=0.0).sqrt()
            z_surf = vz[b].gather(1, near.indices).amin(dim=-1)
            d2j = ((ju[b] - eu[sl, None]).square()
                   + (jv[b] - ev_y[sl, None]).square())          # (m, 21)
            d2j_min, j_arg = d2j.min(dim=-1)
            d_skel = d2j_min.clamp(min=0.0).sqrt()
            z_skel = jz[b].gather(1, j_arg.unsqueeze(1)).squeeze(1)
            g_surf = torch.exp(-d_surf / self.FK_TAU_SURF)
            g_skel = torch.exp(-d_skel / self.FK_TAU_SKEL)
            inv_s = (1.0 / z_surf).clamp(0.0, 5.0) / 5.0
            inv_k = (1.0 / z_skel).clamp(0.0, 5.0) / 5.0
            cols = [g_surf, inv_s * g_surf, g_skel, inv_k * g_skel]
            if self.fk_directional:
                idx0 = near.indices[:, 0]
                du = vu[b].gather(1, idx0.unsqueeze(1)).squeeze(1) - eu[sl]
                dv = vv[b].gather(1, idx0.unsqueeze(1)).squeeze(1) - ev_y[sl]
                den = d_surf.clamp(min=1.0)
                cols += [du / den * g_surf, dv / den * g_surf]
            chunks.append(torch.stack(cols, dim=-1))
        out[flat] = torch.cat(chunks, dim=0)
        return out

    def _decode_active(self, feat, prevpos, nodes=None):
        """Assemble the 51D output from the root head and the fifteen joint decoders.

        `nodes` is `(B, 16, node_dim)` when the frontend emits one feature per MANO joint; joint
        `k` then reads node `k + 1`, since node 0 is the wrist. Otherwise every decoder reads the
        shared vector, which is the S10 arrangement.

        The fifteen decoders are executed as a Python loop on purpose. Batching them into one
        `baddbmm` computes the same math ~0.7 ms faster per recursive step, but the different
        accumulation order perturbs the output by ~1e-6, and the closed loop amplified that to
        0.15 mm of recursive RA on seed 3408 -- past the 0.05 mm reproduction gate that pins
        every recorded number (S36 latency audit, 2026-08-29). Bit-reproducibility wins.
        """
        root = self.root_head(feat)
        prev = prevpos.to(feat.dtype)
        cols = [root]
        for k, head in enumerate(self.joint_heads):
            if self.ablate_joint_heads:
                # The ablation is a genuine silence, not a scaled-down update: with PREDICT_DELTA
                # the output is `prev + out`, so zero here means "this joint does not move".
                cols.append(torch.zeros(feat.shape[0], 3, device=feat.device, dtype=feat.dtype))
                continue
            src = feat if nodes is None else nodes[:, k + 1].to(feat.dtype)
            cols.append(head(torch.cat([src, prev[:, 6 + 3 * k : 9 + 3 * k]], dim=-1)))
        return torch.cat(cols, dim=-1)

    def forward_packet(self, batch):
        """Raw-event forward for an `EventPacketBatch`. Empty packets return bitwise `prev`."""
        if not self.encoder_name:
            raise RuntimeError("forward_packet requires MODEL.ENCODER")
        events = batch.events
        ptr = batch.ptr
        prev = batch.prev_state
        extra = None
        if self.prev_render:
            betas_f, k_f = self._resolve_betas_K(prev, batch.betas, batch.camera_K)
            with torch.no_grad():
                rend = self._render_prev(prev.float(), betas_f, k_f)
            rend = rend.to(dtype=events.dtype, device=events.device)
            from semkine.encoder import query_render
            extra = query_render(rend, events)
        queries = None
        if self.prev_fk:
            betas_f, k_f = self._resolve_betas_K(prev, batch.betas, batch.camera_K)
            # `no_grad` for the same reason the render is: the state is conditioning,
            # not a differentiable pathway.
            with torch.no_grad():
                verts_p, joints_p = self._fk(prev.float(), betas_f)
                extra = self._fk_extra(events, ptr, verts_p, joints_p, k_f)
                if self.joint_readout:
                    queries = self._joint_queries(verts_p, k_f)
            extra = extra.to(dtype=events.dtype)
        nodes = None
        if queries is not None:
            feat, nodes = self.event_encoder(events, ptr, batch.delta_t_s, extra, queries)
        else:
            feat = self.event_encoder(events, ptr, batch.delta_t_s, extra)
        if self.active_head:
            out = self._decode_active(feat, prev, nodes)
        else:
            out = self.pose_head(feat)
        if self.prevpos_embed:
            out = out + self.prev_mlp(prev.to(out.dtype))
        if self.predict_delta:
            delta = out
            if self.zero_event_gate:
                empty = (batch.counts <= 0).unsqueeze(-1)
                delta = torch.where(empty, torch.zeros_like(delta), delta)
            out = delta + prev.to(out.dtype)
        return out

    def forward(self, x, prevpos, betas=None, camera_K=None):
        if self.encoder_name:
            raise RuntimeError("this model reads raw events; call forward_packet")
        lnes = x
        B = lnes.shape[0]
        if self.prev_render:
            betas_f, k_f = self._resolve_betas_K(prevpos, betas, camera_K)
            with torch.no_grad():
                rend = self._render_prev(prevpos.float(), betas_f, k_f)
            rend = rend.to(dtype=lnes.dtype, device=lnes.device)
            x = torch.cat([lnes, rend], dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        out = self.rn(self.conv1(x))
        if self.active_head:
            out = self._decode_active(out, prevpos)
        if self.prevpos_embed:
            out = out + self.prev_mlp(prevpos.to(out.dtype))
        if self.predict_delta:
            delta = out
            if self.zero_event_gate:
                empty = lnes.reshape(B, -1).abs().sum(dim=1, keepdim=True) <= 0
                delta = torch.where(empty, torch.zeros_like(delta), delta)
            out = delta + prevpos.to(out.dtype)
        return out
