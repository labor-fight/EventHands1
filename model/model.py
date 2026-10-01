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
        if not (self.prev_render or self.routed or self.mesh_query or self.fk_graph
                or getattr(self, "mesh_graph", False)):
            # `self.mano` only exists on the conditioning paths, and silently falling back to a
            # parameter-space ratio is the exact failure this docstring documents.
            raise ValueError("TRACK.GAIN_REG_W needs MODEL.PREV_RENDER, ROUTED_READOUT or "
                             "MESH_QUERY for the MANO forward pass")
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
        # region agent log
        if getattr(self, "xyz_mesh", False) and int(self.global_step) % 100 == 0:
            self._agent_last_loss = {"loss": float(loss.detach()), "mano": float(parts["mano_loss"].detach()),
                                     "rot": float(parts["rot_loss"].detach()), "pos": float(parts["pos_loss"].detach())}
        # endregion
        self.log("train_mano_loss", parts["mano_loss"], sync_dist=True)
        self.log("train_pos_loss", parts["pos_loss"], sync_dist=True)
        self.log("train_rot_loss", parts["rot_loss"], sync_dist=True)
        if "loss_distill" in parts:
            self.log("train_loss_distill", parts["loss_distill"], sync_dist=True)
        self._log_so3_fk_parts("train", parts)
        self._log_route_stats("train")
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
        self._log_route_stats("val", on_epoch=True)
        return loss

    def _log_route_stats(self, stage, on_epoch=False):
        """S37 mechanism readings: share of live nodes that reached the hand (within the band)
        and how many joints received evidence. Under the curriculum's large-noise mode the
        first is expected to collapse; that collapse is hypothesis H1 of the pre-registration."""
        if not (getattr(self, "routed", False) or getattr(self, "mesh_query", False)
                or getattr(self, "fk_graph", False) or getattr(self, "mesh_graph", False)
                or getattr(self, "xyz_mesh", False)):
            return
        for key, v in self.route_stats.items():
            self.log(f"{stage}_{key}", v, sync_dist=True, on_epoch=on_epoch)

    def _log_so3_fk_parts(self, stage, parts, on_epoch=False):
        """Named logs for the SO(3)+FK terms; the aliases above are too opaque to read."""
        if self.loss_type != "so3_trans_fk":
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

    # region agent log
    def on_before_optimizer_step(self, optimizer, *args):
        """Debug session 52fff1 (H6): per-module gradient norms every 100 optimizer steps, rank 0,
        only when `EH_AGENT_LOG_52FFF1` names the log file (set by that session's launches only)."""
        import os
        path = os.environ.get("EH_AGENT_LOG_52FFF1")
        trainer = getattr(self, "_trainer", None)
        if not path or trainer is None or int(trainer.global_rank) != 0 or int(self.global_step) % 100:
            return
        import json
        import time

        def gn(params):
            sq = [float(p.grad.detach().float().pow(2).sum()) for p in params if p.grad is not None]
            return math.sqrt(sum(sq)) if sq else None

        data = {"step": int(self.global_step)}
        rf = getattr(self, "root_fusion_head", None)
        if rf is not None and rf.w1.grad is not None:
            data["root_part"] = [math.sqrt(sum(float(p.grad[j].float().pow(2).sum())
                                               for p in (rf.w1, rf.b1, rf.w2, rf.b2)))
                                 for j in range(rf.n_joints)]
        if hasattr(self, "root_head"):
            data["root_head"] = gn(self.root_head.parameters())
        enc = getattr(self, "event_encoder", None)
        if enc is not None and hasattr(enc, "layers"):
            data["edgeconv"] = [gn(layer.parameters()) for layer in enc.layers]
            data["embed"] = gn(enc.embed.parameters())
            if getattr(enc, "readout", False):
                data["proj"] = gn(enc.proj.parameters())
        if hasattr(self, "joint_heads"):
            data["joint_heads"] = gn(self.joint_heads.parameters())
        if hasattr(self, "prev_mlp"):
            data["prev_mlp"] = gn(self.prev_mlp.parameters())
        m = trainer.callback_metrics
        data["train_loss"] = float(m["train_loss"]) if "train_loss" in m else None
        ts = int(time.time() * 1000)
        with open(path, "a") as f:
            f.write(json.dumps({"sessionId": "52fff1", "id": f"log_{ts}_H6", "timestamp": ts,
                                "location": "model.py:on_before_optimizer_step",
                                "message": "per-module grad norms (rank 0)", "data": data,
                                "runId": os.environ.get("EH_AGENT_RUN_52FFF1", "train"),
                                "hypothesisId": "H6"}) + "\n")
    # endregion

    def configure_optimizers(self):
        tcfg = self.cfg.get("TRAIN", {})
        lr = float(tcfg.get("LR", 1e-3))
        # Frozen parameters (TRAIN.TRAINABLE_PREFIXES) stay out of the optimiser; with nothing
        # frozen this is every parameter in the same order as before.
        opt = optim.Adam([p for p in self.parameters() if p.requires_grad], lr=lr)
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

    The whole hand-model path is confined to the rasterizer: the trunk stays a
    plain conv1 adapter followed by an unmodified resnet18.
    """

    #: rasterizable faces and their channel width
    RENDER_FACES = {"sil": 1, "inv": 1, "semsil": 1}
    #: every MODEL key the model or the training script understands
    MODEL_KEYS = frozenset(
        {
            "BACKBONE", "POSE_REPR", "OUTPUT_DIM", "MANO_NCOMPS", "PREDICT_BETA",
            "PREDICT_DELTA", "PREVPOS_EMBED", "PREV_RENDER", "ZERO_EVENT_GATE",
            "ROUTED_READOUT", "ROUTE_BAND_PX", "ROUTE_FRONT_K",
            # S37 root input fusion (docs/S37_EDGE6_ROOTFUSE_PREREG_20260929.md)
            "ROOT_FUSION",
            # S37 root innovation (docs/S37_ROOT_INNOVATION_PREREG.md)
            "ROOT_INNOVATION", "ROOT_INNOV_HIDDEN", "ROOT_INNOV_WIDTH", "PREV_MLP_ROOT",
            "MESH_QUERY", "MESH_QUERY_TOKENS", "MESH_QUERY_BAND_PX", "MESH_QUERY_SIGMA_PX",
            "ENCODER_NODE_ATTRS",
            "FK_GRAPH_VERTS", "FK_GRAPH_BAND_PX", "FK_GRAPH_K", "FK_GRAPH_NODE_ID",
            "MESH_GRAPH_BAND_PX", "MESH_GRAPH_FRONT_PX", "MESH_GRAPH_Z_TOL",
            "MESH_GRAPH_OBS_FLOW", "MESH_GRAPH_NODE_ID",
            # S37-XYZ (docs/S37_XYZ_SCREEN_PREREG_20260929.md)
            "XYZ_BAND_PX", "XYZ_FRONT_PX", "XYZ_Z_TOL", "XYZ_K", "XYZ_GEOM", "XYZ_REL_HIDDEN",
            "XYZ_SLOT_DIM", "XYZ_READOUT_HIDDEN",
            "RENDER_CHANNELS", "RENDER_H", "RENDER_W", "RENDER_SCALE",
            "RENDER_CHUNK", "RENDER_DEPTH_TOL", "INIT_FROM",
            "ACTIVE_HEAD", "ACTIVE_FEAT_DIM", "ACTIVE_HIDDEN",
            "ENCODER", "ENCODER_HIDDEN", "ENCODER_FEAT", "ENCODER_CELL", "ENCODER_LAYERS",
            "ENCODER_K", "ENCODER_MAX_NODES", "ENCODER_WINDOW", "ENCODER_T_SCALE",
            "DISTILL_WEIGHT", "DISTILL_CKPT",
            # x1001 E7
            "POSE_HEAD_HIDDEN",
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
        # S37 (routed readout, 2026-09-07). The graph stays state-free; the previous state enters
        # at the readout only, as a
        # per-node responsibility over the 16 MANO joints computed from its projected FK
        # (`semkine.routed_readout`). Each finger decoder then reads its own joint's routed
        # evidence and nothing else from the events; the root reads the pooled vector plus all
        # sixteen evidence vectors in joint order.
        self.routed = bool(model_cfg.get("ROUTED_READOUT", False))
        self.route_band_px = float(model_cfg.get("ROUTE_BAND_PX", 16.0))
        self.route_front_k = int(model_cfg.get("ROUTE_FRONT_K", 8))
        # How the root reads the routed evidence. `concat` is S37: one linear map of
        # [f; e_0 .. e_15]. `per_joint` (2026-09-29, the user's variant): joint j's evidence is fused
        # with the pooled vector by its own MLP, [f; e_j] -> ACTIVE_HIDDEN -> 6, and the sixteen
        # outputs are summed into the root update.
        self.root_fusion = str(model_cfg.get("ROOT_FUSION", "concat")).lower()
        if self.root_fusion not in ("concat", "per_joint"):
            raise ValueError(f"MODEL.ROOT_FUSION must be concat or per_joint, got {self.root_fusion!r}")
        if self.root_fusion == "per_joint" and not self.routed:
            raise ValueError("ROOT_FUSION: per_joint fuses the routed joint evidence; it needs ROUTED_READOUT")
        # S37 mesh query (2026-09-07, the user's design). The previous state enters in one form
        # only -- its FK mesh -- and the mesh reads the graph: fixed query vertices gather node
        # features, the graph's edge (local motion) summary and event offsets inside a fixed
        # kernel, skinning weights carry that to the joints, each joint decoder reads its own
        # joint's evidence and is gated by its coverage (`semkine.mesh_query`). No prev_mlp, no
        # prev angles in the heads, no rendered node channels, no derived event tokens.
        self.mesh_query = bool(model_cfg.get("MESH_QUERY", False))
        if self.mesh_query and (self.routed or self.prev_render):
            raise ValueError("MESH_QUERY is the only conditioning path of its arm")
        if self.mesh_query and self.prevpos_embed:
            raise ValueError("MESH_QUERY: the previous state enters through its FK mesh only; "
                             "set PREVPOS_EMBED: false")
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
        # S37 FK graph (2026-09-07, the user's design, no geometry branch). The previous state's
        # FK *is* the graph -- joint nodes, sampled surface-vertex nodes, a background node, edges
        # along the mesh / skinning / kinematic tree -- and the events are its observations: every
        # event is handed to the node under its pixel and each node's input is only what its
        # events say (count, offset, spread, time, polarity, local flow). No event graph, no
        # tokens, no rendering, no geometric node or edge features (`semkine.fk_graph`).
        self.fk_graph = self.encoder_name == "fk_graph"
        if self.fk_graph and (self.prev_render or self.routed or self.mesh_query):
            raise ValueError("ENCODER=fk_graph is the only conditioning path of its arm")
        if self.fk_graph and not self.active_head:
            raise ValueError("ENCODER=fk_graph decodes joints from joint nodes; it needs ACTIVE_HEAD")
        # S37 mesh graph (2026-09-18, the user's drawing). The previous state's *full* MANO mesh
        # is the graph -- 778 vertex nodes plus a background node, edges = the mesh faces -- and
        # the events are its observations exactly as in `fk_graph`, handed to the nearest
        # *visible* vertex. After EdgeConv on the mesh, the fixed skinning weights pool the vertex
        # features into sixteen joint evidences (`semkine.mesh_graph`); joint decoder k reads its
        # own joint's evidence, the root reads all sixteen in order plus the background node. The
        # state and the delta stay 51D MANO parameters: FK needs rotations, not joint positions.
        self.mesh_graph = self.encoder_name == "mesh_graph"
        if self.mesh_graph and (self.prev_render or self.routed or self.mesh_query):
            raise ValueError("ENCODER=mesh_graph is the only conditioning path of its arm")
        if self.mesh_graph and not self.active_head:
            raise ValueError("ENCODER=mesh_graph decodes joints from pooled vertex evidence; "
                             "it needs ACTIVE_HEAD")
        # S37-XYZ (2026-09-29, the user's drawing). prev's full mesh in camera XYZ, events associated
        # along their rays, a shared relation encoder, one MeshGNN on the face 1-ring gated by the 3D
        # edges, 16 LBS slots + the residual slot, one common readout that also reads prev[3:51] and
        # writes the only 51D delta (`semkine.xyz_mesh`). No EventGNN, no prev_mlp, no per-joint heads.
        self.xyz_mesh = self.encoder_name == "xyz_mesh"
        if self.xyz_mesh and (self.prev_render or self.routed or self.mesh_query):
            raise ValueError("ENCODER=xyz_mesh is the only conditioning path of its arm")
        if self.xyz_mesh and (self.prevpos_embed or bool(model_cfg.get("ACTIVE_HEAD", False))):
            raise ValueError("ENCODER=xyz_mesh has one common readout: set PREVPOS_EMBED and "
                             "ACTIVE_HEAD false")
        self.distill_weight = float(model_cfg.get("DISTILL_WEIGHT", 0.0))
        self.teacher = None
        hid = int(model_cfg.get("ACTIVE_HIDDEN", 64))
        if self.fk_graph:
            from semkine.fk_graph import FKGraphEncoder, FKGraphSpec
            self._build_mano(model_cfg)
            hidden = int(model_cfg.get("ENCODER_HIDDEN", 128))
            self.fk_spec = FKGraphSpec.from_mano(
                self.mano, n_verts=int(model_cfg.get("FK_GRAPH_VERTS", 192)),
                k_mesh=int(model_cfg.get("FK_GRAPH_K", 6)))
            self.fk_band_px = float(model_cfg.get("FK_GRAPH_BAND_PX", 16.0))
            self.event_encoder = FKGraphEncoder(
                self.fk_spec, hidden=hidden, n_layers=int(model_cfg.get("ENCODER_LAYERS", 3)),
                node_id=bool(model_cfg.get("FK_GRAPH_NODE_ID", True)))
            self.register_buffer("fk_vert_ids", self.fk_spec.vert_ids.clone(), persistent=False)
            self.register_buffer("fk_lbs_wn", self.fk_spec.lbs_wn.clone(), persistent=False)
            self.conv1 = None
            self.rn = None
            assert self.output_dim == 51, "the active head is defined for the 51D layout"
            # root reads the sixteen joint nodes in joint order plus the background node
            self.root_head = nn.Linear(17 * hidden, 6)
            self.joint_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden + 3, hid), nn.ReLU(inplace=True),
                              nn.Linear(hid, 3))
                for _ in range(15)
            ])
        elif self.mesh_graph:
            from semkine.fk_graph import OBS_DIM, FKGraphEncoder
            from semkine.mesh_graph import MeshGraphSpec
            self._build_mano(model_cfg)
            hidden = int(model_cfg.get("ENCODER_HIDDEN", 128))
            self.mg_spec = MeshGraphSpec.from_mano(self.mano)
            self.mg_band_px = float(model_cfg.get("MESH_GRAPH_BAND_PX", 16.0))
            self.mg_front_px = float(model_cfg.get("MESH_GRAPH_FRONT_PX", 1.0))
            self.mg_z_tol = float(model_cfg.get("MESH_GRAPH_Z_TOL", 0.01))
            # The per-node flow fit (b_u, b_v) is the last two observation channels. H6 of the
            # FK-graph pre-registration measured it at zero contribution in 50 ms packets, so the
            # default here is the first six channels; MESH_GRAPH_OBS_FLOW restores all eight.
            self.mg_obs_dim = OBS_DIM if bool(model_cfg.get("MESH_GRAPH_OBS_FLOW", False)) else OBS_DIM - 2
            self.event_encoder = FKGraphEncoder(
                self.mg_spec, hidden=hidden, n_layers=int(model_cfg.get("ENCODER_LAYERS", 3)),
                node_id=bool(model_cfg.get("MESH_GRAPH_NODE_ID", True)), obs_dim=self.mg_obs_dim)
            self.conv1 = None
            self.rn = None
            assert self.output_dim == 51, "the active head is defined for the 51D layout"
            # evidence per joint = [skinning-weighted mean, hard-part max, coverage] over vertices;
            # the root reads the sixteen evidences in joint order plus the background node
            ev_dim = 2 * hidden + 1
            self.root_head = nn.Linear(16 * ev_dim + hidden, 6)
            self.joint_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(ev_dim + 3, hid), nn.ReLU(inplace=True),
                              nn.Linear(hid, 3))
                for _ in range(15)
            ])
        elif self.xyz_mesh:
            from semkine.xyz_mesh import XYZMeshNet
            self._build_mano(model_cfg)
            assert self.output_dim == 51 and self.predict_delta, "xyz_mesh writes a 51D delta"
            self.xyz_band_px = float(model_cfg.get("XYZ_BAND_PX", 16.0))
            self.xyz_front_px = float(model_cfg.get("XYZ_FRONT_PX", 3.0))
            self.xyz_z_tol = float(model_cfg.get("XYZ_Z_TOL", 0.01))
            self.xyz_k = int(model_cfg.get("XYZ_K", 16))
            self.xyz_max_nodes = int(model_cfg.get("ENCODER_MAX_NODES", 2048))
            self.xyz = XYZMeshNet(
                self.mano.weights, self.mano.f,
                hidden=int(model_cfg.get("ENCODER_HIDDEN", 128)),
                n_layers=int(model_cfg.get("ENCODER_LAYERS", 3)),
                rel_hidden=int(model_cfg.get("XYZ_REL_HIDDEN", 64)),
                slot_dim=int(model_cfg.get("XYZ_SLOT_DIM", 32)),
                readout_hidden=int(model_cfg.get("XYZ_READOUT_HIDDEN", 256)),
                geom=bool(model_cfg.get("XYZ_GEOM", True)))
            self.conv1 = None
            self.rn = None
        elif self.encoder_name:
            # S2 / S16. The dense conv trunk is not built: a raw-event encoder that still
            # allocated a resnet would be LNES with extra copies, and INIT_FROM would silently
            # load the wrong weights.
            from semkine.frontends import build_frontend
            feat = int(model_cfg.get("ENCODER_FEAT", model_cfg.get("ACTIVE_FEAT_DIM", 256)))
            # The previous state reaches the frontend the way it does in the arm that actually
            # tracks: two rendered values sampled at each event's own pixel. Comparative, not
            # indexing -- the event's position in the representation does not move with the pose.
            extra = 2 if self.prev_render else 0
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
                node_attrs=str(model_cfg.get("ENCODER_NODE_ATTRS", "token7")),
                readout=not self.mesh_query,
            )
            self.conv1 = None
            self.rn = None
            if self.active_head:
                assert self.output_dim == 51, "the active head is defined for the 51D layout"
                hidden = int(model_cfg.get("ENCODER_HIDDEN", 128))
                if self.mesh_query:
                    # evidence per joint = [node feat, edge summary, offset (2), mass, coverage]
                    ev_dim = hidden + 4 + 3 + 1
                    self.root_head = nn.Linear(ev_dim + 16 * ev_dim, 6)
                    self.joint_heads = nn.ModuleList([
                        nn.Sequential(nn.Linear(ev_dim, hid), nn.ReLU(inplace=True),
                                      nn.Linear(hid, 3))
                        for _ in range(15)
                    ])
                else:
                    if self.routed:
                        # evidence per joint = [weighted mean, max, coverage] over node features
                        ev_dim = 2 * hidden + 1
                        if self.root_fusion == "per_joint":
                            from semkine.routed_readout import PerJointRootFusion
                            self.root_fusion_head = PerJointRootFusion(feat, ev_dim, hid)
                        else:
                            self.root_head = nn.Linear(feat + 16 * ev_dim, 6)
                        head_in = ev_dim
                    else:
                        self.root_head = nn.Linear(feat, 6)
                        head_in = feat
                    self.joint_heads = nn.ModuleList([
                        nn.Sequential(nn.Linear(head_in + 3, hid), nn.ReLU(inplace=True),
                                      nn.Linear(hid, 3))
                        for _ in range(15)
                    ])
            else:
                # x1001 E7: an optional hidden layer makes the absolute readout non-linear
                # (POSE_HEAD_HIDDEN = 0 keeps the single linear layer of every earlier arm).
                ph = int(model_cfg.get("POSE_HEAD_HIDDEN", 0))
                self.pose_head = (nn.Linear(feat, self.output_dim) if ph <= 0 else
                                  nn.Sequential(nn.Linear(feat, ph), nn.ReLU(inplace=True),
                                                nn.Linear(ph, self.output_dim)))
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

        if (self.routed or self.mesh_query) and not (self.encoder_name == "event_gnn"
                                                     and self.active_head):
            raise ValueError("ROUTED_READOUT / MESH_QUERY read event_gnn node features into the "
                             "active joint heads; they need MODEL.ENCODER=event_gnn and ACTIVE_HEAD")
        self._ctx_betas = None
        self._ctx_K = None
        if self.prev_render or self.routed or self.mesh_query:
            self._build_mano(model_cfg)
        if self.prev_render:
            self._register_vertex_codes()
        if self.mesh_query:
            from semkine.mesh_query import MeshQuery
            self.mesh_query_mod = MeshQuery(
                self.mano.weights,
                n_tokens=int(model_cfg.get("MESH_QUERY_TOKENS", 192)),
                band_px=float(model_cfg.get("MESH_QUERY_BAND_PX", 16.0)),
                sigma_px=float(model_cfg.get("MESH_QUERY_SIGMA_PX", 8.0)),
                node_dim=int(model_cfg.get("ENCODER_HIDDEN", 128)),
                edge_dim=self.event_encoder.EDGE_SUMMARY_DIM,
            )
        if self.routed or self.mesh_query or self.fk_graph or self.mesh_graph or self.xyz_mesh:
            #: diagnostics of the last `forward_packet`, for the training loop and the probes
            self.route_stats = {}
            #: set at inference to zero every evidence vector (fk_graph: every observation), for
            #: the same-checkpoint ablation that decides whether the evidence is load-bearing
            self.ablate_evidence = False
            #: probe-only: when set to a `(B, 51)` state, the *geometry* (routing / mesh query /
            #: event assignment) uses it instead of the packet's `prev_state` while everything
            #: else still sees the packet's prev. This is the "oracle routing" arm of the
            #: closed-loop shortcut test.
            self.route_prev_override = None
        if self.fk_graph or self.mesh_graph:
            #: probe-only: `(obs_dim,)` 0/1 mask over the observation channels (H6 sufficiency)
            self.obs_feature_mask = None
        # S37 root innovation (2026-09-29, docs/S37_ROOT_INNOVATION_PREREG.md). A second root path
        # that reads where the events sit relative to prev's projected silhouette, with prev's part
        # lever arms multiplying that residual (`semkine.root_innovation`); and the option to drop
        # the event-blind `prev_mlp` pull on the six root dimensions.
        self.root_innovation = bool(model_cfg.get("ROOT_INNOVATION", False))
        self.prev_mlp_root = bool(model_cfg.get("PREV_MLP_ROOT", True))
        if self.root_innovation:
            if not (self.routed and self.active_head):
                raise ValueError("ROOT_INNOVATION reads the S37 routing; it needs ROUTED_READOUT")
            if self.root_fusion != "concat":
                raise ValueError("ROOT_INNOVATION adds to the S37 concat root head; set ROOT_FUSION: concat")
            from semkine.root_innovation import RootInnovationHead
            self.root_innov_head = RootInnovationHead(
                hidden=int(model_cfg.get("ROOT_INNOV_HIDDEN", 32)),
                width=int(model_cfg.get("ROOT_INNOV_WIDTH", 16)))
            #: set at inference to zero the innovation features (the arm's mechanism ablation)
            self.ablate_innovation = False
        if not self.prev_mlp_root and not self.prevpos_embed:
            raise ValueError("PREV_MLP_ROOT: false only means something with PREVPOS_EMBED")
        # Stage-1 training of an arm on a frozen trunk: only parameters under these prefixes train.
        prefixes = (self.cfg.get("TRAIN", {}) or {}).get("TRAINABLE_PREFIXES")
        if prefixes:
            prefixes = tuple(str(p) for p in prefixes)
            for name, p in self.named_parameters():
                p.requires_grad_(name.startswith(prefixes))
            if not any(p.requires_grad for p in self.parameters()):
                raise ValueError(f"TRAIN.TRAINABLE_PREFIXES {list(prefixes)} match no parameter")

    def _build_mano(self, model_cfg):
        """The MANO layer plus the render-frame intrinsics every conditioning path projects with."""
        if hasattr(self, "mano"):
            return
        from mano_layer import ManoLayer

        mano_npz = self.cfg.get("MANO", {}).get("NPZ", "assets/mano_right.npz")
        self.mano = ManoLayer(mano_npz, add_mean=False)
        self.mano.eval()
        self.render_h = int(model_cfg.get("RENDER_H", self.cfg.get("DATA", {}).get("HEIGHT", 180)))
        self.render_w = int(model_cfg.get("RENDER_W", self.cfg.get("DATA", {}).get("WIDTH", 240)))
        self.render_scale = float(model_cfg.get("RENDER_SCALE", 240.0 / 640.0))
        self.render_chunk = int(model_cfg.get("RENDER_CHUNK", 256))
        self.render_depth_tol = float(model_cfg.get("RENDER_DEPTH_TOL", 5.0e-3))

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

    def _decode_active(self, feat, prevpos, evidence=None, root_extra=None, root_add=None):
        """Assemble the 51D output from the root head and the fifteen joint decoders.

        Without `evidence`, every decoder reads the shared pooled vector and its own joint's
        previous angle, which is the S10 arrangement. With `evidence` `(B, 16, E)` (S37), joint
        decoder `k` reads evidence row `k + 1` -- its own joint's routed nodes, and nothing else
        from the events -- while the root reads the pooled vector (when there is one), all sixteen
        rows in joint order (a permutation-invariant pool over parts would cancel a rotational
        offset field) and, for the FK graph, the background node (`root_extra`). With
        `ROOT_FUSION: per_joint` the root is instead the sum of sixteen per-joint MLPs, head j
        reading `[f; e_j]` (`semkine.routed_readout.PerJointRootFusion`); the heads keep their own
        weights, so the sum stays joint-ordered.

        The fifteen decoders are executed as a Python loop on purpose. Batching them into one
        `baddbmm` computes the same math ~0.7 ms faster per recursive step, but the different
        accumulation order perturbs the output by ~1e-6, and the closed loop amplified that to
        0.15 mm of recursive RA on seed 3408 -- past the 0.05 mm reproduction gate that pins
        every recorded number (S36 latency audit, 2026-08-29). Bit-reproducibility wins.
        """
        ref = feat if feat is not None else evidence
        if evidence is None:
            root = self.root_head(feat)
        elif self.root_fusion == "per_joint":
            root = self.root_fusion_head(feat, evidence.to(ref.dtype))
        else:
            evidence = evidence.to(ref.dtype)
            parts = ([feat] if feat is not None else []) + [evidence.flatten(1)]
            if root_extra is not None:
                parts.append(root_extra.to(ref.dtype))
            root = self.root_head(torch.cat(parts, dim=-1))
        if root_add is not None:
            root = root + root_add.to(root.dtype)
        prev = prevpos.to(ref.dtype)
        cols = [root]
        for k, head in enumerate(self.joint_heads):
            if self.ablate_joint_heads:
                # The ablation is a genuine silence, not a scaled-down update: with PREDICT_DELTA
                # the output is `prev + out`, so zero here means "this joint does not move".
                cols.append(torch.zeros(ref.shape[0], 3, device=ref.device, dtype=ref.dtype))
                continue
            src = feat if evidence is None else evidence[:, k + 1]
            cols.append(head(torch.cat([src, prev[:, 6 + 3 * k : 9 + 3 * k]], dim=-1)))
        return torch.cat(cols, dim=-1)

    def _fk_node_uv(self, prev, betas_f, k_f):
        """Projected pixels `(B, 16 + V, 2)` of the FK-graph nodes under state `prev`: joint nodes
        (skinning-weighted surface centroids, joint order) then the sampled vertices. `no_grad`:
        the state is conditioning, and here it only decides which node an event is handed to."""
        with torch.no_grad():
            verts, _ = self._fk(prev.float(), betas_f)
            pts = torch.cat([torch.einsum("jv,bvc->bjc", self.fk_lbs_wn, verts.float()),
                             verts.float()[:, self.fk_vert_ids]], dim=1)           # (B, 16+V, 3)
            fx, fy, cx, cy = self._intrinsics(k_f)
            x, y, z = pts.unbind(-1)
            z_safe = z.clamp(min=1e-6)
            return torch.stack([fx[:, None] * x / z_safe + cx[:, None],
                                fy[:, None] * y / z_safe + cy[:, None]], dim=-1)

    def _fk_graph_forward(self, batch, prev):
        """S37 FK graph: FK of `prev` -> node pixels -> events assigned and summarised per node ->
        EdgeConv on the fixed graph -> joint nodes and background node for the decoders."""
        from semkine.fk_graph import assign_and_observe
        prev_geo = prev if self.route_prev_override is None else \
            self.route_prev_override.to(prev.device, prev.dtype)
        betas_f, k_f = self._resolve_betas_K(prev_geo, batch.betas, batch.camera_K)
        with torch.no_grad():
            uv = self._fk_node_uv(prev_geo, betas_f, k_f)
            obs, assign = assign_and_observe(batch.events, batch.ptr, uv, batch.delta_t_s,
                                             self.fk_band_px)
            n_ev = int(batch.events.shape[0])
            self.route_stats = {
                "route_frac_routed": (assign < uv.shape[1]).float().mean() if n_ev else obs.new_zeros(()),
                "route_joints_hit": (obs[:, :16, 0] > 0).sum(1).float().mean(),
            }
        if self.obs_feature_mask is not None:
            obs = obs * self.obs_feature_mask.to(obs).view(1, 1, -1)
        if self.ablate_evidence:
            obs = torch.zeros_like(obs)
        h = self.event_encoder(obs)                                                  # (B, N, hidden)
        return h[:, :16], h[:, -1]

    def _mesh_graph_forward(self, batch, prev):
        """S37 mesh graph: FK of `prev` -> 778 projected vertices and their visibility -> events
        handed to the nearest visible vertex and summarised per node -> EdgeConv on the mesh ->
        skinning weights pool the vertices into sixteen joint evidences `(B, 16, 2C+1)`; the
        background node `(B, C)` goes to the root."""
        from semkine.fk_graph import assign_and_observe
        from semkine.mesh_graph import (assign_events_by_lut, lbs_pool_evidence,
                                        nearest_node_lut, visible_vertices)
        prev_geo = prev if self.route_prev_override is None else \
            self.route_prev_override.to(prev.device, prev.dtype)
        betas_f, k_f = self._resolve_betas_K(prev_geo, batch.betas, batch.camera_K)
        with torch.no_grad():
            verts, _ = self._fk(prev_geo.float(), betas_f)                           # (B, 778, 3)
            uv, z = self._project_verts(verts, k_f)                                  # (B, 778, 2)
            vis = visible_vertices(verts, uv, z, self.mano.f, self.render_h, self.render_w,
                                   self.mg_front_px, self.mg_z_tol)                  # (B, 778)
            # jump-flood LUT: O(BHW), independent of event count and of the 778 nodes
            lut = nearest_node_lut(uv, vis, self.render_h, self.render_w, self.mg_band_px)
            assign_pre = assign_events_by_lut(batch.events, lut, background=uv.shape[1])
            obs, assign = assign_and_observe(batch.events, batch.ptr, uv, batch.delta_t_s,
                                             self.mg_band_px, assign_pre=assign_pre)
            obs = obs[..., : self.mg_obs_dim]
            has = obs[:, :-1, 0] > 0                                                 # vertex saw an event
            n_ev = int(batch.events.shape[0])
            self.route_stats = {
                "route_frac_routed": (assign < uv.shape[1]).float().mean() if n_ev else obs.new_zeros(()),
                "mg_frac_visible": vis.float().mean(),
            }
        if self.obs_feature_mask is not None:
            obs = obs * self.obs_feature_mask.to(obs).view(1, 1, -1)
        if self.ablate_evidence:
            # "the mesh saw nothing": observations and the observed-vertex mask both go, so the
            # pool cannot leak *where* events fell through its weights
            obs = torch.zeros_like(obs)
            has = torch.zeros_like(has)
        h = self.event_encoder(obs)                                                  # (B, 779, C)
        e, count = lbs_pool_evidence(h[:, :-1], self.mano.weights, vis, has)
        self.route_stats["route_joints_hit"] = (count > 0).sum(1).float().mean()
        return e, h[:, -1]

    def _xyz_mesh_forward(self, batch, prev):
        """S37-XYZ: FK of `prev` in camera XYZ, event rays through the render-frame intrinsics
        (`_intrinsics`, the same `s K` the events live in, per sample under domrand), the ray/mesh
        association, then the three learned modules of `semkine.xyz_mesh`. Returns the 51D delta."""
        from semkine.encoder import event_tokens
        from semkine.events import EV_X, EV_Y
        from semkine.xyz_mesh import associate_rays, stride_sample
        events, ptr = batch.events, batch.ptr
        B = int(ptr.numel() - 1)
        if events.shape[0] == 0:
            self.route_stats = {}
            return prev.new_zeros(B, 51)
        prev_geo = prev if self.route_prev_override is None else \
            self.route_prev_override.to(prev.device, prev.dtype)
        betas_f, k_f = self._resolve_betas_K(prev_geo, batch.betas, batch.camera_K)
        src, mask = stride_sample(ptr, self.xyz_max_nodes)
        tok = event_tokens(events, ptr, batch.delta_t_s, self.render_h, self.render_w)
        tok = tok[src.reshape(-1)].view(B, -1, tok.shape[-1]) * mask.unsqueeze(-1)
        with torch.no_grad():
            verts, _ = self._fk(prev_geo.float(), betas_f)                           # (B, 778, 3)
            fx, fy, cx, cy = self._intrinsics(k_f)
            ev = events[src.reshape(-1)].view(B, -1, events.shape[1])
            px, py = ev[..., EV_X].float(), ev[..., EV_Y].float()
            d = torch.stack([(px - cx[:, None]) / fx[:, None], (py - cy[:, None]) / fy[:, None],
                             torch.ones_like(px)], dim=-1)
            f_px = (fx * fy).sqrt()
            assoc = associate_rays(d, mask, verts, self.mano.f, self.xyz.vert_faces,
                                   band=self.xyz_band_px / f_px, front_tol=self.xyz_front_px / f_px,
                                   z_tol=self.xyz_z_tol, k=self.xyz_k)
        delta, stats, aux = self.xyz(tok, d, mask, verts, assoc, prev[:, 3:51],
                                     ablate=self.ablate_evidence)
        self.route_stats = stats
        if getattr(self, "xyz_keep_aux", False):
            #: probe-only: the association and the pooled slots of the last packet batch
            self.xyz_last = {"assoc": assoc, "d": d, "mask": mask, "verts": verts, "tok": tok,
                             "delta": delta.detach(), **{k: v.detach() for k, v in aux.items()}}
        # region agent log
        if self.training and self._agent_should_log():
            from semkine.xyz_mesh import agent_log
            with torch.no_grad():
                rot_gap = (prev[:, 3:6] - batch.target[:, 3:6]).norm(dim=-1)            # rad, prev vs GT
                big = rot_gap >= 0.25
                live = mask.float()

                def frac(x, sel):
                    den = (live * sel[:, None]).sum()
                    return float((x.float() * live * sel[:, None]).sum() / den.clamp_min(1)) if den > 0 else None
                agent_log("H2", "model.py:_xyz_mesh_forward", "association stats (train batch)", {
                    "step": int(self.global_step), "B": B, "geom": bool(self.xyz.geom),
                    "valid_all": frac(assoc["valid"], torch.ones_like(big)),
                    "hit_all": frac(assoc["hit"], torch.ones_like(big)),
                    "valid_small_gap": frac(assoc["valid"], ~big), "hit_small_gap": frac(assoc["hit"], ~big),
                    "valid_big_gap": frac(assoc["valid"], big), "hit_big_gap": frac(assoc["hit"], big),
                    "share_big_gap": float(big.float().mean()),
                    "resid_mass": float(stats["xyz_resid_mass"]),
                    "verts_direct": float(stats["xyz_verts_direct"]), "verts_prop": float(stats["xyz_verts_prop"]),
                    "delta_abs_mean": {"t": float(delta[:, :3].float().abs().mean()),
                                       "rot": float(delta[:, 3:6].float().abs().mean()),
                                       "pose": float(delta[:, 6:].float().abs().mean())},
                })
        # endregion
        return delta

    # region agent log
    def _agent_should_log(self) -> bool:
        trainer = getattr(self, "_trainer", None)
        if trainer is None or int(getattr(trainer, "global_rank", 0)) != 0:
            return False
        step = int(self.global_step)
        if step % 100 != 0 or getattr(self, "_agent_last_step", -1) == step:
            return False
        self._agent_last_step = step
        return True

    def on_after_backward(self):
        if not getattr(self, "xyz_mesh", False):
            return
        trainer = getattr(self, "_trainer", None)
        step = int(self.global_step)
        if trainer is None or int(trainer.global_rank) != 0 or step % 100 != 0:
            return
        from semkine.xyz_mesh import agent_log
        groups = {"relation": ["xyz.rel."], "vertex_in": ["xyz.vin."], "edge_gate": ["xyz.edge.", ".gate."],
                  "mesh_msg": [".msg.", ".slf."], "readout": ["xyz.slot_proj.", "xyz.fc1.", "xyz.fc2."]}
        out = {}
        for gname, keys in groups.items():
            sq, n = 0.0, 0
            for name, p in self.named_parameters():
                if p.grad is not None and any(k in name for k in keys):
                    sq += float(p.grad.detach().float().square().sum())
                    n += 1
            out[gname] = {"grad_norm": sq ** 0.5, "n_tensors": n}
        out["weight_norm_fc2"] = float(self.xyz.fc2.weight.detach().float().norm())
        agent_log("H3", "model.py:on_after_backward", "grad norms per learned module", {
            "step": step, "geom": bool(self.xyz.geom), "loss": getattr(self, "_agent_last_loss", None),
            "groups": out})
    # endregion

    def _project_verts(self, verts, k_f):
        """Pinhole projection with the render intrinsics: `(uv (B, V, 2), z (B, V))`."""
        fx, fy, cx, cy = self._intrinsics(k_f)
        x, y, z = verts.unbind(-1)
        z_safe = z.clamp(min=1e-6)
        uv = torch.stack([fx[:, None] * x / z_safe + cx[:, None],
                          fy[:, None] * y / z_safe + cy[:, None]], dim=-1)
        return uv, z_safe

    def _project_prev(self, prev, betas_f, k_f):
        """MANO FK of `prev` projected with the render intrinsics: `(uv (B, 778, 2), z (B, 778))`.

        Under `no_grad`: the state is conditioning, not a differentiable pathway (same rule as
        the render path).
        """
        with torch.no_grad():
            verts, _ = self._fk(prev.float(), betas_f)
            return self._project_verts(verts, k_f)

    def _route_nodes(self, px, py, mask, prev, betas_f, k_f):
        """S37-routed responsibilities `(B, N, 16)` of the sampled nodes over the joints of `prev`:
        the front-most nearest-vertex LBS lookup of `semkine.routed_readout`. Also returns the
        per-node distance to the hand and the vertex id, for the probes."""
        from semkine.routed_readout import route_front_vertex_lbs
        with torch.no_grad():
            uv, z = self._project_prev(prev, betas_f, k_f)
            return route_front_vertex_lbs(px, py, mask, uv, z, self.mano.weights,
                                          self.route_band_px, self.route_front_k)

    def _node_responsibility(self, px, py, mask, prev, betas_f, k_f):
        """Probe interface shared by both S37 arms: `(B, N, 16)` share of each node's evidence
        that reaches each joint under the geometry of `prev`."""
        if self.routed:
            return self._route_nodes(px, py, mask, prev, betas_f, k_f)[0]
        uv, z = self._project_prev(prev, betas_f, k_f)
        return self.mesh_query_mod.node_responsibility(px, py, mask, uv, z)

    def _decode_mesh(self, e, mesh, cov):
        """S37 mesh query: root reads the mesh aggregate and all sixteen joint evidences in joint
        order; joint decoder k reads evidence row k + 1 and nothing else, and is gated by that
        joint's coverage so an unobserved joint does not move (per-joint zero-event identity)."""
        e = e.to(self.root_head.weight.dtype)
        mesh = mesh.to(e.dtype)
        seen = (cov > 0)
        root = self.root_head(torch.cat([mesh, e.flatten(1)], dim=-1)) * seen.any(1, keepdim=True).to(e.dtype)
        cols = [root]
        for k, head in enumerate(self.joint_heads):
            if self.ablate_joint_heads:
                cols.append(torch.zeros(e.shape[0], 3, device=e.device, dtype=e.dtype))
                continue
            cols.append(head(e[:, k + 1]) * seen[:, k + 1: k + 2].to(e.dtype))
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
        evidence = None
        root_add = None
        if self.fk_graph:
            joints_h, background = self._fk_graph_forward(batch, prev)
            out = self._decode_active(None, prev, joints_h, root_extra=background)
        elif self.mesh_graph:
            evidence, background = self._mesh_graph_forward(batch, prev)
            out = self._decode_active(None, prev, evidence, root_extra=background)
        elif self.xyz_mesh:
            out = self._xyz_mesh_forward(batch, prev)
        elif self.mesh_query:
            _, h, g, px, py, mask = self.event_encoder(events, ptr, batch.delta_t_s, None,
                                                       return_nodes=True)
            B = int(ptr.numel() - 1)
            mq = self.mesh_query_mod
            if h.shape[1] == 0:
                e = torch.zeros(B, 16, mq.ev_dim, device=events.device, dtype=h.dtype)
                mesh = torch.zeros(B, mq.ev_dim, device=events.device, dtype=h.dtype)
                cov = torch.zeros(B, 16, device=events.device, dtype=torch.float32)
                self.route_stats = {}
            else:
                prev_geo = prev if self.route_prev_override is None else \
                    self.route_prev_override.to(prev.device, prev.dtype)
                betas_f, k_f = self._resolve_betas_K(prev_geo, batch.betas, batch.camera_K)
                uv, z = self._project_prev(prev_geo, betas_f, k_f)
                e, mesh, cov, self.route_stats = mq(h, g, px, py, mask, uv, z)
            if self.ablate_evidence:
                # "the mesh saw nothing": evidence and coverage both zero, so every gate closes
                # and the arm degenerates to the do-nothing tracker (output = prev)
                e, mesh, cov = torch.zeros_like(e), torch.zeros_like(mesh), torch.zeros_like(cov)
            out = self._decode_mesh(e, mesh, cov)
        else:
            if self.routed:
                from semkine.routed_readout import pool_joint_evidence
                feat, h, _, px, py, mask = self.event_encoder(events, ptr, batch.delta_t_s, extra,
                                                              return_nodes=True)
                if h.shape[1] == 0:
                    evidence = torch.zeros(feat.shape[0], 16, 2 * h.shape[-1] + 1,
                                           device=feat.device, dtype=feat.dtype)
                    self.route_stats = {}
                else:
                    prev_route = prev if self.route_prev_override is None else \
                        self.route_prev_override.to(prev.device, prev.dtype)
                    betas_f, k_f = self._resolve_betas_K(prev_route, batch.betas, batch.camera_K)
                    if self.root_innovation:
                        # the same FK / projection / routing `_route_nodes` computes, kept so the
                        # innovation features reuse them
                        from semkine.routed_readout import route_front_vertex_lbs
                        from semkine.root_innovation import innovation_features, lever_arms
                        with torch.no_grad():
                            verts, joints = self._fk(prev_route.float(), betas_f)
                            uv, zv = self._project_verts(verts, k_f)
                            a, dist, vid = route_front_vertex_lbs(
                                px, py, mask, uv, zv, self.mano.weights,
                                self.route_band_px, self.route_front_k)
                            # fp32 in training and evaluation alike: under bf16 autocast the SDF
                            # blur and the pooling einsum would otherwise drop to bf16 in training only
                            with torch.autocast(device_type=px.device.type, enabled=False):
                                q = innovation_features(px.float(), py.float(), mask, a.float(), vid,
                                                        uv.float(), self.render_h, self.render_w)
                                g = lever_arms(verts.float(), joints[:, self.FK_ROOT_JOINT].float(),
                                               self.mano.weights.float(), *self._intrinsics(k_f))
                            if self.ablate_innovation:
                                q = torch.zeros_like(q)
                        root_add = self.root_innov_head(q, g)
                    else:
                        a, dist, _ = self._route_nodes(px, py, mask, prev_route, betas_f, k_f)
                    evidence, count = pool_joint_evidence(h, a, mask)
                    live = mask.sum(1).clamp_min(1).to(torch.float32)
                    self.route_stats = {
                        "route_frac_routed": ((a.sum(-1) > 0).sum(1).to(torch.float32) / live).mean(),
                        "route_joints_hit": (count > 0).sum(1).to(torch.float32).mean(),
                    }
                if self.ablate_evidence:
                    evidence = torch.zeros_like(evidence)
            else:
                feat = self.event_encoder(events, ptr, batch.delta_t_s, extra)
            if self.active_head:
                out = self._decode_active(feat, prev, evidence, root_add=root_add)
            else:
                out = self.pose_head(feat)
        if self.prevpos_embed:
            pm = self.prev_mlp(prev.to(out.dtype))
            if not self.prev_mlp_root:
                # the event-blind pull on translation and global rotation is removed; the finger
                # rows are the trained S37 ones
                pm = torch.cat([torch.zeros_like(pm[:, :6]), pm[:, 6:]], dim=-1)
            out = out + pm
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
