#!/usr/bin/env python3
"""EventHands absolute-pose models (config-driven 12D / 51D)."""
from __future__ import annotations

import gc
from typing import Any, Dict, Optional

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl

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
        return self._mse_51d_loss(pred, y)

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

    def _mse_51d_loss(self, pred, y):
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
        return loss, {
            "mano_loss": l_local,
            "pos_loss": l_t,
            "rot_loss": l_root,
            "z_loss": F.mse_loss(pred[:, self.slices.transl][:, 2:3], y[:, self.slices.transl][:, 2:3]),
        }

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
        with torch.no_grad():
            est = self.forward_packet(lead).detach()
        take = torch.rand(main.prev_state.shape[0], device=main.prev_state.device) < p
        cond = est.to(main.prev_state.dtype)
        if self.unroll_residual:
            cond = main.prev_state + (est - lead.target).to(main.prev_state.dtype)
        main.prev_state = torch.where(take[:, None], cond, main.prev_state)
        return main

    def _predict_batch(self, batch):
        """One forward that accepts either the legacy 5-tuple or an `EventPacketBatch`."""
        packed = self._unpack_batch(self._maybe_unroll(batch))
        if hasattr(packed, "events"):
            pred = self.forward_packet(packed)
            return pred, packed.target, packed.betas
        x, prevpos, y, betas, camera_K = packed
        return self(x, prevpos, betas=betas, camera_K=camera_K), y, betas

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

    def training_step(self, batch, batch_nb):
        pred, y, betas = self._predict_batch(batch)
        loss, parts = self._compute_loss(pred, y, betas)
        loss, parts = self._maybe_distill(pred, batch, loss, parts)
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
        pred, y, betas = self._predict_batch(batch)
        loss, parts = self._compute_loss(pred, y, betas)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("val_mano_loss", parts["mano_loss"], sync_dist=True, on_epoch=True)
        self.log("val_pos_loss", parts["pos_loss"], sync_dist=True, on_epoch=True)
        self.log("val_rot_loss", parts["rot_loss"], sync_dist=True, on_epoch=True)
        self._log_so3_fk_parts("val", parts, on_epoch=True)
        return loss

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

    def configure_optimizers(self):
        lr = float(self.cfg.get("TRAIN", {}).get("LR", 1e-3))
        opt = optim.Adam(self.parameters(), lr=lr)
        warmup = int(self.cfg.get("TRAIN", {}).get("WARMUP_STEPS", 0))
        if warmup <= 0:
            return opt

        def lr_lambda(step):
            if step < warmup:
                return float(step + 1) / float(warmup)
            return 1.0

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
            "RENDER_CHANNELS", "RENDER_H", "RENDER_W", "RENDER_SCALE",
            "RENDER_CHUNK", "RENDER_DEPTH_TOL", "INIT_FROM",
            "ACTIVE_HEAD", "ACTIVE_FEAT_DIM", "ACTIVE_HIDDEN",
            "ENCODER", "ENCODER_HIDDEN", "ENCODER_FEAT", "ENCODER_CELL",
            "ENCODER_KSSF", "ENCODER_KSSF_HALO", "ENCODER_NODE_DIM", "ENCODER_JOINT_TOKENS",
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
        self.predict_delta = bool(model_cfg.get("PREDICT_DELTA", False))
        self.prevpos_embed = bool(model_cfg.get("PREVPOS_EMBED", False))
        self.prev_render = bool(model_cfg.get("PREV_RENDER", False))
        self.zero_event_gate = bool(model_cfg.get("ZERO_EVENT_GATE", self.prev_render))
        self.render_channels = self._parse_render_channels(model_cfg)
        in_ch = 2 + sum(self.RENDER_FACES[c] for c in self.render_channels)
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
        self.encoder_kssf = False
        self.joint_tokens = False
        #: set at inference to zero the S18 geometric channels on an unchanged checkpoint (G3)
        self.ablate_kssf = False
        hid = int(model_cfg.get("ACTIVE_HIDDEN", 64))
        if self.encoder_name:
            # S2 / S16. The dense conv trunk is not built: a raw-event encoder that still
            # allocated a resnet would be LNES with extra copies, and INIT_FROM would silently
            # load the wrong weights.
            from semkine.frontends import build_frontend
            feat = int(model_cfg.get("ENCODER_FEAT", model_cfg.get("ACTIVE_FEAT_DIM", 256)))
            # S18. `ENCODER_KSSF` replaces the two-channel splat lookup with the twelve-channel
            # kinematic lift, which is a different quantity and not a wider version of the same
            # one: the splat gives silhouette and inverse depth, KSSF gives the contour normal,
            # the skinning weights and the part code, i.e. the terms the event residual is
            # actually built from. The two are mutually exclusive so the input width is defined.
            self.encoder_kssf = bool(model_cfg.get("ENCODER_KSSF", False))
            # E5.5a: replace the visibility hard gate in the geometric lift with the deletion-free
            # halo routing. A representation property, so it is a config bit that the checkpoint
            # carries, not an inference-time switch: the routed statistics change distribution and
            # a gated checkpoint cannot be read through the halo lift.
            self.kssf_halo = bool(model_cfg.get("ENCODER_KSSF_HALO", False))
            if self.encoder_kssf and self.encoder_name not in ("keg", "kinematic_graph"):
                raise ValueError("MODEL.ENCODER_KSSF is defined for the keg frontend only")
            if self.kssf_halo and not self.encoder_kssf:
                raise ValueError("MODEL.ENCODER_KSSF_HALO requires MODEL.ENCODER_KSSF")
            if self.encoder_kssf:
                from semkine.keg import KSSF_CHANNELS
                extra = KSSF_CHANNELS
            else:
                extra = 2 if self.prev_render else 0
            self.event_encoder = build_frontend(
                self.encoder_name,
                height=int(self.cfg.get("DATA", {}).get("HEIGHT", 180)),
                width=int(self.cfg.get("DATA", {}).get("WIDTH", 240)),
                hidden=int(model_cfg.get("ENCODER_HIDDEN", 128)),
                feat_dim=feat,
                extra_channels=extra,
                cell=int(model_cfg.get("ENCODER_CELL", 16)),
                node_dim=int(model_cfg.get("ENCODER_NODE_DIM", 64)),
            )
            self.conv1 = None
            self.rn = None
            # S18/S10. When the frontend already carries a feature per joint, joint `k`'s decoder
            # reads node `k` instead of the shared vector. That is a strictly stronger form of the
            # unique-pathway rule: the trunk has no route to a finger angle *and* no route to
            # another finger's node.
            self.joint_tokens = bool(model_cfg.get("ENCODER_JOINT_TOKENS", False))
            if self.joint_tokens and not hasattr(self.event_encoder, "forward_nodes"):
                raise ValueError("MODEL.ENCODER_JOINT_TOKENS needs a frontend with node outputs")
            if self.joint_tokens and not self.active_head:
                raise ValueError("MODEL.ENCODER_JOINT_TOKENS needs MODEL.ACTIVE_HEAD")
            if self.active_head:
                assert self.output_dim == 51, "the active head is defined for the 51D layout"
                self.root_head = nn.Linear(feat, 6)
                jin = int(getattr(self.event_encoder, "node_dim", 0)) if self.joint_tokens else feat
                self.joint_heads = nn.ModuleList([
                    nn.Sequential(nn.Linear(jin + 3, hid), nn.ReLU(inplace=True),
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

        self._ctx_betas = None
        self._ctx_K = None
        if self.prev_render:
            from mano_layer import ManoLayer

            mano_npz = self.cfg.get("MANO", {}).get("NPZ", "assets/mano_right.npz")
            self.mano = ManoLayer(mano_npz, add_mean=False)
            self.mano.eval()
            self.render_h = int(model_cfg.get("RENDER_H", self.cfg.get("DATA", {}).get("HEIGHT", 180)))
            self.render_w = int(model_cfg.get("RENDER_W", self.cfg.get("DATA", {}).get("WIDTH", 240)))
            self.render_scale = float(model_cfg.get("RENDER_SCALE", 240.0 / 640.0))
            self.render_chunk = int(model_cfg.get("RENDER_CHUNK", 256))
            self.render_depth_tol = float(model_cfg.get("RENDER_DEPTH_TOL", 5.0e-3))
            self._register_vertex_codes()
        # Held outside the module tree: KSSF has no parameters and holds a second reference to
        # `self.mano`, which would duplicate the hand model in `state_dict` and break loading a
        # checkpoint saved before S18.
        self._kssf_holder = []
        if self.encoder_kssf:
            if not self.prev_render:
                raise ValueError("MODEL.ENCODER_KSSF needs MODEL.PREV_RENDER for the hand model")
            from semkine.kssf import KSSF

            k = KSSF(self.mano, height=self.render_h, width=self.render_w,
                     render_scale=self.render_scale)
            k.eval()
            self._kssf_holder.append(k)

    def kssf_on(self, device):
        """The KSSF rasteriser, moved to `device` on first use.

        It sits outside the module tree, so `Module.to` does not reach its buffers; they are the
        face list and the canonical per-vertex tables, all of them fixed, so moving once is enough.
        """
        k = self._kssf_holder[0]
        if k.faces.device != device:
            k.to(device)
        return k

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

    def _decode_active(self, feat, prevpos, nodes=None):
        """Assemble the 51D output from the root head and the fifteen joint decoders.

        `nodes` is `(B, 16, node_dim)` when the frontend emits one feature per MANO joint; joint
        `k` then reads node `k + 1`, since node 0 is the wrist. Otherwise every decoder reads the
        shared vector, which is the S10 arrangement.
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
        groups = None
        if self.encoder_kssf:
            betas_f, k_f = self._resolve_betas_K(prev, batch.betas, batch.camera_K)
            with torch.no_grad():
                # The field is read at the *previous* state, so nothing here needs the answer.
                fields = self.kssf_on(prev.device)(prev.float(), betas_f, k_f, self.pose_repr)
            from semkine.keg import kssf_event_channels
            extra, groups = kssf_event_channels(fields, events, halo=self.kssf_halo)
            if self.ablate_kssf:
                # G3's single-variable ablation: silence the geometry but keep the tokens, so the
                # same checkpoint answers "was the lift load-bearing" rather than "is a smaller
                # network worse". Grouping falls back to the spatial grid inside the frontend.
                extra = torch.zeros_like(extra)
                groups = None
        elif self.prev_render:
            betas_f, k_f = self._resolve_betas_K(prev, batch.betas, batch.camera_K)
            with torch.no_grad():
                rend = self._render_prev(prev.float(), betas_f, k_f)
            from semkine.encoder import query_render
            extra = query_render(rend.to(dtype=events.dtype, device=events.device), events)
        if self.joint_tokens:
            feat, nodes = self.event_encoder.forward_nodes(
                events, ptr, batch.delta_t_s, extra, groups)
        else:
            nodes = None
            feat = (self.event_encoder(events, ptr, batch.delta_t_s, extra, groups)
                    if groups is not None or self.encoder_kssf
                    else self.event_encoder(events, ptr, batch.delta_t_s, extra))
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
