#!/usr/bin/env python3
"""EventHands absolute-pose models (config-driven 12D / 51D)."""
from __future__ import annotations

import collections
import gc
import math
from typing import Any, Dict, Optional

import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl

from semkine import events as EV

from pose_repr import (
    axis_angle_to_quaternion,
    decode_to_mano_inputs,
    get_slices,
    split_losses,
)
from backbones import build_backbone
from sparse_trunk import input_mask


class BaseModel(pl.LightningModule):
    """Original EventHands loss layout preserved for 12D; generalized via slices.

    LOSS.TYPE picks the training objective:

      mse_51d      (default) the historical weighted elementwise MSE
      so3_trans_fk L_rot + L_trans + 2 L_FK, each term on the manifold of the
                   quantity it measures -- see :meth:`_so3_fk_loss`

    Two generic optimiser options, both off by default; with the defaults the optimiser is plain Adam and every saved
    checkpoint carries the raw weights only:

      TRAIN.WEIGHT_DECAY  wd > 0: `torch.optim.AdamW` (decoupled decay) instead of Adam; the weights of Conv / Linear
                          layers decay, BatchNorm parameters and every bias do not (:meth:`_param_groups`)
      TRAIN.EMA_DECAY     d > 0: an exponential moving average of the parameters and the BatchNorm running statistics,
                          updated after every optimiser step (:meth:`_register_ema_hook`). Every SAVED checkpoint's `state_dict` is the EMA model (so
                          evalx / select_checkpoint / load_from_checkpoint evaluate the EMA weights without change); the
                          raw weights go to `raw_state_dict` and a `--resume` continues from them (:meth:`_ema_checkpoint`,
                          :meth:`on_load_checkpoint`). Validation runs on the EMA weights.
    """

    #: Switches of training terms that were removed from this model (teacher distillation, self-conditioned rows,
    #: closed-loop rollout, acceleration loss, gain regulariser). They are constants: the frozen S45-TR N2 training
    #: step (`semkine/s45tr_n2.py`) reads them to refuse any recipe that uses another term.
    distill_weight = 0.0
    self_p = 0.0
    rollout_p = 0.0
    accel_weight = 0.0
    gain_reg_w = 0.0

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
        train_cfg = self.cfg.get("TRAIN", {}) or {}
        self.weight_decay = self._number("TRAIN.WEIGHT_DECAY", train_cfg.get("WEIGHT_DECAY", 0.0))
        if self.weight_decay < 0.0:
            raise ValueError(f"TRAIN.WEIGHT_DECAY must be a finite number >= 0, got {train_cfg.get('WEIGHT_DECAY')!r}")
        self.ema_decay = self._number("TRAIN.EMA_DECAY", train_cfg.get("EMA_DECAY", 0.0))
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(f"TRAIN.EMA_DECAY must be in [0, 1) (0 = off), got {train_cfg.get('EMA_DECAY')!r}")
        #: EMA state (plain attributes, not modules / buffers: they never enter `state_dict`)
        self._ema = None                #: name -> EMA tensor, built on the first update from the live weights
        self._ema_pending = None        #: name -> CPU tensor restored from a checkpoint, moved on first use
        self._ema_steps = 0             #: optimiser steps the EMA has seen (drives the warm-up of the decay)
        self._ema_backup = None         #: raw weights while the validation loop runs on the EMA weights

    @staticmethod
    def _number(name, value) -> float:
        """`value` as a finite float; booleans, strings and NaN / inf are configuration errors, not numbers."""
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
        return float(value)

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
            loss_trans = F.smooth_l1_loss(dec_p["transl"], dec_g["transl"])

            _, joints_p = self._fk(pred, betas)
            with torch.no_grad():
                _, joints_g = self._fk(y, betas)
            root = self.FK_ROOT_JOINT
            rel_p = joints_p - joints_p[:, root : root + 1]
            rel_g = joints_g - joints_g[:, root : root + 1]
            joint_dist = torch.norm(rel_p - rel_g, dim=-1)
            loss_fk = joint_dist.mean()

            loss = (
                self.rot_weight * loss_rot
                + self.trans_weight * loss_trans
                + self.fk_weight * loss_fk
            )
            with torch.no_grad():
                legacy, _ = self._mse_51d_loss(pred, y)
                angle_deg = torch.rad2deg(2.0 * dot.abs().clamp(max=1.0).acos()).mean()
        return loss, {
            "loss_rot": loss_rot,
            "loss_trans": loss_trans,
            "loss_fk": loss_fk,
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

    def _predict_batch(self, batch):
        """One forward that accepts either the legacy 5-tuple or an `EventPacketBatch`.

        Returns the unpacked batch as well, so a second forward can reuse the exact conditioning
        state this one saw.
        """
        packed = self._unpack_batch(batch)
        if hasattr(packed, "events"):
            pred = self.forward_packet(packed)
            return pred, packed.target, packed.betas, packed
        x, prevpos, y, betas, camera_K = packed
        return self(x, prevpos, betas=betas, camera_K=camera_K), y, betas, packed

    def on_save_checkpoint(self, checkpoint):
        """With TRAIN.EMA_DECAY > 0 the saved `state_dict` is the EMA model and the raw weights go to `raw_state_dict`."""
        if self.ema_decay > 0.0:
            self._ema_checkpoint(checkpoint)

    # ------------------------------------------------------------------------------------ weight EMA
    def _ema_live(self):
        """name -> the live tensor of every entry the EMA tracks: the floating-point parameters and the BatchNorm
        running statistics (the keys they have in `state_dict`); every other buffer is left out (constants and
        counters are copied from the raw weights when a checkpoint is written)."""
        live = {}
        for n, p in self.named_parameters():
            if p.is_floating_point():
                live[n] = p.data
        for n, b in self.named_buffers():
            if (n.endswith(".running_mean") or n.endswith(".running_var")) and b.is_floating_point():
                live[n] = b
        return live

    def _ema_ready(self, live):
        """The EMA tensors, built on first use: from the tensors restored by `on_load_checkpoint` if any, else from the
        live weights (an EMA that has seen nothing is the weights); moved to the live tensors' device if they moved."""
        if self._ema is None:
            pend = self._ema_pending or {}
            self._ema = {k: (pend[k].to(device=v.device, dtype=v.dtype).clone() if k in pend else v.detach().clone())
                         for k, v in live.items()}
            self._ema_pending = None
        else:
            for k, v in live.items():
                if self._ema[k].device != v.device:
                    self._ema[k] = self._ema[k].to(v.device)
        return self._ema

    @torch.no_grad()
    def ema_update(self):
        """One EMA step, `ema <- ema + (1 - d) (live - ema)` with d = min(EMA_DECAY, (1 + n) / (10 + n)) after n steps
        (the warm-up lets the average forget the random initial weights quickly). Called after every optimiser step
        (:meth:`_register_ema_hook`); `lerp` leaves an entry that equals the live value exactly unchanged."""
        if self.ema_decay <= 0.0:
            return
        live = self._ema_live()
        ema = self._ema_ready(live)
        n = self._ema_steps
        d = min(self.ema_decay, (1.0 + n) / (10.0 + n))
        for k, v in live.items():
            ema[k].lerp_(v, 1.0 - d)
        self._ema_steps = n + 1

    def _register_ema_hook(self, opt):
        """TRAIN.EMA_DECAY > 0: one EMA update after every `opt.step()` (a torch step post-hook on the optimiser that
        `configure_optimizers` returns). The step runs once per optimiser step, not per accumulated micro-batch, and
        Lightning's checkpoint callback runs after it, so a checkpoint written at step N contains an EMA that has seen
        the N-th update. A hook on the optimiser instead of an `optimizer_step` override leaves the module's Lightning
        hooks as they were (an override makes Lightning warn about accumulation in every run, EMA or not)."""
        if self.ema_decay > 0.0:
            opt.register_step_post_hook(lambda optimizer, args, kwargs: self.ema_update())
        return opt

    def ema_state_dict(self):
        """The model's full `state_dict` with the EMA values in the tracked entries (what a checkpoint's `state_dict`
        holds); raises if the EMA is off. Tensors are copies."""
        if self.ema_decay <= 0.0:
            raise RuntimeError("TRAIN.EMA_DECAY is 0: this model keeps no EMA")
        ema = self._ema_ready(self._ema_live())
        raw = self._raw_state_dict()
        return collections.OrderedDict((k, (ema[k].detach().clone() if k in ema else v)) for k, v in raw.items())

    def _raw_state_dict(self):
        """The raw (trained) weights as a `state_dict` copy, also when called from inside the validation loop, where the
        live tensors hold the EMA values and the raw ones are in `_ema_backup`."""
        sd = self.state_dict()
        out = collections.OrderedDict()
        for k, v in sd.items():
            if torch.is_tensor(v):
                v = (self._ema_backup[k] if self._ema_backup is not None and k in self._ema_backup else v)
                v = v.detach().clone()
            out[k] = v
        return out

    def _ema_checkpoint(self, checkpoint):
        """`on_save_checkpoint` with TRAIN.EMA_DECAY > 0: `state_dict` becomes the EMA model, the raw weights are kept in
        `raw_state_dict`, `ema` records the decay and the number of EMA steps (both dicts follow the key order of
        `checkpoint["state_dict"]`)."""
        sd = checkpoint.get("state_dict")
        if sd is None:
            return
        ema = self._ema_ready(self._ema_live())
        raw = self._raw_state_dict()
        raw = collections.OrderedDict((k, raw[k]) for k in sd)
        checkpoint["raw_state_dict"] = raw
        checkpoint["state_dict"] = collections.OrderedDict(
            (k, (ema[k].detach().clone() if k in ema else raw[k])) for k in sd)
        checkpoint["ema"] = {"decay": float(self.ema_decay), "steps": int(self._ema_steps)}

    @torch.no_grad()
    def _ema_swap_in(self):
        """Validation runs on the EMA weights: write them into the live tensors (the optimiser keeps its references),
        the raw ones wait in `_ema_backup`. Nothing happens without an EMA (EMA off, or no step taken yet)."""
        if self.ema_decay <= 0.0 or self._ema is None or self._ema_backup is not None:
            return
        live = self._ema_live()
        ema = self._ema_ready(live)
        self._ema_backup = {k: v.detach().clone() for k, v in live.items()}
        for k, v in live.items():
            v.copy_(ema[k])

    @torch.no_grad()
    def _ema_swap_out(self):
        if self._ema_backup is None:
            return
        for k, v in self._ema_live().items():
            v.copy_(self._ema_backup[k])
        self._ema_backup = None

    def on_train_start(self):
        """The EMA exists from the first step on: built from the (possibly restored) weights, so the first update already
        blends the initial weights, and the first validation has an EMA to run on."""
        if self.ema_decay > 0.0:
            self._ema_ready(self._ema_live())

    def on_validation_start(self):
        self._ema_swap_in()

    def on_validation_end(self):
        self._ema_swap_out()

    def _is_fit_resume(self):
        """True while a Trainer that is FITTING holds this model, i.e. `on_load_checkpoint` is called by `trainer.fit(...,
        ckpt_path=...)` (Lightning 1.9.5 calls it before `load_state_dict`). `load_from_checkpoint` (evaluation, the
        distillation teacher) builds a model without a Trainer, and `trainer.validate / test` are not FITTING."""
        from pytorch_lightning.trainer.states import TrainerFn
        trainer = getattr(self, "_trainer", None)
        return trainer is not None and getattr(getattr(trainer, "state", None), "fn", None) == TrainerFn.FITTING

    def _ema_resume(self, checkpoint):
        """`trainer.fit(ckpt_path=...)` on a checkpoint written with TRAIN.EMA_DECAY > 0: training continues from the RAW
        weights (`raw_state_dict` replaces `state_dict` before Lightning loads it), and the EMA is restored from the
        checkpoint's `state_dict` (which is the EMA) together with its step count, so it continues without a seam.
        Without `raw_state_dict` (a checkpoint of a non-EMA run) the weights are both the raw weights and the EMA's
        starting point. An EMA-free run that resumes an EMA checkpoint also takes the raw weights."""
        sd = checkpoint.get("state_dict")
        if sd is None:
            return
        if self.ema_decay > 0.0:
            self._ema = None
            self._ema_pending = {k: sd[k].detach().cpu().clone() for k in self._ema_live() if k in sd}
            self._ema_steps = int((checkpoint.get("ema") or {}).get("steps", 0)) if "raw_state_dict" in checkpoint else 0
        if "raw_state_dict" in checkpoint:
            checkpoint["state_dict"] = checkpoint["raw_state_dict"]

    def on_load_checkpoint(self, checkpoint):
        """`trainer.fit(ckpt_path=...)` on a checkpoint written with TRAIN.EMA_DECAY > 0 continues from its raw weights."""
        if self._is_fit_resume():
            self._ema_resume(checkpoint)

    def training_step(self, batch, batch_nb):
        pred, y, betas, packed = self._predict_batch(batch)
        loss, parts = self._compute_loss(pred, y, betas)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_mano_loss", parts["mano_loss"], sync_dist=True)
        self.log("train_pos_loss", parts["pos_loss"], sync_dist=True)
        self.log("train_rot_loss", parts["rot_loss"], sync_dist=True)
        self._log_so3_fk_parts("train", parts)
        self._log_route_stats("train")
        # Under AMP/bf16, log10 must run in fp32 to avoid illegal CUDA engines.
        loss_f = loss.float()
        out = loss_f.log10() if self.log10_loss else loss_f
        return out

    def validation_step(self, batch, batch_nb):
        pred, y, betas, packed = self._predict_batch(batch)
        loss, parts = self._compute_loss(pred, y, betas)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("val_mano_loss", parts["mano_loss"], sync_dist=True, on_epoch=True)
        self.log("val_pos_loss", parts["pos_loss"], sync_dist=True, on_epoch=True)
        self.log("val_rot_loss", parts["rot_loss"], sync_dist=True, on_epoch=True)
        self._log_so3_fk_parts("val", parts, on_epoch=True)
        self._log_route_stats("val", on_epoch=True)
        return loss

    def _log_route_stats(self, stage, on_epoch=False):
        """Mechanism readings of the xyz_mesh model: association hit rates and support sizes of the last packet batch."""
        if not getattr(self, "xyz_mesh", False):
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

    def _param_groups(self, wd):
        """AdamW groups for TRAIN.WEIGHT_DECAY: `weight` of every Conv / Linear layer decays by `wd`; BatchNorm
        weights and biases, every bias and any other parameter do not (weight 0). Frozen parameters stay out."""
        decay, plain, seen = [], [], set()
        for mod in self.modules():
            for pname, p in mod.named_parameters(recurse=False):
                if not p.requires_grad or id(p) in seen:
                    continue
                seen.add(id(p))
                if pname == "weight" and isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                    decay.append(p)
                else:
                    plain.append(p)
        n_train = sum(1 for p in self.parameters() if p.requires_grad)
        assert len(decay) + len(plain) == n_train, "every trainable parameter belongs to exactly one weight-decay group"
        return [{"params": decay, "weight_decay": wd}, {"params": plain, "weight_decay": 0.0}]

    def configure_optimizers(self):
        tcfg = self.cfg.get("TRAIN", {})
        lr = float(tcfg.get("LR", 1e-3))
        # With nothing frozen this is every trainable parameter in the module's own order.
        if self.weight_decay > 0.0:
            # TRAIN.WEIGHT_DECAY (DT2): decoupled decay on the Conv / Linear weights only
            opt = optim.AdamW(self._param_groups(self.weight_decay), lr=lr, weight_decay=self.weight_decay)
        else:
            opt = optim.Adam([p for p in self.parameters() if p.requires_grad], lr=lr)
        self._register_ema_hook(opt)
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

    MODEL.CNN_BACKBONE names the dense trunk `rn`: a key of `backbones.BACKBONES` (default `resnet18`,
    torchvision's resnet18 itself) or of `backbones.SPARSE_BACKBONES` (the S43 sparse trunk, evaluated on
    the active sites of [LNES | render] only). Dense path only (no ENCODER). Not `BACKBONE`: that key is
    dead and sits in many configs.

    MODEL.ENCODER `xyz_mesh` (S37-XYZ, the C1 / S45 recipe): the previous state's mesh in camera XYZ, events
    associated along their rays, one shared relation encoder, a MeshGNN and one common readout writing the
    51D delta (`semkine.xyz_mesh`); it consumes `EventPacketBatch`es through :meth:`forward_packet`.
    """

    #: rasterizable faces and their channel width
    RENDER_FACES = {"sil": 1, "inv": 1, "semsil": 1}
    #: every MODEL key the model or the training script understands
    MODEL_KEYS = frozenset(
        {
            "BACKBONE", "POSE_REPR", "OUTPUT_DIM", "MANO_NCOMPS", "PREDICT_BETA",
            "PREDICT_DELTA", "PREVPOS_EMBED", "PREV_RENDER", "ZERO_EVENT_GATE",
            # S37-XYZ (docs/S37_XYZ_SCREEN_PREREG_20260929.md); XYZ_GEOM only accepts true
            "XYZ_BAND_PX", "XYZ_FRONT_PX", "XYZ_Z_TOL", "XYZ_K", "XYZ_GEOM", "XYZ_REL_HIDDEN",
            "XYZ_SLOT_DIM", "XYZ_READOUT_HIDDEN",
            # XAG (docs/EXPERIMENT_RECORDS.md#xag-paired-c1-20261018): source-label routing, opt-in
            "XAG_ASSOC", "XAG_GNN",
            "RENDER_CHANNELS", "RENDER_H", "RENDER_W", "RENDER_SCALE",
            "RENDER_CHUNK", "RENDER_DEPTH_TOL", "INIT_FROM",
            "ENCODER", "ENCODER_HIDDEN", "ENCODER_LAYERS", "ENCODER_MAX_NODES",
            "CNN_BACKBONE",
            # the S37 per-joint decoder heads were removed; the key stays whitelisted for the frozen S45-line configs
            # that carry `ACTIVE_HEAD: false`, and only `false` is accepted
            "ACTIVE_HEAD",
        }
    )
    #: every TRACK key the model or the dataset understands. Whitelisted so that a key nothing reads cannot turn a
    #: config-only "ablation" into a silent copy of the baseline.
    TRACK_KEYS = frozenset(
        {
            "PREV_NOISE_T", "PREV_NOISE_R", "PREV_NOISE_POSE", "PREV_NOISE_MODE",
            "PREV_NOISE_LARGE_T", "PREV_NOISE_LARGE_R", "PREV_NOISE_LARGE_POSE",
            "PREV_NOISE_P_SMALL", "PREV_NOISE_P_LARGE", "PREV_NOISE_P_CORR",
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
        if model_cfg.get("ACTIVE_HEAD", False):
            raise ValueError("MODEL.ACTIVE_HEAD (the per-joint decoder heads) was removed; only `false` is accepted")
        track_cfg = self.cfg.get("TRACK", {}) or {}
        unknown = sorted(set(track_cfg) - self.TRACK_KEYS)
        if unknown:
            raise ValueError(f"unknown TRACK keys: {unknown}")
        self.predict_delta = bool(model_cfg.get("PREDICT_DELTA", False))
        self.prevpos_embed = bool(model_cfg.get("PREVPOS_EMBED", False))
        self.prev_render = bool(model_cfg.get("PREV_RENDER", False))
        self.zero_event_gate = bool(model_cfg.get("ZERO_EVENT_GATE", self.prev_render))
        self.render_channels = self._parse_render_channels(model_cfg)
        # Two planes per event face. `DATA.EVENT_CHANNELS` is unset in every existing config, which
        # resolves to `("last",)` -- plain LNES, two planes, the historical width.
        self.event_channels = EV.event_channels(self.cfg)
        in_ch = (2 * len(self.event_channels)
                 + sum(self.RENDER_FACES[c] for c in self.render_channels))
        # Keep parameter names conv1 / rn for loading original 12D checkpoints. The xyz_mesh model discards this
        # layer below, but its construction draws random numbers that every initial weight after it depends on.
        self.conv1 = nn.Conv2d(in_ch, 3, kernel_size=3, padding=1)

        self.encoder_name = str(model_cfg.get("ENCODER") or "").lower()
        if self.encoder_name in ("", "none", "lnes"):
            self.encoder_name = ""
        elif self.encoder_name != "xyz_mesh":
            raise ValueError(f"MODEL.ENCODER must be xyz_mesh (or absent for the dense LNES tracker), got "
                             f"{model_cfg.get('ENCODER')!r}")
        # S37-XYZ (2026-09-29, the user's drawing). prev's full mesh in camera XYZ, events associated
        # along their rays, a shared relation encoder, one MeshGNN on the face 1-ring gated by the 3D
        # edges, 16 LBS slots + the residual slot, one common readout that also reads prev[3:51] and
        # writes the only 51D delta (`semkine.xyz_mesh`). No EventGNN, no prev_mlp, no per-joint heads.
        self.xyz_mesh = self.encoder_name == "xyz_mesh"
        from semkine.xag import validate_config as validate_xag_config
        self.xag_assoc_mode, self.xag_gnn_mode = validate_xag_config(model_cfg, xyz_mesh=self.xyz_mesh)
        self.xag_enabled = "XAG_ASSOC" in model_cfg or "XAG_GNN" in model_cfg
        if self.xyz_mesh and self.prev_render:
            raise ValueError("ENCODER=xyz_mesh is the only conditioning path of its arm")
        if self.xyz_mesh and self.prevpos_embed:
            raise ValueError("ENCODER=xyz_mesh has one common readout: set PREVPOS_EMBED false")
        dense_trunk = not self.encoder_name
        self.cnn_backbone = model_cfg.get("CNN_BACKBONE", "resnet18")
        if not isinstance(self.cnn_backbone, str):
            raise ValueError(f"MODEL.CNN_BACKBONE must be a backbone name (a string), got {self.cnn_backbone!r}")
        if "CNN_BACKBONE" in model_cfg and not dense_trunk:
            raise ValueError("MODEL.CNN_BACKBONE replaces the dense ResNet18 trunk; it is only legal for the dense "
                             "path (no ENCODER)")
        if self.xyz_mesh:
            from semkine.xyz_mesh import XYZMeshNet
            self._build_mano(model_cfg)
            assert self.output_dim == 51 and self.predict_delta, "xyz_mesh writes a 51D delta"
            self.xyz_band_px = float(model_cfg.get("XYZ_BAND_PX", 16.0))
            self.xyz_front_px = float(model_cfg.get("XYZ_FRONT_PX", 3.0))
            self.xyz_z_tol = float(model_cfg.get("XYZ_Z_TOL", 0.01))
            self.xyz_k = int(model_cfg.get("XYZ_K", 16))
            self.xyz_max_nodes = int(model_cfg.get("ENCODER_MAX_NODES", 2048))
            if not bool(model_cfg.get("XYZ_GEOM", True)):
                raise ValueError("MODEL.XYZ_GEOM false (the C0 control without the explicit 3D channels) was removed; "
                                 "the xyz_mesh model always reads its geometry")
            self.xyz = XYZMeshNet(
                self.mano.weights, self.mano.f,
                hidden=int(model_cfg.get("ENCODER_HIDDEN", 128)),
                n_layers=int(model_cfg.get("ENCODER_LAYERS", 3)),
                rel_hidden=int(model_cfg.get("XYZ_REL_HIDDEN", 64)),
                slot_dim=int(model_cfg.get("XYZ_SLOT_DIM", 32)),
                readout_hidden=int(model_cfg.get("XYZ_READOUT_HIDDEN", 256)))
            self.conv1 = None
            self.rn = None
        else:
            # `resnet18` (the default) is torchvision's resnet18(num_classes=OUTPUT_DIM) itself: same call, same draws
            self.rn = build_backbone(self.cnn_backbone, self.output_dim)
        # S43 (docs/EXPERIMENT_RECORDS.md#s43-prereg): a `sparse_*` trunk (`sparse_trunk.SparseResNet`) is evaluated on
        # the active sites of [LNES | render] only; `forward` hands it the input's active-site mask
        self.sparse_trunk = bool(getattr(self.rn, "takes_mask", False))
        # PREV_RENDER: false is the S43 ablation A1 (docs/EXPERIMENT_RECORDS.md#s43-ablation): the trunk sees the LNES alone
        # and the previous state enters only through prev_mlp and the delta's skip connection
        if self.sparse_trunk and not self.predict_delta:
            raise ValueError("a sparse CNN_BACKBONE is the S43 trunk: it needs PREDICT_DELTA")
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
        if self.prev_render or (self.sparse_trunk and self.loss_type == "so3_trans_fk"):
            # (S43 ablation A1: the SO3 + FK loss needs MANO even when nothing is rendered)
            self._build_mano(model_cfg)
        if self.prev_render:
            self._register_vertex_codes()
        if self.xyz_mesh:
            #: diagnostics of the last `forward_packet`, for the training loop and the probes
            self.route_stats = {}
            #: set at inference to zero every event record, for the same-checkpoint ablation that decides
            #: whether the evidence is load-bearing
            self.ablate_evidence = False

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
            if self.xag_enabled:
                # An empty rank must still participate in DDP reduction. Keep
                # every trainable parameter connected with an exact zero term.
                self.xyz.xag_last_capture = {}
                self.xyz_last = {"xag_empty": True}
                zero = prev.new_zeros(())
                if self.training:
                    for parameter in self.parameters():
                        if parameter.requires_grad and parameter.numel():
                            zero = zero + parameter.reshape(-1)[0] * 0.0
                return prev.new_zeros(B, 51) + zero
            return prev.new_zeros(B, 51)
        betas_f, k_f = self._resolve_betas_K(prev, batch.betas, batch.camera_K)
        src, mask = stride_sample(ptr, self.xyz_max_nodes)
        tok = event_tokens(events, ptr, batch.delta_t_s, self.render_h, self.render_w)
        tok = tok[src.reshape(-1)].view(B, -1, tok.shape[-1]) * mask.unsqueeze(-1)
        with torch.no_grad():
            verts, _ = self._fk(prev.float(), betas_f)                               # (B, 778, 3)
            fx, fy, cx, cy = self._intrinsics(k_f)
            ev = events[src.reshape(-1)].view(B, -1, events.shape[1])
            px, py = ev[..., EV_X].float(), ev[..., EV_Y].float()
            d = torch.stack([(px - cx[:, None]) / fx[:, None], (py - cy[:, None]) / fy[:, None],
                             torch.ones_like(px)], dim=-1)
            f_px = (fx * fy).sqrt()
            assoc = associate_rays(d, mask, verts, self.mano.f, self.xyz.vert_faces,
                                   band=self.xyz_band_px / f_px, front_tol=self.xyz_front_px / f_px,
                                   z_tol=self.xyz_z_tol, k=self.xyz_k)
            xag_aux = {}
            if self.xag_assoc_mode != "base":
                from semkine.xag import route_association, sample_sources
                source = sample_sources(batch, src, mask)
                assoc, xag_aux = route_association(
                    self.xag_assoc_mode, assoc, d, mask, verts, self.mano.f, self.xyz.vert_faces,
                    self.xyz.lbs_w, self.xyz_band_px / f_px, self.xyz_front_px / f_px,
                    self.xyz_z_tol, source_part=source["xag_part"], source_face=source["xag_face"],
                    source_bary=source["xag_bary"], source_valid=source["xag_valid"], k=self.xyz_k)
                xag_aux.update(source)
        delta, stats, aux = self.xyz(tok, d, mask, verts, assoc, prev[:, 3:51],
                                     ablate=self.ablate_evidence, capture=getattr(self, "xyz_keep_aux", False))
        self.route_stats = stats
        if getattr(self, "xyz_keep_aux", False):
            #: probe-only: the association and the pooled slots of the last packet batch
            self.xyz_last = {"assoc": assoc, "d": d, "mask": mask, "verts": verts, "tok": tok,
                             "src": src, "delta": delta.detach(), **{k: v.detach() for k, v in aux.items()},
                             **xag_aux}
        return delta

    def forward_packet(self, batch):
        """Raw-event forward for an `EventPacketBatch` (MODEL.ENCODER xyz_mesh). Empty packets return bitwise `prev`."""
        if not self.encoder_name:
            raise RuntimeError("forward_packet requires MODEL.ENCODER")
        prev = batch.prev_state
        out = self._xyz_mesh_forward(batch, prev)
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
        if self.sparse_trunk:
            m0 = input_mask(x)
            feats, _ = self.rn.pooled(self.conv1(x) * m0, m0)
            out = self.rn.fc(feats)
        else:
            out = self.rn(self.conv1(x))
        if self.prevpos_embed:
            out = out + self.prev_mlp(prevpos.to(out.dtype))
        if self.predict_delta:
            delta = out
            if self.zero_event_gate:
                empty = lnes.reshape(B, -1).abs().sum(dim=1, keepdim=True) <= 0
                delta = torch.where(empty, torch.zeros_like(delta), delta)
            out = delta + prevpos.to(out.dtype)
        return out
