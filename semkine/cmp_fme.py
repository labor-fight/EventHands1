r"""FastMETRO with event input (Cho et al., ECCV 2022), the event-only hand baseline of EvRGBHand (Jiang et al., CVPR 2024),
ported to the CMP protocol (docs/EXPERIMENT_RECORDS.md#cmp-amend8-20261110; tools/cmp/ext_fme.py).

The network is the official code (semkine/fastmetro_official/, vendored from the EvRGBHand repository): the configuration
of `src/configs/config/fastmetro_ev.yaml` -- ImageNet-pretrained ResNet-34 with frozen BatchNorm (the official event
backbone), 1 x 1 conv to 512, two transformer encoder-decoders (512 / 2048 then 128 / 512, 3 + 3 layers, 8 heads, dropout
0.1) over 21 joint tokens + 195 coarse-vertex tokens with the MANO-195 adjacency as the decoder's attention mask, a linear
3-D regressor and the learnable 195 -> 778 upsampling. Outputs: 21 joints, 195 coarse and 778 fine vertices (metres); no
MANO parameters.

The official recipe is kept where the protocol allows (tools/cmp/ext_fme.py `configure` writes it into CMP.FME):
  loss       `Loss.get_loss_mano` as the FastMETRO arch runs it (no camera, heat-map or auxiliary outputs): SmoothL1
             (beta 0.01, mean) on root-aligned quantities, 100 vertices + 100 coarse vertices + 500 joint tokens +
             500 joints regressed from the fine mesh; ground truth aligned to its wrist, joint tokens to their own
             wrist, predicted vertices and regressed joints to the regressed wrist
  optimiser  AdamW (0.9, 0.999), weight decay 1e-4 on every parameter, lr 1e-4 and 1e-5 for `backbone.layer*`, gradient-norm
             clip 1.0, fp32; cosine to the official eta_min 1e-6 (per group) after CMP's warm-up, per step
  input      the official LNES frame [polarity 0, polarity 1, zeros] (normalised time stamps in [0, 1], last event
             wins): this repository's legacy LNES plus a zero plane
  photometry the official event degrader (`src/utils/augment.py`, event_photometry of the config): Gaussian noise with
             probability 0.1 (std U(0, 0.1), clipped to [0, 1]) and salt-and-pepper with probability 0.1 (rate 0.05:
             polarity plane 0 or 1 set to 1), per sample, here on the GPU
What the protocol changes (registered with the arm): the full 240 x 180 frame zero-padded to 256 x 192 instead of the
192 x 192 crop around the annotated hand box, a fixed 100 ms window, ground-truth joints / vertices from MANO FK of the
51-D label with the sequence betas (the joint tokens are supervised in ManoLayer's OpenPose order), the coarse ground
truth by the official downsampling matrix, and the geometric augmentation as domain randomisation of the full frame.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from model import BaseModel, MNISTModel

from .fastmetro_official.fastmetro import DATA, FastMETRO_Hand_Network

N_JOINTS, N_COARSE, N_FINE = 21, 195, 778


def official_config(net):
    """CMP.FME.NET -> the `model` block FastMETRO_Hand_Network reads (fastmetro_ev.yaml)."""
    return {"model": {"modality": "event", "backbone": str(net["BACKBONE"]), "pretrained": bool(net["PRETRAINED"]),
                      "pos_type": "sine", "hidden_dim_1": int(net["HIDDEN_DIM_1"]),
                      "dim_feedforward_1": int(net["DIM_FEEDFORWARD_1"]), "hidden_dim_2": int(net["HIDDEN_DIM_2"]),
                      "dim_feedforward_2": int(net["DIM_FEEDFORWARD_2"]), "enc_layers": int(net["ENC_LAYERS"]),
                      "dec_layers": int(net["DEC_LAYERS"]), "nhead": int(net["NHEAD"]), "dropout": float(net["DROPOUT"])}}


def joint_regressor(mano_npz):
    """(21, 778): MANO's J_regressor plus the five fingertip vertices (the official `MANO.get_3d_joints`), in ManoLayer's
    OpenPose order. On posed vertices it gives the five tips of ManoLayer's joints exactly and an approximation of the
    wrist and the other fifteen (they are FK joints, not linear in the posed vertices)."""
    d = np.load(mano_npz, allow_pickle=False)
    tips = np.eye(N_FINE, dtype=np.float32)[d["fingertip_vertex_ids"]]
    return np.concatenate([d["J_regressor"].astype(np.float32), tips])[d["mano16_tips_to_openpose21"]]


class FMEModel(BaseModel):
    """CMP.MODEL fme on the frame dataset's batches (lnes, prev, 51-D label, betas, K); the previous state is not read."""

    KIND = "fme"
    _build_mano = MNISTModel._build_mano
    _fk = MNISTModel._fk
    _resolve_betas_K = MNISTModel._resolve_betas_K
    _intrinsics = MNISTModel._intrinsics
    set_hand_context = MNISTModel.set_hand_context

    def __init__(self, cfg=None, num_classes=None):
        super().__init__(cfg)
        c = self.cfg.get("CMP", {}) or {}
        if c.get("MODEL") != self.KIND or "FME" not in c:
            raise ValueError("FMEModel: CMP.MODEL fme with its CMP.FME block")
        if self.pose_repr != "mano_full_axis_angle" or self.output_dim != 51:
            raise ValueError("FMEModel: the labels are 51-D mano_full_axis_angle states")
        f = c["FME"]
        self.net = FastMETRO_Hand_Network(official_config(f["NET"]))
        self.pad = tuple(int(v) for v in f["PAD"])                         # left, right, top, bottom
        self.loss_w = {k: float(v) for k, v in f["LOSS"].items() if k != "BETA"}
        self.loss_beta = float(f["LOSS"]["BETA"])
        self.opt = dict(f["OPT"])
        self.photo = dict(f["PHOTOMETRY"])
        self._build_mano(self.cfg["MODEL"])
        self.register_buffer("down", torch.from_numpy(np.load(DATA / "mano_195_downsampling.npy")), persistent=False)
        self.register_buffer("jreg", torch.from_numpy(joint_regressor(self.cfg["MANO"]["NPZ"])), persistent=False)
        self._ctx_betas = None
        self._ctx_K = None

    # ------------------------------------------------------------------ network
    def frame(self, lnes):
        """(B, H, W, 2) LNES -> the official event frame (B, H', W', 3): a zero third plane, zero-padded borders."""
        l, r, t, b = self.pad
        return F.pad(lnes.float(), (0, 1, l, r, t, b))

    def predict(self, lnes):
        """(B, H, W, 2) LNES -> {'pred_3d_joints' (B, 21, 3), 'pred_vertices_sub' (B, 195, 3), 'pred_vertices' (B, 778, 3)}."""
        return self.net([{"ev_frames": [self.frame(lnes)]}])[0][0]

    def forward(self, x, prevpos=None, betas=None, camera_K=None):
        return self.predict(x)

    def joints_from_mesh(self, vertices):
        return torch.einsum("jv,bvk->bjk", self.jreg, vertices)

    # ------------------------------------------------------------------ official event photometry
    def degrade(self, x):
        """`EvRGBDegrader` on event frames (B, H, W, 2): Gaussian noise, then salt-and-pepper, per sample."""
        p = self.photo
        B, H, W, _ = x.shape
        gauss = torch.rand(B, device=x.device) < float(p["GAUSS_P"])
        if gauss.any():
            # the official `var = (rand * var[1] - var[0]) + var[0]`, i.e. U(0, var[1]), used as the noise std
            std = torch.rand(B, 1, 1, 1, device=x.device) * float(p["GAUSS_VAR"][1])
            x = torch.where(gauss.view(B, 1, 1, 1), (x + torch.randn_like(x) * std).clamp(0.0, 1.0), x)
        sp = torch.rand(B, device=x.device) < float(p["SALT_PEPPER_P"])
        if sp.any():
            flipped = (torch.rand(B, H, W, device=x.device) < float(p["SALT_PEPPER_RATE"])) & sp.view(B, 1, 1)
            salted = torch.rand(B, H, W, device=x.device) > 0.5
            one = torch.ones((), dtype=x.dtype, device=x.device)
            x = torch.stack([torch.where(flipped & salted, one, x[..., 0]),
                             torch.where(flipped & ~salted, one, x[..., 1])], dim=-1)
        return x

    # ------------------------------------------------------------------ official loss (Loss.get_loss_mano, FastMETRO arch)
    def targets(self, y, betas):
        """51-D labels -> ground-truth fine vertices, coarse vertices and 21 joints (metres, event-camera frame)."""
        with torch.autocast(device_type=y.device.type, enabled=False):
            verts, joints = self._fk(y.float(), betas.float())
            return verts, torch.einsum("sv,bvk->bsk", self.down, verts), joints

    def mesh_loss(self, out, y, betas):
        gv, gs, gj = self.targets(y, betas)
        root = gj[:, :1]
        pj, ps, pv = (out[k].float() for k in ("pred_3d_joints", "pred_vertices_sub", "pred_vertices"))
        pjm = self.joints_from_mesh(pv)
        pj_a, pjm_a = pj - pj[:, :1], pjm - pjm[:, :1]
        sl1 = lambda a, b: F.smooth_l1_loss(a, b, beta=self.loss_beta)  # noqa: E731
        parts = {"loss_vertices": sl1(pv - pjm[:, :1], gv - root), "loss_vertices_sub": sl1(ps - pjm[:, :1], gs - root),
                 "loss_3d_joints": sl1(pj_a, gj - root), "loss_3d_joints_reg": sl1(pjm_a, gj - root)}
        w = self.loss_w
        loss = (w["VERTICES"] * parts["loss_vertices"] + w["VERTICES_SUB"] * parts["loss_vertices_sub"]
                + w["JOINTS_3D"] * parts["loss_3d_joints"] + w["JOINTS_3D_FROM_MESH"] * parts["loss_3d_joints_reg"])
        with torch.no_grad():
            parts["joint_error_mm"] = (pj_a - (gj - root)).norm(dim=-1).mean() * 1000.0
            parts["vertex_error_mm"] = ((pv - pjm[:, :1]) - (gv - root)).norm(dim=-1).mean() * 1000.0
        return loss, parts

    # ------------------------------------------------------------------ training
    def training_step(self, batch, batch_nb):
        x, _, y, betas, _ = self._unpack_batch(batch)
        loss, parts = self.mesh_loss(self.predict(self.degrade(x.float())), y, betas)
        self.log("train_loss", loss, prog_bar=True, batch_size=x.shape[0])
        for key, value in parts.items():
            self.log(f"train_{key}", value, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_nb):
        x, _, y, betas, _ = self._unpack_batch(batch)
        loss, parts = self.mesh_loss(self.predict(x.float()), y, betas)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=x.shape[0])
        for key in ("joint_error_mm", "vertex_error_mm"):
            self.log(f"val_{key}", parts[key], on_epoch=True, batch_size=x.shape[0])
        return loss

    def configure_optimizers(self):
        t, o = self.cfg["TRAIN"], self.opt
        lr, lr_backbone, eta_min = float(t["LR"]), float(o["LR_BACKBONE"]), float(o["ETA_MIN"])
        names = tuple(o["BACKBONE_NAMES"])
        named = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        groups = [{"params": [p for n, p in named if not any(b in n for b in names)], "lr": lr},
                  {"params": [p for n, p in named if any(b in n for b in names)], "lr": lr_backbone}]
        opt = torch.optim.AdamW(groups, lr=lr, betas=tuple(float(b) for b in o["BETAS"]),
                                weight_decay=float(o["WEIGHT_DECAY"]))
        warmup, total = int(t["WARMUP_STEPS"]), int(t["MAX_STEPS"])

        def factor(floor):
            def f(step):
                if warmup > 0 and step < warmup:
                    return float(step + 1) / float(warmup)
                p = min(max((step - warmup) / float(max(total - warmup, 1)), 0.0), 1.0)
                return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * p))
            return f
        sched = torch.optim.lr_scheduler.LambdaLR(opt, [factor(eta_min / lr), factor(eta_min / lr_backbone)])
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def configure_gradient_clipping(self, optimizer, optimizer_idx=None, gradient_clip_val=None,
                                    gradient_clip_algorithm=None):
        self.clip_gradients(optimizer, gradient_clip_val=float(self.opt["CLIP_NORM"]), gradient_clip_algorithm="norm")


def model_class(kind):
    if kind != FMEModel.KIND:
        raise ValueError(f"semkine.cmp_fme has no model kind {kind!r}")
    return FMEModel


__all__ = ["FMEModel", "joint_regressor", "model_class", "official_config"]
