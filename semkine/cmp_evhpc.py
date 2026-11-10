r"""EvHandPose with its hand crop, online (docs/EXPERIMENT_RECORDS.md#cmp-amend1-20261110).

`semkine/cmp_evhp.py`'s port (same official FlowNet / Backbone, same losses and schedule) on 128 x 128 crops
(`semkine/cmp_seq_crop.py`) instead of the padded full frame, with the official translation decoding of a cropped view
(core/model/models.py `forward`): the predictor's translation t' = raw x 0.3 + mean is in the crop's virtual camera
and t = (t'_x + t'_z tl_x / f_x, t'_y + t'_z tl_y / f_y, t'_z s) with s the crop scale, tl its top-left and f the
event-frame focal lengths (the box of the packet whose state is decoded). The mean is the crop-frame translation of a
centred hand, (-(c_x - 64) z' / f_x, -(c_y - 64) z' / f_y, z') at the training depth (z' = 0.52 m). The flow pseudo
ground truth is rendered with the crop's intrinsics (f s, c - tl) at 128 x 128. CMP.MODEL epcf = the FlowNet stage,
epc = the pose stage (FlowNet frozen from epcf).
"""
from __future__ import annotations

import torch

from model import BaseModel

from .cmp_evhp import DEPTH_TOL, FLOW_CHUNK, FLOW_SCALE_PX, TRANS_SCALE, EvHPModel, _bary, official_config
from .evhandpose_official.backbone import Backbone, FlowNet

TRANS_MEAN_CROP = (-0.13, -0.06, 0.52)
ROLE = {"epcf": "epf", "epc": "ep"}


class EvHPCropModel(EvHPModel):
    def __init__(self, cfg=None, num_classes=None):
        BaseModel.__init__(self, cfg)
        c = self.cfg["CMP"]
        if c["MODEL"] not in ROLE or self.output_dim != 51 or self.loss_type != "so3_trans_fk":
            raise ValueError("EvHPCropModel: CMP.MODEL epcf / epc, 51-D state, so3_trans_fk")
        self.cmp_kind = c["MODEL"]
        self.kind = ROLE[self.cmp_kind]                                     # the stage role of cmp_evhp.py's methods
        seq = c.get("SEQ", {}) or {}
        self.voxel_bins = int(seq.get("VOXEL_BINS", 5))
        self.annot_len = int(seq.get("ANNOT_LEN", 3))
        self.crop_size = int(c["CROP"]["SIZE"])
        self.ocfg = official_config(self.annot_len)
        self.flow_encoder = FlowNet(self.ocfg)
        self.infer_encoder = Backbone(self.ocfg)
        if self.kind == "ep":
            for p in self.flow_encoder.parameters():
                p.requires_grad = False
        self._build_mano(self.cfg["MODEL"])
        self.register_buffer("trans_mean", torch.tensor(TRANS_MEAN_CROP), persistent=False)
        self.register_buffer("bary", _bary(), persistent=False)
        self._ctx_betas = None
        self._ctx_K = None

    # ------------------------------------------------------------------ network pieces (no padding on the crop)
    def flows(self, voxel):
        out = self.flow_encoder(voxel)
        return out.reshape(out.shape[0], -1, out.shape[-2], out.shape[-1])

    def packet_features(self, lnes, voxel):
        B = self.infer_encoder
        if self.kind == "ep":
            self.flow_encoder.eval()
            with torch.no_grad():
                flow = torch.cat([self.flows(v)[:, -2:] for v in voxel.split(FLOW_CHUNK)])
        else:
            flow = self.flows(voxel)[:, -2:]
        x = torch.cat([lnes.permute(0, 3, 1, 2), flow.to(lnes.dtype)], dim=1)
        for enc in B.event_encoders:
            x = enc(x)
        for enc in B.fusion_encoders:
            x = enc(x)
        return x

    def sequence_head(self, feats, crop):
        """(B, T, 512, h, w) features and (B, T, 7) crops -> (B, T', 51) states of the last T' = annot_len packets."""
        B = self.infer_encoder
        x = feats.permute(0, 2, 1, 3, 4)
        for enc in B.sequence_encoders:
            x = enc(x)
        x = x.permute(0, 2, 1, 3, 4)[:, -self.annot_len:]
        b, t = x.shape[:2]
        x = x.reshape(b * t, *x.shape[2:])
        for i, layer in enumerate(B.predictors):
            x = layer(x)
            if i == 0:
                x = x.view(x.shape[0], -1)
        x = x.view(b, t, -1).float()
        c = crop[:, -t:].float()
        tv = x[..., 45:48] * TRANS_SCALE + self.trans_mean
        s, tlx, tly, fx, fy = c[..., 0], c[..., 1], c[..., 2], c[..., 3], c[..., 4]
        trans = torch.stack([tv[..., 0] + tv[..., 2] * tlx / fx, tv[..., 1] + tv[..., 2] * tly / fy, tv[..., 2] * s], -1)
        return torch.cat([trans, x[..., 48:51], x[..., :45]], dim=-1)

    def sequence_state(self, feats, crop):
        return self.sequence_head(feats, crop)[:, -1]

    # ------------------------------------------------------------------ flow pseudo ground truth in the crop
    @torch.no_grad()
    def mesh_flow_crop(self, s0, s1, betas, crop):
        """cmp_evhp.py `mesh_flow` with the crop's intrinsics (f s, c - tl) on SIZE x SIZE; crop (N, 7)."""
        with torch.autocast(device_type=s0.device.type, enabled=False):
            v0, _ = self._fk(s0.float(), betas.float())
            v1, _ = self._fk(s1.float(), betas.float())
            faces = self.mano.f.long()
            p0 = torch.einsum("sk,nfkd->nfsd", self.bary, v0[:, faces]).reshape(v0.shape[0], -1, 3)
            p1 = torch.einsum("sk,nfkd->nfsd", self.bary, v1[:, faces]).reshape(v1.shape[0], -1, 3)
            c = crop.float()
            fx, fy = c[:, 3] * c[:, 0], c[:, 4] * c[:, 0]
            cx, cy = c[:, 5] - c[:, 1], c[:, 6] - c[:, 2]
            z0, z1 = p0[..., 2].clamp(min=1e-6), p1[..., 2].clamp(min=1e-6)
            u0 = fx[:, None] * p0[..., 0] / z0 + cx[:, None]
            w0 = fy[:, None] * p0[..., 1] / z0 + cy[:, None]
            u1 = fx[:, None] * p1[..., 0] / z1 + cx[:, None]
            w1 = fy[:, None] * p1[..., 1] / z1 + cy[:, None]
            N = u0.shape[0]
            H = W = self.crop_size
            ui, vi = u0.round().long(), w0.round().long()
            ok = (p0[..., 2] > 1e-6) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            pix = (torch.arange(N, device=u0.device)[:, None] * (H * W) + vi * W + ui)[ok]
            depth = torch.full((N * H * W,), 1.0e6, device=u0.device)
            depth.scatter_reduce_(0, pix, z0[ok], reduce="amin", include_self=True)
            front = z0[ok] <= depth[pix] + DEPTH_TOL
            pix = pix[front]
            du = (u1 - u0)[ok][front] / FLOW_SCALE_PX
            dv = (w1 - w0)[ok][front] / FLOW_SCALE_PX
            acc = torch.zeros((N * H * W, 2), device=u0.device)
            cnt = torch.zeros((N * H * W, 1), device=u0.device)
            acc.index_add_(0, pix, torch.stack([du, dv], -1))
            cnt.index_add_(0, pix, torch.ones_like(du).unsqueeze(-1))
            flow = (acc / cnt.clamp(min=1.0)).view(N, H, W, 2).permute(0, 3, 1, 2)
        return flow.contiguous()

    # ------------------------------------------------------------------ training
    def training_step(self, batch, batch_nb):
        if not isinstance(batch, dict) or "crop" not in batch:
            raise TypeError("cropped EvHandPose trains on semkine.cmp_seq_crop batches")
        lnes, voxel, states, crop = batch["lnes"], batch["voxel"], batch["states"], batch["crop"]
        B, K = lnes.shape[:2]
        if self.kind == "epf":
            pred = self.flows(voxel.reshape(B * K, *voxel.shape[2:])).float()
            gt = self.mesh_flow_crop(states[:, :-1].reshape(B * K, -1), states[:, 1:].reshape(B * K, -1),
                                     batch["betas"].unsqueeze(1).expand(B, K, -1).reshape(B * K, -1),
                                     crop.reshape(B * K, -1))
            gt = gt.repeat(1, pred.shape[1] // 2, 1, 1)
            loss = (gt - pred).square().sum() / (B * K * pred.shape[1])
            self.log("train_loss", loss, prog_bar=True, batch_size=B)
            self.log("train_flow_epe_px", ((gt[:, -2:] - pred[:, -2:]).square().sum(1).sqrt().mean()) * FLOW_SCALE_PX,
                     batch_size=B)
            return loss.float().clamp_min(1e-12).log10()
        feats = self.packet_features(lnes.reshape(B * K, *lnes.shape[2:]), voxel.reshape(B * K, *voxel.shape[2:]))
        preds = self.sequence_head(feats.view(B, K, *feats.shape[1:]), crop)
        T = preds.shape[1]
        betas = batch["betas"].unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
        loss, parts = self._compute_loss(preds.reshape(B * T, -1), states[:, -T:].reshape(B * T, -1), betas)
        self.log("train_loss", loss, prog_bar=True, batch_size=B)
        for key in ("loss_rot", "loss_trans", "loss_fk", "rotation_error_deg", "joint_error_mm"):
            self.log(f"train_{key}", parts[key], batch_size=B)
        return loss.float().log10()


__all__ = ["EvHPCropModel", "TRANS_MEAN_CROP"]
