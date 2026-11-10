r"""EvHandPose (Jiang et al., TPAMI 2024) ported to the CMP protocol (docs/EXPERIMENT_RECORDS.md#cmp-prereg-20261110).

The network is the official code (semkine/evhandpose_official/, vendored): FlowNet = MultiResUNet on a 5-bin voxel grid
(base 32 channels, 4 encoders, 2 residual blocks, tanh flow at four scales, upsampled to full resolution); Backbone =
[LNES (2) | finest flow (2)] -> 7x7 stride-2 fusion conv -> max-pool -> ResNet-34 BasicBlock stages [3, 4, 6, 3] (stride 2
each) -> ConvGRU (512, k3) over the packets + BatchNorm3d -> global pool -> 128 -> LeakyReLU -> state. The official
configuration of configs/train_supervision.yml is kept except for what the protocol fixes:
  * input frame: the full 240 x 180 event frame zero-padded to 256 x 192 (divisible by the 64x stride), no crop -- the
    official crop box is built from the annotated joints, which a deployed tracker does not have;
  * output: the 51-D state [translation (x 0.3 + training mean), root axis-angle, 45 pose] with the sequence GT betas
    (every CMP method uses them), instead of 61 values with shape;
  * sub-windows: 50 ms packets (the protocol's update), 12 per sequence, the last 3 supervised (official seq_len 12,
    annot_len 3), loss = so3_trans_fk (every CMP method's loss) on those 3;
  * FlowNet supervision (stage epf): the official flow loss (squared error at all four scales against the pseudo ground
    truth, background included) on mesh flow of the GT MANO surface between the packet's start and end states,
    rendered here by z-buffered surface samples (15 barycentric samples per face); flow is in units of 32 px.
Stage ep freezes FlowNet (the official `flow.fixed`), Adam 5e-4 with L2 weight decay 1e-4 (official) under the common
warm-up + cosine schedule.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from model import BaseModel, MNISTModel

from .evhandpose_official.backbone import Backbone, FlowNet

PAD = (8, 8, 6, 6)                       # left, right, top, bottom: 240 x 180 -> 256 x 192
FLOW_SCALE_PX = 32.0
TRANS_MEAN = (-0.04, 0.03, 0.56)         # training-set mean translation (m)
TRANS_SCALE = 0.3                        # official predictor scale
DEPTH_TOL = 5.0e-3
FLOW_CHUNK = 48                          # packets per frozen FlowNet forward (~2.7 GiB / 12 packets at 256 x 192)


def official_config(annot_len):
    """configs/train_supervision.yml's method / preprocess blocks the official modules read (51 outputs)."""
    return {
        "preprocess": {"repre": "LNES"},
        "method": {
            "block": "BasicBlock",
            "flow": {"usage": True, "num_bins": 5, "base_num_channels": 32, "kernel_size": 3, "num_encoders": 4,
                     "num_residual_blocks": 2, "last_for_output": True},
            "event_encoder": {"usage": True, "num_bins": 1, "base_num_channels": 64, "channel_sizes": []},
            "encoder_fusion": {"channel_sizes": [3, 4, 6, 3]},
            "seq_model": {"model": "ConvGRU", "seq_len": 12, "annot_len": int(annot_len), "num_layers": 1},
            "predictor": {"layer_dims": [128, 51]},
        },
    }


def _bary(n=4):
    pts = [(i / n, j / n, (n - i - j) / n) for i in range(n + 1) for j in range(n + 1 - i)]
    return torch.tensor(pts, dtype=torch.float32)


class EvHPModel(BaseModel):
    """CMP.MODEL epf (FlowNet stage) or ep (pose stage) on semkine/cmp_seq.py batches (VOXEL_BINS 5)."""

    _build_mano = MNISTModel._build_mano
    _fk = MNISTModel._fk
    _resolve_betas_K = MNISTModel._resolve_betas_K
    _intrinsics = MNISTModel._intrinsics
    set_hand_context = MNISTModel.set_hand_context
    LR_FLOOR = 0.02

    def __init__(self, cfg=None, num_classes=None):
        super().__init__(cfg)
        c = self.cfg["CMP"]
        self.kind = c["MODEL"]
        if self.kind not in ("epf", "ep") or self.output_dim != 51 or self.loss_type != "so3_trans_fk":
            raise ValueError("EvHPModel: CMP.MODEL epf / ep, 51-D state, so3_trans_fk")
        seq = c.get("SEQ", {}) or {}
        self.voxel_bins = int(seq.get("VOXEL_BINS", 5))
        self.annot_len = int(seq.get("ANNOT_LEN", 3))
        self.ocfg = official_config(self.annot_len)
        self.flow_encoder = FlowNet(self.ocfg)
        self.infer_encoder = Backbone(self.ocfg)
        if self.kind == "ep":
            for p in self.flow_encoder.parameters():
                p.requires_grad = False
        self._build_mano(self.cfg["MODEL"])
        self.register_buffer("trans_mean", torch.tensor(TRANS_MEAN), persistent=False)
        self.register_buffer("bary", _bary(), persistent=False)
        self._ctx_betas = None
        self._ctx_K = None

    # ------------------------------------------------------------------ network pieces
    def flows(self, voxel):
        """(N, bins, H, W) -> (N, 2 * num_encoders, H', W') on the padded frame (the official FlowNet output layout)."""
        out = self.flow_encoder(F.pad(voxel, PAD))                         # (N, 4, 2, H', W')
        return out.reshape(out.shape[0], -1, out.shape[-2], out.shape[-1])

    def packet_features(self, lnes, voxel):
        """Per-packet fusion feature (N, 512, 3, 4): Backbone up to (not including) the ConvGRU, finest flow only."""
        B = self.infer_encoder
        if self.kind == "ep":
            self.flow_encoder.eval()
            with torch.no_grad():                                           # frozen; chunked (full-frame U-Net memory)
                flow = torch.cat([self.flows(v)[:, -2:] for v in voxel.split(FLOW_CHUNK)])
        else:
            flow = self.flows(voxel)[:, -2:]
        x = torch.cat([F.pad(lnes.permute(0, 3, 1, 2), PAD), flow.to(lnes.dtype)], dim=1)
        for enc in B.event_encoders:                                        # empty in the official configuration
            x = enc(x)
        for enc in B.fusion_encoders:
            x = enc(x)
        return x

    def sequence_head(self, feats):
        """(B, T, 512, h, w) -> (B, T', 51) for the last annot_len packets (T' = min(T, annot_len))."""
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
        trans = x[..., 45:48] * TRANS_SCALE + self.trans_mean
        return torch.cat([trans, x[..., 48:51], x[..., :45]], dim=-1)

    def sequence_state(self, feats):
        """The state of the last packet of a (1, T, 512, h, w) feature sequence (the deployed sliding form)."""
        return self.sequence_head(feats)[:, -1]

    # ------------------------------------------------------------------ flow pseudo ground truth
    @torch.no_grad()
    def mesh_flow(self, s0, s1, betas, camera_K):
        """(N, 2, H, W) flow of the front-most GT MANO surface from s0 to s1, in FLOW_SCALE_PX units (0 elsewhere)."""
        with torch.autocast(device_type=s0.device.type, enabled=False):
            v0, _ = self._fk(s0.float(), betas.float())
            v1, _ = self._fk(s1.float(), betas.float())
            faces = self.mano.f.long()
            p0 = torch.einsum("sk,nfkd->nfsd", self.bary, v0[:, faces]).reshape(v0.shape[0], -1, 3)
            p1 = torch.einsum("sk,nfkd->nfsd", self.bary, v1[:, faces]).reshape(v1.shape[0], -1, 3)
            fx, fy, cx, cy = self._intrinsics(camera_K.float())
            z0, z1 = p0[..., 2].clamp(min=1e-6), p1[..., 2].clamp(min=1e-6)
            u0 = fx[:, None] * p0[..., 0] / z0 + cx[:, None]
            w0 = fy[:, None] * p0[..., 1] / z0 + cy[:, None]
            u1 = fx[:, None] * p1[..., 0] / z1 + cx[:, None]
            w1 = fy[:, None] * p1[..., 1] / z1 + cy[:, None]
            N, P = u0.shape
            H, W = self.render_h, self.render_w
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
        if not isinstance(batch, dict) or "voxel" not in batch:
            raise TypeError("EvHandPose trains on semkine.cmp_seq batches with voxel grids")
        lnes, voxel, states = batch["lnes"], batch["voxel"], batch["states"]
        B, K = lnes.shape[:2]
        if self.kind == "epf":
            flat = voxel.reshape(B * K, *voxel.shape[2:])
            pred = self.flows(flat)                                          # (BK, 8, H', W')
            l, r, t, b = PAD
            pred = pred[..., t:pred.shape[-2] - b, l:pred.shape[-1] - r].float()
            gt = self.mesh_flow(states[:, :-1].reshape(B * K, -1), states[:, 1:].reshape(B * K, -1),
                                batch["betas"].unsqueeze(1).expand(B, K, -1).reshape(B * K, -1),
                                batch["camera_K"].unsqueeze(1).expand(B, K, 3, 3).reshape(B * K, 3, 3))
            gt = gt.repeat(1, pred.shape[1] // 2, 1, 1)
            loss = (gt - pred).square().sum() / (B * K * pred.shape[1])
            self.log("train_loss", loss, prog_bar=True, batch_size=B)
            self.log("train_flow_epe_px", ((gt[:, -2:] - pred[:, -2:]).square().sum(1).sqrt().mean()) * FLOW_SCALE_PX,
                     batch_size=B)
            return loss.float().clamp_min(1e-12).log10()
        feats = self.packet_features(lnes.reshape(B * K, *lnes.shape[2:]), voxel.reshape(B * K, *voxel.shape[2:]))
        preds = self.sequence_head(feats.view(B, K, *feats.shape[1:]))      # (B, annot_len, 51)
        T = preds.shape[1]
        betas = batch["betas"].unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
        loss, parts = self._compute_loss(preds.reshape(B * T, -1), states[:, -T:].reshape(B * T, -1), betas)
        self.log("train_loss", loss, prog_bar=True, batch_size=B)
        for key in ("loss_rot", "loss_trans", "loss_fk", "rotation_error_deg", "joint_error_mm"):
            self.log(f"train_{key}", parts[key], batch_size=B)
        return loss.float().log10()

    def init_from_abs_checkpoint(self, ckpt_path):
        """semkine/train.py's MODEL.INIT_FROM: stage ep takes FlowNet (every `flow_encoder.` tensor) from stage epf."""
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        own = self.state_dict()
        keys = [k for k in own if k.startswith("flow_encoder.")]
        missing = [k for k in keys if k not in sd or sd[k].shape != own[k].shape]
        if missing:
            raise RuntimeError(f"INIT_FROM lacks FlowNet tensors: {missing[:3]}")
        self.load_state_dict({**own, **{k: sd[k] for k in keys}})
        return len(keys)

    def train(self, mode=True):
        super().train(mode)
        if self.kind == "ep":
            self.flow_encoder.eval()                                        # frozen, BatchNorm-free, kept in eval
        return self

    def configure_optimizers(self):
        t = self.cfg["TRAIN"]
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=float(t["LR"]), weight_decay=float(t.get("WEIGHT_DECAY_L2", 0.0)))
        warmup, total = int(t["WARMUP_STEPS"]), int(t["MAX_STEPS"])

        def lr_lambda(step):
            if step < warmup:
                return float(step + 1) / float(warmup)
            p = min(max((step - warmup) / float(total - warmup), 0.0), 1.0)
            return self.LR_FLOOR + (1.0 - self.LR_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * p))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def forward(self, x, prevpos=None, betas=None, camera_K=None):
        raise RuntimeError("EvHandPose reads packet sequences: use packet_features / sequence_head")


__all__ = ["EvHPModel", "FLOW_SCALE_PX", "official_config"]
