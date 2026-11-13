r"""CMP baselines on the dense LNES face (docs/EXPERIMENT_RECORDS.md#cmp-prereg-20261110; tools/cmp/arms.py).

All three are `MNISTModel` subclasses on the same trunk, `rn(conv1(LNES))` with `rn` = torchvision's resnet18 (the
EventHands / S43 dense trunk), and all three build MANO for the so3_trans_fk loss. Opt-in: nothing on B2's or S43's path
imports this module.

  EHModel  EventHands: the 51-D state is `rn.fc` of the pooled feature; the previous state is not read
  EMModel  event encoder + previous-state MLP: delta = fuse([f; prev]) (512 + 51 -> 256 -> 51, zero-initialised last
           layer), zero-event gate, output = prev + delta formed in float32
  EGModel  event encoder + GRU: per-packet pooled features f_k, h_k = GRU(f_k, h_{k-1}) from h_0 = 0 (float32),
           y_k = rn.fc(f_k) + head_h(h_k) with head_h zero-initialised, so the model starts as the EventHands network it
           is initialised from

Sequence stages (CMP.STAGE: seq) train on `semkine.cmp_seq` batches, dicts of stacked tensors: `lnes` (B, K, H, W, C),
`prev0` (B, 51) = GT at the first packet's start + B2's prior noise, `states` (B, K + 1, 51) = clean GT at the first
packet's start and at every packet's end, `betas`, `camera_K`. EMModel then runs EvHandTrack's rol recipe (K packets,
x_{k+1} = y_k with the gradient through the previous state, mean of the K losses, cosine to 0); EGModel the GRU over the
K packets (mean of the K losses, gradient-norm clip 5).
"""
from __future__ import annotations

import torch
from torch import nn

from model import MNISTModel

FUSE_HIDDEN = 256
GRU_HIDDEN = 512
GRU_CLIP = 5.0


def resnet_features(rn, x):
    """torchvision ResNet up to the pooled feature (the input of `rn.fc`)."""
    x = rn.maxpool(rn.relu(rn.bn1(rn.conv1(x))))
    x = rn.layer4(rn.layer3(rn.layer2(rn.layer1(x))))
    return torch.flatten(rn.avgpool(x), 1)


class _Dense(MNISTModel):
    """Shared checks and helpers of the three CMP LNES models."""

    KIND = None

    def __init__(self, cfg=None, num_classes=None):
        super().__init__(cfg, num_classes)
        m = self.cfg["MODEL"]
        c = self.cfg.get("CMP", {}) or {}
        if c.get("MODEL") != self.KIND:
            raise ValueError(f"CMP.MODEL must be {self.KIND!r} for {type(self).__name__}")
        if self.encoder_name or self.prev_render or self.prevpos_embed or self.sparse_trunk:
            raise ValueError("CMP LNES models use the plain dense trunk (no ENCODER / PREV_RENDER / PREVPOS_EMBED)")
        if self.cnn_backbone != "resnet18" or self.output_dim != 51 or self.loss_type != "so3_trans_fk":
            raise ValueError("CMP LNES models: resnet18 trunk, 51-D state, so3_trans_fk")
        self.stage = str(c.get("STAGE", "frame"))
        self.seq = dict(c.get("SEQ", {}) or {})
        if self.stage == "seq":
            self.LR_FLOOR = 0.0                                     # cosine to zero (EvHandTrack's rol schedule)
        self._build_mano(m)

    def features(self, lnes):
        """(N, H, W, C) LNES -> (N, 512) pooled trunk feature."""
        return resnet_features(self.rn, self.conv1(lnes.permute(0, 3, 1, 2).contiguous()))

    @staticmethod
    def empty(lnes):
        """(N, 1) True where the packet has no event (the S43 / B2 zero-event test on the LNES)."""
        return lnes.reshape(lnes.shape[0], -1).abs().sum(dim=1, keepdim=True) <= 0

    def _seq_loss(self, preds, batch):
        """Mean so3_trans_fk loss of (B, K, 51) predictions against states[:, 1:]."""
        B, K, _ = preds.shape
        betas = batch["betas"].unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
        return self._compute_loss(preds.reshape(B * K, -1), batch["states"][:, 1:].reshape(B * K, -1), betas)

    def _log_seq(self, loss, parts, B):
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, batch_size=B)
        for key in ("loss_rot", "loss_trans", "loss_fk", "rotation_error_deg", "joint_error_mm"):
            self.log(f"train_{key}", parts[key], sync_dist=True, batch_size=B)

    def training_step(self, batch, batch_nb):
        if self.stage != "seq":
            return super().training_step(batch, batch_nb)
        if not isinstance(batch, dict):
            raise TypeError("sequence stages train on semkine.cmp_seq batches")
        loss, parts = self.seq_loss(batch)
        self._log_seq(loss, parts, batch["lnes"].shape[0])
        return loss.float().log10()


class EHModel(_Dense):
    """EventHands: absolute 51-D state from the LNES; `forward` is MNISTModel's dense absolute path."""

    KIND = "eh"

    def __init__(self, cfg=None, num_classes=None):
        super().__init__(cfg, num_classes)
        if self.predict_delta or self.zero_event_gate:
            raise ValueError("EventHands regresses the absolute state (PREDICT_DELTA / ZERO_EVENT_GATE false)")
        if self.stage != "frame":
            raise ValueError("EventHands trains on single windows")


class EMModel(_Dense):
    """Event encoder + previous-state MLP."""

    KIND = "em"

    def __init__(self, cfg=None, num_classes=None):
        super().__init__(cfg, num_classes)
        if not (self.predict_delta and self.zero_event_gate):
            raise ValueError("the previous-state MLP writes a gated delta (PREDICT_DELTA / ZERO_EVENT_GATE true)")
        width = self.rn.fc.in_features
        self.rn.fc = nn.Identity()
        self.fuse = nn.Sequential(nn.Linear(width + self.output_dim, FUSE_HIDDEN), nn.ReLU(inplace=True),
                                  nn.Linear(FUSE_HIDDEN, self.output_dim))
        nn.init.zeros_(self.fuse[2].weight)
        nn.init.zeros_(self.fuse[2].bias)
        if self.stage == "seq":
            self.rol_k = int(self.seq["K"])
            self.rol_grad = bool(self.seq.get("GRAD", True))

    def update(self, feat, prev, empty):
        """prev + gated delta in float32."""
        delta = self.fuse(torch.cat([feat, prev.to(feat.dtype)], dim=-1))
        delta = torch.where(empty, torch.zeros_like(delta), delta)
        return prev.float() + delta.float()

    def forward(self, x, prevpos, betas=None, camera_K=None):
        return self.update(self.features(x), prevpos, self.empty(x))

    def seq_loss(self, batch):
        lnes = batch["lnes"]
        B, K = lnes.shape[:2]
        if K != self.rol_k:
            raise ValueError("rollout length differs from CMP.SEQ.K")
        flat = lnes.reshape(B * K, *lnes.shape[2:])
        feats = self.features(flat).view(B, K, -1)                    # the trunk does not read the state
        empty = self.empty(flat).view(B, K, 1)
        x, preds = batch["prev0"], []
        for k in range(K):
            y = self.update(feats[:, k], x, empty[:, k])
            preds.append(y)
            x = y if self.rol_grad else y.detach()
        return self._seq_loss(torch.stack(preds, 1), batch)


class EGModel(_Dense):
    """Event encoder + GRU."""

    KIND = "eg"

    def __init__(self, cfg=None, num_classes=None):
        super().__init__(cfg, num_classes)
        if self.predict_delta or self.zero_event_gate:
            raise ValueError("the GRU model regresses the absolute state")
        width = self.rn.fc.in_features
        self.gru = nn.GRU(width, GRU_HIDDEN, num_layers=1, batch_first=True)
        self.head_h = nn.Linear(GRU_HIDDEN, self.output_dim)
        nn.init.zeros_(self.head_h.weight)
        nn.init.zeros_(self.head_h.bias)

    def run(self, feats, h0=None):
        """(B, K, 512) features -> (B, K, 51) states and the last hidden state; the GRU in float32."""
        with torch.autocast(device_type=feats.device.type, enabled=False):
            hs, hn = self.gru(feats.float(), h0)
        return self.rn.fc(feats).float() + self.head_h(hs).float(), hn

    def forward(self, x, prevpos=None, betas=None, camera_K=None):
        """One packet from a zero hidden state (single-window use, e.g. a validation batch)."""
        return self.run(self.features(x).unsqueeze(1))[0][:, 0]

    def seq_loss(self, batch):
        lnes = batch["lnes"]
        B, K = lnes.shape[:2]
        feats = self.features(lnes.reshape(B * K, *lnes.shape[2:])).view(B, K, -1)
        return self._seq_loss(self.run(feats)[0], batch)

    def configure_gradient_clipping(self, optimizer, optimizer_idx=None, gradient_clip_val=None,
                                    gradient_clip_algorithm=None):
        self.clip_gradients(optimizer, gradient_clip_val=GRU_CLIP, gradient_clip_algorithm="norm")


KINDS = {"eh": EHModel, "em": EMModel, "eg": EGModel}
#: amendment 6 (#cmp-amend6-20261110): model kinds defined in their own modules (each exposes `model_class(kind)`)
EXT_MODELS = {"epxf": "semkine.cmp_evhpx", "epx": "semkine.cmp_evhpx", "ehpe": "semkine.cmp_ehpe",
              "fme": "semkine.cmp_fme"}


def model_class(kind):
    if kind in KINDS:
        return KINDS[kind]
    if kind in ("ep", "epf"):
        from semkine.cmp_evhp import EvHPModel
        return EvHPModel
    if kind in ("epc", "epcf"):                                    # amendment 1: EvHandPose with its hand crop
        from semkine.cmp_evhpc import EvHPCropModel
        return EvHPCropModel
    if kind == "eho":                                              # amendment 4: EventHands as published
        from semkine.cmp_eho import EHOModel
        return EHOModel
    if kind in EXT_MODELS:                                         # amendment 6
        import importlib
        return importlib.import_module(EXT_MODELS[kind]).model_class(kind)
    raise ValueError(f"unknown CMP model kind {kind!r}")


__all__ = ["EGModel", "EHModel", "EMModel", "KINDS", "model_class", "resnet_features"]
