r"""EvHandPose with its hand crop, trained for online use (docs/EXPERIMENT_RECORDS.md#cmp-amend6-20261110).

`semkine/cmp_evhpc.py`'s model unchanged (official FlowNet / Backbone, the official crop-frame translation decoding,
the same losses and schedule); two things of the training differ, both what the online tracker of tools/cmp/nets.py
(kind epc: the box of a packet from the tracker's own state at the packet's start, history from the segment start)
meets and `epc` never trained on:
  box      `semkine/cmp_seq_cropx.py`: every packet cut around GT + prior noise with a zoomed box, so the crop-frame depth
           has to be read from the crop and the decoded depth no longer copies the box's
  history  the pose stage sees sequences of 1 .. K packets (with probability FULL the whole K = 12, else a uniform shorter
           suffix; the last min(length, 3) supervised): the first updates of a segment have fewer than 12 packets
CMP.MODEL epxf = the FlowNet stage (as epcf, on the new crops), epx = the pose stage (FlowNet frozen from epxf).
"""
from __future__ import annotations

import torch

from model import BaseModel

from .cmp_evhp import _bary, official_config
from .cmp_evhpc import TRANS_MEAN_CROP, EvHPCropModel
from .evhandpose_official.backbone import Backbone, FlowNet

ROLE = {"epxf": "epf", "epx": "ep"}
FULL = 0.5
#: stream of the history-length draws
HISTORY_STREAM = 1000003


def slice_history(batch, length):
    """The last `length` packets of a cmp_seq_crop batch (states keep the start of the first kept packet)."""
    return {**batch, "lnes": batch["lnes"][:, -length:], "voxel": batch["voxel"][:, -length:],
            "crop": batch["crop"][:, -length:], "states": batch["states"][:, -(length + 1):]}


class EvHPCropXModel(EvHPCropModel):
    def __init__(self, cfg=None, num_classes=None):
        BaseModel.__init__(self, cfg)                               # EvHPCropModel.__init__ with this arm's kinds
        c = self.cfg["CMP"]
        if c["MODEL"] not in ROLE or self.output_dim != 51 or self.loss_type != "so3_trans_fk":
            raise ValueError("EvHPCropXModel: CMP.MODEL epxf / epx, 51-D state, so3_trans_fk")
        self.cmp_kind = c["MODEL"]
        self.kind = ROLE[self.cmp_kind]
        seq = c.get("SEQ", {}) or {}
        self.voxel_bins = int(seq.get("VOXEL_BINS", 5))
        self.annot_len = int(seq.get("ANNOT_LEN", 3))
        self.crop_size = int(c["CROP"]["SIZE"])
        self.full_prob = float(c["CROP"]["TRACK"]["FULL"])
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

    def history_length(self, k, batch_nb):
        """Packets of this batch's sequences: K with probability FULL, else uniform in 1 .. K - 1 (one draw per batch)."""
        if k < 2:
            return k
        g = torch.Generator().manual_seed(int(self.cfg["SEED"]) * HISTORY_STREAM + int(batch_nb))
        if float(torch.rand(1, generator=g)) < self.full_prob:
            return k
        return int(torch.randint(1, k, (1,), generator=g))

    def training_step(self, batch, batch_nb):
        if self.kind == "ep" and isinstance(batch, dict) and "crop" in batch:
            k = batch["lnes"].shape[1]
            length = self.history_length(k, batch_nb)
            self.log("train_history", float(length), batch_size=batch["lnes"].shape[0])
            if length < k:
                batch = slice_history(batch, length)
        return super().training_step(batch, batch_nb)


def model_class(kind):
    if kind not in ROLE:
        raise ValueError(f"unknown model kind {kind!r}")
    return EvHPCropXModel


__all__ = ["EvHPCropXModel", "FULL", "model_class", "slice_history"]
