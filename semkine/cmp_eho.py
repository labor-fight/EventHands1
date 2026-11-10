r"""EventHands as published (Rudnev et al., ICCV 2021; docs/EXPERIMENT_RECORDS.md#cmp-amend4-20261110).

The official recipe (github.com/r00tman/EventHands `model/model.py`, `model/train.py`, commit abfca72; local copy
/data1/lyq/code/mesh/EventHands) on this repository's data:

  network  conv1 (2 -> 3, 3 x 3, padding 1) + torchvision resnet18(num_classes=12), random initialisation: MNISTModel's
           dense trunk with POSE_REPR mano_pca6 / OUTPUT_DIM 12 is that network, same calls, same parameter names
  output   [alpha6, t3, R3]: the first six MANO PCA coefficients, the root translation (m), the root axis-angle
  loss     (6 / 0.1 MSE(alpha) + 3 / 1e-4 MSE(t) + 3 / 0.05 MSE(R)) / 12, optimised as its log10 (BaseModel's mse_51d
           loss on the 12-D layout is that expression with the official slicing)
  target   the 51-D label [t, R, pose - hands_mean] projected to the 12-D layout, alpha = (pose - hands_mean) pinv(C6)
           with C6 the first six rows of MANO's hands_components (model/pose_repr.py `meta51_to_target`)

`forward` returns the 12-D output as the official network does; `state` decodes it to the engine's 51-D state
[t, R, alpha C6] (model/pose_repr.py `decode_to_mano_inputs`). The previous state is never read.
"""
from __future__ import annotations

import numpy as np
import torch

from model import MNISTModel


class EHOModel(MNISTModel):
    KIND = "eho"

    def __init__(self, cfg=None, num_classes=None):
        super().__init__(cfg, num_classes)
        c = self.cfg.get("CMP", {}) or {}
        if c.get("MODEL") != self.KIND:
            raise ValueError("CMP.MODEL must be 'eho' for EHOModel")
        if (self.encoder_name or self.prev_render or self.prevpos_embed or self.sparse_trunk or self.predict_delta
                or self.zero_event_gate):
            raise ValueError("EventHands: the plain dense trunk, absolute output")
        if (self.cnn_backbone != "resnet18" or self.pose_repr != "mano_pca6" or self.output_dim != 12
                or self.loss_type != "mse_51d" or not self.log10_loss):
            raise ValueError("EventHands: resnet18, mano_pca6 / 12-D output, the official weighted MSE in log10")
        c6 = np.load(self.cfg["MANO"]["NPZ"])["hands_components"].astype(np.float64)[:6]
        self.register_buffer("c6", torch.tensor(c6, dtype=torch.float32), persistent=False)
        self.register_buffer("c6_pinv", torch.tensor(np.linalg.pinv(c6), dtype=torch.float32), persistent=False)

    def to_target(self, y51):
        """51-D label [t, R, residual45] -> the official 12-D layout [alpha6, t, R]."""
        y = y51.float()
        return torch.cat([y[:, 6:51] @ self.c6_pinv, y[:, 0:3], y[:, 3:6]], dim=-1)

    def state(self, out12):
        """12-D output -> the engine's 51-D state [t, R, alpha C6]."""
        o = out12.float()
        return torch.cat([o[:, 6:9], o[:, 9:12], o[:, 0:6] @ self.c6], dim=-1)

    def _compute_loss(self, pred, y, betas=None):
        if y.shape[-1] == 51:
            y = self.to_target(y)
        return super()._compute_loss(pred.float(), y, betas)


__all__ = ["EHOModel"]
