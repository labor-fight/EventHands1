"""Final MANO surface decoding after the causal tracker predicts its 51D state.

This uses the unchanged production `_fk`: residual hand angles get their mean
exactly once. Both computation and device-to-host surface transfer are timed.
No target, visibility or observation state is read or changed here.
"""
from __future__ import annotations

import time
import numpy as np
import torch


@torch.inference_mode()
def decode_surface(model, pose51, betas):
    base = model.base
    reference = next(base.parameters())
    device = reference.device

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    pose = np.asarray(pose51, np.float32)
    beta = np.asarray(betas, np.float32)
    if pose.shape != (51,) or beta.shape != (10,) or not np.isfinite(pose).all() or not np.isfinite(beta).all():
        raise ValueError("finite pose51 and beta10 required")
    synchronize()
    started = time.perf_counter()
    state = torch.as_tensor(pose[None], device=device, dtype=reference.dtype)
    shape = torch.as_tensor(beta[None], device=device, dtype=reference.dtype)
    synchronize()
    transferred = time.perf_counter()
    vertices, joints = base._fk(state, shape)
    synchronize()
    computed = time.perf_counter()
    vertices = vertices[0].float().cpu().numpy().copy()
    joints = joints[0].float().cpu().numpy().copy()
    synchronize()
    finished = time.perf_counter()
    if vertices.shape != (778, 3) or joints.shape != (21, 3) or not np.isfinite(vertices).all() or not np.isfinite(joints).all():
        raise ArithmeticError("invalid reconstructed MANO surface")
    return {"vertices_m": vertices, "joints_m": joints}, {
        "output_mesh_context_transfer_seconds": transferred - started,
        "output_mano_fk_seconds": computed - transferred,
        "output_mesh_transfer_seconds": finished - computed,
        "output_mesh_total_seconds": finished - started,
    }
