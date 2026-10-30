"""Training-only geometric auxiliaries for the opt-in XKF repair experiment.

The deployed DGP solve uses predicted pose, keypoints and confidence. Its
translation loss allows errors in those inputs to compensate. These auxiliaries
hold one geometric input at its training label and detach confidence, testing
whether supervising each geometric input separately improves measurement quality.

They neither change the inference solve nor fix the known approximate ``depth``
augmentation. Labels may be used for training or offline loss scoring, never
for deployed predictions; validation/test labels must not update parameters.
"""
from __future__ import annotations

import torch

from semkine.xkf_direct import dgp_translation


def direct_aux_losses(model, z_d, aux, batch):
    """Return unweighted ``d_aux_2d`` and ``d_aux_3d`` translation MSEs in m².

Both losses average over all B×3 translation coordinates, matching the existing
``l_t``. The caller supplies opt-in coefficients and the existing normalizer.

* 2D auxiliary: DGP(GT zero-translation joints, predicted rays, detached weights).
* 3D auxiliary: DGP(predicted zero-translation joints, GT rays, detached weights).

Ground truth and confidence never receive gradients. Every joint and sample is
retained, including projections outside the image: those remain valid pinhole
coordinates. Nonfinite labels or GT joints behind/at the camera are rejected
explicitly instead of clamping or silently dropping their contributions.
"""
    if batch.target.ndim != 2 or batch.target.shape[-1] != 51 or z_d.shape != batch.target.shape:
        raise ValueError('XKF auxiliary losses require matching (B,51) training states')
    with torch.autocast(device_type=z_d.device.type, enabled=False):
        with torch.no_grad():
            truth = batch.target.detach().float()
            if not bool(torch.isfinite(truth).all()):
                raise ValueError('nonfinite training label in XKF DGP auxiliary')
            betas, camera = model._resolve_betas_K(truth, batch.betas, batch.camera_K)
            pose0 = truth.clone()
            pose0[:, :3] = 0.0
            _, gt_joints0 = model._fk(pose0, betas.detach().float())
            gt_joints = gt_joints0 + truth[:, None, :3]
            if not bool(torch.isfinite(gt_joints).all() & (gt_joints[..., 2] > 1e-3).all()):
                raise ValueError('GT joints must be finite and in front of camera for XKF DGP auxiliary')
            fx, fy, cx, cy = model._intrinsics(camera.detach().float())
            if not bool(torch.isfinite(camera).all() & (fx > 0).all() & (fy > 0).all()):
                raise ValueError('invalid camera intrinsics in XKF DGP auxiliary')
            gt_uv = gt_joints[..., :2] / gt_joints[..., 2:3]

        kp = aux['kp_px'].float()
        pred_uv = torch.stack([(kp[..., 0] - cx[:, None]) / fx[:, None],
                               (kp[..., 1] - cy[:, None]) / fy[:, None]], -1)
        confidence = aux['w'].detach().float()
        translation_2d = dgp_translation(gt_joints0.detach(), pred_uv, confidence)
        translation_3d = dgp_translation(aux['joints0'].float(), gt_uv.detach(), confidence)
        return {
            'd_aux_2d': (translation_2d - truth[:, :3]).square().mean(),
            'd_aux_3d': (translation_3d - truth[:, :3]).square().mean(),
        }


__all__ = ['direct_aux_losses']
