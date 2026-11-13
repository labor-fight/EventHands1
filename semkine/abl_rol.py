r"""ABL rollout model for the ray-association arm (docs/EXPERIMENT_RECORDS.md#abl-amend1-20261111).

semkine/tpd_rol.py's `ROLModel` trains B2-type networks whose association is the router's (PFA under face_route); its
forward re-implements that path and its constructor refuses any other. The ablation (a) (`abgeo`, MODEL.XAG_ASSOC base)
associates events by S45's ray-mesh proximity instead. `RayROLModel` is ROLModel with only the association changed:
  forward   model/model.py `_xyz_mesh_forward` for XAG_ASSOC base (S45's `associate_rays` against the prior mesh, under
            no_grad, as at evaluation), with the FK outside no_grad as in ROLModel, so the prior's gradient reaches the
            mesh points and the gates; the ray association itself (vertices, weights, depth) is not differentiable
  routes    none (there is no router)
Everything else (rollout data, K-packet training step with checkpointing, the start check that the first forward equals
model/model.py's bit for bit, the INIT_FROM check, cosine to 0) is ROLModel's.
"""
from __future__ import annotations

import torch

from . import tpd_rol as TR

_NATIVE = TR.rol_model_class
_CLASS = {}


def ray_rol_model_class():
    if "cls" in _CLASS:
        return _CLASS["cls"]
    from model import MNISTModel
    base = _NATIVE()

    class RayROLModel(base):
        def __init__(self, cfg=None, num_classes=None):
            MNISTModel.__init__(self, cfg, num_classes)
            r = self.cfg["ROL"]
            self.rol_k = int(r["K"])
            self.rol_grad = bool(r["GRAD"])
            self.rol_ckpt = bool(r["CHECKPOINT"])
            self.rol_w = [float(w) for w in r["WEIGHTS"]]
            if len(self.rol_w) != self.rol_k or min(self.rol_w) <= 0:
                raise ValueError("ROL.WEIGHTS: K positive weights")
            if not self.xyz_mesh or self.xag_assoc_mode != "base" or self.loss_type != "so3_trans_fk":
                raise ValueError("RayROLModel trains the ray-association arm (xyz_mesh, XAG_ASSOC base, so3_trans_fk)")
            self._rol_checked = False

        def _xyz_mesh_forward(self, batch, prev, sources=None):
            from .encoder import event_tokens
            from .events import EV_X, EV_Y
            from .xyz_mesh import associate_rays, stride_sample
            events, ptr = batch.events, batch.ptr
            if events.shape[0] == 0:
                return MNISTModel._xyz_mesh_forward(self, batch, prev)
            B = int(ptr.numel() - 1)
            betas_f, k_f = self._resolve_betas_K(prev, batch.betas, batch.camera_K)
            src, mask = stride_sample(ptr, self.xyz_max_nodes)
            tok = event_tokens(events, ptr, batch.delta_t_s, self.render_h, self.render_w)
            tok = tok[src.reshape(-1)].view(B, -1, tok.shape[-1]) * mask.unsqueeze(-1)
            verts, _ = self._fk(prev.float(), betas_f)
            with torch.no_grad():
                fx, fy, cx, cy = self._intrinsics(k_f)
                ev = events[src.reshape(-1)].view(B, -1, events.shape[1])
                px, py = ev[..., EV_X].float(), ev[..., EV_Y].float()
                d = torch.stack([(px - cx[:, None]) / fx[:, None], (py - cy[:, None]) / fy[:, None],
                                 torch.ones_like(px)], dim=-1)
                f_px = (fx * fy).sqrt()
                assoc = associate_rays(d, mask, verts.detach(), self.mano.f, self.xyz.vert_faces,
                                       band=self.xyz_band_px / f_px, front_tol=self.xyz_front_px / f_px,
                                       z_tol=self.xyz_z_tol, k=self.xyz_k)
            delta, stats, _ = self.xyz(tok, d, mask, verts, assoc, prev[:, 3:51], ablate=self.ablate_evidence,
                                       capture=False)
            self.route_stats = stats
            return delta

        @staticmethod
        def _routes(batch, max_nodes):
            return None

    _CLASS["cls"] = RayROLModel
    return RayROLModel


__all__ = ["ray_rol_model_class"]
