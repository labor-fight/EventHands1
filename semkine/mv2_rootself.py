"""MV2 M5 conditional arm: root-block-restricted self-previous-state training on the surface-lifted readout
(docs/EXPERIMENT_RECORDS.md#mv2-prereg-20261026, M5 section 4 RCA; configuration registered as an amendment after RCA
passed, before training).

`RootSelfModel` is `SLRModel` plus one opt-in key, `MODEL.ROOTSELF: true`, which only changes `_maybe_self` (S42
candidate B, `TRACK.SELF_P` with `DATA.CLIP_K` clips). Without the key it is `SLRModel` exactly.

S42-B replaces a selected row's whole previous state by the deployed closed loop's own filtered state s at end - 50 ms
(the clip's step K-2), together with the clip's last LNES and target. ROOTSELF keeps everything of S42-B (same row draw
from the global CUDA RNG, same eval-mode no-grad float32 rollout, same deployed filter, same LNES / target replacement)
except the previous state of the selected rows, which becomes

    [ GT(end - 50) translation + n_t ,  s's root rotation ,  GT(end - 50) fingers + n_f ]

with GT(end - 50) = clip_tgt[K-2] and (n_t, n_f) drawn from the recipe's own TRACK.PREV_NOISE_* model (mixed: small /
large / directed, the translation and finger parts of `SemKineDataset._sample_prev_noise`) by a dedicated CPU generator
(`ROOTSELF_SEED`), so the self-generated, time-correlated error reaches the root block only and the other blocks keep the
S39 training distribution.
"""
from __future__ import annotations

import torch

from semkine.slr import SLRModel

ROOTSELF_KEYS = frozenset({"ROOTSELF", "ROOTSELF_SEED"})


class RootSelfModel(SLRModel):
    MODEL_KEYS = SLRModel.MODEL_KEYS | ROOTSELF_KEYS

    def __init__(self, cfg=None, num_classes=None):
        super().__init__(cfg=cfg, num_classes=num_classes)
        mc = self.cfg.get("MODEL", {}) or {}
        self.rootself = bool(mc.get("ROOTSELF", False))
        stray = sorted(k for k in mc if k in ROOTSELF_KEYS and k != "ROOTSELF")
        if not self.rootself:
            if stray:
                raise ValueError(f"{stray} without MODEL.ROOTSELF: true would silently do nothing")
            return
        if not (self.slr and self.self_p > 0.0):
            raise ValueError("MODEL.ROOTSELF needs MODEL.SLR: true and TRACK.SELF_P > 0 (with DATA.CLIP_K)")
        tc = self.cfg.get("TRACK", {}) or {}
        self.rs_small = (float(tc.get("PREV_NOISE_T", 0.0)), float(tc.get("PREV_NOISE_POSE", 0.0)))
        self.rs_large = (float(tc.get("PREV_NOISE_LARGE_T", 0.0)), float(tc.get("PREV_NOISE_LARGE_POSE", 0.0)))
        self.rs_mixed = str(tc.get("PREV_NOISE_MODE", "gaussian")) == "mixed"
        self.rs_p = (float(tc.get("PREV_NOISE_P_SMALL", 1.0)), float(tc.get("PREV_NOISE_P_LARGE", 0.0)))
        g = torch.Generator(device="cpu")
        g.manual_seed(int(mc.get("ROOTSELF_SEED", 20261026)))
        self._rs_gen_state = g.get_state()

    def rootself_noise(self, n: int, n_fing: int):
        """(n, 3) translation and (n, n_fing) finger noise of the recipe's prev-noise model, float64, own generator."""
        g = torch.Generator(device="cpu")
        g.set_state(self._rs_gen_state)
        f64 = torch.float64
        u = torch.rand(n, generator=g, dtype=f64)
        zt = torch.randn(n, 3, generator=g, dtype=f64)
        zf = torch.randn(n, n_fing, generator=g, dtype=f64)
        mt = torch.rand(n, generator=g, dtype=f64)
        mf = torch.rand(n, generator=g, dtype=f64)
        self._rs_gen_state = g.get_state()
        (st, sf), (lt, lf) = self.rs_small, self.rs_large
        if not self.rs_mixed:
            return zt * st, zf * sf
        ps, pl = self.rs_p
        small, large = (u < ps), (u >= ps) & (u < ps + pl)
        directed = ~(small | large)
        nt = torch.where(small[:, None], zt * st, zt * lt)
        nf = torch.where(small[:, None], zf * sf, zf * lf)
        dt = zt / (zt.norm(dim=1, keepdim=True) + 1e-8) * (mt * lt)[:, None]
        df = zf / (zf.norm(dim=1, keepdim=True) + 1e-8) * (mf * lf)[:, None]
        nt = torch.where(directed[:, None], dt, nt)
        nf = torch.where(directed[:, None], df, nf)
        return nt, nf

    def _maybe_self(self, batch):
        if not getattr(self, "rootself", False):
            return super()._maybe_self(batch)
        # --- S42-B `MNISTModel._maybe_self`, line for line, up to the previous-state assignment
        if not (isinstance(batch, dict) and "main" in batch):
            return batch
        main = batch["main"]
        p = self.self_p_now() if self.training else 0.0
        self._self_stats = {"p": p, "rows": 0}
        if p <= 0.0 or "clip_codes" not in batch:
            return main
        x, prev, y, betas, cam = main
        dev = x.device
        valid_rows = batch["valid"].to(dev).bool().nonzero(as_tuple=False).reshape(-1)
        take = torch.rand(valid_rows.numel(), device=dev) < p
        if not bool(take.any()):
            return main
        rows, ci = valid_rows[take], take.nonzero(as_tuple=False).reshape(-1)
        codes = batch["clip_codes"].to(dev)[ci]
        w = batch["clip_w"].to(dev)[ci].float()
        K = codes.shape[1]
        lnes = codes.float() / w.view(-1, K, 1, 1, 1)
        n50 = batch["clip_n50"].to(dev)[ci]
        b_r, k_r = betas[rows].float(), cam[rows].float()
        was = self.training
        self.eval()
        try:
            with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
                s = batch["clip_init"].to(dev)[ci].float()
                for k in range(K - 1):
                    z = self(lnes[:, k], s, betas=b_r, camera_K=k_r)
                    s = self._self_filter(s, z.float(), n50[:, k], lnes[:, k])
        finally:
            self.train(was)
        x, prev, y = x.clone(), prev.clone(), y.clone()
        x[rows] = lnes[:, K - 1].to(x.dtype)
        # --- ROOTSELF: only the root block carries the deployed loop's own state
        tgt = batch["clip_tgt"].to(dev)[ci]
        gt_prev = tgt[:, K - 2].double()
        nt, nf = self.rootself_noise(int(rows.numel()), int(gt_prev.shape[1]) - 6)
        sl = self.slices
        new = gt_prev.clone()
        new[:, sl.transl] = gt_prev[:, sl.transl] + nt.to(dev)
        new[:, sl.root] = s[:, sl.root].double()
        new[:, 6:] = gt_prev[:, 6:] + nf.to(dev)
        prev[rows] = new.to(prev.dtype)
        y[rows] = tgt[:, K - 1].to(y.dtype)
        self._self_stats["rows"] = int(rows.numel())
        return (x, prev, y, betas, cam)
