#!/usr/bin/env python3
r"""[XYZ-ASSOC] step 4: frozen-weight probes that swap only the GEOMETRY the ray/mesh association is cast on.

Read-only on the repository and on the checkpoints: no model weight, config or library file is touched. The
probe replaces, for the duration of a forward and restoring them afterwards,
  * `semkine.xyz_mesh.associate_rays`    by a wrapper (`_xyz_mesh_forward` imports it inside the call), and
  * `model.xyz.forward`                  by a wrapper that can add `verts_geo=` (the repository's own option).
With the wrappers in pass-through mode the model is bit-identical to the unpatched one (checked at run time).

    A  predicted-old-state association vs ground-truth-old-state association (closed loop; in TF both are the same)
         A0g  own prev mesh, `verts_geo` = own prev mesh (z_prior = sum A zj instead of the hit depth t). Zero point
              of A1: it isolates the change of the z_prior convention alone.
         A1   assignment (vid / A / q / valid / hit) cast on GT pos51[prev_end], dXYZ / z_prior / edge geometry on the
              model's own prev mesh (`verts_geo`). Equivalent to the repository's oracle branch with
              `xyz_target_override = pos51[prev_end]` (checked bitwise on TF packets).
         A2   the whole association (z included) cast on GT pos51[prev_end]; dXYZ = z d - own prev vertex. Leaks GT geometry.
    B  window-start GT geometry vs event-time GT geometry (TF; every other input is the window-start GT prev)
         B0   = base (association on pos51[prev_end]);  B0g = A0g
         B1   each event associated on the GT mesh of its own millisecond (groups of one ms, P1's R3 layout), z from that mesh
         B1g  same assignment, geometry (dXYZ / z_prior / edges) on the window-start mesh (`verts_geo`)
    C  current candidate search vs exhaustive face search (TF and closed loop)
         C1   hit branch over all 1538 faces (nearest positive t); near / none branches unchanged

Protocol, sampling, labels and metrics are those of `tools/xa/probe_xa_assoc.py` (P2), which this file imports and reuses:
`track_sequence` closed loop (rng(0), 50 ms, val_core, noise scale 1), GT-previous-state TF loop (prev = pos51[prev_end],
first step pos51[a], no noise), P1 source labels for M / exact agreement.

Usage (run on the server; the GPU with the least used memory among 2..7 is chosen when CUDA_VISIBLE_DEVICES is unset):
    python tools/xa/probe_xa_geom.py --part tf,cl --noise-seeds 1,2     # main checkpoint; writes {A,B,C}_{tf,closedloop}_<ckpt>.json
    python tools/xa/probe_xa_geom.py --run-dir outputs/semkine/xa_c1_s3408 --part tf,cl
    python tools/xa/probe_xa_geom.py --part cl --resume                  # re-do the closed-loop statistics from outputs/.../cache/
    python tools/xa/probe_xa_geom.py --part dxyz                         # dXYZ-length diagnostic of the frozen-weight caveat
    python tools/xa/probe_xa_geom.py --part report                       # REPORT.md (+ NOTES.md if present)
    python tools/xa/probe_xa_geom.py --random-init --cpu --max-steps 8 --part tf,cl --out-dir /tmp/xa_geom_selftest
"""
from __future__ import annotations

import os
import subprocess
import sys

sys.dont_write_bytecode = True

GPU_POOL = (2, 3, 4, 5, 6, 7)          # the authorised cards
GPU_INFO = None


def _early_gpu_choice():
    """Before torch/CUDA is touched: choose the authorised card with the least used memory (nvidia-smi PCI order)."""
    global GPU_INFO
    if "--cpu" in sys.argv:
        return
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return
    used = {}
    for line in out.strip().splitlines():
        i, u, t = [int(x) for x in line.split(",")]
        used[i] = (u, t)
    if not used:
        return
    pre = os.environ.get("CUDA_VISIBLE_DEVICES")
    chosen = None
    if pre is None:
        pool = {i: used[i][0] for i in GPU_POOL if i in used}
        chosen = min(pool, key=pool.get)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen)
    GPU_INFO = {"used_mib_at_start": {i: used[i][0] for i in sorted(used)}, "total_mib": {i: used[i][1] for i in sorted(used)},
                "authorised": list(GPU_POOL), "preset_CUDA_VISIBLE_DEVICES": pre, "chosen_physical_gpu": chosen,
                "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"), "time": __import__("time").strftime("%F %T")}


_early_gpu_choice()

import argparse                                                     # noqa: E402
import json                                                         # noqa: E402
import math                                                         # noqa: E402
import time                                                         # noqa: E402
from collections import defaultdict                                 # noqa: E402
from pathlib import Path                                            # noqa: E402

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_xa_assoc as P2                                         # noqa: E402  (P2 probe: loading, labels, counts, stats helpers)
import semkine.xyz_mesh as XM                                       # noqa: E402
from semkine import eval_track as ET                                # noqa: E402

REPO = P2.REPO
ST = P2.ST
LOCAL = P2.LOCAL
log = P2.log
OUT_DEFAULT = "outputs/xa/step4/geom"
BOOT_B, BOOT_L = 2000, P2.BLOCK            # block bootstrap: 2000 resamples, block length 20 steps
CAP_MM = 1000.0                          # closed-loop per-step errors are capped here (NaN / inf count as the cap) before any paired statistic
TIE_EPS = 1e-6                              # metres: |t_full - t_cand| below this is the same surface point (a tie)
EXTRA_KEYS = ("n_changed", "n_changed_hand", "n_jchg", "n_jchg_hand", "c_closer", "c_closer_hand", "c_near2hit", "c_near2hit_hand",
              "c_none2hit", "c_none2hit_hand", "c_lost", "c_lost_hand", "c_tie", "c_tie_hand")
# per-step metric names that go into paired tables
TF_STEP = ("ra_mm", "e_rot_deg", "e_t_mm", "e_fing_rad")
CL_STEP = ("ra_mm", "abs_mm", "root_t_mm")
RATIOS = {"M_all": ("n_mis", "n_cf"), "M_has": ("n_mis", "n_cf_has"), "agree_exact": ("n_agree", "n_den"),
          "hit_frac": ("n_hit", "n_live"), "valid_frac": ("n_valid", "n_live"), "joint_changed_frac": ("n_jchg", "n_live"),
          "event_changed_frac": ("n_changed", "n_live")}
TF_VARIANTS = ("base", "pass", "A0g", "A1", "A2", "oracle", "Bident", "B1", "B1g", "C1")
SANITY_ONLY = ("pass", "A1", "A2", "oracle", "Bident")            # run on every `sanity_every`-th packet
CL_VARIANTS = ("base", "pass", "A0g", "A0g2", "A1", "A2", "C1")
GEO_SELF = ("A0g", "A0g2", "A1", "B1g")                                   # model.xyz gets verts_geo = the model's own prev mesh
DESC = {"base": "现行关联（猴补前，未打补丁）", "pass": "打补丁但直通（应与 base 逐位一致）",
        "A0g": "自身 prev 网格关联 + verts_geo=自身 prev（z_prior 口径改为 Σ A·zj；A1 的对照零点）",
        "A0g2": "A0g 的重复运行（GPU 浮点 index_add 原子加的非确定性底，用于判断闭环差值是否超出重复运行噪声）",
        "A1": "分配由真值旧状态 pos51[prev_end] 网格算，dXYZ/z_prior/边几何仍用自身 prev 网格（verts_geo）",
        "A2": "整个 assoc（含 z）由真值旧状态网格算，dXYZ = z·d − 自身 prev 顶点（含 GT 泄漏）",
        "oracle": "仓库 oracle 分支（xyz_assoc_prev=target, override=pos51[prev_end]），用于核对 A1",
        "Bident": "按毫秒分组关联但 50 组都用 prev 网格（应与 base 一致，核对分组/回填）",
        "B1": "每个事件用其所在毫秒的真值网格关联（含 z），其余同 base",
        "B1g": "B1 的分配 + verts_geo=窗首网格（A1 方式）",
        "C1": "命中分支改为对全部 1538 个面求交（取最近正 t），near/none 分支不变"}


# --------------------------------------------------------------------------- statistics
class Boot:
    """Moving-block bootstrap over steps (block 20), resampling inside each sequence; groups pool sequences."""

    def __init__(self, ns: dict, B=BOOT_B, L=BOOT_L, seed=0):
        self.ns, self.B = dict(ns), B
        self.idx, self._ones, self._rs = {}, {}, {}
        for k, (s, n) in enumerate(sorted(self.ns.items())):
            rng = np.random.default_rng(seed * 1000 + k)
            self.idx[s] = np.stack([P2.block_idx(n, rng, L) for _ in range(B)]).astype(np.int32) if n else np.zeros((B, 0), np.int32)

    def ones(self, s):
        if s not in self._ones:
            self._ones[s] = np.ones(self.ns[s])
        return self._ones[s]

    def rsum(self, arr, s):
        """Replicate sums of `arr` (one array per sequence) over the bootstrap blocks, cached per array object."""
        key = (id(arr), s)
        if key not in self._rs:
            self._rs[key] = (arr, np.asarray(arr, np.float64)[self.idx[s]].sum(1))   # keep `arr` alive so its id stays unique
        return self._rs[key][1]

    def stat(self, arrays: dict, fn, group):
        """`arrays` name -> {seq: per-step array}; `fn(**sums)` of sums over the group's steps; returns est and 95% CI."""
        pt = {k: sum(float(np.sum(v[s])) for s in group) for k, v in arrays.items()}
        rp = {k: sum(self.rsum(v[s], s) for s in group) for k, v in arrays.items()}
        est = float(fn(**pt))
        dist = np.asarray(fn(**rp), np.float64)
        dist = dist[np.isfinite(dist)]
        ci = [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))] if len(dist) else [float("nan")] * 2
        return {"est": est, "ci95": ci}


def sdiv(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.divide(a, b, out=np.full(np.broadcast(a, b).shape, np.nan), where=b > 0)


def groups_of(seqs):
    return {"overall": list(seqs), "global": [s for s in seqs if not LOCAL(s)], "local": [s for s in seqs if LOCAL(s)]}


def pair_block(A, B, seqs, boot, step_names, ratio_names, mask=None):
    """Paired differences A - B (A = variant, B = baseline) of per-step metrics and of event-pooled ratios.
    `mask` (seq -> 0/1 array): the step metrics are averaged over the masked-in steps only (ratio metrics stay unmasked)."""
    out = {"step": {}, "ratio": {}}
    for g, ss in groups_of(seqs).items():
        if not ss:
            continue
        n_steps = sum(len(A[s]["j"]) for s in ss) if mask is None else int(sum(np.sum(mask[s]) for s in ss))
        for nm in step_names:
            if mask is None:
                arrs = {"a": {s: A[s][nm] for s in ss}, "b": {s: B[s][nm] for s in ss}, "n": {s: boot.ones(s) for s in ss}}
            else:
                arrs = {"a": {s: A[s][nm] * mask[s] for s in ss}, "b": {s: B[s][nm] * mask[s] for s in ss}, "n": {s: np.asarray(mask[s], np.float64) for s in ss}}
            out["step"].setdefault(nm, {})[g] = {
                "n_steps": n_steps,
                "base": boot.stat(arrs, lambda a, b, n: sdiv(b, n), ss), "variant": boot.stat(arrs, lambda a, b, n: sdiv(a, n), ss),
                "diff": boot.stat(arrs, lambda a, b, n: sdiv(a - b, n), ss)}
        for nm in (ratio_names if mask is None else ()):
            nk, dk = RATIOS[nm]
            if nk not in A[ss[0]] or nk not in B[ss[0]]:
                continue
            arrs = {"na": {s: A[s][nk] for s in ss}, "da": {s: A[s][dk] for s in ss},
                    "nb": {s: B[s][nk] for s in ss}, "db": {s: B[s][dk] for s in ss}}
            out["ratio"].setdefault(nm, {})[g] = {
                "n_steps": n_steps,
                "base": boot.stat(arrs, lambda na, da, nb, db: sdiv(nb, db), ss),
                "variant": boot.stat(arrs, lambda na, da, nb, db: sdiv(na, da), ss),
                "diff": boot.stat(arrs, lambda na, da, nb, db: sdiv(na, da) - sdiv(nb, db), ss)}
    return out


def ratio_block(A, seqs, boot, names):
    """Event-pooled ratios of one variant with block-bootstrap CI (no baseline)."""
    out = {}
    for g, ss in groups_of(seqs).items():
        if not ss:
            continue
        for nm in names:
            nk, dk = RATIOS[nm] if nm in RATIOS else nm
            if nk not in A[ss[0]]:
                continue
            arrs = {"n": {s: A[s][nk] for s in ss}, "d": {s: A[s][dk] for s in ss}}
            key = nm if isinstance(nm, str) else f"{nk}/{dk}"
            out.setdefault(key, {})[g] = boot.stat(arrs, lambda n, d: sdiv(n, d), ss)
    return out


# --------------------------------------------------------------------------- the patch
class GeomPatch:
    """Wrappers around `associate_rays` and `model.xyz.forward`; `variant` selects what they do."""

    def __init__(self, model, device, lbs_hard):
        self.model, self.device, self.lbs_hard = model, device, lbs_hard
        self.variant = None
        self.installed = False
        self.ctx = None
        self.betas = None
        self.step_fn = None
        self.last = None
        self.dt_closer_mm = []
        self.collect_dxyz = False
        self.dxyz_log, self.dxyz_by_seq = [], {}
        self.exh_check = False
        self.exh = defaultdict(int)
        self._last_orig_hit = None
        self.calls = 0
        self._key, self._c = None, {}
        self._orig_assoc = XM.associate_rays
        self._orig_xyz_forward = model.xyz.forward

    # ---- install / uninstall
    def install(self):
        if not self.installed:
            XM.associate_rays = self._assoc
            self.model.xyz.forward = self._xyz_fwd
            self.installed = True

    def uninstall(self):
        if self.installed:
            XM.associate_rays = self._orig_assoc
            delattr(self.model.xyz, "forward")
            self.installed = False

    def set_seq(self, ctx, betas):
        self.ctx, self.betas = ctx, betas
        self._key, self._c = None, {}

    # ---- per-step ground-truth geometry (lazy, cached per step)
    def _cache(self):
        st = self.step_fn()
        key = (id(self.ctx), st.j)
        if key != self._key:
            self._key, self._c = key, {}
        return st, self._c

    @torch.no_grad()
    def verts_prev_gt(self):
        st, c = self._cache()
        if "vp" not in c:
            p = torch.from_numpy(self.ctx.pos51[st.prev_idx].astype(np.float32)).view(1, -1).to(self.device)
            c["vp"] = self.model._fk(p, self.betas)[0]
        return c["vp"]

    @torch.no_grad()
    def verts_ms(self):
        st, c = self._cache()
        if "vm" not in c:
            a = st.end - ST + 1
            p = torch.from_numpy(np.ascontiguousarray(self.ctx.pos51[a:st.end + 1], dtype=np.float32)).to(self.device)
            c["vm"] = self.model._fk(p, self.betas.expand(len(p), -1))[0]
        return c["vm"]

    def ms_rel(self):
        """Window-relative millisecond (0..49) of each sampled event, in the model's sampling order."""
        st, c = self._cache()
        if "ms" not in c:
            a = st.end - ST + 1
            counts = np.diff(self.ctx.offsets[a:st.end + 2]).astype(np.int64)
            ms_abs = np.repeat(np.arange(a, st.end + 1, dtype=np.int64), counts)
            assert len(ms_abs) == st.n_ev
            c["ms"] = ms_abs[P2.sampled_index(st.n_ev)] - a
        return c["ms"]

    # ---- wrappers
    @torch.no_grad()
    def _dxyz_stats(self, d, mask, verts, assoc, verts_geo):
        """Length (cm) of the relation encoder's dXYZ = (z_prior d - V[vid]) / L_REL over the matched (event, vertex) records,
        recomputed with the formulas of `XYZMeshNet.forward` (diagnostic only; nothing here feeds back)."""
        w = assoc["q"].unsqueeze(-1) * assoc["A"] * mask.float().unsqueeze(-1)
        rb, rn, rr = (w > 0).nonzero(as_tuple=True)
        if rb.numel() == 0:
            return None
        vrec, drec = assoc["vid"][rb, rn, rr], d[rb, rn].float()
        wrec = w[rb, rn, rr].float()
        if verts_geo is None:
            zrec = assoc["z"][rb, rn] if assoc["z"].dim() == 2 else assoc["z"][rb, rn, rr]
            vxyz = verts[rb, vrec].float()
        else:
            vxyz = verts_geo.float()[rb, vrec]
            zj = (vxyz * drec).sum(-1) / (drec * drec).sum(-1)
            N = d.shape[1]
            ekey = rb * N + rn
            wsum = torch.zeros(d.shape[0] * N, device=d.device).index_add(0, ekey, wrec)
            zsum = torch.zeros(d.shape[0] * N, device=d.device).index_add(0, ekey, wrec * zj)
            zrec = (zsum / wsum.clamp_min(1e-12))[ekey]
        dx = ((zrec.unsqueeze(-1) * drec - vxyz) / XM.L_REL).norm(dim=-1)
        q = torch.quantile(dx.float(), torch.tensor([0.5, 0.9, 0.99], device=dx.device))
        return {"n": int(dx.numel()), "sum": float(dx.sum()), "p50": float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
                "n_gt5cm": int((dx > 5).sum()), "n_gt10cm": int((dx > 10).sum()), "max": float(dx.max())}

    def _xyz_fwd(self, tok, d, mask, verts, assoc, prev48, ablate=False, verts_geo=None):
        if self.variant in GEO_SELF and verts_geo is None:
            verts_geo = verts                            # the model's own prev mesh (what the oracle branch passes)
        if self.collect_dxyz:
            st = self._dxyz_stats(d, mask, verts, assoc, verts_geo)
            if st is not None:
                self.dxyz_log.append(st)
        return self._orig_xyz_forward(tok, d, mask, verts, assoc, prev48, ablate=ablate, verts_geo=verts_geo)

    def _assoc(self, d, mask, verts, faces, vert_faces, **kw):
        assert d.shape[0] == 1 and "mode" not in kw and "normals" not in kw, "probe is for B=1, hit_first"
        orig = self._orig_assoc(d, mask, verts, faces, vert_faces, **kw)
        self.calls += 1
        v = self.variant
        n = int(mask.sum())
        self._last_orig_hit = orig["hit"][0, :n]
        codes = None
        if v in (None, "pass", "A0g", "A0g2"):
            new = orig
        elif v in ("A1", "A2"):
            new = self._orig_assoc(d, mask, self.verts_prev_gt(), faces, vert_faces, **kw)
        elif v in ("B1", "B1g", "Bident"):
            new = self._grouped(d, mask, verts, faces, vert_faces, kw, ident=(v == "Bident"))
        elif v == "C1":
            new, codes = self._brute(d, mask, verts, faces, orig)
            if self.exh_check:
                self._exhaustive_check(d, mask, verts, faces, vert_faces, kw, new, n)
        else:
            raise ValueError(v)
        if new is orig:
            ch = np.zeros(n, bool)
            jc = np.zeros(n, bool)
        else:
            ch = np.zeros(d.shape[1], bool)
            for key in ("vid", "A"):
                ch |= (new[key] != orig[key]).any(-1)[0].cpu().numpy()
            for key in ("q", "valid", "hit"):
                ch |= (new[key] != orig[key])[0].cpu().numpy()
            eo, en = P2.event_majority(orig, self.lbs_hard), P2.event_majority(new, self.lbs_hard)
            jc = (eo["has"] != en["has"]) | (eo["has"] & en["has"] & (eo["maj"] != en["maj"]))
            ch, jc = ch[:n], jc[:n]
        self.last = {"changed": ch, "jchg": jc, "codes": codes}
        return new

    @torch.no_grad()
    def _exhaustive_check(self, d, mask, verts, faces, vert_faces, kw, new, n):
        """With k = V every vertex is a candidate, so the stock search sees every face: an independent exhaustive reference
        for the hit branch. C1 must reproduce it on every event the reference hits (and must not hit where it does not)."""
        ref = self._orig_assoc(d, mask, verts, faces, vert_faces, **{**kw, "k": int(verts.shape[1]), "chunk": 64})
        rh, nh = ref["hit"][0, :n], new["hit"][0, :n]
        both = rh & nh
        zdiff = (ref["z"][0, :n] - new["z"][0, :n]).abs()
        face_eq = (ref["vid"][0, :n, :3] == new["vid"][0, :n, :3]).all(-1)
        c = self.exh
        c["n_events"] += n
        c["n_ref_hit"] += int(rh.sum())
        c["n_hit_flag_differs"] += int((rh != nh).sum())
        c["n_z_differs_gt_tie"] += int((both & (zdiff > TIE_EPS)).sum())
        c["n_same_t_other_face"] += int((both & (zdiff <= TIE_EPS) & ~face_eq).sum())
        c["n_ref_vs_stock_k16_hit_differs"] += int((rh != self._last_orig_hit).sum())

    @torch.no_grad()
    def _grouped(self, d, mask, verts, faces, vert_faces, kw, ident):
        """P1's R3 layout: events grouped by millisecond (B = 50), each group cast on its own GT mesh, merged back."""
        ml = self.ms_rel()
        n = len(ml)
        assert n == int(mask.sum())
        N = d.shape[1]
        dev = d.device
        V50 = verts.expand(50, -1, -1) if ident else self.verts_ms()
        counts = np.bincount(ml, minlength=50)
        Nm = max(int(counts.max()), 1)
        order = np.argsort(ml, kind="stable")
        sm = ml[order]
        starts = np.searchsorted(sm, np.arange(50))
        colp = np.empty(n, np.int64)
        colp[order] = np.arange(n) - starts[sm]
        ml_t, col_t = torch.from_numpy(ml).to(dev), torch.from_numpy(colp).to(dev)
        dp = d.new_zeros(50, Nm, 3)
        dp[ml_t, col_t] = d[0, :n]
        mk = torch.zeros(50, Nm, dtype=torch.bool, device=dev)
        mk[ml_t, col_t] = True
        kw2 = dict(kw)
        for key in ("band", "front_tol"):
            kw2[key] = kw[key].expand(50).contiguous()
        res = self._orig_assoc(dp, mk, V50.contiguous(), faces, vert_faces, **kw2)
        out = {}
        for key, val in res.items():
            full = torch.zeros((1, N) + tuple(val.shape[2:]), dtype=val.dtype, device=dev)
            full[0, :n] = val[ml_t, col_t]
            out[key] = full
        return out

    @torch.no_grad()
    def _brute(self, d, mask, verts, faces, orig, chunk=256):
        """Exhaustive hit search over all faces; overrides the candidate-search result where it is strictly worse."""
        n = int(mask.sum())
        dev = d.device
        tri = faces.long()
        Vf = verts.float()[0]
        P = Vf[tri]                                                          # (F, 3, 3)
        p0, p1, p2 = P[:, 0][None], P[:, 1][None], P[:, 2][None]
        dd_all = d[0, :n].float()
        tm, fs, uu, vv = [], [], [], []
        for i0 in range(0, n, chunk):
            dd = dd_all[i0:i0 + chunk]
            t, u, v = XM.ray_triangle(dd.view(-1, 1, 3), p0, p1, p2)         # (m, F)
            tmin, fsel = t.min(dim=-1)
            tm.append(tmin); fs.append(fsel)
            uu.append(u.gather(1, fsel[:, None]).squeeze(1)); vv.append(v.gather(1, fsel[:, None]).squeeze(1))
        tmin, fsel, uh, vh = (torch.cat(x) if x else torch.zeros(0, device=dev) for x in (tm, fs, uu, vv))
        is_hit = torch.isfinite(tmin)
        wb = torch.stack([(1.0 - uh - vh).clamp_min(0), uh.clamp_min(0), vh.clamp_min(0), torch.zeros_like(uh)], dim=-1)
        A_hit = wb / wb.sum(-1, keepdim=True).clamp_min(1e-12)
        fv = tri[fsel]
        vid_new = torch.cat([fv, fv[:, :1]], dim=1)
        hit_o, valid_o, t_o = orig["hit"][0, :n], orig["valid"][0, :n], orig["z"][0, :n]
        same_face = (orig["vid"][0, :n, :3] == fv).all(-1)
        override = is_hit & (~hit_o | (tmin < t_o - TIE_EPS))
        lost = hit_o & ~is_hit
        tie_diff = hit_o & is_hit & ~same_face & ~override
        codes = torch.zeros(n, dtype=torch.int8, device=dev)
        codes[override & hit_o] = 1                                          # hit, but a nearer face was missed
        codes[override & ~hit_o & valid_o] = 2                               # near layer -> hit
        codes[override & ~hit_o & ~valid_o] = 3                              # none -> hit
        codes[lost] = 4
        codes[tie_diff] = 5                                                  # another face at the same t (kept as is)
        if bool((override & hit_o).any()):
            self.dt_closer_mm.append(((t_o - tmin)[override & hit_o] * 1000.0).cpu().numpy())
        new = {k: v.clone() for k, v in orig.items()}
        sel = override.nonzero().squeeze(1)
        if sel.numel():
            new["vid"][0, sel] = vid_new[sel]
            new["A"][0, sel] = A_hit[sel]
            new["q"][0, sel] = 1.0
            new["z"][0, sel] = tmin[sel]
            new["valid"][0, sel] = True
            new["hit"][0, sel] = True
        return new, codes.cpu().numpy()


def extra_counts(last, part_s):
    out = {k: 0 for k in EXTRA_KEYS}
    if last is None:
        return out
    hand = part_s >= 0
    ch, jc = last["changed"], last["jchg"]
    out["n_changed"], out["n_changed_hand"] = int(ch.sum()), int((ch & hand).sum())
    out["n_jchg"], out["n_jchg_hand"] = int(jc.sum()), int((jc & hand).sum())
    if last["codes"] is not None:
        for name, c in (("closer", 1), ("near2hit", 2), ("none2hit", 3), ("lost", 4), ("tie", 5)):
            m = last["codes"] == c
            out[f"c_{name}"], out[f"c_{name}_hand"] = int(m.sum()), int((m & hand).sum())
    return out


# --------------------------------------------------------------------------- metrics of a trajectory
def step_metrics(mano, betas, preds, gts, device):
    """Per-step MPJPE root-aligned / absolute (mm, `track_sequence`'s formulas) and the root translation error (mm)."""
    pj, _ = ET.mano_fk(mano, preds, betas, device)
    gj, _ = ET.mano_fk(mano, gts, betas, device)
    ra = np.linalg.norm(ET.root_align(pj) - ET.root_align(gj), axis=-1).mean(-1) * 1000
    ab = np.linalg.norm(pj - gj, axis=-1).mean(-1) * 1000
    rt = np.linalg.norm(preds[:, 0:3].astype(np.float64) - gts[:, 0:3].astype(np.float64), axis=1) * 1000
    rj = np.linalg.norm(pj[:, 0] - gj[:, 0], axis=-1) * 1000
    return {"ra_mm": ra, "abs_mm": ab, "root_t_mm": rt, "root_joint_err_mm": rj}


def gt_stack(ctx):
    return np.stack([ctx.pos51[st.end] for st in ctx.steps]).astype(np.float32)


def frame_weighted(per_seq_means: dict, n: dict):
    tot = sum(n.values())
    return sum(per_seq_means[s] * n[s] for s in n) / max(tot, 1)


def summarize_arrays(arr, seqs):
    """Means / pooled event rates of one variant over each sequence, the two groups and overall."""
    out = {}
    for g, ss in groups_of(seqs).items():
        if not ss:
            continue
        o = {"n_steps": int(sum(len(arr[s]["j"]) for s in ss))}
        for nm in set(TF_STEP + CL_STEP):
            if nm in arr[ss[0]]:
                x = np.concatenate([arr[s][nm] for s in ss])
                o[nm] = float(x.mean())
                o[nm + "_median"] = float(np.median(x))
                o[nm + "_n_nonfinite"] = int((~np.isfinite(x)).sum())
        if "ra_mm" in arr[ss[0]]:
            x = np.concatenate([arr[s]["ra_mm"] for s in ss])
            o["frac_steps_ra_gt_100mm"] = float((x > 100).mean())
        for k in ("ra_mm", "abs_mm", "root_t_mm"):
            if k + "_capped" in arr[ss[0]]:
                o[k + "_n_capped"] = int(sum(np.sum(arr[s][k + "_capped"]) for s in ss))
                o[k + "_n_nonfinite_steps"] = int(sum(np.sum(arr[s][k + "_nonfinite"]) for s in ss))
        sums = {k: float(sum(np.sum(arr[s][k]) for s in ss)) for k in P2.COUNT_KEYS + EXTRA_KEYS if k in arr[ss[0]]}
        o["sums"] = sums
        if all(k in sums for k in P2.COUNT_KEYS):
            o["rates"] = P2.rates(sums)
        out[g] = o
    return out


# --------------------------------------------------------------------------- common setup
def make_args(a):
    return argparse.Namespace(run_dir=a.run_dir, config=a.config, cpu=a.cpu, threads=a.threads, random_init=a.random_init, ckpt=a.ckpt,
                              max_steps=a.max_steps, labels_dir=a.labels_dir)


def ckpt_tag(run_dir):
    return Path(run_dir).name


def build_env(a):
    args = make_args(a)
    with P2.limited_runs(a.max_steps):
        cfg, cfg_path, device, mano, root, seqs, model, ckpt = P2.load_everything(args)
        env = dict(cfg=cfg, cfg_path=cfg_path, device=device, mano=mano, root=root, seqs=seqs, model=model, ckpt=ckpt,
                   cl=P2.build_ctx_labels(root, seqs, REPO / a.labels_dir), args=args)
    return env


def meta_of(a, env, extra=None):
    m = P2.meta(env["args"], env["cfg_path"], env["ckpt"], env["device"], extra)
    m["script"] = "tools/xa/probe_xa_geom.py"
    m["gpu_info"] = GPU_INFO
    m["peak_gpu_mem_alloc_mib"] = (torch.cuda.max_memory_allocated() / 2 ** 20) if env["device"].type == "cuda" else None
    m["seqs"] = [s for s, _ in env["seqs"]]
    m["boot"] = {"resamples": BOOT_B, "block_len_steps": BOOT_L, "scheme": "moving blocks inside each sequence, same indices for variant and baseline"}
    return m


def stack_cols(col: dict):
    return {k: np.asarray(v) for k, v in col.items()}


# --------------------------------------------------------------------------- TF
class Rec:
    def __init__(self):
        self.d = defaultdict(list)
        self.preds = []

    def add(self, pred32, **kw):
        self.preds.append(pred32)
        for k, v in kw.items():
            self.d[k].append(v)


def tf_forward(model, G, v, batch, st, ctx, device):
    """One teacher-forced forward for variant `v`; leaves the model unpatched afterwards."""
    G.last = None
    G.variant = v
    with torch.no_grad():
        if v == "base":
            G.uninstall()
            return model.forward_packet(batch)
        if v == "oracle":
            G.uninstall()
            model.xyz_assoc_prev = "target"
            model.xyz_target_override = torch.from_numpy(ctx.pos51[st.prev_idx].astype(np.float32)).view(1, -1).to(device)
            try:
                return model.forward_packet(batch)
            finally:
                model.xyz_assoc_prev = "self"
                model.xyz_target_override = None
        G.install()
        try:
            return model.forward_packet(batch)
        finally:
            G.uninstall()


def run_tf(a, env, out_dir):
    device, model, mano = env["device"], env["model"], env["mano"]
    tag = ckpt_tag(a.run_dir)
    lbs_hard = model.xyz.lbs_hard.to(device)
    model.xyz_keep_aux = True
    t0 = time.time()
    G = GeomPatch(model, device, lbs_hard)
    seqs = [s for s, _ in env["seqs"]]
    arrs = {v: {} for v in TF_VARIANTS}
    sanity = defaultdict(lambda: defaultdict(float))
    p2_ref = {}
    SAN_PAIRS = (("pass", "base"), ("A2", "base"), ("A1", "A0g"), ("oracle", "A1"), ("Bident", "base"))
    dtc = []
    for seq, d in env["seqs"]:
        ctx, lab = env["cl"][seq]
        P2.set_context(model, ctx, device)
        betas = torch.tensor(ctx.aux["betas"], dtype=torch.float32, device=device).view(1, -1)
        G.set_seq(ctx, betas)
        lab_ms = np.load(lab.path)["ms"]
        recs = {v: Rec() for v in TF_VARIANTS}
        ts = time.time()
        gts = []
        G.dt_closer_mm = []
        for st, ev5, prev, gt, batch in P2.tf_packets(ctx, device):
            part_s, kind_s, fit, idx = lab.step(st, ev5)
            G.step_fn = lambda st=st: st
            gts.append(gt)
            if st.n_ev and st.j % 20 == 0:                         # ms grid of the model's sampling vs the P1 labels
                e0 = int(lab.start[st.j])
                a0 = st.end - ST + 1
                counts = np.diff(ctx.offsets[a0:st.end + 2]).astype(np.int64)
                assert (np.repeat(np.arange(a0, st.end + 1), counts) == lab_ms[e0:e0 + st.n_ev]).all(), "ms differs from P1 labels"
                assert (G.ms_rel() + a0 == lab_ms[e0:e0 + st.n_ev][idx]).all(), "sampled ms differs from P1 labels"
            preds = {}
            G.exh_check = (st.j % a.sanity_every == 0)
            for v in TF_VARIANTS:
                if v in SANITY_ONLY and st.j % a.sanity_every:
                    continue
                pred = tf_forward(model, G, v, batch, st, ctx, device)
                preds[v] = pred
                err = P2.update_errors(pred.double().cpu().numpy()[0], prev, gt)
                if st.n_ev:
                    ev = P2.event_majority(model.xyz_last["assoc"], lbs_hard)
                    cnt, _, _ = P2.count_step(part_s, kind_s, ev)
                    rs = model.route_stats
                    route = {k: float(rs[k]) for k in P2.ROUTE_KEYS}
                else:
                    cnt = {k: 0 for k in P2.COUNT_KEYS}
                    route = {k: float("nan") for k in P2.ROUTE_KEYS}
                recs[v].add(pred.cpu().numpy()[0], j=st.j, **{k: err[k] for k in ("e_rot_deg", "e_rot_geo_deg", "e_fing_rad", "e_t_mm")},
                            **cnt, **route, **extra_counts(G.last, part_s))
            for x, y in SAN_PAIRS:
                if x in preds and y in preds:
                    s = sanity[f"{x}_vs_{y}"]
                    s["n_packets"] += 1
                    s["n_bitwise_equal"] += int(torch.equal(preds[x], preds[y]))
                    s["max_abs_diff"] = max(s["max_abs_diff"], float((preds[x] - preds[y]).abs().max()))
            if (st.j + 1) % 200 == 0:
                log(f"TF {tag} {seq}: {st.j + 1}/{len(ctx.steps)} packets ({time.time() - ts:.0f}s)")
        gts = np.stack(gts)
        dtc.append(np.concatenate(G.dt_closer_mm) if G.dt_closer_mm else np.zeros(0))
        for v, r in recs.items():
            if not r.preds:
                continue
            cols = stack_cols(r.d)
            if v in SANITY_ONLY:
                continue
            cols.update(step_metrics(mano, betas, np.stack(r.preds), gts, device))
            arrs[v][seq] = cols
        log(f"TF {tag} {seq}: done {len(ctx.steps)} packets, {time.time() - ts:.0f}s")
    model.xyz_keep_aux = False
    G.uninstall()
    assert XM.associate_rays is G._orig_assoc, "associate_rays was not restored"
    ns = {s: len(arrs["base"][s]["j"]) for s in seqs}
    boot = Boot(ns)
    TFS = ("base", "A0g", "B1", "B1g", "C1")
    # ---- per-variant summaries and the print that proves the substitution took effect
    summ = {v: summarize_arrays(arrs[v], seqs) for v in TFS}
    for v in TFS:
        for g in ("global", "local"):
            s = summ[v][g]
            r = s["rates"]
            log(f"[TF {tag}] {v:5s} {g:6s} steps={s['n_steps']} replaced events={int(s['sums']['n_changed'])}/{int(s['sums']['n_live'])} "
                f"(joint reassigned {int(s['sums']['n_jchg'])})  M_all={100 * r['M_all']:.2f}% exact={100 * r['agree_exact']:.2f}% "
                f"hit={100 * r['hit_frac']:.2f}% RA={s['ra_mm']:.4f}")
    pairs = {}
    for name, (x, y) in {"B1_vs_base": ("B1", "base"), "B1g_vs_A0g": ("B1g", "A0g"), "B1g_vs_base": ("B1g", "base"),
                         "A0g_vs_base": ("A0g", "base"), "C1_vs_base": ("C1", "base")}.items():
        pairs[name] = pair_block(arrs[x], arrs[y], seqs, boot, TF_STEP, list(RATIOS))
    # ---- C1 hit-set change
    cstats = ratio_block(arrs["C1"], seqs, boot, [("c_closer", "n_live"), ("c_near2hit", "n_live"), ("c_none2hit", "n_live"), ("c_lost", "n_live"),
                                                  ("c_tie", "n_live"), ("n_changed", "n_live"), ("c_closer_hand", "n_hand"),
                                                  ("c_near2hit_hand", "n_hand"), ("c_none2hit_hand", "n_hand"), ("c_lost_hand", "n_hand"),
                                                  ("n_changed_hand", "n_hand"), ("n_hit", "n_live")])
    dt = np.concatenate(dtc) if dtc else np.zeros(0)
    cdt = {"n": int(len(dt))}
    if len(dt):
        cdt.update({"mean_mm": float(dt.mean()), **{f"p{q}": float(np.percentile(dt, q)) for q in (10, 50, 90, 99)}, "max_mm": float(dt.max()),
                    "frac_gt_1mm": float((dt > 1).mean()), "frac_gt_5mm": float((dt > 5).mean())})
    sec = time.time() - t0
    # ---- sanity
    san = {k: dict(v) for k, v in sanity.items()}
    for k, s in san.items():
        log(f"[TF {tag}] sanity {k}: {int(s['n_bitwise_equal'])}/{int(s['n_packets'])} packets bitwise equal, max|diff|={s['max_abs_diff']:.3g}")
    # ---- raw per-step arrays for traceability
    np.savez_compressed(out_dir / f"tf_perstep_{tag}.npz", **{f"{v}|{s}|{k}": x for v in TFS for s, c in arrs[v].items() for k, x in c.items()})
    common = {"meta": meta_of(a, env, {"seconds": sec, "protocol": "TF: prev = pos51[prev_end] (first step pos51[a]), no noise; same packets for every variant",
                                       "n_steps": ns}), "desc": {v: DESC[v] for v in TF_VARIANTS}}
    A_json = {**common, "variants": {v: summ[v] for v in ("base", "A0g")}, "pairs": {"A0g_vs_base": pairs["A0g_vs_base"]},
              "sanity": {k: san[k] for k in san if k.split("_vs_")[0] in ("pass", "A2", "A1", "oracle")},
              "note": "TF: prev is already the GT, so A1 == A0g, A2 == base (bitwise). A0g - base is the z_prior-convention change alone."}
    B_json = {**common, "variants": {v: summ[v] for v in ("base", "A0g", "B1", "B1g")},
              "pairs": {k: pairs[k] for k in ("B1_vs_base", "B1g_vs_A0g", "B1g_vs_base")},
              "sanity": {k: san[k] for k in san if k.startswith("Bident")}}
    C_json = {**common, "variants": {v: summ[v] for v in ("base", "C1")}, "pairs": {"C1_vs_base": pairs["C1_vs_base"]}, "hit_set_change": cstats,
              "nearer_face_depth_gain_mm": cdt, "tie_eps_m": TIE_EPS,
              "exhaustive_check": {"what": "C1 vs the stock search with k=778 (every face a candidate), TF packets with j % sanity_every == 0",
                                   **dict(G.exh)}}
    P2.dump(out_dir / f"A_tf_{tag}.json", A_json)
    P2.dump(out_dir / f"B_tf_{tag}.json", B_json)
    P2.dump(out_dir / f"C_tf_{tag}.json", C_json)
    return arrs


# --------------------------------------------------------------------------- closed loop
class GeomHook(P2.ClosedLoopHook):
    """P2's closed-loop hook (association stats against the P1 labels) plus the predictions and the patch's counters."""

    def __init__(self, model, ctx, lab, lbs_hard, G):
        super().__init__(model, ctx, lab, lbs_hard)
        self.G, self.preds = G, []
        self.xcols = defaultdict(list)

    def __call__(self, kind, prev_t, pred):
        if kind != "step":
            return pred
        j = self.i
        st = self.ctx.steps[j]
        part_s = self.lab.step(st)[0] if st.n_ev else np.zeros(0, np.int64)
        out = super().__call__(kind, prev_t, pred)
        self.preds.append(pred.detach().cpu().numpy()[0])
        ex = extra_counts(self.G.last if st.n_ev else None, part_s)
        for k, v in ex.items():
            self.xcols[k].append(v)
        self.G.last = None
        return out


def run_cl_variant(a, env, G, v, seed=0):
    """One closed-loop pass of variant `v` over val_core. Returns per-seq arrays and track_sequence's own results."""
    device, model, mano, cfg, root = env["device"], env["model"], env["mano"], env["cfg"], env["root"]
    lbs_hard = model.xyz.lbs_hard.to(device)
    rng = np.random.default_rng(seed)                 # one rng across the sequences, as select_checkpoint
    model.xyz_keep_aux = True
    G.variant = v
    if v == "base":
        G.uninstall()
    else:
        G.install()
    arrs, ress, preds_all = {}, {}, {}
    try:
        for seq, d in env["seqs"]:
            ctx, lab = env["cl"][seq]
            betas = torch.tensor(ctx.aux["betas"], dtype=torch.float32, device=device).view(1, -1)
            G.set_seq(ctx, betas)
            hook = GeomHook(model, ctx, lab, lbs_hard, G)
            G.step_fn = lambda ctx=ctx, hook=hook: ctx.steps[hook.i]
            G.last = None
            ts = time.time()
            G.dxyz_log = []
            res = ET.track_sequence(model, mano, cfg, root, d, seq, ST, device, rng, 1.0, 1.0, None, state_hook=hook)
            assert hook.i == len(ctx.steps) == res["n_frames"], (hook.i, len(ctx.steps), res["n_frames"])
            preds = np.stack(hook.preds)
            gts = gt_stack(ctx)
            sm = step_metrics(mano, betas, preds, gts, device)
            assert np.isclose(float(sm["ra_mm"].mean()), res["mpjpe_ra_mm"], rtol=1e-9, atol=1e-9, equal_nan=True), "recomputed RA differs from track_sequence"
            assert np.isclose(float(sm["abs_mm"].mean()), res["mpjpe_abs_mm"], rtol=1e-9, atol=1e-9, equal_nan=True), "recomputed abs MPJPE differs from track_sequence"
            cols = stack_cols(hook.cols)
            cols.update(stack_cols(hook.xcols))
            for k in ("ra_mm", "abs_mm", "root_t_mm"):                 # raw kept; the standard names carry the capped values
                cols[k + "_raw"] = sm[k]
                cols[k + "_nonfinite"] = (~np.isfinite(sm[k])).astype(np.float64)
                cols[k + "_capped"] = (~(sm[k] < CAP_MM)).astype(np.float64)
                sm[k] = np.where(sm[k] < CAP_MM, sm[k], CAP_MM)
            cols.update(sm)
            cols["j"] = np.asarray(hook.cols["j"])
            arrs[seq], ress[seq], preds_all[seq] = cols, res, preds
            s = P2.summarize_p2a(hook.cols)
            arrs[seq]["_summary_p2a"] = s
            if G.collect_dxyz:
                G.dxyz_by_seq[(v, seq)], G.dxyz_log = G.dxyz_log, []
            log(f"CL {v:5s} {seq}: {len(preds)} steps RA={res['mpjpe_ra_mm']:.4f} abs={res['mpjpe_abs_mm']:.4f} ({time.time() - ts:.0f}s)"
                f" nonfinite steps={int(cols['ra_mm_nonfinite'].sum())} capped(>={CAP_MM:.0f}mm) steps={int(cols['ra_mm_capped'].sum())}"
                f" M_all={100 * s['pooled']['M_all']:.2f}% exact={100 * s['pooled']['agree_exact']:.2f}% "
                f"replaced={int(np.sum(arrs[seq]['n_changed']))}/{int(np.sum(arrs[seq]['n_live']))}")
    finally:
        G.uninstall()
        model.xyz_keep_aux = False
        G.variant = None
    return arrs, ress, preds_all


def divergence_stats(arrs, variants, seqs):
    out = {}
    for v in variants:
        out[v] = {}
        for g, ss in groups_of(seqs).items():
            if not ss:
                continue
            n = nd = nr = nrd = 0
            first = []
            for s in ss:
                c = arrs[v][s]
                d_, rid = c["diverged"] > 0, c["run_id"]
                n, nd = n + len(d_), nd + int(d_.sum())
                for r in np.unique(rid):
                    m = rid == r
                    nr += 1
                    if (d_ & m).any():
                        nrd += 1
                        first.append(int(np.argmax((d_ & m)[m])))      # steps since the run's first step
            out[v][g] = {"n_steps": n, "n_diverged": nd, "frac_diverged": nd / max(n, 1), "n_runs": nr, "n_runs_diverged": nrd,
                         "first_diverged_step_in_run": first[:40]}
    return out


def cl_cached(a, env, G, v, out_dir, seed=0):
    """`run_cl_variant` with a per-pass pickle cache (`out_dir/cache/`), so a crash does not lose finished passes (`--resume`)."""
    import pickle
    tag = ckpt_tag(a.run_dir)
    f = out_dir / "cache" / f"cl_{tag}_{v}_seed{seed}.pkl"
    if a.resume and f.exists() and not (a.random_init or a.max_steps):
        log(f"CL {v} seed {seed}: loaded from {f.name}")
        return pickle.loads(f.read_bytes())
    res = run_cl_variant(a, env, G, v, seed=seed)
    if not (a.random_init or a.max_steps):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(pickle.dumps(res))
    return res


def ra_overall(ress):
    n = {s: r["n_frames"] for s, r in ress.items()}
    return {"overall": frame_weighted({s: r["mpjpe_ra_mm"] for s, r in ress.items()}, n),
            "per_seq": {s: r["mpjpe_ra_mm"] for s, r in ress.items()}, "n_frames": n}


def run_cl(a, env, out_dir):
    device, model = env["device"], env["model"]
    tag = ckpt_tag(a.run_dir)
    lbs_hard = model.xyz.lbs_hard.to(device)
    G = GeomPatch(model, device, lbs_hard)
    seqs = [s for s, _ in env["seqs"]]
    t0 = time.time()
    variants = [v for v in CL_VARIANTS if (not a.cl_variants or v in a.cl_variants.split(","))]
    arrs, ress, preds = {}, {}, {}
    for v in variants:
        arrs[v], ress[v], preds[v] = cl_cached(a, env, G, v, out_dir)
        ra = ra_overall(ress[v])
        P2.dump(out_dir / f"cl_progress_{tag}.json", {"finished": list(ress), "ra_by_variant": {k: ra_overall(r) for k, r in ress.items()}})
        log(f"[CL {tag}] {v}: RA overall={ra['overall']:.4f} per-seq={ {s: round(x, 4) for s, x in ra['per_seq'].items()} }")
    assert XM.associate_rays is G._orig_assoc and "forward" not in model.xyz.__dict__, "patch not restored"
    # ---- baseline reproduction: selection JSON (and P2's recorded numbers)
    ref = P2.find_ra_reference(REPO / a.run_dir) if not (a.random_init or a.max_steps) else None
    base_ra = ra_overall(ress["base"])
    repro = None
    if ref:
        repro = P2.compute_ra_check(base_ra, ref)
        repro["bitwise_equal_overall"] = bool(base_ra["overall"] == ref["overall"])
        repro["bitwise_equal_per_seq"] = {s: bool(base_ra["per_seq"][s] == ref["per_seq"][s]) for s in base_ra["per_seq"]} if ref.get("per_seq") else None
        log(f"[CL {tag}] baseline vs {ref['source']}: |diff| overall {repro['abs_diff_overall']:.3e} match={repro['match']} "
            f"bitwise={repro['bitwise_equal_overall']}")
    p2f = REPO / "outputs/xa/step2/p2a_closedloop.json"
    p2_cmp = None
    if p2f.exists() and tag == "xa_c1_s3407" and not (a.random_init or a.max_steps):
        p2 = json.loads(p2f.read_text())["ra"]
        p2_cmp = {"p2_overall": p2["overall"], "p2_per_seq": p2["per_seq"], "this_overall": base_ra["overall"], "this_per_seq": base_ra["per_seq"],
                  "bitwise_equal": bool(p2["overall"] == base_ra["overall"] and all(p2["per_seq"][s] == base_ra["per_seq"][s] for s in base_ra["per_seq"]))}
        log(f"[CL {tag}] baseline vs P2 p2a_closedloop.json: bitwise_equal={p2_cmp['bitwise_equal']}")
    passthrough = {s: {"preds_bitwise_equal": bool(np.array_equal(preds["base"][s], preds["pass"][s])),
                       "ra_equal": bool(ress["base"][s]["mpjpe_ra_mm"] == ress["pass"][s]["mpjpe_ra_mm"])} for s in seqs} if "pass" in preds else None
    if passthrough:
        log(f"[CL {tag}] pass-through patch vs unpatched: {passthrough}")
    ns = {s: len(arrs["base"][s]["j"]) for s in seqs}
    boot = Boot(ns)
    for v in variants:                                   # a step is 'diverged' if any of RA / absolute / root-translation error is >= CAP_MM or non-finite
        for s in seqs:
            c = arrs[v][s]
            c["diverged"] = ((c["ra_mm_capped"] + c["abs_mm_capped"] + c["root_t_mm_capped"]) > 0).astype(np.float64)
    div = divergence_stats(arrs, variants, seqs)
    for v in variants:
        for g in ("global", "local"):
            d_ = div[v][g]
            log(f"[CL {tag}] {v:5s} {g:6s} diverged steps (RA/abs/root-t >= {CAP_MM:.0f} mm or non-finite): {d_['n_diverged']}/{d_['n_steps']}; runs affected {d_['n_runs_diverged']}/{d_['n_runs']}")
    summ = {v: summarize_arrays(arrs[v], seqs) for v in variants}
    # per-sequence P2-style association stats (pooled + route means + prev error), no raw columns
    p2sum = {v: {s: {k: x for k, x in arrs[v][s]["_summary_p2a"].items()} for s in seqs} for v in variants}
    for v in variants:
        for g in ("global", "local"):
            s = summ[v][g]
            r = s["rates"]
            log(f"[CL {tag}] {v:5s} {g:6s} steps={s['n_steps']} replaced events={int(s['sums']['n_changed'])}/{int(s['sums']['n_live'])} "
                f"(joint reassigned {int(s['sums']['n_jchg'])}) M_all={100 * r['M_all']:.2f}% exact={100 * r['agree_exact']:.2f}% "
                f"hit={100 * r['hit_frac']:.2f}% RA={s['ra_mm']:.4f}")
    pair_defs = {"A0g_vs_base": ("A0g", "base"), "A1_vs_A0g": ("A1", "A0g"), "A1_vs_base": ("A1", "base"), "A2_vs_base": ("A2", "base"),
                 "A2_vs_A1": ("A2", "A1"), "C1_vs_base": ("C1", "base"), "A0g2_vs_A0g": ("A0g2", "A0g")}
    pairs = {k: pair_block(arrs[x], arrs[y], seqs, boot, CL_STEP, list(RATIOS)) for k, (x, y) in pair_defs.items() if x in arrs and y in arrs}
    pairs_healthy = {k: pair_block(arrs[x], arrs[y], seqs, boot, CL_STEP, (),
                                   mask={s: (1.0 - arrs[x][s]["diverged"]) * (1.0 - arrs[y][s]["diverged"]) for s in seqs})
                     for k, (x, y) in pair_defs.items() if x in arrs and y in arrs}
    # ---- init-noise floor of the baseline (context for closed-loop differences): other rng seeds of the initial noise
    floor = None
    if a.noise_seeds:
        floor = {}
        for sd in [int(x) for x in a.noise_seeds.split(",")]:
            _, rs_, _ = cl_cached(a, env, G, "base", out_dir, seed=sd)
            floor[str(sd)] = ra_overall(rs_)
            log(f"[CL {tag}] baseline with init-noise rng seed {sd}: RA {floor[str(sd)]['overall']:.4f} {floor[str(sd)]['per_seq']}")
        floor["0"] = base_ra
    C_extra = ratio_block(arrs["C1"], seqs, boot, [("c_closer", "n_live"), ("c_near2hit", "n_live"), ("c_none2hit", "n_live"), ("c_lost", "n_live"),
                                                   ("c_tie", "n_live"), ("n_changed", "n_live"), ("c_closer_hand", "n_hand"), ("c_near2hit_hand", "n_hand"),
                                                   ("c_none2hit_hand", "n_hand"), ("c_lost_hand", "n_hand"), ("n_changed_hand", "n_hand"), ("n_hit", "n_live")]) if "C1" in arrs else None
    # ---- state error along the closed-loop trajectory (variants change the trajectory)
    traj = {v: {s: {"prev_err_t_mm_mean": float(np.mean(arrs[v][s]["prev_err_t_mm"])), "prev_err_rot_deg_mean": float(np.mean(arrs[v][s]["prev_err_rot_deg"])),
                    "prev_err_pose_rad_mean": float(np.mean(arrs[v][s]["prev_err_pose_rad"]))} for s in seqs} for v in variants}
    np.savez_compressed(out_dir / f"cl_perstep_{tag}.npz", **{f"{v}|{s}|{k}": x for v in variants for s in seqs for k, x in arrs[v][s].items()
                                                              if isinstance(x, np.ndarray)})
    ra_by_variant = {v: ra_overall(ress[v]) for v in variants}
    common = {"meta": meta_of(a, env, {"seconds": time.time() - t0, "protocol": "closed loop: track_sequence, rng(0), 50 ms, val_core, noise scale 1, one rng shared across sequences",
                                       "n_steps": ns}), "desc": {v: DESC[v] for v in variants}, "ra_by_variant": ra_by_variant,
              "baseline_repro": {"selection_json": repro, "p2a_closedloop": p2_cmp, "passthrough_patch": passthrough}}
    A_json = {**common, "variants": {v: summ[v] for v in variants if v != "C1"}, "pairs": {k: pairs[k] for k in pairs if k.startswith("A")},
              "pairs_both_healthy": {k: pairs_healthy[k] for k in pairs_healthy if k.startswith("A")}, "divergence": div,
              "p2a_style_stats": {v: p2sum[v] for v in variants if v != "C1"}, "trajectory_state_error": traj, "init_noise_floor_baseline_ra": floor}
    C_json = {**common, "variants": {v: summ[v] for v in ("base", "C1") if v in summ}, "pairs": {"C1_vs_base": pairs.get("C1_vs_base")},
              "pairs_both_healthy": {"C1_vs_base": pairs_healthy.get("C1_vs_base")}, "divergence": {v: div[v] for v in ("base", "C1") if v in div}, "hit_set_change": C_extra,
              "p2a_style_stats": {v: p2sum[v] for v in ("base", "C1") if v in p2sum}, "trajectory_state_error": {v: traj[v] for v in ("base", "C1") if v in traj}}
    P2.dump(out_dir / f"A_closedloop_{tag}.json", A_json)
    P2.dump(out_dir / f"C_closedloop_{tag}.json", C_json)
    return arrs


# --------------------------------------------------------------------------- report
TAGS = ("xa_c1_s3407", "xa_c1_s3408", "xa_c1r_s3407")
GN = {"overall": "overall", "global": "global", "local": "local"}


def _ok(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _n(v, nd, plus=False):
    """Fixed-point, switching to 3 significant digits for diverged (|v| >= 1e5) values."""
    if abs(v) >= 1e5:
        return f"{v:+.3g}" if plus else f"{v:.3g}"
    return f"{v:+.{nd}f}" if plus else f"{v:.{nd}f}"


def fci(x, nd=3, scale=1.0):
    """'+d [lo, hi]*' (star: the 95% interval excludes 0)."""
    if not x or not _ok(x.get("est")):
        return "n/a"
    e = x["est"] * scale
    lo, hi = x["ci95"]
    if not (_ok(lo) and _ok(hi)):
        return f"{_n(e, nd, True)} [n/a]"
    lo, hi = lo * scale, hi * scale
    return f"{_n(e, nd, True)} [{_n(lo, nd, True)}, {_n(hi, nd, True)}]" + ("*" if (lo > 0 or hi < 0) else "")


def fv(x, nd=3, scale=1.0, ci=False):
    if not x or not _ok(x.get("est")):
        return "n/a"
    s = _n(x["est"] * scale, nd)
    if ci and _ok(x["ci95"][0]) and _ok(x["ci95"][1]):
        s += f" [{_n(x['ci95'][0] * scale, nd)}, {_n(x['ci95'][1] * scale, nd)}]"
    return s


def sig(x):
    """True: the 95% interval of the difference excludes 0."""
    return bool(x and _ok(x.get("est")) and _ok(x["ci95"][0]) and _ok(x["ci95"][1]) and (x["ci95"][0] > 0 or x["ci95"][1] < 0))


def fnum(x, nd=3, pct=False):
    if not _ok(x):
        return "n/a"
    return f"{100 * x:.{nd - 1}f}%" if pct else _n(x, nd)


def a_sanity_every(D):
    return 5


def make_report(out_dir):
    out_dir = Path(out_dir)
    L = []
    P = L.append

    def J(name):
        p = out_dir / name
        return json.loads(p.read_text()) if p.exists() else None

    main = TAGS[0]
    Atf, Btf, Ctf = (J(f"{k}_tf_{main}.json") for k in "ABC")
    Adx = J(f"A_dxyz_{main}.json")
    Acl, Ccl = J(f"A_closedloop_{main}.json"), J(f"C_closedloop_{main}.json")
    ref = (Atf or Btf or Ctf or Acl)
    meta = ref["meta"]
    ns_tf = (Btf or Atf)["meta"]["n_steps"] if (Btf or Atf) else {}
    ns_cl = Acl["meta"]["n_steps"] if Acl else {}

    def cap(tag, mode, ns):
        gl = [s for s in ns if not LOCAL(s)]
        ll = [s for s in ns if LOCAL(s)]
        tot = sum(ns.values())
        return (f"检查点 `{tag}` · 模式 **{mode}** · 序列：global = {gl[0] if gl else '-'}（{ns.get(gl[0], 0) if gl else 0} 包），"
                f"local = {ll[0] if ll else '-'}（{ns.get(ll[0], 0) if ll else 0} 包），overall = {tot} 包（帧加权）· "
                f"差值 = 变体 − 基线，方括号 = 95% 区间（步块 bootstrap，块长 {BOOT_L}，{BOOT_B} 次，序列内重采样，变体与基线用同一组索引），`*` = 区间不含 0")

    P("# [XYZ-ASSOC] step4 关联几何探针报告（冻结权重，只换关联所用的几何）\n")
    P(f"- 脚本：`tools/xa/probe_xa_geom.py`（复用 `tools/xa/probe_xa_assoc.py` 的加载 / 闭环 / TF / 标签对齐 / 计数）；产物：`outputs/xa/step4/geom/`；运行日志 `run_<ckpt>.log`；逐步原始数组 `tf_perstep_<ckpt>.npz`、`cl_perstep_<ckpt>.npz`。")
    P(f"- 检查点（均 step=6000，配置 = 各 run 目录的 `config_resolved.yaml`）：主 `outputs/semkine/xa_c1_s3407/xa_c1_s3407-step=6000.ckpt`；种子稳健性 `xa_c1_s3408`、`xa_c1r_s3407` 同名 step=6000。")
    gi = meta.get("gpu_info") or {}
    P(f"- 设备：{meta['device']}（CUDA_VISIBLE_DEVICES={meta['cuda_visible_devices']}，{meta['gpu_name']}，torch {meta['torch']}）；"
      f"启动时各卡已用 MiB（nvidia-smi PCI 序）{gi.get('used_mib_at_start')}；授权 2–7 中显存最少者被选；详见 `gpu_choice.txt`；探针峰值显存分配 {meta.get('peak_gpu_mem_alloc_mib')} MiB。")
    P("- 协议：**闭环** = `eval_track.track_sequence`（`default_rng(0)` 一条流跨序列、50 ms、val_core = zgz_global + zgz_local、初值噪声 scale 1）；**TF** = 真值前态循环（prev = pos51[prev_end]，每序列 run 首步 pos51[a]，不加噪，`make_eval_packet` + `forward_packet`）；"
      "两者都用模型实际看到的 `stride_sample(ptr, 2048)` 事件，M / 精确一致率按 P1 标签（`outputs/xa/step1/labels_zgz_*.npz`）对齐，口径同 P2（M_all = 错配事件 / 手指轮廓前景事件，联合质量多数落在关节 0；精确一致率 = argmax 关节 = src_part，分母 = 有关联的手部事件）。")
    P("- RA = 根对齐 MPJPE（mm，`track_sequence` 同公式）；绝对 MPJPE = 不对齐；根平移误差 = ‖pred[0:3] − gt[0:3]‖（mm）；TF 的 e_rot = ‖(pred−prev)[3:6] − (gt−prev)[3:6]‖（度，轴角向量差，P2 口径），e_t = 平移更新误差（mm），e_fing = 手指更新误差（rad）。\n")
    P(f"- 闭环每步的 RA / 绝对 MPJPE / 根平移误差在进入任何配对统计之前先封顶到 {CAP_MM:.0f} mm（NaN / inf 也计为 {CAP_MM:.0f} mm；各表『封顶步数』列出数量）；"
      "封顶只作用于闭环的逐步数组，选点复现核对用的是 `track_sequence` 自己返回的未封顶均值。\n")
    P("变体定义（A = 预测旧状态 vs 真值旧状态；B = 窗首 vs 事件时刻；C = 候选搜索 vs 全面片搜索）：\n")
    P("| 变体 | 含义 |\n|---|---|")
    for v, t in DESC.items():
        P(f"| {v} | {t} |")
    P("")

    def dxyz_table(block, title):
        P(title)
        P("| 变体 | 序列 | 调用数（其中状态已非有限） | (事件, 顶点) 记录数 | dXYZ 长度均值 (cm) | 逐步 p50 的 中位 / p90 步 (cm) | 逐步 p90 / p99 均值 (cm) | > 5 cm 占比 | > 10 cm 占比 |")
        P("|---|---|---|---|---|---|---|---|---|")
        for v, per in block.items():
            for sq, x in per.items():
                if not x or "n_records" not in x:
                    P(f"| {v} | {sq} | {x['n_calls'] if x else 0}（{x['n_calls_nonfinite'] if x else 0}） | - | - | - | - | - | - |")
                    continue
                P(f"| {v} | {sq} | {x['n_calls']}（{x['n_calls_nonfinite']}） | {x['n_records']} | {_n(x['mean_cm'], 2)} | {_n(x['per_step_p50_cm_median'], 2)} / {_n(x['per_step_p50_cm_p90'], 2)} | "
                  f"{_n(x['per_step_p90_cm_mean'], 2)} / {_n(x['per_step_p99_cm_mean'], 2)} | {100 * x['frac_gt5cm']:.2f}% | {100 * x['frac_gt10cm']:.2f}% |")
        P("")

    # P1 / P2 reference numbers
    p1 = {}
    try:
        st = json.loads((REPO / "outputs/xa/step1/assoc_stats.json").read_text())["sets"]["sampled"]
        for sq in st:
            p1[sq] = {rg: {"M": st[sq][rg]["cur"]["M"]["maj0_rate"], "exact": st[sq][rg]["cur"]["hand"]["exact_given_valid"]} for rg in ("R3", "R0")}
    except Exception:
        pass
    p2a = None
    try:
        p2a = json.loads((REPO / "outputs/xa/step2/p2a_closedloop.json").read_text())
    except Exception:
        pass

    # ------------------------------------------------------------------ 0. key readings
    def kp(D, pair, key, g, nd=3, ratio=False, scale=1.0, unit=""):
        PR = (D or {}).get("pairs", {}).get(pair)
        if not PR:
            return "n/a"
        blk = (PR["ratio"] if ratio else PR["step"]).get(key, {}).get(g)
        if not blk:
            return "n/a"
        d = blk["diff"]
        return f"{fci(d, nd, scale)}{unit}（{'区间不含 0' if sig(d) else '区间含 0，噪声内'}）"

    P("## 0. 关键读数（由脚本按 JSON 生成；差值 = 变体 − 基线，方括号 = 步块 bootstrap 95% 区间）\n")
    if Acl:
        P(f"**A（检查点 `{main}`，闭环，{sum(ns_cl.values())} 步：global {ns_cl.get('zgz_global')} / local {ns_cl.get('zgz_local')}）**\n")
        P(f"（闭环数值：发散步的 RA / 绝对 MPJPE / 根平移误差按 {CAP_MM:.0f} mm 封顶后计入均值；A1 / A2 的 ΔRA 因此主要反映发散步占比，见下一条与 2.2 的条件读数。）")
        for pr, lab in (("A1_vs_A0g", "A1 − A0g（只换分配）"), ("A1_vs_base", "A1 − base"), ("A2_vs_base", "A2 − base（含泄漏）"), ("A0g_vs_base", "A0g − base（z_prior 口径）"), ("A0g2_vs_A0g", "A0g2 − A0g（重复运行底）")):
            P(f"- {lab}：ΔRA overall {kp(Acl, pr, 'ra_mm', 'overall')}；global {kp(Acl, pr, 'ra_mm', 'global')}；local {kp(Acl, pr, 'ra_mm', 'local')}；"
              f"Δ绝对 MPJPE overall {kp(Acl, pr, 'abs_mm', 'overall', 2)}；Δ根平移误差 overall {kp(Acl, pr, 'root_t_mm', 'overall', 2)}。")
        sA1 = Acl["variants"].get("A1", {})
        if sA1:
            P("- A1 的闭环关联统计（按 P1 标签）：" + "；".join(f"{g} M_all = {fnum(sA1[g]['rates']['M_all'], 3, True)}、精确一致率 = {fnum(sA1[g]['rates']['agree_exact'], 3, True)}"
                                                      f"（基线 {fnum(Acl['variants']['base'][g]['rates']['M_all'], 3, True)} / {fnum(Acl['variants']['base'][g]['rates']['agree_exact'], 3, True)}；"
                                                      f"P1 R0 {fnum(p1.get('zgz_' + g, {}).get('R0', {}).get('M'), 3, True)} / {fnum(p1.get('zgz_' + g, {}).get('R0', {}).get('exact'), 3, True)}）"
                                                      for g in ("global", "local")) + "。")
        dv = Acl.get("divergence") or {}
        if dv:
            P("- 闭环发散（RA / 绝对 MPJPE / 根平移误差任一 ≥ 1000 mm 或非有限的步 / 全部步）：" + "；".join(
                f"{v}：" + "、".join(f"{g} {dv[v][g]['n_diverged']}/{dv[v][g]['n_steps']}" for g in ("global", "local")) for v in ("base", "A0g", "A0g2", "A1", "A2") if v in dv) + "。")
        ph = Acl.get("pairs_both_healthy") or {}
        if ph.get("A1_vs_A0g"):
            P(f"- 条件读数（仅两变体都未发散的步）A1 − A0g：ΔRA overall {kp({'pairs': ph}, 'A1_vs_A0g', 'ra_mm', 'overall')}、global {kp({'pairs': ph}, 'A1_vs_A0g', 'ra_mm', 'global')}、local {kp({'pairs': ph}, 'A1_vs_A0g', 'ra_mm', 'local')}。")
        P("")
    if Btf:
        P(f"**B（检查点 `{main}`，TF，{sum(Btf['meta']['n_steps'].values())} 包：global {Btf['meta']['n_steps'].get('zgz_global')} / local {Btf['meta']['n_steps'].get('zgz_local')}）**\n")
        for pr, lab in (("B1_vs_base", "B1 − B0(base)"), ("B1g_vs_A0g", "B1g − B0g(A0g)"), ("B1g_vs_base", "B1g − base")):
            P(f"- {lab}：ΔRA overall {kp(Btf, pr, 'ra_mm', 'overall')}、global {kp(Btf, pr, 'ra_mm', 'global')}、local {kp(Btf, pr, 'ra_mm', 'local')}；"
              f"Δe_rot(°) overall {kp(Btf, pr, 'e_rot_deg', 'overall')}、local {kp(Btf, pr, 'e_rot_deg', 'local')}；Δe_t(mm) overall {kp(Btf, pr, 'e_t_mm', 'overall')}；"
              f"Δe_fing(rad) overall {kp(Btf, pr, 'e_fing_rad', 'overall', 4)}；ΔM_all(pp) global {kp(Btf, pr, 'M_all', 'global', 2, True, 100)}、local {kp(Btf, pr, 'M_all', 'local', 2, True, 100)}。")
        P("")
    if Ctf or Ccl:
        P(f"**C（检查点 `{main}`）**\n")
        for D, mode in ((Ctf, "TF"), (Ccl, "闭环")):
            if D:
                n_ = sum(D["meta"]["n_steps"].values())
                hs = D.get("hit_set_change") or {}
                P(f"- {mode}（{n_} 步）C1 − base：ΔRA overall {kp(D, 'C1_vs_base', 'ra_mm', 'overall')}、global {kp(D, 'C1_vs_base', 'ra_mm', 'global')}、local {kp(D, 'C1_vs_base', 'ra_mm', 'local')}；"
                  + (f"Δe_rot(°) overall {kp(D, 'C1_vs_base', 'e_rot_deg', 'overall')}；Δe_t(mm) overall {kp(D, 'C1_vs_base', 'e_t_mm', 'overall')}；Δe_fing(rad) overall {kp(D, 'C1_vs_base', 'e_fing_rad', 'overall', 4)}；" if mode == "TF" else
                     f"Δ绝对 MPJPE overall {kp(D, 'C1_vs_base', 'abs_mm', 'overall', 2)}；Δ根平移 overall {kp(D, 'C1_vs_base', 'root_t_mm', 'overall', 2)}；")
                  + "命中集合变化占 live 事件：" + "；".join(f"{g} 原 hit 另有更近面 {fv(hs.get('c_closer/n_live', {}).get(g), 3, 100, True)}%、near→hit {fv(hs.get('c_near2hit/n_live', {}).get(g), 3, 100, True)}%、none→hit {fv(hs.get('c_none2hit/n_live', {}).get(g), 3, 100, True)}%"
                                                         for g in ("global", "local")) + "。")
        P("")
    sd_lines = []
    for tag in TAGS[1:]:
        Dt, Dc = J(f"B_tf_{tag}.json"), J(f"A_closedloop_{tag}.json")
        if Dt or Dc:
            sd_lines.append(f"- `{tag}`：TF B1 − base ΔRA overall {kp(Dt, 'B1_vs_base', 'ra_mm', 'overall')}；闭环 A1 − A0g ΔRA overall {kp(Dc, 'A1_vs_A0g', 'ra_mm', 'overall')}（详见第 5 节）。")
    if sd_lines:
        P("**种子稳健性（只列总 RA）**\n")
        for x in sd_lines:
            P(x)
        P("")
    notes = out_dir / "NOTES.md"
    if notes.exists():
        P("### 人工解读\n")
        P(notes.read_text().rstrip() + "\n")

    # ------------------------------------------------------------------ 1. self-checks
    P("## 1. 自检\n")
    if Acl:
        br = Acl["baseline_repro"]
        sj = br.get("selection_json")
        if sj:
            P(f"- **闭环基线 vs 选点结果** `{sj['reference']['source']}`（{main}，step 6000，闭环，{sum(ns_cl.values())} 包）：本次 overall {sj['this_run_overall']:.4f}，选点 {sj['reference']['overall']:.4f}，|差| = {sj['abs_diff_overall']:.2e}，"
              f"逐序列 |差| = { {s: f'{v:.1e}' for s, v in sj['abs_diff_per_seq'].items()} }，**逐位相同 = {sj.get('bitwise_equal_overall')}**（逐序列 {sj.get('bitwise_equal_per_seq')}）。")
        p2c = br.get("p2a_closedloop")
        if p2c:
            P(f"- **闭环基线 vs P2 记录**（`outputs/xa/step2/p2a_closedloop.json`）：overall {p2c['this_overall']:.4f} / global {p2c['this_per_seq'].get('zgz_global', float('nan')):.4f} / local {p2c['this_per_seq'].get('zgz_local', float('nan')):.4f}；"
              f"P2 记录 {p2c['p2_overall']:.4f} / {p2c['p2_per_seq'].get('zgz_global', float('nan')):.4f} / {p2c['p2_per_seq'].get('zgz_local', float('nan')):.4f}；**逐位相同 = {p2c['bitwise_equal']}**。")
        pt = br.get("passthrough_patch")
        if pt:
            P(f"- **猴补前后（直通补丁 vs 未打补丁）** 闭环逐步预测逐位相同：{ {s: v['preds_bitwise_equal'] for s, v in pt.items()} }；RA 逐位相同：{ {s: v['ra_equal'] for s, v in pt.items()} }。")
    if Atf:
        P(f"- 下列 TF 核对变体只在 `j % {a_sanity_every(Atf)} == 0` 的包上运行（{int(next(iter(Atf['sanity'].values()))['n_packets'])} 个包，主检查点全部序列）；其余包只跑 base / A0g / B1 / B1g / C1。")
        for k, s in Atf["sanity"].items():
            P(f"- TF 核对 `{k}`：{int(s['n_bitwise_equal'])}/{int(s['n_packets'])} 个包预测逐位相同，max|Δpred| = {s['max_abs_diff']:.2e}"
              + ("（走 `verts_geo` 的 z_prior 用 float32 `index_add` 原子加，GPU 上求和顺序非确定，故 A1/oracle 与 A0g 之间只差 ~1e-7，不是补丁差异；不走 `verts_geo` 的路径用 float64 累加，逐位相同）" if ("A1" in k or "oracle" in k) else "") + "。")
    if Btf:
        for k, s in Btf["sanity"].items():
            P(f"- TF 核对 `{k}`（50 毫秒分组→回填，用 prev 网格填满 50 组应与整包一次关联一致）：{int(s['n_bitwise_equal'])}/{int(s['n_packets'])} 个包预测逐位相同，max|Δpred| = {s['max_abs_diff']:.2e}。")
    if Ctf:
        ex = Ctf.get("exhaustive_check", {})
        P(f"- C1 与『k = 778（每个顶点都是候选 = 现行函数自己的穷尽搜索）』对照（TF，{int(ex.get('n_events', 0))} 个事件，{int(ex.get('n_ref_hit', 0))} 个穷尽命中）："
          f"命中标志不同 {int(ex.get('n_hit_flag_differs', 0))}、命中深度差 > {TIE_EPS * 1000:.3f} mm 的 {int(ex.get('n_z_differs_gt_tie', 0))}、同深度不同面 {int(ex.get('n_same_t_other_face', 0))}；"
          f"现行 k = 16 与穷尽命中标志不同的事件 {int(ex.get('n_ref_vs_stock_k16_hit_differs', 0))}（= C1 改变的命中集合）。")
    P("- 每个变体的『被替换事件数』：见各表的“事件被替换 / 关节被改派”列（被替换 = vid / A / q / valid / hit 任一与模型自身关联不同；关节被改派 = 有无关联或 argmax 关节不同）；运行日志逐变体逐序列打印了同样数字加 M / 一致率。\n")

    # ------------------------------------------------------------------ helpers for tables
    def abs_table(D, variants, metrics, title):
        P(title)
        hdr = "| 变体 | 序列(包数) | " + " | ".join(m[1] for m in metrics) + " |"
        P(hdr)
        P("|---|---|" + "---|" * len(metrics))
        for v in variants:
            if v not in D["variants"]:
                continue
            for g in ("overall", "global", "local"):
                s = D["variants"][v].get(g)
                if not s:
                    continue
                cells = []
                for key, _, nd in metrics:
                    cells.append(fnum(s.get(key), nd))
                P(f"| {v} | {g} ({s['n_steps']}) | " + " | ".join(cells) + " |")
        P("")

    def pair_table(D, names, labels, steps, title):
        """steps: [(key, header, nd)]."""
        P(title)
        P("| 对比 | 序列(包数) | " + " | ".join(h for _, h, _ in steps) + " |")
        P("|---|---|" + "---|" * len(steps))
        for nm in names:
            if nm not in D["pairs"] or not D["pairs"][nm]:
                continue
            PR = D["pairs"][nm]
            for g in ("overall", "global", "local"):
                if g not in PR["step"][steps[0][0]]:
                    continue
                n = PR["step"][steps[0][0]][g]["n_steps"]
                P(f"| {labels.get(nm, nm)} | {g} ({n}) | " + " | ".join(fci(PR["step"][k][g]["diff"], nd) for k, _, nd in steps) + " |")
        P("")

    def assoc_table(D, names, labels, title, with_changed=True):
        P(title)
        cols = [("M_all", "ΔM_all (pp)"), ("agree_exact", "Δ精确一致率 (pp)"), ("hit_frac", "Δhit 比例 (pp)"), ("valid_frac", "Δvalid 比例 (pp)")]
        P("| 对比 | 序列(包数) | " + " | ".join(h for _, h in cols) + (" | 事件被替换 | 关节被改派 |" if with_changed else " |"))
        P("|---|---|" + "---|" * (len(cols) + (2 if with_changed else 0)))
        for nm in names:
            if nm not in D["pairs"] or not D["pairs"][nm]:
                continue
            PR = D["pairs"][nm]
            for g in ("overall", "global", "local"):
                if g not in PR["ratio"].get("M_all", {}):
                    continue
                n = PR["ratio"]["M_all"][g]["n_steps"]
                cells = [fci(PR["ratio"][k][g]["diff"], 2, 100) for k, _ in cols]
                if with_changed:
                    cells.append(fv(PR["ratio"]["event_changed_frac"][g]["variant"], 2, 100, True) + "%")
                    cells.append(fv(PR["ratio"]["joint_changed_frac"][g]["variant"], 2, 100, True) + "%")
                P(f"| {labels.get(nm, nm)} | {g} ({n}) | " + " | ".join(cells) + " |")
        P("")

    def rate_table(D, variants, title, p1=True):
        P(title)
        P("| 变体 | 序列(包数) | M_all | M_has | 精确一致率 | hit | valid | 事件被替换 | 关节被改派(事件数) | RA (mm) |")
        P("|---|---|---|---|---|---|---|---|---|---|")
        for v in variants:
            if v not in D["variants"]:
                continue
            for g in ("global", "local"):
                s = D["variants"][v].get(g)
                if not s:
                    continue
                r, su = s["rates"], s["sums"]
                P(f"| {v} | {g} ({s['n_steps']}) | {fnum(r['M_all'], 3, True)} | {fnum(r['M_has'], 3, True)} | {fnum(r['agree_exact'], 3, True)} | {fnum(r['hit_frac'], 3, True)} | "
                  f"{fnum(r['valid_frac'], 3, True)} | {int(su.get('n_changed', 0))}/{int(su['n_live'])} | {int(su.get('n_jchg', 0))} | {fnum(s.get('ra_mm'), 4)} |")
        P("")

    # ------------------------------------------------------------------ 2. A
    P("## 2. A 组：预测旧状态关联 vs 真值旧状态关联（隔离前态估计误差）\n")
    if Acl:
        P(cap(main, "闭环", ns_cl) + "\n")
        P("A 组只在闭环有意义（TF 的 prev 已是真值，A1 ≡ A0g、A2 ≡ base，见第 1 节的逐位核对）。对照：A0g = 自身 prev + `verts_geo` = 自身 prev（只改 z_prior 口径）；A1 − A0g 才是『只换分配』的净效应；A2 含 GT 泄漏（dXYZ 用 GT 表面点减自身顶点）。"
          "A0g2 是 A0g 的重复运行，用来给出『GPU 原子加非确定性』造成的闭环差值底。\n")
        abs_table(Acl, ("base", "pass", "A0g", "A0g2", "A1", "A2"),
                  (("ra_mm", "RA (mm)", 4), ("abs_mm", "绝对 MPJPE (mm)", 3), ("root_t_mm", "根平移误差 (mm)", 3), ("ra_mm_median", "RA 逐步中位 (mm)", 3), ("frac_steps_ra_gt_100mm", "RA>100mm 步占比", 4), ("ra_mm_n_capped", "RA 封顶步数(≥1000mm 或非有限)", 0)),
                  "### 2.1 各变体绝对值（闭环）\n")
        labels = {"A0g_vs_base": "A0g − base（z_prior 口径）", "A1_vs_A0g": "A1 − A0g（只换分配，净效应）", "A1_vs_base": "A1 − base", "A2_vs_base": "A2 − base（含泄漏）",
                  "A2_vs_A1": "A2 − A1（泄漏部分）", "A0g2_vs_A0g": "A0g2 − A0g（重复运行底）"}
        pair_table(Acl, list(labels), labels, (("ra_mm", "ΔRA (mm)", 3), ("abs_mm", "Δ绝对 MPJPE (mm)", 3), ("root_t_mm", "Δ根平移误差 (mm)", 3)),
                   "### 2.2 配对差值（闭环）\n")
        if Acl.get("divergence"):
            P(f"闭环发散统计（发散步 = RA / 绝对 MPJPE / 根平移误差任一 ≥ {CAP_MM:.0f} mm 或非有限；一个 run 内一旦发散通常不再恢复，因为 prev 被反馈）：\n")
            P("| 变体 | 序列 | 发散步 / 步数 | 发散占比 | 出现发散的 run / run 数 | 各出现发散的 run 内首个发散步（run 内步序，最多列 6 个） |\n|---|---|---|---|---|---|")
            for v in ("base", "A0g", "A0g2", "A1", "A2"):
                for g in ("global", "local"):
                    d_ = Acl["divergence"].get(v, {}).get(g)
                    if d_:
                        P(f"| {v} | {g} | {d_['n_diverged']} / {d_['n_steps']} | {100 * d_['frac_diverged']:.1f}% | {d_['n_runs_diverged']} / {d_['n_runs']} | {d_['first_diverged_step_in_run'][:6]} |")
            P("")
        if Acl.get("pairs_both_healthy"):
            lh = {k: v + "，仅两变体都未发散的步" for k, v in labels.items()}
            pair_table({"pairs": Acl["pairs_both_healthy"]}, list(lh), lh, (("ra_mm", "ΔRA (mm)", 3), ("abs_mm", "Δ绝对 MPJPE (mm)", 3), ("root_t_mm", "Δ根平移误差 (mm)", 3)),
                       "条件读数：只在『两个变体在该步都未发散』的步上取配对差（对存活有选择，**只作辅助**；括号里的包数 = 入选步数；该口径下 A1 / A2 的发散步被剔除，会低估它们的损害）：\n")
        rate_table(Acl, ("base", "A0g", "A1", "A2"), "### 2.3 闭环关联统计（按 P1 标签；确认 A1 的分配确为『真值旧状态』水平）\n")
        if p1:
            P("参照（同批抽样事件，事件池化）：")
            for sq in p1:
                row = f"- P1 `{sq}`：R0（窗首真值）M = {fnum(p1[sq]['R0']['M'], 3, True)}、精确一致率 = {fnum(p1[sq]['R0']['exact'], 3, True)}；R3（逐毫秒真值）M = {fnum(p1[sq]['R3']['M'], 3, True)}、精确一致率 = {fnum(p1[sq]['R3']['exact'], 3, True)}"
                if p2a:
                    pa = p2a["sequences"][sq]["pooled"]
                    row += f"；P2 闭环基线 M = {fnum(pa['M_all'], 3, True)}、精确一致率 = {fnum(pa['agree_exact'], 3, True)}"
                P(row)
            P("")
        assoc_table(Acl, ("A0g_vs_base", "A1_vs_A0g", "A1_vs_base", "A2_vs_base"), labels, "闭环关联统计差（事件池化比值之差，括号 = 步块 bootstrap 95% 区间；『事件被替换 / 关节被改派』是变体自身占 live 事件的比例）：\n")
        P("闭环轨迹上 prev 偏离真值的程度（逐步均值；变体改变轨迹，故不同变体的 prev 误差不同）与 route_stats 均值：\n")
        P("| 变体 | 序列 | prev 根平移误差 mm | prev 根旋转误差 ° | prev 手指误差 rad | valid | hit | resid 质量 | verts_direct | verts_prop |")
        P("|---|---|---|---|---|---|---|---|---|---|")
        for v in ("base", "A0g", "A1", "A2"):
            for sq in Acl["meta"]["seqs"]:
                t = Acl["trajectory_state_error"][v][sq]
                rsm = Acl["p2a_style_stats"][v][sq]["route_stats_mean"]
                P(f"| {v} | {sq} | {fnum(t['prev_err_t_mm_mean'], 2)} | {fnum(t['prev_err_rot_deg_mean'], 2)} | {fnum(t['prev_err_pose_rad_mean'], 3)} | {fnum(rsm['xyz_valid'], 3, True)} | {fnum(rsm['xyz_hit'], 3, True)} | "
                  f"{fnum(rsm['xyz_resid_mass'], 3, True)} | {fnum(rsm['xyz_verts_direct'], 1)} | {fnum(rsm['xyz_verts_prop'], 1)} |")
        P("")
        if Adx:
            dxyz_table(Adx["closed_loop"], f"### 2.5 输入分布诊断：关系编码器的 dXYZ 长度（闭环，检查点 `{main}`；训练时网络见到的是『关联在自身 prev 网格上』的 dXYZ，base 行即此分布）\n")
        nf = Acl.get("init_noise_floor_baseline_ra")
        if nf:
            P(f"闭环初值噪声底（{main}，基线，仅改初值噪声 rng 种子，其余相同，闭环，{sum(ns_cl.values())} 包）：\n")
            P("| 初值噪声 rng 种子 | RA overall | RA global | RA local |\n|---|---|---|---|")
            for sd in sorted(nf):
                P(f"| {sd} | {nf[sd]['overall']:.4f} | {nf[sd]['per_seq'].get('zgz_global', float('nan')):.4f} | {nf[sd]['per_seq'].get('zgz_local', float('nan')):.4f} |")
            P("")
    if Atf:
        P("### 2.4 TF 下的 A 对照：z_prior 口径改变本身（A0g − base）\n")
        P(cap(main, "TF", Atf["meta"]["n_steps"]) + "\n")
        abs_table(Atf, ("base", "A0g"), (("ra_mm", "RA (mm)", 4), ("e_rot_deg", "e_rot (°)", 3), ("e_t_mm", "e_t (mm)", 3), ("e_fing_rad", "e_fing (rad)", 4)), "TF 绝对值：\n")
        pair_table(Atf, ("A0g_vs_base",), {"A0g_vs_base": "A0g − base"}, (("ra_mm", "ΔRA (mm)", 3), ("e_rot_deg", "Δe_rot (°)", 3), ("e_t_mm", "Δe_t (mm)", 3), ("e_fing_rad", "Δe_fing (rad)", 4)), "")
        P("z_prior 口径的改变在数学上是恒等的：命中事件的 A 是命中点的重心坐标，Σ A·zj = (Σ A·V)·d / (d·d) = (t d)·d / (d·d) = t；近邻事件的现行 z 本就是 Σ A·zj。"
          "所以 A0g 与 base 的差只来自浮点舍入（重心权重的 clamp 与重归一化、float32 与 float64 累加），TF 下表现为 1e-6 量级。\n")

    # ------------------------------------------------------------------ 3. B
    P("## 3. B 组：窗首真值几何 vs 事件时刻真值几何（隔离包内时间错位，TF）\n")
    if Btf:
        P(cap(main, "TF", Btf["meta"]["n_steps"]) + "\n")
        P("B0 = base（关联用 pos51[prev_end] 网格）；B0g = A0g。B1 = 每个事件用其所在毫秒的真值网格（含 z，A2 方式）；B1g = 同一分配 + `verts_geo` = 窗首网格（A1 方式），它的对照零点是 B0g（= A0g）。\n")
        abs_table(Btf, ("base", "A0g", "B1", "B1g"), (("ra_mm", "RA (mm)", 4), ("e_rot_deg", "e_rot (°)", 3), ("e_t_mm", "e_t (mm)", 3), ("e_fing_rad", "e_fing (rad)", 4)), "### 3.1 绝对值（TF）\n")
        lb = {"B1_vs_base": "B1 − B0(base)", "B1g_vs_A0g": "B1g − B0g(A0g)", "B1g_vs_base": "B1g − base"}
        pair_table(Btf, list(lb), lb, (("ra_mm", "ΔRA (mm)", 3), ("e_rot_deg", "Δe_rot (°)", 3), ("e_t_mm", "Δe_t (mm)", 3), ("e_fing_rad", "Δe_fing (rad)", 4)), "### 3.2 配对差值（TF）\n")
        rate_table(Btf, ("base", "A0g", "B1", "B1g"), "### 3.3 关联统计（TF，按 P1 标签）\n")
        assoc_table(Btf, list(lb), lb, "关联统计差（TF）：\n")
        if p1:
            P("衔接 P1：" + "；".join(f"`{sq}` P1 标签层 M(R3) − M(R0) = {100 * (p1[sq]['R3']['M'] - p1[sq]['R0']['M']):+.2f} pp" for sq in p1) + "（P1 在窗内全部抽样事件上算，与本表 B1 − base 的 M_all 差同口径）。\n")
    if Adx:
        dxyz_table(Adx["tf"], f"### 3.4 输入分布诊断：dXYZ 长度（TF，检查点 `{main}`，全部 {sum(Btf['meta']['n_steps'].values()) if Btf else '?'} 包）\n")
    # ------------------------------------------------------------------ 4. C
    P("## 4. C 组：当前候选搜索 vs 全面片搜索（检查候选漏检）\n")
    for D, mode, nm in ((Ctf, "TF", "TF"), (Ccl, "闭环", "闭环")):
        if not D:
            continue
        ns = D["meta"]["n_steps"]
        P(f"### 4.{1 if mode == 'TF' else 2} {mode}\n")
        P(cap(main, mode, ns) + "\n")
        if mode == "TF":
            abs_table(D, ("base", "C1"), (("ra_mm", "RA (mm)", 4), ("e_rot_deg", "e_rot (°)", 3), ("e_t_mm", "e_t (mm)", 3), ("e_fing_rad", "e_fing (rad)", 4)), "绝对值：\n")
            pair_table(D, ("C1_vs_base",), {"C1_vs_base": "C1 − base"}, (("ra_mm", "ΔRA (mm)", 3), ("e_rot_deg", "Δe_rot (°)", 3), ("e_t_mm", "Δe_t (mm)", 3), ("e_fing_rad", "Δe_fing (rad)", 4)), "配对差值：\n")
        else:
            abs_table(D, ("base", "C1"), (("ra_mm", "RA (mm)", 4), ("abs_mm", "绝对 MPJPE (mm)", 3), ("root_t_mm", "根平移误差 (mm)", 3)), "绝对值：\n")
            pair_table(D, ("C1_vs_base",), {"C1_vs_base": "C1 − base"}, (("ra_mm", "ΔRA (mm)", 3), ("abs_mm", "Δ绝对 MPJPE (mm)", 3), ("root_t_mm", "Δ根平移误差 (mm)", 3)), "配对差值：\n")
        rate_table(D, ("base", "C1"), "关联统计：\n")
        assoc_table(D, ("C1_vs_base",), {"C1_vs_base": "C1 − base"}, "关联统计差：\n")
        hs = D.get("hit_set_change") or {}
        P("命中集合变化（占 live 抽样事件 / 手部事件的比例，事件池化，括号 = 步块 bootstrap 95% 区间；『原 hit 但面不同』= 候选集漏掉了更近的面，C1 改用最近面；"
          f"同 t（|Δt| ≤ {TIE_EPS * 1000:.3f} mm）的另一面视为同一命中点，保留原结果，单列）：\n")
        P("| 类别 | 序列 | 占 live 事件 | 占手部事件 |\n|---|---|---|---|")
        for key, name in (("c_closer", "原 hit，另有更近的面（面不同，t 更小）"), ("c_near2hit", "原 near → hit"), ("c_none2hit", "原 none → hit"),
                          ("c_lost", "原 hit → 非 hit（应为 0）"), ("c_tie", "同 t 另一面（保留）"), ("n_changed", "合计被改写")):
            hk = key + "_hand/n_hand" if key != "n_changed" else "n_changed_hand/n_hand"
            for g in ("overall", "global", "local"):
                a_ = hs.get(f"{key}/n_live", {}).get(g)
                b_ = hs.get(hk, {}).get(g)
                if a_ is None:
                    continue
                P(f"| {name} | {g} | {fv(a_, 3, 100, True)}% | {fv(b_, 3, 100, True) + '%' if b_ else 'n/a'} |")
        P("")
        dt = D.get("nearer_face_depth_gain_mm")
        if dt and dt.get("n"):
            P(f"『原 hit 但面不同』事件的深度改进（原 t − 最近面 t，mm；TF 全部包的 {dt['n']} 个事件）：均值 {dt['mean_mm']:.2f}，p10 / p50 / p90 / p99 = {dt['p10']:.2f} / {dt['p50']:.2f} / {dt['p90']:.2f} / {dt['p99']:.2f}，最大 {dt['max_mm']:.2f}；> 1 mm 的占 {100 * dt['frac_gt_1mm']:.1f}%，> 5 mm 的占 {100 * dt['frac_gt_5mm']:.1f}%。\n")

    # ------------------------------------------------------------------ 5. seeds
    P("## 5. 种子稳健性（另两个检查点，只列总 RA 的配对差值）\n")
    P("每行 = 一个检查点 × 一个对比；列 = ΔRA（mm）overall / global / local，格式『差值 [95% 区间]』。TF 与闭环的序列与包数同上（TF 2590 包，闭环 2590 步）。\n")
    P("| 检查点 | 模式 | 对比 | overall | global | local |\n|---|---|---|---|---|---|")
    rows = (("TF", "B", "pairs", "B1_vs_base", "B1 − base"), ("TF", "B", "pairs", "B1g_vs_A0g", "B1g − A0g"), ("TF", "A", "pairs", "A0g_vs_base", "A0g − base"), ("TF", "C", "pairs", "C1_vs_base", "C1 − base"),
            ("闭环", "A", "pairs", "A0g_vs_base", "A0g − base"), ("闭环", "A", "pairs", "A0g2_vs_A0g", "A0g2 − A0g（重复底）"),
            ("闭环", "A", "pairs", "A1_vs_A0g", "A1 − A0g（发散步封顶计入）"), ("闭环", "A", "pairs_both_healthy", "A1_vs_A0g", "A1 − A0g（仅两者都未发散的步）"),
            ("闭环", "A", "pairs", "A1_vs_base", "A1 − base（发散步封顶计入）"), ("闭环", "A", "pairs", "A2_vs_base", "A2 − base（发散步封顶计入）"),
            ("闭环", "A", "pairs_both_healthy", "A2_vs_base", "A2 − base（仅两者都未发散的步）"), ("闭环", "C", "pairs", "C1_vs_base", "C1 − base"))
    base_rows = []
    for tag in TAGS:
        for mode, grp, key, nm, lab in rows:
            D = J(f"{grp}_{'tf' if mode == 'TF' else 'closedloop'}_{tag}.json")
            if not D or not (D.get(key) or {}).get(nm):
                continue
            PR = D[key][nm]["step"]["ra_mm"]
            P(f"| {tag} | {mode} | {lab} | " + " | ".join(f"{fci(PR[g]['diff'], 3)}" + (f" (n={PR[g]['n_steps']})" if key == "pairs_both_healthy" else "") for g in ("overall", "global", "local")) + " |")
        D = J(f"A_closedloop_{tag}.json")
        T = J(f"B_tf_{tag}.json")
        base_rows.append((tag, T, D))
    P("")
    P("闭环发散步（RA / 绝对 MPJPE / 根平移误差任一 ≥ 1000 mm 或非有限）占比，global / local：\n")
    P("| 检查点 | A1 | A2 | base | A0g |\n|---|---|---|---|---|")
    for tag in TAGS:
        D = J(f"A_closedloop_{tag}.json")
        if D and D.get("divergence"):
            P(f"| {tag} | " + " | ".join(" / ".join(f"{D['divergence'][v][g]['n_diverged']}/{D['divergence'][v][g]['n_steps']}" for g in ("global", "local")) for v in ("A1", "A2", "base", "A0g")) + " |")
    P("")
    P("各检查点基线 RA（mm）：\n")
    P("| 检查点 | TF overall / global / local | 闭环 overall / global / local | 闭环 vs 选点结果逐位相同 |\n|---|---|---|---|")
    for tag, T, D in base_rows:
        tf = " / ".join(fnum(T["variants"]["base"][g]["ra_mm"], 4) for g in ("overall", "global", "local")) if T else "n/a"
        cl = " / ".join(fnum(D["variants"]["base"][g]["ra_mm"], 4) for g in ("overall", "global", "local")) if D else "n/a"
        bw = (D["baseline_repro"].get("selection_json") or {}).get("bitwise_equal_overall") if D else None
        P(f"| {tag} | {tf} | {cl} | {bw} |")
    P("")
    P("(显著性在每个检查点内部按步块 bootstrap 给出；三个检查点的方向是否一致是种子稳健性的读数，不构成跨种子的合并区间。)\n")

    # ------------------------------------------------------------------ 6/7. companions and limits
    P("## 6. 伴随变化\n")
    P("- **z_prior 口径**：走 `verts_geo` 路径时 z_prior = Σ w·zj / Σ w（顶点到射线最近点的深度），现行路径用命中深度 t（近邻事件本就是 Σ A·zj）。两者数学恒等（见 2.4：命中点是重心坐标加权，Σ A·zj = t），"
      "A0g − base 因此只反映浮点舍入：TF 下 1e-6 量级、预测逐位不同但数值相同；闭环下这种扰动经反馈放大成轨迹差（A0g − base 的 ΔRA 与 A0g2 − A0g、初值噪声种子之间的离散同量级）。A1 − A0g 已扣除它。")
    P("- **dXYZ / 边几何的来源改变**：A1 / B1g 的 dXYZ、z_prior、边几何仍在自身 prev 网格（verts_geo），只有分配来自别的网格；A2 / B1 的 z 与 X = z·d 来自别的网格，dXYZ 是『别的网格上的表面点 − 自身 prev 顶点』，因而在 prev 偏离真值时 dXYZ 幅度远超训练分布（见 A2 的发散读数）。")
    P("- **闭环轨迹随变体改变**：变体改变预测，进而改变后续各步的 prev、关联与 route_stats；2.3 的轨迹表给出各变体轨迹上的 prev 偏离。TF 下所有变体共享同一批包与同一 prev。")
    P("- **GPU 非确定性**：走 `verts_geo` 路径的 z_prior 用 float32 `index_add`（原子加），同一输入重复两次预测相差 ~1e-7；闭环 A0g2 − A0g 给出这种扰动经反馈放大后的 RA 差，作为闭环差值的『重复运行底』。")
    P("- **事件被替换比例很高但关节改派少**：B1 / A2 / A1 下几乎所有事件的 (vid, A, z) 数值都变（同一面片上的重心坐标 / 深度因网格不同略有差异），但 argmax 关节被改派的比例才是分配层面的改变；两列都列出。\n")
    P("## 7. 局限与偏离\n")
    P("- **冻结权重**：检查点是在『关联用自身 prev 网格』下训练的；A1 / A2 / B1 / B1g 是把另一网格上的关联喂给没见过这种输入分布的网络，读到的是**效应层**（推理时换几何会怎样），不是『训练时就用该几何会怎样』；A2 / B1 还带 GT 泄漏。因此差值为正 / 为零都不能直接外推为训练时的收益上限。")
    P("- **闭环的统计单位**：闭环是一条确定的轨迹（rng(0) 初值噪声，两条序列），步块 bootstrap 只刻画『步级抽样』不确定度，不含『换一次初值噪声 / 换一个种子后整条轨迹不同』的方差；闭环初值噪声底表（2.3 末）与 A0g2 重复运行以及三个检查点的方向一致性是补充读数，不是区间。")
    P("- **序列少**：val_core 只有 zgz_global（1386 包）与 zgz_local（1204 包）两条序列，且 local 序列内相邻包强相关；块长 20 是任务书规定值，可能低估长程自相关。")
    P("- **A1 的实现**：任务书允许两种实现；本探针用猴补 `associate_rays`（把 `verts` 换成 `model._fk(pos51[prev_end])`）+ 包装 `model.xyz.forward`（传入 `verts_geo`=自身 prev 网格），没有使用 `xyz_assoc_prev=\"target\"` 的开关：`track_sequence` 在该开关下会把 override 写成 pos51[end]（窗末真值），无法指定为 prev_end。"
      "核对：在 TF 包上，仓库 oracle 分支（override = pos51[prev_end]）与 A1 补丁的预测只差 float32 原子加噪声（见第 1 节），确认 oracle 分支做的就是『分配用 override 网格、几何用自身 prev 网格』。")
    P(f"- **C1 的并列处理**：暴力搜索的最近面与候选集最近面在同一深度（|Δt| ≤ {TIE_EPS * 1000:.3f} mm）时视为同一命中点，保留原结果（避免仅因面索引顺序改写重心权重）；该类事件单列。其余按任务书：命中分支覆盖 vid / A / z / hit / valid / q，near / none 不变，原非 hit 变 hit 者改为 hit。")
    P("- **B1 的 z**：B1 沿用 A2 方式，z 与 dXYZ 的 X 直接取自该毫秒网格的关联输出；B1g 把几何换回窗首网格。B 组不做闭环（任务书规定 TF）。")
    P("- **事件池化的 M / 一致率** 以 live 抽样事件为分母（≤ 2048/包），与 P2 口径一致；P1 的窗内全事件口径数值略有差别，只做方向对照。")
    P("- **运行经过（与最终脚本的差异）**：TF 部分（三个检查点）由脚本较早的版本产出，TF 的计算路径此后未改，之后只改了闭环的鲁棒性与报告；"
      "首次闭环运行在 A1 变体（xa_c1_s3407，global 序列状态非有限）触发『重算 RA 与 track_sequence 一致』断言而中止，已改为对 NaN 安全、每个闭环 pass 缓存到 `cache/`（`--resume`）、"
      "逐步误差封顶 1000 mm 后重跑，旧日志在 `old_logs/`；`A_dxyz` 首版把非有限步池化进均值，已改为只池化有限步并重跑（旧版在 `old_logs/`）。"
      "闭环的 base / pass 两个 pass 在重跑后与选点结果、P2 记录仍逐位一致（见第 1 节）。")
    P("- **本次未下『关联是否瓶颈』的总结论**；上表只回答『换几何，单步与闭环指标变多少、是否超出噪声』。")
    P("")
    (out_dir / "REPORT.md").write_text("\n".join(L))
    log("wrote", out_dir / "REPORT.md")


# --------------------------------------------------------------------------- main
def dxyz_summary(log_):
    """Pooled / per-step statistics of the dXYZ length (cm); calls with a non-finite statistic (a blown-up state) are counted, not pooled."""
    if not log_:
        return None
    ok = [x for x in log_ if all(math.isfinite(x[k]) for k in ("sum", "p50", "p90", "p99", "max"))]
    out = {"n_calls": len(log_), "n_calls_nonfinite": len(log_) - len(ok)}
    if not ok:
        return out
    n = sum(x["n"] for x in ok)
    p50 = np.array([x["p50"] for x in ok])
    out.update({"n_records": n, "mean_cm": sum(x["sum"] for x in ok) / n,
                "per_step_p50_cm_mean": float(p50.mean()), "per_step_p50_cm_median": float(np.median(p50)), "per_step_p50_cm_p90": float(np.percentile(p50, 90)),
                "per_step_p90_cm_mean": float(np.mean([x["p90"] for x in ok])), "per_step_p99_cm_mean": float(np.mean([x["p99"] for x in ok])),
                "frac_gt5cm": sum(x["n_gt5cm"] for x in ok) / n, "frac_gt10cm": sum(x["n_gt10cm"] for x in ok) / n, "max_cm": max(x["max"] for x in ok)})
    return out


def run_dxyz(a, env, out_dir):
    """Diagnostic for the frozen-weight caveat: the length of the relation encoder's dXYZ input (cm) under the variants.
    TF: base(pass-through) / B1 / B1g. Closed loop: base(pass-through) / A0g / A1 / A2. Nothing else is computed."""
    device, model = env["device"], env["model"]
    tag = ckpt_tag(a.run_dir)
    lbs_hard = model.xyz.lbs_hard.to(device)
    G = GeomPatch(model, device, lbs_hard)
    G.collect_dxyz = True
    out = {"meta": meta_of(a, env, {"what": "length (cm) of dXYZ = (z_prior d - V[vid]) / 0.01 m over (event, vertex) records; per-step quantiles averaged over steps"}),
           "tf": {}, "closed_loop": {}}
    model.xyz_keep_aux = False
    for v in ("pass", "B1", "B1g"):
        for seq, d in env["seqs"]:
            ctx, lab = env["cl"][seq]
            P2.set_context(model, ctx, device)
            betas = torch.tensor(ctx.aux["betas"], dtype=torch.float32, device=device).view(1, -1)
            G.set_seq(ctx, betas)
            G.dxyz_log = []
            for st, ev5, prev, gt, batch in P2.tf_packets(ctx, device):
                G.step_fn = lambda st=st: st
                G.variant = v
                G.install()
                try:
                    with torch.no_grad():
                        model.forward_packet(batch)
                finally:
                    G.uninstall()
            out["tf"].setdefault("base" if v == "pass" else v, {})[seq] = dxyz_summary(G.dxyz_log)
            log(f"dxyz TF {v} {seq}: {out['tf']['base' if v == 'pass' else v][seq]}")
    for v in ("pass", "A0g", "A1", "A2"):
        G.dxyz_by_seq = {}
        run_cl_variant(a, env, G, v)
        for seq, _ in env["seqs"]:
            out["closed_loop"].setdefault("base" if v == "pass" else v, {})[seq] = dxyz_summary(G.dxyz_by_seq.get((v, seq), []))
            log(f"dxyz CL {v} {seq}: {out['closed_loop']['base' if v == 'pass' else v][seq]}")
    P2.dump(out_dir / f"A_dxyz_{tag}.json", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="tf,cl")
    ap.add_argument("--run-dir", default=P2.RUN_DEFAULT)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--labels-dir", default=P2.LABELS_DEFAULT)
    ap.add_argument("--out-dir", default=OUT_DEFAULT)
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--sanity-every", type=int, default=5, help="TF: run the sanity-only variants on every n-th packet")
    ap.add_argument("--cl-variants", default="", help="comma list subset of the closed-loop variants (default all)")
    ap.add_argument("--resume", action="store_true", help="closed loop: reuse finished passes from out_dir/cache")
    ap.add_argument("--noise-seeds", default="", help="closed loop: extra baseline passes with these init-noise rng seeds (noise floor)")
    a = ap.parse_args()
    os.chdir(REPO)
    out_dir = Path(a.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = set(a.part.split(","))
    if parts & {"tf", "cl", "dxyz"}:
        env = build_env(a)
        dev = env["device"]
        info = f"device={dev} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} ckpt={env['ckpt']} cfg={env['cfg_path']}"
        log(info)
        gp = out_dir / "gpu_choice.txt"
        if not gp.exists() or GPU_INFO:
            txt = [f"{ckpt_tag(a.run_dir)} started {time.strftime('%F %T')}: {info}"]
            if GPU_INFO:
                txt.append("GPU memory used (MiB) at start, nvidia-smi PCI order: " + ", ".join(f"{i}:{u}" for i, u in GPU_INFO["used_mib_at_start"].items()))
                txt.append(f"authorised cards {GPU_INFO['authorised']}; chosen (least used among them): physical GPU {GPU_INFO['chosen_physical_gpu']}"
                           f" (preset CUDA_VISIBLE_DEVICES={GPU_INFO['preset_CUDA_VISIBLE_DEVICES']})")
            with open(gp, "a") as fh:
                fh.write("\n".join(txt) + "\n")
        with P2.limited_runs(a.max_steps):               # self-test only: sequences truncated for every reader alike
            if "tf" in parts:
                run_tf(a, env, out_dir)
            if "cl" in parts:
                run_cl(a, env, out_dir)
            if "dxyz" in parts:
                run_dxyz(a, env, out_dir)
    if "report" in parts:
        make_report(out_dir)


if __name__ == "__main__":
    main()
