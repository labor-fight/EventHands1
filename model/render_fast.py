#!/usr/bin/env python3
"""Opt-in, bit-identical, host-synchronisation-free render path of the dense render-and-compare tracker.

Why.  At batch 1 the dense tracker's step is launch bound, and `tools/dt/render_breakdown.py` found that 17 of
the CUDA calls of every forward are `cudaStreamSynchronize` (each one drains the GPU queue and leaves the GPU idle
until the host has issued the next kernels). The forward is launch bound: the GPU is busy for about 0.9-1.1 ms of it,
the rest is the host issuing ~230 tiny kernels (outputs/dt/reports/render_breakdown.md):

  * 15 inside `ManoLayer.forward`: `transform_chain[self.parents[i]]` indexes a Python list with a 0-d CUDA
    tensor, so Python calls `Tensor.__index__` -> `.item()` -> a device-to-host copy + stream sync, once per joint;
  * 2 inside `MNISTModel._render_chunk`: boolean-mask indexing (`pix[valid]`, `z_safe[valid]`) runs `nonzero`.

Everything here computes the *same floating point operations in the same order* as the code it replaces, so the
outputs are bit-for-bit those of `MNISTModel._render_chunk` (`tests/test_dt_render.py` compares `torch.equal`
against the live MAIN implementation on thousands of random states and on hand-made corner cases).  Nothing in this
file changes a default: `MNISTModel` only uses it after `enable_fast_render()`, which exists only if
outputs/dt/reports/render_fast_patch/render_fast_model.patch is applied (the 15 syncs of the stock `ManoLayer` are
removed for every user by render_fast_mano_layer.patch).

Contents
  FastMano            `ManoLayer.forward` op for op, with the parent table as Python ints (no sync); optionally
                      reuses the per-sequence terms (`ShapeTerms`: v_shaped, joint regression, rest-pose joint
                      offsets, constant pads) when the betas are the sequence context, and skips the unused joints.
  rasterize_nosync    sil / inv channels of `_render_chunk` with no boolean indexing and no data-dependent
                      branch: vertices that fail the validity test write into one extra "dump" slot
                      (buffer size B*h*w + 1) that is dropped afterwards.
  rasterize_masked    the original masked implementation, kept as the ablation / reference (same code as
                      `_render_chunk`, factored into the stage functions the breakdown tool times).
  RenderFast          callable with the signature of `MNISTModel._render_prev(prevpos, betas, camera_K)`.
  GraphedForward      (optional, opt-in) CUDA-graph replay of a whole batch-1 forward: either the replica
                      `dense_forward` built on the above, or any callable `fn(x, prev, betas, K)` such as the
                      model's own `forward`. The forward is first run eagerly under sync debug mode "error": a
                      failed capture would poison torch 2.1's caching allocator ("captures_underway == 0"), a
                      host sync is found there instead, cleanly. torch 2.1's profiler never returns after a
                      graph replay: do not profile a graphed forward.

Exactness notes (read before "optimising" anything here).
  * `tensor / python_scalar` is `tensor * (1 / scalar)` on CUDA but a true division on CPU; the `/ 5.0` below is
    therefore left exactly as written in `_render_chunk`.
  * `1.0 / t` is `t.reciprocal() * 1.0`; `* 1.0` is exact, so `t.reciprocal()` is bitwise the same.
  * `sil_flat` only ever holds 0.0 or 1.0, so `.clamp(0, 1)` is the identity and is skipped.
  * `torch.where(valid, pix, dump)` only replaces the index of vertices that the original dropped, and the dump slot
    is dropped, so the other slots see the same set of (index, value) pairs; `amin` and "write 1.0" are
    order-independent, hence the result does not depend on the scatter order.
  * With B == 1 the batch offset `arange(B) * h * w` is the integer 0 and is skipped.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from mano_layer import _transform_mat, batch_rodrigues
from pose_repr import decode_to_mano_inputs

# value a pixel's depth starts at in `_render_chunk` (anything the hand never reaches)
_FAR = 1.0e6


# ------------------------------------------------------------------------------------------------ MANO
class ShapeTerms:
    """Quantities of `ManoLayer.forward` that depend on the betas (and nothing else), computed once at B == 1
    with exactly the operations the layer runs every call, plus the constants it re-creates every call."""

    __slots__ = ("v_shaped", "J", "rel_joints", "rel_joints_col", "joints_homogen", "ident", "ones", "pad")

    def __init__(self, mano, betas: torch.Tensor):
        assert betas.shape[0] == 1, "ShapeTerms is the batch-1 (sequence context) cache"
        device, dtype = betas.device, betas.dtype
        v_shaped = mano.v_template + torch.einsum("bl,mkl->bmk", betas, mano.shapedirs)
        J = torch.einsum("bik,ji->bjk", v_shaped, mano.J_regressor)                     # (1, 16, 3)
        rel_joints = J.clone()
        rel_joints[:, 1:] -= J[:, mano.parents[1:]]
        self.v_shaped = v_shaped
        self.J = J
        self.rel_joints = rel_joints
        self.rel_joints_col = rel_joints.reshape(-1, 3, 1)                                 # (16, 3, 1)
        self.joints_homogen = torch.cat([J, torch.zeros((1, 16, 1), dtype=dtype, device=device)], dim=2)
        self.ident = torch.eye(3, dtype=dtype, device=device)
        self.ones = torch.ones((1, v_shaped.shape[1], 1), dtype=dtype, device=device)
        pad = torch.zeros((16, 1, 4), dtype=dtype, device=device)
        pad[:, :, -1] = 1.0
        self.pad = pad


class FastMano:
    """`ManoLayer.forward` with the same operations in the same order, minus the 15 hidden host syncs.

    `forward(..., shape=ShapeTerms)` additionally reuses the per-sequence terms; `need_joints=False` skips the
    21-joint output (the renderer discards it; the vertices do not depend on it).
    """

    def __init__(self, mano):
        self.mano = mano
        self.parents: List[int] = [int(p) for p in mano.parents.tolist()]   # one sync, at construction

    def shape_terms(self, betas: torch.Tensor) -> ShapeTerms:
        return ShapeTerms(self.mano, betas)

    def forward(self, betas, global_orient, hand_pose, transl=None, shape: Optional[ShapeTerms] = None,
                need_joints: bool = True):
        mano = self.mano
        batch = betas.shape[0]
        device, dtype = betas.device, betas.dtype
        assert shape is None or batch == 1, "ShapeTerms are per-sequence (B == 1) terms"

        pose_hand = hand_pose
        if mano.add_mean:
            pose_hand = pose_hand + mano.hands_mean.view(1, -1)
        full_pose = torch.cat([global_orient, pose_hand], dim=1)  # (B, 48)

        if shape is None:
            v_shaped = mano.v_template + torch.einsum("bl,mkl->bmk", betas, mano.shapedirs)
            J = torch.einsum("bik,ji->bjk", v_shaped, mano.J_regressor)  # (B, 16, 3)
        else:
            v_shaped, J = shape.v_shaped, shape.J

        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(batch, 16, 3, 3)
        ident = torch.eye(3, dtype=dtype, device=device) if shape is None else shape.ident
        pose_feature = (rot_mats[:, 1:] - ident).view(batch, -1)  # (B, 135)
        v_posed = v_shaped + torch.einsum("bl,mkl->bmk", pose_feature, mano.posedirs)

        if shape is None:
            rel_joints = J.clone()
            rel_joints[:, 1:] -= J[:, mano.parents[1:]]
            transforms_mat = _transform_mat(
                rot_mats.view(-1, 3, 3), rel_joints.reshape(-1, 3, 1)
            ).view(batch, 16, 4, 4)
        else:
            transforms_mat = torch.cat(
                [torch.cat([rot_mats.view(-1, 3, 3), shape.rel_joints_col], dim=2), shape.pad], dim=1
            ).view(batch, 16, 4, 4)

        parents = self.parents
        transform_chain = [transforms_mat[:, 0]]
        for i in range(1, mano.num_joints):
            transform_chain.append(torch.matmul(transform_chain[parents[i]], transforms_mat[:, i]))
        transforms = torch.stack(transform_chain, dim=1)  # (B, 16, 4, 4)

        posed_joints = transforms[:, :, :3, 3]  # (B, 16, 3)

        if shape is None:
            joints_homogen = torch.cat(
                [J, torch.zeros((batch, 16, 1), dtype=dtype, device=device)], dim=2
            )
        else:
            joints_homogen = shape.joints_homogen
        rest_mats = torch.matmul(transforms, joints_homogen.unsqueeze(-1))
        joint_rel = transforms.clone()
        joint_rel[:, :, :3, 3] = transforms[:, :, :3, 3] - rest_mats[:, :, :3, 0]

        T = torch.einsum("bnj,bjkl->bnkl", mano.weights.expand(batch, -1, -1), joint_rel)
        if shape is None:
            ones = torch.ones((batch, v_posed.shape[1], 1), dtype=dtype, device=device)
        else:
            ones = shape.ones
        rest_shape_h = torch.cat([v_posed, ones], dim=2)
        verts = torch.matmul(T, rest_shape_h.unsqueeze(-1)).squeeze(-1)[:, :, :3]

        joints21 = None
        if need_joints:
            tips = verts[:, mano.fingertip_vertex_ids]  # (B, 5, 3)
            joints16_tips = torch.cat([posed_joints, tips], dim=1)  # (B, 21, 3)
            joints21 = joints16_tips[:, mano.mano16_tips_to_openpose21]

        if transl is not None:
            verts = verts + transl.unsqueeze(1)
            if need_joints:
                joints21 = joints21 + transl.unsqueeze(1)

        return verts, joints21


# ------------------------------------------------------------------------------------------------ raster
def project_pixels(verts, fx, fy, cx, cy, h: int, w: int):
    """Pinhole projection and the validity test of `_render_chunk`; fx.. are (B, 1).
    Returns (z_safe, ui, vi, valid), all (B, n_v)."""
    x, y, z = verts.unbind(-1)
    z_safe = z.clamp(min=1e-6)
    u = fx * x / z_safe + cx
    v = fy * y / z_safe + cy
    ui = u.round().long()
    vi = v.round().long()
    valid = (z > 1e-6) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    return z_safe, ui, vi, valid


def flat_pixel_index(ui, vi, h: int, w: int, skip_batch_offset: bool = False):
    """Flat index `b * h * w + v * w + u` (B, n_v); with B == 1 the offset is the integer 0 and can be skipped."""
    B, n_v = ui.shape
    if skip_batch_offset and B == 1:
        return vi * w + ui
    b_idx = torch.arange(B, device=ui.device)[:, None].expand(B, n_v)
    return b_idx * (h * w) + vi * w + ui


def masked_select_pixels(pix_all, z_safe, valid):
    """The original boolean-mask indexing: two `nonzero` calls, i.e. two host syncs."""
    return pix_all[valid], z_safe[valid]


def masked_scatter(pix, z_valid, B: int, h: int, w: int, device):
    """The original buffers: z-buffer by `scatter_reduce_(amin)`, silhouette by `scatter_`, guarded by a host
    `if pix.numel() > 0`."""
    depth_flat = torch.full((B * h * w,), _FAR, device=device, dtype=torch.float32)
    sil_flat = torch.zeros((B * h * w,), device=device, dtype=torch.float32)
    if pix.numel() > 0:
        depth_flat.scatter_reduce_(0, pix, z_valid, reduce="amin", include_self=True)
        sil_flat.scatter_(0, pix, torch.ones_like(z_valid))
    return depth_flat, sil_flat


def masked_faces(depth_flat, sil_flat, B: int, h: int, w: int, channels):
    """The original face assembly (sil / inv only)."""
    sil = sil_flat.view(B, h, w).clamp(0, 1)
    faces = {}
    if "sil" in channels:
        faces["sil"] = sil.unsqueeze(-1)
    if "inv" in channels:
        faces["inv"] = (
            ((1.0 / depth_flat.view(B, h, w)).clamp(0.0, 5.0) / 5.0) * sil
        ).unsqueeze(-1)
    return torch.cat([faces[c] for c in channels], dim=-1)


def rasterize_masked(verts, fx, fy, cx, cy, h: int, w: int, channels):
    """`_render_chunk` after the FK, verbatim (sil / inv): the reference / ablation implementation."""
    z_safe, ui, vi, valid = project_pixels(verts, fx, fy, cx, cy, h, w)
    pix_all = flat_pixel_index(ui, vi, h, w)
    pix, z_valid = masked_select_pixels(pix_all, z_safe, valid)
    depth_flat, sil_flat = masked_scatter(pix, z_valid, verts.shape[0], h, w, verts.device)
    return masked_faces(depth_flat, sil_flat, verts.shape[0], h, w, channels)


def dump_pixels(pix_all, valid, n_pix: int):
    """Invalid vertices -> the extra slot `n_pix` (no boolean indexing)."""
    return torch.where(valid, pix_all, n_pix)


def scatter_with_dump(pix, z_safe, n_pix: int, device):
    """z-buffer and silhouette into buffers of n_pix + 1 slots; returns the first n_pix of each."""
    depth_flat = torch.full((n_pix + 1,), _FAR, device=device, dtype=torch.float32)
    sil_flat = torch.zeros((n_pix + 1,), device=device, dtype=torch.float32)
    pix = pix.reshape(-1)
    depth_flat.scatter_reduce_(0, pix, z_safe.reshape(-1), reduce="amin", include_self=True)
    sil_flat.scatter_(0, pix, 1.0)
    return depth_flat[:n_pix], sil_flat[:n_pix]


def nosync_faces(depth_flat, sil_flat, B: int, h: int, w: int, channels):
    sil = sil_flat.view(B, h, w)                       # holds 0.0 / 1.0 only: the original clamp(0, 1) is the identity
    faces = {}
    if "sil" in channels:
        faces["sil"] = sil
    if "inv" in channels:
        faces["inv"] = ((depth_flat.view(B, h, w).reciprocal().clamp(0.0, 5.0) / 5.0) * sil)
    return torch.stack([faces[c] for c in channels], dim=-1)


def rasterize_nosync(verts, fx, fy, cx, cy, h: int, w: int, channels):
    """sil / inv faces of `_render_chunk`, bit for bit, with no host synchronisation."""
    B = verts.shape[0]
    n_pix = B * h * w
    z_safe, ui, vi, valid = project_pixels(verts, fx, fy, cx, cy, h, w)
    pix = dump_pixels(flat_pixel_index(ui, vi, h, w, skip_batch_offset=True), valid, n_pix)
    depth_flat, sil_flat = scatter_with_dump(pix, z_safe, n_pix, verts.device)
    return nosync_faces(depth_flat, sil_flat, B, h, w, channels)


# ------------------------------------------------------------------------------------------------ module
def _key(t: torch.Tensor):
    """Identity of a tensor's *content*: storage address, in-place version counter, shape, device."""
    try:
        version = t._version
    except RuntimeError:                      # inference tensors carry no version counter: never match
        return None
    return (t.data_ptr(), version, tuple(t.shape), t.device)


class _Context:
    # betas_ref / k_ref keep the tensors alive: a freed storage could be re-allocated at the same address with the same
    # (zero) version counter and would then match a stale key
    __slots__ = ("betas_key", "k_key", "shape", "intr", "betas_ref", "k_ref")


class RenderFast:
    """`MNISTModel._render_prev` without host syncs, bit-identical for sil / inv (any other face set falls back to
    the model's own `_render_chunk`).

    raster  boolean-mask free rasteriser (`rasterize_nosync`) instead of the masked one
    fk      `FastMano` (no hidden syncs; vertices only) instead of `model._fk`
    cache   with B == 1 and the betas / K that `set_context` was given: reuse `ShapeTerms` and the render
            intrinsics instead of recomputing them every step. Any other call takes the uncached path.
    """

    def __init__(self, model, raster: bool = True, fk: bool = True, cache: bool = False):
        if cache and not fk:
            raise ValueError("the sequence cache lives in FastMano; use fk=True")
        self.model = model
        self.raster, self.fk, self.cache = bool(raster), bool(fk), bool(cache)
        self.fmano = FastMano(model.mano) if fk else None
        self._ctx: Optional[_Context] = None
        #: B == 1 calls that reused the per-sequence terms / that had a context but did not match it (diagnostics)
        self.cache_hits = 0
        self.cache_misses = 0

    # -- per-sequence context ------------------------------------------------------------------------
    def set_context(self, betas: Optional[torch.Tensor], camera_K: Optional[torch.Tensor]) -> None:
        """Call whenever `MNISTModel.set_hand_context` does, with the tensors it stored (`_ctx_betas` (1, 10),
        `_ctx_K` (1, 3, 3)). The cache only engages for calls whose betas / K *are* those tensors (same storage,
        same in-place version), so a later in-place edit or a different tensor silently takes the uncached path."""
        self._ctx = None
        if not self.cache or betas is None or camera_K is None:
            return
        ctx = _Context()
        ctx.betas_key, ctx.k_key = _key(betas), _key(camera_K)
        ctx.betas_ref, ctx.k_ref = betas, camera_K
        ctx.shape = self.fmano.shape_terms(betas)
        ctx.intr = tuple(t[:, None] for t in self.model._intrinsics(camera_K))
        self._ctx = ctx

    # -- the render ----------------------------------------------------------------------------------
    def __call__(self, prevpos: torch.Tensor, betas: torch.Tensor, camera_K: torch.Tensor) -> torch.Tensor:
        m = self.model
        B = prevpos.shape[0]
        chunk = max(int(m.render_chunk), 1)
        out = [
            self.render_chunk(prevpos[i0: i0 + chunk], betas[i0: i0 + chunk], camera_K[i0: i0 + chunk])
            for i0 in range(0, B, chunk)
        ]
        return out[0] if len(out) == 1 else torch.cat(out, dim=0)

    def render_chunk(self, prevpos, betas, camera_K):
        m = self.model
        if "semsil" in m.render_channels:
            return m._render_chunk(prevpos, betas, camera_K)        # out of scope here: unchanged path
        B = prevpos.shape[0]
        shape = intr = None
        ctx = self._ctx
        if ctx is not None and B == 1:
            if ctx.betas_key is not None and _key(betas) == ctx.betas_key:
                shape = ctx.shape
            if ctx.k_key is not None and _key(camera_K) == ctx.k_key:
                intr = ctx.intr
            if shape is None:
                self.cache_misses += 1
            else:
                self.cache_hits += 1
        if self.fk:
            dec = decode_to_mano_inputs(prevpos, m.pose_repr, m.mano.hands_components, m.mano.hands_mean)
            verts, _ = self.fmano.forward(betas, dec["global_orient"], dec["local_full_aa"], dec["transl"],
                                          shape=shape, need_joints=False)
        else:
            verts, _ = m._fk(prevpos, betas)
        if intr is None:
            intr = tuple(t[:, None] for t in m._intrinsics(camera_K))
        raster = rasterize_nosync if self.raster else rasterize_masked
        return raster(verts, *intr, m.render_h, m.render_w, m.render_channels)


# ------------------------------------------------------------------------------------------------ forward
def dense_forward(model, x, prevpos, betas=None, camera_K=None, render=None):
    """`MNISTModel.forward` of the dense render-and-compare arm (PREV_RENDER, no active head, no raw-event
    encoder), op for op, with the render step pluggable (`render(prev, betas, K)`; default the model's own).
    `tests/test_dt_render.py` pins it bitwise to the live `forward`, so a drift in the model shows up there."""
    if getattr(model, "encoder_name", "") or getattr(model, "active_head", False) or not model.prev_render:
        raise NotImplementedError("dense_forward covers the dense PREV_RENDER arm only")
    lnes = x
    B = lnes.shape[0]
    betas_f, k_f = model._resolve_betas_K(prevpos, betas, camera_K)
    with torch.no_grad():
        rend = (render or model._render_prev)(prevpos.float(), betas_f, k_f)
    rend = rend.to(dtype=lnes.dtype, device=lnes.device)
    x = torch.cat([lnes, rend], dim=-1)
    x = x.permute(0, 3, 1, 2).contiguous()
    out = model.rn(model.conv1(x))
    if model.prevpos_embed:
        out = out + model.prev_mlp(prevpos.to(out.dtype))
    if model.predict_delta:
        delta = out
        if model.zero_event_gate:
            empty = lnes.reshape(B, -1).abs().sum(dim=1, keepdim=True) <= 0
            delta = torch.where(empty, torch.zeros_like(delta), delta)
        out = delta + prevpos.to(out.dtype)
    return out


class GraphedForward:
    """Opt-in: CUDA-graph replay of a batch-1 forward whose every op is host-sync-free.

    At batch 1 the eager forward is bound by the host issuing ~230 small kernels (outputs/dt/reports/render_breakdown.md);
    the replay launches the identical kernels, in the identical order, with the identical arguments, so the output is
    bit-identical and a step costs the GPU's own time. Shapes are static: batch 1, the LNES shape and the 51-D state given
    at `capture`. betas / K are graph *inputs* (static buffers; `set_context` refreshes them), so a new sequence needs no
    re-capture. Inference only (no autograd).

    `fn(x, prev, betas, K)` is what is captured. Default: `dense_forward` with a sync-free `RenderFast` (a replica of the
    dense forward, for standalone use). The model patch passes the model's own `forward` (its `_render_prev` dispatching
    to a `RenderFast`), so whatever the forward does -- later changes included -- is what is replayed; a forward with a
    host sync cannot be captured and raises at `capture`.

        gf = GraphedForward(model); gf.capture(x0, prev0, betas, K)
        pred = gf(x, prev)                     # == model(x, prev), bit for bit
    """

    def __init__(self, model, render: Optional[RenderFast] = None, clone_output: bool = True, fn=None):
        self.model = model
        self.clone_output = bool(clone_output)
        self.capturing = False
        if fn is None:
            self.render = render if render is not None else RenderFast(model, raster=True, fk=True, cache=False)
            if self.render.cache:
                raise ValueError("a cached RenderFast would freeze its per-sequence terms into the graph")
            if not (self.render.raster and self.render.fk):
                raise ValueError("the graph needs the host-sync-free render (raster=True, fk=True)")
            fn = lambda x, p, b, k: dense_forward(model, x, p, b, k, render=self.render)   # noqa: E731
        else:
            self.render = render
        self.fn = fn
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self._x = self._prev = self._betas = self._K = self._out = None

    @property
    def ready(self) -> bool:
        return self.graph is not None

    def matches(self, x, prev) -> bool:
        """True if (x, prev) can be fed to the graph (before capture: any batch-1 input of the right rank)."""
        if x.dim() != 4 or x.shape[0] != 1 or tuple(prev.shape) != (1, self.model.output_dim):
            return False
        if self.graph is None:
            return True
        return (x.shape == self._x.shape and x.dtype == self._x.dtype and x.device == self._x.device
                and prev.device == self._prev.device)

    def _run(self):
        with torch.no_grad():
            return self.fn(self._x, self._prev, self._betas, self._K)

    def capture(self, x, prev, betas, camera_K, warmup: int = 3) -> None:
        """Record the graph; `x` (1, H, W, C) LNES, `prev` (1, 51), `betas` (1, 10), `camera_K` (1, 3, 3)."""
        assert x.shape[0] == 1 and prev.shape[0] == 1, "GraphedForward is a batch-1 graph"
        dev = x.device
        self._x = x.detach().clone()
        self._prev = prev.detach().float().clone()
        self._betas = betas.detach().to(device=dev, dtype=torch.float32).reshape(1, -1).clone()
        self._K = camera_K.detach().to(device=dev, dtype=torch.float32).reshape(1, 3, 3).clone()
        self.capturing = True
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                self._run()                                  # first call plain: lazy library initialisation
                # A capture that fails leaves torch 2.1's caching allocator unusable for the next one ("captures_underway
                # == 0 INTERNAL ASSERT FAILED"), so find out here, eagerly and cleanly: a host synchronisation, which is
                # what makes a capture fail, raises under sync debug mode "error".
                mode = torch.cuda.get_sync_debug_mode()
                torch.cuda.set_sync_debug_mode("error")
                try:
                    for _ in range(max(int(warmup), 1)):
                        self._run()
                finally:
                    torch.cuda.set_sync_debug_mode(mode)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._out = self._run()
            torch.cuda.synchronize()
            self.graph = graph
        except Exception:
            self.graph = None
            torch.cuda.synchronize()
            raise
        finally:
            self.capturing = False

    def set_context(self, betas, camera_K) -> None:
        """New sequence: copy its betas (1, 10) and K (1, 3, 3) into the graph's input buffers (before the capture
        nothing is buffered: the context current at `capture` is used)."""
        if self.graph is None:
            return
        self._betas.copy_(betas.reshape(1, -1))
        self._K.copy_(camera_K.reshape(1, 3, 3))

    def __call__(self, x, prev):
        self._x.copy_(x)
        self._prev.copy_(prev)
        self.graph.replay()
        return self._out.clone() if self.clone_output else self._out
