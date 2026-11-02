r"""S45-SM latency path (docs/EXPERIMENT_RECORDS.md#s45sm-prereg-r2-20261031): the same routes as semkine/s45sm.py, bit for
bit, with the router's redundant work removed. Opt-in; `install` / `uninstall` as semkine/s45sm.py.

  tokens   S45's forward already computes `event_tokens` for the packet batch; `semkine.encoder.event_tokens` is
           wrapped with a one-entry memo (same tensors, same version counters), so the router reuses them.
  nodes    the router takes S45's stride-sampled nodes (`src`, `mask`) from the hook's arguments instead of
           re-sampling (`stride_sample` is deterministic: the same nodes).
  image    the event image without a sort: per (packet, polarity, pixel) the time of the event with the largest row
           index (scatter-amax over row indices, i.e. "the last event wins" exactly as `s45rt_router.event_image`),
           and log1p of the count (an integer-valued float sum, exact).
  faces    for small batches (deployment: one packet) the nearest-face lookup runs on every node at once, without
           `nonzero` / chunk loops; unselected nodes take face 0 as before. Large batches use the original lookup.
  trunk    one packet on a GPU without autocast or gradients (deployment): the frozen trunk is replayed from a CUDA
           graph captured on the fixed 4 x 180 x 240 input (the same kernels, one launch).
  pfa      the empty association and the PFA records are written densely (no `nonzero`, no host sync).
"""
from __future__ import annotations

import torch

from . import encoder as ENC
from . import s45sm as SM
from . import xag as XAG
from . import xyz_mesh as XM
from .encoder import EV_BATCH, EV_P, EV_T, EV_X, EV_Y, _packet_dt
from .s45tr_t0 import project_bary

DENSE_MAX_NODES = 65536
_NATIVE_TOKENS = ENC.event_tokens
_MEMO = {"key": None, "value": None}
_STATE = {"on": False}


def _key(events, ptr, delta_t_s, height, width):
    return (id(events), events.data_ptr(), tuple(events.shape), events._version, id(ptr), ptr.data_ptr(),
            ptr._version, id(delta_t_s), delta_t_s.data_ptr(), delta_t_s._version, int(height), int(width))


def event_tokens(events, ptr, delta_t_s, height, width, eps=ENC.EPS_DT):
    if eps != ENC.EPS_DT:
        return _NATIVE_TOKENS(events, ptr, delta_t_s, height, width, eps)
    key = _key(events, ptr, delta_t_s, height, width)
    if _MEMO["key"] != key:
        _MEMO["value"] = _NATIVE_TOKENS(events, ptr, delta_t_s, height, width)
        _MEMO["key"] = key
    return _MEMO["value"]


def event_image(events, ptr, delta_t_s, height=SM.HEIGHT, width=SM.WIDTH):
    """Equal to `semkine.s45rt_router.event_image`, element for element, without sorting."""
    B = int(ptr.numel() - 1)
    plane = height * width
    n = events.shape[0]
    if n == 0:
        return events.new_zeros((B, 4, height, width), dtype=torch.float32)
    dt = _packet_dt(delta_t_s, events).float()
    b = events[:, EV_BATCH].long()
    x = events[:, EV_X].long().clamp(0, width - 1)
    y = events[:, EV_Y].long().clamp(0, height - 1)
    p = (events[:, EV_P] > 0.5).long()
    t = (events[:, EV_T].float() / dt).clamp(0.0, 1.0)
    pix = (b * 2 + p) * plane + y * width + x                                   # (packet, polarity, pixel)
    idx = torch.arange(n, device=events.device)
    last = torch.full((B * 2 * plane,), -1, dtype=torch.long, device=events.device)
    last.scatter_reduce_(0, pix, idx, reduce="amax", include_self=True)
    time = torch.where(last >= 0, t[last.clamp_min(0)], torch.zeros((), device=events.device))
    count = torch.zeros(B * 2 * plane, dtype=torch.float32, device=events.device).index_add_(
        0, pix, torch.ones(n, dtype=torch.float32, device=events.device))
    return torch.cat([time.view(B, 2, height, width), torch.log1p(count).view(B, 2, height, width)], 1)


_GRAPHS = {}


def trunk(net, image):
    """`net.trunk(image)`; one packet on CUDA without autocast / grad replays a captured graph of the same kernels."""
    if image.shape[0] != 1 or not image.is_cuda or torch.is_grad_enabled() or torch.is_autocast_enabled():
        return net.trunk(image)
    key = (id(net), image.device, image.dtype)
    g = _GRAPHS.get(key)
    if g is None:
        static_in = image.clone()
        stream = torch.cuda.Stream(image.device)
        stream.wait_stream(torch.cuda.current_stream(image.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                net.trunk(static_in)                                                # warm up the kernels' workspaces
        torch.cuda.current_stream(image.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = net.trunk(static_in)
        g = _GRAPHS[key] = (graph, static_in, static_out)
    graph, static_in, static_out = g
    static_in.copy_(image)
    graph.replay()
    return static_out.clone()


def router_forward(net, events, ptr, delta_t_s, src, mask):
    """`net.forward` with S45's nodes and the memoized tokens (both router kinds share this layout)."""
    B = int(ptr.numel() - 1)
    tok = event_tokens(events, ptr, delta_t_s, SM.HEIGHT, SM.WIDTH)
    fmap = trunk(net, event_image(events, ptr, delta_t_s).to(net.stem[0].weight.dtype))
    if tok.shape[0] == 0:
        tok = events.new_zeros((1, 7))
        ev = events.new_zeros((1, events.shape[1]))
        src = torch.zeros_like(src)
    else:
        ev = events
    flat = src.reshape(-1)
    node_ev = ev[flat]
    h4, w4 = fmap.shape[-2:]
    yy = (node_ev[:, EV_Y].long() * h4 // SM.HEIGHT).clamp(0, h4 - 1)
    xx = (node_ev[:, EV_X].long() * w4 // SM.WIDTH).clamp(0, w4 - 1)
    bb = torch.arange(B, device=events.device).repeat_interleave(src.shape[1])
    feat = fmap[bb, :, yy, xx]
    z = net.head(torch.cat([feat.to(tok.dtype), tok[flat]], -1)).view(B, src.shape[1], -1)
    z = z * mask.unsqueeze(-1).to(z.dtype)
    out = {"z": z, "coord": net.coord(z), "onhand": net.onhand(z).squeeze(-1), "src": src, "mask": mask}
    out["patch" if hasattr(net, "patch") else "part"] = (net.patch if hasattr(net, "patch") else net.part)(z)
    return out


def route(net, router, out, select):
    """`net.route` (both kinds) with a dense lookup for small batches; identical faces and barycentrics."""
    if select.numel() > DENSE_MAX_NODES:
        return net.route(out, select)[:2]
    if router == "t0p":
        label = torch.softmax(out["patch"].float(), -1).max(-1).indices
        table, valid, cent = net.patch_faces, net.patch_faces_valid, net.tmpl[net.faces].mean(1)
    else:
        label = out["part"].float().argmax(-1)
        table, valid, cent = net.part_faces, net.part_faces_valid, net.centroid
    point = out["coord"].float() * 0.01
    cand, ok = table[label], valid[label]                                       # (B, N, M)
    d = (cent[cand] - point.unsqueeze(-2)).norm(dim=-1).masked_fill(~ok, float("inf"))
    face = cand.gather(-1, d.argmin(-1, keepdim=True)).squeeze(-1)
    face = torch.where(select, face, torch.zeros_like(face))
    return face, project_bary(point, net.tmpl[net.faces[face]])


@torch.no_grad()
def predicted_sources(batch, src, mask, *, require_depth=False):
    if require_depth:
        raise RuntimeError("S45-SM routes faces; full_oracle depth is not defined for it")
    st = SM._STATE
    net = SM._net(src.device)
    out = router_forward(net, batch.events, batch.ptr, batch.delta_t_s, src, mask)
    routed = mask & (torch.sigmoid(out["onhand"].float()) >= st["onhand_tau"])
    face, bary = route(net, st["router"], out, routed)
    zero = torch.zeros_like(face)
    return {"xag_source_index": zero, "xag_source_ms": zero, "xag_part": net.face_part[face], "xag_face": face,
            "xag_bary": bary.to(torch.float32), "xag_valid": routed}


def empty_association(d, mask, verts, faces, vert_faces, band, front_tol, z_tol, k=16, chunk=XM.ASSOC_CHUNK,
                      mode="hit_first", normals=None, gap=0.01):
    """`s45sm.empty_association` (S45's all-zero record set) without the native function's `nonzero`."""
    if mode != "hit_first":
        raise ValueError("PFA replaces the hit_first association only")
    B, N, _ = d.shape
    dev = d.device
    return {"vid": torch.zeros(B, N, XM.R_MAX, dtype=torch.long, device=dev),
            "A": torch.zeros(B, N, XM.R_MAX, dtype=torch.float32, device=dev),
            "q": torch.zeros(B, N, dtype=torch.float32, device=dev), "z": torch.zeros(B, N, dtype=torch.float32, device=dev),
            "valid": torch.zeros(B, N, dtype=torch.bool, device=dev), "hit": torch.zeros(B, N, dtype=torch.bool, device=dev)}


@torch.no_grad()
def pfa_association(mode, assoc, d, mask, verts, faces, vert_faces, lbs_weights, band, front_tol, z_tol, *,
                    source_part, source_face, source_bary, source_valid, source_depth=None, k=16, chunk=4096):
    """`s45sm.pfa_association`, written densely over every node (no `nonzero`); identical records."""
    if mode != "face_route":
        raise ValueError("PFA is installed under MODEL.XAG_ASSOC: face_route")
    if assoc["vid"].shape != (*mask.shape, XM.R_MAX) or assoc["z"].shape != mask.shape:
        raise ValueError("PFA expects the four-record hit_first association shapes")
    known = source_valid & (source_face >= 0) & (source_face < len(faces))
    known = known & torch.isfinite(source_bary).all(-1) & (source_bary >= 0).all(-1) & (source_bary.sum(-1) > 0)
    eligible = known & mask
    fv = faces.long()[source_face.clamp(0, len(faces) - 1)]                              # (B, N, 3)
    bary = source_bary.float()
    bary = bary / bary.sum(-1, keepdim=True).clamp_min(1e-30)
    B = mask.shape[0]
    point = (bary.unsqueeze(-1) * verts.float()[torch.arange(B, device=mask.device).view(B, 1, 1), fv]).sum(-2)
    e3, e1 = eligible.unsqueeze(-1), eligible
    out = {"vid": torch.where(e3, torch.cat([fv, fv[..., :1]], -1), torch.zeros_like(assoc["vid"])),
           "A": torch.where(e3, torch.cat([bary, torch.zeros_like(bary[..., :1])], -1).to(assoc["A"].dtype),
                            torch.zeros_like(assoc["A"])),
           "q": e1.to(assoc["q"].dtype),
           "z": torch.where(e1, point[..., 2].clamp_min(SM.Z_MIN).to(assoc["z"].dtype), torch.zeros_like(assoc["z"])),
           "valid": e1.clone(), "hit": e1.clone()}
    return out, {"xag_known": known & mask, "xag_eligible": eligible, "xag_replaced": eligible,
                 "xag_fallback": mask & ~eligible, "xag_recalled": eligible, "xag_none": mask & ~eligible}


def install(router, checkpoint, assoc, onhand_tau=0.5):
    state = SM.install(router, checkpoint, assoc, onhand_tau)
    XAG.sample_sources = predicted_sources
    ENC.event_tokens = event_tokens
    if assoc == "pfa":
        XAG.route_association = pfa_association
        XM.associate_rays = empty_association
    _STATE["on"] = True
    return {**state, "fast": True}


def uninstall():
    ENC.event_tokens = _NATIVE_TOKENS
    _MEMO.update(key=None, value=None)
    _GRAPHS.clear()
    _STATE["on"] = False
    SM.uninstall()


def installed():
    state = SM.installed()
    return None if state is None else {**state, "fast": _STATE["on"]}


__all__ = ["empty_association", "event_image", "event_tokens", "install", "installed", "pfa_association",
           "predicted_sources", "route", "router_forward", "trunk", "uninstall"]
