"""EVF event nodes and their features (docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104).

Nodes. A node is the first event of its cell (packet, time bin, polarity, VOXEL_PX x VOXEL_PX pixels), with
`TIME_BINS` bins over the packet duration Delta_t. Whether an event becomes a node depends only on the events before it,
so the set grows by appending as events arrive (the asynchronous runtime, `semkine/evf_async.py`, keeps the same set).
A packet with more than `cap` nodes keeps `cap` of them at a uniform stride in event order (training packets up to
300 ms only; a 50 ms evaluation packet above the cap is reported, the asynchronous equivalence requires none).

Event attributes, prefix-stable: polarity, t / Delta_t, the time since the previous event (in packet order) of the same
polarity at the same pixel, and since the previous one of the opposite polarity there, both over Delta_t (1 when there
is none; a per-pixel timestamp map gives both in O(1) per event), log(Delta_t / 50 ms).

Features of a node, "event minus prediction" (`node_features(..., vf=True)`): its pixel relative to the projected root
joint in hand-scale units s; what it reads from the virtual frame (`semkine/evf_render.py sample`): inside, signed
distance to the predicted silhouette (in s units and in pixels / 16) with the outward direction, distance to the
nearest predicted occluding contour with the direction from it, root-relative depth and template coordinates of the
predicted surface under / nearest to the pixel and of the nearest contour point (optionally LBS weights). Only relative
quantities: no absolute depth, no absolute image position. The control without the virtual frame (`vf=False`) reads the
event attributes and the pixel position in the image (x / W, y / H) only.

Packet statistics (`packet_from_sums`), from sums over the nodes (so incremental): event and node counts, Delta_t, the
node cloud's mean and covariance relative to the root in s units next to the predicted silhouette's, their log size
ratio, the fractions inside / near a contour, the mean |signed distance| / s, the silhouette's area in s^2; the
control: counts, Delta_t and the node cloud's image moments.
"""
from __future__ import annotations

import torch

from semkine.evf_render import FAR_PX, sample
from semkine.events import EV_BATCH, EV_P, EV_T, EV_X, EV_Y

VOXEL_PX = 3
TIME_BINS = 5
MAX_NODES = 6144
PX_SCALE = 16.0
DT_REF_S = 0.05
NEAR_PX = 2.0
EVENT_DIM = 5
NODE_DIM_VF = 7 + 21
NODE_DIM_NOVF = 7
LBS_DIM = 32                 # two 16-joint LBS weight vectors (surface, contour), optional
PACKET_DIM_VF = 18
PACKET_DIM_NOVF = 8
MOMENT_DIM_VF = 8            # sums: rx, ry, rx^2, ry^2, rx ry, inside, near, |ds| / s
MOMENT_DIM_NOVF = 5          # sums: x, y, x^2, y^2, x y (image units)


@torch.no_grad()
def node_keys(events, delta_t_s, H: int, W: int, voxel: int = VOXEL_PX, bins: int = TIME_BINS):
    b = events[:, EV_BATCH].long()
    x = events[:, EV_X].long().clamp(0, W - 1)
    y = events[:, EV_Y].long().clamp(0, H - 1)
    p = (events[:, EV_P] > 0.5).long()
    dt = delta_t_s.float().clamp_min(1e-6)[b]
    tb = (events[:, EV_T].float() / dt * bins).floor().long().clamp(0, bins - 1)
    gw, gh = (W + voxel - 1) // voxel, (H + voxel - 1) // voxel
    return (((b * bins + tb) * 2 + p) * gh + y // voxel) * gw + x // voxel


@torch.no_grad()
def first_of_keys(key: torch.Tensor) -> torch.Tensor:
    """Row indices (ascending) of the first row of every distinct key."""
    sk, order = torch.sort(key, stable=True)
    first = torch.ones_like(sk, dtype=torch.bool)
    first[1:] = sk[1:] != sk[:-1]
    return torch.sort(order[first]).values


@torch.no_grad()
def select_nodes(events: torch.Tensor, ptr: torch.Tensor, delta_t_s: torch.Tensor, H: int, W: int,
                 voxel: int = VOXEL_PX, bins: int = TIME_BINS, cap: int = MAX_NODES):
    """`(src (B, N) long, mask (B, N) bool, n_nodes (B,) long before the cap)`; src indexes `events` rows."""
    dev = events.device
    B = int(ptr.numel() - 1)
    if int(events.shape[0]) == 0:
        return (torch.zeros(B, 1, dtype=torch.long, device=dev), torch.zeros(B, 1, dtype=torch.bool, device=dev),
                torch.zeros(B, dtype=torch.long, device=dev))
    nodes = first_of_keys(node_keys(events, delta_t_s, H, W, voxel, bins))
    nb = events[nodes, EV_BATCH].long()
    count = torch.bincount(nb, minlength=B)
    start = torch.cumsum(count, 0) - count
    rank = torch.arange(nodes.numel(), device=dev) - start[nb]
    c = count[nb].clamp_min(1)
    lvl = torch.div(rank * cap, c, rounding_mode="floor")
    prev_lvl = torch.div((rank - 1).clamp_min(0) * cap, c, rounding_mode="floor")
    keep = (count[nb] <= cap) | (rank == 0) | (lvl != prev_lvl)
    nodes, nb = nodes[keep], nb[keep]
    kcount = torch.bincount(nb, minlength=B)
    kstart = torch.cumsum(kcount, 0) - kcount
    krank = torch.arange(nodes.numel(), device=dev) - kstart[nb]
    N = max(int(kcount.max()), 1)
    src = torch.zeros(B, N, dtype=torch.long, device=dev)
    mask = torch.zeros(B, N, dtype=torch.bool, device=dev)
    src[nb, krank] = nodes
    mask[nb, krank] = True
    return src, mask, count


@torch.no_grad()
def previous_times(group: torch.Tensor, pol: torch.Tensor, t: torch.Tensor):
    """For every row (rows in packet order): the time of the previous row of the same group (pixel) with the same
    polarity and with the opposite polarity, `-1` where there is none."""
    n = int(t.shape[0])
    dev = t.device
    idx = torch.arange(n, device=dev)
    order = torch.sort(group * n + idx).indices                       # by (group, packet order)
    g, ps, ts = group[order], pol[order], t[order]
    start = torch.ones(n, dtype=torch.bool, device=dev)
    start[1:] = g[1:] != g[:-1]
    gstart = torch.cummax(torch.where(start, idx, torch.zeros_like(idx)), 0).values
    out = []
    for q in (0, 1):
        last = torch.cummax(torch.where(ps == q, idx, torch.full_like(idx, -1)), 0).values
        prev = torch.cat([last.new_full((1,), -1), last[:-1]])           # strictly before
        ok = prev >= gstart
        out.append(torch.where(ok, ts[prev.clamp_min(0)], torch.full_like(ts, -1.0)))
    same = torch.where(ps == 1, out[1], out[0])
    opp = torch.where(ps == 1, out[0], out[1])
    res_s, res_o = torch.empty_like(t), torch.empty_like(t)
    res_s[order], res_o[order] = same, opp
    return res_s, res_o


@torch.no_grad()
def attributes_from_times(pol, t, same_t, opp_t, dt):
    def since(x):
        return torch.where(x >= 0, ((t - x) / dt).clamp(0.0, 1.0), torch.ones_like(t))
    return torch.stack([2.0 * pol.float() - 1.0, t / dt, since(same_t), since(opp_t), torch.log(dt / DT_REF_S)], -1)


@torch.no_grad()
def event_attributes(events: torch.Tensor, ptr: torch.Tensor, delta_t_s: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """`(N, EVENT_DIM)` prefix-stable attributes of every event (see the module docstring)."""
    if int(events.shape[0]) == 0:
        return events.new_zeros((0, EVENT_DIM))
    b = events[:, EV_BATCH].long()
    pix = b * (H * W) + events[:, EV_Y].long().clamp(0, H - 1) * W + events[:, EV_X].long().clamp(0, W - 1)
    pol = (events[:, EV_P] > 0.5).long()
    t = events[:, EV_T].float()
    same_t, opp_t = previous_times(pix, pol, t)
    return attributes_from_times(pol, t, same_t, opp_t, delta_t_s.float().clamp_min(1e-6)[b])


def _scatter_rows(flat, values, B, N):
    out = values.new_zeros((B * N,) + tuple(values.shape[1:]))
    out[flat] = values
    return out.view(B, N, *values.shape[1:])


@torch.no_grad()
def node_rows(events, attrs, rows, bi, frame, template, H: int, W: int, vf: bool = True, lbs=None):
    """Features `(M, D)` and moment terms `(M, MOMENT_DIM_*)` of the nodes at event rows `rows` of samples `bi`."""
    px = events[rows, EV_X].long().clamp(0, W - 1)
    py = events[rows, EV_Y].long().clamp(0, H - 1)
    ev = attrs[rows].float()
    if not vf:
        x, y = px.float() / W, py.float() / H
        return torch.cat([ev, torch.stack([x, y], -1)], -1), torch.stack([x, y, x * x, y * y, x * y], -1)
    fr = frame
    s = fr.scale[bi]
    rx, ry = (px.float() - fr.root_u[bi]) / s, (py.float() - fr.root_v[bi]) / s
    smp = sample(fr, bi, px, py, template, fr.root_z, lbs)
    ds, dc = smp["ds"], smp["dc"]
    parts = [ev, torch.stack([rx, ry], -1), smp["inside"].float()[:, None], (ds / s)[:, None],
             (ds / PX_SCALE)[:, None], smp["dir_s"], (dc / s)[:, None], (dc / PX_SCALE)[:, None], smp["dir_c"],
             (smp["dz_s"] / 0.1)[:, None], smp["tmpl_s"] / 0.1, (smp["dz_c"] / 0.1)[:, None], smp["tmpl_c"] / 0.1,
             torch.stack([smp["valid_s"].float(), smp["valid_c"].float(), smp["far_s"].float(),
                          smp["far_c"].float()], -1)]
    if lbs is not None:
        parts += [smp["lbs_s"], smp["lbs_c"]]
    mom = torch.stack([rx, ry, rx * rx, ry * ry, rx * ry, smp["inside"].float(), (dc <= NEAR_PX).float(),
                       ds.abs() / s], -1)
    return torch.cat(parts, -1), mom


@torch.no_grad()
def node_features(events, attrs, src, mask, frame, template, H: int, W: int, vf: bool = True, lbs=None):
    """`(feat (B, N, D), moment sums (B, MOMENT_DIM_*))`; padded slots are zero."""
    B, N = src.shape
    flat = mask.reshape(-1).nonzero().squeeze(1)
    rows = src.reshape(-1)[flat]
    bi = torch.div(flat, N, rounding_mode="floor")
    feat, mom = node_rows(events, attrs, rows, bi, frame, template, H, W, vf, lbs)
    return _scatter_rows(flat, feat, B, N), _scatter_rows(flat, mom, B, N).sum(1)


@torch.no_grad()
def packet_from_sums(n_events, n_nodes, delta_t_s, count, sums, frame, vf: bool = True) -> torch.Tensor:
    """`(B, PACKET_DIM_*)` packet statistics from the node count and moment sums."""
    den = count.float().clamp_min(1.0)
    has = (count > 0).float()
    mean = sums.float() / den[:, None]
    head = torch.stack([torch.log1p(n_events.float() / 1000.0), torch.log1p(n_nodes.float() / 1000.0),
                        torch.log(delta_t_s.float().clamp_min(1e-6) / DT_REF_S)], -1)
    mx, my = mean[:, 0], mean[:, 1]
    mom = torch.stack([mx, my, mean[:, 2] - mx * mx, mean[:, 3] - my * my, mean[:, 4] - mx * my], -1) * has[:, None]
    if not vf:
        return torch.cat([head, mom], -1)
    sm = frame.sil_moments
    ratio = 0.5 * (torch.log((mom[:, 2] + mom[:, 3]).clamp_min(0.0) + 1e-3) - torch.log(sm[:, 2] + sm[:, 3] + 1e-3))
    area = torch.log1p(frame.sil_area / frame.scale.square())
    tail = torch.stack([ratio * has, mean[:, 5], mean[:, 6], mean[:, 7], area], -1)
    return torch.cat([head, mom, sm, tail], -1)


__all__ = ["EVENT_DIM", "FAR_PX", "LBS_DIM", "MAX_NODES", "MOMENT_DIM_NOVF", "MOMENT_DIM_VF", "NODE_DIM_NOVF",
           "NODE_DIM_VF", "PACKET_DIM_NOVF", "PACKET_DIM_VF", "TIME_BINS", "VOXEL_PX", "attributes_from_times",
           "event_attributes", "first_of_keys", "node_features", "node_keys", "node_rows", "packet_from_sums",
           "previous_times", "select_nodes"]
