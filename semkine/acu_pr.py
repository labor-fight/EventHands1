r"""ACU PR: a causal, asynchronous point-graph surface router (docs/EXPERIMENT_RECORDS.md#acu-s1c-prereg-20261104).

The deployed router r_s2d draws the whole 50 ms packet as an event image and runs a CNN, so an early event's route
depends on later events. PR computes every node's route from the node itself and *strictly earlier* nodes only, with
cached per-node features (no event image, no CNN):

  nodes      causal stride: inside a readout interval I (length w), keep the j-th arriving event iff j % k == 0 and
             j // k < N_CAP, k = max(1, ceil(rate * w / N_CAP)), rate = the event rate of the interval before I (known
             when I starts; k = 1 without one). History nodes (the H = 50 ms before I) come from the same rule with their
             own predecessor's rate (`k_hist`).
  layer 0    e_i = MLP0(x/W, y/H, polarity, log time since the last same / opposite polarity event at the pixel among all
             earlier events, capped at SAE_CAP = 10 ms): a function of earlier events only
  layer 1    h1_i = MLP1(max over {i} and its K1 nearest earlier nodes j with t_i - t_j <= TAU, |x_i - x_j|, |y_i - y_j|
             within the R1 grid neighbourhood, of A0 e_j + G1 (dx, dy, dt, p_i p_j))
  context    c_i = MLPc(mean of h1 over earlier nodes within TAU in the 3x3 coarse cells around i, the same over the whole
             image, log counts)
  layer 2    h2_i = MLP2(max over {i} and its K2 nearest earlier nodes (R2) of A1 h1_j + G2 (...))
  heads      [e, h1, h2, c] -> part (16) / rest-template point (cm) / on-hand logit, decoded to (face, barycentrics) with
             S45-SM's tables (part, then the nearest face centroid of that part), as r_s2d
A window node's outputs depend on events within 2 * TAU + SAE_CAP = 50 ms = H before it, so a 50 ms history makes the
batched computation equal to the event-by-event one. The labels (XAG pseudo-sources) supervise training only.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from . import s45sm as SM
from .encoder import sae_times
from .events import EV_BATCH, EV_P, EV_T, EV_X, EV_Y

H_S = 0.050
N_CAP = 2048
TAU_S = 0.020
R1, R2, K1, K2 = 6, 16, 8, 8
CTX_CELL = 30
WIDTH, HEIGHT = 240, 180
CW = 48
HEAD = 96
T_UNIT = 1e-6                    # node times are compared in integer microseconds
SAE_CAP_S = 0.010                # the pixel-memory feature saturates here: 2 * TAU + SAE_CAP <= H keeps a window node's
                                 # outputs a function of the H-ms history and the window alone


# --------------------------------------------------------------------------- causal nodes
def causal_stride(ptr: torch.Tensor, k: torch.Tensor, cap: int = N_CAP):
    """`(src (B, N) long, mask (B, N) bool)` like `stride_sample`: per packet the events j with j % k == 0, j // k < cap,
    in arrival order; `k (B,)` comes from the past (never from this packet's own count)."""
    dev = ptr.device
    counts = (ptr[1:] - ptr[:-1]).clamp(min=0)
    k = k.to(dev).long().clamp_min(1)
    n = torch.minimum((counts + k - 1) // k, torch.full_like(counts, int(cap)))
    N = max(int(n.max()) if n.numel() else 0, 1)
    pos = torch.arange(N, device=dev)
    mask = pos.unsqueeze(0) < n.unsqueeze(1)
    src = ptr[:-1].unsqueeze(1) + pos.unsqueeze(0) * k.unsqueeze(1)
    src = torch.where(mask, src, torch.zeros_like(src))
    return src, mask


def stride_k(n_prev, len_prev_s, w_s, cap: int = N_CAP):
    """k = max(1, ceil(rate * w / cap)), rate = n(I_prev) / |I_prev| (0 when there is no previous interval)."""
    n_prev, len_prev_s, w_s = (torch.as_tensor(v, dtype=torch.float64) for v in (n_prev, len_prev_s, w_s))
    rate = torch.where(len_prev_s > 0, n_prev / len_prev_s.clamp_min(1e-12), torch.zeros_like(n_prev))
    return torch.clamp(torch.ceil(rate * w_s / cap), min=1).long()


# --------------------------------------------------------------------------- causal neighbourhoods
def _segments(b, gx, gy, ncx, ncy):
    return (b * (ncy * ncx) + gy.clamp(0, ncy - 1) * ncx + gx.clamp(0, ncx - 1)).long()


@torch.no_grad()
def causal_neighbours(b, pos, t_us, x, y, cell, K, tau_us, query=None):
    """K nearest strictly earlier nodes (same sample, pos_j < pos_i, t_i - t_j <= tau) within the 3x3 grid cells of size
    `cell` around each query node; `(idx (Q, K) long, valid (Q, K) bool)`, idx into the node list.

    Nodes are given in stream order per sample (pos increasing with time). Candidates per neighbouring cell are its K most
    recent earlier nodes in the time window; the K nearest in space among the 9K candidates are kept (ties: earlier
    candidate order)."""
    n = b.shape[0]
    dev = b.device
    q = torch.arange(n, device=dev) if query is None else query
    ncx, ncy = WIDTH // cell + 2, HEIGHT // cell + 2
    gx, gy = (x // cell).long() + 1, (y // cell).long() + 1
    seg = _segments(b, gx, gy, ncx, ncy)
    P = int(pos.max().item()) + 2 if n else 2
    key_p = seg * P + pos
    order = torch.argsort(key_p)
    kp, ks, kt = key_p[order], seg[order], t_us[order]
    T = int(t_us.max().item() - t_us.min().item()) + int(tau_us) + 2 if n else 2
    t0 = t_us.min() if n else torch.zeros((), device=dev, dtype=torch.long)
    key_t = ks * T + (kt - t0)
    Q = q.shape[0]
    cand = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            s2 = _segments(b[q], gx[q] + dx, gy[q] + dy, ncx, ncy)
            up = torch.searchsorted(kp, s2 * P + pos[q])
            lo = torch.searchsorted(key_t, s2 * T + (t_us[q] - int(tau_us) - t0))
            off = torch.arange(1, K + 1, device=dev)
            c = up.unsqueeze(1) - off.unsqueeze(0)                                    # (Q, K)
            ok = (c >= lo.unsqueeze(1)) & (c >= 0)
            c = c.clamp_min(0)
            ok = ok & (ks[c] == s2.unsqueeze(1))
            cand.append((order[c], ok))
    ci = torch.cat([c for c, _ in cand], 1)                                           # (Q, 9K)
    cv = torch.cat([v for _, v in cand], 1)
    d2 = (x[ci] - x[q].unsqueeze(1)).float().square() + (y[ci] - y[q].unsqueeze(1)).float().square()
    d2 = d2.masked_fill(~cv, float("inf"))
    kk = min(K, ci.shape[1])
    best = d2.topk(kk, dim=1, largest=False)
    idx = ci.gather(1, best.indices)
    valid = torch.isfinite(best.values)
    return idx, valid


@torch.no_grad()
def causal_cell_sums(b, pos, t_us, x, y, feat, cell, tau_us):
    """For every node i: sums of `feat` and counts over strictly earlier nodes of the same sample with t_i - t_j <= tau,
    (a) in each of the 3x3 coarse cells around i -> (n, 9, C), (9) counts; (b) over the whole image -> (n, C), (n,)."""
    n, C = feat.shape
    dev = feat.device
    ncx, ncy = WIDTH // cell + 2, HEIGHT // cell + 2
    gx, gy = (x // cell).long() + 1, (y // cell).long() + 1
    P = int(pos.max().item()) + 2 if n else 2
    t0 = t_us.min() if n else torch.zeros((), device=dev, dtype=torch.long)
    T = int(t_us.max().item() - t_us.min().item()) + int(tau_us) + 2 if n else 2

    def window_sums(seg):
        key_p = seg * P + pos
        order = torch.argsort(key_p)
        kp, ks, kt = key_p[order], seg[order], t_us[order]
        key_t = ks * T + (kt - t0)
        cs = torch.cat([torch.zeros(1, C, device=dev, dtype=torch.float64), feat[order].double().cumsum(0)])
        cn = torch.arange(n + 1, device=dev, dtype=torch.float64)
        return kp, key_t, cs, cn

    out_c, out_n = [], []
    seg_all = _segments(b, gx, gy, ncx, ncy)
    kp, key_t, cs, cn = window_sums(seg_all)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            s2 = _segments(b, gx + dx, gy + dy, ncx, ncy)
            up = torch.searchsorted(kp, s2 * P + pos)
            lo = torch.searchsorted(key_t, s2 * T + (t_us - int(tau_us) - t0))
            lo = torch.minimum(lo, up)
            out_c.append((cs[up] - cs[lo]).float())
            out_n.append((cn[up] - cn[lo]).float())
    seg_b = b.long()
    kp, key_t, cs, cn = window_sums(seg_b)
    up = torch.searchsorted(kp, seg_b * P + pos)
    lo = torch.minimum(torch.searchsorted(key_t, seg_b * T + (t_us - int(tau_us) - t0)), up)
    return torch.stack(out_c, 1), torch.stack(out_n, 1), (cs[up] - cs[lo]).float(), (cn[up] - cn[lo]).float()


# --------------------------------------------------------------------------- the network
class EdgeLayer(nn.Module):
    """max over {i} + causal neighbours of (A f_j + G geom_ij), then a 2-layer MLP; A f_j is computed once per node."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.a = nn.Linear(c_in, c_out)
        self.g = nn.Linear(4, c_out, bias=False)
        self.post = nn.Sequential(nn.ReLU(), nn.Linear(c_out, c_out), nn.ReLU())

    def forward(self, f, idx, valid, x, y, t_us, p, radius, tau_us):
        af = self.a(f)                                                                # (n, C)
        n = f.shape[0]
        self_geom = torch.zeros(n, 4, device=f.device, dtype=af.dtype)
        self_geom[:, 3] = 1.0
        m = af + self.g(self_geom)
        if idx.numel():
            geom = torch.stack([(x[idx] - x.unsqueeze(1)).float() / radius, (y[idx] - y.unsqueeze(1)).float() / radius,
                                (t_us.unsqueeze(1) - t_us[idx]).float() / float(tau_us),
                                (p[idx] * p.unsqueeze(1)).float()], -1).to(af.dtype)
            msg = af[idx] + self.g(geom)                                              # (n, K, C)
            msg = msg.masked_fill(~valid.unsqueeze(-1), -1e4)
            m = torch.maximum(m, msg.amax(1))
        return self.post(m)


class PRNet(nn.Module):
    """The causal point-graph router; buffers and decoding tables as S45-SM's SRNet (`route` is SRNet's)."""

    def __init__(self, faces: torch.Tensor, verts: torch.Tensor, face_part: torch.Tensor, width: int = CW, head: int = HEAD):
        super().__init__()
        self.design = {"width": int(width), "head": int(head), "tau_s": TAU_S, "r1": R1, "r2": R2, "k1": K1, "k2": K2,
                       "ctx_cell": CTX_CELL, "h_s": H_S, "n_cap": N_CAP}
        c = int(width)
        self.e = nn.Sequential(nn.Linear(5, c), nn.ReLU())
        self.l1 = EdgeLayer(c, c)
        self.ctx = nn.Sequential(nn.Linear(2 * c + 10, c), nn.ReLU(), nn.Linear(c, c), nn.ReLU())
        self.l2 = EdgeLayer(c, c)
        self.mlp = nn.Sequential(nn.Linear(4 * c, head), nn.ReLU(inplace=True), nn.Linear(head, head), nn.ReLU(inplace=True))
        self.part = nn.Linear(head, SM.N_PARTS)
        self.coord = nn.Linear(head, 3)
        self.onhand = nn.Linear(head, 1)
        self.register_buffer("faces", faces.long().clone(), persistent=False)
        self.register_buffer("tmpl", verts.float().clone(), persistent=False)
        self.register_buffer("face_part", face_part.long().clone(), persistent=False)
        members = [torch.nonzero(face_part == k).flatten() for k in range(SM.N_PARTS)]
        wdt = max(len(m) for m in members)
        table = torch.zeros(SM.N_PARTS, wdt, dtype=torch.long)
        ok = torch.zeros(SM.N_PARTS, wdt, dtype=torch.bool)
        for k, m in enumerate(members):
            table[k, :len(m)], ok[k, :len(m)] = m, True
        self.register_buffer("part_faces", table, persistent=False)
        self.register_buffer("part_faces_valid", ok, persistent=False)
        self.register_buffer("centroid", verts.float()[faces.long()].mean(1), persistent=False)

    route = SM.SRNet.route

    def forward(self, events, ptr, delta_t_s, hist_events, hist_ptr, k_win, k_hist):
        """Window events `(N, 5)` [b, x, y, t_rel >= 0, p] / `ptr`, history events of the H before the window `(Nh, 5)`
        [b, x, y, t_rel < 0, p] / `hist_ptr`, the causal strides `k_win`, `k_hist (B,)`. Returns S45-SM's router dict
        for the window nodes (`src`, `mask` index the window events, as `stride_sample` does)."""
        B = int(ptr.numel() - 1)
        dev = events.device
        src, mask = causal_stride(ptr, k_win)
        hsrc, hmask = causal_stride(hist_ptr, k_hist)
        # all events of a sample in time order (history first): the pixel memory reads every earlier event
        allev = torch.cat([hist_events, events], 0)
        nb_h = (hist_ptr[1:] - hist_ptr[:-1])
        nb_w = (ptr[1:] - ptr[:-1])
        bid = torch.cat([hist_events[:, EV_BATCH], events[:, EV_BATCH]]).long()
        rank = torch.cat([torch.arange(hist_events.shape[0], device=dev) - hist_ptr[:-1][hist_events[:, EV_BATCH].long()],
                          torch.arange(events.shape[0], device=dev) - ptr[:-1][events[:, EV_BATCH].long()]
                          + nb_h[events[:, EV_BATCH].long()]])
        order = torch.argsort(bid * (int(rank.max().item()) + 1 if rank.numel() else 1) + rank)
        allev = allev[order]
        cnt = nb_h + nb_w
        aptr = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), cnt.cumsum(0)])
        same, opp = sae_times(allev, aptr, HEIGHT, WIDTH, fallback=SAE_CAP_S)
        same, opp = same.clamp(max=SAE_CAP_S), opp.clamp(max=SAE_CAP_S)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.shape[0], device=dev)
        # node list per sample: its history nodes, then its window nodes, each in arrival (= time) order
        hb, hj = hmask.nonzero(as_tuple=True)
        wb, wj = mask.nonzero(as_tuple=True)
        width_h = hsrc.shape[1]
        key = torch.cat([hb * (width_h + src.shape[1]) + hj, wb * (width_h + src.shape[1]) + width_h + wj])
        h_idx, w_idx = hsrc[hb, hj], src[wb, wj]
        node_ev = torch.cat([hist_events[h_idx], events[w_idx]], 0)
        node_all = torch.cat([inv[h_idx], inv[hist_events.shape[0] + w_idx]], 0)      # rows in `allev`
        nb = torch.cat([hb, wb])
        is_win = torch.cat([torch.zeros_like(hb, dtype=torch.bool), torch.ones_like(wb, dtype=torch.bool)])
        win_slot = torch.cat([torch.full_like(hb, -1), wj])
        o2 = torch.argsort(key)
        node_ev, node_all, nb, is_win, win_slot = node_ev[o2], node_all[o2], nb[o2], is_win[o2], win_slot[o2]
        n = node_ev.shape[0]
        x = node_ev[:, EV_X].round().long()
        y = node_ev[:, EV_Y].round().long()
        t_us = torch.round(node_ev[:, EV_T].double() / T_UNIT).long()
        p = torch.where(node_ev[:, EV_P] > 0.5, 1.0, -1.0).to(node_ev.dtype)
        first = torch.searchsorted(nb, torch.arange(B, device=dev))
        pos = torch.arange(n, device=dev) - first[nb]
        lim = math.log1p(SAE_CAP_S * 1e3)
        f0 = torch.stack([node_ev[:, EV_X] / WIDTH, node_ev[:, EV_Y] / HEIGHT, p,
                          torch.log1p(same[node_all] * 1e3) / lim, torch.log1p(opp[node_all] * 1e3) / lim], -1)
        wdt = self.e[0].weight.dtype
        e = self.e(f0.to(wdt))
        tau = int(round(TAU_S / T_UNIT))
        i1, v1 = causal_neighbours(nb, pos, t_us, x, y, R1, K1, tau)
        h1 = self.l1(e, i1, v1, x, y, t_us, p, float(R1), tau)
        cs, cn, gs, gn = causal_cell_sums(nb, pos, t_us, x, y, h1.float(), CTX_CELL, tau)
        loc = cs.sum(1) / cn.sum(1, keepdim=True).clamp_min(1.0)
        glob = gs / gn.unsqueeze(-1).clamp_min(1.0)
        cnts = torch.log1p(torch.cat([cn, gn.unsqueeze(-1)], 1))
        c = self.ctx(torch.cat([loc, glob, cnts], -1).to(wdt))
        i2, v2 = causal_neighbours(nb, pos, t_us, x, y, R2, K2, tau)
        h2 = self.l2(h1, i2, v2, x, y, t_us, p, float(R2), tau)
        z_all = self.mlp(torch.cat([e, h1, h2, c], -1))
        # back to the (B, N) layout of the window nodes
        z = torch.zeros(B, src.shape[1], z_all.shape[-1], device=dev, dtype=z_all.dtype)
        z[nb[is_win], win_slot[is_win]] = z_all[is_win]
        return {"z": z, "part": self.part(z), "coord": self.coord(z), "onhand": self.onhand(z).squeeze(-1),
                "src": src, "mask": mask}


def build(npz, **design):
    with np.load(str(npz), allow_pickle=False) as zf:
        verts = torch.tensor(np.asarray(zf["v_template"], np.float32))
        faces = torch.tensor(np.asarray(zf["f"], np.int64))
        part = torch.from_numpy(SM.face_part_table(np.asarray(zf["weights"]), np.asarray(zf["f"], np.int64)))
    return PRNet(faces, verts, part, **design)


def n_params(net):
    return int(sum(p.numel() for p in net.parameters()))


def save_checkpoint(net, path, **meta):
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "faces": net.faces.cpu(),
                "tmpl": net.tmpl.cpu(), "face_part": net.face_part.cpu(), "kind": "acu_pr", "design": net.design, **meta},
               str(path))


def load(path, map_location="cpu"):
    obj = torch.load(str(path), map_location=map_location)
    if obj.get("kind") != "acu_pr":
        raise ValueError("not an ACU PR checkpoint: " + str(path))
    d = obj["design"]
    net = PRNet(obj["faces"], obj["tmpl"], obj["face_part"], width=d["width"], head=d["head"])
    net.load_state_dict(obj["state_dict"], strict=True)
    return net.eval(), obj


__all__ = ["CW", "H_S", "HEAD", "N_CAP", "PRNet", "TAU_S", "build", "causal_cell_sums", "causal_neighbours",
           "causal_stride", "load", "n_params", "save_checkpoint", "stride_k"]
