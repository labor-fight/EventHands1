r"""ACU PR2: PR with a causal multi-scale shape context, the one registered PR revision allowed by the s1c rule
(docs/EXPERIMENT_RECORDS.md#acu-pr2-prereg-20261104).

PR's context pools h1 over its 3 x 3 coarse cells into one mean (its only layout cues are nine log counts) and its edge
layers read the 8 nearest earlier nodes, so a node sees a few pixels of structure; r_s2d's CNN sees the whole hand. PR2
keeps PR's nodes, layer 0, layers 1 / 2, context and heads unchanged and adds a shape context of every window node i:

  cells    for each scale s in SCALES = (8, 20, 50) px, the 5 x 5 image-aligned cells of size s around i's cell; per cell
           the count (log1p) and the mean of the fixed per-node features (polarity, PR's two pixel-memory times) over the
           strictly earlier nodes j of the sample with t_i - t_j <= TAU_C = 40 ms
  global   over the same nodes of the whole image: log1p count, mean polarity, i's offset from their centroid in units of
           their RMS radius (clipped to +-4), log of that radius
  phase    i's position inside its cell at each scale (x mod s / s, y mod s / s)
  s_i      MLP_s(cells, global, phase): 75 * 4 + 5 + 6 -> SC_HIDDEN -> 48, then the head reads [e, h1, h2, c, s]
           (5 * 48 -> HEAD2 -> HEAD2 -> part / point / on-hand)

The summed features are fixed functions of the events (no learned input), so the sums need no gradient; they are exact
int64 fixed-point prefix sums (semkine/acu_pr_det.py's rule: every summand rounded to 2^-20, deterministic on every
device). A window node's shape context depends on events within TAU_C + SAE_CAP = 50 ms = H before it, so PR's
batched = event-by-event equality is kept. Only window nodes get a shape context (history nodes are inputs only).

    install()    semkine.acu_pr.{build, load, save_checkpoint, n_params} -> this module's (load reads both kinds)
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from . import acu_pr as PR
from . import s45sm as SM
from .encoder import sae_times
from .events import EV_BATCH, EV_P, EV_T, EV_X, EV_Y

SCALES = (8, 20, 50)
RADIUS = 2
TAU_C_S = 0.040
N_SC = 3                           # summed per-node features: polarity, pixel-memory same / opposite
N_GLOBAL = 5
HEAD2 = 64                         # PR2's head width (PR: 96) and shape-context hidden width: PR's cost per node
SC_HIDDEN = 32
FIX_BITS = 20
SCALE_FIX = float(2 ** FIX_BITS)
_native = {"build": PR.build, "load": PR.load, "save_checkpoint": PR.save_checkpoint, "n_params": PR.n_params}


def _prefix(feat_sorted: torch.Tensor) -> torch.Tensor:
    q = torch.round(feat_sorted.double() * SCALE_FIX).long()
    if q.numel():
        lim = int(q.abs().max().item()) * q.shape[0]
        if lim >= 2 ** 62:
            raise OverflowError(f"fixed-point window sums would overflow int64 (bound {lim})")
    z = torch.zeros(1, q.shape[1], device=q.device, dtype=torch.long)
    return torch.cat([z, q.cumsum(0)])


@torch.no_grad()
def causal_grid_sums(b, pos, t_us, x, y, feat, cell, radius, tau_us, query):
    """For each query node i: per cell of the (2r + 1)^2 image-aligned cells (size `cell`) around i's cell, the sums of
    `feat` and the counts over strictly earlier nodes j of the same sample (pos_j < pos_i) with t_i - t_j <= tau:
    `(sums (Q, (2r+1)^2, C) float32, counts (Q, (2r+1)^2) float32)`. `cell` = None: the whole image (one cell)."""
    n, C = feat.shape
    dev = feat.device
    if cell is None:
        seg = b.long()
        offsets = [(0, 0)]
        sq = lambda dx, dy: b[query].long()                                           # noqa: E731
    else:
        m = int(radius)
        ncx, ncy = PR.WIDTH // cell + 1 + 2 * m, PR.HEIGHT // cell + 1 + 2 * m
        gx = (x // cell).long().clamp(0, PR.WIDTH // cell) + m
        gy = (y // cell).long().clamp(0, PR.HEIGHT // cell) + m
        seg = b.long() * (ncy * ncx) + gy * ncx + gx
        offsets = [(dx, dy) for dy in range(-m, m + 1) for dx in range(-m, m + 1)]
        sq = lambda dx, dy: b[query].long() * (ncy * ncx) + (gy[query] + dy) * ncx + (gx[query] + dx)  # noqa: E731
    P = int(pos.max().item()) + 2 if n else 2
    t0 = t_us.min() if n else torch.zeros((), device=dev, dtype=torch.long)
    T = int(t_us.max().item() - t_us.min().item()) + int(tau_us) + 2 if n else 2
    key_p = seg * P + pos
    order = torch.argsort(key_p)
    kp, ks, kt = key_p[order], seg[order], t_us[order]
    key_t = ks * T + (kt - t0)
    cs = _prefix(feat[order])
    # all offsets of the neighbourhood in one lookup (Q, O): the same integer arithmetic as one lookup per offset
    s2 = torch.stack([sq(dx, dy) for dx, dy in offsets], 1)
    up = torch.searchsorted(kp, (s2 * P + pos[query].unsqueeze(1)).flatten()).view(s2.shape)
    lo = torch.searchsorted(key_t, (s2 * T + (t_us[query] - int(tau_us) - t0).unsqueeze(1)).flatten()).view(s2.shape)
    lo = torch.minimum(lo, up)
    return ((cs[up] - cs[lo]).double() / SCALE_FIX).float(), (up - lo).float()


class PR2Net(PR.PRNet):
    """PR + the causal multi-scale shape context of the window nodes (see the module docstring)."""

    def __init__(self, faces, verts, face_part, width: int = PR.CW, head: int = HEAD2, sc_hidden: int = SC_HIDDEN):
        super().__init__(faces, verts, face_part, width=width, head=head)
        c = int(width)
        n_in = len(SCALES) * (2 * RADIUS + 1) ** 2 * (N_SC + 1) + N_GLOBAL + 2 * len(SCALES)
        self.design.update(kind="pr2", scales=list(SCALES), radius=RADIUS, tau_c_s=TAU_C_S, sc_in=n_in,
                           sc_hidden=int(sc_hidden))
        self.sc = nn.Sequential(nn.Linear(n_in, int(sc_hidden)), nn.ReLU(), nn.Linear(int(sc_hidden), c), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(5 * c, head), nn.ReLU(inplace=True), nn.Linear(head, head),
                                 nn.ReLU(inplace=True))

    def shape_context(self, nb, pos, t_us, x, y, xf, yf, p, same_n, opp_n, query):
        tau = int(round(TAU_C_S / PR.T_UNIT))
        feat = torch.stack([p.double(), same_n.double(), opp_n.double()], -1)
        parts = []
        for s in SCALES:
            sm, cn = causal_grid_sums(nb, pos, t_us, x, y, feat, s, RADIUS, tau, query)
            mean = sm / cn.unsqueeze(-1).clamp_min(1.0)
            parts.append(torch.cat([torch.log1p(cn).unsqueeze(-1), mean], -1).flatten(1))
        g = torch.stack([torch.ones_like(xf, dtype=torch.float64), p.double(), xf.double() / PR.WIDTH,
                         yf.double() / PR.HEIGHT, (xf.double() / PR.WIDTH) ** 2 + (yf.double() / PR.HEIGHT) ** 2], -1)
        gs, gn = causal_grid_sums(nb, pos, t_us, x, y, g, None, 0, tau, query)
        gs, gn = gs[:, 0].double(), gn[:, 0].double()
        k = gn.clamp_min(1.0)
        cx, cy = gs[:, 2] / k, gs[:, 3] / k
        rad = torch.sqrt((gs[:, 4] / k - cx ** 2 - cy ** 2).clamp_min(1e-6))
        rx = ((xf[query].double() / PR.WIDTH - cx) / rad).clamp(-4, 4)
        ry = ((yf[query].double() / PR.HEIGHT - cy) / rad).clamp(-4, 4)
        empty = gn < 0.5
        rx, ry = rx.masked_fill(empty, 0.0), ry.masked_fill(empty, 0.0)
        glob = torch.stack([torch.log1p(gn), gs[:, 1] / k, rx, ry, torch.log(rad).masked_fill(empty, 0.0)], -1).float()
        phase = torch.stack([v for s in SCALES for v in (torch.remainder(x[query], s).float() / s,
                                                         torch.remainder(y[query], s).float() / s)], -1)
        return torch.cat(parts + [glob, phase], -1)

    def forward(self, events, ptr, delta_t_s, hist_events, hist_ptr, k_win, k_hist):
        """PR's signature and output dict (`src`, `mask` index the window events); see `semkine.acu_pr.PRNet.forward`."""
        B = int(ptr.numel() - 1)
        dev = events.device
        src, mask = PR.causal_stride(ptr, k_win)
        hsrc, hmask = PR.causal_stride(hist_ptr, k_hist)
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
        same, opp = sae_times(allev, aptr, PR.HEIGHT, PR.WIDTH, fallback=PR.SAE_CAP_S)
        same, opp = same.clamp(max=PR.SAE_CAP_S), opp.clamp(max=PR.SAE_CAP_S)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.shape[0], device=dev)
        hb, hj = hmask.nonzero(as_tuple=True)
        wb, wj = mask.nonzero(as_tuple=True)
        width_h = hsrc.shape[1]
        key = torch.cat([hb * (width_h + src.shape[1]) + hj, wb * (width_h + src.shape[1]) + width_h + wj])
        h_idx, w_idx = hsrc[hb, hj], src[wb, wj]
        node_ev = torch.cat([hist_events[h_idx], events[w_idx]], 0)
        node_all = torch.cat([inv[h_idx], inv[hist_events.shape[0] + w_idx]], 0)
        nb = torch.cat([hb, wb])
        is_win = torch.cat([torch.zeros_like(hb, dtype=torch.bool), torch.ones_like(wb, dtype=torch.bool)])
        win_slot = torch.cat([torch.full_like(hb, -1), wj])
        o2 = torch.argsort(key)
        node_ev, node_all, nb, is_win, win_slot = node_ev[o2], node_all[o2], nb[o2], is_win[o2], win_slot[o2]
        n = node_ev.shape[0]
        x = node_ev[:, EV_X].round().long()
        y = node_ev[:, EV_Y].round().long()
        t_us = torch.round(node_ev[:, EV_T].double() / PR.T_UNIT).long()
        p = torch.where(node_ev[:, EV_P] > 0.5, 1.0, -1.0).to(node_ev.dtype)
        first = torch.searchsorted(nb, torch.arange(B, device=dev))
        pos = torch.arange(n, device=dev) - first[nb]
        lim = math.log1p(PR.SAE_CAP_S * 1e3)
        same_n = torch.log1p(same[node_all] * 1e3) / lim
        opp_n = torch.log1p(opp[node_all] * 1e3) / lim
        f0 = torch.stack([node_ev[:, EV_X] / PR.WIDTH, node_ev[:, EV_Y] / PR.HEIGHT, p, same_n, opp_n], -1)
        wdt = self.e[0].weight.dtype
        e = self.e(f0.to(wdt))
        tau = int(round(PR.TAU_S / PR.T_UNIT))
        i1, v1 = PR.causal_neighbours(nb, pos, t_us, x, y, PR.R1, PR.K1, tau)
        h1 = self.l1(e, i1, v1, x, y, t_us, p, float(PR.R1), tau)
        cs, cn, gs, gn = PR.causal_cell_sums(nb, pos, t_us, x, y, h1.float(), PR.CTX_CELL, tau)
        loc = cs.sum(1) / cn.sum(1, keepdim=True).clamp_min(1.0)
        glob = gs / gn.unsqueeze(-1).clamp_min(1.0)
        cnts = torch.log1p(torch.cat([cn, gn.unsqueeze(-1)], 1))
        c = self.ctx(torch.cat([loc, glob, cnts], -1).to(wdt))
        i2, v2 = PR.causal_neighbours(nb, pos, t_us, x, y, PR.R2, PR.K2, tau)
        h2 = self.l2(h1, i2, v2, x, y, t_us, p, float(PR.R2), tau)
        q = is_win.nonzero(as_tuple=True)[0]
        s_in = self.shape_context(nb, pos, t_us, x, y, node_ev[:, EV_X], node_ev[:, EV_Y], p, same_n, opp_n, q)
        s = self.sc(s_in.to(wdt))
        z_w = self.mlp(torch.cat([e[q], h1[q], h2[q], c[q], s], -1))
        z = torch.zeros(B, src.shape[1], z_w.shape[-1], device=dev, dtype=z_w.dtype)
        z[nb[q], win_slot[q]] = z_w
        return {"z": z, "part": self.part(z), "coord": self.coord(z), "onhand": self.onhand(z).squeeze(-1),
                "src": src, "mask": mask}


def build(npz, **design):
    with np.load(str(npz), allow_pickle=False) as zf:
        verts = torch.tensor(np.asarray(zf["v_template"], np.float32))
        faces = torch.tensor(np.asarray(zf["f"], np.int64))
        part = torch.from_numpy(SM.face_part_table(np.asarray(zf["weights"]), np.asarray(zf["f"], np.int64)))
    return PR2Net(faces, verts, part, **design)


def n_params(net):
    return int(sum(p.numel() for p in net.parameters()))


def save_checkpoint(net, path, **meta):
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "faces": net.faces.cpu(),
                "tmpl": net.tmpl.cpu(), "face_part": net.face_part.cpu(), "kind": "acu_pr2", "design": net.design,
                **meta}, str(path))


def load(path, map_location="cpu"):
    """A PR2 checkpoint, or a PR one through PR's own loader."""
    obj = torch.load(str(path), map_location=map_location)
    if obj.get("kind") == "acu_pr":
        return _native["load"](path, map_location=map_location)
    if obj.get("kind") != "acu_pr2":
        raise ValueError("not an ACU PR / PR2 checkpoint: " + str(path))
    d = obj["design"]
    net = PR2Net(obj["faces"], obj["tmpl"], obj["face_part"], width=d["width"], head=d["head"],
                 sc_hidden=d["sc_hidden"])
    net.load_state_dict(obj["state_dict"], strict=True)
    return net.eval(), obj


def install() -> str:
    PR.build, PR.load, PR.save_checkpoint, PR.n_params = build, load, save_checkpoint, n_params
    return "acu_pr2"


def uninstall() -> None:
    PR.build, PR.load = _native["build"], _native["load"]
    PR.save_checkpoint, PR.n_params = _native["save_checkpoint"], _native["n_params"]


__all__ = ["PR2Net", "RADIUS", "SCALES", "TAU_C_S", "build", "causal_grid_sums", "install", "load", "n_params",
           "save_checkpoint", "uninstall"]
