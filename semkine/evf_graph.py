"""EVF graph version B (docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104 starting point B; registered with its round).

Level 0 = the point version's nodes and features (semkine/evf_nodes.py), a shared MLP. Levels 1..3 pool them into
coarser space-time cells (CELLS: pixels, time bins over Delta_t; polarity merged), every level:
  pool      max of the level below over the members of a cell (a node per non-empty cell), member count; the node sits
            at the members' centroid (mean pixel, mean t / Delta_t)
  inject    the virtual frame read at the rounded centroid (the 21 node features of semkine/evf_render.py `sample`, the
            centroid relative to the projected root in hand-scale units) and at the cell's own scale (the fraction of
            the cell inside the predicted silhouette and on a predicted contour, from box-averaged maps): multi-scale
            virtual-frame features and relative quantities at every level
  graph     edges to the nodes of the 3 x 3 neighbouring cells of the same time bin and of the same cell in the
            adjacent time bins (a space-time radius of one cell, 11 edges at most incl. the self edge); edge features =
            relative displacement / cell size and relative time only; message
            m_ij = relu(A h_i + B h_j + C e_ij + c), h_i <- h_i + W max_j m_ij (self included)
The readout is A's (17 soft queries, packet statistics, prev48, the projective translation head), over the nodes of
levels 1..3 with a learned level embedding. The previous state enters through the virtual-frame readings only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from semkine import evf_nodes as EN
from semkine.evf_render import FAR_PX, sample

#: (cell size in pixels, time bins over Delta_t) of levels 1..3
CELLS = ((6, 5), (12, 5), (24, 1))
VF_DIM = 2 + 21 + 2            # centroid relative position, virtual-frame sample, cell-scale occupancy
EDGE_DIM = 3


def _vf_parts(smp, rx, ry, s):
    ds, dc = smp["ds"], smp["dc"]
    return [torch.stack([rx, ry], -1), smp["inside"].float()[:, None], (ds / s)[:, None],
            (ds / EN.PX_SCALE)[:, None], smp["dir_s"], (dc / s)[:, None], (dc / EN.PX_SCALE)[:, None], smp["dir_c"],
            (smp["dz_s"] / 0.1)[:, None], smp["tmpl_s"] / 0.1, (smp["dz_c"] / 0.1)[:, None], smp["tmpl_c"] / 0.1,
            torch.stack([smp["valid_s"].float(), smp["valid_c"].float(), smp["far_s"].float(),
                         smp["far_c"].float()], -1)]


@torch.no_grad()
def build_levels(events, src, mask, delta_t_s, frame, template, H: int, W: int, vf: bool = True):
    """The pooling structure and per-level inputs (no gradient). Returns a list over levels 1..3 of dicts:
    parent (B, N_{l-1}) long index into level l (-1 at padded slots), mask (B, N_l), nbr (B, N_l, K) long (-1 none),
    edge (B, N_l, K, 3), vf (B, N_l, VF_DIM or 2), count (B, N_l)."""
    B, N0 = src.shape
    dev = src.device
    flat = mask.reshape(-1).nonzero().squeeze(1)
    rows = src.reshape(-1)[flat]
    b0 = torch.div(flat, N0, rounding_mode="floor")
    px = events[rows, 1].float()
    py = events[rows, 2].float()
    tt = (events[rows, 3].float() / delta_t_s.float().clamp_min(1e-6)[b0]).clamp(0.0, 1.0 - 1e-6)
    sil = cont = None
    if vf:
        sil = (frame.face >= 0).float()[:, None]
        cont = frame.contour.float()[:, None]
    levels = []
    child_slot = flat                                   # flat padded index of each live L0 node in the level below
    child_b = b0
    child_n = N0
    for c, nb in CELLS:
        gh, gw = (H + c - 1) // c, (W + c - 1) // c
        tb = (tt * nb).floor().long()
        key = ((b0 * nb + tb) * gh + (py.long() // c)) * gw + (px.long() // c)
        ukey, inv = torch.unique(key, return_inverse=True)
        n_cl = int(ukey.numel())
        cb = torch.div(ukey, nb * gh * gw, rounding_mode="floor")
        count_b = torch.bincount(cb, minlength=B)
        start = torch.cumsum(count_b, 0) - count_b
        rank = torch.arange(n_cl, device=dev) - start[cb]
        N = max(int(count_b.max()), 1)
        # centroids and member counts per cluster
        ones = torch.ones_like(px)
        cnt = torch.zeros(n_cl, device=dev).index_add_(0, inv, ones)
        cx = torch.zeros(n_cl, device=dev).index_add_(0, inv, px) / cnt
        cy = torch.zeros(n_cl, device=dev).index_add_(0, inv, py) / cnt
        ct = torch.zeros(n_cl, device=dev).index_add_(0, inv, tt) / cnt
        # the parent of every node of the level below (via its L0 members; all members of a child share its cell)
        parent = torch.full((B * child_n,), -1, dtype=torch.long, device=dev)
        parent[child_slot] = rank[inv]
        lvl_mask = torch.zeros(B, N, dtype=torch.bool, device=dev)
        lvl_mask[cb, rank] = True
        # neighbours: the clusters of the 3 x 3 cells in the same / adjacent time bins
        grid = torch.full((B, nb + 2, gh + 2, gw + 2), -1, dtype=torch.long, device=dev)
        ctb = torch.div(ukey, gh * gw, rounding_mode="floor") % nb
        cgy = torch.div(ukey, gw, rounding_mode="floor") % gh
        cgx = ukey % gw
        grid[cb, ctb + 1, cgy + 1, cgx + 1] = rank
        # the 3 x 3 cells of the same time bin and the same cell in the adjacent bins (a space-time radius of one cell)
        offs = [(0, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)] + ([(-1, 0, 0), (1, 0, 0)] if nb > 1 else [])
        nbr_c = torch.stack([grid[cb, ctb + 1 + dt, cgy + 1 + dy, cgx + 1 + dx] for dt, dy, dx in offs], -1)
        K = len(offs)
        nbr = torch.full((B, N, K), -1, dtype=torch.long, device=dev)
        nbr[cb, rank] = nbr_c
        pos = torch.zeros(B, N, 3, device=dev)
        pos[cb, rank] = torch.stack([cx / c, cy / c, ct], -1)
        npos = pos[torch.arange(B, device=dev)[:, None, None], nbr.clamp_min(0)]
        edge = (npos - pos.unsqueeze(2)) * (nbr >= 0).unsqueeze(-1)
        # virtual-frame readings at the centroid and at the cell's scale
        cnt_pad = torch.zeros(B, N, device=dev)
        cnt_pad[cb, rank] = cnt
        if vf:
            ipx = cx.round().long().clamp(0, W - 1)
            ipy = cy.round().long().clamp(0, H - 1)
            s = frame.scale[cb]
            smp = sample(frame, cb, ipx, ipy, template, frame.root_z)
            rx, ry = (cx - frame.root_u[cb]) / s, (cy - frame.root_v[cb]) / s
            occ_s = F.avg_pool2d(sil, c, c, ceil_mode=True, count_include_pad=False)[:, 0]
            occ_c = F.avg_pool2d(cont, c, c, ceil_mode=True, count_include_pad=False)[:, 0]
            multi = torch.stack([occ_s[cb, cgy, cgx], occ_c[cb, cgy, cgx]], -1)
            vfeat = torch.cat(_vf_parts(smp, rx, ry, s) + [multi], -1)
        else:
            vfeat = torch.stack([cx / W, cy / H], -1)
        vf_pad = torch.zeros(B, N, vfeat.shape[-1], device=dev)
        vf_pad[cb, rank] = vfeat
        levels.append({"parent": parent.view(B, child_n), "mask": lvl_mask, "nbr": nbr, "edge": edge, "vf": vf_pad,
                       "count": cnt_pad})
        child_slot = cb * N + rank
        child_slot = child_slot[inv]                    # per L0 node: its cluster's flat slot at this level
        child_n = N
    return levels


class GraphLevel(nn.Module):
    def __init__(self, d_in, vf_dim, d):
        super().__init__()
        self.inj = nn.Linear(d_in + vf_dim + 1, d)
        self.a = nn.Linear(d, d)
        self.b = nn.Linear(d, d, bias=False)
        self.c = nn.Linear(EDGE_DIM, d, bias=False)
        self.w = nn.Linear(d, d)

    def forward(self, h_child, lvl):
        """Pool the level below (`h_child (B, N_child, d_in)`), inject, one graph round."""
        B, Nc, D = h_child.shape
        parent = lvl["parent"]
        N = lvl["mask"].shape[1]
        live = parent >= 0
        idx = parent.clamp_min(0)
        pooled = torch.full((B, N, D), -1e4, device=h_child.device, dtype=h_child.dtype)
        src = h_child.masked_fill(~live.unsqueeze(-1), -1e4)
        pooled = pooled.scatter_reduce(1, idx.unsqueeze(-1).expand(B, Nc, D), src, reduce="amax")
        m = lvl["mask"].unsqueeze(-1)
        pooled = pooled.masked_fill(~m, 0.0)
        x = torch.cat([pooled, lvl["vf"].to(pooled.dtype), torch.log1p(lvl["count"]).unsqueeze(-1).to(pooled.dtype)],
                      -1)
        h = torch.relu(self.inj(x)) * m
        nbr = lvl["nbr"]
        K = nbr.shape[-1]
        ok = (nbr >= 0) & m
        bh = self.b(h)
        gathered = bh[torch.arange(B, device=h.device)[:, None, None], nbr.clamp_min(0)]      # (B, N, K, d)
        msg = torch.relu(self.a(h).unsqueeze(2) + gathered + self.c(lvl["edge"].to(h.dtype)))
        msg = msg.masked_fill(~ok.unsqueeze(-1), 0.0).amax(2)
        return (h + self.w(msg)) * m


class EVFGraphNet(nn.Module):
    def __init__(self, node_dim: int, packet_dim: int, hidden: int = 48, slot_dim: int = 32, readout_hidden: int = 256,
                 prev48: bool = True, vf: bool = True):
        super().__init__()
        from semkine.evf_net import N_SLOTS
        self.vf = bool(vf)
        self.phi = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden),
                                 nn.ReLU(inplace=True))
        vfd = VF_DIM if self.vf else 2
        self.levels = nn.ModuleList([GraphLevel(hidden, vfd, hidden) for _ in CELLS])
        self.level_emb = nn.Parameter(torch.zeros(len(CELLS), hidden))
        self.att = nn.Linear(hidden, N_SLOTS)
        self.val = nn.Linear(hidden, int(slot_dim))
        self.prev48 = bool(prev48)
        self.fc1 = nn.Linear(N_SLOTS * int(slot_dim) + N_SLOTS + int(packet_dim) + (48 if self.prev48 else 0),
                             int(readout_hidden))
        self.fc2 = nn.Linear(int(readout_hidden), 51)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, feat, mask, levels, packet, prev48=None):
        from semkine.evf_net import MASK_FILL
        h = self.phi(feat) * mask.unsqueeze(-1)
        hs, ms = [], []
        for k, (layer, lvl) in enumerate(zip(self.levels, levels)):
            h = layer(h, lvl)
            hs.append(h + self.level_emb[k].to(h.dtype))
            ms.append(lvl["mask"])
        hh, mm = torch.cat(hs, 1), torch.cat(ms, 1)
        logit = self.att(hh).float()
        value = self.val(hh).float()
        w = torch.softmax(logit.masked_fill(~mm.unsqueeze(-1), MASK_FILL), dim=1) * mm.unsqueeze(-1)
        w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
        slots = torch.einsum("bnk,bnd->bkd", w, value)
        mass = torch.log1p((torch.sigmoid(logit) * mm.unsqueeze(-1)).sum(1))
        parts = [slots.flatten(1), mass, packet.float()]
        if self.prev48:
            parts.append(prev48.float())
        return self.fc2(torch.relu(self.fc1(torch.cat(parts, -1)))).float()


__all__ = ["CELLS", "EVFGraphNet", "FAR_PX", "GraphLevel", "VF_DIM", "build_levels"]
