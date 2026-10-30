"""AGG778 probe modules (docs/EXPERIMENT_RECORDS.md#agg778-prereg-20261024 Stage B).

* `Tail`: a fresh instance of C1's own downstream (input projection without bias, 3 `XYZMeshLayer` rounds on the fixed
  1-ring, `lbs_slot_pool`, the 17-group `slot_proj`, `fc1(+prev48)`, zero-initialised `fc2`). Every task-level arm
  gets one, so arms differ only in the per-vertex content they feed it.
* `RecordOp`: C1's record -> vertex operator with a fresh phi (records 13-d as `XYZMeshNet.forward` L527-569), plus the
  three single-factor variants of the plan: `matpt` (records also carry A_k and the material-point in-plane offset),
  `mass` (mass-consistent pooling: weighted mean, weighted RMS in place of the unweighted max, soft support
  min(mass, 1)), `fanout` (each event spread to its m nearest projected vertices inside the association band).
* `KernelOp`: the XVM point-set kernel features (gather within rho = f r / z, pair features, mean / max / log count).
* `VertexHead`: the two per-vertex readouts (per-vertex MLP; per-vertex head after 3 fresh `XYZMeshLayer` rounds),
  output m (2, vertex frame / rho), L (3, information Cholesky) and the 3-D displacement (3, / L_REL).
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from semkine.xyz_mesh import L_REL, Z_REF, Z_SCALE, XYZMeshLayer, lbs_slot_pool

V = 778
FANOUT_M = 8
FANOUT_SIGMA_PX = 8.0


# ------------------------------------------------------------------------------------------------ task level
class Tail(nn.Module):
    def __init__(self, f_in: int, xyz, hidden: int = 128, layers: int = 3, edge_dim: int = 32, slot_dim: int = 32,
                 readout: int = 256):
        super().__init__()
        self.inp = nn.Linear(f_in, hidden, bias=False)
        self.edge = nn.Linear(4, edge_dim)
        self.layers = nn.ModuleList([XYZMeshLayer(hidden, edge_dim) for _ in range(layers)])
        n_slots = xyz.N_SLOTS
        self.slot_proj = nn.Conv1d(n_slots * (2 * hidden + 2), n_slots * slot_dim, 1, groups=n_slots)
        self.fc1 = nn.Linear(n_slots * slot_dim + 48, readout)
        self.fc2 = nn.Linear(readout, 51)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.register_buffer("ring_idx", xyz.ring_idx.clone(), persistent=False)
        self.register_buffer("ring_mask", xyz.ring_mask.clone(), persistent=False)
        self.register_buffer("lbs_w", xyz.lbs_w.clone(), persistent=False)
        self.register_buffer("lbs_hard", xyz.lbs_hard.clone(), persistent=False)

    def forward(self, feat, O, edge_g, bslot, prev48):
        B = feat.shape[0]
        H = self.inp(feat) * O.unsqueeze(-1)
        gemb = torch.relu(self.edge(edge_g))
        S = O
        for layer in self.layers:
            H, S = layer(H, S, gemb, self.ring_idx, self.ring_mask)
        e = lbs_slot_pool(H, S, O, self.lbs_w, self.lbs_hard)
        slots = torch.cat([e.float(), bslot.unsqueeze(1).float()], 1)
        zs = torch.relu(self.slot_proj(slots.reshape(B, -1, 1))).view(B, -1)
        h = torch.relu(self.fc1(torch.cat([zs, prev48.float()], -1)))
        return self.fc2(h)


# ------------------------------------------------------------------------------------------------ records
def c1_records(cap: dict) -> dict:
    """The C1 record set of a capture (exactly `XYZMeshNet.forward` L521-547 with geometry on): batch / vertex index,
    weight q A, 13-d feature, plus the A_k and material-point quantities the `matpt` variant adds."""
    assoc, d, mask, verts, tok = cap["assoc"], cap["d"], cap["mask"], cap["verts"], cap["tok"]
    w = assoc["q"].unsqueeze(-1) * assoc["A"] * mask.float().unsqueeze(-1)
    rb, rn, rr = (w > 0).nonzero(as_tuple=True)
    vrec = assoc["vid"][rb, rn, rr]
    wrec = w[rb, rn, rr].float()
    drec = d[rb, rn].float()
    zrec = assoc["z"][rb, rn].float()
    X = zrec.unsqueeze(-1) * drec
    vx = verts[rb, vrec].float()
    dxyz = (X - vx) / L_REL
    zn = ((zrec - Z_REF) / Z_SCALE).unsqueeze(-1)
    feat = torch.cat([tok[rb, rn].float(), dxyz, zn, torch.ones_like(zn), assoc["hit"][rb, rn].float().unsqueeze(-1)], -1)
    # material point of the event on the anchor mesh: sum_j A_j V_j over the event's own records
    A = assoc["A"].float() * (assoc["q"].unsqueeze(-1) > 0).float()
    An = A / A.sum(-1, keepdim=True).clamp_min(1e-12)
    Vr = torch.gather(verts.float().unsqueeze(1).expand(-1, d.shape[1], -1, -1), 2,
                      assoc["vid"].unsqueeze(-1).expand(-1, -1, -1, 3))                   # (B, N, 4, 3)
    Xp = (An.unsqueeze(-1) * Vr).sum(2)                                                   # (B, N, 3)
    inplane = (Xp[rb, rn] - vx) / L_REL                                                  # vertex -> material point
    akn = An[rb, rn, rr].unsqueeze(-1)
    return {"rb": rb, "rn": rn, "vrec": vrec, "wrec": wrec, "feat": feat, "inplane": inplane, "ak": akn}


@torch.no_grad()
def fanout_records(cap: dict, geo: dict, band_px: float = 16.0, m: int = FANOUT_M,
                   sigma_px: float = FANOUT_SIGMA_PX) -> dict:
    """Each live event with q > 0 spread to its m nearest projected anchor vertices within `band_px` (Gaussian weights
    renormalised to the event's q), records built as in C1 (z = the association's z, dXYZ to each vertex)."""
    assoc, d, mask, verts, tok = cap["assoc"], cap["d"], cap["mask"], cap["verts"], cap["tok"]
    B, N, _ = d.shape
    fx, fy, cx, cy = geo["fx"], geo["fy"], geo["cx"], geo["cy"]
    ex = torch.stack([d[..., 0] * fx[:, None] + cx[:, None], d[..., 1] * fy[:, None] + cy[:, None]], -1)  # event px
    d2 = torch.cdist(ex.float(), geo["uv"].float()).square()                               # (B, N, V)
    val, idx = d2.topk(m, dim=-1, largest=False)
    ok = (val <= band_px ** 2) & mask.unsqueeze(-1) & (assoc["q"] > 0).unsqueeze(-1)
    k = torch.exp(-val / (2 * sigma_px ** 2)) * ok.float()
    w = k / k.sum(-1, keepdim=True).clamp_min(1e-12) * assoc["q"].unsqueeze(-1).float()
    rb, rn, rr = (w > 0).nonzero(as_tuple=True)
    vrec = idx[rb, rn, rr]
    wrec = w[rb, rn, rr]
    zrec = assoc["z"][rb, rn].float()
    X = zrec.unsqueeze(-1) * d[rb, rn].float()
    dxyz = (X - verts[rb, vrec].float()) / L_REL
    zn = ((zrec - Z_REF) / Z_SCALE).unsqueeze(-1)
    feat = torch.cat([tok[rb, rn].float(), dxyz, zn, torch.ones_like(zn), assoc["hit"][rb, rn].float().unsqueeze(-1)], -1)
    return {"rb": rb, "rn": rn, "vrec": vrec, "wrec": wrec, "feat": feat}


def pool_records(phi, rb, vrec, wrec, B: int, mode: str = "c1"):
    """C1 pooling (weighted mean, unweighted max, log1p mass, binary support) or the mass-consistent variant
    (weighted mean, weighted RMS, log1p mass, soft support min(mass, 1)). Returns x (B, V, 2C+1), O (B, V)."""
    C = phi.shape[-1]
    key = rb * V + vrec
    dev = phi.device
    mass = torch.zeros(B * V, device=dev, dtype=torch.float32).index_add(0, key, wrec)
    num = torch.zeros(B * V, C, device=dev, dtype=torch.float32).index_add(0, key, wrec.unsqueeze(-1) * phi)
    has = mass > 0
    mean = num / mass.clamp_min(1e-12).unsqueeze(-1) * has.unsqueeze(-1)
    if mode == "c1":
        peak = torch.full((B * V, C), -1e4, device=dev, dtype=phi.dtype).scatter_reduce(
            0, key.unsqueeze(-1).expand(-1, C), phi, reduce="amax") * has.unsqueeze(-1)
        O = has.float()
    elif mode == "mass":
        sq = torch.zeros(B * V, C, device=dev, dtype=torch.float32).index_add(0, key, wrec.unsqueeze(-1) * phi * phi)
        peak = torch.sqrt((sq / mass.clamp_min(1e-12).unsqueeze(-1)).clamp_min(1e-12)) * has.unsqueeze(-1)
        O = mass.clamp(max=1.0)
    else:
        raise ValueError(mode)
    x = torch.cat([mean, peak, torch.log1p(mass).unsqueeze(-1)], -1)
    return x.view(B, V, 2 * C + 1), O.view(B, V)


class RecordOp(nn.Module):
    """Fresh phi + pooling on a record set; `extra` adds the matpt inputs (A_k, in-plane offset)."""

    def __init__(self, extra: bool = False, pool: str = "c1", hidden: int = 128, rel_hidden: int = 64):
        super().__init__()
        self.extra, self.pool = extra, pool
        self.rel = nn.Sequential(nn.Linear(13 + (4 if extra else 0), rel_hidden), nn.ReLU(inplace=True),
                                 nn.Linear(rel_hidden, hidden))
        self.out_dim = 2 * hidden + 1

    def forward(self, rec: dict, B: int):
        f = rec["feat"]
        if self.extra:
            f = torch.cat([f, rec["ak"], rec["inplane"]], -1)
        phi = self.rel(f)
        return pool_records(phi, rec["rb"], rec["vrec"], rec["wrec"], B, self.pool)


# ------------------------------------------------------------------------------------------------ kernel
class KernelOp(nn.Module):
    """XVM point-set kernel features: pair features of the <= k events within rho = f r / z of every visible vertex
    (`semkine.xvm.gather` / `pair_features`), a 2-layer kernel, then mean / max / log1p(count)."""

    def __init__(self, r_m: float = 0.01, k: int = 64, hidden: int = 64):
        super().__init__()
        from semkine import xvm as XV
        self.r, self.k = r_m, k
        self.kernel = nn.Sequential(nn.Linear(XV.PAIR_IN, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden),
                                    nn.ReLU(inplace=True))
        self.out_dim = 2 * hidden + 1

    def forward(self, geo: dict, ev: dict):
        from semkine import xvm as XV
        rho = (geo["f"][:, None] * self.r / geo["z"].clamp(min=1e-3)).float()
        idx, ok, _ = XV.gather(geo["uv"], geo["vis"], rho, ev["xy"], ev["mask"], self.k)
        feats = XV.pair_features(geo["uv"], geo["R"], rho, geo["cos"], ev, idx, ok, 50.0, ev["in_sil"])
        h = self.kernel(feats)
        okf = ok.unsqueeze(-1).to(h.dtype)
        cnt = okf.sum(2)
        mean = (h * okf).sum(2) / cnt.clamp(min=1.0)
        mx = h.masked_fill(~ok.unsqueeze(-1), 0.0).amax(2)
        return torch.cat([mean, mx, torch.log1p(cnt)], -1), (cnt.squeeze(-1) > 0).float()


# ------------------------------------------------------------------------------------------------ vertex heads
class VertexHead(nn.Module):
    """Per-vertex readout on (B, V, F) features + |n.d| + support: `mlp` (per vertex) or `gnn` (3 fresh rounds)."""

    def __init__(self, f_in: int, readout: str, xyz=None, hidden: int = 128, edge_dim: int = 32):
        super().__init__()
        self.readout = readout
        if readout == "mlp":
            self.net = nn.Sequential(nn.Linear(f_in + 2, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden),
                                     nn.ReLU(inplace=True), nn.Linear(hidden, 8))
        elif readout == "gnn":
            self.inp = nn.Linear(f_in, hidden, bias=False)
            self.edge = nn.Linear(4, edge_dim)
            self.layers = nn.ModuleList([XYZMeshLayer(hidden, edge_dim) for _ in range(3)])
            self.net = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 8))
            self.register_buffer("ring_idx", xyz.ring_idx.clone(), persistent=False)
            self.register_buffer("ring_mask", xyz.ring_mask.clone(), persistent=False)
        else:
            raise ValueError(readout)

    def forward(self, feat, O, cos, edge_g=None):
        side = torch.stack([cos.float(), (O > 0).float()], -1)
        if self.readout == "mlp":
            o = self.net(torch.cat([feat, side], -1))
        else:
            H = self.inp(feat) * O.unsqueeze(-1)
            gemb = torch.relu(self.edge(edge_g))
            S = O
            for layer in self.layers:
                H, S = layer(H, S, gemb, self.ring_idx, self.ring_mask)
            o = self.net(torch.cat([H, side], -1))
        m = o[..., :2]
        L = torch.zeros(*o.shape[:-1], 2, 2, device=o.device, dtype=o.dtype)
        L[..., 0, 0] = F.softplus(o[..., 2]) + 1e-3
        L[..., 1, 1] = F.softplus(o[..., 3]) + 1e-3
        L[..., 1, 0] = o[..., 4]
        return m, L, o[..., 5:8]
