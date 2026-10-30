"""AURA probe modules (docs/EXPERIMENT_RECORDS.md#aura-prereg-20261024 Stage C).

The instrument is AGG778 R2's: a fresh phi + C1 pooling feeding a fresh C1 downstream (`tools/agg778/heads.Tail`:
bias-free input projection x O, 3 `XYZMeshLayer` rounds on the anchor mesh, LBS slots, 17-group `slot_proj`, fc1, zero
fc2), state-blind (prev48 = 0), residual slot 0 except the "R" arms (rebuilt by the arm's own phi from that arm's
residual mass r = (1 - q) mask). Arms differ only in the association that builds the records; three need a different
node layout and get a derived tail whose common parameters are built first in Tail's order (so under one seed they start
from the same values as Tail's), extras last in a CPU fork:

    SplitTail   surface and near records pooled into two blocks, inp_near a copy of inp
    ShellTail   2V nodes (skin + shell at V + 1 cm n), skin-shell edges, shell nodes with their skin vertex's LBS weights
    JointTail   per-joint pooled phi straight into the 16 joint slots (the coarse 2D baseline, no vertex graph)

`RecordOpX` is AGG778's RecordOp (13-d records -> 64 -> 128) with an optional zero-initialised side input (the ctr flag
of the contour association), built last.
"""
from __future__ import annotations

import torch
from torch import nn

from semkine import aura as AU
from semkine.xyz_mesh import L_REL, Z_REF, Z_SCALE, lbs_slot_pool
from tools.agg778.heads import Tail

V = 778
HIDDEN = 128


# ------------------------------------------------------------------------------------------------ records
def records(assoc, d, mask, verts, tok, shell_pos=None) -> dict:
    """C1's record set (`XYZMeshNet.forward`, geometry on) of any association dict (R records, z (B, N) or (B, N, R)).
    With `shell_pos` (B, 2V, 3) near records live on shell node V + vid and their dXYZ is relative to it."""
    w = assoc["q"].unsqueeze(-1) * assoc["A"] * mask.float().unsqueeze(-1)
    rb, rn, rr = (w > 0).nonzero(as_tuple=True)
    vrec = assoc["vid"][rb, rn, rr]
    wrec = w[rb, rn, rr].float()
    drec = d[rb, rn].float()
    z = assoc["z"]
    zrec = (z[rb, rn] if z.dim() == 2 else z[rb, rn, rr]).float()
    X = zrec.unsqueeze(-1) * drec
    hit = assoc["hit"][rb, rn]
    node = vrec
    if shell_pos is None:
        vx = verts[rb, vrec].float()
    else:
        node = vrec + V * (~hit).long()
        vx = shell_pos[rb, node].float()
    dxyz = (X - vx) / L_REL
    zn = ((zrec - Z_REF) / Z_SCALE).unsqueeze(-1)
    feat = torch.cat([tok[rb, rn].float(), dxyz, zn, torch.ones_like(zn), hit.float().unsqueeze(-1)], -1)
    out = {"rb": rb, "rn": rn, "vrec": vrec, "node": node, "wrec": wrec, "feat": feat, "hit": hit}
    if "ctr" in assoc:
        out["ctr"] = assoc["ctr"][rb, rn, rr].float().unsqueeze(-1)
    return out


def pool(phi, rb, node, wrec, B: int, Vn: int = V):
    """C1 pooling (`tools/agg778/heads.pool_records` mode c1, any node count): (B, Vn, 2C+1) [weighted mean,
    unweighted max, log1p mass], O (B, Vn) binary support."""
    C = phi.shape[-1]
    key = rb * Vn + node
    dev = phi.device
    mass = torch.zeros(B * Vn, device=dev, dtype=torch.float32).index_add(0, key, wrec)
    num = torch.zeros(B * Vn, C, device=dev, dtype=torch.float32).index_add(0, key, wrec.unsqueeze(-1) * phi)
    has = mass > 0
    mean = num / mass.clamp_min(1e-12).unsqueeze(-1) * has.unsqueeze(-1)
    peak = torch.full((B * Vn, C), -1e4, device=dev, dtype=phi.dtype).scatter_reduce(
        0, key.unsqueeze(-1).expand(-1, C), phi, reduce="amax") * has.unsqueeze(-1)
    x = torch.cat([mean, peak, torch.log1p(mass).unsqueeze(-1)], -1)
    return x.view(B, Vn, 2 * C + 1), has.float().view(B, Vn), mass.view(B, Vn)


class RecordOpX(nn.Module):
    """Fresh phi on 13-d records (AGG778 RecordOp's structure and initial values) + an optional zero-initialised,
    bias-free side input into the first layer, built last in a CPU fork."""

    def __init__(self, side: int = 0, hidden: int = HIDDEN, rel_hidden: int = 64):
        super().__init__()
        self.rel = nn.Sequential(nn.Linear(13, rel_hidden), nn.ReLU(inplace=True), nn.Linear(rel_hidden, hidden))
        self.out_dim = 2 * hidden + 1
        self.side_dim = int(side)
        if side:
            with torch.random.fork_rng(devices=[]):
                self.side = nn.Linear(side, rel_hidden, bias=False)
            nn.init.zeros_(self.side.weight)

    def phi(self, rec):
        if self.side_dim:
            return self.rel[2](self.rel[1](self.rel[0](rec["feat"]) + self.side(rec["ctr"])))
        return self.rel(rec["feat"])

    def forward(self, rec, B: int, Vn: int = V, split: bool = False):
        phi = self.phi(rec)
        if not split:
            x, O, _ = pool(phi, rec["rb"], rec["node"], rec["wrec"], B, Vn)
            return x, O
        h = rec["hit"]
        xh, Oh, _ = pool(phi[h], rec["rb"][h], rec["node"][h], rec["wrec"][h], B, Vn)
        xn, On, _ = pool(phi[~h], rec["rb"][~h], rec["node"][~h], rec["wrec"][~h], B, Vn)
        return (xh, xn), torch.maximum(Oh, On)


def residual_slot(phi_net, r, tok, mask):
    """C1's residual summary (r-weighted mean, max, share, has) from residual mass r (B, N) with the arm's own phi;
    geometry, valid and hit are zero for residual records as in `XYZMeshNet.forward`."""
    B = r.shape[0]
    eb, en = (r > 0).nonzero(as_tuple=True)
    tr = tok[eb, en].float()
    feat = torch.cat([tr, tr.new_zeros(tr.shape[0], 6)], -1)
    phi = phi_net(feat).float()
    Cc = phi.shape[-1]
    rw = r[eb, en].float()
    rs = torch.zeros(B, device=r.device).index_add(0, eb, rw)
    rnum = torch.zeros(B, Cc, device=r.device).index_add(0, eb, rw.unsqueeze(-1) * phi)
    rhas = (rs > 0).unsqueeze(-1)
    rmean = rnum / rs.clamp_min(1e-12).unsqueeze(-1) * rhas
    rpeak = torch.full((B, Cc), -1e4, device=r.device).scatter_reduce(
        0, eb.unsqueeze(-1).expand(-1, Cc), phi, reduce="amax") * rhas
    live = mask.float().sum(1).clamp_min(1.0)
    return torch.cat([rmean, rpeak, (rs / live).unsqueeze(-1), rhas.float()], -1)


def occupancy_features(assoc, mask, kind: str):
    """Per-vertex occupancy inputs (no event content): occ = [log1p surface mass, log1p near mass, has];
    occ_tot = [log1p total mass, has]; occ_surf = [log1p surface mass, has_surface] (support = surface only)."""
    m_hit, m_near = AU.occupancy(assoc, mask, V)
    if kind == "occ":
        O = ((m_hit + m_near) > 0).float()
        return torch.stack([torch.log1p(m_hit), torch.log1p(m_near), O], -1), O
    if kind == "occ_tot":
        O = ((m_hit + m_near) > 0).float()
        return torch.stack([torch.log1p(m_hit + m_near), O], -1), O
    if kind == "occ_surf":
        O = (m_hit > 0).float()
        return torch.stack([torch.log1p(m_hit), O], -1), O
    raise ValueError(kind)


def joint_slots(phi, rec, B: int, live):
    """(B, 16, 2C+2) joint slots from joint-binned records: [weighted mean phi, max phi, mass share, has]."""
    x, has, mass = pool(phi, rec["rb"], rec["vrec"], rec["wrec"], B, 16)
    C = phi.shape[-1]
    share = mass / live.clamp_min(1.0).unsqueeze(-1)
    return torch.cat([x[..., :2 * C], share.unsqueeze(-1), has.unsqueeze(-1)], -1)


# ------------------------------------------------------------------------------------------------ tails
def _readout(t, e, bslot, prev48):
    B = e.shape[0]
    slots = torch.cat([e.float(), bslot.unsqueeze(1).float()], 1)
    zs = torch.relu(t.slot_proj(slots.reshape(B, -1, 1))).view(B, -1)
    h = torch.relu(t.fc1(torch.cat([zs, prev48.float()], -1)))
    return t.fc2(h)


class SplitTail(Tail):
    """Two input blocks (surface, near) into one node feature: H = (inp(x_hit) + inp_near(x_near)) O."""

    def __init__(self, f_in: int, xyz, **kw):
        super().__init__(f_in, xyz, **kw)
        with torch.random.fork_rng(devices=[]):
            self.inp_near = nn.Linear(f_in, self.inp.out_features, bias=False)
        with torch.no_grad():
            self.inp_near.weight.copy_(self.inp.weight)

    def forward(self, feat, O, edge_g, bslot, prev48):
        fh, fn = feat
        H = (self.inp(fh) + self.inp_near(fn)) * O.unsqueeze(-1)
        gemb = torch.relu(self.edge(edge_g))
        S = O
        for layer in self.layers:
            H, S = layer(H, S, gemb, self.ring_idx, self.ring_mask)
        return _readout(self, lbs_slot_pool(H, S, O, self.lbs_w, self.lbs_hard), bslot, prev48)


class ShellTail(Tail):
    """2V nodes: skin (surface records) and shell V + 1 cm n (near records); skin 1-ring, shell 1-ring, skin-shell
    edges; edge geometry on the node positions; shell nodes pool into their skin vertex's LBS slots."""

    def __init__(self, f_in: int, xyz, **kw):
        super().__init__(f_in, xyz, **kw)
        idx2, m2 = AU.shell_tables(self.ring_idx, self.ring_mask)
        self.register_buffer("idx2", idx2, persistent=False)
        self.register_buffer("m2", m2, persistent=False)
        self.register_buffer("lbs_w2", torch.cat([self.lbs_w, self.lbs_w], 0), persistent=False)
        self.register_buffer("lbs_hard2", torch.cat([self.lbs_hard, self.lbs_hard], 0), persistent=False)

    def forward(self, feat, O, pos, bslot, prev48):
        H = self.inp(feat) * O.unsqueeze(-1)
        gemb = torch.relu(self.edge(AU.node_edge_geometry(pos, self.idx2, self.m2)))
        S = O
        for layer in self.layers:
            H, S = layer(H, S, gemb, self.idx2, self.m2)
        return _readout(self, lbs_slot_pool(H, S, O, self.lbs_w2, self.lbs_hard2), bslot, prev48)


class JointTail(Tail):
    """The coarse 2D baseline: joint slots straight into the readout (Tail's input projection and graph unused)."""

    def forward(self, jslots, bslot, prev48):
        return _readout(self, jslots, bslot, prev48)
