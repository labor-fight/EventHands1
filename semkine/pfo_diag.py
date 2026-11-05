r"""PFO diagnostics (docs/EXPERIMENT_RECORDS.md#pfo-prereg-20261103): S45's `XYZMeshNet.forward` with test-time
interventions and stage captures, for frozen trained networks only. Opt-in: `install()` replaces the forward,
`uninstall()` restores it. With no intervention set the outputs equal the native forward exactly
(tests/test_pfo.py).

Interventions (`set_interventions(...)`; each changes one input or stage of the trained network at test time; the
network is not retrained, so a change in error is a co-adapted use measurement, not a capacity statement):
  dxyz_zero      record offsets dXYZ := 0
  dxyz_shuffle   record offsets permuted among the records of the same packet
  zn_const       record depth channel zn := 0 (z = Z_REF)
  tok_xy_mean    record image position (token cols 0, 1) := its packet mean
  tok_time_mean  record timing (token cols 3..6) := their packet means
  tok_pol_zero   record polarity (token col 2) := 0
  agg_mean_only  vertex input: per-vertex max part := 0
  agg_max_only   vertex input: per-vertex mean part := 0
  agg_mass_zero  vertex input: log1p(mass) := 0
  gnn_skip       no message rounds: H3 := H0, S := O
  gnn_rewire     message rounds on a vertex-permuted 1-ring (every vertex keeps its degree, neighbours are random)
  resid_zero     residual slot := 0
  prev_fing_mean readout's previous-state fingers (prev[6:51]) := the training mean pose (root rotation kept)
Captures (`set_capture(True)`): per call, slots (B, 17, 258), slots from H0 with direct support (B, 17, 258), zs, h, delta.
"""
from __future__ import annotations

import torch

from . import xyz_mesh as XM

INTERVENTIONS = ("dxyz_zero", "dxyz_shuffle", "zn_const", "tok_xy_mean", "tok_time_mean", "tok_pol_zero",
                 "agg_mean_only", "agg_max_only", "agg_mass_zero", "gnn_skip", "gnn_rewire", "resid_zero",
                 "prev_fing_mean")
_NATIVE = {"forward": XM.XYZMeshNet.forward}
_STATE = {"on": False, "intervene": frozenset(), "capture": False, "captured": [], "mean_pose": None,
          "gen": None, "perm": None}


def set_interventions(names, mean_pose=None, seed=0):
    names = frozenset(names)
    unknown = names - set(INTERVENTIONS)
    if unknown:
        raise ValueError(f"unknown interventions {sorted(unknown)}")
    if "prev_fing_mean" in names and mean_pose is None:
        raise ValueError("prev_fing_mean needs the training mean pose")
    _STATE.update(intervene=names, mean_pose=None if mean_pose is None else torch.as_tensor(mean_pose).float(),
                  gen=torch.Generator().manual_seed(int(seed)), perm=None)


def set_capture(on=True):
    _STATE.update(capture=bool(on), captured=[])


def captured():
    return _STATE["captured"]


def _packet_mean(values, rows, B):
    """Per-packet mean of `values (R, k)` over the rows of each packet, broadcast back to the rows."""
    s = torch.zeros(B, values.shape[1], device=values.device, dtype=torch.float64).index_add(0, rows, values.double())
    n = torch.zeros(B, device=values.device, dtype=torch.float64).index_add(0, rows, torch.ones_like(rows, dtype=torch.float64))
    return (s / n.clamp_min(1.0).unsqueeze(-1))[rows].to(values.dtype)


def _packet_perm(rows, B):
    """A permutation of record indices that only moves records within their packet (seeded generator)."""
    order = torch.arange(rows.numel(), device=rows.device)
    out = order.clone()
    for b in range(B):
        idx = order[rows == b]
        if idx.numel() > 1:
            p = torch.randperm(idx.numel(), generator=_STATE["gen"]).to(rows.device)
            out[idx] = idx[p]
    return out


def _rewired(net):
    if _STATE["perm"] is None:
        V = net.ring_idx.shape[0]
        _STATE["perm"] = torch.randperm(V, generator=_STATE["gen"])
    perm = _STATE["perm"].to(net.ring_idx.device)
    idx = torch.where(net.ring_mask > 0, perm[net.ring_idx], net.ring_idx)
    return idx


def forward(self, tok, d, mask, verts, assoc, prev48, ablate: bool = False, verts_geo=None, capture: bool = False,
            h3_transform=None):
    """`XYZMeshNet.forward` (sign / XAG H3 transform paths excluded) with the module's interventions and captures."""
    if verts_geo is not None or self.sign or h3_transform is not None or self.xag_h3_transform is not None:
        raise ValueError("pfo_diag covers the S45 / S45-SM forward only")
    iv = _STATE["intervene"]
    B, N, _ = tok.shape
    V = verts.shape[1]
    C = self.hidden
    dev = tok.device
    wdt = self.rel[0].weight.dtype
    acc = torch.float32 if self.training else torch.float64
    mask_f = mask.to(torch.float32)
    w = assoc["q"].unsqueeze(-1) * assoc["A"] * mask_f.unsqueeze(-1)
    r = (1.0 - assoc["q"]) * mask_f
    if ablate:
        w = torch.zeros_like(w)
        r = torch.zeros_like(r)
    rb, rn, rr = (w > 0).nonzero(as_tuple=True)
    vrec = assoc["vid"][rb, rn, rr]
    wrec = w[rb, rn, rr]
    drec = d[rb, rn].float()
    zrec = assoc["z"][rb, rn]
    vxyz = verts[rb, vrec].float()
    X = zrec.unsqueeze(-1) * drec
    dxyz = (X - vxyz) / XM.L_REL
    zn = ((zrec - XM.Z_REF) / XM.Z_SCALE).unsqueeze(-1)
    trec = tok[rb, rn].float()
    if iv:
        if "dxyz_zero" in iv:
            dxyz = torch.zeros_like(dxyz)
        if "dxyz_shuffle" in iv:
            dxyz = dxyz[_packet_perm(rb, B)]
        if "zn_const" in iv:
            zn = torch.zeros_like(zn)
        if {"tok_xy_mean", "tok_time_mean", "tok_pol_zero"} & iv:
            trec = trec.clone()
            if "tok_xy_mean" in iv:
                trec[:, 0:2] = _packet_mean(trec[:, 0:2], rb, B)
            if "tok_time_mean" in iv:
                trec[:, 3:7] = _packet_mean(trec[:, 3:7], rb, B)
            if "tok_pol_zero" in iv:
                trec[:, 2] = 0.0
    cols = [trec, dxyz, zn, torch.ones_like(zn), assoc["hit"][rb, rn].float().unsqueeze(-1)]
    feat = torch.cat(cols, dim=-1)
    phi = self.rel(feat.to(wdt))
    key = rb * V + vrec
    mass = torch.zeros(B * V, device=dev, dtype=acc).index_add(0, key, wrec.to(acc))
    num = torch.zeros(B * V, C, device=dev, dtype=acc).index_add(0, key, wrec.to(acc).unsqueeze(-1) * phi.to(acc))
    has = mass > 0
    mean = (num / mass.clamp_min(1e-12).unsqueeze(-1)).to(torch.float32) * has.unsqueeze(-1)
    peak = torch.full((B * V, C), -1e4, device=dev, dtype=torch.float32).scatter_reduce(
        0, key.unsqueeze(-1).expand(-1, C), phi.float(), reduce="amax")
    peak = peak * has.unsqueeze(-1)
    lmass = torch.log1p(mass.to(torch.float32)).unsqueeze(-1)
    if "agg_mean_only" in iv:
        peak = torch.zeros_like(peak)
    if "agg_max_only" in iv:
        mean = torch.zeros_like(mean)
    if "agg_mass_zero" in iv:
        lmass = torch.zeros_like(lmass)
    x = torch.cat([mean, peak, lmass], -1)
    O = has.view(B, V).to(torch.float32)
    H = self.vin(x.view(B, V, 2 * C + 1).to(wdt))
    H0 = H
    S = O
    if "gnn_skip" not in iv:
        ring = _rewired(self) if "gnn_rewire" in iv else self.ring_idx
        if "gnn_rewire" in iv:
            vf = verts.float()
            K = ring.shape[1]
            dv = vf[:, ring.reshape(-1)].view(B, V, K, 3) - vf.unsqueeze(2)
            g = torch.cat([dv, dv.norm(dim=-1, keepdim=True)], dim=-1) / XM.L_EDGE * self.ring_mask.view(1, V, K, 1)
        else:
            g = self.edge_geometry(verts)
        gemb = torch.relu(self.edge(g.to(wdt)))
        for layer in self.layers:
            H, S = layer(H, S, gemb, ring, self.ring_mask)
    e = XM.lbs_slot_pool(H, S, O, self.lbs_w, self.lbs_hard)
    eb, en = (r > 0).nonzero(as_tuple=True)
    tr = tok[eb, en].float()
    feat_r = torch.cat([tr, torch.zeros(tr.shape[0], self.rel_in - tr.shape[1], device=dev)], dim=-1)
    phi_r = self.rel(feat_r.to(wdt))
    rw = r[eb, en]
    rs = torch.zeros(B, device=dev, dtype=acc).index_add(0, eb, rw.to(acc))
    rnum = torch.zeros(B, C, device=dev, dtype=acc).index_add(0, eb, rw.to(acc).unsqueeze(-1) * phi_r.to(acc))
    rhas = (rs > 0).unsqueeze(-1)
    rmean = (rnum / rs.clamp_min(1e-12).unsqueeze(-1)).to(torch.float32) * rhas
    rpeak = torch.full((B, C), -1e4, device=dev, dtype=torch.float32).scatter_reduce(
        0, eb.unsqueeze(-1).expand(-1, C), phi_r.float(), reduce="amax") * rhas
    live = mask_f.sum(1).clamp_min(1.0)
    bslot = torch.cat([rmean, rpeak, (rs.to(torch.float32) / live).unsqueeze(-1), rhas.to(torch.float32)], dim=-1)
    if "resid_zero" in iv:
        bslot = torch.zeros_like(bslot)
    slots = torch.cat([e.to(torch.float32), bslot.unsqueeze(1)], dim=1)
    if "prev_fing_mean" in iv:
        prev48 = prev48.clone()
        prev48[:, 3:48] = _STATE["mean_pose"].to(prev48.device, prev48.dtype)
    zs = torch.relu(self.slot_proj(slots.reshape(B, -1, 1).to(wdt))).view(B, -1)
    h = torch.relu(self.fc1(torch.cat([zs, prev48.to(zs.dtype)], dim=-1)))
    delta = self.fc2(h)
    if _STATE["capture"]:
        e0 = XM.lbs_slot_pool(H0, O, O, self.lbs_w, self.lbs_hard)
        slots0 = torch.cat([e0.to(torch.float32), bslot.unsqueeze(1)], dim=1)
        _STATE["captured"].append({k: v.detach().float().cpu() for k, v in
                                   {"slots": slots, "slots_h0": slots0, "zs": zs, "h": h, "delta": delta,
                                    "n_direct": O.sum(1), "n_prop": S.sum(1)}.items()})
    n_live = mask_f.sum()
    stats = {"xyz_valid": assoc["valid"].float().sum() / n_live.clamp_min(1.0),
             "xyz_hit": assoc["hit"].float().sum() / n_live.clamp_min(1.0),
             "xyz_resid_mass": r.sum() / n_live.clamp_min(1.0), "xyz_verts_direct": O.sum(1).mean(),
             "xyz_verts_prop": S.sum(1).mean()}
    return delta, stats, {"slots": slots, "O": O, "S": S, "w": w, "r": r}


def install():
    if XM.XYZMeshNet.forward is not _NATIVE["forward"]:
        raise RuntimeError("XYZMeshNet.forward is already patched")
    XM.XYZMeshNet.forward = forward
    _STATE["on"] = True


def uninstall():
    XM.XYZMeshNet.forward = _NATIVE["forward"]
    _STATE.update(on=False, intervene=frozenset(), capture=False, captured=[], perm=None)


__all__ = ["INTERVENTIONS", "captured", "forward", "install", "set_capture", "set_interventions", "uninstall"]
