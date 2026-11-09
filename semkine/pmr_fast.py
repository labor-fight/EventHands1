r"""PMR fast evaluation path (docs/EXPERIMENT_RECORDS.md#pmr-s3c-prereg-20261109): the same function as
semkine/pmr_trunk.py's `PMRNet` in fewer, larger kernels, for the cost readings only. `fastify(net)` patches one network
instance (the module and its classes are unchanged):

  frames      the kinematic chain by tree depth (three batched products instead of fifteen)
  statistics  the raw moment sums of all nodes in one contraction over a stacked per-event vector
              [1, res, P~, P~ (x) res, P~ (x) P~] (P~ = P - J_0), the central moments derived per node as before
  phi         the matched and unmatched event encodings in one call
  propagate   per round one map W = [W_down; W_up; W_hub2part; W_part2hub] (C -> 4C) and one gather / gate / scatter
              over the 62 edges with the per-type mean folded into the scatter matrix
The arithmetic is reassociated, so the outputs equal the reference up to floating-point rounding (checked on CPU in
float64 and on GPU at evaluation precision; tests/test_pmr_fast.py); the readings use the reference implementation.
"""
from __future__ import annotations

import types

import torch

from . import pmr_trunk as PM
from . import xyz_mesh as XM

DEPTH = ((1, 4, 7, 10, 13), (2, 5, 8, 11, 14), (3, 6, 9, 12, 15))


def node_frames_fast(prev48, hands_mean):
    B = prev48.shape[0]
    pose = torch.cat([prev48[:, 0:3], prev48[:, 3:48] + hands_mean.view(1, -1).to(prev48.dtype)], 1).view(B, 16, 3)
    rot = PM.rodrigues(pose)
    G = torch.zeros_like(rot)
    G[:, 0] = rot[:, 0]
    for level in DEPTH:
        idx = list(level)
        par = [PM.PARENT[j] for j in idx]
        G[:, idx] = G[:, par] @ rot[:, idx]
    eye = torch.eye(3, dtype=prev48.dtype, device=prev48.device).expand(B, 1, 3, 3)
    return torch.cat([eye, G[:, list(PM.PARENT[1:])]], 1)


def moments_fast(weight, P, res, origin, frame, ref):
    """`semkine.pmr_trunk.moments` with one contraction for the raw sums; `ref` (B, 1, 3) the expansion point."""
    B, N, _ = P.shape
    Pt = P - ref
    Ot = origin - ref
    one = torch.ones(B, N, 1, dtype=P.dtype, device=P.device)
    stack = torch.cat([one, res, Pt, (Pt.unsqueeze(-1) * res.unsqueeze(-2)).flatten(-2),
                       (Pt.unsqueeze(-1) * Pt.unsqueeze(-2)).flatten(-2)], -1)          # (B, N, 1 + 3 + 3 + 9 + 9)
    S = torch.einsum("bnk,bnf->bkf", weight, stack)
    M, Sr, SP = S[..., 0], S[..., 1:4], S[..., 4:7]
    SPr, SPP = S[..., 7:16].unflatten(-1, (3, 3)), S[..., 16:25].unflatten(-1, (3, 3))
    Mc = M.clamp_min(1e-12)
    Ml = M.unsqueeze(-1)
    M4 = Mc.view(*Mc.shape, 1, 1)
    mr = Sr / Mc.unsqueeze(-1)
    ml = (SP - Ml * Ot) / Mc.unsqueeze(-1)
    T = (SPr - Ot.unsqueeze(-1) * Sr.unsqueeze(-2)) / M4
    Gm = (SPP - Ot.unsqueeze(-1) * SP.unsqueeze(-2) - SP.unsqueeze(-1) * Ot.unsqueeze(-2)
          + Ml.unsqueeze(-1) * Ot.unsqueeze(-1) * Ot.unsqueeze(-2)) / M4
    Ft = frame.transpose(-1, -2)
    mr = torch.einsum("bkij,bkj->bki", Ft, mr) / PM.L_RES
    ml = torch.einsum("bkij,bkj->bki", Ft, ml) / PM.L_LEV
    T = (Ft @ T @ frame) / (PM.L_LEV * PM.L_RES)
    Gm = (Ft @ Gm @ frame) / (PM.L_LEV * PM.L_LEV)
    iu = torch.triu_indices(3, 3, device=Gm.device)
    feats = torch.cat([torch.log1p(M).unsqueeze(-1), mr, ml, T.flatten(-2), Gm[..., iu[0], iu[1]]], -1)
    return M, feats * (M > 0).unsqueeze(-1).to(feats.dtype)


def _geometry(self, mask, d, verts, assoc, prev48, ablate, acc):
    mask_f = mask.to(acc)
    q = assoc["q"].to(acc)
    w = q.unsqueeze(-1) * assoc["A"].to(acc) * mask_f.unsqueeze(-1)
    r = (1.0 - q) * mask_f
    if ablate:
        w = torch.zeros_like(w)
        r = torch.zeros_like(r)
    vf = verts.to(acc)
    m, P, res = PM.event_geometry(w, d.to(acc), assoc["z"].to(acc), assoc["vid"], vf)
    return {"w": w, "r": r, "m": m, "P": P, "res": res, "mask_f": mask_f,
            "omega": PM.part_weights(w, assoc["vid"], self.lbs_w.to(acc)),
            "J": torch.einsum("jv,bvk->bjk", self.jreg.to(acc), vf),
            "F": node_frames_fast(prev48.to(acc), self.hands_mean.to(acc))}


def _statistics(self, g):
    B = g["m"].shape[0]
    J = g["J"]
    ref = J[:, :1]
    W = torch.cat([g["omega"], g["m"].unsqueeze(-1)], -1)                                   # 16 parts + the hand
    eye = torch.eye(3, dtype=J.dtype, device=J.device).expand(B, 1, 3, 3)
    M, feats = moments_fast(W, g["P"], g["res"], torch.cat([J, J[:, :1]], 1), torch.cat([g["F"], eye], 1), ref)
    Mp, sp = M[:, :16], feats[:, :16]
    view = g["F"][..., 2, :]
    sp = torch.cat([sp, view * torch.ones_like(Mp).unsqueeze(-1)], -1)
    return Mp, sp, M[:, 16], feats[:, 16]


def _node_inputs(self, tok, d, mask, verts, assoc, prev48, ablate=False):
    B, N, _ = tok.shape
    dev = tok.device
    acc = torch.float32 if self.training else torch.float64
    with torch.autocast(device_type=dev.type, enabled=False):
        g = self.geometry(mask, d, verts, assoc, prev48, ablate, acc)
        Mp, sp, Mh, sh = self.statistics(g)
        on = (g["m"] > 0).unsqueeze(-1).float()
        zn = ((assoc["z"].float() - XM.Z_REF) / XM.Z_SCALE).unsqueeze(-1)
        f_on = torch.cat([tok.float(), g["res"].float() / PM.L_RES, zn * on, on], -1)
        f_off = torch.cat([tok.float(), torch.zeros(B, N, 5, device=dev)], -1)
    wdt = self.phi[0].weight.dtype
    both = self.phi(torch.cat([f_on, f_off], 1).to(wdt))
    with torch.autocast(device_type=dev.type, enabled=False):
        phi_on, phi_off = both[:, :N].to(acc), both[:, N:].to(acc)
        Phi_p = torch.einsum("bnk,bnc->bkc", g["omega"], phi_on) / Mp.clamp_min(1e-12).unsqueeze(-1)
        Phi_h = torch.einsum("bn,bnc->bc", g["m"], phi_on) / Mh.clamp_min(1e-12).unsqueeze(-1)
        rs = g["r"].sum(1)
        Phi_r = torch.einsum("bn,bnc->bc", g["r"], phi_off) / rs.clamp_min(1e-12).unsqueeze(-1)
        live = g["mask_f"].sum(1).clamp_min(1.0)
        p3 = prev48.to(acc).view(B, PM.N_PARTS, 3)
        x_part = torch.cat([Phi_p, sp, p3], -1)
        x_hub = torch.cat([Phi_h, sh, Phi_r, (rs / live).unsqueeze(-1), (rs > 0).to(acc).unsqueeze(-1), p3[:, 0]], -1)
    return {"x_part": x_part, "x_hub": x_hub, "Mp": Mp, "Mh": Mh, "sp": sp, "sh": sh, **g}


def _propagate(self, h, gate=None, messages=True):
    if not messages:
        return PM.PMRNet.propagate(self, h, gate, messages)
    C = h.shape[-1]
    for rd, Wcat in zip(self.rounds, self._fast_w):
        Mall = (h @ Wcat.to(h.dtype).t()).view(*h.shape[:2], len(PM.EDGE_TYPES), C)        # (B, 17, 4, C)
        msg = gate * Mall[:, self.e_src, self.e_type]                                      # (B, 62, C)
        agg = torch.einsum("ev,bec->bvc", self._fast_scatter.to(msg.dtype), msg)
        h = h + torch.relu(rd.slf(h) + agg)
    return h


def fastify(net):
    """Patch one PMRNet instance (variant r / noprop / mlp) with the fast path; returns it."""
    if not isinstance(net, PM.PMRNet):
        raise TypeError("fastify expects a semkine.pmr_trunk.PMRNet")
    net.geometry = types.MethodType(_geometry, net)
    net.statistics = types.MethodType(_statistics, net)
    net.node_inputs = types.MethodType(_node_inputs, net)
    if net.variant == "r":
        S = torch.zeros(len(net.e_src), PM.N_NODES, device=net.e_src.device)
        for k in range(len(PM.EDGE_TYPES)):
            sel = getattr(net, f"e_sel{k}")
            S[sel] = getattr(net, f"e_scatter{k}")
        net._fast_scatter = S
        net._fast_w = [torch.cat([lin.weight for lin in rd.msg], 0).detach() for rd in net.rounds]
        net.propagate = types.MethodType(_propagate, net)
    return net


__all__ = ["fastify", "moments_fast", "node_frames_fast"]
