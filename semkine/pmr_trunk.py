r"""PMR (redesign of the shared-gate propagation module P, docs/EXPERIMENT_RECORDS.md#pmr-prereg-20261109): a part-level
trunk for B2's recipe in which the only path from one part's evidence to another joint's increment is the propagation
module P, opt-in replacement of `semkine.xyz_mesh.XYZMeshNet` (`install(variant)`).

B2 (PFO2 `fgs`) pools its vertex features into 17 slots and reads the 51-D increment with one MLP over the concatenated
slots; that MLP already moves every part's evidence to every joint, so its vertex MeshGNN ablates to zero. Here the graph
is the kinematic level, where cross-part information exists (gshare probe, tree_flow.md: own part 6.20 -> all parts
4.68 mm), and the readout is local:

    events (B2's routes: r_s2d + PFA) -> per-event residual res_i = z_i d_i - P_i (camera frame; P_i = routed point on
        the prior mesh) and soft part weights omega_ip = q_i sum_r A_ir W[v_ir, p] (LBS weights of the routed point)
    node p (16 parts):  x_p = [ mean_i omega_ip phi(e_i) (learned, C) | moments of (lever, residual) about the prior
                        joint J_p, in the prior frame of p's parent, and the view axis in that frame (25) |
                        prev48 slice of joint p (3) ]
    hand node (hub):    x_h = [ mean over on-hand events of phi (C) | the same moments of the whole hand about J_0 (22) |
                        residual slot: mean phi of the unmatched events (C), their share, presence | root slice (3) ]
    h^0 = per-node linear encoders
    P (variant "r"):    L rounds  h_v <- h_v + relu(W_s h_v + sum_t mean_{u -t-> v} g_uv * W_t h_u),
                        t in {parent->child, child->parent, hub->part, part->hub}; the gate g_uv = sigmoid of an
                        embedding of the prior FK geometry of the edge (joint vector, length, foreshortening, depth,
                        type), computed once per forward and shared by all rounds
    readout (local):    d theta_j = head_j(h_j), j = 1..15; [d transl, d theta_root] = head_hub(h_hub)

Every statistic is a sum over events (event-additive); nothing is max-pooled. Variants (the screen's three arms):
  r       the module above
  noprop  every message removed (the per-node rounds h <- h + relu(W_s h) and the local heads kept): a finger joint
          reads its own part only, the root / translation the whole-hand statistics only
  mlp     no P and no local heads: concat(h^0 of the 17 nodes) -> fc1 (readout_hidden) -> relu -> fc2 -> 51, B2's
          readout form over the same node inputs (the bypass control)
The native constructor arguments are accepted (hidden = C, n_layers = L, rel_hidden = phi's hidden width,
readout_hidden = the mlp control's hidden width); slot_dim / edge_dim / sign are B2's (edge_dim = the gate embedding).
All modules of all variants are built in the same order from the same draws; the variant then deletes what it does not
use, so shared parameters start equal across the three arms at one seed. The last layer of every head is
zero-initialised (initial increment 0, as B2's fc2). Geometry and statistics are computed outside autocast (float32 in
training, float64 at evaluation, as B2's pooling) and with gradient, so a rollout training that makes the previous state
differentiable (semkine/tpd_rol.py) differentiates through them; phi and the learned maps run in the ambient autocast.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import xyz_mesh as XM

N_PARTS = 16
HUB = 16
N_NODES = N_PARTS + 1
#: MANO kinematic tree (kintree_table row 0): part p's parent joint
PARENT = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14)
#: metres per unit of the event residual and of the lever (event point - joint)
L_RES = 0.01
L_LEV = 0.02
#: metres per unit of an edge's joint vector
L_EDGE = 0.02
#: phi input: token7 + residual (3) + z (1) + matched flag (1)
PHI_IN = 7 + 3 + 1 + 1
#: fixed statistics: log mass, mean residual (3), mean lever (3), lever x residual (9), lever x lever (6 upper), view (3)
PART_STATS = 1 + 3 + 3 + 9 + 6 + 3
HUB_STATS = PART_STATS - 3
EDGE_IN = 3 + 1 + 1 + 1 + 4
EDGE_TYPES = ("down", "up", "hub2part", "part2hub")
JOINT_HEAD = 64
HUB_HEAD = 128
VARIANTS = ("r", "noprop", "mlp")
_NATIVE = {"cls": XM.XYZMeshNet}
_STATE = {"variant": None}


def mano_npz():
    return Path(__file__).resolve().parents[1] / "assets/mano_right.npz"


def edges():
    """(src, dst, type) of the 62 directed edges: the tree both ways, hub <-> every part."""
    out = []
    for c, p in enumerate(PARENT):
        if p >= 0:
            out.append((p, c, 0))
            out.append((c, p, 1))
    for p in range(N_PARTS):
        out.append((HUB, p, 2))
        out.append((p, HUB, 3))
    return out


def rodrigues(aa):
    """(..., 3) axis-angle -> (..., 3, 3)."""
    shape = aa.shape[:-1]
    a = aa.reshape(-1, 3)
    th = a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = a / th
    K = torch.zeros(a.shape[0], 3, 3, dtype=a.dtype, device=a.device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    th = th.unsqueeze(-1)
    eye = torch.eye(3, dtype=a.dtype, device=a.device).expand_as(K)
    R = eye + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)
    return R.view(*shape, 3, 3)


def node_frames(prev48, hands_mean):
    """(B, 16, 3, 3) the prior frame of every part node: the global rotation of the part's parent joint (camera frame),
    the identity for the palm (its increment, the root's, is a camera-frame rotation). prev48 = [root, 15 x finger]
    axis-angles relative to the MANO mean; the MANO pose adds `hands_mean` to the fingers."""
    B = prev48.shape[0]
    pose = torch.cat([prev48[:, 0:3], prev48[:, 3:48] + hands_mean.view(1, -1).to(prev48.dtype)], 1).view(B, 16, 3)
    rot = rodrigues(pose)
    G = [rot[:, 0]]
    for j in range(1, N_PARTS):
        G.append(G[PARENT[j]] @ rot[:, j])
    eye = torch.eye(3, dtype=prev48.dtype, device=prev48.device).expand(B, 3, 3)
    return torch.stack([eye] + [G[PARENT[j]] for j in range(1, N_PARTS)], 1)


def event_geometry(w, d, z, vid, verts):
    """Per event: on-hand mass m (B, N), routed point P (B, N, 3) and residual res = z d - P (B, N, 3), camera frame,
    zero where m = 0. w (B, N, R) = q A mask."""
    m = w.sum(-1)
    B = verts.shape[0]
    bidx = torch.arange(B, device=verts.device).view(B, 1, 1)
    P = (w.unsqueeze(-1) * verts[bidx, vid]).sum(2) / m.clamp_min(1e-12).unsqueeze(-1)
    on = (m > 0).unsqueeze(-1)
    res = torch.where(on, z.unsqueeze(-1) * d - P, torch.zeros_like(P))
    return m, torch.where(on, P, torch.zeros_like(P)), res


def part_weights(w, vid, lbs_w):
    """(B, N, 16) soft part weights omega_ip = sum_r w_ir W[v_ir, p] (their sum over p = the event's on-hand mass)."""
    return torch.einsum("bnr,bnrp->bnp", w, lbs_w[vid].to(w.dtype))


def moments(weight, P, res, origin, frame):
    """Weighted moments of (lever = P - origin, residual) per node, in the node frame.

    weight (B, N, K), P / res (B, N, 3), origin (B, K, 3), frame (B, K, 3, 3) (columns = the frame's axes in camera
    coordinates). Returns the mass (B, K) and (B, K, 22) = [log1p mass, mean res / L_RES, mean lever / L_LEV,
    lever x res / (L_LEV L_RES) (9), lever x lever / L_LEV^2 (upper triangle, 6)], zero for a node without mass. The raw
    sums are taken about the batch item's first origin (small cancellation)."""
    M = weight.sum(1)                                                         # (B, K)
    Mc = M.clamp_min(1e-12)
    ref = origin[:, :1]
    Pt = P - ref
    Ot = origin - ref
    Sr = torch.einsum("bnk,bnc->bkc", weight, res)
    SP = torch.einsum("bnk,bnc->bkc", weight, Pt)
    SPr = torch.einsum("bnk,bnc,bnd->bkcd", weight, Pt, res)
    SPP = torch.einsum("bnk,bnc,bnd->bkcd", weight, Pt, Pt)
    Ml = M.unsqueeze(-1)
    M4 = Mc.view(*Mc.shape, 1, 1)
    mr = Sr / Mc.unsqueeze(-1)
    ml = (SP - Ml * Ot) / Mc.unsqueeze(-1)
    T = (SPr - Ot.unsqueeze(-1) * Sr.unsqueeze(-2)) / M4
    G = (SPP - Ot.unsqueeze(-1) * SP.unsqueeze(-2) - SP.unsqueeze(-1) * Ot.unsqueeze(-2)
         + Ml.unsqueeze(-1) * Ot.unsqueeze(-1) * Ot.unsqueeze(-2)) / M4
    Ft = frame.transpose(-1, -2)
    mr = torch.einsum("bkij,bkj->bki", Ft, mr) / L_RES
    ml = torch.einsum("bkij,bkj->bki", Ft, ml) / L_LEV
    T = (Ft @ T @ frame) / (L_LEV * L_RES)
    G = (Ft @ G @ frame) / (L_LEV * L_LEV)
    iu = torch.triu_indices(3, 3, device=G.device)
    feats = torch.cat([torch.log1p(M).unsqueeze(-1), mr, ml, T.flatten(-2), G[..., iu[0], iu[1]]], -1)
    return M, feats * (M > 0).unsqueeze(-1).to(feats.dtype)


def edge_features(J, src, dst, etype):
    """(B, E, EDGE_IN): the vector between the edge's two anchors (a part's prior joint; the wrist joint J_0 for the hub)
    / L_EDGE, its length, its image-plane share |v_xy| / |v| (foreshortening), the destination anchor's depth, the type."""
    anchor = torch.cat([J, J[:, :1]], 1)
    v = anchor[:, dst] - anchor[:, src]
    ln = v.norm(dim=-1, keepdim=True)
    fs = v[..., :2].norm(dim=-1, keepdim=True) / ln.clamp_min(1e-9)
    zd = (anchor[:, dst, 2:3] - XM.Z_REF) / XM.Z_SCALE
    oh = torch.nn.functional.one_hot(etype, len(EDGE_TYPES)).to(J.dtype).unsqueeze(0).expand(J.shape[0], -1, -1)
    return torch.cat([v / L_EDGE, ln / L_EDGE, fs, zd, oh], -1)


class NodeLinear(nn.Conv1d):
    """K independent linear maps applied to (B, K, in): a grouped 1 x 1 convolution (B2's `slot_proj` form; fan-in =
    `n_in` per group, so the initialisation is nn.Linear's, and the cost harness counts its multiply-accumulates)."""

    def __init__(self, k, n_in, n_out, zero=False):
        super().__init__(k * n_in, k * n_out, 1, groups=k)
        self.k, self.n_in, self.n_out = k, n_in, n_out
        if zero:
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        B = x.shape[0]
        return super().forward(x.reshape(B, self.k * self.n_in, 1)).view(B, self.k, self.n_out)


class PMRRound(nn.Module):
    """One round of P: per edge type a bias-free message map; one bias-free self map shared by the nodes."""

    def __init__(self, hidden):
        super().__init__()
        self.msg = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in EDGE_TYPES])
        self.slf = nn.Linear(hidden, hidden, bias=False)


class PMRNet(nn.Module):
    variant = "r"

    def __init__(self, lbs_weights, faces, hidden=128, n_layers=3, rel_hidden=64, slot_dim=32, readout_hidden=256,
                 edge_dim=32, sign=False):
        super().__init__()
        if self.variant not in VARIANTS:
            raise ValueError(f"PMR variant in {VARIANTS}")
        if sign:
            raise ValueError("PMR has no XVM sign channel")
        C = int(hidden)
        V = int(lbs_weights.shape[0])
        self.hidden = C
        self.rounds_n = int(n_layers)
        # B2 attributes the model reads (association tables, the XAG capture slot)
        self.register_buffer("vert_faces", XM.vertex_face_table(faces, V), persistent=False)
        self.register_buffer("lbs_w", lbs_weights.detach().float().clone(), persistent=False)
        self.xag_last_capture = {}
        self.xag_capture_enabled = False
        self.xag_h3_transform = None
        with np.load(str(mano_npz()), allow_pickle=False) as z:
            self.register_buffer("jreg", torch.tensor(np.asarray(z["J_regressor"], np.float32)), persistent=False)
            self.register_buffer("hands_mean", torch.tensor(np.asarray(z["hands_mean"], np.float32)), persistent=False)
            kt = np.asarray(z["kintree_table"])[0].astype(np.int64)
        if tuple(-1 if i == 0 else int(p) for i, p in enumerate(kt)) != PARENT:
            raise ValueError("MANO kinematic tree differs from PMR's")
        E = edges()
        self.register_buffer("e_src", torch.tensor([e[0] for e in E]), persistent=False)
        self.register_buffer("e_dst", torch.tensor([e[1] for e in E]), persistent=False)
        self.register_buffer("e_type", torch.tensor([e[2] for e in E]), persistent=False)
        indeg = torch.zeros(len(EDGE_TYPES), N_NODES)
        for _, t, k in E:
            indeg[k, t] += 1
        # per edge type: its edge indices and the (E_t, 17) scatter onto the destinations with the mean over the type's
        # in-edges (a dense product: the aggregation is deterministic on every device)
        for k in range(len(EDGE_TYPES)):
            sel = torch.tensor([i for i, e in enumerate(E) if e[2] == k])
            S = torch.zeros(len(sel), N_NODES)
            S[torch.arange(len(sel)), self.e_dst[sel]] = 1.0 / indeg[k, self.e_dst[sel]]
            self.register_buffer(f"e_sel{k}", sel, persistent=False)
            self.register_buffer(f"e_scatter{k}", S, persistent=False)
        # ---- every variant builds every module in this order (the same draws at one seed)
        self.phi = nn.Sequential(nn.Linear(PHI_IN, rel_hidden), nn.ReLU(inplace=True), nn.Linear(rel_hidden, C))
        self.enc_part = NodeLinear(N_PARTS, C + PART_STATS + 3, C)
        self.enc_hub = nn.Linear(2 * C + HUB_STATS + 2 + 3, C)
        self.edge_emb = nn.Linear(EDGE_IN, edge_dim)
        self.gate = nn.Linear(edge_dim, C)
        self.rounds = nn.ModuleList([PMRRound(C) for _ in range(self.rounds_n)])
        self.head_j1 = NodeLinear(N_PARTS - 1, C, JOINT_HEAD)
        self.head_j2 = NodeLinear(N_PARTS - 1, JOINT_HEAD, 3, zero=True)
        self.head_h1 = nn.Linear(C, HUB_HEAD)
        self.head_h2 = nn.Linear(HUB_HEAD, 6)
        self.fc1 = nn.Linear(N_NODES * C, int(readout_hidden))
        self.fc2 = nn.Linear(int(readout_hidden), 51)
        for lin in (self.head_h2, self.fc2):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        # ---- the variant keeps what it uses
        if self.variant in ("r", "noprop"):
            del self.fc1, self.fc2
        if self.variant == "noprop":
            del self.edge_emb, self.gate
            for rd in self.rounds:
                del rd.msg
        if self.variant == "mlp":
            del self.edge_emb, self.gate, self.rounds, self.head_j1, self.head_j2, self.head_h1, self.head_h2

    # ---------------------------------------------------------------- pieces (shared with the probes)
    def geometry(self, mask, d, verts, assoc, prev48, ablate, acc):
        """Event weights, residuals, prior joints and node frames, in `acc` (outside autocast)."""
        mask_f = mask.to(acc)
        q = assoc["q"].to(acc)
        w = q.unsqueeze(-1) * assoc["A"].to(acc) * mask_f.unsqueeze(-1)       # (B, N, R)
        r = (1.0 - q) * mask_f                                                 # (B, N)
        if ablate:
            w = torch.zeros_like(w)
            r = torch.zeros_like(r)
        vf = verts.to(acc)
        m, P, res = event_geometry(w, d.to(acc), assoc["z"].to(acc), assoc["vid"], vf)
        return {"w": w, "r": r, "m": m, "P": P, "res": res, "mask_f": mask_f,
                "omega": part_weights(w, assoc["vid"], self.lbs_w.to(acc)),      # (B, N, 16)
                "J": torch.einsum("jv,bvk->bjk", self.jreg.to(acc), vf),       # (B, 16, 3) prior joints
                "F": node_frames(prev48.to(acc), self.hands_mean.to(acc))}    # (B, 16, 3, 3)

    def statistics(self, g):
        """The fixed node statistics: (B, 16, PART_STATS) and (B, HUB_STATS) with their masses."""
        B = g["m"].shape[0]
        Mp, sp = moments(g["omega"], g["P"], g["res"], g["J"], g["F"])
        view = g["F"][..., 2, :]                                                 # the camera z axis in the node frame
        sp = torch.cat([sp, view * torch.ones_like(Mp).unsqueeze(-1)], -1)
        eye = torch.eye(3, dtype=sp.dtype, device=sp.device).expand(B, 1, 3, 3)
        Mh, sh = moments(g["m"].unsqueeze(-1), g["P"], g["res"], g["J"][:, :1], eye)
        return Mp, sp, Mh[:, 0], sh[:, 0]

    def node_inputs(self, tok, d, mask, verts, assoc, prev48, ablate=False):
        """The 17 node inputs and the pieces the probes read."""
        B, N, _ = tok.shape
        dev = tok.device
        acc = torch.float32 if self.training else torch.float64
        with torch.autocast(device_type=dev.type, enabled=False):
            g = self.geometry(mask, d, verts, assoc, prev48, ablate, acc)
            Mp, sp, Mh, sh = self.statistics(g)
            on = (g["m"] > 0).unsqueeze(-1).float()
            zn = ((assoc["z"].float() - XM.Z_REF) / XM.Z_SCALE).unsqueeze(-1)
            f_on = torch.cat([tok.float(), g["res"].float() / L_RES, zn * on, on], -1)
            f_off = torch.cat([tok.float(), torch.zeros(B, N, 5, device=dev)], -1)
        wdt = self.phi[0].weight.dtype
        phi_on = self.phi(f_on.to(wdt))                                         # ambient autocast
        phi_off = self.phi(f_off.to(wdt))
        with torch.autocast(device_type=dev.type, enabled=False):
            phi_on, phi_off = phi_on.to(acc), phi_off.to(acc)
            Phi_p = torch.einsum("bnk,bnc->bkc", g["omega"], phi_on) / Mp.clamp_min(1e-12).unsqueeze(-1)
            Phi_h = torch.einsum("bn,bnc->bc", g["m"], phi_on) / Mh.clamp_min(1e-12).unsqueeze(-1)
            rs = g["r"].sum(1)
            Phi_r = torch.einsum("bn,bnc->bc", g["r"], phi_off) / rs.clamp_min(1e-12).unsqueeze(-1)
            live = g["mask_f"].sum(1).clamp_min(1.0)
            p3 = prev48.to(acc).view(B, N_PARTS, 3)
            x_part = torch.cat([Phi_p, sp, p3], -1)
            x_hub = torch.cat([Phi_h, sh, Phi_r, (rs / live).unsqueeze(-1), (rs > 0).to(acc).unsqueeze(-1), p3[:, 0]], -1)
        return {"x_part": x_part, "x_hub": x_hub, "Mp": Mp, "Mh": Mh, "sp": sp, "sh": sh, **g}

    def encode(self, x_part, x_hub):
        wdt = self.enc_hub.weight.dtype
        return torch.cat([self.enc_part(x_part.to(wdt)), self.enc_hub(x_hub.to(wdt)).unsqueeze(1)], 1)   # (B, 17, C)

    def gates(self, J):
        g = edge_features(J.to(self.edge_emb.weight.dtype), self.e_src, self.e_dst, self.e_type)
        return torch.sigmoid(self.gate(torch.relu(self.edge_emb(g))))          # (B, 62, C)

    def propagate(self, h, gate=None, messages=True):
        """The rounds of P; `messages=False` = the per-node updates only (the noprop arm, the zeroed-message audit)."""
        for rd in self.rounds:
            upd = rd.slf(h)
            if messages:
                for k, lin in enumerate(rd.msg):
                    sel = getattr(self, f"e_sel{k}")
                    msg = gate[:, sel] * lin(h)[:, self.e_src[sel]]
                    upd = upd + torch.einsum("ev,bec->bvc", getattr(self, f"e_scatter{k}").to(msg.dtype), msg)
            h = h + torch.relu(upd)
        return h

    def readout(self, h):
        dj = self.head_j2(torch.relu(self.head_j1(h[:, 1:N_PARTS])))            # (B, 15, 3)
        dh = self.head_h2(torch.relu(self.head_h1(h[:, HUB])))                  # (B, 6) = [transl, root]
        return torch.cat([dh, dj.flatten(1)], -1)

    def forward(self, tok, d, mask, verts, assoc, prev48, ablate: bool = False, verts_geo=None, capture: bool = False,
                h3_transform=None, messages: bool = True):
        if verts_geo is not None or h3_transform is not None or self.xag_h3_transform is not None:
            raise ValueError("PMR covers the S45 / S45-SM forward only")
        n = self.node_inputs(tok, d, mask, verts, assoc, prev48, ablate)
        h0 = self.encode(n["x_part"], n["x_hub"])
        if self.variant == "mlp":
            hL = h0
            delta = self.fc2(torch.relu(self.fc1(h0.reshape(h0.shape[0], -1))))
        else:
            use = self.variant == "r" and messages
            hL = self.propagate(h0, self.gates(n["J"]) if use else None, messages=use)
            delta = self.readout(hL)
        n_live = mask.float().sum()
        stats = {"xyz_valid": assoc["valid"].float().sum() / n_live.clamp_min(1.0),
                 "xyz_hit": assoc["hit"].float().sum() / n_live.clamp_min(1.0),
                 "xyz_resid_mass": n["r"].float().sum() / n_live.clamp_min(1.0),
                 "pmr_parts_live": (n["Mp"] > 0).float().sum(1).mean()}
        aux = {"w": n["w"], "r": n["r"]}
        if capture or self.xag_capture_enabled:
            aux.update({k: n[k].detach() for k in ("x_part", "x_hub", "Mp", "Mh", "J", "F")})
            aux.update(h0=h0.detach(), hL=hL.detach())
        return delta, stats, aux


def install(variant):
    if variant not in VARIANTS:
        raise ValueError(f"PMR variant in {VARIANTS}")
    if XM.XYZMeshNet is not _NATIVE["cls"]:
        raise RuntimeError("XYZMeshNet already patched")
    XM.XYZMeshNet = type(f"PMRNet_{variant}", (PMRNet,), {"variant": variant})
    _STATE["variant"] = variant
    return variant


def uninstall():
    XM.XYZMeshNet = _NATIVE["cls"]
    _STATE["variant"] = None


def installed():
    return _STATE["variant"]


__all__ = ["EDGE_TYPES", "HUB", "N_NODES", "N_PARTS", "PARENT", "PMRNet", "VARIANTS", "edge_features", "edges",
           "event_geometry", "install", "installed", "moments", "node_frames", "part_weights", "rodrigues", "uninstall"]
