r"""MRF: the receptive-field module of the S37-XYZ C1 MeshGNN, isolated (docs/EXPERIMENT_RECORDS.md#mrf-prereg-20261024).

C1's module G (`XYZMeshNet.forward` L572-581) is the shared 3-D edge embedding plus three `XYZMeshLayer` rounds on the
face 1-ring; every round also grows the support, S' = S or any(S_l), so after three rounds features are inpainted onto
vertices without records and the LBS slots pool over them. `MODEL.XYZ_RF` (opt-in; absent or `legacy` builds the plain
`XYZMeshNet`, bit for bit) replaces G by ONE variant; association, records, phi, pooling, vin, LBS slots, residual slot,
readout and loss stay C1's:

* `self3`       no cross-vertex messages, no inpainting: H' = H + O relu(W_s H), S = O. The message, gate and edge maps
                are still built (every C1 tensor keeps its initial value) and frozen.
* `noprop`      C1's rounds with the support held at O: messages only between directly supported vertices, updates and
                LBS pooling on O only.
* `ring4np`     C1's rounds on the 4-hop geodesic ball (exact BFS on the face graph, self excluded; 12 hops after three
                rounds) instead of the 1-ring, S = O; the same parameters, the edge embedding reads the camera-frame
                vectors to the ball's members. Computed in checkpointed chunks of target vertices.
* `global_loc`  C1's rounds, each followed by the XAG `GlobalResidual` block with its attention restricted to the vertex
                itself and its 1-ring: the capacity-matched local control of XAG F+ (`XAG_GNN: global_residual`), built
                last in an isolated RNG scope exactly as F+ builds its blocks, so it starts from F+'s initial values.

The probe tails of tools/mrf also use `ring4` (the ball with support expansion), `none` (no rounds), `diffuse`
(parameter-free frontier-mean inpainting, then three per-vertex layers), `nogeo` (C1's rounds with the edge geometry
zeroed) and `global` (XAG F+). Nothing in production imports this module unless XYZ_RF is set.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .xag import GlobalResidual
from .xyz_mesh import L_EDGE, L_REL, Z_REF, Z_SCALE, XYZMeshNet, lbs_slot_pool, mesh_ring

RF_MODES = ("legacy", "self3", "noprop", "ring4np", "global_loc")
PROBE_MODES = RF_MODES + ("ring4", "none", "diffuse", "nogeo", "global")
BALL_HOPS = 4
BALL_CHUNK = 64


# --------------------------------------------------------------------------- neighbourhoods
@torch.no_grad()
def hop_matrix(faces: torch.Tensor, n_verts: int, max_hops: int) -> torch.Tensor:
    """`(V, V)` long geodesic hop distance on the face graph up to `max_hops`, `max_hops + 1` beyond."""
    idx, emask = mesh_ring(faces, n_verts)
    A = torch.zeros(n_verts, n_verts, dtype=torch.float64)
    rows = torch.arange(n_verts).unsqueeze(1).expand_as(idx)
    A[rows[emask > 0], idx[emask > 0]] = 1.0
    D = torch.full((n_verts, n_verts), max_hops + 1, dtype=torch.long)
    reach = torch.eye(n_verts, dtype=torch.float64)
    D[reach > 0] = 0
    for h in range(1, max_hops + 1):
        reach = ((reach + reach @ A) > 0).to(torch.float64)
        D[(reach > 0) & (D > h)] = h
    return D


@torch.no_grad()
def ball_table(faces: torch.Tensor, n_verts: int, hops: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """The `hops`-ball of every vertex, self excluded, ordered by (hop, vertex id): `(idx (V, K) long, mask (V, K)
    float)`, padded with the vertex itself and mask 0 as `mesh_ring`."""
    D = hop_matrix(faces, n_verts, hops)
    members = []
    for v in range(n_verts):
        m = ((D[v] >= 1) & (D[v] <= hops)).nonzero().squeeze(1)
        members.append(m[torch.argsort(D[v, m] * n_verts + m)])
    K = max(len(m) for m in members)
    idx = torch.arange(n_verts).unsqueeze(1).repeat(1, K)
    mask = torch.zeros(n_verts, K)
    for v, m in enumerate(members):
        idx[v, :len(m)] = m
        mask[v, :len(m)] = 1.0
    return idx, mask


class BallIndex(nn.Module):
    """The k-hop ball for the checkpointed rounds: targets sorted by ball size and cut into chunks of `chunk`, each
    padded only to its own largest ball. Buffers only (non-persistent), so the state dict is unchanged."""

    def __init__(self, faces: torch.Tensor, n_verts: int, hops: int = BALL_HOPS, chunk: int = BALL_CHUNK):
        super().__init__()
        idx, mask = ball_table(faces, n_verts, hops)
        size = mask.sum(1)
        order = torch.argsort(size, stable=True)
        self.hops = int(hops)
        self.n_chunks = 0
        for c0 in range(0, n_verts, int(chunk)):
            rows = order[c0:c0 + int(chunk)]
            k = max(1, int(size[rows].max()))
            self.register_buffer(f"rows{self.n_chunks}", rows.clone(), persistent=False)
            self.register_buffer(f"nbr{self.n_chunks}", idx[rows, :k].clone(), persistent=False)
            self.register_buffer(f"msk{self.n_chunks}", mask[rows, :k].clone(), persistent=False)
            self.n_chunks += 1
        self.register_buffer("inv", torch.argsort(order), persistent=False)

    def chunks(self):
        for c in range(self.n_chunks):
            yield getattr(self, f"rows{c}"), getattr(self, f"nbr{c}"), getattr(self, f"msk{c}")


def local_keys(ring_idx: torch.Tensor, ring_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """`(V, 1 + K)` attention keys of `global_loc`: the vertex itself (slot 0) and its 1-ring."""
    V = ring_idx.shape[0]
    keys = torch.cat([torch.arange(V, device=ring_idx.device).unsqueeze(1), ring_idx], 1)
    kmask = torch.cat([torch.ones(V, 1, device=ring_mask.device, dtype=ring_mask.dtype), ring_mask], 1)
    return keys, kmask


# --------------------------------------------------------------------------- rounds
def fixed_round(layer, H, S, gemb, idx, emask):
    """`XYZMeshLayer.forward` with the support held, S' = S (`noprop`)."""
    B, V, C = H.shape
    K = idx.shape[1]
    flat = idx.reshape(-1)
    Mn = layer.msg(H)[:, flat].view(B, V, K, C)
    Sn = S[:, flat].view(B, V, K) * emask.unsqueeze(0).to(S.dtype)
    gate = torch.sigmoid(layer.gate(gemb))
    w = Sn.unsqueeze(-1).to(Mn.dtype)
    agg = (gate * Mn * w).sum(2) / Sn.sum(2, keepdim=True).clamp_min(1.0).to(Mn.dtype)
    H = H + S.unsqueeze(-1).to(H.dtype) * torch.relu(layer.slf(H) + agg)
    return H, S


def self_round(layer, H, S):
    """The per-vertex part of a round only: H' = H + S relu(W_s H) (`self3`)."""
    return H + S.unsqueeze(-1).to(H.dtype) * torch.relu(layer.slf(H)), S


def diffuse_round(H, S, idx, emask):
    """Parameter-free inpainting: vertices reached this round take the mean of their supported 1-ring neighbours;
    supported vertices keep their value; S' = S or any(S_l)."""
    B, V, C = H.shape
    K = idx.shape[1]
    flat = idx.reshape(-1)
    Sn = S[:, flat].view(B, V, K) * emask.unsqueeze(0).to(S.dtype)
    Hn = H[:, flat].view(B, V, K, C)
    mean = (Hn * Sn.unsqueeze(-1).to(H.dtype)).sum(2) / Sn.sum(2, keepdim=True).clamp_min(1.0).to(H.dtype)
    S_next = torch.maximum(S, Sn.amax(2))
    new = ((S_next > 0) & ~(S > 0)).unsqueeze(-1)
    return torch.where(new, mean, H), S_next


def _ball_chunk(Mm, S, verts, rows, nbr, msk, w_edge, b_edge, w_gate, b_gate, geom: bool):
    B, V, C = Mm.shape
    Vc, K = nbr.shape
    flat = nbr.reshape(-1)
    with torch.no_grad():
        vf = verts.float()
        dv = vf[:, flat].view(B, Vc, K, 3) - vf[:, rows].unsqueeze(2)
        g = torch.cat([dv, dv.norm(dim=-1, keepdim=True)], -1) / L_EDGE * msk.view(1, Vc, K, 1)
        if not geom:
            g = torch.zeros_like(g)
    gemb = torch.relu(F.linear(g.to(w_edge.dtype), w_edge, b_edge))
    gate = torch.sigmoid(F.linear(gemb, w_gate, b_gate))
    Mn = Mm[:, flat].view(B, Vc, K, C)
    Sn = S[:, flat].view(B, Vc, K) * msk.unsqueeze(0).to(S.dtype)
    w = Sn.unsqueeze(-1).to(Mn.dtype)
    agg = (gate * Mn * w).sum(2) / Sn.sum(2, keepdim=True).clamp_min(1.0).to(Mn.dtype)
    return agg, Sn.amax(2)


def ball_round(layer, edge, H, S, verts, ball: BallIndex, expand: bool, geom: bool = True):
    """`XYZMeshLayer` semantics on the k-hop ball: the gate reads the shared edge embedding of the camera-frame vector
    to every ball member; aggregation in checkpointed target chunks. `expand`: S' = S or any(S_l over the ball)."""
    Mm = layer.msg(H)
    parts, grown = [], []
    for rows, nbr, msk in ball.chunks():
        a, s = checkpoint(_ball_chunk, Mm, S, verts, rows, nbr, msk, edge.weight, edge.bias, layer.gate.weight,
                          layer.gate.bias, geom, use_reentrant=False, preserve_rng_state=False)
        parts.append(a)
        grown.append(s)
    agg = torch.cat(parts, 1)[:, ball.inv]
    S_next = torch.maximum(S, torch.cat(grown, 1)[:, ball.inv]) if expand else S
    H = H + S_next.unsqueeze(-1).to(H.dtype) * torch.relu(layer.slf(H) + agg)
    return H, S_next


class LocalResidual(GlobalResidual):
    """The XAG `GlobalResidual` block (same parameters; the same initial values when built at the same RNG state) with
    every vertex's attention restricted to itself (always a key) and its supported 1-ring, computed on that key set."""

    def forward(self, H, S, lbs_weights, keys, kmask):
        support = S > 0
        B, V, C = H.shape
        K1 = keys.shape[1]
        value = self.norm1(H)
        position = self.identity(lbs_weights.to(H.dtype)).unsqueeze(0)
        query = value + position                                             # LBS identity never enters the values
        mha = self.attention
        E, h = mha.embed_dim, mha.num_heads
        d = E // h
        W, b = mha.in_proj_weight, mha.in_proj_bias
        q = F.linear(query, W[:E], b[:E]).view(B, V, h, d)
        k = F.linear(query, W[E:2 * E], b[E:2 * E]).view(B, V, h, d)
        v = F.linear(value, W[2 * E:], b[2 * E:]).view(B, V, h, d)
        flat = keys.reshape(-1)
        kn = k[:, flat].view(B, V, K1, h, d)
        vn = v[:, flat].view(B, V, K1, h, d)
        logits = torch.einsum("bvhd,bvkhd->bvkh", q, kn).float() * (float(d) ** -0.5)
        usable = support[:, flat].view(B, V, K1) & (kmask.unsqueeze(0) > 0)
        usable[:, :, 0] = True
        attn = torch.softmax(logits.masked_fill(~usable.unsqueeze(-1), float("-inf")), dim=2)
        out = torch.einsum("bvkh,bvkhd->bvhd", attn.to(vn.dtype), vn).reshape(B, V, E)
        update = mha.out_proj(out)
        live = support.unsqueeze(-1).to(H.dtype)
        H = (H + update * live) * live
        H = (H + self.ff(self.norm2(H)) * live) * live
        return H


def run_g(variant, H, O, *, layers, edge, edge_g=None, verts=None, ring_idx=None, ring_mask=None, lbs_w=None,
          ball=None, blocks=None, keys=None, kmask=None):
    """Module G of one variant from H0 on the direct support O; returns (H3, S). `legacy` is C1's loop exactly."""
    S = O
    if variant == "none":
        return H, S
    if variant == "self3":
        for layer in layers:
            H, S = self_round(layer, H, S)
        return H, S
    if variant == "diffuse":
        for _ in layers:
            H, S = diffuse_round(H, S, ring_idx, ring_mask)
        for layer in layers:
            H, S = self_round(layer, H, S)
        return H, S
    if variant in ("ring4", "ring4np"):
        for layer in layers:
            H, S = ball_round(layer, edge, H, S, verts, ball, expand=variant == "ring4")
        return H, S
    if variant not in ("legacy", "noprop", "nogeo", "global", "global_loc"):
        raise ValueError(f"unknown receptive-field variant {variant!r}")
    g = torch.zeros_like(edge_g) if variant == "nogeo" else edge_g
    gemb = torch.relu(edge(g.to(edge.weight.dtype)))
    for i, layer in enumerate(layers):
        if variant == "noprop":
            H, S = fixed_round(layer, H, S, gemb, ring_idx, ring_mask)
        else:
            H, S = layer(H, S, gemb, ring_idx, ring_mask)
        if variant == "global":
            H = blocks[i](H, S, lbs_w)
        elif variant == "global_loc":
            H = blocks[i](H, S, lbs_w, keys, kmask)
    return H, S


# --------------------------------------------------------------------------- model variant
class MRFXYZMeshNet(XYZMeshNet):
    """`XYZMeshNet` with one receptive-field variant; the forward is `XYZMeshNet.forward` on the C1 recipe (hit_first,
    geometry, legacy residual / GNN) with module G replaced (`rf="legacy"` reproduces `XYZMeshNet`; tests only)."""

    def __init__(self, lbs_weights, faces, *args, rf: str = "legacy", **kwargs):
        super().__init__(lbs_weights, faces, *args, **kwargs)
        if rf not in RF_MODES:
            raise ValueError(f"MRFXYZMeshNet needs one of {RF_MODES}, got {rf!r}")
        if (self.contour or not self.geom or self.residual_mode != "legacy" or self.xag_gnn != "legacy"
                or self.sign):
            raise ValueError("XYZ_RF is defined on the C1 recipe only (hit_first, geometry, legacy residual / GNN)")
        if self.hidden != 128 or len(self.layers) != 3:
            raise ValueError("XYZ_RF variants keep C1's 128 channels and three rounds")
        self.rf = rf
        V = int(lbs_weights.shape[0])
        # F+ builds its blocks last inside a CPU fork; built here at the same RNG state, `global_loc` starts from the
        # same values, and the stream after construction is C1's
        with torch.random.fork_rng(devices=[]):
            if rf == "global_loc":
                self.mrf_global = nn.ModuleList([LocalResidual() for _ in self.layers])
        if rf == "global_loc":
            keys, kmask = local_keys(self.ring_idx, self.ring_mask)
            self.register_buffer("mrf_keys", keys, persistent=False)
            self.register_buffer("mrf_kmask", kmask, persistent=False)
        if rf == "ring4np":
            self.mrf_ball = BallIndex(faces, V)
        if rf == "self3":
            # built for the shared initial values, never used: frozen (DDP find_unused_parameters=False)
            self.edge.requires_grad_(False)
            for layer in self.layers:
                layer.msg.requires_grad_(False)
                layer.gate.requires_grad_(False)

    def receptive_field(self, H, O, verts, wdt):
        rf = self.rf
        edge_g = self.edge_geometry(verts) if rf in ("legacy", "noprop", "global_loc") else None
        return run_g(rf, H, O, layers=self.layers, edge=self.edge, edge_g=edge_g, verts=verts,
                     ring_idx=self.ring_idx, ring_mask=self.ring_mask, lbs_w=self.lbs_w,
                     ball=getattr(self, "mrf_ball", None), blocks=getattr(self, "mrf_global", None),
                     keys=getattr(self, "mrf_keys", None), kmask=getattr(self, "mrf_kmask", None))

    def forward(self, tok, d, mask, verts, assoc, prev48, ablate: bool = False, verts_geo=None,
                capture: bool = False, h3_transform=None, verts_gt=None):
        if verts_geo is not None or verts_gt is not None:
            raise ValueError("XYZ_RF variants do not take the oracle geometry inputs")
        B, N, _ = tok.shape
        V = verts.shape[1]
        C = self.hidden
        dev = tok.device
        wdt = self.rel[0].weight.dtype
        acc = torch.float32 if self.training else torch.float64
        mask_f = mask.to(torch.float32)
        w = assoc["q"].unsqueeze(-1) * assoc["A"] * mask_f.unsqueeze(-1)                # (B, N, R)
        r = (1.0 - assoc["q"]) * mask_f                                                 # (B, N)
        if ablate:
            w = torch.zeros_like(w)
            r = torch.zeros_like(r)
        # ---- matched records through phi, then onto the vertices (unchanged)
        rb, rn, rr = (w > 0).nonzero(as_tuple=True)
        vrec = assoc["vid"][rb, rn, rr]
        wrec = w[rb, rn, rr]
        drec = d[rb, rn].float()
        zrec = assoc["z"][rb, rn]
        vxyz = verts[rb, vrec].float()
        X = zrec.unsqueeze(-1) * drec
        dxyz = (X - vxyz) / L_REL
        zn = ((zrec - Z_REF) / Z_SCALE).unsqueeze(-1)
        feat = torch.cat([tok[rb, rn].float(), dxyz, zn, torch.ones_like(zn),
                          assoc["hit"][rb, rn].float().unsqueeze(-1)], dim=-1)
        phi = self.rel(feat.to(wdt))                                                    # (R, C)
        key = rb * V + vrec
        mass = torch.zeros(B * V, device=dev, dtype=acc).index_add(0, key, wrec.to(acc))
        num = torch.zeros(B * V, C, device=dev, dtype=acc).index_add(
            0, key, wrec.to(acc).unsqueeze(-1) * phi.to(acc))
        has = mass > 0
        mean = (num / mass.clamp_min(1e-12).unsqueeze(-1)).to(torch.float32) * has.unsqueeze(-1)
        peak = torch.full((B * V, C), -1e4, device=dev, dtype=torch.float32).scatter_reduce(
            0, key.unsqueeze(-1).expand(-1, C), phi.float(), reduce="amax")
        peak = peak * has.unsqueeze(-1)
        x = torch.cat([mean, peak, torch.log1p(mass.to(torch.float32)).unsqueeze(-1)], -1)
        O = has.view(B, V).to(torch.float32)
        H = self.vin(x.view(B, V, 2 * C + 1).to(wdt))
        keep_capture = capture or self.xag_capture_enabled
        H0 = H if keep_capture else None
        # ---- module G: the selected receptive-field variant
        H, S = self.receptive_field(H, O, verts, wdt)
        transform = h3_transform if h3_transform is not None else self.xag_h3_transform
        if transform is not None:
            replacement = transform(H, S.detach().clone(), O.detach().clone())
            if replacement.shape != H.shape or replacement.device != H.device or replacement.dtype != H.dtype:
                raise ValueError("XAG H3 transform must preserve shape, device and dtype")
            H = replacement
        H3 = H if keep_capture else None
        e = lbs_slot_pool(H, S, O, self.lbs_w, self.lbs_hard)                            # (B, 16, 2C+2)
        # ---- residual (unmatched / weakly matched) summary with the same phi (unchanged)
        eb, en = (r > 0).nonzero(as_tuple=True)
        tr = tok[eb, en].float()
        feat_r = torch.cat([tr, torch.zeros(tr.shape[0], self.rel_in - tr.shape[1], device=dev)], dim=-1)
        phi_r = self.rel(feat_r.to(wdt))
        rw = r[eb, en]
        rs = torch.zeros(B, device=dev, dtype=acc).index_add(0, eb, rw.to(acc))
        rnum = torch.zeros(B, C, device=dev, dtype=acc).index_add(
            0, eb, rw.to(acc).unsqueeze(-1) * phi_r.to(acc))
        rhas = (rs > 0).unsqueeze(-1)
        rmean = (rnum / rs.clamp_min(1e-12).unsqueeze(-1)).to(torch.float32) * rhas
        rpeak = torch.full((B, C), -1e4, device=dev, dtype=torch.float32).scatter_reduce(
            0, eb.unsqueeze(-1).expand(-1, C), phi_r.float(), reduce="amax") * rhas
        live = mask_f.sum(1).clamp_min(1.0)
        bslot = torch.cat([rmean, rpeak, (rs.to(torch.float32) / live).unsqueeze(-1),
                           rhas.to(torch.float32)], dim=-1)
        slots = torch.cat([e.to(torch.float32), bslot.unsqueeze(1)], dim=1)             # (B, 17, 2C+2)
        # ---- common readout (unchanged)
        zs = torch.relu(self.slot_proj(slots.reshape(B, -1, 1).to(wdt))).view(B, -1)
        h = torch.relu(self.fc1(torch.cat([zs, prev48.to(zs.dtype)], dim=-1)))
        delta = self.fc2(h)
        n_live = mask_f.sum()
        stats = {
            "xyz_valid": assoc["valid"].float().sum() / n_live.clamp_min(1.0),
            "xyz_hit": assoc["hit"].float().sum() / n_live.clamp_min(1.0),
            "xyz_resid_mass": r.sum() / n_live.clamp_min(1.0),
            "xyz_verts_direct": O.sum(1).mean(),
            "xyz_verts_prop": S.sum(1).mean(),
        }
        aux = {"slots": slots, "O": O, "S": S, "w": w, "r": r}
        if keep_capture:
            self.xag_last_capture = {key_: value.detach().clone() for key_, value in {
                "H0": H0, "H3": H3, "O": O, "S": S, "slots": slots,
                "zs": zs, "bslot": bslot, "prev48": prev48}.items()}
            aux.update({key_: value for key_, value in self.xag_last_capture.items() if key_ not in aux})
        return delta, stats, aux


__all__ = ["BALL_CHUNK", "BALL_HOPS", "BallIndex", "LocalResidual", "MRFXYZMeshNet", "PROBE_MODES", "RF_MODES",
           "ball_round", "ball_table", "diffuse_round", "fixed_round", "hop_matrix", "local_keys", "run_g",
           "self_round"]
