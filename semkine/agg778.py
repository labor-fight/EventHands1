r"""AGG778 aggregation variants of the S37-XYZ C1 module (docs/EXPERIMENT_RECORDS.md#agg778-prereg-20261024).

`MODEL.XYZ_AGG` (opt-in; absent or `legacy` builds the plain `XYZMeshNet`, bit for bit) selects ONE single-factor change
of the record -> 778-vertex aggregation, everything else (association, residual slot, MeshGNN, LBS slots, readout,
loss) identical:

* `matpt`  records also carry the event's barycentric share A_k and the material-point in-plane offset
           (sum_j A_j V_j - V_k) / L_REL, through a zero-initialised, bias-free map into the relation encoder's
           first layer (the C1 function at initialisation, as `XYZ_SIGN`)
* `mass`   mass-consistent pooling: weighted mean, weighted RMS in place of the unweighted max, soft support
           O = min(mass, 1) (the support the MeshGNN and the LBS slots read)
* `fanout` each event with q > 0 is spread to its m = 8 nearest projected prev vertices inside the association band
           (Gaussian weights, sigma = band / 2, renormalised to q), in place of the hit-face / near-layer records
* `vin`    the 257 -> 128 vertex input gets a parallel 257 -> 256 -> 128 ReLU path with a zero-initialised output
           (the C1 function at initialisation)
* `noabs`  (AGG778 R2, docs/EXPERIMENT_RECORDS.md#agg778-r2-prereg-20261024) the relation encoder never sees the
           absolute cues: the token's image position x/W, y/H is zeroed in the records and in the residual slot, and the
           absolute depth zn in the records (no new parameters)

New parameters are built last, inside an isolated RNG scope, so every C1 parameter keeps its initial value and the
data / augmentation streams are unchanged. Nothing in production imports this module unless XYZ_AGG is set.
"""
from __future__ import annotations

import torch
from torch import nn

from .xyz_mesh import L_REL, Z_REF, Z_SCALE, XYZMeshNet, lbs_slot_pool

AGG_MODES = ("legacy", "matpt", "mass", "fanout", "vin", "noabs")
#: record columns zeroed by `noabs`: token x/W, y/H and zn (record layout: tok7 | dXYZ3 | zn | valid | hit)
NOABS_RECORD_COLS = (0, 1, 10)
NOABS_TOKEN_COLS = (0, 1)
FANOUT_M = 8


class AggXYZMeshNet(XYZMeshNet):
    """`XYZMeshNet` with one aggregation variant; the forward is `XYZMeshNet.forward` with branches in the record and
    pooling section only."""

    def __init__(self, *args, agg: str = "legacy", **kwargs):
        super().__init__(*args, **kwargs)
        if agg not in AGG_MODES or agg == "legacy":
            raise ValueError(f"AggXYZMeshNet needs one of {AGG_MODES[1:]}, got {agg!r}")
        if (self.contour or not self.geom or self.residual_mode != "legacy" or self.xag_gnn != "legacy"
                or self.sign):
            raise ValueError("XYZ_AGG is defined on the C1 recipe only (hit_first, geometry, legacy residual / GNN)")
        self.agg = agg
        rel_hidden = self.rel[0].out_features
        with torch.random.fork_rng(devices=[]):
            if agg == "matpt":
                self.rel_extra = nn.Linear(4, rel_hidden, bias=False)
                nn.init.zeros_(self.rel_extra.weight)
            elif agg == "vin":
                self.vin_a = nn.Linear(2 * self.hidden + 1, 2 * self.hidden, bias=False)
                self.vin_b = nn.Linear(2 * self.hidden, self.hidden, bias=False)
                nn.init.zeros_(self.vin_b.weight)

    @torch.no_grad()
    def fanout(self, d, mask, verts, q, band, chunk: int = 64):
        """(B, N, m) vertex ids and weights: the m nearest projected vertices (ray units) inside `band` (B,)."""
        B, N, _ = d.shape
        m = min(FANOUT_M, verts.shape[1])
        vray = verts[..., :2].float() / verts[..., 2:3].float().clamp_min(1e-6)
        vid = torch.zeros(B, N, m, dtype=torch.long, device=d.device)
        w = torch.zeros(B, N, m, dtype=torch.float32, device=d.device)
        for b0 in range(0, B, chunk):
            sl = slice(b0, b0 + chunk)
            d2 = torch.cdist(d[sl, :, :2].float(), vray[sl]).square()
            val, idx = d2.topk(m, dim=-1, largest=False)
            bd = band[sl].float().view(-1, 1, 1)
            ok = (val <= bd ** 2) & mask[sl].unsqueeze(-1) & (q[sl] > 0).unsqueeze(-1)
            k = torch.exp(-val / (2 * (bd / 2) ** 2)) * ok.float()
            w[sl] = k / k.sum(-1, keepdim=True).clamp_min(1e-12) * q[sl].float().unsqueeze(-1)
            vid[sl] = idx
        return vid, w

    def forward(self, tok, d, mask, verts, assoc, prev48, ablate: bool = False, verts_geo=None,
                capture: bool = False, h3_transform=None, verts_gt=None, band=None):
        if verts_geo is not None or verts_gt is not None:
            raise ValueError("XYZ_AGG variants do not take the oracle geometry inputs")
        B, N, _ = tok.shape
        V = verts.shape[1]
        C = self.hidden
        dev = tok.device
        wdt = self.rel[0].weight.dtype
        acc = torch.float32 if self.training else torch.float64
        mask_f = mask.to(torch.float32)
        r = (1.0 - assoc["q"]) * mask_f                                                 # (B, N)
        if self.agg == "fanout":
            if band is None:
                raise ValueError("XYZ_AGG=fanout needs the association band (B,) in ray units")
            vid_all, w = self.fanout(d, mask, verts, assoc["q"], band)
        else:
            vid_all = assoc["vid"]
            w = assoc["q"].unsqueeze(-1) * assoc["A"] * mask_f.unsqueeze(-1)            # (B, N, R)
        if ablate:
            w = torch.zeros_like(w)
            r = torch.zeros_like(r)
        # ---- matched records through phi, then onto the vertices
        rb, rn, rr = (w > 0).nonzero(as_tuple=True)
        vrec = vid_all[rb, rn, rr]
        wrec = w[rb, rn, rr]
        drec = d[rb, rn].float()
        zrec = assoc["z"][rb, rn]
        vxyz = verts[rb, vrec].float()
        X = zrec.unsqueeze(-1) * drec
        dxyz = (X - vxyz) / L_REL
        zn = ((zrec - Z_REF) / Z_SCALE).unsqueeze(-1)
        cols = [tok[rb, rn].float(), dxyz, zn, torch.ones_like(zn), assoc["hit"][rb, rn].float().unsqueeze(-1)]
        feat = torch.cat(cols, dim=-1)
        if self.agg == "noabs":
            feat[:, list(NOABS_RECORD_COLS)] = 0.0
        if self.agg == "matpt":
            A = assoc["A"].float() * (assoc["q"].unsqueeze(-1) > 0).float()
            An = A / A.sum(-1, keepdim=True).clamp_min(1e-12)
            Vr = torch.gather(verts.float().unsqueeze(1).expand(-1, N, -1, -1), 2,
                              assoc["vid"].unsqueeze(-1).expand(-1, -1, -1, 3))
            Xp = (An.unsqueeze(-1) * Vr).sum(2)                                           # material point (B, N, 3)
            extra = torch.cat([An[rb, rn, rr].unsqueeze(-1), (Xp[rb, rn] - vxyz) / L_REL], -1)
            phi = self.rel[2](self.rel[1](self.rel[0](feat.to(wdt)) + self.rel_extra(extra.to(wdt))))
        else:
            phi = self.rel(feat.to(wdt))                                                # (R, C)
        key = rb * V + vrec
        mass = torch.zeros(B * V, device=dev, dtype=acc).index_add(0, key, wrec.to(acc))
        num = torch.zeros(B * V, C, device=dev, dtype=acc).index_add(
            0, key, wrec.to(acc).unsqueeze(-1) * phi.to(acc))
        has = mass > 0
        mean = (num / mass.clamp_min(1e-12).unsqueeze(-1)).to(torch.float32) * has.unsqueeze(-1)
        if self.agg == "mass":
            sq = torch.zeros(B * V, C, device=dev, dtype=acc).index_add(
                0, key, wrec.to(acc).unsqueeze(-1) * phi.to(acc) * phi.to(acc))
            peak = torch.sqrt((sq / mass.clamp_min(1e-12).unsqueeze(-1)).clamp_min(1e-12)).to(torch.float32)
            peak = peak * has.unsqueeze(-1)
            O = mass.clamp(max=1.0).view(B, V).to(torch.float32)
        else:
            peak = torch.full((B * V, C), -1e4, device=dev, dtype=torch.float32).scatter_reduce(
                0, key.unsqueeze(-1).expand(-1, C), phi.float(), reduce="amax")
            peak = peak * has.unsqueeze(-1)
            O = has.view(B, V).to(torch.float32)
        x = torch.cat([mean, peak, torch.log1p(mass.to(torch.float32)).unsqueeze(-1)], -1)
        xv = x.view(B, V, 2 * C + 1).to(wdt)
        H = self.vin(xv)
        if self.agg == "vin":
            H = H + self.vin_b(torch.relu(self.vin_a(xv)))
        keep_capture = capture or self.xag_capture_enabled
        H0 = H if keep_capture else None
        # ---- MeshGNN on the fixed 1-ring (unchanged)
        g = self.edge_geometry(verts)
        gemb = torch.relu(self.edge(g.to(wdt)))
        S = O
        for layer in self.layers:
            H, S = layer(H, S, gemb, self.ring_idx, self.ring_mask)
        transform = h3_transform if h3_transform is not None else self.xag_h3_transform
        if transform is not None:
            replacement = transform(H, S.detach().clone(), O.detach().clone())
            if replacement.shape != H.shape or replacement.device != H.device or replacement.dtype != H.dtype:
                raise ValueError("XAG H3 transform must preserve shape, device and dtype")
            H = replacement
        H3 = H if keep_capture else None
        e = lbs_slot_pool(H, S, O, self.lbs_w, self.lbs_hard)                            # (B, 16, 2C+2)
        # ---- residual (unmatched / weakly matched) summary with the same phi (unchanged but for `noabs`)
        eb, en = (r > 0).nonzero(as_tuple=True)
        tr = tok[eb, en].float()
        if self.agg == "noabs":
            tr[:, list(NOABS_TOKEN_COLS)] = 0.0
        feat_r =torch.cat([tr, torch.zeros(tr.shape[0], self.rel_in - tr.shape[1], device=dev)], dim=-1)
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
            "xyz_verts_direct": (O > 0).float().sum(1).mean(),
            "xyz_verts_prop": (S > 0).float().sum(1).mean(),
        }
        aux = {"slots": slots, "O": O, "S": S, "w": w, "r": r}
        if keep_capture:
            self.xag_last_capture = {key_: value.detach().clone() for key_, value in {
                "H0": H0, "H3": H3, "O": O, "S": S, "slots": slots,
                "zs": zs, "bslot": bslot, "prev48": prev48}.items()}
            aux.update({key_: value for key_, value in self.xag_last_capture.items() if key_ not in aux})
        return delta, stats, aux


__all__ = ["AGG_MODES", "AggXYZMeshNet", "FANOUT_M", "NOABS_RECORD_COLS", "NOABS_TOKEN_COLS"]
