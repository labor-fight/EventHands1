r"""ACU line A: the identity + contour-normal observation interface CA (docs/EXPERIMENT_RECORDS.md#acu-s1-prereg-20261104).

B2 (PFO2 `fgs`) places every routed event at the exact rest-template point its router predicts (PFA) and encodes the
event's 3-D offset from that point's face vertices (dXYZ). The router's error inside a part is a persistent along-finger
field (`#acu-diag-20261104` D2), so the offset carries it as a consistent false innovation. CA keeps the router's
identity only and lets the previous state supply where on the identified finger the event is:

    identity     g_i = group(part_i) in {palm, thumb, index, middle, ring, pinky}, from the routes the batch carries
                 (`semkine.xag.sample_sources`: a router or, for diagnostics, the XAG pseudo-labels); never the state
    contour      the previous state's mesh M(x_s) rasterised into the event grid (semkine/evf_render.py: exact
                 z-buffer, silhouette boundary + occluding contours); seeds = contour pixels whose face belongs to g;
                 one jump-flooding pass per group gives every pixel its nearest seed c
    association  c_i = nearest contour seed of group g_i to the event pixel u_i (an O(1) lookup); the record goes to the
                 seed's face (its three vertices, perspective-correct barycentrics at the seed pixel, depth of that
                 surface point), q = 1
    innovation   r_i = nu(c_i)^T (u_i - c_i), the signed distance along the outward image normal nu of the seed face
                 (positive outside); |u_i - c_i| > R_MAX_PX or no seed within the JFA reach -> residual slot
    record       B2's 13 relation-encoder inputs with dXYZ (3) replaced by (r_i / RHO_PX, nu_x, nu_y); token7, zn
                 (seed depth), valid, hit unchanged -> the same parameters and initial values as B2's trunk

A wrong previous state can only slide an event's association along its finger's contour (no normal innovation); it
cannot move the event to another finger, because the identity never reads the state. Everything here is
parameter-free and runs under no_grad.

Opt-in: `install(variant)` sets the trunk class (before the model is built), `install_hooks()` replaces the face-route
association after the routing hooks are installed, `install_intrinsics(MNISTModel)` records the per-sample render-frame
intrinsics of the forward (the association hook does not receive them). `uninstall()` restores everything.
"""
from __future__ import annotations

import torch
from torch import nn  # noqa: F401  (kept for subclasses)

from . import evf_render as RD
from . import pfo_trunk4 as T4
from . import xag as XAG
from . import xyz_mesh as XM

#: MANO part (argmax skinning joint) -> hand group: 0 palm, 1 thumb, 2 index, 3 middle, 4 ring, 5 pinky
GROUP_OF_PART = (0, 2, 2, 2, 3, 3, 3, 5, 5, 5, 4, 4, 4, 1, 1, 1)
N_GROUPS = 6
#: an event farther than this from its identity's nearest contour seed goes to the residual slot (pixels)
R_MAX_PX = 16.0
#: the residual enters the record in units of RHO_PX pixels
RHO_PX = 4.0
#: depth floor of a seed's surface point (metres), as PFA's
Z_MIN = 0.05
_STATE = {"variant": None, "hooks": None, "intr": None, "hw": None, "intr_native": None, "tables": {}}


# --------------------------------------------------------------------------- tables
def face_groups(faces: torch.Tensor, lbs_weights: torch.Tensor) -> torch.Tensor:
    """(F,) long: the group of every face = the group of the majority skinning part of its three vertices (ties: lowest
    part), the face-part rule of S45-SM's routers."""
    key = (str(faces.device), int(faces.shape[0]))
    t = _STATE["tables"].get(key)
    if t is None:
        vpart = lbs_weights.detach().float().argmax(1)                                   # (V,)
        fp = vpart[faces.long()]                                                         # (F, 3)
        counts = torch.zeros(fp.shape[0], 16, device=faces.device, dtype=torch.long)
        counts.scatter_add_(1, fp, torch.ones_like(fp))
        part = counts.argmax(1)                                                          # first max = lowest part
        t = torch.tensor(GROUP_OF_PART, device=faces.device, dtype=torch.long)[part]
        _STATE["tables"][key] = t
    return t


# --------------------------------------------------------------------------- the rendered contours
@torch.no_grad()
def contour_frame(verts, faces, fgroup, fx, fy, cx, cy, H, W):
    """Rasterised previous state and the per-group nearest-contour-seed maps `sy, sx (B, 6, H, W)` (-1 = none)."""
    with torch.autocast(device_type=verts.device.type, enabled=False):
        verts = verts.float()
        B = verts.shape[0]
        u, v, z = RD.project(verts, fx.float(), fy.float(), cx.float(), cy.float())
        face, _ = RD.rasterize(u, v, z, faces, H, W)
        sil = face >= 0
        depth = torch.zeros(B, H, W, device=verts.device)
        bi, py, px = sil.nonzero(as_tuple=True)
        if bi.numel():
            _, dz = RD.surface_at(u, v, z, faces, bi, face[bi, py, px], px.float(), py.float())
            depth[bi, py, px] = dz
        _, contour = RD.contour_seeds(face, depth)
        grp = torch.where(sil, fgroup[face.clamp_min(0)], torch.full_like(face, -1))
        gid = torch.arange(N_GROUPS, device=verts.device).view(1, N_GROUPS, 1, 1)
        seeds = contour.unsqueeze(1) & (grp.unsqueeze(1) == gid)                           # (B, 6, H, W)
        sy, sx, _ = RD.jfa(seeds.reshape(B * N_GROUPS, H, W))
        return {"u": u, "v": v, "z": z, "face": face, "sy": sy.view(B, N_GROUPS, H, W),
                "sx": sx.view(B, N_GROUPS, H, W)}


@torch.no_grad()
def ca_association(mode, assoc, d, mask, verts, faces, vert_faces, lbs_weights, band, front_tol, z_tol, *,
                   source_part, source_face, source_bary, source_valid, source_depth=None, k=16, chunk=4096):
    """Every live node with an identity -> the nearest contour seed of its group on the previous-state render."""
    if mode != "face_route":
        raise ValueError("CA is installed under MODEL.XAG_ASSOC: face_route")
    if assoc["vid"].shape != (*mask.shape, XM.R_MAX) or assoc["z"].shape != mask.shape:
        raise ValueError("CA expects the four-record hit_first association shapes")
    if _STATE["intr"] is None:
        raise RuntimeError("CA needs the forward's intrinsics (install_intrinsics)")
    fx, fy, cx, cy = (t.float() for t in _STATE["intr"])
    H, W = _STATE["hw"]
    B, N = mask.shape
    dev = mask.device
    if fx.shape[0] != B:
        raise RuntimeError("stale intrinsics: batch size differs")
    fgroup = face_groups(faces, lbs_weights)
    known = source_valid & mask & (source_part >= 0) & (source_part < 16)
    out = {name: torch.zeros_like(assoc[name]) for name in ("vid", "A", "q", "z", "valid", "hit")}
    geo = torch.zeros(B, N, 3, device=dev, dtype=torch.float32)
    assigned = torch.zeros_like(mask)
    b, n = known.nonzero(as_tuple=True)
    if b.numel():
        fr = contour_frame(verts, faces, fgroup, fx, fy, cx, cy, H, W)
        g = torch.tensor(GROUP_OF_PART, device=dev, dtype=torch.long)[source_part[b, n].long().clamp(0, 15)]
        pxf = d[b, n, 0].float() * fx[b] + cx[b]
        pyf = d[b, n, 1].float() * fy[b] + cy[b]
        px = pxf.round().long().clamp(0, W - 1)
        py = pyf.round().long().clamp(0, H - 1)
        ys = fr["sy"][b, g, py, px].long()
        xs = fr["sx"][b, g, py, px].long()
        ok = ys >= 0
        ysc, xsc = ys.clamp_min(0), xs.clamp_min(0)
        offx, offy = pxf - xsc.float(), pyf - ysc.float()
        dist = torch.sqrt(offx * offx + offy * offy)
        fc = fr["face"][b, ysc, xsc]
        ok = ok & (fc >= 0) & (dist <= R_MAX_PX)
        bary, depth = RD.surface_at(fr["u"], fr["v"], fr["z"], faces, b, fc, xsc.float(), ysc.float())
        fv = faces.long()[fc.clamp_min(0)]                                               # (M, 3)
        Vf = verts.float()[b.unsqueeze(1), fv]                                           # (M, 3, 3)
        X = (bary.unsqueeze(-1) * Vf).sum(1)
        nrm = torch.cross(Vf[:, 1] - Vf[:, 0], Vf[:, 2] - Vf[:, 0], dim=-1)
        nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        Z = X[:, 2].clamp_min(1e-6)
        ex = fx[b] * (nrm[:, 0] - X[:, 0] * nrm[:, 2] / Z) / Z
        ey = fy[b] * (nrm[:, 1] - X[:, 1] * nrm[:, 2] / Z) / Z
        en = torch.sqrt(ex * ex + ey * ey)
        ok = ok & (en > 1e-9) & (depth > 0)
        nux, nuy = ex / en.clamp_min(1e-12), ey / en.clamp_min(1e-12)
        r = nux * offx + nuy * offy
        sel = ok.nonzero().flatten()
        bb, nn_ = b[sel], n[sel]
        out["vid"][bb, nn_] = torch.cat([fv[sel], fv[sel][:, :1]], -1)
        out["A"][bb, nn_] = torch.cat([bary[sel], torch.zeros_like(bary[sel][:, :1])], -1).to(out["A"].dtype)
        out["q"][bb, nn_] = 1
        out["z"][bb, nn_] = depth[sel].clamp_min(Z_MIN).to(out["z"].dtype)
        out["valid"][bb, nn_] = True
        out["hit"][bb, nn_] = True
        geo[bb, nn_] = torch.stack([r[sel] / RHO_PX, nux[sel], nuy[sel]], -1)
        assigned[bb, nn_] = True
    out["ca_geo"] = geo
    return out, {"xag_known": known, "xag_eligible": known, "xag_replaced": assigned, "xag_fallback": mask & ~assigned,
                 "xag_recalled": assigned, "xag_none": mask & ~assigned}


# --------------------------------------------------------------------------- the trunk
class ACUMeshNetCA(T4.PFOMeshNet4):
    """B2's composite trunk (semkine/pfo_trunk4.py, variant "gshare" or "nomp") whose relation records read the CA
    geometry (r / RHO_PX, nu) in the three dXYZ slots; every module, width and initial value is the base trunk's."""

    variant = "gshare"

    def forward(self, tok, d, mask, verts, assoc, prev48, ablate: bool = False, verts_geo=None, capture: bool = False,
                h3_transform=None):
        if verts_geo is not None or self.sign or h3_transform is not None or self.xag_h3_transform is not None:
            raise ValueError("the CA trunk covers the S45 / S45-SM forward only")
        if self.flags:
            raise ValueError("the CA trunk takes no pos / hframe / noprev flag")
        if "ca_geo" not in assoc:
            raise RuntimeError("the CA trunk needs the CA association (install_hooks)")
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
        zrec = assoc["z"][rb, rn]
        geo = assoc["ca_geo"][rb, rn].float()
        zn = ((zrec - XM.Z_REF) / XM.Z_SCALE).unsqueeze(-1)
        feat = torch.cat([tok[rb, rn].float(), geo, zn, torch.ones_like(zn),
                          assoc["hit"][rb, rn].float().unsqueeze(-1)], dim=-1)
        phi = self.rel(feat.to(wdt))
        key = rb * V + vrec
        mass = torch.zeros(B * V, device=dev, dtype=acc).index_add(0, key, wrec.to(acc))
        num = torch.zeros(B * V, C, device=dev, dtype=acc).index_add(0, key, wrec.to(acc).unsqueeze(-1) * phi.to(acc))
        has = mass > 0
        mean = (num / mass.clamp_min(1e-12).unsqueeze(-1)).to(torch.float32) * has.unsqueeze(-1)
        peak = torch.full((B * V, C), -1e4, device=dev, dtype=torch.float32).scatter_reduce(
            0, key.unsqueeze(-1).expand(-1, C), phi.float(), reduce="amax")
        peak = peak * has.unsqueeze(-1)
        x = torch.cat([mean, peak, torch.log1p(mass.to(torch.float32)).unsqueeze(-1)], -1)
        O = has.view(B, V).to(torch.float32)
        H = self.vin(x.view(B, V, 2 * C + 1).to(wdt))
        S = O
        if self.mp != "nomp":
            g = T4._NATIVE["cls"].edge_geometry(self, verts)
            gemb = torch.relu(self.edge(g.to(wdt)))
            if self.mp == "gshare":
                gate = torch.sigmoid(self.gate_shared(gemb))
                for layer in self.layers:
                    H, S = layer(H, S, gate, self.ring_idx, self.ring_mask)
            else:
                for layer in self.layers:
                    H, S = layer(H, S, gemb, self.ring_idx, self.ring_mask)
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
        slots = torch.cat([e.to(torch.float32), bslot.unsqueeze(1)], dim=1)
        zs = torch.relu(self.slot_proj(slots.reshape(B, -1, 1).to(wdt))).view(B, -1)
        h = torch.relu(self.fc1(torch.cat([zs, prev48.to(zs.dtype)], dim=-1)))
        delta = self.fc2(h)
        n_live = mask_f.sum()
        stats = {"xyz_valid": assoc["valid"].float().sum() / n_live.clamp_min(1.0),
                 "xyz_hit": assoc["hit"].float().sum() / n_live.clamp_min(1.0),
                 "xyz_resid_mass": r.sum() / n_live.clamp_min(1.0), "xyz_verts_direct": O.sum(1).mean(),
                 "xyz_verts_prop": S.sum(1).mean()}
        return delta, stats, {"slots": slots, "O": O, "S": S, "w": w, "r": r}


# --------------------------------------------------------------------------- install
def install(variant="gshare"):
    """The CA trunk class (before the model is built); variant "gshare" (B2) or "nomp"."""
    if variant not in ("gshare", "nomp"):
        raise ValueError("CA trunk variant gshare | nomp")
    if XM.XYZMeshNet is not T4._NATIVE["cls"]:
        raise RuntimeError("XYZMeshNet already patched")
    XM.XYZMeshNet = type(f"ACUMeshNetCA_{variant}", (ACUMeshNetCA,), {"variant": variant})
    _STATE["variant"] = variant
    return variant


def install_hooks():
    """Replace the face-route association (PFA, already installed by the routing) by CA."""
    from . import s45sm as SM
    from . import s45sm_fast as FAST
    if XAG.route_association not in (SM.pfa_association, FAST.pfa_association):
        raise RuntimeError("CA replaces an installed PFA association only")
    if XM.associate_rays not in (SM.empty_association, FAST.empty_association):
        raise RuntimeError("CA needs the empty ray association of PFA")
    _STATE["hooks"] = XAG.route_association
    XAG.route_association = ca_association
    return {"route_association": "semkine.acu_ca.ca_association", "replaced": _STATE["hooks"].__module__}


def install_intrinsics(model_cls):
    """Record the render-frame intrinsics and grid size of every forward (`MNISTModel._intrinsics` wrapped)."""
    if _STATE["intr_native"] is not None:
        return
    native = model_cls._intrinsics

    def recorded(self, camera_K):
        out = native(self, camera_K)
        _STATE["intr"] = tuple(t.detach() for t in out)
        _STATE["hw"] = (int(self.render_h), int(self.render_w))
        return out
    model_cls._intrinsics = recorded
    _STATE["intr_native"] = (model_cls, native)


def uninstall():
    if _STATE["hooks"] is not None:
        XAG.route_association = _STATE["hooks"]
    if _STATE["intr_native"] is not None:
        cls, native = _STATE["intr_native"]
        cls._intrinsics = native
    if _STATE["variant"] is not None:
        XM.XYZMeshNet = T4._NATIVE["cls"]
    _STATE.update(variant=None, hooks=None, intr=None, hw=None, intr_native=None)


def installed():
    return {"variant": _STATE["variant"], "hooks": _STATE["hooks"] is not None,
            "intrinsics": _STATE["intr_native"] is not None}


__all__ = ["ACUMeshNetCA", "GROUP_OF_PART", "N_GROUPS", "RHO_PX", "R_MAX_PX", "ca_association", "contour_frame",
           "face_groups", "install", "install_hooks", "install_intrinsics", "installed", "uninstall"]
