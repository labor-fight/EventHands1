r"""PFO2 stage-1 trunk (docs/EXPERIMENT_RECORDS.md#pfo2-s1-prereg-20261103): semkine/pfo_trunk4.py's composite trunk with
the base B2's MeshGNN form ("l1" one round, or "gshare" three rounds with one shared edge gate) and the stage-1 single
changes that can make the MeshGNN load-bearing, each opt-in:

  self | rewire   the message graph: every vertex's 1-ring replaced by K copies of itself (no information between
                  vertices, same parameters), or degree-preserving random rewiring (tools of semkine/pfo_trunk.py)
  diff            differential messages m_{l->j} = gate_jl * W_m (H_l - H_j) (a surface gradient; same parameters, the
                  same matrix product W_m H once per vertex)
  disp            a per-vertex displacement head u_v = W_u H3_v (3 outputs, 1 cm units, in the previous hand frame),
                  supervised by the previous-to-GT vertex displacement (auxiliary loss, semkine/pfo2_train.py); the field
                  is fitted to the 51-D state by a parameter-free damped least squares through the analytic LBS
                  (skinning) Jacobian of the previous state (`lbs_jacobian`, blend shapes held fixed), and that solution
                  enters the readout through a zero-initialised map (initial function = the base's)
Variant strings: "<form>[+self|+rewire][+diff][+disp]", form in (l1, gshare). The native constructor runs first (the
same random draws as the base); every added parameter is built inside an isolated RNG scope, so at the same seed the
shared parameters start equal to the base's. With no change the forward is pfo_trunk4's, operation for operation.
`install(variant)` replaces `semkine.xyz_mesh.XYZMeshNet`; the LBS Jacobian needs the previous state and the betas of
the batch, which `install_context(model_cls)` records from `MNISTModel._resolve_betas_K` (opt-in, PFO2 processes only).
"""
from __future__ import annotations

import torch
from torch import nn

from . import pfo_trunk as T1
from . import pfo_trunk2 as T2
from . import pfo_trunk4 as T4
from . import xyz_mesh as XM

FORMS = ("l1", "gshare")
MODS = ("self", "rewire", "diff", "disp")
DISP_UNIT = 0.01                         # head outputs in cm
FD_STEP = 1e-3                           # central differences of the MANO forward (rad)
LS_SCALE = (0.01,) * 3 + (0.1,) * 48     # state scaling of the least squares (m, rad), as tools/pfo/gn.py
LS_DAMP = 1e-6
_NATIVE = {"cls": XM.XYZMeshNet, "resolve": None}
_STATE = {"variant": None}


def parse(variant):
    parts = [p for p in variant.split("+") if p]
    forms = [p for p in parts if p in FORMS]
    mods = {p for p in parts if p in MODS}
    bad = [p for p in parts if p not in FORMS and p not in MODS]
    if bad or len(forms) != 1 or {"self", "rewire"} <= mods:
        raise ValueError(f"bad PFO2 variant {variant!r}")
    return forms[0], mods


def jacobian(fk, prev, betas, step=FD_STEP, chunk=8192):
    """(B, V, 3, 51) d vertices / d state at `prev` (B, 51): translation columns analytic (identity), root and finger
    columns by central differences of `fk` (the model's batched MANO forward, (params (N, 51), betas (N, 10)) ->
    vertices (N, V, 3)); no gradient."""
    with torch.no_grad():
        B = prev.shape[0]
        p = prev.detach().float()
        eye = torch.eye(51, device=p.device, dtype=p.dtype)[3:] * step                     # (48, 51)
        stack = torch.cat([p.unsqueeze(1) + eye.unsqueeze(0), p.unsqueeze(1) - eye.unsqueeze(0)], 1).reshape(-1, 51)
        bstack = betas.detach().float().repeat_interleave(96, 0)
        outs = [fk(stack[i:i + chunk], bstack[i:i + chunk]) for i in range(0, stack.shape[0], chunk)]
        V = torch.cat(outs).view(B, 2, 48, -1, 3)
        Jr = (V[:, 0] - V[:, 1]).permute(0, 2, 3, 1) / (2 * step)                           # (B, V, 3, 48)
        Jt = torch.eye(3, device=p.device, dtype=p.dtype).view(1, 1, 3, 3).expand(B, Jr.shape[1], 3, 3)
        return torch.cat([Jt, Jr], -1)


def _left_jacobian(aa):
    """(N, 3) axis-angle -> (N, 3, 3) left Jacobian of SO(3): d exp([a]x) = [J_l(a) da]x exp([a]x)."""
    phi = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    a = aa / phi
    phi = phi.unsqueeze(-1)
    K = torch.zeros(aa.shape[0], 3, 3, dtype=aa.dtype, device=aa.device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -a[:, 2], a[:, 1], a[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -a[:, 0], -a[:, 1], a[:, 0]
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device)
    sin_c = torch.sin(phi) / phi
    return sin_c * eye + (1 - sin_c) * a.unsqueeze(-1) * a.unsqueeze(-2) + (1 - torch.cos(phi)) / phi * K


def lbs_jacobian(mano, prev, betas):
    """(B, V, 3, 51) analytic linear-blend-skinning Jacobian of the MANO vertices at `prev` (B, 51; layout transl,
    global orient, 45 local axis-angles without the mean) with the shape and pose blend shapes held fixed: translation
    columns = identity; joint k's three columns = the rotation of its subtree about its posed joint by the world-frame
    increment w = R_parent(k) J_l(theta_k) d theta, i.e. dV_v = w x sum_{j in subtree(k)} w_vj (x_vj - P_k), x_vj the
    vertex carried by joint j. About 2 M multiply-adds per sample; no gradient."""
    from mano_layer import _transform_mat, batch_rodrigues
    with torch.no_grad():
        B = prev.shape[0]
        dt = mano.v_template.dtype
        p = prev.detach().to(dt)
        betas = betas.detach().to(dt)
        pose = torch.cat([p[:, 3:6], p[:, 6:51] + mano.hands_mean.view(1, -1)], 1)                   # (B, 48)
        v_shaped = mano.v_template + torch.einsum("bl,mkl->bmk", betas, mano.shapedirs)
        J = torch.einsum("bik,ji->bjk", v_shaped, mano.J_regressor)
        rot = batch_rodrigues(pose.reshape(-1, 3)).view(B, 16, 3, 3)
        eye = torch.eye(3, device=p.device, dtype=dt)
        v_posed = v_shaped + torch.einsum("bl,mkl->bmk", (rot[:, 1:] - eye).reshape(B, -1), mano.posedirs)
        rel = J.clone()
        rel[:, 1:] -= J[:, mano._parents_py[1:]]
        mats = _transform_mat(rot.reshape(-1, 3, 3), rel.reshape(-1, 3, 1)).view(B, 16, 4, 4)
        chain = [mats[:, 0]]
        for i in range(1, 16):
            chain.append(chain[mano._parents_py[i]] @ mats[:, i])
        G = torch.stack(chain, 1)                                                                      # (B, 16, 4, 4)
        P = G[:, :, :3, 3]                                                                             # posed joints
        Jh = torch.cat([J, torch.zeros(B, 16, 1, device=p.device, dtype=dt)], 2)
        T = G.clone()
        T[:, :, :3, 3] = G[:, :, :3, 3] - (G @ Jh.unsqueeze(-1))[:, :, :3, 0]
        x = torch.einsum("bjkl,bvl->bjvk", T[:, :, :3, :3], v_posed) + T[:, :, :3, 3].unsqueeze(2)     # (B, 16, V, 3)
        W = mano.weights                                                                               # (V, 16)
        sub = _subtree(mano).to(device=p.device, dtype=dt)                                             # (16 k, 16 j)
        wx = W.t().unsqueeze(0).unsqueeze(-1) * x                                                      # (B, 16 j, V, 3)
        S = torch.einsum("kj,bjvc->bkvc", sub, wx)                                                     # sum over subtree
        Wk = torch.einsum("kj,vj->kv", sub, W)                                                         # (16, V)
        arm = S - Wk.view(1, 16, -1, 1) * P.unsqueeze(2)                                               # (B, 16, V, 3)
        Rpar = torch.cat([eye.expand(B, 1, 3, 3), G[:, mano._parents_py[1:], :3, :3]], 1)              # (B, 16, 3, 3)
        omega = Rpar @ _left_jacobian(pose.reshape(-1, 3)).view(B, 16, 3, 3)                            # cols: d theta_i
        # dV / d theta_{k,i} = omega[:, k, :, i] x arm[:, k, v, :]
        cols = torch.cross(omega.permute(0, 1, 3, 2).unsqueeze(3).expand(-1, -1, -1, arm.shape[2], -1),
                           arm.unsqueeze(2).expand(-1, -1, 3, -1, -1), dim=-1)                          # (B, 16, 3, V, 3)
        Jr = cols.permute(0, 3, 4, 1, 2).reshape(B, arm.shape[2], 3, 48)
        Jt = eye.view(1, 1, 3, 3).expand(B, Jr.shape[1], 3, 3)
        return torch.cat([Jt, Jr], -1)


_SUBTREE = {}


def _subtree(mano):
    """(16, 16) 0/1: entry (k, j) = 1 when joint j is k or a descendant of k."""
    key = tuple(mano._parents_py)
    if key not in _SUBTREE:
        par = mano._parents_py
        M = torch.zeros(16, 16)
        for j in range(16):
            k = j
            while k >= 0:
                M[k, j] = 1.0
                k = par[k] if k > 0 else -1
        _SUBTREE[key] = M
    return _SUBTREE[key]


def least_squares(J, u, scale, damp=LS_DAMP):
    """argmin_x sum_v |J_v x - u_v|^2 / V + damp |x / scale|^2, per sample; J (B, V, 3, 51), u (B, V, 3) -> (B, 51)."""
    B, V = u.shape[:2]
    A = J.reshape(B, V * 3, 51).double()
    H = A.transpose(1, 2) @ A / V
    H = H + torch.diag_embed((damp / scale.double() ** 2).expand(B, 51))
    g = A.transpose(1, 2) @ u.reshape(B, V * 3, 1).double() / V
    return torch.linalg.solve(H, g).squeeze(-1).to(u.dtype)


class PFO2MeshNet(T4.PFOMeshNet4):
    variant = "l1"
    mods = frozenset()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "self" in self.mods:
            V, K = self.ring_idx.shape
            self.ring_idx = torch.arange(V).unsqueeze(1).repeat(1, K)
        elif "rewire" in self.mods:
            idx, mask, info = T1.rewired_ring(self.ring_idx.cpu(), self.ring_mask.cpu())
            self.ring_idx, self.ring_mask = idx, mask
            self.rewire_info = info
        if "disp" in self.mods:
            with torch.random.fork_rng(devices=[]):
                self.disp_head = nn.Linear(self.hidden, 3)
                self.disp_in = nn.Linear(51, self.fc1.out_features, bias=False)
            nn.init.zeros_(self.disp_in.weight)
            self.register_buffer("ls_scale", torch.tensor(LS_SCALE), persistent=False)
        self.pfo2_ctx = None
        self.pfo2_aux = {}

    def rounds(self, H, S, verts, wdt):
        """The message rounds of the form (pfo_trunk4's loop), with the diff option."""
        g = _NATIVE["cls"].edge_geometry(self, verts)
        gemb = torch.relu(self.edge(g.to(wdt)))
        shared = torch.sigmoid(self.gate_shared(gemb)) if self.mp == "gshare" else None
        for layer in self.layers:
            if "diff" not in self.mods:
                H, S = layer(H, S, shared if shared is not None else gemb, self.ring_idx, self.ring_mask)
                continue
            B, V, C = H.shape
            K = self.ring_idx.shape[1]
            flat = self.ring_idx.reshape(-1)
            M = layer.msg(H)
            Mn = M[:, flat].view(B, V, K, C) - M.unsqueeze(2)
            Sn = S[:, flat].view(B, V, K) * self.ring_mask.unsqueeze(0).to(S.dtype)
            gate = shared if shared is not None else torch.sigmoid(layer.gate(gemb))
            w = Sn.unsqueeze(-1).to(Mn.dtype)
            agg = (gate * Mn * w).sum(2) / Sn.sum(2, keepdim=True).clamp_min(1.0).to(Mn.dtype)
            S_next = torch.maximum(S, Sn.amax(2))
            H = H + S_next.unsqueeze(-1).to(H.dtype) * torch.relu(layer.slf(H) + agg)
            S = S_next
        return H, S

    def disp_slot(self, H, verts, prev48):
        """u (B, V, 3) in the previous hand frame and the scaled least-squares state (B, 51)."""
        ctx = self.pfo2_ctx
        if ctx is None:
            raise RuntimeError("the disp head needs the previous state and betas (pfo2_trunk.install_context)")
        R = T2.rodrigues(prev48[:, 0:3].float())
        u = self.disp_head(H).float() * DISP_UNIT
        J = lbs_jacobian(ctx["mano"], ctx["prev"], ctx["betas"])
        Jh = torch.einsum("bik,bvil->bvkl", R, J)                                               # R^T J (hand frame)
        x = least_squares(Jh, u, self.ls_scale)
        self.pfo2_aux = {"u": u, "verts": verts.detach(), "R": R.detach()}
        return x / self.ls_scale

    def forward(self, tok, d, mask, verts, assoc, prev48, ablate: bool = False, verts_geo=None, capture: bool = False,
                h3_transform=None):
        if verts_geo is not None or self.sign or h3_transform is not None or self.xag_h3_transform is not None:
            raise ValueError("PFO2 trunk covers the S45 / S45-SM forward only")
        if self.flags:
            raise ValueError("PFO2 trunk takes no pfo_trunk4 flags")
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
        feat = torch.cat([tok[rb, rn].float(), dxyz, zn, torch.ones_like(zn),
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
        H, S = self.rounds(H, S, verts, wdt)
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
        pre = self.fc1(torch.cat([zs, prev48.to(zs.dtype)], dim=-1))
        if "disp" in self.mods:
            pre = pre + self.disp_in(self.disp_slot(H, verts, prev48).to(pre.dtype))
        h = torch.relu(pre)
        delta = self.fc2(h)
        n_live = mask_f.sum()
        stats = {"xyz_valid": assoc["valid"].float().sum() / n_live.clamp_min(1.0),
                 "xyz_hit": assoc["hit"].float().sum() / n_live.clamp_min(1.0),
                 "xyz_resid_mass": r.sum() / n_live.clamp_min(1.0), "xyz_verts_direct": O.sum(1).mean(),
                 "xyz_verts_prop": S.sum(1).mean()}
        return delta, stats, {"slots": slots, "O": O, "S": S, "w": w, "r": r}


def install(variant):
    form, mods = parse(variant)
    if XM.XYZMeshNet is not _NATIVE["cls"]:
        raise RuntimeError("XYZMeshNet already patched")
    XM.XYZMeshNet = type(f"PFO2MeshNet_{variant.replace('+', '_')}", (PFO2MeshNet,),
                         {"variant": form, "mods": frozenset(mods)})
    _STATE["variant"] = variant
    return variant


def install_context(model_cls):
    """Record (fk, prev, betas) on the trunk at every XYZ forward: `MNISTModel._resolve_betas_K` is the forward's first
    call with both; the record is read by the disp head (no change to any computation)."""
    if _NATIVE["resolve"] is not None:
        return
    native = model_cls._resolve_betas_K
    _NATIVE["resolve"] = (model_cls, native)

    def resolve(self, prevpos, betas, camera_K):
        out = native(self, prevpos, betas, camera_K)
        xyz = getattr(self, "xyz", None)
        if isinstance(xyz, PFO2MeshNet):
            xyz.pfo2_ctx = {"prev": prevpos, "betas": out[0], "mano": self.mano,
                            "fk": lambda p, b, _m=self: _m._fk(p, b)[0]}
        return out
    model_cls._resolve_betas_K = resolve


def uninstall():
    XM.XYZMeshNet = _NATIVE["cls"]
    if _NATIVE["resolve"] is not None:
        cls, native = _NATIVE["resolve"]
        cls._resolve_betas_K = native
        _NATIVE["resolve"] = None
    _STATE["variant"] = None


def installed():
    return _STATE["variant"]


__all__ = ["FORMS", "MODS", "PFO2MeshNet", "install", "install_context", "installed", "jacobian", "least_squares",
           "parse", "uninstall"]
