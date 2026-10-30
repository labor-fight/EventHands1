#!/usr/bin/env python3
"""MV2 M10 / TTA (P6-E7-R3) + SDT (P4-E1-R3): the transfer-tier pre-audit of the surface-lifted operators. Zero training, no
event data; SDT reads FIT7 ground-truth POSES only (formal mode; smoke uses synthetic poses).

(a) static constant audit: an AST scan of the new operator code (default `semkine/slr.py`) for MANO constants accessed outside
    the declared body-descriptor sites: integer literals in {778, 1538, 21, 16, 15, 45, 48, 51, 135} (vertex / face / joint /
    pose-dimension counts), attribute accesses `.mano`, and pose-layout slice bounds {1, 3, 6, 45, 48, 51}. Module-level
    STRIDE / CHANNELS / N_OFF (backbone and sampling constants) are exempt; so are the declared adapter sites
    (`SLRModel.__init__`, `slr_geometry`, `_trunk_with_root_head`: the places that read the body). Tier 1 needs 0 sites.
    The scanner is validated on a snippet with a known count (tool validity).
(b) synthetic second body: a serial revolute chain (product-of-exponentials FK with soft LBS weights; 2-link, 3-link planar and
    a 6-DoF "SO-100-like" chain with synthetic dimensions): FK adjoint virtual-work identity
    <d xi, A^T f> = <A d xi, f> with A^T by an analytic torque formula and A d xi by forward-mode autograd (fp32 scaled error
    <= 1e-5; fp64 <= 1e-12), the analytic A^T f against autograd VJP, a deliberately wrong adjoint (descendant terms dropped)
    that must be flagged, G-buffer shapes through `semkine.aura.visible` / `semkine.xyz_mesh.vertex_normals` / `mesh_ring`, and
    the re-bodying probe: `SurfaceLiftedReadout` built on the chain's skinning matrix (must fail for K != 16: hard-wired
    16-joint / 51-d layout), against a 16-link chain where it must run (so the failure is the layout, not the probe).
(c) asset inventory: the declared table of candidate second bodies / sensors with the verification state the ledger
    (#m10-real-data-20261025) supports; tier 2 needs >= 1 second body with full-chain visibility AND units AND extrinsics AND clock
    verified. Nothing is downloaded.
(d) SDT skinning-degree ladder on MANO: A = LBS + posedirs, B = no posedirs, C = hardened (argmax one-hot) weights, D = both; vertex
    displacement of B / C / D against A (mean / p95 / max, mm). Gate: p95(C - A) <= 5 mm lets a rigid second body count as a
    same-class test; > 14 mm kills it.

    python tools/mv2/m10_tta.py --smoke
    python tools/mv2/m10_tta.py --out outputs/mv2/m10_tta
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mv2 import common as C  # noqa: E402

SCAN_FILES = ("semkine/slr.py",)
MANO_INTS = frozenset({778, 1538, 21, 16, 15, 45, 48, 51, 135})
LAYOUT_BOUNDS = frozenset({1, 3, 6, 45, 48, 51})
EXEMPT_MODULE_NAMES = frozenset({"STRIDE", "CHANNELS", "N_OFF"})
DESCRIPTOR_SITES = frozenset({"SLRModel.__init__", "SLRModel.slr_geometry", "SLRModel._trunk_with_root_head"})
ADJOINT_TOL_FP32 = 1e-5
ADJOINT_TOL_FP64 = 1e-12
SDT_P95_ALLOW_MM = 5.0               # = 1/3 of the 14 mm KS4 capture range (rounded)
SDT_P95_KILL_MM = 14.0
SDT_SEED = 20261026


# ----------------------------------------------------------------------------------------------- (a) static audit
class _Scanner(ast.NodeVisitor):
    def __init__(self, src: str, descriptor_sites, exempt_names):
        self.lines = src.splitlines()
        self.desc, self.exempt = set(descriptor_sites), set(exempt_names)
        self.stack: List[str] = []
        self.sites: List[dict] = []
        self.skip_depth = 0

    def _scope(self) -> str:
        return ".".join(self.stack) if self.stack else "<module>"

    def _in_descriptor(self) -> bool:
        s = self._scope()
        return any(s == d or s.startswith(d + ".") for d in self.desc)

    def _add(self, kind: str, node, detail: str):
        if self._in_descriptor() or self.skip_depth:
            return
        self.sites.append({"kind": kind, "line": node.lineno, "scope": self._scope(), "detail": detail,
                           "text": self.lines[node.lineno - 1].strip()[:140]})

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node):
        names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if not self.stack and names & self.exempt:
            return                                              # module-level backbone / sampling constants
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, int) and not isinstance(node.value, bool) and node.value in MANO_INTS:
            self._add("int_literal", node, str(node.value))

    def visit_Attribute(self, node):
        if node.attr == "mano":
            self._add("mano_attr", node, "mano")
        self.generic_visit(node)

    def visit_Slice(self, node):
        for b in (node.lower, node.upper):
            if isinstance(b, ast.Constant) and isinstance(b.value, int) and b.value in LAYOUT_BOUNDS:
                self._add("layout_slice", node, f"{ast.unparse(node)}")
                break
        self.generic_visit(node)


def scan_source(src: str, descriptor_sites=DESCRIPTOR_SITES, exempt_names=EXEMPT_MODULE_NAMES) -> List[dict]:
    sc = _Scanner(src, descriptor_sites, exempt_names)
    sc.visit(ast.parse(src))
    return sc.sites


def scanner_self_test() -> dict:
    snippet = '''
STRIDE = {"layer1": 4, "layer3": 16}
CHANNELS = {"layer3": 256}
class Op(nn.Module):
    def __init__(self):
        self.e = nn.Embedding(16, 8)
        self.w = self.mano.weights
    def forward(self, x):
        y = x[:, 1:]
        z = y.reshape(2, 45)
        return z + x[:, 3:6] + 7 + 1.5
class Adapter:
    def __init__(self):
        self.n = 778
        self.f = self.mano.f
    def geometry(self):
        return x[:, 6:] + 51
'''
    sites = scan_source(snippet, descriptor_sites={"Adapter.__init__"})
    kinds = sorted((s["kind"], s["detail"]) for s in sites)
    expected = sorted([("int_literal", "16"), ("mano_attr", "mano"), ("layout_slice", "1:"), ("int_literal", "45"),
                       ("layout_slice", "3:6"), ("layout_slice", "6:"), ("int_literal", "51")])
    return {"ok": kinds == expected, "found": kinds, "expected": expected}


def static_audit(files: Sequence[str] = SCAN_FILES) -> dict:
    out: Dict[str, dict] = {}
    total = 0
    for f in files:
        sites = scan_source((REPO / f).read_text(encoding="utf-8"))
        by_scope: Dict[str, int] = {}
        for s in sites:
            by_scope[s["scope"]] = by_scope.get(s["scope"], 0) + 1
        out[f] = {"n_sites": len(sites), "by_scope": by_scope, "sites": sites}
        total += len(sites)
    return {"files": out, "n_sites": total, "descriptor_sites": sorted(DESCRIPTOR_SITES),
            "exempt_module_names": sorted(EXEMPT_MODULE_NAMES), "mano_ints": sorted(MANO_INTS),
            "layout_bounds": sorted(LAYOUT_BOUNDS)}


# ----------------------------------------------------------------------------------------------- (b) synthetic body
def _rodrigues(axis: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """(3,), () -> (3, 3) rotation about the unit `axis` (functional: no in-place writes, so forward-mode AD works)."""
    z = torch.zeros((), dtype=axis.dtype)
    K = torch.stack([torch.stack([z, -axis[2], axis[1]]), torch.stack([axis[2], z, -axis[0]]),
                     torch.stack([-axis[1], axis[0], z])])
    eye = torch.eye(3, dtype=axis.dtype)
    return eye + torch.sin(ang) * K + (1 - torch.cos(ang)) * (K @ K)


class ChainBody:
    """A serial revolute chain, product-of-exponentials FK about rest-frame axes through the rest joint positions, with a
    triangulated cylinder skin per link and soft LBS weights (the far ring of a link blends 50 / 50 with the next link).
    `axes` (K, 3) unit vectors, `lengths` (K,) link lengths (m); the base joint sits at `base`."""

    def __init__(self, axes: np.ndarray, lengths: Sequence[float], ring: int = 6, soft: bool = True, seed: int = 0,
                 base=(0.0, 0.0, 0.6), dtype=torch.float32):
        self.dtype = dtype
        self.K = len(lengths)
        ax = torch.tensor(axes, dtype=dtype)
        self.axes = ax / ax.norm(dim=1, keepdim=True)
        rng = np.random.default_rng(seed)
        dirs = rng.standard_normal((self.K, 3))
        dirs[:, 2] = np.abs(dirs[:, 2]) * 0.2
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        c = [np.asarray(base, np.float64)]
        for k in range(self.K):
            c.append(c[-1] + np.asarray(lengths[k]) * dirs[k])
        self.c_rest = torch.tensor(np.stack(c[:self.K]), dtype=dtype)                       # (K, 3) joint positions
        verts, faces, W = [], [], []
        for k in range(self.K):
            d = torch.tensor(dirs[k], dtype=dtype)
            e1 = torch.linalg.cross(d, torch.tensor([0.0, 0.0, 1.0], dtype=dtype))
            if e1.norm() < 1e-6:
                e1 = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
            e1 = e1 / e1.norm()
            e2 = torch.linalg.cross(d, e1)
            ang = torch.arange(ring, dtype=dtype) * (2 * np.pi / ring)
            circ = 0.012 * (torch.cos(ang)[:, None] * e1 + torch.sin(ang)[:, None] * e2)     # (ring, 3)
            p0, p1 = torch.tensor(c[k], dtype=dtype), torch.tensor(c[k + 1], dtype=dtype)
            base_idx = 2 * ring * k
            verts += [p0 + circ, p1 + circ]
            for r in range(ring):
                a, b = base_idx + r, base_idx + (r + 1) % ring
                faces += [[a, b, base_idx + ring + r], [b, base_idx + ring + (r + 1) % ring, base_idx + ring + r]]
            w = torch.zeros(2 * ring, self.K, dtype=dtype)
            w[:ring, k] = 1.0
            if soft and k + 1 < self.K:
                w[ring:, k], w[ring:, k + 1] = 0.5, 0.5
            else:
                w[ring:, k] = 1.0
            W.append(w)
        self.v_rest = torch.cat(verts)                                                       # (V, 3)
        self.faces = torch.tensor(faces, dtype=torch.long)
        self.W = torch.cat(W)                                                                # (V, K)
        self.V = self.v_rest.shape[0]

    # --- kinematics
    def transforms(self, xi: torch.Tensor):
        """M_j (K, 4, 4) = E_0 ... E_j with E_m the rotation by xi_m about axis_m through c_m (rest frame)."""
        Ms, M = [], torch.eye(4, dtype=xi.dtype)
        for m in range(self.K):
            R = _rodrigues(self.axes[m], xi[m])
            t = self.c_rest[m] - R @ self.c_rest[m]
            E = torch.cat([torch.cat([R, t.unsqueeze(1)], 1), torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=xi.dtype)], 0)
            M = M @ E
            Ms.append(M)
        return torch.stack(Ms)

    def fk(self, xi: torch.Tensor) -> torch.Tensor:
        Ms = self.transforms(xi)
        vh = torch.cat([self.v_rest, torch.ones(self.V, 1, dtype=xi.dtype)], 1)              # (V, 4)
        U = torch.einsum("jab,vb->vja", Ms, vh)[..., :3]                                     # (V, K, 3): M_j vbar_i
        return (self.W.unsqueeze(-1) * U).sum(1)

    def adjoint_apply(self, xi: torch.Tensor, f: torch.Tensor, drop_descendants: bool = False) -> torch.Tensor:
        """A^T f (K,) by the torque formula: tau_k = a_k . sum_i (r_ik x f_i), r_ik = sum_{j>=k} W_ij (M_j vbar_i) - S_ik c_k,
        with the CURRENT axis a_k = R_{0..k-1} axis_k and joint position c_k = M_{k-1} cbar_k.
        `drop_descendants=True` is the deliberately wrong version (j == k only) used as a negative control."""
        Ms = self.transforms(xi)
        vh = torch.cat([self.v_rest, torch.ones(self.V, 1, dtype=xi.dtype)], 1)
        U = torch.einsum("jab,vb->vja", Ms, vh)[..., :3]
        Wu = self.W.unsqueeze(-1) * U
        if drop_descendants:
            R, S = Wu, self.W
        else:
            R = torch.flip(torch.cumsum(torch.flip(Wu, [1]), 1), [1])                         # sum_{j >= k}
            S = torch.flip(torch.cumsum(torch.flip(self.W, [1]), 1), [1])
        a_cur, c_cur = [], []
        M_prev = torch.eye(4, dtype=xi.dtype)
        for k in range(self.K):
            a_cur.append(M_prev[:3, :3] @ self.axes[k])
            c_cur.append(M_prev[:3, :3] @ self.c_rest[k] + M_prev[:3, 3])
            M_prev = Ms[k]
        a_cur, c_cur = torch.stack(a_cur), torch.stack(c_cur)
        r = R - S.unsqueeze(-1) * c_cur.unsqueeze(0)                                          # (V, K, 3)
        return torch.einsum("kc,vkc->k", a_cur, torch.linalg.cross(r, f.unsqueeze(1).expand_as(r), dim=-1))


def so100_like(dtype=torch.float32) -> ChainBody:
    """A 6-DoF arm with the joint axes of an SO-100-style serial chain (pan about z, lift / elbow / wrist-flex about y,
    wrist-roll about the link, gripper about x) and SYNTHETIC link lengths: only the kinematic structure is exercised, the
    real URDF / units / extrinsics are unverified (#m10-real-data-20261025)."""
    axes = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]], np.float64)
    return ChainBody(axes, [0.08, 0.11, 0.11, 0.06, 0.04, 0.03], dtype=dtype, seed=3)


def adjoint_identity(body: ChainBody, n_trials: int = 20, seed: int = SDT_SEED) -> dict:
    g = torch.Generator().manual_seed(seed)
    dt = body.dtype
    e_id, e_vjp, e_wrong = [], [], []
    for _ in range(n_trials):
        xi = (torch.randn(body.K, generator=g, dtype=dt) * 0.5)
        dxi = torch.randn(body.K, generator=g, dtype=dt)
        f = torch.randn(body.V, 3, generator=g, dtype=dt)
        _, Adxi = torch.func.jvp(body.fk, (xi,), (dxi,))                                      # forward-mode autograd
        _, vjp_fn = torch.func.vjp(body.fk, xi)
        (At_f_auto,) = vjp_fn(f)
        At_f = body.adjoint_apply(xi, f)
        lhs, rhs = float(dxi @ At_f), float((Adxi * f).sum())
        e_id.append(abs(lhs - rhs) / float(dxi.norm() * At_f.norm()))
        e_vjp.append(float((At_f - At_f_auto).norm() / At_f_auto.norm()))
        wrong = body.adjoint_apply(xi, f, drop_descendants=True)
        e_wrong.append(float((wrong - At_f_auto).norm() / At_f_auto.norm()))
    return {"identity_err_max": float(max(e_id)), "analytic_vs_vjp_err_max": float(max(e_vjp)),
            "wrong_adjoint_err_min": float(min(e_wrong)), "n": n_trials}


def gbuffer_checks(body: ChainBody) -> dict:
    """The surface operators' input shapes on the synthetic body: visibility, projection, normals, 1-ring."""
    from semkine.aura import visible, project, IMG_H, IMG_W
    from semkine.xyz_mesh import vertex_normals, mesh_ring
    B = 2
    xi = torch.zeros(B, body.K)
    verts = torch.stack([body.fk(xi[b].to(body.dtype)) for b in range(B)]).float()
    intr = tuple(torch.full((B,), v) for v in (250.0, 250.0, 120.0, 90.0))
    uv = project(verts, intr)
    vis = visible(verts, intr)
    nrm = vertex_normals(verts, body.faces)
    idx, emask = mesh_ring(body.faces, body.V)
    return {"uv_shape": list(uv.shape), "vis_shape": list(vis.shape), "vis_dtype": str(vis.dtype),
            "any_visible": bool(vis.any()), "normals_shape": list(nrm.shape),
            "normals_unit": bool(torch.allclose(nrm.norm(dim=-1), torch.ones(B, body.V), atol=1e-4)),
            "in_image_frac": float(((uv[..., 0] >= 0) & (uv[..., 0] < IMG_W) & (uv[..., 1] >= 0) & (uv[..., 1] < IMG_H)).float().mean()),
            "ring_shape": list(idx.shape), "ring_mask_shape": list(emask.shape), "V": body.V,
            "ok": bool(list(uv.shape) == [B, body.V, 2] and list(vis.shape) == [B, body.V] and vis.any()
                       and list(nrm.shape) == [B, body.V, 3] and idx.shape == emask.shape and idx.shape[0] == body.V)}


def rebody_probe(body: ChainBody) -> dict:
    """Run `semkine.slr.SurfaceLiftedReadout` on this body's skinning matrix; compare with a 16-link chain (the MANO joint
    count) where it must run."""
    from semkine.slr import SurfaceLiftedReadout
    from semkine.xyz_mesh import mesh_ring, vertex_normals
    from semkine.aura import visible, project

    def attempt(b: ChainBody) -> dict:
        try:
            B = 2
            idx, emask = mesh_ring(b.faces, b.V)
            op = SurfaceLiftedReadout(b.W.float(), b.faces, idx, emask)
            verts = torch.stack([b.fk(torch.zeros(b.K, dtype=b.dtype)) for _ in range(B)]).float()
            intr = tuple(torch.full((B,), v) for v in (250.0, 250.0, 120.0, 90.0))
            geo = {"uv": project(verts, intr), "z": verts[..., 2], "normals": vertex_normals(verts, b.faces),
                   "vis": visible(verts, intr), "fx": intr[0]}
            maps = {t: torch.randn(B, ch, -(-180 // s), -(-240 // s)) for t, ch, s in
                    (("layer1", 64, 4), ("layer2", 128, 8), ("layer3", 256, 16))}
            out = op(maps, geo, 180, 240)
            return {"ran": True, "out_shape": list(out.shape)}
        except Exception as err:
            return {"ran": False, "error": f"{type(err).__name__}: {str(err).splitlines()[0][:160]}"}

    rng16 = np.random.default_rng(0)
    axes16 = rng16.standard_normal((16, 3))
    body16 = ChainBody(axes16, [0.02] * 16, ring=4, dtype=torch.float32, seed=5)
    r_other, r_16 = attempt(body), attempt(body16)
    return {"other_body_K": body.K, "other_body": r_other, "mano_sized_body": r_16,
            "layout_is_hard_wired": bool((not r_other["ran"]) and r_16["ran"] and r_16.get("out_shape", [0, 0])[-1] == 51)}


# ----------------------------------------------------------------------------------------------- (c) assets
ASSETS = [
    {"name": "E-VLA-MultiLight (SO-100 + DAVIS346)", "role": "second_body", "body": "SO-100 arm, 6-d state at 30 Hz",
     "sensor": "DAVIS346 346x260", "full_chain_visible": None, "units_verified": None, "extrinsics_verified": None,
     "clock_verified": None, "license": "apache-2.0 (HF dataset card; code separately)",
     "note": "state units / normalisation / URDF / extrinsics / full-chain visibility not verified; not downloaded",
     "source": "#m10-real-data-20261025"},
    {"name": "E-POSE (UR10, rigid object 6-DoF)", "role": "rigid_check", "body": "rigid object", "sensor": "DAVIS346c",
     "full_chain_visible": False, "units_verified": None, "extrinsics_verified": None, "clock_verified": None,
     "license": "not inferred from the article licence", "note": "not an articulated chain; wrist-mounted camera",
     "source": "#m10-real-data-20261025"},
    {"name": "EvRealHands / EvHandPose", "role": "real_sensor_same_body", "body": "MANO right/left hand, 15 Hz labels",
     "sensor": "DAVIS346 346x260", "full_chain_visible": True, "units_verified": None, "extrinsics_verified": None,
     "clock_verified": None, "license": "no explicit data licence on the author page",
     "note": "fast motion evaluated on 2-D keypoints; same body as training, so not a second body", "source": "#m10-real-data-20261025"},
    {"name": "Ev2Hands-R", "role": "real_sensor_same_body", "body": "two MANO hands, Captury 3-D joints", "sensor": "DAVIS346C",
     "full_chain_visible": True, "units_verified": None, "extrinsics_verified": None, "clock_verified": None,
     "license": "code CC-BY-NC 4.0; data terms not explicit", "note": "same body class", "source": "#m10-real-data-20261025"},
    {"name": "EEH-R (EventEgoHands++)", "role": "real_sensor_same_body", "body": "MANO, OptiTrack + glove", "sensor": "DAVIS346 346x260",
     "full_chain_visible": True, "units_verified": None, "extrinsics_verified": None, "clock_verified": None,
     "license": "request form", "note": "same body class; versioned annotations", "source": "#m10-real-data-20261025"},
]
VERIFY_FLAGS = ("full_chain_visible", "units_verified", "extrinsics_verified", "clock_verified")


def asset_inventory(assets=ASSETS) -> dict:
    qual = [a["name"] for a in assets if a["role"] == "second_body" and all(a.get(f) is True for f in VERIFY_FLAGS)]
    return {"n_assets": len(assets), "second_bodies": [a["name"] for a in assets if a["role"] == "second_body"],
            "qualifying_second_bodies": qual, "n_qualifying": len(qual), "assets": assets}


# ----------------------------------------------------------------------------------------------- (d) SDT
def sdt_ladder(poses: np.ndarray, betas: np.ndarray, chunk: int = 256) -> dict:
    """Vertex displacement (mm) of the B / C / D rungs against A on 51-d poses (root axis-angle + finger residuals; the
    translation is irrelevant). `betas` (N, 10) or (10,)."""
    from mano_layer import ManoLayer
    npz = str(REPO / "assets/mano_right.npz")
    A = ManoLayer(npz, add_mean=True).eval()
    B_ = ManoLayer(npz, add_mean=True).eval()
    C_ = ManoLayer(npz, add_mean=True).eval()
    D_ = ManoLayer(npz, add_mean=True).eval()
    B_.posedirs.zero_()
    D_.posedirs.zero_()
    hard = torch.nn.functional.one_hot(A.weights.argmax(1), A.weights.shape[1]).float()
    C_.weights.copy_(hard)
    D_.weights.copy_(hard)
    betas = np.broadcast_to(np.asarray(betas, np.float32).reshape(-1, 10), (len(poses), 10))
    disp = {"B_no_posedirs": [], "C_hard_weights": [], "D_both": []}
    with torch.no_grad():
        for i in range(0, len(poses), chunk):
            p = torch.from_numpy(np.asarray(poses[i:i + chunk], np.float32))
            b = torch.from_numpy(np.ascontiguousarray(betas[i:i + chunk]))
            args = (b, p[:, 3:6], p[:, 6:51])
            va = A(*args)[0]
            for name, layer in (("B_no_posedirs", B_), ("C_hard_weights", C_), ("D_both", D_)):
                disp[name].append((layer(*args)[0] - va).norm(dim=-1).numpy() * 1000.0)
    out = {}
    for k, v in disp.items():
        v = np.concatenate(v).ravel()
        out[k] = {"mean_mm": float(v.mean()), "p95_mm": float(np.percentile(v, 95)), "max_mm": float(v.max())}
    ow = A.weights.numpy()
    out["skinning"] = {"mean_max_weight": float(ow.max(1).mean()), "frac_vertices_max_weight_lt_0p9": float((ow.max(1) < 0.9).mean()),
                       "mean_effective_joints": float(np.exp(-(ow * np.log(np.clip(ow, 1e-12, 1))).sum(1)).mean())}
    out["n_poses"] = int(len(poses))
    return out


def synthetic_poses(n: int = 64, seed: int = SDT_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = np.zeros((n, 51), np.float32)
    p[:, 3:6] = rng.standard_normal((n, 3)) * 0.3
    p[:, 6:] = np.clip(rng.standard_normal((n, 45)) * 0.35, -1.2, 1.2)
    return p


def fit7_poses(every_ms: int = 250):
    """FIT7 base-recording GT poses (root + fingers; no events read) with each sequence's betas."""
    poses, betas = [], []
    for seq in C.base_seqs(C.FIT7):
        _, _, aux, pos51 = C.load_seq(seq)
        for ra, rb in np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2):
            t = np.arange(ra, rb, every_ms)
            if len(t):
                poses.append(np.asarray(pos51[t], np.float32))
                betas.append(np.repeat(np.asarray(aux["betas"], np.float32).reshape(1, -1)[:, :10], len(t), 0))
    return np.concatenate(poses), np.concatenate(betas)


# ----------------------------------------------------------------------------------------------- decision
def decide(result: dict) -> dict:
    """Gates of TTA (tools/mv2/prereg_m10.md section 1) and the SDT gate.

    tool validity (else `tool_defect`): the scanner reproduces its known count on the self-test snippet; the wrong adjoint is
    flagged (> 1e-3); the fp64 identity is exact (<= 1e-12) on every synthetic body.
    tier 1 (shared operator, per-body retraining): static audit sites == 0.
    second-body unit checks: fp32 adjoint identity <= 1e-5 on every synthetic body, analytic adjoint = autograd VJP (<= 1e-5), and
    the G-buffer shapes ok.
    tier 2 (shared encoder + FK adapter): tier 1 and the unit checks and >= 1 second body in the inventory with full-chain visibility,
    units, extrinsics and clock all verified.
    tier 3 (frozen weights, only FK swapped): never released (the operator is not body-free: `layout_is_hard_wired`, or sites > 0).
    SDT: p95(C - A) <= 5 mm => rigid second body may count as a same-class test; > 14 mm => killed; else descriptive."""
    if not result["scanner_self_test"]["ok"]:
        return {"verdict": "tool_defect", "notes": ["static scanner failed its self-test"]}
    for name, r in result["bodies"].items():
        if r["fp32"]["wrong_adjoint_err_min"] <= 1e-3:
            return {"verdict": "tool_defect", "notes": [f"{name}: the wrong adjoint was not flagged"]}
        if r["fp64"]["identity_err_max"] > ADJOINT_TOL_FP64 or r["fp64"]["analytic_vs_vjp_err_max"] > ADJOINT_TOL_FP64 * 1e3:
            return {"verdict": "tool_defect", "notes": [f"{name}: fp64 adjoint not exact"]}
    n_sites = result["static_audit"]["n_sites"]
    unit_ok = all(r["fp32"]["identity_err_max"] <= ADJOINT_TOL_FP32 and r["fp32"]["analytic_vs_vjp_err_max"] <= ADJOINT_TOL_FP32
                  and r["gbuffer"]["ok"] for r in result["bodies"].values())
    tier1 = n_sites == 0
    n_qual = result["assets"]["n_qualifying"]
    tier2 = bool(tier1 and unit_ok and n_qual >= 1)
    out: Dict[str, object] = {"tier1": tier1, "tier1_sites": n_sites, "second_body_unit_checks": bool(unit_ok),
                              "assets_qualifying": n_qual, "tier2": tier2, "tier3": False, "notes": []}
    rb = result.get("rebody")
    out["layout_is_hard_wired"] = bool(rb["layout_is_hard_wired"]) if rb else None
    if not tier1:
        out["notes"].append(f"{n_sites} MANO-constant sites outside the descriptor: tier 1 not satisfied; M10 reduces to "
                            "real-sensor same-body tests until the operator is descriptor-parametrised")
    if n_qual == 0:
        out["notes"].append("no second body with full-chain visibility + units + extrinsics + clock verified: cross-body claim "
                            "deleted; M10 reduces to a single-body real-sensor test (EvRealHands)")
    sdt = result.get("sdt")
    if sdt is None:
        out["sdt"] = "not_run"
    else:
        p95 = sdt["C_hard_weights"]["p95_mm"]
        out["sdt_C_minus_A_p95_mm"] = p95
        out["sdt"] = ("rigid_second_body_same_class_allowed" if p95 <= SDT_P95_ALLOW_MM else
                      "rigid_second_body_killed" if p95 > SDT_P95_KILL_MM else "descriptive")
    out["verdict"] = f"tier1={'open' if tier1 else 'closed'}|tier2={'open' if tier2 else 'closed'}|tier3=closed|sdt={out['sdt']}"
    return out


# ----------------------------------------------------------------------------------------------- main
def build_result(sdt_poses=None, sdt_betas=None, files=SCAN_FILES) -> dict:
    bodies = {}
    specs = {"chain2_planar": lambda dt: ChainBody(np.array([[0, 0, 1], [0, 0, 1]], np.float64), [0.1, 0.1], dtype=dt, seed=1),
             "chain3_planar": lambda dt: ChainBody(np.array([[0, 0, 1]] * 3, np.float64), [0.08, 0.08, 0.06], dtype=dt, seed=2),
             "so100_like_6dof": so100_like}
    for name, mk in specs.items():
        b32, b64 = mk(torch.float32), mk(torch.float64)
        bodies[name] = {"K": b32.K, "V": b32.V, "fp32": adjoint_identity(b32), "fp64": adjoint_identity(b64),
                        "gbuffer": gbuffer_checks(b32)}
    res = {"scanner_self_test": scanner_self_test(), "static_audit": static_audit(files), "bodies": bodies,
           "rebody": rebody_probe(so100_like()), "assets": asset_inventory()}
    if sdt_poses is not None:
        res["sdt"] = sdt_ladder(sdt_poses, sdt_betas)
    return res


def smoke(args) -> int:
    t0 = time.time()
    ok = True
    st = scanner_self_test()
    print(f"[m10_tta smoke] scanner self-test on a known snippet: found {len(st['found'])} / expected {len(st['expected'])} "
          f"sites, ok={st['ok']}")
    ok &= st["ok"]
    sa = static_audit()
    print(f"  static audit ran on {len(sa['files'])} file(s) "
          f"(by-scope breakdown has {len(next(iter(sa['files'].values()))['by_scope'])} scopes)")
    for name, mk in (("chain2", lambda dt: ChainBody(np.array([[0, 0, 1], [0, 0, 1]], np.float64), [0.1, 0.1], dtype=dt, seed=1)),
                     ("so100_like", so100_like)):
        b = mk(torch.float32)
        ai = adjoint_identity(b, n_trials=6)
        gb = gbuffer_checks(b)
        ctl = ai["wrong_adjoint_err_min"] > 1e-3
        id32 = ai["identity_err_max"] <= ADJOINT_TOL_FP32 and ai["analytic_vs_vjp_err_max"] <= ADJOINT_TOL_FP32
        print(f"  body {name}: K={b.K} V={b.V}; fp32 adjoint identity + analytic-vs-autograd within 1e-5 over {ai['n']} trials: {id32}; "
              f"wrong-adjoint control flagged={ctl}; G-buffer shapes ok={gb['ok']} (uv {gb['uv_shape']}, vis {gb['vis_shape']}, "
              f"normals {gb['normals_shape']})")
        ok &= ctl and gb["ok"] and id32
        a64 = adjoint_identity(mk(torch.float64), n_trials=6)
        exact = a64["identity_err_max"] <= ADJOINT_TOL_FP64 and a64["analytic_vs_vjp_err_max"] <= ADJOINT_TOL_FP64 * 1e3
        print(f"    fp64 identity / analytic-vs-autograd within tolerance: {exact}")
        ok &= exact
    rb = rebody_probe(so100_like())
    print(f"  re-body probe: operator on the 6-link body ran={rb['other_body']['ran']}, on a 16-link body ran="
          f"{rb['mano_sized_body']['ran']} (out {rb['mano_sized_body'].get('out_shape')}); layout_is_hard_wired={rb['layout_is_hard_wired']}")
    ok &= rb["mano_sized_body"]["ran"]
    inv = asset_inventory()
    print(f"  asset inventory: {inv['n_assets']} entries, {len(inv['second_bodies'])} second-body candidates")
    poses = synthetic_poses(32)
    sd = sdt_ladder(poses, np.zeros(10))
    print(f"  SDT ladder on {sd['n_poses']} synthetic poses: rungs {sorted(k for k in sd if k not in ('skinning', 'n_poses'))} computed")
    ok &= set(sd) >= {"B_no_posedirs", "C_hard_weights", "D_both"}
    full = build_result(poses, np.zeros(10))
    dec = decide(full)
    print(f"  decide() plumbing on the smoke result: keys {sorted(dec)}")
    ok &= "verdict" in dec and dec["verdict"] != "tool_defect"
    out_dir = REPO / "outputs" / "mv2" / "smoke" / "m10_tta"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke.json").write_text(json.dumps({"scanner": st["ok"], "ok": bool(ok)}, indent=1))
    print(f"[m10_tta smoke] {'PASS' if ok else 'FAIL'} ({time.time() - t0:.1f} s)")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--files", nargs="*", default=list(SCAN_FILES))
    a = ap.parse_args()
    if a.smoke:
        raise SystemExit(smoke(a))
    if not a.out:
        raise SystemExit("--out outputs/mv2/m10_tta (formal) or --smoke")
    C.verify_protocol()
    poses, betas = fit7_poses()
    res = build_result(poses, betas, files=tuple(a.files))
    res["decision"] = decide(res)
    C.save_json(Path(a.out) / "result.json", res)
    C.receipt(a.out, {"probe": "m10_tta", "verdict": res["decision"]["verdict"], "scanned": a.files},
              scripts=["tools/mv2/m10_tta.py", "tools/mv2/common.py"] + list(a.files))
    print(json.dumps(res["decision"], indent=1, default=C._np))


if __name__ == "__main__":
    main()
