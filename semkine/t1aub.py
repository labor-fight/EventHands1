r"""T1A-UB (docs/EXPERIMENT_RECORDS.md#t1aub-prereg-20261102): opt-in model pieces of the S45-T1a upper-bound study.

T1a = S45 with token7 replaced in the relation record by the node feature of a causal kNN event graph
(semkine/s45tr_t1a.py). `install(spec)` replaces `semkine.xyz_mesh.XYZMeshNet` by a subclass of whatever class is
installed then (T1a's when the arm keeps the event graph, S45's otherwise) that may, per registered arm:

  readout  rebuild the common readout (grouped slot projection, fc1, fc2) at registered widths, inside an isolated RNG
           scope seeded with READOUT_SEED after every other module: every other initial tensor and every later random
           draw stay the base's; fc2 stays zero-initialised. (MODEL.XYZ_SLOT_DIM / XYZ_READOUT_HIDDEN keep S45's 32 /
           256 because semkine/xag.py validates them whenever MODEL.XAG_ASSOC is set; the widths live in T1AUB.READOUT.)
  gtfeat   (diagnostic oracle, not deployable) append the label surface location of every sampled event -- the XAG
           pseudo-source's rest-template point in centimetres and a valid flag -- to the node features before phi,
           through four zero-initialised input columns of phi's first layer, so the initial function is the base's.
           MODEL.XAG_ASSOC face_route only brings the sources into the batch; `install_gtfeat_route` replaces the routing
           step by a pass-through that records them, so the association stays S45's ray association.
  reparam  (evaluation / deployment only, `set_reparam`) T1a's EdgeConv rounds in the algebraically equal form
           relu(W_h h_j - W_h h_i + W_d dp + b): W_h h once per node instead of W [h_j - h_i; dp] once per edge.
           The MACs of these functional products are added to `MACS` while `count_macs()` is active (thop does not see
           them).

Nothing changes unless `install` is called; `uninstall` restores what it replaced.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from . import s45sm as _SM  # noqa: F401  (its import-time check needs the native XAG / association functions)
from . import s45tr_t1a as T1A
from . import xag as XAG
from . import xyz_mesh as XM
from .event_gnn import T_COL

NATIVE_NET = T1A._ORIGINAL
if NATIVE_NET.__module__ != "semkine.xyz_mesh" or NATIVE_NET.__name__ != "XYZMeshNet":
    raise ImportError("semkine.t1aub must be imported before any XYZMeshNet hook is installed")
NATIVE_ROUTE = XAG.route_association
#: seed of the isolated RNG scope a rebuilt readout draws its initial values from
READOUT_SEED = 1102
#: label feature: rest-template point (cm, 3) + valid flag (1)
GT_DIM = 4
CM = 0.01
#: phi input columns after the node feature: dXYZ (3), zn, valid, hit
REC_TAIL = XM.REL_IN - 7
_STATE = {"spec": None, "base": None, "cls": None, "stash": None, "gtfeat_route": False, "reparam": False}
MACS = {"on": False, "count": 0}


@contextmanager
def count_macs():
    """Count the MACs of this module's functional products (the reparameterised EdgeConv) in MACS['count']."""
    MACS.update(on=True, count=0)
    try:
        yield MACS
    finally:
        MACS["on"] = False


def _add_macs(n):
    if MACS["on"]:
        MACS["count"] += int(n)


# --------------------------------------------------------------------------- pieces
def rebuild_readout(net, slot_dim: int, hidden: int):
    """The common readout at `slot_dim` / `hidden`, drawn in an isolated scope seeded with READOUT_SEED."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(READOUT_SEED)
        net.slot_proj = nn.Conv1d(net.N_SLOTS * net.slot_in, net.N_SLOTS * int(slot_dim), 1, groups=net.N_SLOTS)
        net.fc1 = nn.Linear(net.N_SLOTS * int(slot_dim) + 48, int(hidden))
        net.fc2 = nn.Linear(int(hidden), 51)
    nn.init.zeros_(net.fc2.weight)
    nn.init.zeros_(net.fc2.bias)


def widen_phi(net, extra: int):
    """Insert `extra` zero-initialised input columns into phi's first layer right after the node-feature block."""
    old = net.rel[0]
    width = net.rel_in - REC_TAIL
    with torch.random.fork_rng(devices=[]):
        new = nn.Linear(old.in_features + int(extra), old.out_features)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :width] = old.weight[:, :width]
        new.weight[:, width + int(extra):] = old.weight[:, width:]
        new.bias.copy_(old.bias)
    net.rel[0] = new
    net.rel_in += int(extra)


def template(npz):
    with np.load(str(npz), allow_pickle=False) as z:
        return (torch.tensor(np.asarray(z["v_template"], np.float32)), torch.tensor(np.asarray(z["f"], np.int64)))


def gt_features(net, mask):
    """`(B, N, 4)` float32: the recorded label source's rest-template point (cm) and a valid flag; zero elsewhere."""
    st = _STATE["stash"]
    _STATE["stash"] = None
    if st is None or st["face"].shape != mask.shape:
        raise RuntimeError("gtfeat needs the XAG sources of this very forward (MODEL.XAG_ASSOC face_route + "
                           "DATA.XAG_SOURCE_ROOT and the pass-through route installed)")
    face, bary = st["face"], st["bary"].float()
    n_faces = net.t1aub_faces.shape[0]
    ok = st["valid"] & mask & (face >= 0) & (face < n_faces)
    ok = ok & torch.isfinite(bary).all(-1) & (bary >= 0).all(-1) & (bary.sum(-1) > 0)
    w = bary.clamp_min(0.0)
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
    point = (w.unsqueeze(-1) * net.t1aub_tmpl[net.t1aub_faces[face.clamp(0, n_faces - 1)]]).sum(-2) / CM
    out = torch.cat([point, torch.ones_like(point[..., :1])], -1)
    return torch.where(ok.unsqueeze(-1), out, torch.zeros_like(out))


def reparam_features(gnn, tok, mask):
    """`semkine.s45tr_t1a.node_features` (no gradients) with every EdgeConv round as relu(W_h h_j - W_h h_i + W_d dp + b)."""
    m = mask.unsqueeze(-1)
    feat = tok * m
    p = torch.stack([feat[..., 0], feat[..., 1], feat[..., T_COL] * gnn.t_scale], dim=-1) * m
    idx, dp, emask = gnn._edges(p.to(torch.float32), mask)
    h = torch.relu(gnn.embed(feat.to(gnn.embed.weight.dtype))) * m
    B, N, C = h.shape
    k = idx.shape[-1]
    rows = (idx + (torch.arange(B, device=idx.device) * N).view(B, 1, 1)).reshape(-1)
    em = emask.to(h.dtype).unsqueeze(-1)
    den = emask.sum(2, keepdim=True).clamp_min(1.0).to(h.dtype)
    dpt = dp.to(h.dtype)
    for layer in gnn.layers:
        w, b = layer.lin.weight, layer.lin.bias
        a = h @ w[:, :C].t()
        pre = a.reshape(B * N, C).index_select(0, rows).view(B, N, k, C) - a.unsqueeze(2) + dpt @ w[:, C:].t() + b
        h = (h + (torch.relu(pre) * em).sum(2) / den) * m
        _add_macs(B * N * C * C + B * N * k * dp.shape[-1] * C)
    return h


@torch.no_grad()
def passthrough_route(mode, assoc, d, mask, verts, faces, vert_faces, lbs_weights, band, front_tol, z_tol, *,
                      source_part, source_face, source_bary, source_valid, k=16, chunk=4096):
    """gtfeat's routing step: record the sampled label sources for the model and keep S45's ray association unchanged."""
    if mode != "face_route":
        raise ValueError("gtfeat brings the sources in under MODEL.XAG_ASSOC face_route")
    _STATE["stash"] = {"face": source_face, "bary": source_bary, "valid": source_valid & mask}
    none = torch.zeros_like(mask)
    return assoc, {"xag_known": source_valid & mask, "xag_eligible": none, "xag_replaced": none}


# --------------------------------------------------------------------------- the subclass
def make_class(base, spec):
    class T1aubXYZMeshNet(base):
        """S45 (or T1a) with the arm's registered readout widths / label feature; see the module docstring."""

        SPEC = dict(spec)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            s = self.SPEC
            self.t1aub_eg = hasattr(self, "evgnn")
            if bool(s["eg"]) != self.t1aub_eg:
                raise RuntimeError("the installed base class does not match T1AUB.EG")
            ro = s.get("readout")
            if ro:
                rebuild_readout(self, int(ro["slot_dim"]), int(ro["hidden"]))
            self.t1aub_gtfeat = bool(s.get("gtfeat"))
            if self.t1aub_gtfeat:
                widen_phi(self, GT_DIM)
                tmpl, faces = template(s["mano_npz"])
                self.register_buffer("t1aub_tmpl", tmpl, persistent=False)
                self.register_buffer("t1aub_faces", faces, persistent=False)

        def forward(self, tok, d, mask, verts, assoc, prev48, *args, **kwargs):
            if self.t1aub_eg:
                if _STATE["reparam"] and not torch.is_grad_enabled():
                    h = reparam_features(self.evgnn, tok, mask)
                else:
                    h = T1A.node_features(self.evgnn, tok, mask)
            else:
                h = tok
            if self.t1aub_gtfeat:
                h = torch.cat([h.float(), gt_features(self, mask)], -1)
            return NATIVE_NET.forward(self, h, d, mask, verts, assoc, prev48, *args, **kwargs)

    return T1aubXYZMeshNet


def install(spec):
    """Install the arm's XYZMeshNet subclass on top of the class installed now (call after s45tr_t1a.install for EG
    arms). `spec`: {"eg": bool, "readout": {"slot_dim", "hidden"} or None, "gtfeat": bool, "mano_npz": path}."""
    if _STATE["cls"] is not None:
        raise RuntimeError("T1A-UB already installed")
    base = XM.XYZMeshNet
    if bool(spec["eg"]) != T1A.installed():
        raise RuntimeError("T1AUB.EG must match the s45tr_t1a hook (install it first for event-graph arms)")
    cls = make_class(base, spec)
    _STATE.update(spec=dict(spec), base=base, cls=cls)
    XM.XYZMeshNet = cls
    if spec.get("gtfeat"):
        install_gtfeat_route()
    return installed()


def install_gtfeat_route():
    if XAG.route_association is not NATIVE_ROUTE:
        raise RuntimeError("another hook already patched semkine.xag.route_association")
    XAG.route_association = passthrough_route
    _STATE["gtfeat_route"] = True


def set_reparam(on: bool):
    """Deployment form of T1a's EdgeConv (no-gradient forwards only); returns the previous setting."""
    old = _STATE["reparam"]
    _STATE["reparam"] = bool(on)
    return old


def uninstall():
    if _STATE["base"] is not None and XM.XYZMeshNet is _STATE["cls"]:
        XM.XYZMeshNet = _STATE["base"]
    if _STATE["gtfeat_route"] and XAG.route_association is passthrough_route:
        XAG.route_association = NATIVE_ROUTE
    _STATE.update(spec=None, base=None, cls=None, stash=None, gtfeat_route=False, reparam=False)


def installed():
    if _STATE["cls"] is None:
        return None
    return {**_STATE["spec"], "gtfeat_route": _STATE["gtfeat_route"], "reparam": _STATE["reparam"]}


__all__ = ["GT_DIM", "MACS", "NATIVE_NET", "READOUT_SEED", "count_macs", "gt_features", "install", "installed",
           "make_class", "passthrough_route", "rebuild_readout", "reparam_features", "set_reparam", "template",
           "uninstall", "widen_phi"]
