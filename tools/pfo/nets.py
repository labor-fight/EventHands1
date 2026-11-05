"""PFO networks (docs/EXPERIMENT_RECORDS.md#pfo-prereg-20261103): one loader for every network the study reads.

`load(name, route=None, tau=None, device)` returns a `Net` whose `step(ctx, end, prev)` maps the 51-D state at the window
start to the state after the 50 ms packet ending at `end` (one process = one network: the routing hooks are module
patches). Kinds:
  routed  a trunk trained with PFA routes (S45-SM / PFO arms); test-time routes = the config's router unless `route`
          names another one:
            router            the config's router (deployed)
            sr1 / t0p         that router instead (zero-training swap)
            label             the XAG label routes (diagnostic only, not deployable)
            label_onhand      router locations, label on-hand set
            label_part        router on-hand set and template point, label part (nearest face of the label part)
            label_onhand_part label on-hand set and part, router template point
            label_loc         router on-hand set, label face / barycentrics where a label exists
            label_shuffle     label on-hand set, label (face, bary) permuted among the routed nodes of the packet
            router_shuffle    router on-hand set, router (face, bary) permuted among the routed nodes of the packet
          mixes use sr1 (part head) as the router; `tau` overrides the on-hand threshold.
  label   a trunk trained with label routes (pfagtdz): label routes, or any router / mix above
  s43     the dense LNES tracker S43 (s43n_sp), W50 LNES, `forward(x, prev)`
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import NETS, SR1, T0P, XAG_SOURCE_ROOT, require

ROUTES = ("router", "sr1", "t0p", "label", "label_onhand", "label_part", "label_onhand_part", "label_loc",
          "label_shuffle", "router_shuffle")
MIXES = ("label_onhand", "label_part", "label_onhand_part", "label_loc", "label_shuffle", "router_shuffle")
NEEDS_LABELS = ("label",) + MIXES


@dataclass
class Context:
    seq: str
    events: object
    offsets: object
    tsub: object
    aux: dict
    gt: object
    betas: object
    K: object


def sequence(root, directory, seq, device):
    import torch
    from semkine import eval_track as ET
    events, offsets, aux, gt = ET.load_sequence(Path(root), directory, seq)
    tsub_path = Path(root) / directory / f"{seq}_tsub.npy"
    tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
    betas = torch.tensor(np.asarray(aux["betas"], np.float32), device=device).view(1, -1)
    K = torch.tensor(np.asarray(aux["camera_K"], np.float32), device=device).view(1, 3, 3)
    return Context(seq, events, offsets, tsub, aux, gt, betas, K)


class Net:
    def __init__(self, name, kind, model, cfg, device, route, tau, identity):
        self.name, self.kind, self.model, self.cfg, self.device = name, kind, model, cfg, device
        self.route, self.tau, self.identity = route, tau, identity

    def bind(self, ctx):
        self.model.set_hand_context(ctx.betas, ctx.K)

    def step(self, ctx, end, prev):
        import torch
        from semkine import eval_track as ET
        prev_t = torch.from_numpy(np.ascontiguousarray(prev, np.float32)).view(1, -1).to(self.device)
        if self.kind == "s43":
            from semkine import events as EV
            x = torch.from_numpy(ET.build_lnes(ctx.events, ctx.offsets, int(end), 50, EV.event_channels(self.cfg)))
            pred = self.model(x.unsqueeze(0).to(self.device), prev_t)
        else:
            ev5 = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, int(end), 50)
            packet = ET.make_eval_packet(ev5, prev_t, ctx.betas, ctx.K, 50, self.device)
            packet = ET.attach_xag_sources(packet, self.cfg, ctx.seq, ctx.offsets, int(end), 50, self.device)
            pred = self.model.forward_packet(packet)
        value = pred.detach().cpu().numpy()[0].copy()
        if value.shape != (51,) or not np.isfinite(value).all():
            raise FloatingPointError(f"nonfinite endpoint retained as failure: {ctx.seq}:{end}")
        return value


# --------------------------------------------------------------------------- route mixing (diagnostics only)
_MIX = {"mode": None, "gen": None}


def _faces_in_part(net, part, point):
    import torch
    from semkine.s45tr_t0 import project_bary
    cand, ok = net.part_faces[part], net.part_faces_valid[part]
    d = (net.centroid[cand] - point.unsqueeze(-2)).norm(dim=-1).masked_fill(~ok, float("inf"))
    face = cand.gather(-1, d.argmin(-1, keepdim=True)).squeeze(-1)
    return face, project_bary(point, net.tmpl[net.faces[face]])


def _permute(face, bary, routed):
    import torch
    face, bary = face.clone(), bary.clone()
    for b in range(routed.shape[0]):
        idx = routed[b].nonzero().flatten()
        if idx.numel() > 1:
            p = idx[torch.randperm(idx.numel(), generator=_MIX["gen"]).to(idx.device)]
            face[b, idx], bary[b, idx] = face[b, p], bary[b, p]
    return face, bary


def mixed_sources(batch, src, mask, *, require_depth=False):
    """The router's and the labels' routes at S45's nodes, combined per `_MIX['mode']` (see the module docstring)."""
    import torch
    from semkine import s45sm as SM
    from semkine import s45sm_fast as FAST
    require(not require_depth, "routes only")
    st = SM._STATE
    net = SM._net(src.device)
    out = FAST.router_forward(net, batch.events, batch.ptr, batch.delta_t_s, src, mask)
    r_sel = mask & (torch.sigmoid(out["onhand"].float()) >= st["onhand_tau"])
    r_face, r_bary = FAST.route(net, st["router"], out, mask)          # a router location for every live node
    lab = SM._NATIVE["sample_sources"](batch, src, mask)
    F_ = net.faces.shape[0]
    l_bary = lab["xag_bary"].float()
    l_ok = lab["xag_valid"] & mask & (lab["xag_face"] >= 0) & (lab["xag_face"] < F_)
    l_ok = l_ok & torch.isfinite(l_bary).all(-1) & (l_bary >= 0).all(-1) & (l_bary.sum(-1) > 0)
    l_face = lab["xag_face"].clamp(0, F_ - 1)
    l_bary = torch.where(l_ok.unsqueeze(-1), l_bary / l_bary.sum(-1, keepdim=True).clamp_min(1e-12),
                         torch.zeros_like(l_bary))
    point = out["coord"].float() * 0.01
    mode = _MIX["mode"]
    if mode == "label_onhand":
        routed, face, bary = l_ok, r_face, r_bary
    elif mode in ("label_part", "label_onhand_part"):
        routed = r_sel if mode == "label_part" else l_ok
        part = torch.where(l_ok, net.face_part[l_face], out["part"].float().argmax(-1))
        face, bary = _faces_in_part(net, part, point)
    elif mode == "label_loc":
        routed = r_sel
        face = torch.where(l_ok, l_face, r_face)
        bary = torch.where(l_ok.unsqueeze(-1), l_bary, r_bary)
    elif mode == "label_shuffle":
        routed = l_ok
        face, bary = _permute(l_face, l_bary, routed)
    elif mode == "router_shuffle":
        routed = r_sel
        face, bary = _permute(r_face, r_bary, routed)
    else:
        raise ValueError(mode)
    face = torch.where(routed, face, torch.zeros_like(face))
    zero = torch.zeros_like(face)
    return {"xag_source_index": zero, "xag_source_ms": zero, "xag_part": net.face_part[face], "xag_face": face,
            "xag_bary": bary.to(torch.float32), "xag_valid": routed & mask}


# --------------------------------------------------------------------------- loading
def _router_for(route, cfg):
    s = cfg["S45SM"]
    if route in (None, "router"):
        return s["ROUTER_KIND"], s["ROUTER_CHECKPOINT"]
    if route == "t0p":
        return "t0p", str(T0P)
    return "sr", str(SR1)                                                   # sr1 and every mix


def load(name, route=None, tau=None, device=None, spec=None):
    """Load network `name` (a key of `common.NETS`, or `spec` = {kind, checkpoint, config}) with its test-time routes."""
    import copy
    import torch
    from config import load_config
    from model import MNISTModel
    device = device or torch.device("cuda")
    spec = spec or NETS[name]
    kind = spec["kind"]
    cfg = copy.deepcopy(load_config(str(spec["config"])))
    identity = {"net": name, "kind": kind, "checkpoint": str(spec["checkpoint"]), "config": str(spec["config"])}
    if kind == "s43":
        require(route in (None, "router") and tau is None, "S43 has no routes")
        model = MNISTModel.load_from_checkpoint(str(spec["checkpoint"]), cfg=cfg, map_location=device, strict=True)
        return Net(name, kind, model.to(device).eval(), cfg, device, None, None, identity)
    route = route or ("label" if kind == "label" else "router")
    require(route in ROUTES, f"unknown route {route}")
    from semkine import eval_track as ET
    from semkine import s45sm_fast as FAST
    from semkine import xag as XAG
    if route == "label":
        from semkine import s45sm_oracle as O
        cfg["DATA"]["XAG_SOURCE_ROOT"] = str(XAG_SOURCE_ROOT)
        O.install()
        identity["routing"] = {"router": "label", "assoc": "pfa"}
    else:
        rkind, rckpt = _router_for(route, cfg) if kind == "routed" else (("t0p", str(T0P)) if route == "t0p"
                                                                          else ("sr", str(SR1)))
        if rkind in ("dw", "srp"):
            from tools.s45sm.routing import NEEDS, register_kinds
            register_kinds(NEEDS[rkind])
        t = float(cfg.get("S45SM", {}).get("ONHAND_TAU", 0.5)) if tau is None else float(tau)
        state = FAST.install(rkind, rckpt, "pfa", t)
        identity["routing"] = {k: state[k] for k in ("router", "checkpoint", "assoc", "onhand_tau")}
        if route in MIXES:
            cfg["DATA"]["XAG_SOURCE_ROOT"] = str(XAG_SOURCE_ROOT)
            _MIX.update(mode=route, gen=torch.Generator().manual_seed(0))
            XAG.sample_sources = mixed_sources
            identity["routing"]["mix"] = route
        else:
            ET.attach_xag_sources = lambda batch, *args, **kwargs: batch     # deployed: no label source is read
            cfg["DATA"].pop("XAG_SOURCE_ROOT", None)
    identity["route"] = route
    model = MNISTModel.load_from_checkpoint(str(spec["checkpoint"]), cfg=cfg, map_location=device, strict=True)
    return Net(name, kind, model.to(device).eval(), cfg, device, route, tau, identity)


__all__ = ["Context", "MIXES", "NEEDS_LABELS", "Net", "ROUTES", "load", "mixed_sources", "sequence"]
