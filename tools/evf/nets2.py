"""EVF network kinds "evf" and "evf2" (semkine/evf_model2.py) for PFO's frozen readout engine and cost harness (docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104).

`install()` wraps `tools.pfo.nets.load` (the engine imports that module and calls `N.load` at run time): a spec of kind
"evf" ({kind, checkpoint, config}) loads `semkine.evf_model.EVFModel` and returns PFO's `Net`, whose `step` cuts the
50 ms window, builds the evaluation packet and calls `forward_packet` exactly as for the routed networks (no route, no
label source is read). Every other kind goes to PFO's loader unchanged.
"""
from __future__ import annotations

_INSTALLED = {}


def load_evf(name, spec, device):
    """`spec.kind` evf (EVFModel) or evf2 (EVFModel2)."""
    import copy
    import torch
    from config import load_config
    from semkine import eval_track as ET
    from semkine.evf_model import EVFModel
    from semkine.evf_model2 import EVFModel2
    from tools.pfo import nets as PN
    cfg = copy.deepcopy(load_config(str(spec["config"])))
    cls = EVFModel2 if spec["kind"] == "evf2" else EVFModel
    model = cls.load_from_checkpoint(str(spec["checkpoint"]), cfg=cfg, map_location=device or torch.device("cuda"),
                                          strict=True)
    ET.attach_xag_sources = lambda batch, *args, **kwargs: batch          # deployed: no label source is read
    identity = {"net": name, "kind": spec["kind"], "checkpoint": str(spec["checkpoint"]), "config": str(spec["config"]),
                "route": None, "evf": cfg["EVF"]}
    return PN.Net(name, spec["kind"], model.to(device).eval(), cfg, device, None, None, identity)


def install():
    from tools.pfo import nets as PN
    if "load" in _INSTALLED:
        return
    native = PN.load
    _INSTALLED["load"] = native

    def load(name, route=None, tau=None, device=None, spec=None):
        if spec is not None and spec["kind"] in ("evf", "evf2"):
            if route not in (None, "router") or tau is not None:
                raise ValueError("the EVF networks have no routes")
            return load_evf(name, spec, device)
        return native(name, route=route, tau=tau, device=device, spec=spec)
    PN.load = load


__all__ = ["install", "load_evf"]
