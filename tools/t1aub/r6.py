"""T1A-UB round 6: router-location features with S45's own association ("rfeat"; CPU only).

The round-1 next step written for gtf = FEATURE_SUFFICES (docs/EXPERIMENT_RECORDS.md#t1aub-r1-prereg-20261102): the
deployable version of gtf, i.e. every sampled event gets the surface location a router predicts from the events alone
(the rest-template point of its routed face, and a valid flag: semkine/t1aub.py's gtfeat feature, fed from the router
instead of the labels), while the association stays S45's own ray casting against the previous mesh (no PFA, no face
route). The arm is a round-1-format arm (`registry_r<k>.json`) with both `gtfeat` and `route` = {router, assoc
"face_route"}: a combination the frozen round-1 generator refuses, so this layer supplies
  expected_raw       the arm's config: the same arm with PFA routes, then T1AUB.ROUTE.ASSOC face_route and T1AUB.GTFEAT;
                     no XAG source root (no labels anywhere)
  install_for_config the router first in its `face_route` mode (it replaces only the sampled sources), then this study's
                     subclass with the location feature (its pass-through keeps S45's association and hands the router's
                     sources to the feature); evaluation attaches no label sources
and `patch_common()` redirects the frozen round-1 layer to them on top of tools/t1aub/r5.py's resolution. Every other
network falls through unchanged.
"""
from __future__ import annotations

import copy

from tools.t1aub import common as C
from tools.t1aub import r5

_NATIVE = {"expected_raw": C.expected_raw, "install_for_config": C.install_for_config, "arm_spec": C.arm_spec}


def is_rfeat_spec(spec):
    route = spec.get("route") or {}
    return bool(spec.get("gtfeat")) and route.get("router") not in (None, "gt") and route.get("assoc") == "face_route"


def expected_raw(arm, phase, base=None):
    spec = _NATIVE["arm_spec"](arm)
    if not is_rfeat_spec(spec):
        return _NATIVE["expected_raw"](arm, phase, base)
    proxy = {k: copy.deepcopy(v) for k, v in spec.items() if k != "gtfeat"}
    proxy["route"] = {**proxy["route"], "assoc": "pfa"}
    native_spec = C.arm_spec
    C.arm_spec = lambda name: proxy if name == arm else native_spec(name)
    try:
        cfg = _NATIVE["expected_raw"](arm, phase, base)
    finally:
        C.arm_spec = native_spec
    C.require("XAG_SOURCE_ROOT" not in cfg["DATA"], "router-location features read no labels")
    cfg["T1AUB"]["ROUTE"]["ASSOC"] = "face_route"
    cfg["T1AUB"]["GTFEAT"] = True
    return cfg


def round_files(k):
    """[(path, text)] of every rfeat arm config of round k (both phases)."""
    base = C.s45_raw()
    out = []
    for arm, spec in C.registry()["arms"].items():
        if spec["round"] == k and is_rfeat_spec(spec):
            for phase in C.phases(arm):
                out.append((C.config_path(arm, phase), C.dump(expected_raw(arm, phase, base))))
    return out


def check_configs(k):
    return {str(p): p.is_file() and p.read_text() == text for p, text in round_files(k)}


def install_for_config(cfg, *, evaluation, fast=False, reparam=False):
    b = cfg["T1AUB"]
    route = b.get("ROUTE") or {}
    if not (b.get("GTFEAT") and route.get("ASSOC") == "face_route" and route.get("ROUTER") not in (None, "gt")):
        return _NATIVE["install_for_config"](cfg, evaluation=evaluation, fast=fast, reparam=reparam)
    from semkine import t1aub as M
    from semkine import s45tr_t1a
    from tools.s45sm.routing import NEEDS, register_kinds
    state = {"arm": b["ARM"], "eg": bool(b["EG"])}
    if b["EG"]:
        C.require(b["T1A_GNN"] == C.T1A_GNN, "T1a event-graph settings changed")
        s45tr_t1a.install(b["T1A_GNN"])
    register_kinds(NEEDS[route["ROUTER_KIND"]])
    if fast:
        from semkine import s45sm_fast as R
    else:
        from semkine import s45sm as R
    r = R.install(route["ROUTER_KIND"], route["ROUTER_CHECKPOINT"], "face_route", float(route["ONHAND_TAU"]))
    ro = b.get("READOUT")
    M.install({"eg": bool(b["EG"]), "gtfeat": True, "mano_npz": cfg["MANO"]["NPZ"],
               "readout": {"slot_dim": int(ro["SLOT_DIM"]), "hidden": int(ro["HIDDEN"])} if ro else None})
    state["model"] = M.installed()
    state["route"] = {k: r[k] for k in ("router", "checkpoint", "assoc", "onhand_tau")}
    state["route"]["fast"] = bool(fast)
    state["rfeat"] = True
    if evaluation:
        from semkine import eval_track as ET
        ET.attach_xag_sources = lambda batch, *args, **kwargs: batch
    if reparam:
        C.require(b["EG"], "reparam applies to the event graph")
        M.set_reparam(True)
        state["reparam"] = True
    return state


def patch_common():
    """r5's resolution (oracle cells, round->=2 cells, round-1 layer) plus this layer's configs and hooks."""
    r5.patch_common()
    C.expected_raw, C.install_for_config = expected_raw, _chained_install()


def _chained_install():
    resolved = C.install_for_config         # r5 / r2's (arbitration) after r5.patch_common()

    def install(cfg, *, evaluation, fast=False, reparam=False):
        b = cfg["T1AUB"]
        route = b.get("ROUTE") or {}
        if b.get("GTFEAT") and route.get("ASSOC") == "face_route" and route.get("ROUTER") not in (None, "gt"):
            return install_for_config(cfg, evaluation=evaluation, fast=fast, reparam=reparam)
        return resolved(cfg, evaluation=evaluation, fast=fast, reparam=reparam)
    return install


__all__ = ["check_configs", "expected_raw", "install_for_config", "is_rfeat_spec", "patch_common", "round_files"]
