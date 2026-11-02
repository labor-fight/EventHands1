"""T1A-UB round 5: diagnostic oracle cells on top of the round-1 and round->=2 layers (CPU only).

An oracle cell = a routed arm's trunk checkpoints evaluated with the label routes (S45-SM's PFA oracle, as the round-1 `gt`
arm) in place of a router at test time: how much of a deployable network's remaining gap is the router's accuracy, with the
trunk fixed. It is label-assisted (never deployable, never in a budget). Registered in `tools/t1aub/registry5_r<k>.json`
({"round": k, "cells": {name: {"trunk": arm, "router": "gt", "note"}}}); its config is the trunk's formal config with the
label route block and the XAG source root, as the `gt` arm's. `evaluate5.py` runs the frozen round-1 evaluator with
`common.resolve` / `load_config_for` / `install_for_config` / `family` redirected here; names this layer does not register
fall through to tools/t1aub/r2.py, then to the round-1 layer.
"""
from __future__ import annotations

import json

from tools.t1aub import common as C
from tools.t1aub import r2

PACKAGE = C.PACKAGE
CELL5_KEYS = {"trunk", "router", "note"}


def registry_files():
    return sorted(PACKAGE.glob("registry5_r*.json"), key=lambda p: int(p.stem.split("_r")[1]))


def registry():
    out = {"rounds": {}, "cells": {}}
    r1, rr = C.registry(), r2.registry()
    taken = set(r1["arms"]) | set(r1["cells"]) | set(rr["arms"]) | set(rr["cells"]) | {"t1a"}
    for f in registry_files():
        r = json.loads(f.read_text())
        k = int(r["round"])
        C.require(k >= 5 and f.stem == f"registry5_r{k}", "oracle-cell registry file name differs from its round: " + f.name)
        out["rounds"][k] = r.get("prereg")
        for name, spec in (r.get("cells") or {}).items():
            C.require(name not in taken and name not in out["cells"], "network registered twice: " + name)
            out["cells"][name] = {**spec, "round": k}
    return out


def cell_spec(name):
    cells = registry()["cells"]
    C.require(name in cells, "unregistered oracle cell " + repr(name))
    s = cells[name]
    C.require(set(s) - {"round"} <= CELL5_KEYS and s["router"] == "gt", "oracle cells: trunk + router gt")
    return s


def is_r5(name):
    base, _ = C.split_step(name)
    return base in registry()["cells"]


def config_path(name):
    return C.CONFIGS / f"t1aub_cell_{name}.yaml"


def expected_cell(name, base=None):
    spec = cell_spec(name)
    trunk = C.arm_spec(spec["trunk"])
    C.require("formal" in C.phases(spec["trunk"]) and trunk.get("route"), "an oracle cell's trunk is a routed arm")
    cfg = C.expected_raw(spec["trunk"], "formal", base)
    cfg["DATA"]["XAG_SOURCE_ROOT"] = str(C.XAG_SOURCES)
    cfg["T1AUB"]["ROUTE"] = C._route_block({"router": "gt", "assoc": "pfa"})
    cfg["T1AUB"]["ROUND"] = int(spec["round"])
    cfg["T1AUB"]["CELL"] = name
    cfg["T1AUB"]["TRUNK"] = spec["trunk"]
    cfg["T1AUB"]["ORACLE"] = True
    cfg["TRAIN"].pop("LR_SCHEDULE", None)
    return cfg


def round_files(k):
    base = C.s45_raw()
    return [(config_path(n), C.dump(expected_cell(n, base))) for n, s in registry()["cells"].items() if s["round"] == k]


def check_configs(k):
    return {str(p): p.is_file() and p.read_text() == text for p, text in round_files(k)}


# --------------------------------------------------------------------------- resolution
def resolve(name):
    if not is_r5(name):
        return r2.resolve(name)
    base, step = C.split_step(name)
    config = config_path(base)
    C.config_frozen(config)
    return C.arm_checkpoint(cell_spec(base)["trunk"], step), config, "cell", step


def load_config_for(name):
    if not is_r5(name):
        return r2.load_config_for(name)
    from config import load_config
    return load_config(str(resolve(name)[1]))


def install_for_config(cfg, *, evaluation, fast=False, reparam=False):
    return r2.install_for_config(cfg, evaluation=evaluation, fast=fast, reparam=reparam)


def family(name):
    return "cell" if is_r5(name) else r2.family(name)


def patch_common():
    """Redirect the frozen round-1 layer's resolution, configs and hooks to this module (in this process)."""
    C.resolve, C.load_config_for, C.install_for_config, C.family = resolve, load_config_for, install_for_config, family


__all__ = [n for n in dir() if not n.startswith("_")]
