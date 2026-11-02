"""T1A-UB rounds >= 2: registry, configs, hooks and network resolution on top of the frozen round-1 layer (CPU only).

Round 1's `common.py` / `evaluate.py` are frozen; later rounds register in `tools/t1aub/registry2_r<k>.json` (a file name the
round-1 registry does not read) and are resolved here. A round->=2 cell = a round-1 arm's trunk checkpoints with a test-time
router and, optionally, state arbitration of that router's parts (semkine/t1aub_arb.py: `arb` = {"mode": "point" | "ray",
"sigma_px": number | null}); its config is the trunk's formal config with the router's route block and `T1AUB.ARB`.
`evaluate2.py` runs the frozen round-1 evaluator with `common.resolve` / `load_config_for` / `install_for_config`
redirected to this module (`patch_common`), so traces, readouts and rows are produced by the same code.
"""
from __future__ import annotations

import json
from pathlib import Path

from tools.t1aub import common as C

PACKAGE = C.PACKAGE
CELL2_KEYS = {"trunk", "router", "arb", "note"}
_NATIVE = {"resolve": C.resolve, "load_config_for": C.load_config_for, "install_for_config": C.install_for_config,
           "family": C.family}


def registry_files():
    return sorted(PACKAGE.glob("registry2_r*.json"), key=lambda p: int(p.stem.split("_r")[1]))


def registry():
    out = {"rounds": {}, "cells": {}, "arms": {}}
    r1 = C.registry()
    for f in registry_files():
        r = json.loads(f.read_text())
        k = int(r["round"])
        C.require(k >= 2 and f.stem == f"registry2_r{k}", "round >= 2 registry file name differs from its round: " + f.name)
        out["rounds"][k] = r.get("prereg")
        for what in ("cells", "arms"):
            for name, spec in (r.get(what) or {}).items():
                C.require(name not in out["cells"] and name not in out["arms"] and name not in r1["arms"]
                          and name not in r1["cells"] and name != "t1a", "network registered twice: " + name)
                out[what][name] = {**spec, "round": k}
    return out


def cell_spec(name):
    cells = registry()["cells"]
    C.require(name in cells, "unregistered round>=2 cell " + repr(name))
    s = cells[name]
    C.require(set(s) - {"round"} <= CELL2_KEYS, "unknown cell fields in " + name)
    if "arb" in s:
        a = s["arb"]
        C.require(a["mode"] in ("point", "ray") and (a["mode"] == "point" or a.get("sigma_px") is None),
                  "arb: mode point (sigma_px or null) or ray")
    return s


def is_r2(name):
    base, _ = C.split_step(name)
    reg = registry()
    return base in reg["cells"] or base in reg["arms"]


def config_path(name):
    return C.CONFIGS / f"t1aub_cell_{name}.yaml"


def expected_cell(name, base=None):
    spec = cell_spec(name)
    trunk = C.arm_spec(spec["trunk"])
    C.require("formal" in C.phases(spec["trunk"]) and trunk.get("route"), "a cell's trunk is a routed round-1 arm")
    cfg = C.expected_raw(spec["trunk"], "formal", base)
    r = C.router_info(spec["router"])
    C.require(r["kind"] in ("dw", "srp", "t0p"), "round>=2 cells route with a patch router")
    cfg["DATA"].pop("XAG_SOURCE_ROOT", None)
    cfg["T1AUB"]["ROUTE"] = C._route_block({"router": spec["router"], "assoc": "pfa",
                                           "onhand_tau": trunk["route"].get("onhand_tau", 0.5)})
    cfg["T1AUB"]["ROUND"] = int(spec["round"])
    cfg["T1AUB"]["CELL"] = name
    cfg["T1AUB"]["TRUNK"] = spec["trunk"]
    if "arb" in spec:
        cfg["T1AUB"]["ARB"] = {"MODE": spec["arb"]["mode"], "SIGMA_PX": spec["arb"].get("sigma_px")}
    cfg["TRAIN"].pop("LR_SCHEDULE", None)
    return cfg


def round_files(k):
    base = C.s45_raw()
    return [(config_path(n), C.dump(expected_cell(n, base))) for n, s in registry()["cells"].items() if s["round"] == k]


def write_configs(k):
    out = []
    for path, text in round_files(k):
        C.require(not path.exists(), "config exists: " + str(path))
        path.write_text(text)
        out.append(path)
    return out


def check_configs(k):
    return {str(p): p.is_file() and p.read_text() == text for p, text in round_files(k)}


# --------------------------------------------------------------------------- resolution and hooks
def resolve(name):
    if not is_r2(name):
        return _NATIVE["resolve"](name)
    base, step = C.split_step(name)
    config = config_path(base)
    C.config_frozen(config)
    return C.arm_checkpoint(cell_spec(base)["trunk"], step), config, "cell", step


def load_config_for(name):
    if not is_r2(name):
        return _NATIVE["load_config_for"](name)
    from config import load_config
    return load_config(str(resolve(name)[1]))


def install_for_config(cfg, *, evaluation, fast=False, reparam=False):
    state = _NATIVE["install_for_config"](cfg, evaluation=evaluation, fast=fast, reparam=reparam)
    arb = cfg["T1AUB"].get("ARB")
    if arb:
        from semkine import t1aub_arb as A
        state["arb"] = A.install(arb["SIGMA_PX"], mode=arb["MODE"])
    return state


def family(name):
    return "cell" if is_r2(name) else _NATIVE["family"](name)


def patch_common():
    """Redirect the frozen round-1 layer's resolution, configs and hooks to this module (in this process)."""
    C.resolve, C.load_config_for, C.install_for_config, C.family = resolve, load_config_for, install_for_config, family


def test_router(name):
    base, _ = C.split_step(name)
    return cell_spec(base)["router"] if is_r2(name) else C.test_router(name)


__all__ = [n for n in dir() if not n.startswith("_")]
