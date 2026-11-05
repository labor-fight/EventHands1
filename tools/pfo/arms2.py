"""PFO phase-2b arms (docs/EXPERIMENT_RECORDS.md#pfo-p2b-prereg-20261103): one change against the base arm B
(tools/pfo/arms.py `b`), configs generated from B's.

Each arm = {"trunk": variant of semkine/pfo_trunk.py ("base", "nomp", ...) or semkine/pfo_trunk2.py ("pos", "hframe",
"noprev"), "set": {dotted config key: value}, "bank": {"root", "p"} or None (own-loop prior, semkine/s45tr_prior.py N4),
"router": None (sr1) or a PFO router name (tools/pfo/router_train.py)}.
"""
from __future__ import annotations

import copy

import yaml

from .arms import SEEDS, build_config as build_b, run_dir, run_name  # noqa: F401
from .common import CONFIGS, OUT, SM, SR1, T0P, require

TRUNK2 = ("pos", "hframe", "noprev")
_SEC = "pfo-p2b-prereg-20261103"
PFASRDZ_BANK = str(SM / "banks/pfasrdz")
ARMS2 = {
    "noise": {"trunk": "base", "set": {"TRACK.PREV_NOISE_P_SMALL": 0.2, "TRACK.PREV_NOISE_P_LARGE": 0.5,
                                       "TRACK.PREV_NOISE_P_CORR": 0.3}, "section": _SEC},
    "bank": {"trunk": "base", "bank": {"root": PFASRDZ_BANK, "p": 0.5}, "section": _SEC},
    "pos": {"trunk": "pos", "section": _SEC},
    "noprev": {"trunk": "noprev", "section": _SEC},
    "t0ptrain": {"trunk": "base", "train_router": "t0p", "section": _SEC},
}
#: training-time routers (frozen, events-only; the deployed router stays the config's / sr1)
TRAIN_ROUTERS = {"t0p": ("t0p", str(T0P))}


def spec(arm):
    require(arm in ARMS2, "unregistered phase-2b arm " + arm)
    return ARMS2[arm]


def _set(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def build_config(arm, seed, phase="formal"):
    s = spec(arm)
    cfg = copy.deepcopy(build_b("b", seed, phase))
    name = run_name(arm, seed, phase)
    cfg["TRAIN"].update(OUTPUT_DIR=str(OUT / "runs" / name), RUN_NAME=name)
    cfg["EVAL"]["OUTPUT_DIR"] = str(OUT / "runs" / name / "eval")
    cfg["PFO"].update(ARM=arm, TRUNK=s["trunk"], SECTION=s["section"])
    if s.get("router"):
        from .router_train import checkpoint
        cfg["PFO"].update(ROUTER=s["router"], ROUTER_KIND="pfr", ROUTER_CHECKPOINT=str(checkpoint(s["router"])))
    if s.get("train_router"):
        # training reads the frozen training-time router's routes; evaluation and deployment the PFO block's EVAL_* router
        kind, ckpt = TRAIN_ROUTERS[s["train_router"]]
        cfg["PFO"].update(EVAL_ROUTER=cfg["PFO"]["ROUTER"], EVAL_ROUTER_KIND=cfg["PFO"]["ROUTER_KIND"],
                          EVAL_ROUTER_CHECKPOINT=cfg["PFO"]["ROUTER_CHECKPOINT"],
                          ROUTER=s["train_router"], ROUTER_KIND=kind, ROUTER_CHECKPOINT=ckpt)
    if s.get("bank"):
        cfg["PFO"]["BANK"] = dict(s["bank"])
    for k, v in s.get("set", {}).items():
        _set(cfg, k, v)
    return cfg


def config_path(arm, seed, phase="formal"):
    return CONFIGS / f"{run_name(arm, seed, phase)}.yaml"


def write_config(arm, seed, phase="formal"):
    p = config_path(arm, seed, phase)
    text = yaml.safe_dump(build_config(arm, seed, phase), sort_keys=False)
    if p.exists():
        require(p.read_text() == text, "config on disk differs from the registry: " + str(p))
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


__all__ = ["ARMS2", "PFASRDZ_BANK", "TRAIN_ROUTERS", "TRUNK2", "build_config", "config_path", "spec", "write_config"]
