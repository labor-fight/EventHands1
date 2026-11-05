"""PFO arm registry, data-driven (docs/EXPERIMENT_RECORDS.md#pfo-p2c-prereg-20261103 and later sections): the phase-2b
arms (tools/pfo/arms2.py, unchanged) plus the arms of every registered phase file tools/pfo/arms_<phase>.json
({arm: spec}, the spec keys of arms2: trunk, set, bank, router, train_router, section). A phase file is written once,
before any reading of its arms, and frozen with its section; later phases add new files, never edit old ones.
`build_config` / `write_config` / `config_path` / `spec` behave as arms2's for every arm.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import arms2 as A2
from .common import require

HERE = Path(__file__).resolve().parent


def _load():
    arms = dict(A2.ARMS2)
    for f in sorted(HERE.glob("arms_*.json")):
        for name, s in json.loads(f.read_text()).items():
            require(name not in arms, f"arm {name} registered twice ({f.name})")
            arms[name] = s
    return arms


ARMS3 = _load()


def spec(arm):
    require(arm in ARMS3, "unregistered PFO arm " + arm)
    return ARMS3[arm]


def build_config(arm, seed, phase="formal"):
    if arm in A2.ARMS2:
        return A2.build_config(arm, seed, phase)
    saved = dict(A2.ARMS2)
    try:
        A2.ARMS2[arm] = ARMS3[arm]
        return A2.build_config(arm, seed, phase)
    finally:
        A2.ARMS2.clear()
        A2.ARMS2.update(saved)


def config_path(arm, seed, phase="formal"):
    return A2.config_path(arm, seed, phase)


def write_config(arm, seed, phase="formal"):
    import yaml
    p = config_path(arm, seed, phase)
    text = yaml.safe_dump(build_config(arm, seed, phase), sort_keys=False)
    if p.exists():
        require(p.read_text() == text, "config on disk differs from the registry: " + str(p))
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


__all__ = ["ARMS3", "build_config", "config_path", "spec", "write_config"]
