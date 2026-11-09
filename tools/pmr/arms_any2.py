"""PMR registry union 2 (docs/EXPERIMENT_RECORDS.md#pmr-s4c-prereg-20261109): tools/pmr/arms_any.py (frozen with S4b) with the
S4c registry added. tools/pmr/arms.py's interface over every
PMR registry (arms: D1; arms2: the ladder; arms3: the rounds rungs; arms4 / arms5: the S4 / S4b refinements), dispatching by
arm name,
so the rollout and reading entries of later stages can take any registered base arm.
"""
from __future__ import annotations

import importlib

from .common import require

REGISTRIES = ("tools.pmr.arms", "tools.pmr.arms2", "tools.pmr.arms3", "tools.pmr.arms4", "tools.pmr.arms5", "tools.pmr.arms6")


def _owner(arm):
    hits = [r for r in REGISTRIES if arm in importlib.import_module(r).arms()]
    require(len(hits) == 1, f"arm {arm} registered {len(hits)} times")
    return importlib.import_module(hits[0])


def arms():
    out = {}
    for r in REGISTRIES:
        out.update(importlib.import_module(r).arms())
    return out


def parse(arm):
    return _owner(arm).parse(arm)


def run_name(arm, seed, phase="formal"):
    return _owner(arm).run_name(arm, seed, phase)


def run_dir(arm, seed, phase="formal"):
    return _owner(arm).run_dir(arm, seed, phase)


def config_path(arm, seed, phase="formal"):
    return _owner(arm).config_path(arm, seed, phase)


def build_config(arm, seed, phase="formal"):
    return _owner(arm).build_config(arm, seed, phase)


def write_config(arm, seed, phase="formal"):
    return _owner(arm).write_config(arm, seed, phase)


def designs(arm):
    return _owner(arm).DESIGNS


class _Designs(dict):
    """DESIGNS lookup across the registries (tools/pmr/run.py reads DESIGNS[design][0])."""

    def __missing__(self, key):
        for r in REGISTRIES:
            d = importlib.import_module(r).DESIGNS
            if key in d:
                return d[key]
        raise KeyError(key)


DESIGNS = _Designs()
install_trunk = importlib.import_module("tools.pmr.arms").install_trunk
VARIANTS = importlib.import_module("tools.pmr.arms").VARIANTS

__all__ = ["DESIGNS", "REGISTRIES", "VARIANTS", "arms", "build_config", "config_path", "designs", "install_trunk",
           "parse", "run_dir", "run_name", "write_config"]
