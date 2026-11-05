"""PFO router registry, data-driven: the routers of tools/pfo/router_train.py (unchanged) plus every registered phase file
tools/pfo/routers_<phase>.json ({name: {"design": {...SlimSR...}, "kd": null | KD dict, "steps": 6000}}); written once
before any reading of its routers and frozen with its section."""
from __future__ import annotations

import json
from pathlib import Path

from . import router_train as RT
from .common import require

HERE = Path(__file__).resolve().parent


def _load():
    routers = dict(RT.ROUTERS)
    for f in sorted(HERE.glob("routers_*.json")):
        for name, s in json.loads(f.read_text()).items():
            require(name not in routers, f"router {name} registered twice ({f.name})")
            routers[name] = s
    return routers


ROUTERS3 = _load()
KD = RT.KD
