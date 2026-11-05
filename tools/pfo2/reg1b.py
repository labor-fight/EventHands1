"""PFO2 arm (b) registry (docs/EXPERIMENT_RECORDS.md#pfo2-s1b-prereg-20261103): tools/pfo2/reg1.py's registry for arms
whose trunk is a semkine/pfo2_trunk_b.py variant ("<form>[+self|+rewire]+mom"), read from tools/pfo2/arms1b_*.json.
`build_config` is reg1's procedure (the base's recipe, this arm's names and PFO block, the S1 block) on these arms.
"""
from __future__ import annotations

import json
from pathlib import Path

from tools.pfo2 import arms as A
from tools.pfo2 import reg1 as G
from tools.pfo2.common import require

HERE = Path(__file__).resolve().parent


def _load():
    from semkine.pfo2_trunk_b import parse
    arms = {}
    for f in sorted(HERE.glob("arms1b_*.json")):
        for name, s in json.loads(f.read_text()).items():
            require(name not in arms and name not in A.ARMS and name not in G.ARMS1, f"arm {name} registered twice")
            form, _, mom = parse(s["trunk"])
            require(mom and A.recipe(s["base"])["trunk"] == form, f"{name}: a +mom variant of its base's form")
            arms[name] = s
    return arms


ARMS1B = _load()


def spec(arm):
    require(arm in ARMS1B, "unregistered PFO2 arm (b) " + arm)
    return ARMS1B[arm]


def build_config(arm, seed, phase="formal"):
    spec(arm)
    saved = dict(G.ARMS1)
    try:
        G.ARMS1[arm] = ARMS1B[arm]
        return G.build_config(arm, seed, phase)
    finally:
        G.ARMS1.clear()
        G.ARMS1.update(saved)


def config_path(arm, seed, phase="formal"):
    return A.config_path(arm, seed, phase)


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


__all__ = ["ARMS1B", "build_config", "config_path", "spec", "write_config"]
