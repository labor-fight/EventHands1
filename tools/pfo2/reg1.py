"""PFO2 stage-1 arm registry (docs/EXPERIMENT_RECORDS.md#pfo2-s1-prereg-20261103): every arm is the base B2's recipe with
one change, trained by tools/pfo2/train_s1.py (deterministic) and read by tools/pfo2/run_s1.py.

Arm files tools/pfo2/arms1_<stage>.json ({arm: spec}; the prefix differs from tools/pfo2/arms.py's `arms_*.json`, which
must not see these files), written once before any reading of their arms. A spec:
  base      the PFO2 base arm (tools/pfo2/arms.py, "fl1" or "fgs") whose recipe the arm starts from
  trunk     the semkine/pfo2_trunk.py variant string ("l1", "l1+self+disp", ...); its form must be the base's
  set       optional {dotted config key: value} applied after the base recipe
  losses    optional {"disp": weight} (semkine/pfo2_train.py)
  dual      optional {"lam_max", "start", "end"}: arm A2's double forward (semkine/pfo2_dual.py)
  section   the registration anchor; other keys are bookkeeping (replay_of, control_of, ...)
Configs: tools/pfo2/arms.py's config of the base (PFO's recipe through tools/pfo/arms2.py), with this arm's run name,
directories, PFO block (MODULE "pfo2", TRUNK = the variant) and an S1 block (losses). Phases as tools/pfo2/arms.py.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from tools.pfo2 import arms as A
from tools.pfo2.common import SEEDS, require

HERE = Path(__file__).resolve().parent


def _set(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def _load():
    from semkine.pfo2_trunk import parse
    arms = {}
    for f in sorted(HERE.glob("arms1_*.json")):
        for name, s in json.loads(f.read_text()).items():
            require(name not in arms and name not in A.ARMS, f"arm {name} registered twice ({f.name})")
            base = A.spec(s["base"])
            form, _ = parse(s["trunk"])
            require(A.recipe(s["base"])["trunk"] == form, f"{name}: trunk form differs from its base's")
            require(base.get("module", "trunk4") == "trunk4", f"{name}: base must be a trunk4 arm")
            arms[name] = s
    return arms


ARMS1 = _load()


def spec(arm):
    require(arm in ARMS1, "unregistered PFO2 stage-1 arm " + arm)
    return ARMS1[arm]


def build_config(arm, seed, phase="formal"):
    require(seed in SEEDS and phase in A.PHASES, "unregistered seed / phase")
    s = spec(arm)
    cfg = copy.deepcopy(A.build_config(s["base"], seed, phase))
    name = A.run_name(arm, seed, phase)
    cfg["TRAIN"].update(OUTPUT_DIR=str(A.run_dir(arm, seed, phase)), RUN_NAME=name)
    cfg["EVAL"]["OUTPUT_DIR"] = str(A.run_dir(arm, seed, phase) / "eval")
    cfg["PFO"].update(ARM=arm, TRUNK=s["trunk"], MODULE="pfo2", SECTION=s["section"], BASE_ARM=s["base"])
    for k, v in s.get("set", {}).items():
        _set(cfg, k, v)
    cfg["S1"] = {"LOSSES": dict(s.get("losses", {})), "DUAL": dict(s.get("dual", {}))}
    return cfg


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


__all__ = ["ARMS1", "build_config", "config_path", "spec", "write_config"]
