"""S45-SM arm configs: one layered generator (and the check that the YAML files on disk are exactly its output).

An arm's YAML = S45 (`c1sf_so3_s3407`) + S45-AT dz's depth augmentation (`dz`) + `MODEL.XAG_ASSOC: face_route` with
the routes of a frozen events-only router (`S45SM` block) + identity / budget / paths, and then, per registry field:
  round >= 2   `S45SM.FAST` (the round-1 files have no such key)
  bank         `S45SM.BANK {FROM, ROOT, P}`
  sampler      `S45SM.SAMPLER` + `S45SM.REPEAT_OF` (a repeat equals its arm in router / assoc / onhand_tau / dz / fast / bank)
  lr_schedule  `TRAIN.LR_SCHEDULE: cosine` in the formal run only (the warmup of 500 steps it relies on is checked)
An oracle (label-route PFA) is S45-AT t1ddz's YAML with the oracle identity.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .paths import CONFIGS, ROOT, ROOT_ALIAS, S45_CONFIG, SAME_STEPS, T1DDZ_CONFIG, T1DDZ_IDENTITY, require
from .registry import (arm_spec, bank_dir, config_path, oracle_spec, oracles, output_dir, phases, registry, router_checkpoint,
                       router_spec, run_name, sampler, schedule)

REPEAT_SAME = ("router", "assoc", "onhand_tau", "dz", "fast", "bank")


def base_config():
    return yaml.safe_load(S45_CONFIG.read_text())


def oracle_base_config():
    return yaml.safe_load(T1DDZ_CONFIG.read_text())


def expected_raw(arm, phase, s45_raw):
    """The arm's YAML content (see the module docstring); `s45_raw` is S45's config."""
    require(phase in ("preflight", "formal"), "unregistered phase")
    spec = arm_spec(arm)
    rspec = router_spec(spec["router"])
    cfg = copy.deepcopy(s45_raw)
    require(cfg.pop("C1SF", None) == {"ARM": "so3"}, "S45 config identity changed")
    require("XAG_ASSOC" not in cfg["MODEL"] and "XAG_GNN" not in cfg["MODEL"], "S45 config changed")
    if spec["dz"]:
        cfg["AUG"]["DOMRAND"]["SCALE_MODE"] = "depth"
    cfg["MODEL"]["XAG_ASSOC"] = "face_route"
    cfg["MODEL"]["XAG_GNN"] = "legacy"
    cfg["S45SM"] = {"ARM": arm, "ROUTER": spec["router"], "ROUTER_KIND": rspec["kind"],
                    "ROUTER_CHECKPOINT": str(router_checkpoint(spec["router"])), "ASSOC": spec["assoc"],
                    "ONHAND_TAU": float(spec["onhand_tau"])}
    out = output_dir(arm, phase)
    cfg["TRAIN"]["RUN_NAME"] = run_name(arm, phase)
    cfg["TRAIN"]["OUTPUT_DIR"] = str(out)
    cfg["EVAL"]["OUTPUT_DIR"] = str(out / "eval")
    if phase == "preflight":
        cfg["TRAIN"]["MAX_STEPS"] = 50
        cfg["TRAIN"]["VAL_CHECK_INTERVAL"] = 25
    if spec["round"] >= 2:
        cfg["S45SM"]["FAST"] = bool(spec.get("fast", False))
    if spec.get("bank"):
        b = spec["bank"]
        cfg["S45SM"]["BANK"] = {"FROM": b["from"], "ROOT": str(bank_dir(b["from"])), "P": float(b.get("p", 0.5))}
    if sampler(arm) == "dedicated":
        base = arm_spec(spec["repeat_of"])
        require({k: spec.get(k) for k in REPEAT_SAME} == {k: base.get(k) for k in REPEAT_SAME},
                "a repeat differs from its arm")
        cfg["S45SM"]["SAMPLER"] = "dedicated"
        cfg["S45SM"]["REPEAT_OF"] = spec["repeat_of"]
    if schedule(arm) == "cosine":
        require(int(cfg["TRAIN"]["WARMUP_STEPS"]) == SAME_STEPS, "the warmup the identity check relies on changed")
        if phase == "formal":
            cfg["TRAIN"]["LR_SCHEDULE"] = "cosine"
    return cfg


def expected_raw_oracle(arm, phase, base_raw):
    """S45-AT t1ddz's YAML with the S45-SM oracle identity and this arm's names / paths."""
    require(phase in ("preflight", "formal"), "unregistered phase")
    spec = oracle_spec(arm)
    require(spec["base"] == "t1ddz" and spec["assoc"] == "pfa", "oracle arms are t1ddz + PFA")
    cfg = copy.deepcopy(base_raw)
    require(cfg.pop("S45AT", None) == T1DDZ_IDENTITY, "t1ddz config identity changed")
    require(cfg["MODEL"].get("XAG_ASSOC") == "face_route" and cfg["DATA"].get("XAG_SOURCE_ROOT"), "t1ddz routes changed")
    cfg["S45SM"] = {"ARM": arm, "ROUTER": "gt", "ASSOC": "pfa", "BASE": "s45at_t1ddz"}
    out = output_dir(arm, phase)
    cfg["TRAIN"]["RUN_NAME"] = run_name(arm, phase)
    cfg["TRAIN"]["OUTPUT_DIR"] = str(out)
    cfg["EVAL"]["OUTPUT_DIR"] = str(out / "eval")
    if phase == "preflight":
        cfg["TRAIN"]["MAX_STEPS"] = 50
        cfg["TRAIN"]["VAL_CHECK_INTERVAL"] = 25
    return cfg


def expected_for(name, phase, s45_raw=None, oracle_raw=None):
    """The expected YAML content of an arm or an oracle."""
    if name in oracles():
        return expected_raw_oracle(name, phase, oracle_raw if oracle_raw is not None else oracle_base_config())
    return expected_raw(name, phase, s45_raw if s45_raw is not None else base_config())


def expected_state(cfg):
    s = cfg["S45SM"]
    return {"router": s["ROUTER_KIND"], "checkpoint": s["ROUTER_CHECKPOINT"], "assoc": s["ASSOC"],
            "onhand_tau": float(s["ONHAND_TAU"])}


def dump(cfg, root_alias=None):
    """The YAML text of a config. `root_alias` (verification in a copy of the repository only) replaces this tree's
    root by the root the files on disk were written under: the configs hold absolute paths below the repository
    (default: $S45SM_ROOT_ALIAS, see paths.py)."""
    text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    root_alias = root_alias or ROOT_ALIAS
    return text.replace(str(ROOT), str(root_alias)) if root_alias else text


def round_networks(round_):
    """Arms then oracles of a round, in registry order."""
    reg = registry()["arms"]
    return [a for a, s in reg.items() if s["round"] == round_] + [o for o, s in oracles().items() if s["round"] == round_]


def write_configs(round_, out_dir=None, root_alias=None):
    """Write the configs of one round (an arm registered with `phases` gets only those); refuses to overwrite.
    `out_dir` redirects the files (verification); the content is the same."""
    target = Path(out_dir) if out_dir else CONFIGS
    target.mkdir(parents=True, exist_ok=True)
    s45, orc = base_config(), oracle_base_config()
    paths = []
    for name in round_networks(round_):
        for phase in phases(name):
            path = target / config_path(name, phase).name
            require(not path.exists(), "config exists: " + str(path))
            path.write_text(dump(expected_for(name, phase, s45, orc), root_alias))
            paths.append(path)
    return paths


def check_configs(round_=None, root_alias=None):
    """{path: bool}: whether each registered config file on disk equals what the generator produces (bytes).
    A missing file is False; a file on disk that no generator produces is listed under "unexpected"."""
    s45, orc = base_config(), oracle_base_config()
    rounds = [round_] if round_ is not None else sorted(registry()["rounds"])
    result, expected = {}, set()
    for r in rounds:
        for name in round_networks(r):
            for phase in phases(name):
                path = config_path(name, phase)
                expected.add(path.name)
                result[str(path)] = path.is_file() and path.read_text() == dump(expected_for(name, phase, s45, orc), root_alias)
    if round_ is None:
        for path in sorted(CONFIGS.glob("*.yaml")):
            if path.name not in expected:
                result[str(path)] = False
    return result
