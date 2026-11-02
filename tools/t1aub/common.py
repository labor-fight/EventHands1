"""T1A-UB: identity, registry, configs, hooks and network resolution (CPU only; imports nothing heavy at module level).

The study (docs/EXPERIMENT_RECORDS.md#t1aub-prereg-20261102): estimate the performance upper bound of the S45-T1a
framework (event input -> event-MANO interaction -> mesh features -> incremental state update) and approach it within
T1a's own budget (parameters, latency and FLOPs each <= T1a's).

Networks
  t1a      the baseline: S45-TR T1a (`s45tr_t1a_s3407`, step 5000 / 5500 / 6000), read from S45-TR's receipts.
  arms     fresh 6000-step trainings registered in `tools/t1aub/registry_r<k>.json` (`arms`): S45's frozen config plus
           the arm's registered changes (`T1AUB` block); trained by tools/t1aub/train.py under the strict contract.
  cells    zero-training networks (`cells`): an arm's trunk checkpoints with another router at test time. Training-time
           routes are not deployed, so a cell's cost is its trunk plus its test router.
Arm spec fields: eg (keep T1a's event graph), route ({router: <S45-SM router name> | "gt", assoc: "pfa", onhand_tau}),
gtfeat, readout ({slot_dim, hidden}), dz, sampler (legacy_none | dedicated), lr_schedule (constant | cosine), phases,
replay_of (a preflight-only replay of the baseline's preflight).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from tools.s45tr.common import now, read, require, save, sha  # noqa: E402,F401

STUDY = "T1AUB_20261102"
OUT = ROOT / "outputs/t1aub/20261102"
CONFIGS = ROOT / "configs/t1aub_20261102"
MANIFESTS = OUT / "manifests"
PHYSICAL_GPUS = (2, 3, 4, 5, 6, 7)
LATENCY_GPU = 4
SEED = 3407
STEPS = (5000, 5500, 6000)
VALIDATION = {"batches": 126, "samples": 64094}
SAME_STEPS = 500

S45_CONFIG = ROOT / "configs/c1sf_20261029/r2/c1sf_so3_s3407.yaml"
S45_CONFIG_SHA256 = "521d2d4aa2558d1eac33a77e23d972a1258fdc7733c4b35406e62215eaccd29f"
S45_RUN = ROOT / "outputs/c1sf/20261029/runs/c1sf_so3_s3407"
S45_E1_TRACE = ROOT / "outputs/c1sf/20261029/evaluations/so3/e1_step6000_repeat1_none/states.npz"
S45TR = ROOT / "outputs/s45tr/20261030"
T1A_CONFIG = ROOT / "configs/s45tr_20261030/s45tr_t1a_s3407.yaml"
T1A_RUN = S45TR / "runs/s45tr_t1a_s3407"
T1A_PREFLIGHT = S45TR / "preflight/s45tr_t1a_s3407"
S45_PREFLIGHT = S45TR / "preflight/s45tr_s45replay_s3407"
S45AT_SAMP_RUN = ROOT / "outputs/s45at/20261031/runs/s45at_s45samp_s3407"
T1A_GNN = {"HIDDEN": 96, "K": 8, "WINDOW": 32, "LAYERS": 3, "T_SCALE": 1.0}
XAG_SOURCES = ROOT / "outputs/xag/sources"
#: T1a's main row (S45-TR, step 6000) and its cost budget: each of the three costs separately <= T1a
T1A_ROW = {"local": 20.614303588867188, "global": 9.860526084899902, "ra": 14.859579411068479}
BUDGET = {"params": 494323, "macs": 732061888, "latency_ms_published": 15.588630643725516}
SAMPLERS = ("legacy_none", "dedicated")
SCHEDULES = ("constant", "cosine")
ROUTE_ASSOCS = ("pfa",)
ARM_KEYS = {"eg", "route", "gtfeat", "readout", "dz", "sampler", "lr_schedule", "phases", "replay_of", "note"}
CELL_KEYS = {"trunk", "router", "note"}


# --------------------------------------------------------------------------- registry
def registry_files():
    return sorted(PACKAGE.glob("registry_r*.json"), key=lambda p: int(p.stem.split("_r")[1]))


def registry():
    """{"rounds": {k: prereg}, "arms": {name: spec + round}, "cells": {name: spec + round}} over every round file."""
    out = {"rounds": {}, "arms": {}, "cells": {}}
    for f in registry_files():
        r = json.loads(f.read_text())
        k = int(r["round"])
        require(f.stem == f"registry_r{k}", "registry file name differs from its round: " + f.name)
        out["rounds"][k] = r.get("prereg")
        for what in ("arms", "cells"):
            for name, spec in (r.get(what) or {}).items():
                require(name not in out["arms"] and name not in out["cells"] and name != "t1a",
                        "network registered twice: " + name)
                out[what][name] = {**spec, "round": k}
    return out


def arm_spec(name):
    arms = registry()["arms"]
    require(name in arms, "unregistered T1A-UB arm " + repr(name))
    s = arms[name]
    require(set(s) - {"round"} <= ARM_KEYS, f"unknown arm fields in {name}: {sorted(set(s) - ARM_KEYS - {'round'})}")
    return s


def cell_spec(name):
    cells = registry()["cells"]
    require(name in cells, "unregistered T1A-UB cell " + repr(name))
    s = cells[name]
    require(set(s) - {"round"} <= CELL_KEYS, "unknown cell fields in " + name)
    return s


def family(name):
    """'baseline' | 'arm' | 'cell' for a network name (late-step suffixes `_5000` / `_5500` stripped)."""
    base, _ = split_step(name)
    if base == "t1a":
        return "baseline"
    reg = registry()
    if base in reg["arms"]:
        return "arm"
    if base in reg["cells"]:
        return "cell"
    raise RuntimeError("unregistered T1A-UB network " + repr(name))


def split_step(name):
    head, _, tail = name.rpartition("_")
    if tail in ("5000", "5500") and head:
        return head, int(tail)
    return name, 6000


def phases(arm):
    return tuple(arm_spec(arm).get("phases", ("preflight", "formal")))


def sampler(arm):
    s = arm_spec(arm).get("sampler", "legacy_none")
    require(s in SAMPLERS, "unregistered sampler " + repr(s))
    return s


def schedule(arm):
    s = arm_spec(arm).get("lr_schedule", "constant")
    require(s in SCHEDULES, "unregistered schedule " + repr(s))
    return s


def run_name(arm, phase="formal"):
    return f"t1aub_{arm}_s3407" + ("_preflight" if phase == "preflight" else "")


def config_path(name, phase="formal"):
    if family(name) == "cell":
        return CONFIGS / f"t1aub_cell_{name}.yaml"
    return CONFIGS / (run_name(name, phase) + ".yaml")


def output_dir(arm, phase="formal"):
    return OUT / ("preflight" if phase == "preflight" else "runs") / run_name(arm)


# --------------------------------------------------------------------------- routers (S45-SM's, by name)
def router_info(name):
    """{kind, checkpoint, params} of an S45-SM router (verified ready), or the label oracle for 'gt'."""
    if name == "gt":
        return {"kind": "gt", "checkpoint": None, "params": 0}
    from tools.s45sm.registry import router_ready, router_spec
    spec = router_spec(name)
    ckpt = router_ready(name)
    return {"kind": spec["kind"], "checkpoint": str(ckpt), "params": int(spec.get("params") or 0)}


# --------------------------------------------------------------------------- configs
def s45_raw():
    import yaml
    from tools.frozen import accepts
    require(accepts(S45_CONFIG, S45_CONFIG_SHA256), "S45 config changed")
    return yaml.safe_load(S45_CONFIG.read_text())


def _route_block(route):
    r = router_info(route["router"])
    if r["kind"] == "gt":
        require(route["assoc"] == "pfa", "label routes enter through PFA")
        return {"ROUTER": "gt", "ASSOC": "pfa"}
    require(route["assoc"] in ROUTE_ASSOCS, "unregistered association " + repr(route["assoc"]))
    return {"ROUTER": route["router"], "ROUTER_KIND": r["kind"], "ROUTER_CHECKPOINT": r["checkpoint"],
            "ASSOC": route["assoc"], "ONHAND_TAU": float(route.get("onhand_tau", 0.5))}


def expected_raw(arm, phase, base=None):
    """An arm's YAML content: S45's config + the arm's registered changes + identity / paths."""
    require(phase in phases(arm), f"phase {phase} not registered for {arm}")
    spec = arm_spec(arm)
    cfg = copy.deepcopy(base if base is not None else s45_raw())
    require(cfg.pop("C1SF", None) == {"ARM": "so3"}, "S45 config identity changed")
    require("XAG_ASSOC" not in cfg["MODEL"] and "XAG_SOURCE_ROOT" not in cfg["DATA"], "S45 config changed")
    block = {"ARM": arm, "ROUND": int(spec["round"]), "EG": bool(spec.get("eg", True))}
    if block["EG"]:
        block["T1A_GNN"] = dict(T1A_GNN)
    if spec.get("readout"):
        ro = spec["readout"]
        block["READOUT"] = {"SLOT_DIM": int(ro["slot_dim"]), "HIDDEN": int(ro["hidden"])}
    route = spec.get("route")
    if route:
        block["ROUTE"] = _route_block(route)
    if spec.get("gtfeat"):
        require(not route, "gtfeat keeps S45's ray association")
        block["GTFEAT"] = True
    if route or spec.get("gtfeat"):
        cfg["MODEL"]["XAG_ASSOC"] = "face_route"
        cfg["MODEL"]["XAG_GNN"] = "legacy"
    if spec.get("gtfeat") or (route and route["router"] == "gt"):
        cfg["DATA"]["XAG_SOURCE_ROOT"] = str(XAG_SOURCES)
    if spec.get("dz"):
        cfg["AUG"]["DOMRAND"]["SCALE_MODE"] = "depth"
    if sampler(arm) != "legacy_none":
        block["SAMPLER"] = sampler(arm)
    if schedule(arm) == "cosine":
        require(int(cfg["TRAIN"]["WARMUP_STEPS"]) == SAME_STEPS, "the warmup the identity check relies on changed")
        if phase == "formal":
            cfg["TRAIN"]["LR_SCHEDULE"] = "cosine"
    if spec.get("replay_of"):
        require(spec["replay_of"] == "t1a" and phases(arm) == ("preflight",), "replays are preflight-only T1a replays")
        block["REPLAY_OF"] = "t1a"
    cfg["T1AUB"] = block
    out = output_dir(arm, phase)
    cfg["TRAIN"]["RUN_NAME"] = run_name(arm, phase)
    cfg["TRAIN"]["OUTPUT_DIR"] = str(out)
    cfg["EVAL"]["OUTPUT_DIR"] = str(out / "eval")
    if phase == "preflight":
        cfg["TRAIN"]["MAX_STEPS"] = 50
        cfg["TRAIN"]["VAL_CHECK_INTERVAL"] = 25
    return cfg


def expected_cell(cell, base=None):
    """A cell's evaluation config: its trunk arm's formal config with the test router in place of the training routes."""
    spec = cell_spec(cell)
    trunk = arm_spec(spec["trunk"])
    require("formal" in phases(spec["trunk"]), "a cell's trunk needs a formal training")
    require(trunk.get("route"), "cells swap the router of a routed trunk")
    cfg = expected_raw(spec["trunk"], "formal", base)
    r = router_info(spec["router"])
    require(r["kind"] != "gt", "a cell's test router is deployable")
    cfg["DATA"].pop("XAG_SOURCE_ROOT", None)
    cfg["T1AUB"]["ROUTE"] = _route_block({"router": spec["router"], "assoc": trunk["route"]["assoc"],
                                          "onhand_tau": trunk["route"].get("onhand_tau", 0.5)})
    cfg["T1AUB"]["CELL"] = cell
    cfg["T1AUB"]["TRUNK"] = spec["trunk"]
    cfg["TRAIN"].pop("LR_SCHEDULE", None)
    return cfg


def dump(cfg):
    import yaml
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)


def round_files(k):
    """[(path, content)] of every config a round registers (arms: their phases; cells: one evaluation config)."""
    reg = registry()
    base = s45_raw()
    files = []
    for arm, spec in reg["arms"].items():
        if spec["round"] == k:
            for phase in phases(arm):
                files.append((config_path(arm, phase), dump(expected_raw(arm, phase, base))))
    for cell, spec in reg["cells"].items():
        if spec["round"] == k:
            files.append((config_path(cell), dump(expected_cell(cell, base))))
    return files


def write_configs(k):
    CONFIGS.mkdir(parents=True, exist_ok=True)
    written = []
    for path, text in round_files(k):
        require(not path.exists(), "config exists: " + str(path))
        path.write_text(text)
        written.append(path)
    return written


def check_configs(k):
    return {str(p): p.is_file() and p.read_text() == text for p, text in round_files(k)}


# --------------------------------------------------------------------------- hooks
def install_for_config(cfg, *, evaluation, fast=False, reparam=False):
    """Install every opt-in hook a T1A-UB config needs, before the model is built; returns the installed identity.

    Order: the T1a event-graph class (EG arms), this study's subclass, then the routes (an S45-SM router, fast or not,
    or the label PFA oracle). `evaluation`: router networks attach no ground-truth XAG source to their packets.
    `reparam`: T1a's EdgeConv in its deployment form (no-gradient forwards only)."""
    from semkine import t1aub as M        # imports semkine.s45sm first (native-hook check)
    from semkine import s45tr_t1a
    b = cfg["T1AUB"]
    state = {"arm": b["ARM"], "eg": bool(b["EG"])}
    if b["EG"]:
        require(b["T1A_GNN"] == T1A_GNN, "T1a event-graph settings changed")
        s45tr_t1a.install(b["T1A_GNN"])
    ro = b.get("READOUT")
    M.install({"eg": bool(b["EG"]), "gtfeat": bool(b.get("GTFEAT", False)), "mano_npz": cfg["MANO"]["NPZ"],
               "readout": {"slot_dim": int(ro["SLOT_DIM"]), "hidden": int(ro["HIDDEN"])} if ro else None})
    state["model"] = M.installed()
    route = b.get("ROUTE")
    if route and route["ROUTER"] == "gt":
        from semkine import s45sm_oracle as O
        state["route"] = O.install()
    elif route:
        from tools.s45sm.routing import NEEDS, register_kinds
        register_kinds(NEEDS[route["ROUTER_KIND"]])
        if fast:
            from semkine import s45sm_fast as R
        else:
            from semkine import s45sm as R
        r = R.install(route["ROUTER_KIND"], route["ROUTER_CHECKPOINT"], route["ASSOC"], float(route["ONHAND_TAU"]))
        state["route"] = {k: r[k] for k in ("router", "checkpoint", "assoc", "onhand_tau")}
        state["route"]["fast"] = bool(fast)
        if evaluation:
            from semkine import eval_track as ET
            ET.attach_xag_sources = lambda batch, *args, **kwargs: batch
    if reparam:
        require(b["EG"], "reparam applies to the event graph")
        M.set_reparam(True)
        state["reparam"] = True
    return state


def router_net():
    """The installed S45-SM router network (for its parameter / MAC count), or None."""
    from semkine import s45sm as SM
    return SM._STATE["net"]


# --------------------------------------------------------------------------- frozen files
def manifests():
    return sorted(MANIFESTS.glob("*.json"))


def frozen_files():
    files = {}
    for m in manifests():
        files.update(read(m)["file_sha256"])
    return files


def verify_protocol():
    """Every file any frozen manifest of this study lists still has its frozen SHA; every registered round is frozen."""
    from tools.frozen import accepts
    files = frozen_files()
    for name, digest in files.items():
        require(accepts(name, digest), "Frozen T1A-UB file changed: " + name)
    frozen_rounds = {read(m).get("round") for m in manifests()}
    for k in registry()["rounds"]:
        require(k in frozen_rounds, f"round {k} is registered but not frozen")
    return files


def config_frozen(config):
    from tools.frozen import accepts
    rel = str(Path(config).resolve().relative_to(ROOT))
    files = frozen_files()
    require(rel in files and accepts(config, files[rel]), "Config not frozen: " + rel)


def section(tag):
    """(sha256, length) of a ledger section between its BEGIN / END markers in docs/EXPERIMENT_RECORDS.md."""
    text = (ROOT / "docs/EXPERIMENT_RECORDS.md").read_text()
    begin, end = f"<!-- EXPERIMENT_RECORD:{tag}:BEGIN -->", f"<!-- EXPERIMENT_RECORD:{tag}:END -->"
    require(text.count(begin) == 1 and text.count(end) == 1, "pre-registration markers missing: " + tag)
    body = text[text.index(begin):text.index(end) + len(end)]
    return hashlib.sha256(body.encode()).hexdigest(), len(body)


#: the files that define a training and its acceptance; evaluation / reading tools are frozen by each round's manifest
TRAIN_FILES = ("semkine/t1aub.py", "tools/t1aub/common.py", "tools/t1aub/train.py", "tools/t1aub/accept.py")


def code_identity():
    """SHA-256 of the (path, SHA-256) list of TRAIN_FILES: a preflight accepted under other training code does not
    authorise a formal run."""
    listing = [[f, sha(ROOT / f)] for f in TRAIN_FILES]
    return hashlib.sha256(json.dumps(listing, sort_keys=True).encode()).hexdigest()


def require_one_gpu():
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, PHYSICAL_GPUS)), "one GPU among 2-7")


# --------------------------------------------------------------------------- resolution
def t1a_checkpoint(step):
    receipt = read(S45TR / "receipts/formal_t1a.json")
    require(receipt["status"] == "PASS", "T1a has no verified formal training")
    ckpt = T1A_RUN / f"s45tr_t1a_s3407-step={step}.ckpt"
    require(sha(ckpt) == receipt["checkpoint_sha256_by_step"][str(step)], "T1a checkpoint changed since its receipt")
    return ckpt


def arm_checkpoint(arm, step):
    receipt = read(OUT / "receipts" / f"formal_{arm}.json")
    require(receipt["status"] == "PASS", "arm has no verified formal training: " + arm)
    ckpt = output_dir(arm) / f"{run_name(arm)}-step={step}.ckpt"
    require(sha(ckpt) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since its receipt")
    return ckpt


def resolve(name):
    """(checkpoint, config path, family, step) of a network name; a cell resolves to its trunk's checkpoint."""
    base, step = split_step(name)
    fam = family(name)
    if fam == "baseline":
        return t1a_checkpoint(step), T1A_CONFIG, fam, step
    if fam == "arm":
        config = config_path(base)
        config_frozen(config)
        return arm_checkpoint(base, step), config, fam, step
    config = config_path(base)
    config_frozen(config)
    return arm_checkpoint(cell_spec(base)["trunk"], step), config, fam, step


def load_config_for(name):
    """The loaded config of a network; the baseline gets its S45-TR T1a config with an equivalent T1AUB block."""
    from config import load_config
    _, config, fam, _ = resolve(name)
    cfg = load_config(str(config))
    if fam == "baseline":
        require(cfg["S45TR"] == {"ARM": "t1a", "HOOK": "t1a", "T1A_GNN": T1A_GNN}, "T1a config identity changed")
        cfg["T1AUB"] = {"ARM": "t1a", "ROUND": 0, "EG": True, "T1A_GNN": dict(T1A_GNN)}
    return cfg


def test_router(name):
    """The S45-SM router a network routes with at test time (a cell's, or a routed arm's own), else None."""
    base, _ = split_step(name)
    fam = family(name)
    if fam == "cell":
        return cell_spec(base)["router"]
    if fam == "arm" and arm_spec(base).get("route") and arm_spec(base)["route"]["router"] != "gt":
        return arm_spec(base)["route"]["router"]
    return None


__all__ = [n for n in dir() if not n.startswith("_")]
