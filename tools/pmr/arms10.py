"""PMR registry 10 (docs/EXPERIMENT_RECORDS.md#rdr-trunk-prereg-20261110): D1 R trained and read with an RDR router
(semkine/rdr_router.py, outputs/rdr/20261110/routers/v<k>) instead of r_s2d; tools/pmr/arms.py's interface.

  d1rv1 / d1rv2 / d1rv3   D1 R (design d1, variant r) + RDR router v1 / v2 / v3, trained from scratch with it
  d1rw1 / d1rw2 / d1rw3   "warm" base arms: D1 R's own accepted run and checkpoint (r_s2d-trained) with the router block
                          pointing at RDR v1 / v2 / v3. Never trained themselves; they exist so that the rollout registry
                          (tools/pmr/arms_rol2.py) builds d1rw<k>_rol5 = rol5 fine-tune of D1 R's step-6000 checkpoint
                          with the RDR router in the loop.
Every config is D1 R's config of the same seed and phase with only the router name and checkpoint of the PFO block and the
names / directories / identity keys changed. The hook kind stays "pfr": semkine/rdr_hooks.py makes that kind load an RDR
checkpoint by its own type.
"""
from __future__ import annotations

import copy

from tools.pmr import arms as A

from .common import CONFIGS, OUT, ROOT, SEEDS, require

SECTION = "rdr-trunk-prereg-20261110"
ROUTERS = ("v1", "v2", "v3")
MODULE = A.DESIGNS["d1"][0]
DESIGNS = {"d1": (MODULE, SECTION)}
PHASES = ("preflight", "formal")
BASE = "d1r"
install_trunk = A.install_trunk
VARIANTS = A.VARIANTS


def router_checkpoint(router):
    return ROOT / "outputs/rdr/20261110/routers" / router / f"{router}_final.ckpt"


def arms():
    return {**{f"d1r{r}": ("d1", "r") for r in ROUTERS}, **{f"d1rw{r[1:]}": ("d1", "r") for r in ROUTERS}}


def warm(arm):
    return arm.startswith("d1rw")


def router_of(arm):
    return "v" + arm[len("d1rw"):] if warm(arm) else arm[len("d1r"):]


def parse(arm):
    a = arms()
    require(arm in a, "unregistered PMR arm " + arm)
    return a[arm]


def run_name(arm, seed, phase="formal"):
    if warm(arm):
        return A.run_name(BASE, seed, phase)
    return f"pmr_{arm}_s{seed}" + ("" if phase == "formal" else f"_{phase}")


def run_dir(arm, seed, phase="formal"):
    if warm(arm):
        return A.run_dir(BASE, seed, phase)
    return OUT / "runs" / run_name(arm, seed, phase)


def config_path(arm, seed, phase="formal"):
    require(not warm(arm), "a warm base arm is never trained")
    return CONFIGS / f"{run_name(arm, seed, phase)}.yaml"


def build_config(arm, seed, phase="formal"):
    design, variant = parse(arm)
    require(seed in SEEDS and phase in PHASES, "unregistered seed / phase")
    router = router_of(arm)
    cfg = copy.deepcopy(A.build_config(BASE, seed, phase))
    if not warm(arm):
        name = run_name(arm, seed, phase)
        cfg["TRAIN"].update(OUTPUT_DIR=str(run_dir(arm, seed, phase)), RUN_NAME=name)
        cfg["EVAL"]["OUTPUT_DIR"] = str(run_dir(arm, seed, phase) / "eval")
    require(cfg["PFO"]["ROUTER_KIND"] == "pfr" and cfg["PFO"]["ROUTER"] == "r_s2d", "base arm keeps B2's router block")
    cfg["PFO"].update(ARM=arm, SECTION=SECTION, ROUTER=f"rdr_{router}", ROUTER_CHECKPOINT=str(router_checkpoint(router)))
    cfg["PMR"] = {"DESIGN": design, "VARIANT": variant, "MODULE": MODULE, "BASE_ARM": BASE}
    return cfg


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


__all__ = ["BASE", "DESIGNS", "MODULE", "PHASES", "ROUTERS", "SECTION", "VARIANTS", "arms", "build_config", "config_path",
           "install_trunk", "parse", "router_checkpoint", "run_dir", "run_name", "write_config"]
