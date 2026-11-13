"""PMR registry 11, the ABL ablation arms (docs/EXPERIMENT_RECORDS.md#abl-prereg-20261111); tools/pmr/arms.py's interface.

Every config is D1 R's config of the same seed and phase (tools/pmr/arms.py `build_config("d1r", ...)`: B2's recipe) with
only the PMR block (design / variant / module), the router block, names / directories / identity keys changed:
  abgeo                  (a) S45's geometric ray association against the prior mesh (MODEL.XAG_ASSOC base) instead of the
                             ESMNet router's surface correspondence: no router at all (router "none", assoc "ray")
  abnr                   (b) no event-prior geometric residual (semkine/pmr_trunk11.py rnores), router r_s2d
  abnrv<k> abmv<k>       path 1: (b) / (c) R-mlp / (d) static edge-type gates (pmr_trunk2 rconst) / (e) no hand-part
  abg0v<k> abh0v<k>          communication (pmr_trunk2 rnohub), each with the RDR router v<k> (k = 2, 3)
The path-2 arms (c) / (d) / (e) are the existing d1m / d1g0 / d1h0 (registries 1 and 2), (f) is a rollout entry (rol1).
"""
from __future__ import annotations

import copy

from tools.pmr import arms as A

from .common import CONFIGS, OUT, ROOT, SEEDS, require

SECTION = "abl-prereg-20261111"
BASE = "d1r"
DESIGN_SPECS = {"x_r": ("semkine.pmr_trunk", "r"), "x_nr": ("semkine.pmr_trunk11", "rnores"),
                "x_m": ("semkine.pmr_trunk", "mlp"), "x_g0": ("semkine.pmr_trunk2", "rconst"),
                "x_h0": ("semkine.pmr_trunk2", "rnohub")}
DESIGNS = {d: (m, SECTION) for d, (m, _) in DESIGN_SPECS.items()}
ARMS = {"abgeo": ("x_r", "geo"), "abnr": ("x_nr", "s"),
        **{f"ab{d}v{k}": (f"x_{d}", f"v{k}") for d in ("nr", "m", "g0", "h0") for k in (2, 3)}}
PHASES = ("preflight", "formal")
install_trunk = A.install_trunk
VARIANTS = A.VARIANTS


def rdr_checkpoint(router):
    return ROOT / "outputs/rdr/20261110/routers" / router / f"{router}_final.ckpt"


def arms():
    return {a: (d, DESIGN_SPECS[d][1]) for a, (d, _) in ARMS.items()}


def parse(arm):
    a = arms()
    require(arm in a, "unregistered PMR arm " + arm)
    return a[arm]


def router_of(arm):
    return ARMS[arm][1]


def run_name(arm, seed, phase="formal"):
    return f"pmr_{arm}_s{seed}" + ("" if phase == "formal" else f"_{phase}")


def run_dir(arm, seed, phase="formal"):
    return OUT / "runs" / run_name(arm, seed, phase)


def config_path(arm, seed, phase="formal"):
    return CONFIGS / f"{run_name(arm, seed, phase)}.yaml"


def build_config(arm, seed, phase="formal"):
    design, variant = parse(arm)
    require(seed in SEEDS and phase in PHASES, "unregistered seed / phase")
    module = DESIGN_SPECS[design][0]
    cfg = copy.deepcopy(A.build_config(BASE, seed, phase))
    name = run_name(arm, seed, phase)
    cfg["TRAIN"].update(OUTPUT_DIR=str(run_dir(arm, seed, phase)), RUN_NAME=name)
    cfg["EVAL"]["OUTPUT_DIR"] = str(run_dir(arm, seed, phase) / "eval")
    p = cfg["PFO"]
    require(p["ROUTER_KIND"] == "pfr" and p["ROUTER"] == "r_s2d" and p["ASSOC"] == "pfa", "base arm keeps B2's router block")
    p.update(ARM=arm, SECTION=SECTION)
    router = router_of(arm)
    if router == "geo":
        p.update(ROUTER="none", ROUTER_KIND="none", ROUTER_CHECKPOINT=None, ASSOC="ray")
        cfg["MODEL"]["XAG_ASSOC"] = "base"
    elif router != "s":
        p.update(ROUTER=f"rdr_{router}", ROUTER_CHECKPOINT=str(rdr_checkpoint(router)))
    cfg["PMR"] = {"DESIGN": design, "VARIANT": variant, "MODULE": module, "BASE_ARM": BASE}
    return cfg


def write_config(arm, seed, phase="formal"):
    import yaml
    path = config_path(arm, seed, phase)
    text = yaml.safe_dump(build_config(arm, seed, phase), sort_keys=False)
    if path.exists():
        require(path.read_text() == text, "config on disk differs from the registry: " + str(path))
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


__all__ = ["ARMS", "BASE", "DESIGNS", "DESIGN_SPECS", "PHASES", "SECTION", "VARIANTS", "arms", "build_config",
           "config_path", "install_trunk", "parse", "rdr_checkpoint", "router_of", "run_dir", "run_name", "write_config"]
