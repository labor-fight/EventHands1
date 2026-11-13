#!/usr/bin/env python3
"""Route-first EvHandTrack on the real data (docs/EXPERIMENT_RECORDS.md#real-rfs-20261112; user 2026-11-12: "EvHandTrack
先路由，残差槽只收手边事件，这部分给我实验").

Two arms added in-process to the real registry (tools/real/evht.py, frozen, unchanged on disk):
  rfs        = d1r (EvHandTrack's from-scratch arm, its real-data config) + RFS, 6000 steps
  rfs_rol5   = d1r_rol5 (K = 5 rollout fine-tune, 1500 steps) + RFS, warm start = rfs's step 6000
RFS (semkine/rfs.py): the router evaluates every event of a packet; nodes are sampled (the same rule, budget 2048)
among the on-hand events (p >= TAU) and the off-hand events within NEAR_PX canvas px of an on-hand event of the packet
(square window; 16 px of the 346 x 260 canvas ~ 30 px of the 640 x 480 grid). Config block RFS {TAU, NEAR_PX}. The
router (the study's r_s2d), trunk, PFA, residual slot, losses, recipe and data are EvHandTrack's.

    python tools/real/rfs.py configs                                   # write the configs (formal + smoke)
    CUDA_VISIBLE_DEVICES=<k> python tools/real/rfs.py train --config configs/real_20261110/pmr/pmr_rfs[_rol5]_s<seed>[_smoke].yaml
    python tools/real/rfs.py accept --arm rfs|rfs_rol5 --seed S [--phase smoke]
    CUDA_VISIBLE_DEVICES=<k> python tools/real/rfs.py read --arm rfs_rol5 --seed S        # streaming E1, test + dev

Route-first everywhere (docs/EXPERIMENT_RECORDS.md#rfsu-20261113; user 2026-11-13: "全文统一成先路由"): the paper's
real-data ablations with RFS -- rfs_rol1 (Single-step only, warm start = rfs), rfsm(_rol5) (Global MLP update), rfsg0(_rol5)
(State-independent gating), rfsgeo(_rol5) (w/o surface association: S45's ray association; the namespace's r_s2d is loaded
for the on-hand decision of the sampling only, config RFS.ROUTER_CHECKPOINT / ROUTER_SHA256) -- and, in the EvRealHands
namespace (ERH_NAMESPACE=1, or tools/erh/run.py tools.real.rfs ...), rfs / rfs_rol5. The arms of #real-rfs-20261112 keep
their configs byte for byte.

r_s2dx2 everywhere (user 2026-11-13: "论文用 r_s2dx2，真实数据，仿真数据，和 EvRealHands ... 都换成 r_s2dx2"): the same
ablations with the namespace's r_s2dx2 -- rfsr_rol1 (warm start = rfsr), rfsrm(_rol5), rfsrg0(_rol5), rfsrgeo(_rol5) (the
r_s2dx2 gates the sampling only) -- and, in the EvRealHands namespace, rfsr / rfsr_rol5 with EvRealHands' own r_s2dx2
(tools/real/router_variant.py run there; x2_router()). `configs --arms a,b` writes only those arms.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ARMS_RFS = {"rfs": "d1r", "rfs_rol5": "d1r_rol5", "rfsr": "d1r", "rfsr_rol5": "d1r_rol5",
            # #rfsu-20261113: the paper's ablations
            "rfs_rol1": "d1r_rol1", "rfsm": "d1m", "rfsm_rol5": "d1m_rol5", "rfsg0": "d1g0", "rfsg0_rol5": "d1g0_rol5",
            "rfsgeo": "abgeo", "rfsgeo_rol5": "abgeo_rol5",
            # user 2026-11-13 ("论文用 r_s2dx2，真实数据，仿真数据，和 EvRealHands ... 都换成 r_s2dx2"): the ablations with
            # the namespace's r_s2dx2
            "rfsr_rol1": "d1r_rol1", "rfsrm": "d1m", "rfsrm_rol5": "d1m_rol5", "rfsrg0": "d1g0",
            "rfsrg0_rol5": "d1g0_rol5", "rfsrgeo": "abgeo", "rfsrgeo_rol5": "abgeo_rol5"}
BASES_RFS = ("rfs", "rfsr", "rfsm", "rfsg0", "rfsgeo", "rfsrm", "rfsrg0", "rfsrgeo")
ROLS_RFS = {"rfs_rol5": "rfs", "rfsr_rol5": "rfsr", "rfs_rol1": "rfs", "rfsm_rol5": "rfsm", "rfsg0_rol5": "rfsg0",
            "rfsgeo_rol5": "rfsgeo", "rfsr_rol1": "rfsr", "rfsrm_rol5": "rfsrm", "rfsrg0_rol5": "rfsrg0",
            "rfsrgeo_rol5": "rfsrgeo"}
#: the arms of #real-rfs-20261112 (their configs keep SECTION); EvRealHands trains the full model only
FIRST = ("rfs", "rfs_rol5", "rfsr", "rfsr_rol5")
ERH_ARMS = ("rfs", "rfs_rol5", "rfsr", "rfsr_rol5")
SECTION_U = "rfsu-20261113"
#: #rfsu-20261113 arms: memory cap of the router pass over a training batch (semkine/rfs.py `_spans`; a simulated run
#: ran out of memory on a 64-packet chunk); the arms of #real-rfs-20261112 keep the 64-packet chunks
MAX_ROWS = 4_000_000
#: rfsr / rfsr_rol5 = rfs / rfs_rol5 with the router chosen by the pre-registered rule (#real-rfs-20261112, 16:20 UTC),
#: named in CHOSEN (written once by the decision, never edited); the rfsr* ablations (#rfsu-20261113) use the same
#: router. In the EvRealHands namespace: EvRealHands' own r_s2dx2 (tools/real/router_variant.py there: r_s2d's
#: EvRealHands recipe with only the design changed) -- x2_router()
ROUTER_ARMS = ("rfsr", "rfsr_rol5", "rfsr_rol1", "rfsrm", "rfsrm_rol5", "rfsrg0", "rfsrg0_rol5", "rfsrgeo", "rfsrgeo_rol5")
CHOSEN = ROOT / "outputs/real/20261110/routers/chosen.json"
ERH_X2 = "r_s2dx2"
TAU = 0.5
NEAR_PX = 16
SECTION = "real-rfs-20261112"


def extend():
    """The real registry with the two RFS arms (configs = their EvHandTrack arm's + the RFS block)."""
    from tools.real import evht as X
    if "rfs" in X.ARMS:
        return X
    X.BASES = X.BASES + BASES_RFS
    X.ROLS = {**X.ROLS, **ROLS_RFS}
    X.ARMS = X.BASES + tuple(X.ROLS)
    native_sim, native_build = X.sim_config, X.build_config

    def sim_config(arm, seed):
        return native_sim(ARMS_RFS.get(arm, arm), seed)

    def build_config(arm, seed, phase="formal"):
        cfg = native_build(arm, seed, phase)
        if arm in ARMS_RFS:
            cfg["PFO"]["ARM"] = arm
            cfg["RFS"] = {"TAU": TAU, "NEAR_PX": NEAR_PX, "SECTION": section(arm), "BASE_ARM": ARMS_RFS[arm]}
            if section(arm) == SECTION_U:
                cfg["RFS"]["MAX_ROWS"] = MAX_ROWS
            if cfg["PFO"]["ROUTER_KIND"] == "none":      # w/o surface association: the router gates the sampling only
                r = x2_router()[1] if arm in ROUTER_ARMS else X.router(phase)
                cfg["RFS"].update(ROUTER_CHECKPOINT=str(r), ROUTER_SHA256=_sha(r))
        if arm in ROUTER_ARMS:
            variant, path = x2_router()
            if cfg["PFO"]["ROUTER_KIND"] != "none":
                cfg["PFO"]["ROUTER_CHECKPOINT"] = str(path)
                cfg["REAL"]["ROUTER"] = str(path)
            cfg["RFS"]["ROUTER"] = variant
        return cfg
    X.sim_config, X.build_config = sim_config, build_config
    return X


def _erh():
    from tools.real import ns
    return bool(getattr(ns, "ERH", False))


def section(arm):
    return SECTION if (arm in FIRST and not _erh()) else SECTION_U


def _sha(path):
    from tools.cmp.common import sha
    return sha(path)


def sampling_router(cfg):
    """An arm without a router of its own (rfsgeo*: S45's ray association): load the namespace's r_s2d into S45-SM's
    state for RFS's on-hand decision only -- no association / source hook is installed, the model never reads it."""
    r = cfg.get("RFS", {}).get("ROUTER_CHECKPOINT")
    if not r:
        return None
    from tools.real import evht as X
    X.require(_sha(r) == cfg["RFS"]["ROUTER_SHA256"], "the sampling router changed since the config: " + r)
    from semkine import pfo_router as PR
    from semkine import rdr_hooks as RH
    from semkine import s45sm as SM
    X.require(SM._STATE["net"] is None, "a router is installed already")
    RH.enable()
    PR.register()
    SM._STATE.update(router=PR.KIND, net=SM._load_router(PR.KIND, r), checkpoint=str(r), onhand_tau=float(TAU), on={})
    print("RFS sampling router (on-hand decision only):", r, flush=True)
    return r


def x2_router():
    """(variant, checkpoint) of the namespace's r_s2dx2: the real study's chosen router, or EvRealHands' own."""
    if not _erh():
        return chosen_router()
    import json
    from tools.real import ns
    path = ns.OUT / "routers" / ERH_X2 / "r_s2d_final.ckpt"
    if not (path.exists() and json.loads(path.with_suffix(".json").read_text()).get("status") == "COMPLETE"):
        raise SystemExit("EvRealHands r_s2dx2 is not trained yet: " + str(path))
    return ERH_X2, path


def chosen_router():
    import json
    obj = json.loads(CHOSEN.read_text())
    path = Path(obj["checkpoint"])
    if not path.exists():
        raise SystemExit("chosen router checkpoint missing: " + str(path))
    return obj["variant"], path


def configs(arms=None):
    X = extend()
    from tools.real import ns
    default = [a for a in ARMS_RFS if (a in ERH_ARMS if _erh() else (a not in ROUTER_ARMS or CHOSEN.exists()))]
    if _erh():
        default = [a for a in default if a not in ROUTER_ARMS or (ns.OUT / "routers" / ERH_X2 / "r_s2d_final.ckpt").exists()]
    for arm in (arms or default):
        for seed in ns.SEEDS:
            for phase in X.PHASES:
                print(X.write_config(arm, seed, phase))


def train(config):
    import yaml
    X = extend()
    from tools.real import ns
    path = Path(config).resolve()
    cfg = yaml.safe_load(path.read_text())
    arm = cfg["PFO"]["ARM"]
    X.require(arm in ARMS_RFS, "not an RFS arm: " + arm)
    X.parse(arm)
    X.enter()
    from semkine import rfs as RF
    print("RFS:", RF.install(cfg["RFS"]["TAU"], cfg["RFS"]["NEAR_PX"], packet=True, max_rows=cfg["RFS"].get("MAX_ROWS")),
          flush=True)
    sampling_router(cfg)
    if X.kind(arm) == "rol":
        from tools.pmr import train_rol7 as W
        T = W.W.T
    else:
        from tools.pmr import train11 as W
        T = W.T
    T.build_config, T.config_path, T.parse = X.build_config, X.config_path, X.parse
    out = Path(cfg["TRAIN"]["OUTPUT_DIR"])
    if out.exists():
        dst = ns.OUT / "pack" / "preempted" / f"{out.name}__{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out), str(dst))
        print(f"fresh-only: archived the earlier attempt {out.name} to {dst}", flush=True)
    sys.argv = [sys.argv[0], "--config", str(path)]
    T.main()
    print("RFS stats:", RF.stats(), flush=True)


def accept(arm, seed, phase):
    X = extend()
    if arm in ROUTER_ARMS:                     # the acceptance's router check reads X.router(phase): the namespace's r_s2dx2
        _, path = x2_router()
        X.router = lambda phase="formal": path
    X.accept(arm, seed, phase)


def read(arm, seed, task, phase, limit=None):
    X = extend()
    from semkine import rfs as RF
    RF.install(TAU, NEAR_PX, packet=False)
    RF.install_stream()
    sampling_router(X.build_config(arm, seed, phase))
    X.read(arm, seed, task, phase, limit)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verb", choices=("configs", "train", "accept", "read"))
    ap.add_argument("--config")
    ap.add_argument("--arm", choices=tuple(ARMS_RFS))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--task", default="e1")
    ap.add_argument("--phase", default="formal")
    ap.add_argument("--limit", type=int, help="engineering smoke: this many endpoints per sequence")
    ap.add_argument("--arms", help="configs: comma-separated arms (default: every arm whose router exists)")
    a = ap.parse_args()
    if a.verb == "configs":
        configs(a.arms.split(",") if a.arms else None)
    elif a.verb == "train":
        train(a.config)
    elif a.verb == "accept":
        accept(a.arm, a.seed, a.phase)
    else:
        read(a.arm, a.seed, a.task, a.phase, a.limit)


if __name__ == "__main__":
    main()
