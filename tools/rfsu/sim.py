#!/usr/bin/env python3
"""Route-first EvHandTrack on the simulated data (docs/EXPERIMENT_RECORDS.md#rfsu-20261113; user 2026-11-13:
"全文统一成先路由").

The paper's simulated EvHandTrack and its figure ablations retrained with route-first sampling (semkine/rfs.py, as
tools/real/rfs.py on the real data): the router evaluates every event of a packet, nodes are drawn (the same rule, budget
2048) among the on-hand events and the off-hand events within NEAR_PX canvas px of an on-hand event; the residual slot
keeps only that vicinity. NEAR_PX = 11 on the 240 x 180 simulated canvas = the real data's 16 px of 346 scaled by
240 / 346 (both ~30 px of the 640 x 480 grid, the near30 rule of #real-amend5-run-20261111 14:35 UTC).

Arms (each = the frozen PMR registry's formal config of its simulated arm, same seed; only names / output directories,
PFO.ARM / PHASE, the rollout warm start and the RFS block change):
  rfs           d1r          EvHandTrack's trunk from scratch, 6000 steps (B2's recipe, 2 x 512 DDP)
  rfs_rol5      d1r_rol5     Full model (K = 5 rollout fine-tune, 1500 steps, warm start = rfs step 6000)
  rfs_rol1      d1r_rol1     Single-step only (K = 1, warm start = rfs step 6000)
  rfsm(_rol5)   d1m(_rol5)   Global MLP update
  rfsg0(_rol5)  d1g0(_rol5)  State-independent gating
  rfsgeo(_rol5) abgeo(_rol5) w/o surface association (S45's ray association; the simulated r_s2d is loaded for the
                             on-hand decision of the sampling only: RFS.ROUTER_CHECKPOINT / ROUTER_SHA256)
  rfsr, rfsr_rol5, rfsr_rol1, rfsrm(_rol5), rfsrg0(_rol5), rfsrgeo(_rol5): the same arms with the simulated r_s2dx2
                             (user 2026-11-13: "论文用 r_s2dx2 ... 仿真数据 ... 都换成 r_s2dx2"; tools/rfsu/router_sim.py,
                             r_s2d's simulated recipe with only the design changed): PFO.ROUTER_CHECKPOINT (or, rfsrgeo*,
                             RFS.ROUTER_CHECKPOINT) points to it and RFS.ROUTER = r_s2dx2; nothing else changes
Training goes through the frozen tools/pmr/train11.py (bases) / train_rol7.py (rollouts) with their registry rebound to
this one, RFS installed in every DDP rank (Lightning relaunches rank 1 as `sim.py --config <config>`). Readings: the
streaming form (semkine/tpd_str.py, EvHandTrack's main-row form) through PFO's frozen E1 engine on zgz, with RFS's
streaming sampler. Outputs under outputs/rfsu/20261113/sim, configs under configs/rfsu_20261113/sim.

    python tools/rfsu/sim.py configs
    CUDA_VISIBLE_DEVICES=a,b NCCL_P2P_DISABLE=1 python tools/rfsu/sim.py train --config configs/rfsu_20261113/sim/pmr_<arm>_s<seed>[_smoke].yaml
    python tools/rfsu/sim.py accept --arm A --seed S [--phase smoke]
    CUDA_VISIBLE_DEVICES=k python tools/rfsu/sim.py read --arm A --seed S [--phase smoke --limit N]
"""
from __future__ import annotations

import argparse
import copy
import json
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

OUT = ROOT / "outputs/rfsu/20261113/sim"
CONFIGS = ROOT / "configs/rfsu_20261113/sim"
SECTION = "rfsu-20261113"
SEEDS = (3407, 3408, 3409)
PHASES = ("formal", "smoke")
TAU = 0.5
NEAR_PX = 11
#: memory cap of the router pass over a training batch (semkine/rfs.py `_spans`): sim:rfs:3407's first attempt ran out of
#: memory on a 64-packet chunk (6.6 GiB in the router head; 2026-11-13 00:07 UTC)
MAX_ROWS = 4_000_000
BASES = {"rfs": "d1r", "rfsm": "d1m", "rfsg0": "d1g0", "rfsgeo": "abgeo",
         # user 2026-11-13 ("论文用 r_s2dx2 ... 仿真数据 ... 都换成 r_s2dx2"): the same arms with the simulated r_s2dx2
         "rfsr": "d1r", "rfsrm": "d1m", "rfsrg0": "d1g0", "rfsrgeo": "abgeo"}
ROLS = {"rfs_rol5": ("rfs", "d1r_rol5"), "rfs_rol1": ("rfs", "d1r_rol1"), "rfsm_rol5": ("rfsm", "d1m_rol5"),
        "rfsg0_rol5": ("rfsg0", "d1g0_rol5"), "rfsgeo_rol5": ("rfsgeo", "abgeo_rol5"),
        "rfsr_rol5": ("rfsr", "d1r_rol5"), "rfsr_rol1": ("rfsr", "d1r_rol1"), "rfsrm_rol5": ("rfsrm", "d1m_rol5"),
        "rfsrg0_rol5": ("rfsrg0", "d1g0_rol5"), "rfsrgeo_rol5": ("rfsrgeo", "abgeo_rol5")}
#: the arms with the simulated r_s2dx2 (tools/rfsu/router_sim.py: r_s2d's simulated recipe, only the design changed)
X2_ARMS = ("rfsr", "rfsrm", "rfsrg0", "rfsrgeo", "rfsr_rol5", "rfsr_rol1", "rfsrm_rol5", "rfsrg0_rol5", "rfsrgeo_rol5")
ARMS = tuple(BASES) + tuple(ROLS)
FINAL = {"base": 6000, "rol": 1500}
SMOKE = {"base": {"MAX_STEPS": 20, "WARMUP_STEPS": 5, "SAVE_EVERY_N_STEPS": 20, "VAL_CHECK_INTERVAL": 20},
         "rol": {"MAX_STEPS": 10, "WARMUP_STEPS": 5, "SAVE_EVERY_N_STEPS": 10, "VAL_CHECK_INTERVAL": 10}}
SIM_ROUTER = ROOT / "outputs/pfo/20261103/routers/r_s2d/r_s2d_final.ckpt"
SIM_X2 = OUT / "routers" / "r_s2dx2" / "r_s2dx2_final.ckpt"
FORM = "str"
CODE = ("tools/rfsu/sim.py", "semkine/rfs.py", "semkine/tpd_str.py")


def require(cond, msg):
    if not cond:
        raise SystemExit("rfsu.sim: " + msg)


def kind(arm):
    return "rol" if arm in ROLS else "base"


def sim_arm(arm):
    return ROLS[arm][1] if arm in ROLS else BASES[arm]


def parse(arm):
    require(arm in ARMS, "not a route-first simulated arm: " + str(arm))
    return (ROLS[arm][0], arm.rpartition("_")[2]) if arm in ROLS else (arm, None)


def run_name(arm, seed, phase="formal"):
    return f"pmr_{arm}_s{seed}" + ("" if phase == "formal" else f"_{phase}")


def run_dir(arm, seed, phase="formal"):
    return OUT / "runs" / run_name(arm, seed, phase)


def config_path(arm, seed, phase="formal"):
    return CONFIGS / f"{run_name(arm, seed, phase)}.yaml"


def checkpoint(arm, seed, phase="formal", step=None):
    step = step if step is not None else (FINAL[kind(arm)] if phase == "formal" else SMOKE[kind(arm)]["MAX_STEPS"])
    return run_dir(arm, seed, phase) / f"{run_name(arm, seed, phase)}-step={step}.ckpt"


def _sha(path):
    from tools.cmp.common import sha
    return sha(path)


def registry():
    from tools.pmr import arms_any6  # noqa: F401  (registry 11 and rol1)
    from tools.pmr import arms_any2 as U
    from tools.pmr import arms_rol2 as R2
    return U, R2


def build_config(arm, seed, phase="formal"):
    parse(arm)
    require(seed in SEEDS and phase in PHASES, "unregistered seed / phase")
    U, R2 = registry()
    k = kind(arm)
    src = sim_arm(arm)
    cfg = copy.deepcopy(R2.build_config(src, seed, "formal") if k == "rol" else U.build_config(src, seed, "formal"))
    rn = run_name(arm, seed, phase)
    t = cfg["TRAIN"]
    require(t["DEVICES"] == 2 and t["BATCH_SIZE_PER_GPU"] == 512, "the simulated recipe is 2 x 512")
    t.update(OUTPUT_DIR=str(run_dir(arm, seed, phase)), RUN_NAME=rn)
    if phase == "smoke":
        t.update(SMOKE[k], LIMIT_VAL_BATCHES=2)
    cfg["EVAL"]["OUTPUT_DIR"] = str(run_dir(arm, seed, phase) / "eval")
    p = cfg["PFO"]
    p.update(ARM=arm, PHASE=phase)
    if k == "rol":
        base = ROLS[arm][0]
        p["BASE_ARM"] = base
        cfg["MODEL"]["INIT_FROM"] = str(checkpoint(base, seed, phase))
    cfg["RFS"] = {"TAU": TAU, "NEAR_PX": NEAR_PX, "SECTION": SECTION, "SIM_ARM": src, "MAX_ROWS": MAX_ROWS}
    r = router_of(arm)
    if p["ROUTER_KIND"] == "none":                    # w/o surface association: the router gates the sampling only
        cfg["RFS"].update(ROUTER_CHECKPOINT=str(r), ROUTER_SHA256=_sha(r))
    else:
        require(p["ROUTER"] == "r_s2d" and Path(p["ROUTER_CHECKPOINT"]) == SIM_ROUTER, "EvHandTrack's router block")
        p["ROUTER_CHECKPOINT"] = str(r)
    if arm in X2_ARMS:
        cfg["RFS"]["ROUTER"] = "r_s2dx2"
    return cfg


def router_of(arm):
    """The simulated r_s2d, or (X2_ARMS) the simulated r_s2dx2 -- which must be complete."""
    if arm not in X2_ARMS:
        return SIM_ROUTER
    require(SIM_X2.exists() and read_json(SIM_X2.with_suffix(".json")).get("status") == "COMPLETE",
            "the simulated r_s2dx2 is not trained yet: " + str(SIM_X2))
    return SIM_X2


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


def read_json(p):
    return json.loads(Path(p).read_text())


def save_json(p, obj, exclusive=False):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and p.exists():
        raise SystemExit("exists: " + str(p))
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")
    os.replace(tmp, p)
    return p


def sampling_router(cfg):
    """rfsgeo*: S45's ray association has no router; load the simulated r_s2d into S45-SM's state for RFS's on-hand
    decision only (no association / source hook is installed; the model never reads it)."""
    r = cfg.get("RFS", {}).get("ROUTER_CHECKPOINT")
    if not r:
        return None
    require(_sha(r) == cfg["RFS"]["ROUTER_SHA256"], "the sampling router changed since the config: " + r)
    from semkine import pfo_router as PR
    from semkine import rdr_hooks as RH
    from semkine import s45sm as SM
    require(SM._STATE["net"] is None, "a router is installed already")
    RH.enable()
    PR.register()
    SM._STATE.update(router=PR.KIND, net=SM._load_router(PR.KIND, r), checkpoint=str(r), onhand_tau=float(TAU), on={})
    print("RFS sampling router (on-hand decision only):", r, flush=True)
    return r


# ------------------------------------------------------------------------------------------------- training
def train(config):
    import yaml
    path = Path(config).resolve()
    cfg = yaml.safe_load(path.read_text())
    arm, seed, phase = cfg["PFO"]["ARM"], cfg["PFO"]["SEED"], cfg["PFO"]["PHASE"]
    parse(arm)
    require(path == config_path(arm, seed, phase).resolve(), "unregistered config path")
    require(cfg == build_config(arm, seed, phase), "config differs from the registry")
    rank0 = os.environ.get("LOCAL_RANK") in (None, "0")
    out = Path(cfg["TRAIN"]["OUTPUT_DIR"])
    if rank0 and out.exists():                        # fresh-only: an interrupted attempt is archived, never resumed
        dst = OUT / "preempted" / f"{out.name}__{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out), str(dst))
        print(f"fresh-only: archived the earlier attempt {out.name} to {dst}", flush=True)
    if kind(arm) == "rol":
        init = Path(cfg["MODEL"]["INIT_FROM"])
        if phase == "formal":
            rec = read_json(OUT / "receipts" / f"{run_name(ROLS[arm][0], seed, phase)}.json")
            require(rec["status"] == "PASS" and _sha(init) == rec["checkpoint_sha256_by_step"][str(FINAL["base"])],
                    "warm start is not the accepted base checkpoint")
        else:
            require(init.exists(), "missing smoke base checkpoint " + str(init))
    from semkine import rfs as RF
    state = RF.install(cfg["RFS"]["TAU"], cfg["RFS"]["NEAR_PX"], packet=True, max_rows=cfg["RFS"]["MAX_ROWS"])
    print("RFS:", state, flush=True)
    sampler = sampling_router(cfg)
    if kind(arm) == "rol":
        from tools.pmr import train_rol7 as W
        T = W.W.T
    else:
        from tools.pmr import train11 as W
        T = W.T
    T.build_config, T.config_path, T.parse = build_config, config_path, parse
    T.CODE = T.CODE + CODE
    if rank0:
        save_json(OUT / "launch" / f"{run_name(arm, seed, phase)}__{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json",
                  {"arm": arm, "seed": seed, "phase": phase, "config": str(path), "config_sha256": _sha(path),
                   "rfs": state, "sampling_router": sampler, "code_sha256": {c: _sha(ROOT / c) for c in CODE},
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    sys.argv = [sys.argv[0], "--config", str(path)]
    T.main()
    print("RFS stats:", RF.stats(), flush=True)


# ------------------------------------------------------------------------------------------------- acceptance
def accept(arm, seed, phase="formal"):
    """Engineering acceptance: the checkpoint grid, this config unchanged since launch, 2 x 512 at the registered step
    count, deterministic, the router block (r_s2d + PFA, or the ray association), the trunk, a rollout's warm start
    = the accepted base checkpoint and its start check; RFS installed with this config's rule (the launch record).
    Recorded, not gating: the data stream equals B2's (route-first changes the nodes, not the packets)."""
    from tools.pmr import arms_any2 as U
    from tools.pmr import ablrt as RT
    rn = run_name(arm, seed, phase)
    out = OUT / "receipts" / f"{rn}.json"
    require(not out.exists(), "accepted once: " + rn)
    d = run_dir(arm, seed, phase)
    cfg = build_config(arm, seed, phase)
    launch, meta, det = read_json(d / "pmr_launch.json"), read_json(d / "training_metadata.json"), read_json(d / "pmr_det.json")
    k = kind(arm)
    steps, every = cfg["TRAIN"]["MAX_STEPS"], cfg["TRAIN"]["SAVE_EVERY_N_STEPS"]
    grid = list(range(every, steps + 1, every))
    p = cfg["PFO"]
    design, variant = U.parse(BASES[parse(arm)[0]])
    mine = sorted((OUT / "launch").glob(f"{rn}__*.json"))
    rfs_rec = read_json(mine[-1]) if mine else {}
    ok = {"steps": all(checkpoint(arm, seed, phase, s).exists() for s in grid),
          "seed": meta["seed"] == seed and launch["pfo"]["SEED"] == seed,
          "max_steps": meta["max_steps"] == steps,
          "devices": meta["devices"] == 2 and meta["batch_size_per_gpu"] == 512,
          "config": launch["config_sha256"] == _sha(launch["config"]) and Path(launch["config"]) == config_path(
              arm, seed, phase).resolve() and read_yaml(launch["config"]) == cfg,
          "deterministic": bool(det["deterministic_algorithms"]) and det["trainer_deterministic"],
          "router": RT.router_ok(launch, cfg) if p["ROUTER_KIND"] == "none" else (
              launch["routing"]["router"] == "pfr" and launch["routing"]["assoc"] == "pfa"
              and float(launch["routing"]["onhand_tau"]) == 0.5
              and launch["router_checkpoint_sha256"] == _sha(p["ROUTER_CHECKPOINT"])),
          "trunk": launch["trunk"]["design"] == design and launch["trunk"]["variant"] == variant,
          "rfs": rfs_rec.get("rfs") == {"rfs": True, "tau": TAU, "near_px": NEAR_PX, "packet": True, "max_rows": MAX_ROWS}
          and rfs_rec.get("config_sha256") == launch["config_sha256"]
          and rfs_rec.get("sampling_router") == cfg["RFS"].get("ROUTER_CHECKPOINT")}
    if k == "rol":
        base = ROLS[arm][0]
        brec = read_json(OUT / "receipts" / f"{run_name(base, seed, phase)}.json")
        ok["rollout"] = launch["rol"] == cfg["ROL"]
        ok["init_from_base"] = (brec["status"] == "PASS" and launch["init_from"] == cfg["MODEL"]["INIT_FROM"]
                                and launch["init_from_sha256"] == brec["checkpoint_sha256_by_step"][
                                    str(FINAL["base"] if phase == "formal" else SMOKE["base"]["MAX_STEPS"])])
        starts = [read_json(x) for x in sorted(d.glob("rol_start_check_rank*.json"))]
        ok["start_check"] = len(starts) == 2 and all(x["delta_bitwise_equal"] and x["init_equal"] and x["global_step"] == 0
                                                     for x in starts)
    extra = {}
    if k == "base" and phase == "formal":
        try:
            from tools.pmr.common import b2_run
            rows = lambda r: {i: [x["sha256"] for x in read_json(Path(r) / f"pair_stream_rank{i}.json")[:2]]
                              for i in (0, 1)}
            extra["data_stream_equals_b2"] = rows(d) == rows(b2_run(phase, seed))
        except Exception as exc:  # noqa: BLE001  (recorded only)
            extra["data_stream_equals_b2"] = f"not compared: {exc!r}"
    from tools.cmp.common import now
    receipt = {"run": rn, "arm": arm, "seed": seed, "phase": phase, "checks": ok, "recorded": extra,
               "status": "PASS" if all(ok.values()) else "FAIL", "accepted_at_utc": now(),
               "checkpoint_sha256_by_step": {str(s): _sha(checkpoint(arm, seed, phase, s)) for s in grid
                                             if checkpoint(arm, seed, phase, s).exists()}}
    save_json(out, receipt, exclusive=True)
    print(receipt["status"], ok, extra, flush=True)
    require(receipt["status"] == "PASS", "acceptance failed: " + rn)


def read_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())


# ------------------------------------------------------------------------------------------------- readings
def read(arm, seed, task="e1", phase="formal", limit=None):
    import yaml
    from tools.cmp.common import now
    from tools.pmr import arms_any6  # noqa: F401
    from tools.pmr import arms_any2 as U
    from semkine import rdr_hooks as RH
    from tools.pmr import ablrt as RT
    from semkine import rfs as RF
    RF.install(TAU, NEAR_PX, packet=False)
    RF.install_stream()
    cfg = build_config(arm, seed, phase)
    sampling_router(cfg)
    RH.enable()
    RT.patch_loader()
    from tools.pfo import engine as E
    native_require = E.require

    def require_(cond, msg):                          # a single borrowed card (CUDA_VISIBLE_DEVICES = its UUID)
        if msg == "one GPU among 2-7" and os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-") \
                and "," not in os.environ["CUDA_VISIBLE_DEVICES"]:
            return
        native_require(cond, msg)
    E.require = require_
    from tools.pmr import run as R
    from tools.pfo2.run import _eval_cfg
    rn = run_name(arm, seed, phase)
    ck = checkpoint(arm, seed, phase)
    if phase == "formal":
        rec = read_json(OUT / "receipts" / f"{rn}.json")
        require(rec["status"] == "PASS", "training not accepted: " + rn)
        require(_sha(ck) == rec["checkpoint_sha256_by_step"][str(FINAL[kind(arm)])], "checkpoint changed since acceptance")
        spec_dir, E.OUT = OUT / "specs", OUT
    else:
        require(ck.exists(), "missing smoke checkpoint " + str(ck))
        spec_dir, E.OUT = OUT / "smoke" / "specs", OUT / "smoke"
        os.environ["PFO_SMOKE_DIR"] = str(OUT / "smoke")             # the engine's output root for limited readings
    cfg["MODEL"].pop("INIT_FROM", None)
    cfg.pop("ROL", None)
    ecfg = _eval_cfg(cfg, rn)
    epath = CONFIGS / ("eval" if phase == "formal" else "eval_smoke") / f"{rn}.yaml"
    text = yaml.safe_dump(ecfg, sort_keys=False)
    if epath.exists():
        require(epath.read_text() == text, "eval config differs: " + str(epath))
    else:
        epath.parent.mkdir(parents=True, exist_ok=True)
        epath.write_text(text)
    design, variant = U.parse(BASES[parse(arm)[0]])
    spec = {"kind": "routed", "checkpoint": str(ck), "checkpoint_sha256": _sha(ck), "config": str(epath), "arm": arm,
            "design": design, "variant": variant, "messages": "on"}
    sp = save_json(spec_dir / f"{rn}.json", spec)
    R.prepare(sp)
    R.deployed_router_for_mixes()
    from semkine import tpd_str as TF
    TF.install()
    TF.engine_hooks()
    out = E.run(f"{rn}__{FORM}", task, None, None, (), str(sp), limit=limit)
    save_json(Path(out) / "form.json", {"form": FORM, "rfs": {"tau": TAU, "near_px": NEAR_PX}, "written_utc": now()},
              exclusive=True)
    print("RFS stats:", RF.stats(), flush=True)


def report(arm, seed, task="e1"):
    return OUT / "readings" / f"{run_name(arm, seed)}__{FORM}" / task / "report.json"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "--config":          # Lightning's DDP relaunch of rank 1 (same argv as rank 0's T.main)
        return train(argv[1])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verb", choices=("configs", "train", "accept", "read"))
    ap.add_argument("--config")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--task", default="e1")
    ap.add_argument("--phase", default="formal", choices=PHASES)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--arms", help="configs: comma-separated arms (default: every arm whose router exists)")
    a = ap.parse_args(argv)
    if a.verb == "configs":
        for arm in (a.arms.split(",") if a.arms else [x for x in ARMS if x not in X2_ARMS or SIM_X2.exists()]):
            for seed in SEEDS:
                for phase in PHASES:
                    print(write_config(arm, seed, phase))
    elif a.verb == "train":
        train(a.config)
    elif a.verb == "accept":
        accept(a.arm, a.seed, a.phase)
    else:
        require((a.phase == "smoke") == (a.limit is not None), "--limit only (and always) with --phase smoke")
        read(a.arm, a.seed, a.task, a.phase, a.limit)


# ESMNet without the part loss (docs/EXPERIMENT_RECORDS.md#nopart-prereg-20261113; user 2026-11-13, queued after the
# simulated main seeds): tools/nopart/sim_ext.py adds the arms rfsrnp / rfsrnp_rol5. Every arm above is unchanged
# (tests/test_nopart.py). Guarded: if the extension cannot load, this module stays exactly as written above.
try:
    from tools.nopart import sim_ext as _nopart_ext
    _nopart_ext.extend(sys.modules[__name__])
except Exception as _exc:  # noqa: BLE001
    print("rfsu.sim: nopart extension not loaded:", repr(_exc), file=sys.stderr)


if __name__ == "__main__":
    main()
