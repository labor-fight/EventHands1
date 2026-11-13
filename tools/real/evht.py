#!/usr/bin/env python3
"""REAL amendment 5 (docs/EXPERIMENT_RECORDS.md#real-amend5-20261111): EvHandTrack and its paper ablations on the real data.

User 2026-11-11: "Evhandtrack这个方法也进行实验也加进去。在真实数据集下进行试验。" and "补上我图中的消融试验，在真实数据集下的"
(the paper's ablation table: (a) w/o surface association, Global MLP update, State-independent gating; (b) Single-step
only; Full model). Arms (each = the simulated-data registry's config of the same arm and seed -- tools/pmr/arms_any2.py
and tools/pmr/arms_rol2.py, registry 11 and rol1 added by tools/pmr/arms_any6.py, all frozen -- moved to the real data):

  d1r            EvHandTrack's trunk from scratch (PMR D1 R on B2's recipe, 6000 steps)
  d1r_rol5       Full model = EvHandTrack (K = 5 closed-loop rollout fine-tune from d1r step 6000, 1500 steps)
  d1r_rol1       Single-step only (the same fine-tune with K = 1)
  abgeo(_rol5)   w/o surface association (S45's ray association to the prior mesh instead of the ESMNet router)
  d1m(_rol5)     Global MLP update (part features concatenated + MLP instead of graph propagation and local readout)
  d1g0(_rol5)    State-independent gating (static edge-type gates instead of the prior-geometry gates)

Real-data changes, the same for every arm:
  data       DATA.ROOT / SPLITS_MANIFEST / HEIGHT / WIDTH = the real canvas n346 (346 x 260, splits_real.json: 12 training
             subjects; test kqj, lqh; dev wh), MODEL.RENDER_H / W / SCALE for it (as for the CMP arms of this study)
  pixels     every length the 240-wide recipe gives in pixels scaled by 346 / 240 so that it keeps its angle: the
             domain-randomisation shift (as for the CMP arms) and the ray association's band / front tolerance
  router     PFO.ROUTER_CHECKPOINT = r_s2d's recipe trained on the real training subjects (tools/real/evht_router.py;
             its per-event labels from tools/real/xag_build.py; a smoke config takes the router smoke's weights);
             abgeo has no router
  card       one card, 512 packets x 2 accumulation (the CMP contract of this study: one card, effective batch 1024)
             instead of 2 x 512 DDP; 6 data workers (12 for the runs in WORKERS_12, started before the change); no
             validation passes (TRAIN.LIMIT_VAL_BATCHES 0, as the CMP arms:
             the fixed last step is read)
  identity   names, output and config directories under outputs/real/20261110/pmr and configs/real_20261110/pmr; a
             REAL block. A rollout arm starts from its own base arm's real step-6000 checkpoint.
Readings: the streaming form (semkine/tpd_str.py; EvHandTrack's main-row form; the ABL paper table read every row in it)
through PFO's frozen E1 engine on the real test sequences, then the dev sequences (the CMP arms' order and grid).

    python tools/real/evht.py configs                      # write every config (formal + smoke)
    python tools/real/evht.py accept --arm A --seed S [--phase smoke]
    python tools/real/evht.py read --arm A --seed S [--task e1] [--phase smoke --limit N]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if os.environ.get("ERH_NAMESPACE") == "1":                          # EvRealHands (tools/erh/ns.py, tools/erh/run.py)
    from tools.erh import ns as _erh  # noqa: E402
    _erh.install()
from tools.real import ns  # noqa: E402

OUT = ns.OUT / "pmr"
CONFIGS = ns.CONFIGS / "pmr"
ROUTER_DIR = ns.OUT / "routers" / "r_s2d"
ROUTER = ROUTER_DIR / "r_s2d_final.ckpt"
ROUTER_SMOKE = ns.OUT / "routers" / "r_s2d_smoke" / "r_s2d_final.ckpt"     # the router smoke's 20-step weights
XAG = ns.OUT / "xag_sources"
SECTION = ns.SECTION if getattr(ns, "ERH", False) else "real-amend5-20261111"
CANVAS = "n346"
H, W = ns.CANVAS[CANVAS]["H"], ns.CANVAS[CANVAS]["W"]
PIX = W / 240.0
BASES = ("d1r", "abgeo", "d1m", "d1g0")
ROLS = {"d1r_rol5": "d1r", "d1r_rol1": "d1r", "abgeo_rol5": "abgeo", "d1m_rol5": "d1m", "d1g0_rol5": "d1g0"}
ARMS = BASES + tuple(ROLS)
PHASES = ("formal", "smoke")
FINAL = {"base": 6000, "rol": 1500}
SMOKE = {"base": {"MAX_STEPS": 20, "WARMUP_STEPS": 5, "SAVE_EVERY_N_STEPS": 20, "VAL_CHECK_INTERVAL": 20},
         "rol": {"MAX_STEPS": 10, "WARMUP_STEPS": 5, "SAVE_EVERY_N_STEPS": 10, "VAL_CHECK_INTERVAL": 10}}
NUM_WORKERS = 6
#: runs of the real study started (18:55 UTC and earlier) with 12 data workers, kept as they ran; every later run has 6
#: (host memory: ~2.4 GB per worker, the training's main process is the bottleneck, not the workers;
#: #real-amend5-run-20261111). A loader setting: the samples, their order and the recipe are unchanged.
WORKERS_12 = {("abgeo", 3407), ("abgeo", 3408), ("abgeo", 3409), ("abgeo_rol5", 3409), ("d1r", 3407), ("d1r", 3408),
              ("d1r", 3409), ("d1m", 3407), ("d1m", 3408)}
#: the paper's ablation table (row label, arm read)
PAPER = (("w/o surface association", "abgeo_rol5"), ("Global MLP update", "d1m_rol5"),
         ("State-independent gating", "d1g0_rol5"), ("Single-step only", "d1r_rol1"), ("Full model", "d1r_rol5"))
FORM = "str"


def require(cond, msg):
    if not cond:
        raise SystemExit("evht: " + msg)


def kind(arm):
    return "rol" if arm in ROLS else "base"


def run_name(arm, seed, phase="formal"):
    return f"pmr_{arm}_s{seed}" + ("" if phase == "formal" else f"_{phase}")


def run_dir(arm, seed, phase="formal"):
    return OUT / "runs" / run_name(arm, seed, phase)


def config_path(arm, seed, phase="formal"):
    return CONFIGS / f"{run_name(arm, seed, phase)}.yaml"


def checkpoint(arm, seed, phase="formal", step=None):
    step = step if step is not None else (FINAL[kind(arm)] if phase == "formal" else SMOKE[kind(arm)]["MAX_STEPS"])
    return run_dir(arm, seed, phase) / f"{run_name(arm, seed, phase)}-step={step}.ckpt"


def parse(arm):
    require(arm in ARMS, "not an arm of REAL amendment 5: " + str(arm))
    return (ROLS[arm], arm.rpartition("_")[2]) if arm in ROLS else (arm, None)


def sim_config(arm, seed):
    """The frozen simulated-data registry's formal config of the arm and seed."""
    from tools.pmr import arms_any6  # noqa: F401  (registry 11 and rol1)
    if arm in ROLS:
        from tools.pmr import arms_rol2 as R
        return R.build_config(arm, seed, "formal")
    from tools.pmr import arms_any2 as U
    return U.build_config(arm, seed, "formal")


def build_config(arm, seed, phase="formal"):
    parse(arm)
    require(seed in ns.SEEDS and phase in PHASES, "unregistered seed / phase")
    cfg = copy.deepcopy(sim_config(arm, seed))
    k = kind(arm)
    name = run_name(arm, seed, phase)
    t = cfg["TRAIN"]
    require(t["DEVICES"] == 2 and t["BATCH_SIZE_PER_GPU"] == 512 and t.get("ACCUMULATE_GRAD_BATCHES", 1) == 1,
            "the simulated recipe is 2 x 512")
    t.update(OUTPUT_DIR=str(run_dir(arm, seed, phase)), RUN_NAME=name, DEVICES=1, ACCUMULATE_GRAD_BATCHES=2,
             NUM_WORKERS=12 if phase == "smoke" or (not getattr(ns, "ERH", False) and (arm, seed) in WORKERS_12)
             else NUM_WORKERS,
             LIMIT_VAL_BATCHES=0)
    if phase == "smoke":
        t.update(SMOKE[k])
    cfg["EVAL"]["OUTPUT_DIR"] = str(run_dir(arm, seed, phase) / "eval")
    d = cfg["DATA"]
    d.update(ROOT=str(ns.CANVAS[CANVAS]["root"]), SPLITS_MANIFEST=str(ns.splits(CANVAS)), HEIGHT=H, WIDTH=W)
    m = cfg["MODEL"]
    m.update(RENDER_H=H, RENDER_W=W, RENDER_SCALE=W / 640.0, XYZ_BAND_PX=float(m["XYZ_BAND_PX"]) * PIX,
             XYZ_FRONT_PX=float(m["XYZ_FRONT_PX"]) * PIX)
    dr = cfg["AUG"]["DOMRAND"]
    require(dr.get("ENABLED") and float(dr["SHIFT_PX"]) == ns.B2_SHIFT_PX, "B2's domain randomisation")
    dr["SHIFT_PX"] = float(dr["SHIFT_PX"]) * PIX
    p = cfg["PFO"]
    p["PHASE"] = phase
    if p["ROUTER_KIND"] != "none":
        require(p["ROUTER"] == "r_s2d" and p["ROUTER_KIND"] == "pfr", "EvHandTrack's router block")
        p["ROUTER_CHECKPOINT"] = str(router(phase))
    if k == "rol":
        base = ROLS[arm]
        m["INIT_FROM"] = str(checkpoint(base, seed, phase))
    cfg["REAL"] = {"SECTION": SECTION, "CANVAS": CANVAS, "SPLITS_SHA256": _split_sha(),
                   "CONVERSION": str(ns.CANVAS[CANVAS]["root"] / "conversion.json"), "SIM_ARM": arm,
                   "PIXEL_SCALE": PIX, "ROUTER": str(router(phase)) if p["ROUTER_KIND"] != "none" else None}
    return cfg


def router(phase="formal"):
    return ROUTER if phase == "formal" else ROUTER_SMOKE


_SPLIT = {}


def _split_sha():
    if "sha" not in _SPLIT:
        from tools.cmp.common import sha
        _SPLIT["sha"] = sha(ns.splits(CANVAS))
    return _SPLIT["sha"]


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


def enter():
    """The router modules' frame (module constants and the event-image defaults bound at definition) and the
    evaluator's frame = the real canvas. Call before anything builds a router image or an evaluation packet."""
    from semkine import eval_track as ET
    from semkine import s45rt_router as RT
    from semkine import s45sm as SM
    from semkine import s45sm_fast as FAST
    SM.HEIGHT, SM.WIDTH = H, W
    RT.HEIGHT, RT.WIDTH = H, W
    RT.event_image.__defaults__ = (H, W)
    FAST.event_image.__defaults__ = (H, W)
    ET.H, ET.W = H, W


# ------------------------------------------------------------------------------------------------- acceptance
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


def accept(arm, seed, phase="formal"):
    """Engineering acceptance of a finished training: every checkpoint of the grid exists, the launch record names this
    config (unchanged since), the trunk / router block / warm start are the registered ones, the metadata is one card
    x 512 x 2 at the registered step count, the deterministic policy was on; receipts/<run>.json."""
    from tools.cmp.common import now, sha
    rn = run_name(arm, seed, phase)
    out = OUT / "receipts" / f"{rn}.json"
    require(not out.exists(), "accepted once: " + rn)
    d = run_dir(arm, seed, phase)
    cfg = build_config(arm, seed, phase)
    launch, meta, det = read_json(d / "pmr_launch.json"), read_json(d / "training_metadata.json"), read_json(d / "pmr_det.json")
    k = kind(arm)
    steps = cfg["TRAIN"]["MAX_STEPS"]
    every = cfg["TRAIN"]["SAVE_EVERY_N_STEPS"]
    grid = list(range(every, steps + 1, every))
    p = cfg["PFO"]
    ok = {"steps": all(checkpoint(arm, seed, phase, s).exists() for s in grid),
          "seed": meta["seed"] == seed and launch["pfo"]["SEED"] == seed,
          "max_steps": meta["max_steps"] == steps,
          "card": meta["devices"] == 1 and meta["batch_size_per_gpu"] == 512 and meta.get("accumulate_grad_batches") == 2,
          "config": launch["config_sha256"] == sha(launch["config"]) and Path(launch["config"]) == config_path(
              arm, seed, phase).resolve() and read_json_yaml(launch["config"]) == cfg,
          "deterministic": bool(det["deterministic_algorithms"]) and det["trainer_deterministic"],
          "router": (launch["router_checkpoint_sha256"] is None) if p["ROUTER_KIND"] == "none" else (
              launch["router_checkpoint_sha256"] == sha(router(phase)) == read_json(router(phase).with_suffix(
                  ".json"))["checkpoint_sha256"]),
          "trunk": launch["trunk"]["module"] == cfg["PMR"]["MODULE"] and launch["trunk"]["design"] == cfg["PMR"]["DESIGN"]}
    if k == "rol":
        base = ROLS[arm]
        brec = read_json(OUT / "receipts" / f"{run_name(base, seed, phase)}.json")
        ok["rollout"] = launch["rol"] == cfg["ROL"]
        ok["init_from_base"] = (brec["status"] == "PASS" and launch["init_from"] == cfg["MODEL"]["INIT_FROM"]
                                and launch["init_from_sha256"] == brec["checkpoint_sha256_by_step"][
                                    str(FINAL["base"] if phase == "formal" else SMOKE["base"]["MAX_STEPS"])])
        starts = [read_json(x) for x in sorted(d.glob("rol_start_check_rank*.json"))]
        ok["start_check"] = bool(starts) and all(x["delta_bitwise_equal"] and x["init_equal"] and x["global_step"] == 0
                                                 for x in starts)
    receipt = {"run": rn, "arm": arm, "seed": seed, "phase": phase, "checks": ok,
               "status": "PASS" if all(ok.values()) else "FAIL", "accepted_at_utc": now(),
               "checkpoint_sha256_by_step": {str(s): sha(checkpoint(arm, seed, phase, s)) for s in grid
                                             if checkpoint(arm, seed, phase, s).exists()}}
    save_json(out, receipt, exclusive=True)
    print(receipt["status"], ok, flush=True)
    require(receipt["status"] == "PASS", "acceptance failed: " + rn)


def read_json_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())


# ------------------------------------------------------------------------------------------------- readings
def patch_engine():
    """PFO's frozen engine on the real data (tools/real/run.py `patch_engine` without the CMP loader): the sequence list
    (test, then dev; endpoint counts unchecked; looked up in val / test / train of splits_real.json), the output root,
    and the one-GPU check also accepting a single borrowed card (CUDA_VISIBLE_DEVICES = its UUID)."""
    from tools.pfo import common as PC
    from tools.pfo import engine as E
    E.OUT = PC.OUT = OUT
    E.ZGZ = ns.TEST + ns.DEV
    E.COUNTS = {}
    native_require = E.require

    def require_(cond, msg):
        if msg == "one GPU among 2-7" and os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-") \
                and "," not in os.environ["CUDA_VISIBLE_DEVICES"]:
            return
        native_require(cond, msg)
    E.require = require_

    def sequences(cfg, names):
        from semkine.dataset import sequences_for_split, splits_manifest
        root = Path(cfg["DATA"]["ROOT"])
        require(root == ns.CANVAS[CANVAS]["root"], "not the real data root")
        found = {}
        for split in ("val", "test", "train"):
            found.update(dict(sequences_for_split(root, split, splits_manifest(cfg))))
        require(all(s in found for s in names), "missing sequences")
        return root, [(s, found[s]) for s in names]
    E.sequences = sequences
    return E


def read(arm, seed, task="e1", phase="formal", limit=None):
    import yaml
    from tools.cmp.common import now, sha
    enter()
    from tools.pmr import arms_any6  # noqa: F401
    from semkine import rdr_hooks as RH
    from tools.pmr import ablrt as RT
    RH.enable()
    RT.patch_loader()
    E = patch_engine()
    from tools.pmr import run as R
    from tools.pfo2.run import _eval_cfg
    rn = run_name(arm, seed, phase)
    ck = checkpoint(arm, seed, phase)
    if phase == "formal":
        rec = read_json(OUT / "receipts" / f"{rn}.json")
        require(rec["status"] == "PASS", "training not accepted: " + rn)
        require(sha(ck) == rec["checkpoint_sha256_by_step"][str(FINAL[kind(arm)])], "checkpoint changed since acceptance")
        spec_dir = OUT / "specs"
    else:
        require(ck.exists(), "missing smoke checkpoint " + str(ck))
        spec_dir = OUT / "smoke" / "specs"
        os.environ["PFO_SMOKE_DIR"] = str(OUT / "smoke")
    cfg = build_config(arm, seed, phase)
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
    spec = {"kind": "routed", "checkpoint": str(ck), "checkpoint_sha256": sha(ck), "config": str(epath), "arm": arm,
            "design": cfg["PMR"]["DESIGN"], "variant": cfg["PMR"]["VARIANT"], "messages": "on"}
    sp = save_json(spec_dir / f"{rn}.json", spec)
    R.prepare(sp)
    R.deployed_router_for_mixes()
    from semkine import tpd_str as TF
    TF.install()
    TF.engine_hooks()
    out = E.run(f"{rn}__{FORM}", task, None, None, (), str(sp), limit=limit)
    save_json(Path(out) / "form.json", {"form": FORM, "written_utc": now()}, exclusive=True)


def report(arm, seed, task="e1"):
    return OUT / "readings" / f"{run_name(arm, seed)}__{FORM}" / task / "report.json"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verb", choices=("configs", "accept", "read"))
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--task", default="e1")
    ap.add_argument("--phase", default="formal", choices=PHASES)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    if a.verb == "configs":
        for arm in ARMS:
            for seed in ns.SEEDS:
                for phase in PHASES:
                    if phase == "smoke" and seed != 3407:
                        continue
                    print(write_config(arm, seed, phase))
    elif a.verb == "accept":
        accept(a.arm, a.seed, a.phase)
    else:
        require((a.phase == "smoke") == (a.limit is not None), "--limit only (and always) with --phase smoke")
        read(a.arm, a.seed, a.task, a.phase, a.limit)


if __name__ == "__main__":
    main()
