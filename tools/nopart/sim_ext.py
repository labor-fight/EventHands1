"""ESMNet without the part loss, simulated trunk arms (docs/EXPERIMENT_RECORDS.md#nopart-prereg-20261113).

`extend(S)` adds two arms to tools/rfsu/sim.py `S` (called from a guarded hook at the end of that file); every existing
arm is untouched (tests/test_nopart.py checks that build_config of every existing arm, seed and phase is unchanged):

  rfsrnp       = rfsr      (route-first EvHandTrack trunk from scratch, 6000 steps, 2 x 512 DDP)
  rfsrnp_rol5  = rfsr_rol5 (K = 5 rollout fine-tune, 1500 steps, warm start = rfsrnp step 6000)

with only PFO.ROUTER_CHECKPOINT = the simulated r_s2dx2np (tools/nopart/router_sim.py: r_s2dx2's recipe without the
part loss) and the label RFS.ROUTER = "r_s2dx2np". Every process of these arms enables semkine/nopart.py, so the router
decodes with c_i alone. Runs, receipts and readings live in RFSU's simulated folders (the lanes look there); the router in
outputs/nopart/20261113/sim/routers.

The router trains inside the first rfsrnp unit: rank 0 of `train` for rfsrnp (formal) trains it on the lane's two cards
when it is missing, marks the external queue unit `router:simnp`, then runs the zero-training router comparison against
r_s2dx2 (tools/nopart/router_compare.py, descriptive; a failure there never stops the training).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SECTION = "nopart-prereg-20261113"
NP_OUT = ROOT / "outputs/nopart/20261113"
ROUTER = NP_OUT / "sim/routers/r_s2dx2np/r_s2dx2np_final.ckpt"
PREFLIGHT = NP_OUT / "sim/routers/r_s2dx2np_preflight/r_s2dx2np_final.ckpt"
BASE, ROL = "rfsrnp", "rfsrnp_rol5"
ARMS = (BASE, ROL)
UNIT = "router:simnp"
PY = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"


def log(msg):
    p = NP_OUT / "logs" / "sim_ext.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} pid {os.getpid()} {msg}\n")


def router_complete(path=ROUTER):
    j = Path(path).with_suffix(".json")
    return Path(path).exists() and j.exists() and json.loads(j.read_text()).get("status") == "COMPLETE"


def mark(status, note):
    subprocess.run([PY, str(ROOT / "tools/rfsu/lane.py"), "mark", "--unit", UNIT, "--status", status, "--note", note],
                   cwd=str(ROOT), check=False)


def unit_status():
    """router:simnp's status in RFSU's lane state (PENDING when absent or unreadable)."""
    try:
        st = json.loads((ROOT / "outputs/rfsu/20261113/lanes/state.json").read_text())
        return st.get(UNIT, {}).get("status", "PENDING")
    except Exception:  # noqa: BLE001
        return "PENDING"


def ensure_router():
    """Rank 0, before rfsrnp's base training: train r_s2dx2np on this process's cards when it is missing (fresh-only: a
    partial router folder is archived first), mark router:simnp, then the router comparison (descriptive)."""
    if not router_complete():
        d = ROUTER.parent
        if d.exists():
            dst = NP_OUT / "sim/routers_preempted" / f"{d.name}__{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(d), str(dst))
            log(f"archived a partial router folder to {dst}")
        env = {k: v for k, v in os.environ.items() if k not in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR",
                                                               "MASTER_PORT", "NODE_RANK")}
        env.update(NCCL_P2P_DISABLE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        out = NP_OUT / "logs" / "router_r_s2dx2np.log"
        out.parent.mkdir(parents=True, exist_ok=True)
        log(f"router training starts on CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}")
        with open(out, "a") as f:
            r = subprocess.run([PY, str(ROOT / "tools/nopart/router_sim.py"), "--router", "r_s2dx2np", "--phase", "formal"],
                               cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0 or not router_complete():
            mark("FAIL", f"router training rc={r.returncode}")
            log(f"router training FAILED rc={r.returncode}")
            raise SystemExit("nopart: the r_s2dx2np router failed (see outputs/nopart/20261113/logs/router_r_s2dx2np.log)")
    if unit_status() != "DONE":                     # also after a restart that found the router complete but unmarked
        sha = json.loads(ROUTER.with_suffix(".json").read_text()).get("checkpoint_sha256")
        mark("DONE", f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} checkpoint_sha256={sha}")
        log("router DONE " + str(sha))
    compare = NP_OUT / "router_compare" / "DONE"
    if not compare.exists():
        try:
            env = {k: v for k, v in os.environ.items() if k not in ("LOCAL_RANK", "RANK", "WORLD_SIZE")}
            env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "").split(",")[0]
            out = NP_OUT / "logs" / "router_compare.log"
            with open(out, "a") as f:
                r = subprocess.run([PY, str(ROOT / "tools/nopart/router_compare.py"), "all"], cwd=str(ROOT), env=env,
                                   stdout=f, stderr=subprocess.STDOUT, timeout=3 * 3600)
            log(f"router comparison rc={r.returncode}")
        except Exception as exc:  # noqa: BLE001  (descriptive; never stops the training)
            log(f"router comparison failed: {exc!r}")


def extend(S):
    """Add rfsrnp / rfsrnp_rol5 to tools/rfsu/sim.py's module `S` (all assignments at the end)."""
    if getattr(S, "NOPART_ARMS_ADDED", False):
        return
    bases = dict(S.BASES, **{BASE: "d1r"})
    rols = dict(S.ROLS, **{ROL: (BASE, "d1r_rol5")})
    native_router_of, native_build, native_train, native_read = S.router_of, S.build_config, S.train, S.read

    def router_of(arm):
        if arm in ARMS:
            return ROUTER                              # the path; its completion is checked where it is used
        return native_router_of(arm)

    def build_config(arm, seed, phase="formal"):
        cfg = native_build(arm, seed, phase)
        if arm in ARMS:
            if phase == "smoke":
                cfg["PFO"]["ROUTER_CHECKPOINT"] = str(PREFLIGHT)
            cfg["RFS"]["ROUTER"] = "r_s2dx2np"
        return cfg

    def train(config):
        import yaml
        cfg = yaml.safe_load(Path(config).read_text())
        if cfg.get("PFO", {}).get("ARM") in ARMS:
            from semkine import nopart as NP
            NP.enable()
            msg = f"train {cfg['PFO']['ARM']} s{cfg['PFO'].get('SEED')} {cfg['PFO'].get('PHASE')}: c_i-only decode enabled, " \
                  f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}"
            print("nopart:", msg, flush=True)
            log(msg)
            if cfg["PFO"]["ARM"] == BASE and cfg["PFO"].get("PHASE") == "formal" and \
                    os.environ.get("LOCAL_RANK") in (None, "0"):
                ensure_router()
            ck = Path(cfg["PFO"]["ROUTER_CHECKPOINT"])
            S.require(router_complete(ck), "the r_s2dx2np router is not complete: " + str(ck))
        return native_train(config)

    def read(arm, seed, task="e1", phase="formal", limit=None):
        if arm in ARMS:
            from semkine import nopart as NP
            NP.enable()
        return native_read(arm, seed, task, phase, limit)

    S.BASES, S.ROLS = bases, rols
    S.ARMS = tuple(bases) + tuple(rols)
    S.router_of, S.build_config, S.train, S.read = router_of, build_config, train, read
    S.NOPART_ARMS_ADDED = True


__all__ = ["ARMS", "BASE", "PREFLIGHT", "ROL", "ROUTER", "UNIT", "ensure_router", "extend", "router_complete"]
