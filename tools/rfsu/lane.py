#!/usr/bin/env python3
"""Route-first unification driver (docs/EXPERIMENT_RECORDS.md#rfsu-20261113): GPU lanes pulling units from one queue.

A unit = one arm and seed of one study namespace: train the base (6000) -> accept -> train the fine-tune -> accept ->
streaming E1 reading of the fine-tune (and of the base when `read_base`). Every stage is skipped when its receipt / report
exists, so a lane restarted after a gpu-borrow preemption resumes its unit (the trainers are fresh-only: an interrupted
run directory is archived and the run starts again; an interrupted reading directory is archived here). A failed stage
fails the unit (kept, never retried); the lane goes on with the next unit.

  ns real   tools/real/rfs.py                          one card (512 x 2 accumulation)
  ns erh    tools/erh/run.py tools.real.rfs            one card, the EvRealHands namespace
  ns sim    tools/rfsu/sim.py                          two cards (2 x 512 DDP)

Lanes: `run --lane NAME` with CUDA_VISIBLE_DEVICES = its card(s) (direct indices, or the UUIDs gpu-borrow sets). A
two-card lane takes two-card units while any is unclaimed, then splits into two one-card lanes. A unit is claimed only
when its `after` units are DONE. STOP (outputs/rfsu/20261113/STOP) stops claiming; running units finish.

    python tools/rfsu/lane.py run --lane p45 [--wait-free]
    python tools/rfsu/lane.py status
    python tools/rfsu/lane.py skip-blocked [--loop 300]

Units that can never start (an `after` unit FAIL / SKIPPED / SUPERSEDED) are marked SKIPPED; an external unit (gpus 0) may
name an `owner` unit whose run marks it -- if the owner ends without marking it, it is marked DONE (owner DONE) or FAIL.
Lanes do this at every claim; `skip-blocked --loop` does it for lanes started before this rule existed (2026-11-13).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/rfsu/20261113"
QUEUE = BASE / "queue.json"
STATE = BASE / "lanes" / "state.json"
LOCK = BASE / "lanes" / "lock"
LOGS = BASE / "logs"
STOP = BASE / "STOP"
PY = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"
POLICY_ENV = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "3407", "OMP_NUM_THREADS": "4",
              "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
              "PYTHONDONTWRITEBYTECODE": "1"}
NS = {"real": {"out": ROOT / "outputs/real/20261110/pmr", "cfg": ROOT / "configs/real_20261110/pmr",
               "tool": [PY, "tools/real/rfs.py"]},
      "erh": {"out": ROOT / "outputs/erh/20261111/pmr", "cfg": ROOT / "configs/erh_20261111/pmr",
              "tool": [PY, "tools/erh/run.py", "tools.real.rfs"]},
      "sim": {"out": BASE / "sim", "cfg": ROOT / "configs/rfsu_20261113/sim", "tool": [PY, "tools/rfsu/sim.py"]}}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(lane, msg):
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / f"lane_{lane}.log", "a") as f:
        f.write(f"{now()} {msg}\n")


class Locked:
    def __enter__(self):
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(LOCK, "a+")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *a):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(st):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1) + "\n")
    os.replace(tmp, STATE)


def queue():
    return json.loads(QUEUE.read_text())


def rn(arm, seed):
    return f"pmr_{arm}_s{seed}"


def receipt_ok(ns, arm, seed):
    p = NS[ns]["out"] / "receipts" / f"{rn(arm, seed)}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("status") == "PASS"


def report(ns, arm, seed):
    return NS[ns]["out"] / "readings" / f"{rn(arm, seed)}__str" / "e1" / "report.json"


DEAD = ("FAIL", "SKIPPED", "SUPERSEDED")


def _skip_blocked(st, q):
    """In place, to a fixed point (caller holds the lock): a unit with an `after` unit FAIL / SKIPPED / SUPERSEDED can never
    start -> SKIPPED; an external unit (gpus 0) whose `owner` -- the unit whose run produces and marks it -- has ended without
    marking it -> DONE when the owner is DONE (it could not have finished without it), else FAIL. So lanes and the cost block
    never wait forever on a unit that cannot run. Returns the ids changed."""
    changed, again = [], True
    while again:
        again = False
        for u in q:
            if u["id"] in st:
                continue
            own = u.get("owner")
            if u["gpus"] == 0 and own and st.get(own, {}).get("status") in DEAD + ("DONE",):
                s = st[own]["status"]
                st[u["id"]] = {"lane": "skip", "status": "DONE" if s == "DONE" else "FAIL", "start": now(), "end": now(),
                               "note": f"owner {own} ended {s} without marking it"}
                changed.append(u["id"])
                again = True
                continue
            bad = [d for d in u.get("after", []) if st.get(d, {}).get("status") in DEAD]
            if bad:
                st[u["id"]] = {"lane": "skip", "status": "SKIPPED", "start": now(), "end": now(),
                               "note": f"after {bad[0]} {st[bad[0]]['status']}"}
                changed.append(u["id"])
                again = True
    return changed


def skip_blocked():
    with Locked():
        st = load_state()
        changed = _skip_blocked(st, queue())
        if changed:
            save_state(st)
    return changed


def claim(lane, ngpu):
    """The first unclaimed unit for `ngpu` cards whose `after` units are DONE: (unit, more) -- more = whether unclaimed
    units of this size remain (waiting on dependencies). Units that can never start are marked first (_skip_blocked)."""
    with Locked():
        st = load_state()
        if _skip_blocked(st, queue()):
            save_state(st)
        mine = [u for u in queue() if u["gpus"] == ngpu and st.get(u["id"], {}).get("lane") == lane
                and st[u["id"]]["status"] == "RUNNING"]
        if mine:
            return mine[0], True
        if STOP.exists():
            return None, False
        free = [u for u in queue() if u["gpus"] == ngpu and u["id"] not in st]
        for u in free:
            if all(st.get(d, {}).get("status") == "DONE" for d in u.get("after", [])):
                st[u["id"]] = {"lane": lane, "status": "RUNNING", "start": now(), "cvd": os.environ.get("CUDA_VISIBLE_DEVICES")}
                save_state(st)
                return u, True
        return None, bool(free)


def finish(uid, status, note=""):
    with Locked():
        st = load_state()
        st[uid].update(status=status, end=now(), note=note)
        save_state(st)


def stage(lane, uid, name, cmd, env):
    LOGS.mkdir(parents=True, exist_ok=True)
    path = LOGS / f"{uid.replace(':', '_')}__{name}.log"
    log(lane, f"{uid} {name} start: {' '.join(cmd)}")
    with open(path, "a") as f:
        f.write(f"\n===== {now()} {' '.join(cmd)} (CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')})\n")
        f.flush()
        r = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
    log(lane, f"{uid} {name} exit {r.returncode}")
    return r.returncode == 0


def archive_partial_reading(ns, arm, seed):
    d = NS[ns]["out"] / "readings" / f"{rn(arm, seed)}__str"
    if d.exists() and not report(ns, arm, seed).exists():
        dst = NS[ns]["out"] / "readings_preempted" / f"{d.name}__{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), str(dst))
        return dst
    return None


def run_unit(lane, u, cvd):
    ns, base, rol, seed = u["ns"], u["base"], u["rol"], u["seed"]
    tool, cfgd = NS[ns]["tool"], NS[ns]["cfg"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": cvd, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    if ns == "sim":
        env["NCCL_P2P_DISABLE"] = "1"
    renv = {**env, **POLICY_ENV, "CUDA_VISIBLE_DEVICES": cvd.split(",")[0]}
    for arm, train in ((base, u.get("train_base", True)), (rol, True)):
        ok = receipt_ok(ns, arm, seed)
        if ok is False:
            return "FAIL", f"{arm} acceptance FAIL on disk"
        if ok:
            continue
        if not train:
            return "FAIL", f"{arm} is not accepted and this unit does not train it"
        if not stage(lane, u["id"], f"train_{arm}", tool + ["train", "--config", str(cfgd / f"{rn(arm, seed)}.yaml")], env):
            return "FAIL", f"train {arm}"
        if not stage(lane, u["id"], f"accept_{arm}", tool + ["accept", "--arm", arm, "--seed", str(seed)], env):
            return "FAIL", f"accept {arm}"
    for arm in ([rol] + ([base] if u.get("read_base") else [])):
        if report(ns, arm, seed).exists():
            continue
        moved = archive_partial_reading(ns, arm, seed)
        if moved:
            log(lane, f"{u['id']} archived a partial reading to {moved}")
        if not stage(lane, u["id"], f"read_{arm}", tool + ["read", "--arm", arm, "--seed", str(seed)], renv):
            return "FAIL", f"read {arm}"
    return "DONE", ""


def gpu_free(cvd):
    """Direct cards only: no compute process on any of them."""
    for c in cvd.split(","):
        out = subprocess.run(["nvidia-smi", "-i", c, "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout
        if out.strip():
            return False
    return True


def lane_loop(lane, cvd, ngpu, wait_free):
    if wait_free:
        quiet = 0
        while quiet < 4:
            quiet = quiet + 1 if gpu_free(cvd) else 0
            time.sleep(30)
        log(lane, f"cards {cvd} free, starting")
    while True:
        u, more = claim(lane, ngpu)
        if u is None:
            if more and not STOP.exists():
                time.sleep(60)
                continue
            log(lane, f"no {ngpu}-card unit left ({'STOP' if STOP.exists() else 'queue empty'})")
            return
        log(lane, f"claimed {u['id']} on {cvd}")
        try:
            status, note = run_unit(lane, u, cvd)
        except Exception as exc:  # noqa: BLE001
            status, note = "FAIL", repr(exc)
        finish(u["id"], status, note)
        log(lane, f"{u['id']} {status} {note}")


def run(lane, wait_free):
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cards = [c for c in cvd.split(",") if c]
    if not cards or len(cards) > 2:
        raise SystemExit("lane: one or two cards in CUDA_VISIBLE_DEVICES")
    log(lane, f"lane start pid {os.getpid()} cards {cards}")
    if len(cards) == 2:
        lane_loop(lane, cvd, 2, wait_free)
        threads = [threading.Thread(target=lane_loop, args=(f"{lane}.{i}", c, 1, False)) for i, c in enumerate(cards)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        lane_loop(lane, cvd, 1, wait_free)
    log(lane, "lane end")


def status():
    st = load_state()
    for u in queue():
        s = st.get(u["id"], {})
        print(f"{u['id']:28s} {s.get('status', 'PENDING'):8s} {s.get('lane', ''):8s} {s.get('start', '')} "
              f"{s.get('end', '')} {s.get('note', '')}")


def mark(uid, status_, note):
    """External units (`gpus` 0, e.g. router:sim): their own scripts mark them DONE / FAIL; lanes never claim them."""
    with Locked():
        st = load_state()
        u = [q for q in queue() if q["id"] == uid]
        if not u or u[0]["gpus"] != 0:
            raise SystemExit(f"{uid}: not an external unit of the queue")
        st.setdefault(uid, {"lane": "external", "start": now()}).update(status=status_, end=now(), note=note)
        save_state(st)


def skip_loop(every):
    """`_skip_blocked` every `every` s until no unit is PENDING / RUNNING (lanes started before 2026-11-13 04:31 UTC run an
    older claim() without it; this covers them)."""
    while True:
        for uid in skip_blocked():
            log("skip", f"{uid} marked: {load_state()[uid]['status']} {load_state()[uid]['note']}")
        st = load_state()
        if all(st.get(u["id"], {}).get("status") not in (None, "RUNNING") for u in queue()):
            log("skip", "no unit PENDING / RUNNING; exit")
            return
        time.sleep(every)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verb", choices=("run", "status", "mark", "skip-blocked"))
    ap.add_argument("--lane")
    ap.add_argument("--wait-free", action="store_true")
    ap.add_argument("--unit")
    ap.add_argument("--status", choices=("RUNNING", "DONE", "FAIL"))
    ap.add_argument("--note", default="")
    ap.add_argument("--loop", type=int, help="skip-blocked: repeat every N s until no unit is PENDING / RUNNING")
    a = ap.parse_args()
    if a.verb == "run":
        run(a.lane, a.wait_free)
    elif a.verb == "mark":
        mark(a.unit, a.status, a.note)
    elif a.verb == "skip-blocked":
        if a.loop:
            skip_loop(a.loop)
        else:
            print(skip_blocked())
    else:
        status()


if __name__ == "__main__":
    main()
