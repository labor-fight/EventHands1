#!/usr/bin/env python3
"""Move a running formal training off its card to any card with room (docs/EXPERIMENT_RECORDS.md#real-amend5-run-20261111).

tools/real/move.py moves a training to one named card and refuses when another card would also have room. On
2026-11-12 three cards were idle while card 3 ran three trainings, so any idle card will do. This tool keeps move.py's
procedure (its helpers are imported unchanged) but does not edit the policy: it relies on the limits steer.py already set
(max_jobs = least-loaded live card's GPU tasks + 1), and checks, before waiting and again right before the kill, that
after the kill the source card has no room (its GPU tasks >= max_jobs, or its trainings of the arm >= max_per_arm) while
at least one other live card has room.

  1  checks as above; the task is a RUNNING formal training whose pid runs the task's config
  2  wait for a loadable checkpoint at most --max-loss-min old with no save in progress (move.wait_checkpoint)
  3  SIGKILL the training's process tree; wait until the source packer has reaped it (FAILED)
  4  under the table lock: the task -> PENDING (requeued: moved off card <src>), dependants SKIPPED since the kill ->
     PENDING; the training resumes from that checkpoint wherever a packer starts it (tools/real/train.py, amendment 2)
  5  wait until a card runs it

    python tools/real/move_off.py --task erh:train:ehpe:3407 [--max-loss-min 5] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

from tools.real import move as M  # noqa: E402
from tools.real import pack as P  # noqa: E402


def room(key):
    """(source blocked after the kill, other live cards with room after the kill, limits) under the current policy."""
    st = M.state()
    T, t = st["tasks"], st["tasks"][key]
    pol = json.loads(P.POLICY.read_text())
    cap = pol.get("max_jobs", P.DEFAULT_POLICY["max_jobs"])
    arm_cap = dict(P.DEFAULT_POLICY.get("max_per_arm", {}), **pol.get("max_per_arm", {})).get(t["arm"], 99)
    src = str(t["card"])

    def arm_count(c):
        return sum(1 for u in T.values() if u["status"] == "RUNNING" and u.get("card") == c
                   and u["arm"] == t["arm"] and u["kind"] in ("smoke", "train"))
    after = {c: (M.gpu_count(T, c) - (c == src), arm_count(c) - (c == src)) for c in M.live_cards(st)}
    blocked = after[src][0] >= cap or after[src][1] >= arm_cap
    open_ = [c for c, (n, na) in after.items() if c != src and n < cap and na < arm_cap]
    return blocked, open_, (cap, arm_cap, after)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--max-loss-min", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    key = a.task

    t = M.state()["tasks"][key]
    if t["kind"] != "train" or t["status"] != "RUNNING":
        raise SystemExit(f"{key}: not a RUNNING formal training ({t['kind']}, {t['status']})")
    src, pid, cfg = str(t["card"]), int(t["pid"]), t["cmd"][-1]
    if cfg not in M.cmdline(pid):
        raise SystemExit(f"pid {pid} does not run {cfg}")
    blocked, open_, lim = room(key)
    run = Path(t["outputs"][0])
    M.say(f"{key}: off card {src}; pid {pid}; run {run.name}; max_jobs {lim[0]}, max_per_arm {lim[1]}; "
          f"(jobs, arm trainings) after the kill {lim[2]}; source blocked {blocked}; cards with room {open_}")
    if not blocked or not open_:
        raise SystemExit("no move: the source card would take it again, or no other card has room")
    if a.dry_run:
        M.say("dry run: newest checkpoint", M.newest(run))
        return

    step, ckpt = M.wait_checkpoint(run, a.max_loss_min * 60)
    blocked, open_, lim = room(key)
    if M.state()["tasks"][key]["status"] != "RUNNING" or not blocked or not open_:
        raise SystemExit(f"no move after the wait: status {M.state()['tasks'][key]['status']}, source blocked "
                         f"{blocked}, cards with room {open_} ({lim})")
    M.say(f"resume point {ckpt.name} (global_step {step}, {(time.time() - ckpt.stat().st_mtime) / 60:.1f} min old); "
          f"cards with room {open_}")

    procs = M.tree(pid)
    kill_at = P.now()
    for p in reversed(procs):
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
    M.say(f"SIGKILL {len(procs)} processes (training {pid} and its workers)")
    t0 = time.time()
    while M.state()["tasks"][key]["status"] == "RUNNING":
        if time.time() - t0 > 180:
            raise SystemExit("the source packer did not reap the training within 180 s -- fix by hand")
        time.sleep(1)

    with P.locked() as s:
        TT = s["tasks"]
        u = TT[key]
        if u["status"] != "FAILED":
            raise SystemExit(f"{key} is {u['status']} after the kill, expected FAILED")
        u.update(status="PENDING", card=None, pid=None)
        u.pop("rc", None)
        u.pop("ended", None)
        u.setdefault("requeued", []).append({"at": P.now(), "reason": f"moved off card {src} after {ckpt.name} "
                                             "(tools/real/move_off.py, #real-amend5-run-20261111)"})
        reset, changed = [], True
        while changed:
            changed = False
            for k, v in TT.items():
                if (v["status"] == "SKIPPED" and v.get("ended", "") >= kill_at
                        and any(d == key or d in reset for d in v["deps"])):
                    v["status"] = "PENDING"
                    v.pop("ended", None)
                    reset.append(k)
                    changed = True
    M.say(f"{key} -> PENDING; dependants restored: {reset}")

    t0 = time.time()
    while True:
        u = M.state()["tasks"][key]
        if u["status"] == "RUNNING":
            M.say(f"{key}: RUNNING on card {u.get('card')} (pid {u.get('pid')})")
            return
        if time.time() - t0 > 600:
            M.say(f"warning: no card started {key} within 600 s (status {u['status']})")
            return
        time.sleep(2)


if __name__ == "__main__":
    main()
