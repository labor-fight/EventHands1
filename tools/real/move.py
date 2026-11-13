#!/usr/bin/env python3
"""Move a running formal training of the real study to another card (docs/EXPERIMENT_RECORDS.md#real-amend3-20261111).

The packers (tools/real/pack.py) never move a running task, and a packer that reaps a killed child marks it FAILED (its
dependants SKIPPED). A formal training resumes from its newest loadable checkpoint when it starts again
(tools/real/train.py, amendment 2), so a training is moved by killing it right after a checkpoint and letting only the
target card's packer start it again:

  1  checks: the task is a RUNNING formal training on another card; its pid runs the task's config; the target card's
     packer is alive and, with temporary limits max_jobs = target's GPU tasks + 1 and max_per_arm[arm] = target's
     trainings of the arm + 1, every other live card is blocked after the kill (by its job count or its arm count)
  2  wait until the newest loadable checkpoint is at most --max-loss-min old (else wait for the next one) and no save is
     in progress
  3  write those temporary limits to policy.json, so only the target card has room (recorded in policy["edits"])
  4  SIGKILL the training's process tree; wait until the source packer has reaped it (FAILED)
  5  under the table lock: the task -> PENDING (requeued: moved), dependants SKIPPED since the kill -> PENDING
  6  wait until the target card runs it, then restore max_jobs (recorded)

    python tools/real/move.py --task train:fme:3409 --to 0 [--max-loss-min 5] [--dry-run]
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

from tools.real import pack as P  # noqa: E402

LIVE_S = 60


def say(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def state():
    return json.loads(P.STATE.read_text())


def gpu_count(T, card):
    return sum(1 for t in T.values() if t["status"] == "RUNNING" and t.get("card") == card and t["gpu"])


def live_cards(st):
    return [c for c, i in st["cards"].items()
            if i.get("alive", True) and "exited" not in i and time.time() - i.get("beat", 0) < LIVE_S]


def tree(pid):
    kids = {}
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            kids.setdefault(ppid, []).append(int(d.name))
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        todo.extend(kids.get(p, []))
    return out


def cmdline(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return ""


def write_policy(pol):
    tmp = P.POLICY.with_suffix(".tmp")
    tmp.write_text(json.dumps(pol, indent=1) + "\n")
    os.replace(tmp, P.POLICY)


def newest(run):
    import torch
    best = None
    for p in run.glob("*.ckpt"):
        try:
            step = int(torch.load(str(p), map_location="cpu", weights_only=False).get("global_step", -1))
        except Exception:                                                  # noqa: BLE001
            continue
        if best is None or step > best[0]:
            best = (step, p)
    return best


def wait_checkpoint(run, max_loss_s):
    """Return (step, path) of a loadable checkpoint written at most max_loss_s ago, with no save in progress."""
    while True:
        files = list(run.glob("*.ckpt"))
        latest = max((f.stat().st_mtime for f in files), default=0.0)
        age = time.time() - latest
        if files and 15 <= age <= max_loss_s:
            sizes = {f: f.stat().st_size for f in files}
            time.sleep(3)
            if all(f.stat().st_size == s for f, s in sizes.items()):
                best = newest(run)
                if best is not None and time.time() - best[1].stat().st_mtime <= max_loss_s:
                    return best
        say(f"waiting for a fresh checkpoint in {run.name} (newest {age / 60:.1f} min old)")
        time.sleep(20)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--to", required=True)
    ap.add_argument("--max-loss-min", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    key, dst = a.task, str(a.to)

    st = state()
    T = st["tasks"]
    t = T[key]
    if t["kind"] != "train" or t["status"] != "RUNNING":
        raise SystemExit(f"{key}: not a RUNNING formal training ({t['kind']}, {t['status']})")
    src, pid = str(t["card"]), int(t["pid"])
    if src == dst:
        raise SystemExit(f"{key} already runs on card {dst}")
    cfg = t["cmd"][-1]
    if cfg not in cmdline(pid):
        raise SystemExit(f"pid {pid} does not run {cfg}")
    live = live_cards(st)
    if dst not in live or src not in live:
        raise SystemExit(f"cards live: {live}; need source {src} and target {dst}")
    arm = t["arm"]

    def arm_count(c):
        return sum(1 for u in T.values() if u["status"] == "RUNNING" and u.get("card") == c
                   and u["arm"] == arm and u["kind"] in ("smoke", "train"))
    # temporary limits: the target has room (cap jobs, arm_cap trainings of this arm); every other live card is
    # blocked after the kill by the job count or by the per-arm count
    cap, arm_cap = gpu_count(T, dst) + 1, arm_count(dst) + 1
    after = {c: (gpu_count(T, c) - (c == src), arm_count(c) - (c == src)) for c in live if c != dst}
    open_ = [c for c, (n, na) in after.items() if n < cap and na < arm_cap]
    if open_:
        raise SystemExit(f"cards {open_} would also have room (jobs, {arm} trainings after the kill: {after}; "
                         f"target {dst}: {gpu_count(T, dst)}, {arm_count(dst)})")
    run = Path(t["outputs"][0])
    say(f"{key}: card {src} -> card {dst}; pid {pid}; run {run.name}; (jobs, {arm}) after the kill {after}; "
        f"temporary max_jobs {cap}, max_per_arm.{arm} {arm_cap}")
    if a.dry_run:
        say("dry run: newest checkpoint", newest(run))
        return

    step, ckpt = wait_checkpoint(run, a.max_loss_min * 60)
    say(f"resume point {ckpt.name} (global_step {step}, {(time.time() - ckpt.stat().st_mtime) / 60:.1f} min old)")

    pol0 = json.loads(P.POLICY.read_text())
    orig = pol0.get("max_jobs", P.DEFAULT_POLICY["max_jobs"])
    orig_arm = dict(pol0.get("max_per_arm", {})).get(arm)
    pol = dict(pol0, max_jobs=cap, max_per_arm={**pol0.get("max_per_arm", {}), arm: arm_cap})
    pol.setdefault("edits", []).append({"at": P.now(), "change": f"max_jobs {orig} -> {cap}, max_per_arm.{arm} "
                                        f"{orig_arm} -> {arm_cap} (temporary)",
                                        "why": f"tools/real/move.py: only card {dst} may start {key}"})
    write_policy(pol)
    time.sleep(12)                                                        # every packer re-reads the policy

    procs = tree(pid)
    kill_at = P.now()
    for p in reversed(procs):
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
    say(f"SIGKILL {len(procs)} processes (training {pid} and its workers)")
    t0 = time.time()
    while state()["tasks"][key]["status"] == "RUNNING":
        if time.time() - t0 > 180:
            raise SystemExit("the source packer did not reap the training within 180 s; policy max_jobs left at "
                             f"{cap} -- fix by hand")
        time.sleep(1)

    with P.locked() as s:
        TT = s["tasks"]
        u = TT[key]
        if u["status"] != "FAILED":
            raise SystemExit(f"{key} is {u['status']} after the kill, expected FAILED")
        u.update(status="PENDING", card=None, pid=None)
        u.pop("rc", None)
        u.pop("ended", None)
        u.setdefault("requeued", []).append({"at": P.now(), "reason": f"moved card {src} -> card {dst} after "
                                             f"{ckpt.name} (tools/real/move.py, #real-amend3-20261111)"})
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
    say(f"{key} -> PENDING; dependants restored: {reset}")

    t0 = time.time()
    while True:
        u = state()["tasks"][key]
        if u["status"] == "RUNNING":
            break
        if time.time() - t0 > 300:
            say(f"warning: no card started {key} within 300 s")
            break
        time.sleep(2)
    pol = json.loads(P.POLICY.read_text())
    pol["max_jobs"] = orig
    if orig_arm is None:
        pol.get("max_per_arm", {}).pop(arm, None)
    else:
        pol.setdefault("max_per_arm", {})[arm] = orig_arm
    pol.setdefault("edits", []).append({"at": P.now(), "change": f"max_jobs {cap} -> {orig}, max_per_arm.{arm} "
                                        f"{arm_cap} -> {orig_arm} (restored)",
                                        "why": f"tools/real/move.py: {key} started on card {u.get('card')}"})
    write_policy(pol)
    say(f"{key}: {u['status']} on card {u.get('card')} (pid {u.get('pid')}); max_jobs {orig}, "
        f"max_per_arm.{arm} {orig_arm} restored")


if __name__ == "__main__":
    main()
