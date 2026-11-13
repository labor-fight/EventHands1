#!/usr/bin/env python3
"""Late pair lanes for the route-first unification queue (docs/EXPERIMENT_RECORDS.md#rfsu-20261113).

The simulated units need two cards and are the longest part of the queue; the one-card units (EvRealHands, real ablations)
drain earlier, and their lanes then exit while two-card units may still be unclaimed. This watcher starts a two-card lane on
two cards freed that way, so they take the remaining simulated units instead of sitting idle. It never touches a running lane.

  - watched one-card lanes: g6, g7 (direct cards) and g2, g3 (borrowed); a lane has exited when no process runs
    `tools/rfsu/lane.py run --lane <name>` any more;
  - two freed direct cards: `CUDA_VISIBLE_DEVICES=a,b lane.py run --lane p<ab> --wait-free` in a new session;
    a pair with a borrowed card (0-3) goes whole through gpu-borrow (user unit rfsu-lane-p<ab>, `ulimit -n 65536`);
  - exits as soon as no two-card unit is unclaimed.

    setsid nohup python tools/rfsu/pairup.py > outputs/rfsu/20261113/logs/pairup.out 2>&1 &
Log: outputs/rfsu/20261113/logs/pairup.log
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.rfsu import lane as L  # noqa: E402

WATCH = {"g6": 6, "g7": 7, "g2": 2, "g3": 3}
BORROWED = {0, 1, 2, 3}
GPU_BORROW = Path.home() / ".local/bin/gpu-borrow"
LOG = L.LOGS / "pairup.log"


def say(msg):
    L.LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"{L.now()} {msg}\n")


def lane_running(name):
    """A process whose argv is `... tools/rfsu/lane.py run ... --lane <name> ...`."""
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        args = [a.decode(errors="replace") for a in argv if a]
        if any(a.endswith("tools/rfsu/lane.py") for a in args) and "run" in args:
            for i, a in enumerate(args[:-1]):
                if a == "--lane" and args[i + 1] == name:
                    return True
    return False


def busy(card):
    out = subprocess.run(["nvidia-smi", "-i", str(card), "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    return bool(out.strip())


def unclaimed_two_card():
    with L.Locked():
        st = L.load_state()
        return [u["id"] for u in L.queue() if u["gpus"] == 2 and u["id"] not in st]


def launch(a, b):
    name = f"p{a}{b}"
    if a in BORROWED or b in BORROWED:
        cmd = ["systemd-run", "--user", f"--unit=rfsu-lane-{name}", "--collect", f"--working-directory={ROOT}",
               "-p", "KillMode=control-group", str(GPU_BORROW), "run", "--gpus", f"{a},{b}", "--", "bash", "-c",
               f"ulimit -n 65536; exec {L.PY} tools/rfsu/lane.py run --lane {name}"]
        env = {**os.environ}
        env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
        say(f"launched {name} through gpu-borrow: exit {r.returncode} {r.stdout.strip()} {r.stderr.strip()}")
        return r.returncode == 0
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": f"{a},{b}"}
    out = open(L.LOGS / f"lane_{name}.out", "a")
    p = subprocess.Popen([L.PY, "tools/rfsu/lane.py", "run", "--lane", name, "--wait-free"], cwd=str(ROOT), env=env,
                         stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    say(f"launched {name} on direct cards {a},{b}: pid {p.pid}")
    return True


def main():
    used = set()
    say(f"start pid {os.getpid()} watching {WATCH}")
    while True:
        left = unclaimed_two_card()
        if not left:
            say("no two-card unit unclaimed; exit")
            return
        free = [c for name, c in WATCH.items()
                if c not in used and not lane_running(name) and (c in BORROWED or not busy(c))]
        free.sort(key=lambda c: (c in BORROWED, c))          # direct pairs first, then mixed, then borrowed
        if len(free) >= 2:
            a, b = sorted(free[:2])
            say(f"cards {a},{b} freed with {len(left)} two-card units unclaimed: {left}")
            if launch(a, b):
                used |= {a, b}
                continue
        time.sleep(120)


if __name__ == "__main__":
    main()
