#!/usr/bin/env python3
"""Restart order after the 2026-11-12 host crash (docs/EXPERIMENT_RECORDS.md#real-amend5-run-20261111).

The host stopped at ~01:13 UTC (no clean shutdown; rebooted 01:49 UTC) and every task of the shared packer table died.
On restart the real-data ablation's remaining trainings (seven rol5 fine-tunes, fresh-only, deterministic) and
FastMETRO 3408 go first, beside EvRealHands' three FastMETRO trainings (resumed; EvRealHands' long pole, ~8 h alone).
Every other EvRealHands training waits on the external task erh:hold:mem again (EvHandTrack d1r, EventHPE, EventHands,
EvHandPose pose stage: all far shorter than FastMETRO, so holding them does not move the EvRealHands finish).

This gate releases them one by one, in ORDER, every SPACING_S, once at most REAL_LEFT real-data trainings
(`ev:train:` / `train:` keys) are still pending or running, or at the latest at DEADLINE_UTC; each release also needs
MemAvailable >= MEM_GIB and memory pressure (PSI some avg60) < PSI_MAX. It never starts, stops or edits anything else.
State: outputs/real/20261110/pack/holdgate.json; log on stdout.

    setsid nohup python tools/real/holdgate.py > outputs/real/20261110/pack/holdgate.log 2>&1 &
"""
from __future__ import annotations

import calendar
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

from tools.real import pack as P  # noqa: E402

HOLD = "erh:hold:mem"
SEEDS = (3407, 3408, 3409)
ORDER = ([f"erh:ev:train:d1r:{s}" for s in SEEDS] + [f"erh:train:ehpe:{s}" for s in SEEDS]
         + [f"erh:train:eho:{s}" for s in SEEDS] + [f"erh:train:epx:{s}" for s in SEEDS])
REAL_LEFT = 2
DEADLINE_UTC = "2026-11-12T04:30:00Z"
SPACING_S = 120
MEM_GIB = 60.0
PSI_MAX = 30.0
STATE = P.PACK / "holdgate.json"


def say(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def host():
    avail = int([x for x in open("/proc/meminfo") if x.startswith("MemAvailable")][0].split()[1]) / 2 ** 20
    some = [x for x in open("/proc/pressure/memory") if x.startswith("some")][0]
    return avail, float(some.split("avg60=")[1].split()[0])


def hold():
    """Put erh:hold:mem back on every not-yet-released task of ORDER (idempotent)."""
    with P.locked() as st:
        T = st["tasks"]
        if T[HOLD]["status"] != "EXTERNAL":
            raise SystemExit(f"{HOLD} is {T[HOLD]['status']}, expected EXTERNAL")
        done = []
        for key in ORDER:
            t = T[key]
            if t["status"] in ("PENDING", "RUNNING") and HOLD not in t["deps"]:
                t["deps"].append(HOLD)
                t.setdefault("requeued", []).append({"at": P.now(), "reason": "held again after the 2026-11-12 "
                                                     "host crash: real-data rol5 fine-tunes first (tools/real/holdgate.py)"})
                done.append(key)
    return done


def real_left(T):
    return sum(1 for k, t in T.items() if (k.startswith("ev:train:") or k.startswith("train:"))
               and t["status"] in ("PENDING", "RUNNING"))


def main():
    s = json.loads(STATE.read_text()) if STATE.exists() else {"held": [], "released": [], "last": 0.0}
    if not s["held"]:
        s["held"] = hold()
        STATE.write_text(json.dumps(s, indent=1) + "\n")
        say("held again:", s["held"])
    deadline = calendar.timegm(time.strptime(DEADLINE_UTC, "%Y-%m-%dT%H:%M:%SZ"))
    while len(s["released"]) < len(s["held"]):
        T = json.loads(P.STATE.read_text())["tasks"]
        left = real_left(T)
        avail, psi = host()
        if (left <= REAL_LEFT or time.time() >= deadline) and avail >= MEM_GIB and psi < PSI_MAX \
                and time.time() - s["last"] >= SPACING_S:
            key = [k for k in s["held"] if k not in s["released"]][0]
            with P.locked() as st:
                t = st["tasks"][key]
                if HOLD in t["deps"]:
                    t["deps"].remove(HOLD)
                    t.setdefault("requeued", []).append({"at": P.now(), "reason": "released by tools/real/holdgate.py"})
            s["released"].append(key)
            s["last"] = time.time()
            tmp = STATE.with_suffix(".tmp")
            tmp.write_text(json.dumps(s, indent=1) + "\n")
            os.replace(tmp, STATE)
            say(f"released {key} (real-data trainings left {left}, MemAvailable {avail:.0f} GiB, PSI avg60 {psi:.1f})")
        time.sleep(20)
    say("holdgate: all released")


if __name__ == "__main__":
    main()
