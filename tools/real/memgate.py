#!/usr/bin/env python3
"""Host-memory gate of the shared packer table (docs/EXPERIMENT_RECORDS.md#real-amend5-run-20261111).

The packers (tools/real/pack.py, frozen) check GPU memory only; host memory (503 GiB) is what limits how many trainings
the cards can take (an EvHandTrack training ~15 GB with 6 data workers, a CMP training ~15-28 GB, the two-card router
~40 GB with 6 workers per rank). This gate takes the held steps below one at a time, in this order, each once the host
has MemAvailable >= its threshold and memory pressure (PSI some avg60) < 30 %, and at least SPACING_S after the previous
step (so that the jobs it released have reached their memory):
  release <task> <hold>   remove the hold dependency from a pending task (ev:hold:mem, erh:hold:mem)
  router                  start the EvRealHands router chain (outputs/erh/20261111/pmr/router_chain3.sh)
  open <hold>             set an external hold task DONE (erh:hold:<seed>)
It never kills, edits a recipe, or touches a running task. State: outputs/real/20261110/pack/memgate.json; log on stdout.

    setsid nohup python tools/real/memgate.py > outputs/real/20261110/pack/memgate.log 2>&1 &
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

from tools.real import pack as P  # noqa: E402

#: thresholds = the released job's host memory + ~50 GiB margin. Since 2026-11-11 20:50 UTC the EvRealHands steps put
#: FastMETRO first (~19 h per seed, the long pole): opening erh:hold:<seed> starts only erh:train:fme:<seed>, every other
#: EvRealHands training (epf -> ehpe, the EvHandTrack d1r after the router, eho, epxf -> epx) also waits on erh:hold:mem,
#: removed here task by task (longest chains first). The first two steps ran under the earlier list. 21:05 UTC: epf
#: (-> ehpe, ~10 h) moved before the router (its chain router -> d1r -> rol5 -> reading is ~6 h); an EvRealHands CMP
#: training measured ~8 GiB (FastMETRO, 6 data workers), so its threshold is 60 GiB.
STEPS = (("release", "ev:train:d1m:3409", "ev:hold:mem", 85),
         ("release", "ev:train:d1g0:3409", "ev:hold:mem", 85),
         ("open", "erh:hold:3407", None, 80),
         ("open", "erh:hold:3408", None, 80),
         ("open", "erh:hold:3409", None, 80),
         *[("release", f"erh:train:epf:{s}", "erh:hold:mem", 60) for s in (3407, 3408, 3409)],
         ("router", None, None, 95),
         *[("release", f"erh:ev:train:d1r:{s}", "erh:hold:mem", 65) for s in (3407, 3408, 3409)],
         *[("release", f"erh:train:eho:{s}", "erh:hold:mem", 60) for s in (3407, 3408, 3409)],
         *[("release", f"erh:train:epxf:{s}", "erh:hold:mem", 60) for s in (3407, 3408, 3409)])
SPACING_S = 600
PSI_MAX = 30.0
STATE = P.PACK / "memgate.json"
CHAIN = ROOT / "outputs/erh/20261111/pmr/router_chain3.sh"


def say(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def host():
    avail = int([x for x in open("/proc/meminfo") if x.startswith("MemAvailable")][0].split()[1]) / 2 ** 20
    some = [x for x in open("/proc/pressure/memory") if x.startswith("some")][0]
    return avail, float(some.split("avg60=")[1].split()[0])


def do(kind, key, hold):
    if kind == "release":
        with P.locked() as st:
            t = st["tasks"][key]
            if hold in t["deps"] and t["status"] == "PENDING":
                t["deps"].remove(hold)
                t.setdefault("requeued", []).append({"at": P.now(), "reason": "memory hold released by tools/real/memgate.py"})
        return f"released {key}"
    if kind == "router":
        log = ROOT / "outputs/erh/20261111/pmr/router_chain3.log"
        subprocess.Popen(["setsid", "nohup", str(CHAIN)], cwd=str(ROOT), stdout=open(log, "a"), stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL)
        return "started the EvRealHands router chain"
    with P.locked() as st:
        t = st["tasks"][key]
        if t["status"] == "EXTERNAL":
            t.update(status="DONE", ended=P.now(), released_by="tools/real/memgate.py")
    return f"opened {key}"


def main():
    s = json.loads(STATE.read_text()) if STATE.exists() else {"done": [], "last": 0.0}
    say("memgate start; done so far", s["done"])
    while len(s["done"]) < len(STEPS):
        kind, key, hold, need = STEPS[len(s["done"])]
        avail, psi = host()
        if avail >= need and psi < PSI_MAX and time.time() - s["last"] >= SPACING_S:
            msg = do(kind, key, hold)
            s["done"].append({"step": [kind, key, hold, need], "at": P.now(), "avail_gib": round(avail, 1), "psi60": psi})
            s["last"] = time.time()
            tmp = STATE.with_suffix(".tmp")
            tmp.write_text(json.dumps(s, indent=1) + "\n")
            os.replace(tmp, STATE)
            say(msg, f"(MemAvailable {avail:.0f} GiB, PSI avg60 {psi:.1f})")
        time.sleep(60)
    say("memgate: all steps done")


if __name__ == "__main__":
    main()
