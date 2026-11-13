#!/usr/bin/env python3
"""Insert the NOPART units into RFSU's lane queue (docs/EXPERIMENT_RECORDS.md#nopart-prereg-20261113), once, under the lanes'
own lock: the external unit router:simnp (marked by tools/nopart/sim_ext.py) and the two-card units sim:rfsrnp:<seed>,
placed right after sim:rfsr:3409 (user 2026-11-13: after the simulated main seeds, before the other simulated ablations).
The queue as it was is copied to outputs/nopart/20261113/queue_before_insert.json.

    python tools/nopart/queue_add.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
NOTE = "#nopart-prereg-20261113 (session eventhands1-a1)"
UNITS = [{"id": "router:simnp", "gpus": 0, "owner": "sim:rfsrnp:3407",
          "note": "r_s2dx2np, trained inside sim:rfsrnp:3407's train stage; " + NOTE}]
for seed in (3407, 3408, 3409):
    UNITS.append({"id": f"sim:rfsrnp:{seed}", "ns": "sim", "base": "rfsrnp", "rol": "rfsrnp_rol5", "seed": seed, "gpus": 2,
                  "after": ["configs:sim"] + ([] if seed == 3407 else ["router:simnp"]), "note": NOTE})
ANCHOR = "sim:rfsr:3409"
BACKUP = ROOT / "outputs/nopart/20261113/queue_before_insert.json"


def main():
    from tools.rfsu import lane as L
    with L.Locked():
        q = L.queue()
        ids = [u["id"] for u in q]
        if any(u["id"] in ids for u in UNITS):
            raise SystemExit("already inserted")
        i = ids.index(ANCHOR) + 1
        if BACKUP.exists():
            raise SystemExit("backup exists: " + str(BACKUP))
        BACKUP.parent.mkdir(parents=True, exist_ok=True)
        BACKUP.write_text(L.QUEUE.read_text())
        new = q[:i] + UNITS + q[i:]
        tmp = L.QUEUE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(new, indent=1) + "\n")
        os.replace(tmp, L.QUEUE)
    print("inserted", [u["id"] for u in UNITS], "after", ANCHOR, "; queue", len(q), "->", len(new))


if __name__ == "__main__":
    main()
