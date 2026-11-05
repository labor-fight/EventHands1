#!/usr/bin/env python3
"""ACU PR revision (docs/EXPERIMENT_RECORDS.md#acu-pr2-prereg-20261104): tools/acu/router_pr.py (frozen, unchanged: r_s2d's
recipe, seed, steps, deterministic Trainer) training PR2 (semkine/acu_pr2.py) with PR's deterministic window sums, under
the namespace "q" (tools/acu/ns_q.py): the router goes to outputs/acu/20261104q/routers/pr/. Lightning's DDP re-runs this
script in the second rank, so both ranks train the same network.

    python tools/acu/router_pr2.py --phase preflight|formal
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.acu import ns_q  # noqa: E402

ns_q.enter()

from tools.acu import router_pr as R  # noqa: E402
from tools.acu.common import OUT, now, save, sha  # noqa: E402

CODE = ("tools/acu/router_pr2.py", "tools/acu/ns_q.py", "tools/acu/router_pr.py", "semkine/acu_pr2.py",
        "semkine/acu_pr_det.py", "semkine/acu_pr.py", "semkine/acu_hist.py")


def main():
    phase = sys.argv[sys.argv.index("--phase") + 1] if "--phase" in sys.argv else None
    if os.environ.get("LOCAL_RANK") in (None, "0") and phase:
        save(OUT / "receipts" / f"pr2_launch_{phase}.json",
             {"namespace": ns_q.NS, "network": "semkine.acu_pr2.PR2Net", "router_dir": str(R.router_dir(phase)),
              "code_sha256": {c: sha(ROOT / c) for c in CODE}, "written_utc": now()}, exclusive=True)
    R.main()


if __name__ == "__main__":
    main()
