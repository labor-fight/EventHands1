#!/usr/bin/env python3
"""ACU engineering amendment 2 (docs/EXPERIMENT_RECORDS.md#acu-amend2-20261104): tools/acu/router_pr.py with
semkine/acu_pr_det.py installed (PR's window sums as exact integer prefix sums, deterministic on CUDA). Usage, recipe and
outputs are router_pr.py's; rank 0 also writes receipts/pr_det_<phase>.json (the amendment's code identity). Lightning's
DDP re-runs this script in the second rank, so both ranks install the same sums.

    python tools/acu/router_prd.py --phase preflight|formal
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

from semkine import acu_pr_det  # noqa: E402
from tools.acu import router_pr  # noqa: E402
from tools.acu.common import OUT, now, save, sha  # noqa: E402


def main():
    acu_pr_det.install()
    phase = sys.argv[sys.argv.index("--phase") + 1] if "--phase" in sys.argv else None
    if os.environ.get("LOCAL_RANK") in (None, "0") and phase in router_pr.STEPS:
        save(OUT / "receipts" / f"pr_det_{phase}.json",
             {"phase": phase, "wrapper_sha256": sha(__file__), "det_sha256": sha(ROOT / "semkine/acu_pr_det.py"),
              "router_pr_sha256": sha(ROOT / "tools/acu/router_pr.py"), "fix_bits": acu_pr_det.FIX_BITS,
              "started_utc": now()}, exclusive=True)
    router_pr.main()


if __name__ == "__main__":
    main()
