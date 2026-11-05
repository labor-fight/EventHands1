#!/usr/bin/env python3
"""ACU engineering amendment 2 (docs/EXPERIMENT_RECORDS.md#acu-amend2-20261104): tools/acu/train_s2.py with
semkine/acu_pr_det.py installed (the PR routes of the causal pipeline use exact integer window sums, deterministic on
CUDA). Usage, checks and outputs are train_s2.py's; rank 0 also writes receipts/det__<config stem>.json (the
amendment's code identity). Lightning's DDP re-runs this script in the second rank, so both ranks install the same sums.

    python tools/acu/train_s2d.py --config <registered config>
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
from tools.acu import train_s2  # noqa: E402
from tools.acu.common import OUT, now, save, sha  # noqa: E402


def main():
    acu_pr_det.install()
    if os.environ.get("LOCAL_RANK") in (None, "0") and len(sys.argv) == 3 and sys.argv[1] == "--config":
        cfg = Path(sys.argv[2]).resolve()
        save(OUT / "receipts" / f"det__{cfg.stem}.json",
             {"config": str(cfg), "config_sha256": sha(cfg), "wrapper_sha256": sha(__file__),
              "det_sha256": sha(ROOT / "semkine/acu_pr_det.py"), "train_s2_sha256": sha(ROOT / "tools/acu/train_s2.py"),
              "fix_bits": acu_pr_det.FIX_BITS, "started_utc": now()}, exclusive=True)
    train_s2.main()


if __name__ == "__main__":
    main()
