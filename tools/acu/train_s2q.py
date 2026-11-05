#!/usr/bin/env python3
"""ACU PR revision (docs/EXPERIMENT_RECORDS.md#acu-pr2-prereg-20261104): tools/acu/train_s2.py (frozen, unchanged) under the
namespace "q" (tools/acu/ns_q.py): PR2 routes the causal pipeline (deterministic window sums), configs from
configs/acu_20261104q/, runs and receipts under outputs/acu/20261104q/. Rank 0 also writes receipts/q__<config stem>.json.
Lightning's DDP re-runs this script in the second rank, so both ranks enter the same namespace.

    python tools/acu/train_s2q.py --config <registered q config>
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

from tools.acu import train_s2  # noqa: E402
from tools.acu.common import OUT, now, save, sha  # noqa: E402

CODE = ("tools/acu/train_s2q.py", "tools/acu/ns_q.py", "tools/acu/train_s2.py", "semkine/acu_pr2.py",
        "semkine/acu_pr_det.py")


def main():
    if os.environ.get("LOCAL_RANK") in (None, "0") and len(sys.argv) == 3 and sys.argv[1] == "--config":
        cfg = Path(sys.argv[2]).resolve()
        save(OUT / "receipts" / f"q__{cfg.stem}.json",
             {"namespace": ns_q.NS, "config": str(cfg), "config_sha256": sha(cfg),
              "code_sha256": {c: sha(ROOT / c) for c in CODE}, "started_utc": now()}, exclusive=True)
    train_s2.main()


if __name__ == "__main__":
    main()
