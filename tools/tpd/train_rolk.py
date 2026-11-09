#!/usr/bin/env python3
"""TPD stage ROL-k training (docs/EXPERIMENT_RECORDS.md#tpd-rolk-prereg-20261106): tools/tpd/train_final.py unchanged,
inside tools/tpd/ns_rolk.py (arm rolk registered; the model with ROL's floor). Lightning relaunches rank 1 with
`train_rolk.py --config <path>`, the form this entry takes.

    python tools/tpd/train_rolk.py --config configs/tpd_20261105/rolk/tpd_rolk_s<seed>[_preflight].yaml
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.tpd import ns_rolk  # noqa: E402

ns_rolk.enter()
from tools.tpd import train_final  # noqa: E402

if __name__ == "__main__":
    train_final.main()
