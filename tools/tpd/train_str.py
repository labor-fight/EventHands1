#!/usr/bin/env python3
"""TPD stage STR training (docs/EXPERIMENT_RECORDS.md#tpd-str-prereg-20261106): tools/tpd/train_final.py unchanged,
inside tools/tpd/ns_str.py (arm str registered; streaming datasets, collate and model). Lightning relaunches rank 1 with
`train_str.py --config <path>`, the form this entry takes.

    python tools/tpd/train_str.py --config configs/tpd_20261105/str/tpd_str_s<seed>[_preflight].yaml
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.tpd import ns_str  # noqa: E402

ns_str.enter()
from tools.tpd import train_final  # noqa: E402

train_final.CODE = train_final.CODE + ("semkine/tpd_str.py", "tools/tpd/ns_str.py", "tools/tpd/ns_rolk.py",
                                       "tools/tpd/train_str.py")

if __name__ == "__main__":
    train_final.main()
