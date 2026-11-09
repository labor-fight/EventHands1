#!/usr/bin/env python3
"""TPD stage ROL-c: tools/tpd/train_rol.py unchanged, under tools/tpd/ns_seed.py (seed 3409 registered)
(docs/EXPERIMENT_RECORDS.md#tpd-rolc-prereg-20261105).

    python tools/tpd/train_rolc.py --config configs/tpd_20261105/tpd_rol5_s3409.yaml
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.tpd import ns_seed  # noqa: E402

ns_seed.enter()
from tools.tpd import train_rol  # noqa: E402

if __name__ == "__main__":
    train_rol.main()
