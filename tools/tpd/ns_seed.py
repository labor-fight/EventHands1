"""TPD namespace for seed 3409 of the ROL registry (docs/EXPERIMENT_RECORDS.md#tpd-rolc-prereg-20261105): the frozen ROL
tools (tools/tpd/arms_rol.py, train_rol.py, run_rol.py) read their seed list from tools/tpd/common.py at import; `enter()`
extends it to (3407, 3408, 3409) before any of them is imported, so they run seed 3409 unchanged (B2 3409 = PFO2's
accepted pfo2_fgs_s3409)."""
from __future__ import annotations

import sys


def enter():
    for m in ("tools.tpd.arms_rol", "tools.tpd.train_rol", "tools.tpd.run_rol", "tools.tpd.analyze_rol"):
        if m in sys.modules:
            raise RuntimeError("ns_seed.enter() must run before " + m + " is imported")
    from tools.tpd import common as C
    C.SEEDS = (3407, 3408, 3409)
    return C.SEEDS
