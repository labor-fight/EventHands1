"""PMR registry union 3 (docs/EXPERIMENT_RECORDS.md#pmr-s8-prereg-20261109): tools/pmr/arms_any2.py (frozen) with the S6
and S7 registries (tools/pmr/arms7.py, arms8.py) appended to its registry tuple at import, so every tool built on arms_any2
(the S5 rollout registry tools/pmr/arms_rol2.py and its entries, tools/pmr/cost4.py, tools/pmr/analyze7.py) also takes
d1hub, d1ghandhp and d1ghandhub as base arms. Import it before those tools are used (one network per process).
"""
from __future__ import annotations

from tools.pmr import arms_any2 as U

EXTRA = ("tools.pmr.arms7", "tools.pmr.arms8")
if not set(EXTRA) <= set(U.REGISTRIES):
    U.REGISTRIES = U.REGISTRIES + EXTRA

__all__ = ["EXTRA"]
