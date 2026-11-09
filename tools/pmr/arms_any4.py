"""PMR registry union 4 (docs/EXPERIMENT_RECORDS.md#pmr-s9-prereg-20261109): tools/pmr/arms_any3.py (frozen with S8) and the
D2 registry tools/pmr/arms9.py appended to arms_any2's registry tuple at import, so the S5 rollout registry and its entries
also take d1hubfull as a base arm. Import it before those tools are used (one network per process).
"""
from __future__ import annotations

from tools.pmr import arms_any3  # noqa: F401
from tools.pmr import arms_any2 as U

EXTRA = ("tools.pmr.arms9",)
if not set(EXTRA) <= set(U.REGISTRIES):
    U.REGISTRIES = U.REGISTRIES + EXTRA

__all__ = ["EXTRA"]
