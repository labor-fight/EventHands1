"""PMR registry union 5 (docs/EXPERIMENT_RECORDS.md#rdr-trunk-prereg-20261110): tools/pmr/arms_any4.py plus registry 10
(D1 R with an RDR router), appended to arms_any2's registry tuple at import."""
from __future__ import annotations

from tools.pmr import arms_any4  # noqa: F401
from tools.pmr import arms_any2 as U

EXTRA = ("tools.pmr.arms10",)
if not set(EXTRA) <= set(U.REGISTRIES):
    U.REGISTRIES = U.REGISTRIES + EXTRA

__all__ = ["EXTRA"]
