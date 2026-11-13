"""PMR registry union 6 (docs/EXPERIMENT_RECORDS.md#abl-prereg-20261111): union 5 (registries 1-10) plus registry 11 (the
ABL arms), appended to arms_any2's registry tuple at import, and the rollout entry rol1 (K = 1: the same fine-tune with
single-step updates, TPD's control) added to tools/pmr/arms_rol2.py's table. Import it before those tools are used."""
from __future__ import annotations

from tools.pmr import arms_any5  # noqa: F401
from tools.pmr import arms_any2 as U
from tools.pmr import arms_rol2 as R2

EXTRA = ("tools.pmr.arms11",)
if not set(EXTRA) <= set(U.REGISTRIES):
    U.REGISTRIES = U.REGISTRIES + EXTRA
R2.ROL.setdefault("rol1", {"k": 1, "grad": False})

__all__ = ["EXTRA"]
