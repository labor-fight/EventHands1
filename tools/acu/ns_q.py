"""ACU PR revision namespace "q" (docs/EXPERIMENT_RECORDS.md#acu-pr2-prereg-20261104): the registered s1c / s2 / s3 tools
(frozen, imported unchanged) re-run with PR2 (semkine/acu_pr2.py) in place of PR, under their own roots
outputs/acu/20261104q and configs/acu_20261104q (siblings of the study's roots: tools/acu/jobs_s3.py locates PFO2 / PFO
readings from OUT.parent.parent).

`enter()` must run before any tools.acu module other than tools.acu.common is imported (every frozen tool binds
tools.acu.common.OUT / CONFIGS at import): it rebinds the two roots, installs PR's deterministic window sums
(semkine/acu_pr_det.py) and PR2 (semkine.acu_pr.build / load / save_checkpoint / n_params; load reads PR and PR2 kinds).
"""
from __future__ import annotations

import sys

NS = "q"
#: the namespace itself and the modules that enter it before importing anything else from tools.acu
_SELF = ("tools.acu", "tools.acu.common", "tools.acu.ns_q", "tools.acu.jobs_q", "tools.acu.jobs_s3q", "tools.acu.qrun",
         "tools.acu.router_pr2", "tools.acu.train_s2q")


def enter() -> str:
    import tools.acu.common as C
    if getattr(C, "ACU_NAMESPACE", None) == NS:
        return NS
    early = sorted(m for m in sys.modules if m.startswith("tools.acu.") and m not in _SELF)
    C.require(not early, "tools.acu modules imported before the namespace: " + ", ".join(early))
    C.ACU_BASE_OUT, C.ACU_BASE_CONFIGS = C.OUT, C.CONFIGS
    C.OUT, C.CONFIGS = C.OUT.with_name(C.OUT.name + NS), C.CONFIGS.with_name(C.CONFIGS.name + NS)
    C.ACU_NAMESPACE = NS
    from semkine import acu_pr2 as P2
    from semkine import acu_pr_det as PD
    PD.install()
    P2.install()
    return NS


def base_out():
    """The study's own root (outputs/acu/20261104), also inside the namespace."""
    import tools.acu.common as C
    return getattr(C, "ACU_BASE_OUT", C.OUT)
