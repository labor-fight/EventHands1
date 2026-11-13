#!/usr/bin/env python3
"""Run a real-study tool in the EvRealHands namespace (tools/erh/ns.py, docs/EXPERIMENT_RECORDS.md#erh-prereg-20261111).

    python tools/erh/run.py tools.real.train --config configs/erh_20261111/cmp_<arm>_s<seed>[_smoke].yaml
    python tools/erh/run.py tools.real.run e1 --arm A --seed S
    python tools/erh/run.py tools.real.evht_train --config configs/erh_20261111/pmr/pmr_<arm>_s<seed>.yaml
    python tools/erh/run.py tools.real.evht read --arm A --seed S
The module runs as __main__ with argv = [its file, the remaining arguments]; every trainer here uses one card, so no
DDP relaunch reads this argv, except tools.real.evht_router (two cards): Lightning relaunches rank 1 with
`sys.executable` + this process's argv, i.e. this wrapper with the same arguments, which installs the namespace again.
"""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Fix 2026-11-12 (docs/EXPERIMENT_RECORDS.md#real-amend5-run-20261111): PFO's frozen engine and FastMETRO's reading
#: call tools/xcl/evaluate.py `configure_environment`, which re-executes `sys.executable *sys.argv` with POLICY_ENV when
#: PYTHONHASHSEED is not 3407. Below, sys.argv names the real-study module's own file, so that re-exec ran
#: tools/real/run.py outside this namespace (erh:e1:epx:3407 tried the real study's reading directory and stopped at the
#: engine's "preserve existing reading" check; nothing was overwritten). A reading re-executes this wrapper with the
#: same POLICY_ENV first, so the engine's check finds PYTHONHASHSEED = 3407 and does not re-execute.
READINGS = {("tools.real.run", "e1"), ("tools.real.evht", "read")}
if len(sys.argv) > 2 and (sys.argv[1], sys.argv[2]) in READINGS and os.environ.get("PYTHONHASHSEED") != "3407":
    from tools.xcl.evaluate import POLICY_ENV
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, **POLICY_ENV})

from tools.erh import ns  # noqa: E402

ns.install()

if __name__ == "__main__":
    module = sys.argv[1]
    spec_file = Path(ROOT / (module.replace(".", "/") + ".py"))
    if not spec_file.exists():
        raise SystemExit("no such module file: " + str(spec_file))
    os.environ["ERH_NAMESPACE"] = "1"
    sys.argv = [str(spec_file)] + sys.argv[2:]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
