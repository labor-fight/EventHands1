#!/usr/bin/env python3
"""PFO router training with registered config overrides (docs/EXPERIMENT_RECORDS.md#pfo-p3-prereg-20261103): tools/pfo/
router_train.py's procedure, unchanged, for any router of tools/pfo/routers3.py, with the router spec's optional "set"
({dotted config key: value}) applied to the data config after `tools/s45sm/router_train.py` `_data` builds it (e.g. the
hot-pixel rate). Lightning relaunches rank 1 with this argv, so both ranks apply the same overrides.

    python tools/pfo/router_train3.py --router NAME --phase preflight|formal
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.pfo import router_train as RT  # noqa: E402
from tools.pfo import routers3 as R3  # noqa: E402

RT.ROUTERS.clear()
RT.ROUTERS.update(R3.ROUTERS3)


def _patch_data(router):
    from tools.s45sm import router_train as S
    native = S._data
    overrides = R3.ROUTERS3[router].get("set", {})

    def data(spec):
        cfg = native(spec)
        for dotted, value in overrides.items():
            node = cfg
            keys = dotted.split(".")
            for k in keys[:-1]:
                node = node[k]
            node[keys[-1]] = value
        return cfg
    S._data = data


if __name__ == "__main__":
    name = sys.argv[sys.argv.index("--router") + 1]
    _patch_data(name)
    RT.main()
