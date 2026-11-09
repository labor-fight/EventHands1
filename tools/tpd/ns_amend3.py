"""TPD engineering amendment 3 (docs/EXPERIMENT_RECORDS.md#tpd-amend3-20261105): the K >= 5 rollout trainings ran /dev/shm
out (two of them prefetching ~2.5 GB / 0.8 GB batches through 12-16 workers x 2 ranks x prefetch 2); their reruns use
TRAIN.PREFETCH_FACTOR 1. Every sample's augmentation is a function of (seed, index) and the sampler's order does not
depend on the loader's resources, so the data and numerics are unchanged.

`enter()` (before any other tools.tpd import): seed 3409 registered (tools/tpd/ns_seed.py), the ROL and synthesis
registries write their configs under configs/tpd_20261105/amend3/ (frozen configs untouched) with PREFETCH_FACTOR 1 for
K >= 5 and PFO.AMENDMENT = tpd-amend3-20261105; run names and output directories stay the registered ones (the failed
attempts are archived under outputs/tpd/20261105/formal_failed/)."""
from __future__ import annotations

AMENDMENT = "tpd-amend3-20261105"
_STATE = {"on": False}


def enter():
    if _STATE["on"]:
        return
    from tools.tpd import ns_seed
    ns_seed.enter()
    from tools.tpd import arms_final as F
    from tools.tpd import arms_rol as A
    from tools.tpd.common import CONFIGS
    for mod in (A, F):
        native = mod.build_config

        def build(arm, seed, phase="formal", _native=native):
            cfg = _native(arm, seed, phase)
            if int(cfg["ROL"]["K"]) >= 5:
                cfg["TRAIN"]["PREFETCH_FACTOR"] = 1
                cfg["PFO"]["AMENDMENT"] = AMENDMENT
            return cfg
        mod.build_config = build
        mod.CONFIGS = CONFIGS / "amend3"
    _STATE["on"] = True
