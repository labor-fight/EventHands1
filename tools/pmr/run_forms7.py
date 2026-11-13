#!/usr/bin/env python3
"""ABL descriptive reading (docs/EXPERIMENT_RECORDS.md#abl-results-20261111, written after the step-1 rollout collapse): a
from-scratch base arm (6000 steps, no rollout fine-tune) read in the streaming form, so step 1 can be compared before the
fine-tune in the same form as the ablation table. tools/pmr/run.py's `arm_spec` / `prepare` (union registry 6, RDR enabled,
the loader taking arms without a router, exactly as tools/pmr/run11.py) and tools/pmr/run_forms.py's streaming install
(semkine/tpd_str.py), both unchanged.

    python -m tools.pmr.run_forms7 read --arm d1r|abgeo --seed S --task e1     tag pmr_<arm>_s<seed>__str
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms_any6  # noqa: E402,F401
from semkine import rdr_hooks as RH  # noqa: E402
from tools.pmr import ablrt as RT  # noqa: E402
from tools.pmr import arms_any2 as U  # noqa: E402
from tools.pmr import run as R  # noqa: E402
from tools.pmr.common import OUT, now, require, save  # noqa: E402

RH.enable()
RT.patch_loader()
R.A = U
ARMS = ("d1r", "abgeo")


def reading(arm, seed, task):
    require(arm in ARMS, "the two step-1 bases only")
    from tools.pfo import engine as E
    from semkine import tpd_str as TF
    tag, sp = R.arm_spec(arm, seed)
    R.prepare(sp)
    R.deployed_router_for_mixes()
    TF.install()
    TF.engine_hooks()
    E.OUT = OUT
    out = E.run(f"{tag}__str", task, None, None, (), str(sp))
    save(Path(out) / "form.json", {"form": "str", "written_utc": now()}, exclusive=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("read")
    s.add_argument("--arm", required=True, choices=ARMS)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--task", required=True, choices=("e1",))
    a = p.parse_args(argv)
    reading(a.arm, a.seed, a.task)


if __name__ == "__main__":
    main()
