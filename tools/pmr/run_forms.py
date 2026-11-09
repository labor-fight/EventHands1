#!/usr/bin/env python3
"""PMR stage S3c input-form readings (docs/EXPERIMENT_RECORDS.md#pmr-s3c-prereg-20261109; report only): a PMR rollout
arm's accepted checkpoint read, with zero training, in TPD's causal (semkine/tpd_causal.py) or streaming
(semkine/tpd_str.py) input form instead of the ordinary packet form. Both replace only the node sampler and the event
tokens before the trunk, so they apply to the PMR trunk unchanged; the router r_s2d stays per packet (exempt,
#tpd-prereg-20261105).

    python -m tools.pmr.run_forms read --arm A_rol5 --seed S --form causal|str --task e1|e3
        tag pmr_<arm>_s<seed>__causal / __str under outputs/pmr/20261109/readings
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr.common import OUT, now, save  # noqa: E402

FORMS = ("causal", "str")


def reading(arm, seed, form, task):
    from tools.pfo import engine as E
    from tools.pmr import run as R
    from tools.pmr import run_rol2 as RR
    tag, sp = RR.arm_spec(arm, seed)
    if form == "causal":
        from semkine import tpd_causal as TF
    else:
        from semkine import tpd_str as TF
    R.prepare(sp)
    R.deployed_router_for_mixes()
    TF.install()
    TF.engine_hooks()
    E.OUT = OUT
    out = E.run(f"{tag}__{form}", task, None, None, (), str(sp))
    save(Path(out) / "form.json", {"form": form, "written_utc": now()}, exclusive=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("read")
    s.add_argument("--arm", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--form", required=True, choices=FORMS)
    s.add_argument("--task", required=True, choices=("e1", "e3"))
    a = p.parse_args(argv)
    reading(a.arm, a.seed, a.form, a.task)


if __name__ == "__main__":
    main()
