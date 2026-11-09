#!/usr/bin/env python3
"""PMR stage S3c asynchronous checks (docs/EXPERIMENT_RECORDS.md#pmr-s3c-prereg-20261109; report only): TPD's frozen checks
(tools/tpd/asynccheck.py: prefix + order on the ordinary or the causal form; tools/tpd/asynccheck_str.py: prefix + order
+ partition on the streaming form) run on a PMR network, with TPD's trunk preparation replaced by the PMR one
(tools/pmr/run.py `prepare`, the trunk the spec names). CPU; receipts under outputs/pmr/20261109/receipts/.

    python -m tools.pmr.asynccheck --form plain|causal|str --arm A_rol5 --seed S
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr.common import OUT  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--form", required=True, choices=("plain", "causal", "str"))
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args(argv)
    from tools.pmr import run as R
    from tools.pmr import run_rol2 as RR
    from tools.tpd import run_rol as TR
    tag, sp = RR.arm_spec(a.arm, a.seed)
    TR.prepare = R.prepare                                          # the PMR trunk instead of B2's gshare trunk
    name = f"{tag}__{a.form}"
    if a.form == "str":
        from tools.tpd import asynccheck_str as C
        return C.main(["--name", name, "--spec", str(sp), "--out-root", str(OUT)])
    from tools.tpd import asynccheck as C
    args = ["--name", name, "--spec", str(sp), "--out-root", str(OUT)]
    return C.main(args + (["--causal"] if a.form == "causal" else []))


if __name__ == "__main__":
    main()
