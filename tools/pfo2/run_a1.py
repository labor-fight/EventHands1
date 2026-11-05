#!/usr/bin/env python3
"""PFO2 acceptance of arms trained by tools/pfo2/train_det.py (docs/EXPERIMENT_RECORDS.md#pfo2-amend1-20261103):
the run must carry its deterministic-policy record (pfo2_det.json: Trainer deterministic, torch deterministic algorithms
enabled at fit start, the entry's SHA) before tools/pfo2/run.py's acceptance (unchanged) is written.

    python -m tools.pfo2.run_a1 accept --arm A --seed S
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo2 import arms as A  # noqa: E402
from tools.pfo2 import run as R  # noqa: E402
from tools.pfo2.common import read, require, sha  # noqa: E402


def deterministic_record(arm, seed, phase="formal"):
    rec = read(A.run_dir(arm, seed, phase) / "pfo2_det.json")
    require(rec["trainer_deterministic"] is True and rec["deterministic_algorithms"] is True,
            "the run did not train with deterministic algorithms")
    require(rec["entry_sha256"] == sha(ROOT / "tools/pfo2/train_det.py"), "deterministic entry changed")
    return rec


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("accept")
    s.add_argument("--arm", required=True)
    s.add_argument("--seed", required=True, type=int)
    a = p.parse_args(argv)
    deterministic_record(a.arm, a.seed)
    R.accept(a.arm, a.seed)


if __name__ == "__main__":
    main()
