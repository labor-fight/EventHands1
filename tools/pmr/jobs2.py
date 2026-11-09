#!/usr/bin/env python3
"""PMR job lists after engineering amendments (docs/EXPERIMENT_RECORDS.md#pmr-amend1-20261109), appended with
`chain.py add`; tools/pmr/jobs.py (frozen) is unchanged.

    python -m tools.pmr.jobs2 amend1 --design d1 --out outputs/pmr/20261109/jobs/amend1_d1.json
        the screen's cost readings of the three PMR arms through tools/pmr/cost2.py (GPU 4 exclusive, default flags,
        chained after the design's B2 cost job), then the written screen (tools/pmr/analyze.py, unchanged) under a new key
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms as A  # noqa: E402
from tools.pmr.common import PY, require  # noqa: E402


def amend1(design, after):
    p = f"s1:{design}"
    J, prev = [], after
    for c in A.VARIANTS:
        key = f"{p}:cost:r2:{design}{c}"
        J.append({"key": key, "kind": "exclusive", "deps": [prev], "mem_gib": 15, "note": "cost block (amend1, default flags)",
                  "command": [PY, "-m", "tools.pmr.cost2", "run", "--arm", f"{design}{c}", "--policy", "default"]})
        prev = key
    J.append({"key": f"{p}:screen:r2", "kind": "cpu", "deps": [prev], "note": "written screen (amend1)",
              "command": [PY, "-m", "tools.pmr.analyze", "screen", "--design", design]})
    return J


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("amend1")
    s.add_argument("--design", required=True)
    s.add_argument("--after", default=None, help="the B2 cost job of the screen (default s1:<design>:cost:b2)")
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "written once: " + str(out))
    jobs = amend1(a.design, a.after or f"s1:{a.design}:cost:b2")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
