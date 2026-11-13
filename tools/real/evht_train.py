#!/usr/bin/env python3
"""REAL amendment 5 training entry (docs/EXPERIMENT_RECORDS.md#real-amend5-20261111): the frozen PMR / ABL entries
unchanged -- tools/pmr/train11.py (from scratch: tools/pmr/train.py with the union registry 6 and the router block of
tools/pmr/ablrt.py: r_s2d + PFA or none = S45's ray association) and tools/pmr/train_rol7.py (rollout fine-tunes:
tools/pmr/train_rol2.py with train_rol6's patches and the ray arm's rollout model) -- with their registry check rebound to
the real registry (tools/real/evht.py `build_config` / `config_path` / `parse`) and the router frame set to the real
canvas (tools/real/evht.py `enter`). One card (DEVICES 1), so Lightning launches no second rank.

Fresh-only, as the PMR entries: a run directory left by a killed attempt (a borrowed card taken back) is archived to
outputs/real/20261110/pack/preempted/ and the training starts again from step 0 (AGENTS.md, gpu-borrow: fresh-only
trainings restart from scratch after a preemption).

    python tools/real/evht_train.py --config configs/real_20261110/pmr/pmr_<arm>_s<seed>[_smoke].yaml
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402

from tools.real import evht as X  # noqa: E402
from tools.real import ns  # noqa: E402


def main():
    X.require(len(sys.argv) == 3 and sys.argv[1] == "--config", "usage: evht_train.py --config <registered config>")
    path = Path(sys.argv[2]).resolve()
    cfg = yaml.safe_load(path.read_text())
    arm = cfg["PFO"]["ARM"]
    X.parse(arm)
    X.enter()
    if X.kind(arm) == "rol":
        from tools.pmr import train_rol7 as W
        T = W.W.T
    else:
        from tools.pmr import train11 as W
        T = W.T
    T.build_config, T.config_path, T.parse = X.build_config, X.config_path, X.parse
    out = Path(cfg["TRAIN"]["OUTPUT_DIR"])
    if out.exists():
        dst = ns.OUT / "pack" / "preempted" / f"{out.name}__{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out), str(dst))
        print(f"fresh-only: archived the earlier attempt {out.name} to {dst}", flush=True)
    T.main()


if __name__ == "__main__":
    main()
