#!/usr/bin/env python3
"""Engineering smoke of a PMR training on a borrowed pair (docs/EXPERIMENT_RECORDS.md#pmr-amend4-20261109; never read):
the frozen tools/pmr/train.py unchanged, with its registry rebound to D1-R's smoke config (B2's preflight shortened to
20 steps) whose output directory is outputs/pmr/20261109/borrow_smoke/run; the config is written next to it.

    python tools/pmr/borrow_smoke.py write
    gpu-borrow run --gpus 0,1 --state-dir <dir>/borrow -- python tools/pmr/borrow_job.py --key smoke:d1r ... -- \\
        env NCCL_P2P_DISABLE=1 python tools/pmr/borrow_smoke.py --config <dir>/pmr_d1r_s3407_smoke.yaml
Lightning relaunches rank 1 with this same argv.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402

from tools.pmr import arms as A  # noqa: E402
from tools.pmr import train as T  # noqa: E402
from tools.pmr.common import OUT, require  # noqa: E402

DIR = OUT / "borrow_smoke"
CFG = DIR / "pmr_d1r_s3407_smoke.yaml"


def build_config(arm, seed, phase="formal"):
    require((arm, seed, phase) == ("d1r", 3407, A.SMOKE), "the borrow smoke only")
    cfg = A.build_config(arm, seed, phase)
    cfg["TRAIN"]["OUTPUT_DIR"] = str(DIR / "run")
    cfg["EVAL"]["OUTPUT_DIR"] = str(DIR / "run" / "eval")
    return cfg


def config_path(arm, seed, phase="formal"):
    return CFG


T.build_config, T.config_path = build_config, config_path

if __name__ == "__main__":
    if sys.argv[1:] == ["write"]:
        DIR.mkdir(parents=True, exist_ok=True)
        require(not CFG.exists(), "smoke config exists")
        CFG.write_text(yaml.safe_dump(build_config("d1r", 3407, A.SMOKE), sort_keys=False))
        sys.exit(0)
    T.main()
