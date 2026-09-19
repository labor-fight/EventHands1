#!/usr/bin/env python3
"""Is the ~39 mm plateau caused by throwing away 95% of the events?

`event_gnn` keeps `ENCODER_MAX_NODES` events per packet and the training run had to cap that at
2048 to fit 44 GiB, which is ~5% of an average packet. That is the arm's most obvious suspect, so
it gets tested before anything else is blamed.

The test is inference-only and needs no retraining: one already-selected checkpoint, scored by the
same recursive protocol at several node budgets. Node count is not a learned shape -- the message
MLP and the readout are per-node and per-edge -- so a model trained at 2048 runs unchanged at 512
or 4096, and the only thing that moves is how much of the packet it sees.

Read it as: if RA is flat across a 8x range of retained evidence, the subsampling is not what is
holding the arm at 39 mm and the next hypothesis has to come from somewhere else.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--step", type=int, required=True, help="which grid checkpoint to probe")
    ap.add_argument("--nodes", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    run = Path(a.run_dir)
    ckpt = next(iter(run.glob(f"*step={a.step}.ckpt")), None)
    assert ckpt is not None, f"no step={a.step} checkpoint in {run}"
    meta = json.loads((run / "training_metadata.json").read_text())
    cfg_src = next(iter(sorted(run.glob("*.yaml"))), None) or Path(meta["config_path"])
    base = yaml.safe_load(Path(cfg_src).read_text())
    trained = base["MODEL"]["ENCODER_MAX_NODES"]

    results = {}
    for n in a.nodes:
        work = Path(f"/tmp/s36_nodes_{n}")
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        # A symlink keeps the probe from copying a checkpoint per arm.
        (work / ckpt.name).symlink_to(ckpt.resolve())
        shutil.copy(run / "training_metadata.json", work / "training_metadata.json")
        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg["MODEL"]["ENCODER_MAX_NODES"] = n
        (work / "cfg.yaml").write_text(yaml.safe_dump(cfg))

        p = subprocess.run([a.python, "tools/select_checkpoint.py", "--run-dir", str(work),
                            "--config", str(work / "cfg.yaml")],
                           cwd=REPO, capture_output=True, text=True)
        sel = next(iter(sorted(work.glob("selection_val_core_step50*.json"))), None)
        if sel is None:
            print(f"nodes={n:<6d} FAILED\n{p.stdout[-800:]}\n{p.stderr[-800:]}")
            continue
        d = json.loads(sel.read_text())["selected"]
        results[n] = d["mpjpe_ra_mm"]
        tag = "  <- trained at this budget" if n == trained else ""
        print(f"nodes={n:<6d} RA={d['mpjpe_ra_mm']:8.4f}  abs={d['mpjpe_abs_mm']:8.3f}{tag}",
              flush=True)

    if len(results) > 1:
        lo, hi = min(results.values()), max(results.values())
        print(f"\nRA spread across {min(results)}..{max(results)} nodes: {hi - lo:.4f} mm")


if __name__ == "__main__":
    main()
