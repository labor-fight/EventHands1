#!/usr/bin/env python3
"""Reduce a checkpoint directory to one run's step grid.

An interrupted run leaves `...-step=500.ckpt` behind; the restart writes `...-step=500-v1.ckpt`
beside it. `select_checkpoint.py` globs `*step=*.ckpt`, so the grid it scores would mix two
trajectories and could select a checkpoint from the run that was abandoned. Keeping the newest file
per step recovers the completed run, since it wrote every step later than the interrupted one did.

The displaced files are moved, not deleted, so the state before the fix stays inspectable.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--apply", action="store_true", help="without this, only reports")
    a = ap.parse_args()

    for d in a.dirs:
        run = Path(d)
        by_step: dict[int, list[Path]] = {}
        for p in run.glob("*step=*.ckpt"):
            m = re.search(r"step=(\d+)", p.name)
            if m:
                by_step.setdefault(int(m.group(1)), []).append(p)
        moved = []
        for step, paths in sorted(by_step.items()):
            if len(paths) == 1:
                continue
            paths.sort(key=lambda p: p.stat().st_mtime)
            moved += paths[:-1]
        print(f"{run.name}: {len(by_step)} steps, {len(moved)} superseded")
        if not a.apply:
            for p in moved:
                print("  would move", p.name)
            continue
        dest = run / "superseded"
        dest.mkdir(exist_ok=True)
        for p in moved:
            p.rename(dest / p.name)
        # The surviving file for a step may carry the `-v1` suffix; give it the canonical name so
        # the grid reads as one run.
        for step, paths in sorted(by_step.items()):
            live = [p for p in paths if p.exists()]
            if not live:
                continue
            keep = max(live, key=lambda p: p.stat().st_mtime)
            canon = keep.with_name(re.sub(r"-v\d+(?=\.ckpt$)", "", keep.name))
            if canon != keep and not canon.exists():
                keep.rename(canon)
                print("  renamed", keep.name, "->", canon.name)
        for p in run.glob("last*.ckpt"):
            if p.name != "last.ckpt":
                p.rename(dest / p.name)


if __name__ == "__main__":
    main()
