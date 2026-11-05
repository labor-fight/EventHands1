#!/usr/bin/env python3
"""PFO2 frozen manifests for amendments and later stages (docs/EXPERIMENT_RECORDS.md#pfo2-amend1-20261103 onwards):
the stage's file list is a spec tools/pfo2/manifest_<stage>.json ({"files": [path or glob, ...]}, written once with the
stage), so this tool itself never changes.

    python -m tools.pfo2.freeze2 write --stage X     outputs/pfo2/20261103/manifests/X.json (written once)
    python -m tools.pfo2.freeze2 verify --stage X    re-hash; exit non-zero on any difference
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo2.common import OUT, now, read, require, save, sha  # noqa: E402

HERE = Path(__file__).resolve().parent


def expand(stage):
    spec = HERE / f"manifest_{stage}.json"
    files = [str(spec.relative_to(ROOT))]
    for item in read(spec)["files"]:
        if any(c in item for c in "*?["):
            hits = sorted(str(p.relative_to(ROOT)) for p in ROOT.glob(item) if p.is_file())
            require(hits, "glob matches nothing: " + item)
            files += hits
        else:
            files.append(item)
    return list(dict.fromkeys(files))


def write(stage):
    dest = OUT / "manifests" / f"{stage}.json"
    require(not dest.exists(), "manifest written once: " + str(dest))
    files = expand(stage)
    missing = [f for f in files if not (ROOT / f).exists()]
    require(not missing, f"missing files: {missing[:5]}")
    save(dest, {"stage": stage, "written_at_utc": now(), "files": {f: sha(ROOT / f) for f in files}}, exclusive=True)
    print(len(files), "files ->", dest)


def verify(stage):
    m = read(OUT / "manifests" / f"{stage}.json")
    bad = [f for f, h in m["files"].items() if not (ROOT / f).exists() or sha(ROOT / f) != h]
    require(not bad, f"frozen files changed: {bad[:5]}")
    print("manifest", stage, "OK", len(m["files"]), "files")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("verb", choices=("write", "verify"))
    p.add_argument("--stage", required=True)
    a = p.parse_args(argv)
    (write if a.verb == "write" else verify)(a.stage)


if __name__ == "__main__":
    main()
