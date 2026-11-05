#!/usr/bin/env python3
"""EVF freeze manifests (docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104): the SHA-256 of every file a stage's runs and
readings depend on, written before any reading of that stage; frozen files are never edited (new behaviour goes to new
files).

    python -m tools.evf.freeze write --stage S      tools/evf/manifest_<S>.json (a list of repo paths) ->
                                                    outputs/evf/20261104/manifests/<S>.json (once)
    python -m tools.evf.freeze verify --stage S     exit 1 if any listed file changed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

from tools.evf.common import OUT, now, read, require, save, sha  # noqa: E402


def files(stage):
    lst = json.loads((ROOT / "tools/evf" / f"manifest_{stage}.json").read_text())
    require(isinstance(lst, list) and lst and len(set(lst)) == len(lst), "manifest list")
    return lst


def write(stage):
    out = OUT / "manifests" / f"{stage}.json"
    require(not out.exists(), "frozen once: " + str(out))
    lst = files(stage)
    save(out, {"stage": stage, "files": {f: sha(ROOT / f) for f in lst},
               "list_sha256": sha(ROOT / "tools/evf" / f"manifest_{stage}.json"), "frozen_at_utc": now()},
         exclusive=True)
    print("frozen", stage, len(lst), "files")


def verify(stage):
    m = read(OUT / "manifests" / f"{stage}.json")
    bad = [f for f, h in m["files"].items() if not (ROOT / f).exists() or sha(ROOT / f) != h]
    print("verify", stage, "OK" if not bad else f"CHANGED {bad}")
    sys.exit(1 if bad else 0)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("verb", choices=("write", "verify"))
    p.add_argument("--stage", required=True)
    a = p.parse_args(argv)
    (write if a.verb == "write" else verify)(a.stage)


if __name__ == "__main__":
    main()
