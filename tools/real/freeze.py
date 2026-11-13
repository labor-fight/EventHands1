#!/usr/bin/env python3
"""Freeze / verify the real study's files (docs/EXPERIMENT_RECORDS.md#real-prereg-20261110).

    python tools/real/freeze.py write|verify --stage r1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/real/20261110"
DATA = Path("/data1/lyq/code/mesh/EventHands/data")
ROOTS = ("hand_data_real_n346", "hand_data_real_n256")   # amendment 1 (r1 / r1b: hand_data_real51)
CMP_CODE = ("tools/cmp/train.py", "tools/cmp/arms.py", "tools/cmp/common.py", "tools/cmp/run.py", "tools/cmp/nets.py",
            "tools/cmp/ext_epx.py", "tools/cmp/ext_ehpe.py", "semkine/cmp_models.py", "semkine/cmp_seq.py",
            "semkine/cmp_evhp.py", "semkine/cmp_evhpc.py", "semkine/cmp_evhpx.py", "semkine/cmp_seq_crop.py",
            "semkine/cmp_seq_cropx.py", "semkine/cmp_eho.py", "semkine/cmp_ehpe.py", "semkine/cmp_seq_ehpe.py",
            "semkine/train.py", "semkine/dataset.py", "semkine/eval_track.py", "tools/pfo/engine.py", "tools/pfo/nets.py",
            "tools/prepare_hand_data.py")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def files():
    out = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tools/real").glob("*.py"))
    out += sorted(str(p.relative_to(ROOT)) for p in (ROOT / "configs/real_20261110").glob("*.yaml"))
    out += list(CMP_CODE) + [str((OUT / "trans_means.json").relative_to(ROOT))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verb", choices=("write", "verify"))
    ap.add_argument("--stage", default="r1")
    a = ap.parse_args()
    man = OUT / "manifests" / f"{a.stage}.json"
    if a.verb == "write":
        if man.exists():
            sys.exit("manifest exists: " + str(man))
        man.parent.mkdir(parents=True, exist_ok=True)
        rec = {"stage": a.stage, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "files": {f: sha(ROOT / f) for f in files()},
               "data": {f"{r}/{n}": sha(DATA / r / n) for r in ROOTS for n in ("splits_real.json", "conversion.json")}}
        man.write_text(json.dumps(rec, indent=1) + "\n")
        print("wrote", man, len(rec["files"]), "files")
    else:
        rec = json.loads(man.read_text())
        bad = [f for f, h in rec["files"].items() if sha(ROOT / f) != h]
        base = DATA if "/" in next(iter(rec["data"])) else DATA / "hand_data_real51"
        bad += [n for n, h in rec["data"].items() if sha(base / n) != h]
        print("VERIFY", "PASS" if not bad else "FAIL", bad)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
