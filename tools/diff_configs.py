#!/usr/bin/env python3
"""Assert two YAML configs differ only in MODEL and LOSS sections."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--allow", nargs="*", default=["MODEL", "LOSS", "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR"])
    args = ap.parse_args()
    with args.a.open() as f:
        ca = yaml.safe_load(f)
    with args.b.open() as f:
        cb = yaml.safe_load(f)

    def flatten(d, prefix=""):
        out = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(flatten(v, key))
            else:
                out[key] = v
        return out

    fa, fb = flatten(ca), flatten(cb)
    allowed_prefixes = tuple(args.allow)
    diffs = []
    for k in sorted(set(fa) | set(fb)):
        if any(k == p or k.startswith(p + ".") for p in allowed_prefixes):
            continue
        if fa.get(k) != fb.get(k):
            diffs.append((k, fa.get(k), fb.get(k)))
    if diffs:
        print("Unexpected config diffs:", file=sys.stderr)
        for k, a, b in diffs:
            print(f"  {k}: {a!r} vs {b!r}", file=sys.stderr)
        sys.exit(1)
    print(f"OK: {args.a} and {args.b} differ only in allowed keys {args.allow}")


if __name__ == "__main__":
    main()
