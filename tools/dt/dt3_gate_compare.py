#!/usr/bin/env python3
"""DT3 deploy gates: bit-identity of a new evaluation npz against a recorded one (CPU, read-only).

    python tools/dt/dt3_gate_compare.py NEW.npz REC.npz [--prefix 'model|'] [--prefix 'tf|']

Selects the arrays whose key starts with any `--prefix` (no `--prefix`: every key) in both files and compares them key by key
with `np.array_equal` (NaN at the same place counts as equal for float arrays, so two bit-identical runs with NaN entries pass);
dtype and shape must match too. If NEW.json and REC.json (the siblings with the same stem) both exist, their
`model.overall.mpjpe_ra_mm` is compared with `==` (no tolerance). Prints the first differing key and its first differing index
on axis 0 (the step for the per-step arrays), then one line per further difference.

Exit status: 0 = every selected array (and the json RA, when both jsons exist) identical; 1 = any difference, a key on one side
only, a `--prefix` that selects nothing, or a missing / unreadable json RA; 2 = usage error or an unreadable npz.
Strict equality is deliberate (same GPU model as the recording); there is no tolerance option.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def first_diff_index(a: np.ndarray, b: np.ndarray):
    """First index on axis 0 where `a` and `b` differ (over the common length when only axis 0 differs); None for 0-d arrays
    or when no difference is found in the common part."""
    if a.ndim == 0 or b.ndim == 0 or a.ndim != b.ndim or a.shape[1:] != b.shape[1:]:
        return None
    n = min(a.shape[0], b.shape[0])
    x, y = a[:n], b[:n]
    if x.dtype.kind in "fc" and y.dtype.kind in "fc":
        neq = (x != y) & ~(np.isnan(x) & np.isnan(y))
    else:
        neq = x != y
    rows = np.nonzero(neq.reshape(n, -1).any(axis=1))[0] if n else np.array([], int)
    if rows.size:
        return int(rows[0])
    return n if a.shape[0] != b.shape[0] else None


def same(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))


def describe(key: str, a: np.ndarray, b: np.ndarray) -> str:
    parts = [key]
    if a.dtype != b.dtype:
        parts.append(f"dtype {a.dtype} vs {b.dtype}")
    if a.shape != b.shape:
        parts.append(f"shape {a.shape} vs {b.shape}")
    i = first_diff_index(a, b)
    if i is not None:
        parts.append(f"step {i}")
        if i < min(a.shape[0], b.shape[0]):
            ra, rb = np.asarray(a[i]).ravel(), np.asarray(b[i]).ravel()
            ne = ra != rb
            if ra.dtype.kind in "fc" and rb.dtype.kind in "fc":
                ne &= ~(np.isnan(ra) & np.isnan(rb))
            j = int(np.nonzero(ne)[0][0])
            at = [i] + [int(t) for t in np.unravel_index(j, a.shape[1:])] if a.ndim > 1 else [i]
            parts.append(f"at {at}: new {ra[j].item()!r} rec {rb[j].item()!r}")
            if a.shape == b.shape and a.dtype.kind in "fciu" and b.dtype.kind in "fciu":
                d = np.abs(a.astype(np.float64) - b.astype(np.float64))
                d = d[~np.isnan(d)]
                n = int((a.reshape(a.shape[0], -1) != b.reshape(b.shape[0], -1)).any(axis=1).sum()) if a.ndim else 1
                parts.append(f"{n} differing steps, max|diff| {float(d.max()) if d.size else float('nan')!r}")
    elif a.ndim == 0 and b.ndim == 0:
        parts.append(f"new {a.item()!r} rec {b.item()!r}")
    return "; ".join(parts)


def json_ra(path: Path):
    j = json.loads(path.read_text())
    return j["model"]["overall"]["mpjpe_ra_mm"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("new", help="new evaluation npz")
    ap.add_argument("rec", help="recorded evaluation npz")
    ap.add_argument("--prefix", action="append", default=[], help="key prefix to compare, e.g. 'model|' (repeatable; default all keys)")
    a = ap.parse_args(argv)
    try:
        zn, zr = np.load(a.new, allow_pickle=False), np.load(a.rec, allow_pickle=False)
        kn, kr = set(zn.files), set(zr.files)
    except Exception as e:                                          # missing file, not an npz
        print(f"error: cannot read the npz: {e}", file=sys.stderr)
        return 2
    pre = tuple(a.prefix)
    sel = (lambda k: k.startswith(pre)) if pre else (lambda k: True)
    sn, sr = {k for k in kn if sel(k)}, {k for k in kr if sel(k)}
    fail = []
    for p in pre:
        if not any(k.startswith(p) for k in kn | kr):
            fail.append(f"--prefix {p!r} selects no key in either file")
    if not (sn | sr):
        fail.append("no key selected")
    only_new, only_rec = sorted(sn - sr), sorted(sr - sn)
    fail += [f"missing in REC: {k}" for k in only_new] + [f"missing in NEW: {k}" for k in only_rec]
    common = sorted(sn & sr)
    diffs = []
    for k in common:
        x, y = zn[k], zr[k]
        if not same(x, y):
            diffs.append(describe(k, x, y))
    scope = " ".join(repr(p) for p in pre) if pre else "all keys"
    print(f"npz {a.new}\n vs {a.rec}\n prefixes {scope}: {len(common)} arrays compared, {len(common) - len(diffs)} identical, "
          f"{len(diffs)} differ, {len(only_new) + len(only_rec)} on one side only")
    if diffs:
        print(f"FIRST DIFF: {diffs[0]}")
        for d in diffs[1:20]:
            print(f"  diff: {d}")
        if len(diffs) > 20:
            print(f"  ... {len(diffs) - 20} more")
    jn, jr = Path(a.new).with_suffix(".json"), Path(a.rec).with_suffix(".json")
    if jn.exists() and jr.exists():
        try:
            rn, rr = json_ra(jn), json_ra(jr)
        except Exception as e:                                      # no model.overall.mpjpe_ra_mm, bad json
            fail.append(f"json model.overall.mpjpe_ra_mm unreadable: {type(e).__name__}: {e}")
        else:
            eq = rn == rr
            print(f"json model.overall.mpjpe_ra_mm: new {rn!r} {'==' if eq else '!='} rec {rr!r}")
            if not eq:
                fail.append(f"json model.overall.mpjpe_ra_mm {rn!r} != {rr!r}")
    else:
        print(f"json: not compared ({', '.join(str(p) for p in (jn, jr) if not p.exists())} missing)")
    for f in fail:
        print(f"FAIL: {f}")
    ok = not diffs and not fail
    print("PASS: identical" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
