#!/usr/bin/env python3
"""EVF engineering check for the second registry (tools/evf/arms2.py; tools/evf/replay.py, frozen with round 1).
EVF engineering check (docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104): two 3-step trainings of the same config are
bitwise identical (every state-dict tensor, every optimiser state, the learning rate) at each of the three step
checkpoints; and a screen run's checkpoints equal the formal run's at the shared steps.

    python -m tools.evf.replay2 smoke3 --arm A --seed S          smoke3 vs smoke3b
    python -m tools.evf.replay2 prefix --arm A --seed S          screen vs formal at 500 .. 2000
Writes receipts/cmp_<name>.json (once); exit 1 on any mismatch.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.evf import arms2 as A  # noqa: E402
from tools.evf.common import OUT, now, require, save  # noqa: E402


def flat(obj, prefix=""):
    import torch
    if torch.is_tensor(obj):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            yield from flat(obj[k], f"{prefix}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from flat(v, f"{prefix}/{i}")
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        yield prefix, obj


def compare(pa, pb):
    import torch
    a = torch.load(pa, map_location="cpu")
    b = torch.load(pb, map_location="cpu")
    out = {"tensors": 0, "mismatch": []}
    for part in ("state_dict", "optimizer_states", "lr_schedulers"):
        fa, fb = dict(flat(a.get(part), part)), dict(flat(b.get(part), part))
        if fa.keys() != fb.keys():
            out["mismatch"].append(f"{part}: keys differ")
            continue
        for k in fa:
            va, vb = fa[k], fb[k]
            if torch.is_tensor(va):
                out["tensors"] += 1
                if not (torch.is_tensor(vb) and va.dtype == vb.dtype and va.shape == vb.shape and torch.equal(va, vb)):
                    out["mismatch"].append(k)
            elif va != vb:
                out["mismatch"].append(k)
    if a.get("global_step") != b.get("global_step"):
        out["mismatch"].append("global_step")
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("smoke3", "prefix"))
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args(argv)
    if a.what == "smoke3":
        pairs = [(A.run_dir(a.arm, a.seed, "smoke3") / f"{A.run_name(a.arm, a.seed, 'smoke3')}-step={s}.ckpt",
                  A.run_dir(a.arm, a.seed, "smoke3b") / f"{A.run_name(a.arm, a.seed, 'smoke3b')}-step={s}.ckpt")
                 for s in (1, 2, 3)]
    else:
        pairs = [(A.run_dir(a.arm, a.seed, "screen") / f"{A.run_name(a.arm, a.seed, 'screen')}-step={s}.ckpt",
                  A.run_dir(a.arm, a.seed, "formal") / f"{A.run_name(a.arm, a.seed, 'formal')}-step={s}.ckpt")
                 for s in (500, 1000, 1500, 2000)]
    rows = []
    for pa, pb in pairs:
        require(pa.exists() and pb.exists(), f"missing {pa} / {pb}")
        rows.append({"a": str(pa), "b": str(pb), **compare(pa, pb)})
    status = "PASS" if all(not r["mismatch"] for r in rows) and all(r["tensors"] > 0 for r in rows) else "FAIL"
    out = OUT / "receipts" / f"cmp_{a.what}_{A.run_name(a.arm, a.seed)}.json"
    save(out, {"what": a.what, "arm": a.arm, "seed": a.seed, "status": status, "rows": rows, "at_utc": now()},
         exclusive=True)
    print(status, [(r["tensors"], len(r["mismatch"])) for r in rows], flush=True)
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
