#!/usr/bin/env python3
"""PFO2 bitwise training checks (docs/EXPERIMENT_RECORDS.md#pfo2-prereg-20261103): two Lightning checkpoints are equal
when the model state, every optimiser state tensor / scalar, the scheduler state and the global step are identical
(metadata such as hyper-parameters and paths is not compared), and the first-batch data-stream hashes of both ranks
(pair_stream_rank{0,1}.json) agree. CPU only; one receipt per check, written once.

    python -m tools.pfo2.replay f1rep --phase preflight|head500
        PFO2's entry on F's recipe against the PFO F run (seed 3407): step 40 of the preflight, step 500 of the formal run
    python -m tools.pfo2.replay base --arm fl1|fgs --seed S --phase preflight|head500
        the base arm (semkine/pfo_trunk4.py) against its single-change reference (semkine/pfo_trunk3.py): step 40 of the
        preflights (seed 3407), or step 500 of the formal run against the reference's head500 run
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo2.arms import run_dir, run_name, spec  # noqa: E402
from tools.pfo2.common import OUT, now, pfo_run, read, require, save, sha  # noqa: E402

STEP = {"preflight": 40, "head500": 500}


def _cmp(a, b, where, bad, counter):
    import torch
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        counter[0] += 1
        same = (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and a.shape == b.shape and torch.equal(a, b))
        if not same:
            diff = None
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape:
                diff = float((a.double() - b.double()).abs().max()) if a.numel() else 0.0
            bad.append({"at": where, "max_abs_diff": diff})
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            bad.append({"at": where, "keys": sorted(map(str, set(a) ^ set(b)))[:10]})
        for k in sorted(set(a) & set(b), key=str):
            _cmp(a[k], b[k], f"{where}/{k}", bad, counter)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            bad.append({"at": where, "len": [len(a), len(b)]})
        for i, (x, y) in enumerate(zip(a, b)):
            _cmp(x, y, f"{where}[{i}]", bad, counter)
        return
    if callable(a) or callable(b):
        return
    if a != b:
        bad.append({"at": where, "values": [repr(a)[:80], repr(b)[:80]]})


def compare(a_ckpt, b_ckpt):
    import torch
    A = torch.load(str(a_ckpt), map_location="cpu", weights_only=False)
    B = torch.load(str(b_ckpt), map_location="cpu", weights_only=False)
    bad, counter = [], [0]
    for key in ("state_dict", "optimizer_states", "lr_schedulers", "global_step"):
        require(key in A and key in B, f"checkpoint lacks {key}")
        _cmp(A[key], B[key], key, bad, counter)
    return {"tensors": counter[0], "mismatches": len(bad), "first": bad[:10]}


def streams(a_dir, b_dir):
    out = {}
    for rank in (0, 1):
        fa, fb = Path(a_dir) / f"pair_stream_rank{rank}.json", Path(b_dir) / f"pair_stream_rank{rank}.json"
        out[rank] = fa.exists() and fb.exists() and read(fa) == read(fb)
    return out


def check(label, a_ckpt, b_ckpt, a_dir, b_dir):
    dest = OUT / "receipts" / f"cmp_{label}.json"
    require(not dest.exists(), "checked once: " + str(dest))
    require(Path(a_ckpt).exists() and Path(b_ckpt).exists(), "missing checkpoint")
    res = compare(a_ckpt, b_ckpt)
    st = streams(a_dir, b_dir)
    ok = res["mismatches"] == 0 and res["tensors"] > 0 and all(st.values())
    receipt = {"label": label, "a": str(a_ckpt), "b": str(b_ckpt), "a_sha256": sha(a_ckpt), "b_sha256": sha(b_ckpt),
               **res, "pair_streams_equal": st, "status": "PASS" if ok else "FAIL", "checked_at_utc": now()}
    save(dest, receipt, exclusive=True)
    print(receipt["status"], label, {k: receipt[k] for k in ("tensors", "mismatches", "pair_streams_equal")}, flush=True)
    require(ok, "bitwise check failed: " + label)
    return receipt


def ckpt(directory, name, step):
    return Path(directory) / f"{name}-step={step}.ckpt"


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="verb", required=True)
    f = sub.add_parser("f1rep")
    f.add_argument("--phase", required=True, choices=tuple(STEP))
    b = sub.add_parser("base")
    b.add_argument("--arm", required=True)
    b.add_argument("--seed", required=True, type=int)
    b.add_argument("--phase", required=True, choices=tuple(STEP))
    a = p.parse_args(argv)
    step = STEP[a.phase]
    if a.verb == "f1rep":
        ph_pfo = "preflight" if a.phase == "preflight" else "formal"
        pd = pfo_run("f1", 3407, ph_pfo)
        qd = run_dir("f1rep", 3407, a.phase)
        return check(f"f1rep_{a.phase}", ckpt(pd, pd.name, step), ckpt(qd, run_name("f1rep", 3407, a.phase), step), pd, qd)
    ref = spec(a.arm)["reference"]
    if a.phase == "preflight":
        require(a.seed == 3407, "preflights run at seed 3407")
        rd, qd = run_dir(ref, 3407, "preflight"), run_dir(a.arm, 3407, "preflight")
        return check(f"{a.arm}_preflight", ckpt(rd, rd.name, step), ckpt(qd, qd.name, step), rd, qd)
    rd, qd = run_dir(ref, a.seed, "head500"), run_dir(a.arm, a.seed, "formal")
    return check(f"{a.arm}_s{a.seed}_head500", ckpt(rd, rd.name, step), ckpt(qd, qd.name, step), rd, qd)


if __name__ == "__main__":
    main()
