#!/usr/bin/env python3
"""EVF asynchronous-synchronous equivalence (docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104, final configuration):
every zgz E1 endpoint of an accepted point-model run, from the state the closed loop actually fed it (the E1 reading's
`incoming`), with the packet's events pushed in arrival order in 1 ms chunks into `semkine/evf_async.py`, against the
synchronous step (the E1 reading's `prediction`, and a fresh synchronous step on the same GPU).

    python -m tools.evf.asynccheck --tag TAG [--chunk-ms 1]       TAG = an EVF reading tag with an e1 reading
Writes readings/<TAG>/async/report.json: max |async - sync| per sequence and overall (the rule: <= 1e-4), the number of
packets above the node cap (the asynchronous runtime refuses them; the rule needs none), per-step async time.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.evf.common import OUT, ZGZ, now, read, require, save  # noqa: E402

TOL = 1e-4


def main(argv=None):
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--chunk-ms", type=int, default=1)
    a = p.parse_args(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, range(2, 8))), "one GPU among 2-7")
    from tools.xcl import evaluate as XE
    XE.configure_environment()
    XE.set_policy()
    from semkine import eval_track as ET
    from semkine.evf_async import AsyncPoint
    from tools.evf import nets2 as N2
    from tools.pfo import nets as PN
    from tools.pfo.engine import endpoint_rows, sequences
    d = OUT / "readings" / a.tag
    launch = read(d / "e1" / "launch.json")
    out = d / "async"
    require(not out.exists(), "measured once: " + str(out))
    out.mkdir(parents=True)
    sp = read(OUT / "specs" / f"{a.tag}.json")
    require(sp["checkpoint"] == launch["identity"]["checkpoint"], "spec and reading disagree")
    device = torch.device("cuda")
    net = N2.load_evf(a.tag, {**sp, "checkpoint": Path(sp["checkpoint"]), "config": Path(sp["config"])}, device)
    model = net.model
    rt = AsyncPoint(model)
    z = np.load(d / "e1" / "states.npz")
    root, seqs = sequences(net.cfg, ZGZ)
    rep = {"tag": a.tag, "chunk_ms": a.chunk_ms, "started_utc": now(), "per_sequence": {}}
    worst = 0.0
    with torch.no_grad():
        for seq, directory in seqs:
            ctx = PN.sequence(root, directory, seq, device)
            net.bind(ctx)
            rows = endpoint_rows(ctx.aux)
            inc, pred = z[f"{seq}/incoming"], z[f"{seq}/prediction"]
            require(len(inc) == len(rows), "endpoint count")
            dmax, dfresh, capped, times = 0.0, 0.0, 0, []
            for i, (_, end, _, _) in enumerate(rows):
                ev5 = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, int(end), 50)
                prev = torch.from_numpy(np.ascontiguousarray(inc[i], np.float32)).view(1, -1).to(device)
                pk = ET.make_eval_packet(ev5, prev, ctx.betas, ctx.K, 50, device)
                sync = model.forward_packet(pk)[0]
                if int(model.evf_stats["nodes"][0]) > int(model.evf["MAX_NODES"]):
                    capped += 1
                    continue
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                rt.begin(prev, ctx.betas, ctx.K, 0.05)
                ev = torch.from_numpy(ev5[:, 1:]).to(device)
                ms = np.floor(ev5[:, 3] * 1000.0 / a.chunk_ms).astype(np.int64)
                cut = np.flatnonzero(np.diff(ms)) + 1
                for lo, hi in zip(np.r_[0, cut], np.r_[cut, len(ms)]):
                    rt.push(ev[lo:hi])
                y = rt.query()[0]
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1e3)
                dmax = max(dmax, float((y - torch.from_numpy(pred[i]).to(device)).abs().max()))
                dfresh = max(dfresh, float((y - sync).abs().max()))
            rep["per_sequence"][seq] = {"max_abs_vs_reading": dmax, "max_abs_vs_fresh_sync": dfresh, "capped": capped,
                                        "n": len(rows), "async_step_ms_mean": float(np.mean(times)) if times else None}
            worst = max(worst, dmax, dfresh)
            print(seq, rep["per_sequence"][seq], flush=True)
    rep.update(max_abs=worst, rule=f"<= {TOL} and no capped packet",
               status="PASS" if worst <= TOL and all(v["capped"] == 0 for v in rep["per_sequence"].values()) else "FAIL",
               finished_utc=now())
    save(out / "report.json", rep, exclusive=True)
    print(rep["status"], worst, flush=True)


if __name__ == "__main__":
    main()
