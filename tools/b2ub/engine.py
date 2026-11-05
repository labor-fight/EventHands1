#!/usr/bin/env python3
"""B2UB U2 task `e1c` (docs/EXPERIMENT_RECORDS.md#b2ub-prereg-20261104): PFO's engine closed loop `e1`
(tools/pfo/engine.py, unchanged: zgz_global then zgz_local, 2590 endpoints, W50, segment start = GT + the protocol's
initial noise from one np.random.default_rng(0) shared over the sequences) with one change: before the prediction is fed
back as the next previous state, the degrees of freedom in the clamp set are replaced with the GT at that endpoint.

  T = translation [0:3], R = root rotation [3:6], F = fingers [6:51]; clamp sets none, T, R, F, TR, TF, RF, TRF
  The scored output is still the network's own prediction at every step (as in e1); "none" is e1 itself and "TRF" is the
  GT previous state at every endpoint but the first of each segment (= e3 there).

One task per process, one physical GPU among 2-7; readings/<tag>__clamp-<S>/e1c/{launch.json, states.npz, report.json}
under outputs/b2ub/20261104 (fresh-only; failures preserved).
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.b2ub.common import COUNTS, OUT, ZGZ, now, read, require, save, sha  # noqa: E402

BLOCKS = {"T": slice(0, 3), "R": slice(3, 6), "F": slice(6, 51)}
CLAMPS = ("none", "T", "R", "F", "TR", "TF", "RF", "TRF")
TASK = "e1c"


def clamp_slices(clamp):
    require(clamp in CLAMPS, "unknown clamp set " + str(clamp))
    return () if clamp == "none" else tuple(BLOCKS[c] for c in clamp)


def feed_back(pred, truth, slices):
    """The next previous state: the prediction with the clamped blocks replaced by the GT (float32, a copy)."""
    nxt = np.array(pred, np.float32, copy=True)
    for sl in slices:
        nxt[sl] = truth[sl]
    return nxt


def tag(net_name, route, clamp):
    from tools.pfo.engine import net_tag
    return net_tag(net_name, route) + f"__clamp-{clamp}"


def run(net_name, spec, clamp, route=None, limit=None):
    import torch
    from tools.xcl import evaluate as XE
    from tools.pfo import engine as E
    from tools.pfo import nets as N
    from semkine import eval_track as ET
    slices = clamp_slices(clamp)
    if limit is None:
        require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, range(2, 8))), "one GPU among 2-7")
        XE.configure_environment()
        XE.set_policy()
    spec_obj = read(spec)
    spec_obj = {**spec_obj, "checkpoint": Path(spec_obj["checkpoint"]), "config": Path(spec_obj["config"])}
    base = OUT if limit is None else Path(os.environ["PFO_SMOKE_DIR"])
    out = base / "readings" / tag(net_name, route, clamp) / TASK
    require(not out.exists(), "preserve existing reading: " + str(out))
    out.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = N.load(net_name, route=route, tau=None, device=device, spec=spec_obj)
    launch = {"net": net_name, "task": TASK, "clamp": clamp, "route": route, "identity": net.identity,
              "checkpoint_sha256": sha(net.identity["checkpoint"]), "config_sha256": sha(net.identity["config"]),
              "engine_sha256": sha(__file__), "pfo_engine_sha256": sha(Path(E.__file__)),
              "nets_sha256": sha(Path(N.__file__)), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "limit": limit,
              "started_utc": now()}
    save(out / "launch.json", launch, exclusive=True)
    from mano_layer import ManoLayer
    mano = ManoLayer(net.cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    names = ZGZ
    if limit is not None and os.environ.get("PFO_SMOKE_SEQS"):
        names = tuple(os.environ["PFO_SMOKE_SEQS"].split(","))           # engineering smoke on training recordings
    root, seqs = E.sequences(net.cfg, names)
    arrays, blocks = {}, {}
    shared_rng = np.random.default_rng(0)
    with torch.no_grad():
        for seq, directory in seqs:
            ctx = N.sequence(root, directory, seq, device)
            net.bind(ctx)
            rows = E.endpoint_rows(ctx.aux)
            if seq in COUNTS:
                require(len(rows) == COUNTS[seq], "endpoint denominator changed")
            g = lambda ms: np.asarray(ctx.gt[int(ms)], np.float32).copy()  # noqa: E731
            beta = torch.tensor(np.asarray(ctx.aux["betas"], np.float32), device=device).view(1, -1)
            rec = {}

            def put(key, value):
                rec.setdefault(key, []).append(np.asarray(value))
            n = len(rows) if limit is None else min(len(rows), limit)
            runs = np.asarray(ctx.aux["valid_runs_ms"], np.int64).reshape(-1, 2)
            i = 0
            for rid, (a, b) in enumerate(runs):
                ends = np.arange(a + 49, b, 50, dtype=np.int64)
                if not len(ends):
                    continue
                prev = g(a) + ET.sample_init_noise(net.cfg, shared_rng, 1.0)
                prev_ms = int(a)
                for end in ends:
                    if i >= n:
                        break
                    require(rows[i][:3] == (rid, int(end), prev_ms), "endpoint grid mismatch")
                    i += 1
                    pred = net.step(ctx, end, prev)
                    put("incoming", prev); put("prediction", pred); put("target", g(end))
                    put("gt_previous", g(prev_ms)); put("end_ms", int(end)); put("run_id", rid)
                    put("elapsed_ms", int(end - a))
                    prev = pred if not slices else feed_back(pred, g(end), slices)
                    prev_ms = int(end)
            for key, v in rec.items():
                arrays[f"{seq}/{key}"] = np.stack(v)
            m = E.score(mano, arrays[f"{seq}/prediction"], arrays[f"{seq}/target"], beta, device)
            for key, v in m.items():
                arrays[f"{seq}/{key}"] = v
            blocks[seq] = {k: float(v.mean()) for k, v in m.items()}
            blocks[seq]["n"] = int(len(rec["prediction"]))
            save(out / "progress.json", {"done": seq, "at": now()})
    np.savez_compressed(out / "states.npz", **arrays)
    pooled = {}
    if all(s in blocks for s in ZGZ):
        nn_ = sum(blocks[s]["n"] for s in ZGZ)
        pooled = {k: sum(blocks[s][k] * blocks[s]["n"] for s in ZGZ) / nn_
                  for k in ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm", "wrist_mm", "root_deg",
                            "finger_deg")}
    save(out / "report.json", {**launch, "status": "COMPLETE", "finished_utc": now(), "per_sequence": blocks,
                               "overall": pooled, "states_sha256": sha(out / "states.npz")}, exclusive=True)
    print("COMPLETE", out, {s: round(blocks[s].get("mpjpe_ra_mm", float("nan")), 3) for s in blocks}, flush=True)
    return out


__all__ = ["BLOCKS", "CLAMPS", "TASK", "clamp_slices", "feed_back", "run", "tag"]
