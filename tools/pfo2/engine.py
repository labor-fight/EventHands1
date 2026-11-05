#!/usr/bin/env python3
"""PFO2 probe tasks (docs/EXPERIMENT_RECORDS.md#pfo2-p0-prereg-20261103) on PFO's frozen engine machinery
(tools/pfo/engine.py helpers and tools/pfo/nets.py, unchanged): same endpoints, same injected errors (the PFO `pert`
task's RNG salts), same scoring; one task per process, one physical GPU among 2-7; readings under
outputs/pfo2/20261103/readings/<tag>/<task>/ (fresh-only).

  pert_rf   zgz: the E3 step plus the R15 and F15 injections of PFO's `pert` (identical injected states, so retention
            equals the `pert` reading's R15 / F15 for the same network and routes)
  cap_inj   P0-C: the slots (17 x 258) and prev48 the readout receives, at GT + R15 and GT + F15 (PFO `pert` draws),
            on the nine training local recordings (`fit`) and zgz (`test`), every endpoint, with the injected delta, the
            network's outputs at the injected and at the GT state
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pfo import engine as E  # noqa: E402
from tools.pfo2.common import COUNTS, OUT, TRAIN_LOCAL, ZGZ, now, read, require, save, sha  # noqa: E402

TASKS = ("pert_rf", "cap_inj")
RF = ("R15", "F15")


def kind_index(kind):
    return list(E.PERT_KINDS).index(kind)


class Capture:
    """Records the readout inputs (aux slots, prev48) of the latest trunk call of one network, and counts the calls."""

    def __init__(self, model):
        self.calls, self.last = 0, None
        xyz = model.xyz
        native = xyz.forward

        def forward(tok, d, mask, verts, assoc, prev48, *args, **kwargs):
            delta, stats, aux = native(tok, d, mask, verts, assoc, prev48, *args, **kwargs)
            self.calls += 1
            self.last = (aux["slots"].detach().float().cpu().numpy(), prev48.detach().float().cpu().numpy())
            return delta, stats, aux
        xyz.forward = forward

    def take(self, n_before):
        """The capture of the step just run, or None when the packet was empty (the trunk was not called)."""
        if self.calls == n_before:
            return None
        require(self.calls == n_before + 1, "one trunk call per step")
        return self.last


def run(net_name, task, route=None, spec=None, limit=None, sidecar=None):
    import torch
    from tools.xcl import evaluate as XE
    from tools.pfo import nets as N
    require(task in TASKS, "unknown task " + task)
    if limit is None:
        require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, range(2, 8))), "one GPU among 2-7")
        XE.configure_environment()
        XE.set_policy()
    spec_obj = read(spec)
    spec_obj = {**spec_obj, "checkpoint": Path(spec_obj["checkpoint"]), "config": Path(spec_obj["config"])}
    base = OUT if limit is None else Path(os.environ["PFO_SMOKE_DIR"])
    tag = E.net_tag(net_name, route)
    out = base / "readings" / tag / task
    require(not out.exists(), "preserve existing reading: " + str(out))
    out.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = N.load(net_name, route=route, tau=None, device=device, spec=spec_obj)
    launch = {"net": net_name, "task": task, "route": route, "identity": net.identity,
              "checkpoint_sha256": sha(net.identity["checkpoint"]), "config_sha256": sha(net.identity["config"]),
              "pfo_engine_sha256": sha(Path(E.__file__)), "engine_sha256": sha(__file__),
              "nets_sha256": sha(Path(N.__file__)), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "limit": limit,
              "started_utc": now()}
    save(out / "launch.json", launch, exclusive=True)
    from mano_layer import ManoLayer
    mano = ManoLayer(net.cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    cap = Capture(net.model) if task == "cap_inj" else None
    groups = [("test", ZGZ)] if task == "pert_rf" else [("fit", TRAIN_LOCAL), ("test", ZGZ)]
    if limit is not None and os.environ.get("PFO_SMOKE_SEQS"):
        groups = [(g, tuple(os.environ["PFO_SMOKE_SEQS"].split(","))) for g, _ in groups[:1]]
    prior = E.prior_noise_fn(net.cfg)
    arrays, blocks = {}, {}
    with torch.no_grad():
        for group, names in groups:
            root, seqs = E.sequences(net.cfg, names)
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
                stride = 1
                n = len(rows) if limit is None else min(len(rows), limit)
                for i in range(0, n, stride):
                    _, end, prev_ms, _ = rows[i]
                    truth = g(prev_ms)
                    base_pred = net.step(ctx, end, truth)
                    put("prediction", base_pred); put("target", g(end)); put("gt_previous", truth)
                    put("end_ms", int(end))
                    for kind in RF:
                        x, dlt, sl = E.inject(truth, kind, E.rng_for(seq, int(end), E.SALT["pert"] + kind_index(kind)),
                                              prior)
                        k1 = cap.calls if cap else 0
                        y = net.step(ctx, end, x)
                        dd = dlt[sl]
                        put(f"{kind}/prediction", y); put(f"{kind}/delta", dlt)
                        put(f"{kind}/rho", float(np.dot(y[sl].astype(np.float64) - base_pred[sl], dd)
                                                 / max(np.dot(dd, dd), 1e-12)))
                        if cap:
                            c = cap.take(k1)
                            put(f"{kind}/captured", c is not None)
                            put(f"{kind}/slots", (c[0][0] if c else np.zeros((17, 258), np.float32)).astype(np.float16))
                            put(f"{kind}/prev48", c[1][0] if c else np.zeros(48, np.float32))
                for key, v in rec.items():
                    arrays[f"{seq}/{key}"] = np.stack(v)
                m = E.score(mano, arrays[f"{seq}/prediction"], arrays[f"{seq}/target"], beta, device)
                blocks[seq] = {"group": group, "n": int(len(rec["prediction"])),
                               **{k: float(v.mean()) for k, v in m.items()}}
                for kind in RF:
                    mk = E.score(mano, arrays[f"{seq}/{kind}/prediction"], arrays[f"{seq}/target"], beta, device)
                    for key, v in mk.items():
                        arrays[f"{seq}/{kind}/{key}"] = v
                    rho = arrays[f"{seq}/{kind}/rho"]
                    blocks[seq][kind] = {**{k: float(v.mean()) for k, v in mk.items()},
                                         "rho_mean": float(rho.mean()), "rho_median": float(np.median(rho))}
                save(out / "progress.json", {"done": seq, "at": now()})
    np.savez_compressed(out / "states.npz", **arrays)
    report = {**launch, "status": "COMPLETE", "finished_utc": now(), "per_sequence": blocks,
              "states_sha256": sha(out / "states.npz")}
    if sidecar is not None:
        report["sidecar"] = sidecar()
    save(out / "report.json", report, exclusive=True)
    print("COMPLETE", out, {s: round(b.get("mpjpe_ra_mm", float("nan")), 3) for s, b in blocks.items()}, flush=True)
    return out


__all__ = ["Capture", "RF", "TASKS", "kind_index", "run"]
