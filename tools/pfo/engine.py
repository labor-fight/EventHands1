#!/usr/bin/env python3
"""PFO readout engine (docs/EXPERIMENT_RECORDS.md#pfo-prereg-20261103): the same loop for every network, one task per process,
one physical GPU among 2-7.

    python -m tools.pfo.engine run --net N --task T [--route R] [--tau X] [--intervene a,b] [--spec FILE]

Tasks (zgz = zgz_global then zgz_local, 2590 endpoints; tl = the nine training local recordings, in-sample):
  e1 / tl_e1     own closed loop: segment start = GT + the protocol's initial noise (np.random.default_rng(0) shared over
                 the sequences, as XCL `trace` / evalx), then prev <- prediction; identical loop for S43 (W50 LNES)
  e3 / tl_e3     GT previous state every step (single-step error; S43's teacher forcing)
  pert           zgz: GT previous + one injected error per kind and endpoint (T10 / T30 mm translation, R5 / R15 deg root,
                 F05 / F15 rad i.i.d. finger axis-angle, N = the training prior-noise contract), plus the E3 step itself;
                 per kind the output error and the retention rho = <f(x+d) - f(x), d> / |d|^2 on the perturbed block
  loop1          zgz: one step from the network's own E1 incoming state (requires this network's e1 reading)
  f1             zgz: every 8th endpoint, the packet fed 20 times from the GT / mean-pose / own-E1 state
  cap_tl / cap_zgz_loop / cap_zgz_e3
                 stage captures (semkine/pfo_diag.py) for the information probes: training local recordings with the
                 training prior noise (every 2nd endpoint), zgz from the own E1 incoming state, zgz E3
Every task writes readings/<net-tag>/<task>/{launch.json, states.npz, report.json} (fresh-only; failures preserved).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pfo.common import NETS, OUT, TRAIN_LOCAL, ZGZ, COUNTS, now, read, require, save, sha  # noqa: E402

TASKS = ("e1", "e3", "tl_e1", "tl_e3", "pert", "loop1", "f1", "cap_tl", "cap_zgz_loop", "cap_zgz_e3")
PERT_KINDS = {"T10": ("T", 0.010), "T30": ("T", 0.030), "R5": ("R", np.deg2rad(5.0)), "R15": ("R", np.deg2rad(15.0)),
              "F05": ("F", 0.05), "F15": ("F", 0.15), "N": ("N", None)}
F1_STRIDE, F1_ITERS, F1_SAVE = 8, 20, (1, 2, 3, 5, 10, 20)
SALT = {"pert": 101, "cap": 202}


def net_tag(net, route=None, tau=None, intervene=()):
    tag = net
    if route and route not in ("router",):
        tag += f"__{route}"
    if tau is not None:
        tag += f"__tau{tau:g}"
    if intervene:
        tag += "__iv-" + "-".join(sorted(intervene))
    return tag


def reading_dir(net, route=None, tau=None, intervene=(), task=None):
    d = OUT / "readings" / net_tag(net, route, tau, intervene)
    return d / task if task else d


def rng_for(seq, end_ms, salt):
    from tools.s45loc.common import rng_for as R
    return R(seq, end_ms, salt)


def rodrigues(v):
    from tools.s45loc.common import rodrigues as R
    return R(v)


def log_so3(m):
    from tools.s45loc.common import log_so3 as L
    return L(m)


def inject(state, kind, rng, prior_noise):
    """GT state + one error of `kind`; returns (perturbed state, delta in the 51-D parameter vector, block slice)."""
    out = np.array(state, np.float32, copy=True)
    base, mag = PERT_KINDS[kind]
    if base == "T":
        u = rng.standard_normal(3)
        out[0:3] = (out[0:3].astype(np.float64) + mag * u / np.linalg.norm(u)).astype(np.float32)
        sl = slice(0, 3)
    elif base == "R":
        u = rng.standard_normal(3)
        out[3:6] = log_so3(rodrigues(mag * u / np.linalg.norm(u)) @ rodrigues(out[3:6])).astype(np.float32)
        sl = slice(3, 6)
    elif base == "F":
        out[6:51] = (out[6:51].astype(np.float64) + mag * rng.standard_normal(45)).astype(np.float32)
        sl = slice(6, 51)
    else:
        out = (out.astype(np.float64) + prior_noise(rng)).astype(np.float32)
        sl = slice(0, 51)
    delta = out.astype(np.float64) - np.asarray(state, np.float64)
    return out, delta, sl


def prior_noise_fn(cfg):
    """The training prior-noise contract (semkine/dataset.py `_sample_prev_noise`, mixed mode) of the config."""
    t = cfg.get("TRACK", {}) or {}
    small = (float(t.get("PREV_NOISE_T", 0.005)), float(t.get("PREV_NOISE_R", 0.05)), float(t.get("PREV_NOISE_POSE", 0.05)))
    large = (float(t.get("PREV_NOISE_LARGE_T", 0.05)), float(t.get("PREV_NOISE_LARGE_R", 0.3)),
             float(t.get("PREV_NOISE_LARGE_POSE", 0.3)))
    mix = (float(t.get("PREV_NOISE_P_SMALL", 0.5)), float(t.get("PREV_NOISE_P_LARGE", 0.3)))

    def fill(rng, a, b, c):
        n = np.empty(51, np.float64)
        n[0:3] = rng.standard_normal(3) * a
        n[3:6] = rng.standard_normal(3) * b
        n[6:51] = rng.standard_normal(45) * c
        return n

    def directed(rng, n, mag):
        v = rng.standard_normal(n)
        return v / (np.linalg.norm(v) + 1e-8) * rng.uniform(0.0, mag)

    def draw(rng):
        u = float(rng.random())
        if u < mix[0]:
            return fill(rng, *small)
        if u < mix[0] + mix[1]:
            return fill(rng, *large)
        return np.concatenate([directed(rng, 3, large[0]), directed(rng, 3, large[1]), directed(rng, 45, large[2])])
    return draw


def mean_pose():
    from tools.s45tr.readouts import pose_basis
    return pose_basis()[0].astype(np.float32)


def sequences(cfg, names):
    from semkine.dataset import sequences_for_split, splits_manifest
    root = Path(cfg["DATA"]["ROOT"])
    found = {}
    for split in ("val_core", "train"):
        found.update(dict(sequences_for_split(root, split, splits_manifest(cfg))))
    require(all(s in found for s in names), "missing sequences")
    return root, [(s, found[s]) for s in names]


def endpoint_rows(aux):
    from tools.xcl.evaluate import endpoint_grid
    return endpoint_grid(aux)


def score(mano, pred, target, betas, device):
    from tools.xcl.evaluate import score as S
    return S(mano, np.asarray(pred, np.float32), np.asarray(target, np.float32), betas, device)


def run(net_name, task, route=None, tau=None, intervene=(), spec=None, limit=None):
    import torch
    from tools.xcl import evaluate as XE
    from tools.pfo import nets as N
    require(task in TASKS, "unknown task " + task)
    if limit is None:
        require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, range(2, 8))), "one GPU among 2-7")
        XE.configure_environment()
        XE.set_policy()
    spec_obj = read(spec) if spec else None
    if spec_obj:
        spec_obj = {**spec_obj, "checkpoint": Path(spec_obj["checkpoint"]), "config": Path(spec_obj["config"])}
    base = OUT if limit is None else Path(os.environ["PFO_SMOKE_DIR"])
    out = base / reading_dir(net_name, route, tau, intervene, task).relative_to(OUT)
    require(not out.exists(), "preserve existing reading: " + str(out))
    out.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diag = bool(intervene) or task.startswith("cap_")
    if diag:
        from semkine import pfo_diag as D
        D.install()
        D.set_interventions(intervene, mean_pose=mean_pose() if "prev_fing_mean" in intervene else None)
        D.set_capture(task.startswith("cap_"))
    net = N.load(net_name, route=route, tau=tau, device=device, spec=spec_obj)
    launch = {"net": net_name, "task": task, "route": route, "tau": tau, "intervene": sorted(intervene),
              "identity": net.identity, "checkpoint_sha256": sha(net.identity["checkpoint"]),
              "config_sha256": sha(net.identity["config"]), "engine_sha256": sha(__file__),
              "nets_sha256": sha(Path(N.__file__)), "diag_sha256": sha(ROOT / "semkine/pfo_diag.py") if diag else None,
              "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "limit": limit, "started_utc": now()}
    save(out / "launch.json", launch, exclusive=True)
    from mano_layer import ManoLayer
    mano = ManoLayer(net.cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    names = TRAIN_LOCAL if task in ("tl_e1", "tl_e3", "cap_tl") else ZGZ
    if limit is not None and os.environ.get("PFO_SMOKE_SEQS"):
        names = tuple(os.environ["PFO_SMOKE_SEQS"].split(","))           # engineering smoke on training recordings
    root, seqs = sequences(net.cfg, names)
    arrays, blocks = {}, {}
    loop = None
    if task in ("loop1", "f1", "cap_zgz_loop"):
        src = base / reading_dir(net_name, route, tau, intervene if task != "cap_zgz_loop" else (), "e1").relative_to(OUT) / "states.npz"
        require(src.exists(), "needs this network's e1 reading: " + str(src))
        loop = np.load(src, allow_pickle=False)
    prior = prior_noise_fn(net.cfg)
    mu = mean_pose() if task == "f1" else None
    shared_rng = np.random.default_rng(0)
    from semkine import eval_track as ET
    with torch.no_grad():
        for seq, directory in seqs:
            ctx = N.sequence(root, directory, seq, device)
            net.bind(ctx)
            rows = endpoint_rows(ctx.aux)
            if seq in COUNTS:
                require(len(rows) == COUNTS[seq], "endpoint denominator changed")
            g = lambda ms: np.asarray(ctx.gt[int(ms)], np.float32).copy()  # noqa: E731
            beta = torch.tensor(np.asarray(ctx.aux["betas"], np.float32), device=device).view(1, -1)
            rec = {}

            def put(key, value):
                rec.setdefault(key, []).append(np.asarray(value))
            n = len(rows) if limit is None else min(len(rows), limit)
            if task in ("e1", "tl_e1"):
                rng = shared_rng if task == "e1" else np.random.default_rng(0)
                runs = np.asarray(ctx.aux["valid_runs_ms"], np.int64).reshape(-1, 2)
                i = 0
                for rid, (a, b) in enumerate(runs):
                    ends = np.arange(a + 49, b, 50, dtype=np.int64)
                    if not len(ends):
                        continue
                    prev = g(a) + ET.sample_init_noise(net.cfg, rng, 1.0)
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
                        prev, prev_ms = pred, int(end)
            elif task in ("e3", "tl_e3", "cap_zgz_e3"):
                for i in range(n):
                    _, end, prev_ms, _ = rows[i]
                    put("prediction", net.step(ctx, end, g(prev_ms))); put("incoming", g(prev_ms))
                    put("target", g(end)); put("gt_previous", g(prev_ms)); put("end_ms", int(end))
            elif task == "pert":
                for i in range(n):
                    _, end, prev_ms, _ = rows[i]
                    truth = g(prev_ms)
                    base = net.step(ctx, end, truth)
                    put("prediction", base); put("target", g(end)); put("gt_previous", truth); put("end_ms", int(end))
                    for k, kind in enumerate(PERT_KINDS):
                        x, dlt, sl = inject(truth, kind, rng_for(seq, int(end), SALT["pert"] + k), prior)
                        y = net.step(ctx, end, x)
                        dd = dlt[sl]
                        put(f"{kind}/prediction", y)
                        put(f"{kind}/delta", dlt)
                        put(f"{kind}/rho", float(np.dot(y[sl].astype(np.float64) - base[sl], dd) / max(np.dot(dd, dd), 1e-12)))
            elif task in ("loop1", "cap_zgz_loop"):
                inc = loop[f"{seq}/incoming"]
                require(len(inc) == len(rows) or limit is not None, "own-loop endpoint count differs")
                for i in range(n):
                    _, end, prev_ms, _ = rows[i]
                    put("prediction", net.step(ctx, end, inc[i])); put("incoming", inc[i])
                    put("target", g(end)); put("gt_previous", g(prev_ms)); put("end_ms", int(end))
                    if task == "loop1":
                        put("e3_prediction", net.step(ctx, end, g(prev_ms)))
            elif task == "f1":
                inc = loop[f"{seq}/incoming"]
                for i in range(0, n, F1_STRIDE):
                    _, end, prev_ms, _ = rows[i]
                    truth = g(prev_ms)
                    meanpose = truth.copy()
                    meanpose[6:51] = mu
                    for start, x0 in (("gt", truth), ("mean", meanpose), ("loop", inc[i])):
                        x = np.asarray(x0, np.float32).copy()
                        for k in range(1, F1_ITERS + 1):
                            x = net.step(ctx, end, x)
                            if k in F1_SAVE:
                                put(f"{start}_iter{k}", x)
                    put("index", i); put("end_ms", int(end)); put("target", g(end)); put("gt_previous", truth)
            elif task == "cap_tl":
                for i in range(0, n, 2):
                    _, end, prev_ms, _ = rows[i]
                    x, _, _ = inject(g(prev_ms), "N", rng_for(seq, int(end), SALT["cap"]), prior)
                    put("prediction", net.step(ctx, end, x)); put("incoming", x)
                    put("target", g(end)); put("gt_previous", g(prev_ms)); put("end_ms", int(end))
            for key, v in rec.items():
                arrays[f"{seq}/{key}"] = np.stack(v)
            if "prediction" in rec:
                m = score(mano, arrays[f"{seq}/prediction"], arrays[f"{seq}/target"], beta, device)
                for key, v in m.items():
                    arrays[f"{seq}/{key}"] = v
                blocks[seq] = {k: float(v.mean()) for k, v in m.items()}
                blocks[seq]["n"] = int(len(rec["prediction"]))
            if task == "pert":
                for kind in PERT_KINDS:
                    m = score(mano, arrays[f"{seq}/{kind}/prediction"], arrays[f"{seq}/target"], beta, device)
                    for key, v in m.items():
                        arrays[f"{seq}/{kind}/{key}"] = v
                    blocks[seq][kind] = {**{k: float(v.mean()) for k, v in m.items()},
                                         "rho_mean": float(arrays[f"{seq}/{kind}/rho"].mean()),
                                         "rho_median": float(np.median(arrays[f"{seq}/{kind}/rho"]))}
            if task == "loop1":
                m_in = score(mano, arrays[f"{seq}/incoming"], arrays[f"{seq}/gt_previous"], beta, device)
                m_e3 = score(mano, arrays[f"{seq}/e3_prediction"], arrays[f"{seq}/target"], beta, device)
                blocks[seq]["incoming"] = {k: float(v.mean()) for k, v in m_in.items()}
                blocks[seq]["e3"] = {k: float(v.mean()) for k, v in m_e3.items()}
                for key, v in m_in.items():
                    arrays[f"{seq}/incoming_err/{key}"] = v
            if task == "f1":
                tgt = arrays[f"{seq}/target"]
                for start in ("gt", "mean", "loop"):
                    for k in F1_SAVE:
                        m = score(mano, arrays[f"{seq}/{start}_iter{k}"], tgt, beta, device)
                        blocks.setdefault(seq, {})[f"{start}_iter{k}"] = float(m["mpjpe_ra_mm"].mean())
                m0 = score(mano, np.concatenate([arrays[f"{seq}/gt_previous"][:, :6],
                                                 np.repeat(mu[None], len(tgt), 0)], 1), tgt, beta, device)
                blocks[seq]["mean_start"] = float(m0["mpjpe_ra_mm"].mean())
            save(out / "progress.json", {"done": seq, "at": now()})
    if task.startswith("cap_"):
        from semkine import pfo_diag as D
        caps = D.captured()
        for key in caps[0]:
            arrays[f"cap/{key}"] = torch.cat([c[key] for c in caps]).numpy().astype(np.float32)
    np.savez_compressed(out / "states.npz", **arrays)
    pooled = {}
    if all(s in blocks for s in ZGZ) and "mpjpe_ra_mm" in blocks[ZGZ[0]]:
        nn_ = sum(blocks[s]["n"] for s in ZGZ)
        pooled = {k: sum(blocks[s][k] * blocks[s]["n"] for s in ZGZ) / nn_
                  for k in ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm", "wrist_mm", "root_deg",
                            "finger_deg")}
    save(out / "report.json", {**launch, "status": "COMPLETE", "finished_utc": now(), "per_sequence": blocks,
                               "overall": pooled, "states_sha256": sha(out / "states.npz")}, exclusive=True)
    print("COMPLETE", out, {s: round(blocks[s].get("mpjpe_ra_mm", float("nan")), 3) for s in blocks}, flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--net", required=True)
    r.add_argument("--task", required=True, choices=TASKS)
    r.add_argument("--route")
    r.add_argument("--tau", type=float)
    r.add_argument("--intervene", default="")
    r.add_argument("--spec", help="JSON {kind, checkpoint, config} for a network outside common.NETS")
    r.add_argument("--limit", type=int, help="engineering smoke only: first endpoints, scratch directory")
    a = p.parse_args(argv)
    iv = tuple(x for x in a.intervene.split(",") if x)
    if not a.spec:
        require(a.net in NETS, "unknown network " + a.net)
    run(a.net, a.task, a.route, a.tau, iv, a.spec, a.limit)


if __name__ == "__main__":
    main()
