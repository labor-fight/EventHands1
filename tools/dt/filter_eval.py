#!/usr/bin/env python3
"""DT round, task filter-eval: constant-gain causal filtering of the render-and-compare tracker (zero training).

`semkine.anchored.FilteredTracker(trk, a_root, a_rest, a_trans)` feeds back `prev` moved toward the tracker's
own output by fixed gains (geodesic on the root rotation, linear on the 45 finger parameters and on the
translation); (1, 1, 1) is the raw tracker and an event-free packet holds `prev`. This file evaluates one
run's checkpoint under a list of gain triples with exactly the protocol of `tools/tracking/evalx.py eval`
(the loop is `evalx.run_sequence`, the summary `evalx.evaluate`; model, config, split and leak check are
built the way `evalx.cmd_eval` builds them) and writes one json (and the per-step arrays) per triple. Gain
triples are written ROOT,REST(fingers),TRANS. `--report` turns the jsons of several runs into the markdown /
json summary, every row with its delta versus (1, 1, 1).

Gains are fixed a priori and never chosen on the evaluation subject (AGENTS.md: zgz is both the development
and the test subject). Only the pre-registered primary (0.5, 1.0, 0.5) may be adopted; every other triple is
a descriptive sensitivity row.

    python tools/dt/filter_eval.py --run-dir outputs/semkine/rt_cnntrack_s3407 --ckpt last \\
        --gains '1,1,1;0.5,1,0.5' --controls --tf --perturb --out-dir outputs/dt/filter/rt_cnntrack_s3407
    python tools/dt/filter_eval.py --law-check --run-dir outputs/semkine/rt_cnntrack_s3407 --ckpt last \\
        --in-dir outputs/dt/filter/rt_cnntrack_s3407 --device cpu --threads 2
    python tools/dt/filter_eval.py --report --runs rt_cnntrack_s3407 rt_cnntrack_s3408 \\
        --in-dir outputs/dt/filter --report-out outputs/dt/reports/filter_cnntrack

`--law-check` (CPU, no GPU needed) replays the first step of the stored perturbation trials with the real network
and splits the first-step retention of a root perturbation into its parts: the filtered output changes by
g = (1 - a) e + a f (e the injected rotation, f the tracker's own output change; rotation vectors), so the retention
is |g| / |e|, which is bounded by (1 - a) + a |f| / |e| and equals it only when f is parallel to e. Writes
`law_check.json` next to the evaluations; `--report` includes it when present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "tracking")]
from config import load_config                                   # noqa: E402
from mano_layer import ManoLayer                                 # noqa: E402
from model import MNISTModel                                     # noqa: E402
from semkine.anchored import FilteredTracker                     # noqa: E402
from semkine.dataset import sequences_for_split                  # noqa: E402
import evalx as EX                                               # noqa: E402

BASE = (1.0, 1.0, 1.0)
PRIMARY = (0.5, 1.0, 0.5)                      # pre-registered
SENSITIVITY = ((0.5, 1.0, 1.0), (1.0, 1.0, 0.5), (0.5, 0.5, 0.5), (0.25, 1.0, 0.25), (0.75, 1.0, 0.75),
               (0.5, 0.75, 0.5))               # descriptive only
RECORDED_NAME = "evalx_val_core_last_tf_pert.json"
SANITY_TOL = 1e-6
SUPP_SUFFIX = "pert"                           # tag suffix of the supplementary perturbation-only evaluations
N_BOOT = 20000                                 # block-bootstrap resamples of the report's paired intervals
NEAR_ZERO = 0.02                               # an interval end this close to 0 is inside the Monte-Carlo error of the percentile: its SD over
                                               # 30 resampling seeds is 0.003-0.004 mm at 20000 resamples (0.008-0.013 at 2000) for the 14-block
                                               # RA differences of this report, and the seed 3408 primary's upper end moved between -0.009 and +0.005
LAW_CHECK_NAME = "law_check.json"
LAW_THETAS = (10.0, 20.0)                      # the two injected angles of `evalx.perturb_trials`


# ------------------------------------------------------------------------------------------ gains
def parse_gains(spec: str):
    """'ROOT,REST,TRANS;ROOT,REST,TRANS;...' -> [(root, rest, trans), ...]; every gain in [0, 1]."""
    out = []
    for part in str(spec).split(";"):
        part = part.strip()
        if not part:
            continue
        vals = [v.strip() for v in part.split(",")]
        if len(vals) != 3:
            raise ValueError(f"gain triple {part!r}: expected ROOT,REST,TRANS")
        g = tuple(float(v) for v in vals)
        if not all(0.0 <= v <= 1.0 for v in g):
            raise ValueError(f"gain triple {part!r}: every gain must lie in [0, 1]")
        if g in out:
            raise ValueError(f"gain triple {part!r} given twice")
        out.append(g)
    if not out:
        raise ValueError("no gain triple given")
    return out


def gain_tag(g) -> str:
    """(0.5, 1.0, 0.5) -> 'r0.5_f1.0_t0.5'."""
    return f"r{float(g[0])}_f{float(g[1])}_t{float(g[2])}"


def md5_of(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------ evaluation
def build_run(run: Path, ckpt: str, device, config=None):
    """The model, config and checkpoint of one run, as `evalx.cmd_eval` builds them."""
    cfg_path = config or next(iter(sorted(run.glob("*.yaml"))), None) or json.loads(
        (run / "training_metadata.json").read_text())["config_path"]
    cfg = load_config(cfg_path)
    path, step, sel_ra = EX.find_ckpt(run, ckpt)
    model = MNISTModel.load_from_checkpoint(str(path), cfg=cfg, map_location=device).to(device).eval()
    return cfg, model, path, step, sel_ra


def eval_args(controls=False, tf=False, perturb=False):
    """The fields `evalx.evaluate` reads from its `a`: the fixed protocol (50 ms windows) plus the modes."""
    return argparse.Namespace(controls=bool(controls), tf=bool(tf), perturb=bool(perturb), window_mode="fixed",
                              window_ms=EX.STEP, min_events=0, max_window_ms=300)


def _write_atomic(path: Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def cmd_eval(a):
    if a.threads:
        torch.set_num_threads(a.threads)
    run = Path(a.run_dir)
    out_dir = Path(a.out_dir)
    gains = parse_gains(a.gains) if a.gains else []
    if not gains and not a.raw:
        raise SystemExit("nothing to do: give --gains and/or --raw")
    device = torch.device(a.device) if a.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, model, ckpt, step, sel_ra = build_run(run, a.ckpt, device, a.config)
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    mani = Path(a.manifest) if a.manifest else Path(cfg["DATA"]["SPLITS_MANIFEST"])
    seqs = sequences_for_split(root, a.split, mani)
    trained = json.loads((run / "training_metadata.json").read_text()).get("train_sequences", [])
    leak = {s.split("_")[0] for s, _ in seqs} & {s.split("_")[0] for s in trained}
    assert not leak, f"{a.split} subjects {sorted(leak)} are in {run.name}'s training set"
    ns = eval_args(a.controls, a.tf, a.perturb)
    env = {"torch": torch.__version__, "device": str(device),
           "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "md5": {p: md5_of(REPO / p) for p in ("semkine/anchored.py", "tools/tracking/evalx.py", "model/model.py")}}
    jobs = ([("raw" + (f"_{a.suffix}" if a.suffix else ""), None)] if a.raw else []) + [
        (gain_tag(g) + (f"_{a.suffix}" if a.suffix else ""), g) for g in gains]
    for tag, g in jobs:
        net = model if g is None else FilteredTracker(model, *g).to(device).eval()
        label = f"{run.name} step={step} {tag}"
        t0 = time.time()
        summary, arrays = EX.evaluate(net, cfg, mano, root, seqs, device, ns, label=label)
        out = {"run": run.name, "ckpt": str(ckpt), "step": step, "split": a.split, "manifest": str(mani),
               "filter": None if g is None else {"class": "semkine.anchored.FilteredTracker", "tag": tag,
                                                  "a_root": g[0], "a_rest": g[1], "a_trans": g[2]},
               "gains": None if g is None else {"root": g[0], "rest": g[1], "trans": g[2]},
               "flags": {"controls": ns.controls, "tf": ns.tf, "perturb": ns.perturb}}
        out.update(summary)
        out["window"] = {"mode": ns.window_mode, "ms": ns.window_ms, "min_events": ns.min_events,
                         "max_ms": ns.max_window_ms}
        out["wall_s"] = time.time() - t0
        out["env"] = env
        out_dir.mkdir(parents=True, exist_ok=True)
        if not a.no_npz:
            np.savez_compressed(out_dir / f"{tag}.npz", **arrays)
        _write_atomic(out_dir / f"{tag}.json", json.dumps(out, indent=1))
        print(f"wrote {out_dir / (tag + '.json')}  ({out['wall_s']:.0f} s)", flush=True)


# ------------------------------------------------------------------------------------------ report
def seq_names(m: dict):
    """Sequence entries of a `summary[mode]` dict (the keys holding per-sequence statistics)."""
    return [k for k, v in m.items() if isinstance(v, dict) and "n_frames" in v]


def summarize(j: dict) -> dict:
    """The numbers the report tabulates, from one evaluation json: accuracy (overall / global / local), motion
    ratios, failures and the jitter decomposition (`model.jitter`). global = sequences without `_local`."""
    m = j["model"]
    seqs = seq_names(m)
    loc = [s for s in seqs if "_local" in s]
    glo = [s for s in seqs if "_local" not in s]

    def wm(ss, fn):
        n = sum(m[s]["n_frames"] for s in ss)
        return float(sum(fn(m[s]) * m[s]["n_frames"] for s in ss) / n) if n else float("nan")

    def three(fn, overall=None):
        return {"overall": float(overall) if overall is not None else wm(seqs, fn),
                "global": wm(glo, fn), "local": wm(loc, fn)}

    def acc(key):
        return three(lambda e: e[key][0], m["overall"][key])

    s = {"n_frames": int(m["n_frames"]),
         "ra_mm": acc("mpjpe_ra_mm"), "mpvpe_ra_mm": acc("mpvpe_ra_mm"), "root_rot_deg": acc("root_rot_deg"),
         "abs_mpjpe_mm": acc("mpjpe_abs_mm"), "transl_mm": acc("transl_mm"),
         "root_speed_ratio": three(lambda e: e["motion"]["root_speed_ratio"]),
         "finger_speed_ratio": three(lambda e: e["motion"]["finger_speed_ratio"]),
         "failures": {"episodes": int(sum(m[q]["failure"]["episodes"] for q in seqs)),
                      "bad_steps": int(sum(round(m[q]["failure"]["bad_step_frac"] * m[q]["n_frames"]) for q in seqs)),
                      "bad_step_frac": wm(seqs, lambda e: e["failure"]["bad_step_frac"]),
                      "fail_time_frac": wm(seqs, lambda e: e["failure"]["fail_time_frac"]),
                      "longest_s": float(max(m[q]["failure"]["longest_s"] for q in seqs))}}
    if "jitter" in m:
        s["jitter"] = {k: float(v) for k, v in m["jitter"].items()}
    return s


def pooled_by_events(j: dict) -> dict:
    """Event-count buckets (events per 50 ms packet) pooled over the sequences, step-weighted. A sequence
    contributes a bucket only when it holds >= 20 steps of it (`evalx.evaluate`)."""
    m, acc = j["model"], {}
    for s in seq_names(m):
        for b, v in m[s].get("by_events", {}).items():
            o = acc.setdefault(b, {"n": 0, "ra": 0.0, "rot": 0.0})
            o["n"] += v["n"]
            o["ra"] += v["n"] * v["mpjpe_ra_mm"]
            o["rot"] += v["n"] * v["root_rot_deg"]
    return {b: {"n": o["n"], "mpjpe_ra_mm": o["ra"] / o["n"], "root_rot_deg": o["rot"] / o["n"]} for b, o in acc.items()}


def tree_delta(cur, ref):
    """Same-shaped nested dict of `cur - ref` over the numeric leaves present in both."""
    if isinstance(cur, dict) and isinstance(ref, dict):
        return {k: tree_delta(cur[k], ref[k]) for k in cur if k in ref}
    return float(cur) - float(ref)


def tree_mean(trees):
    """Leaf-wise mean of same-shaped nested dicts (keys common to all)."""
    t0 = trees[0]
    if isinstance(t0, dict):
        return {k: tree_mean([t[k] for t in trees]) for k in t0 if all(k in t for t in trees)}
    return float(np.mean([float(t) for t in trees]))


def leaf_diffs(a, b, path=""):
    """`(path, |a - b|)` for every numeric leaf present in both nested structures; a structural mismatch
    (list lengths, None versus a number) gives `inf`. Strings and keys present in only one side are skipped."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k in b:
                yield from leaf_diffs(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            yield path, float("inf")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                yield from leaf_diffs(x, y, f"{path}[{i}]")
    elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str):
        return
    elif a is None and b is None:
        yield path, 0.0
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        yield path, (0.0 if (math.isnan(a) and math.isnan(b)) else abs(float(a) - float(b)))
    elif (a is None) != (b is None):
        yield path, float("inf")


def compare_runs(new: dict, rec: dict) -> dict:
    """Largest absolute difference over every numeric leaf two evaluation jsons share."""
    d = list(leaf_diffs(rec, new))
    worst = max(d, key=lambda t: t[1]) if d else ("", float("nan"))
    return {"n_leaves": len(d), "max_abs_diff": float(worst[1]), "worst_path": worst[0],
            "n_exact": int(sum(1 for _, v in d if v == 0.0))}


def compare_npz(new: Path, rec: Path) -> dict:
    """Bitwise comparison of the per-step arrays (names present in both) of two evaluations."""
    zn, zr = np.load(new), np.load(rec)
    common = sorted(set(zn.files) & set(zr.files))
    same = [k for k in common if zn[k].shape == zr[k].shape and np.array_equal(zn[k], zr[k])]
    return {"n_arrays_compared": len(common), "n_arrays_bitwise_identical": len(same),
            "differing": [k for k in common if k not in same][:10]}


def paired_block_ci(delta, blocks, n_boot=2000, seed=0):
    """Mean of `delta` and its 95% percentile interval, resampling blocks (10 s, `evalx.block_ci`) with
    replacement; `delta` is the per-step difference of two evaluations of the same steps."""
    ub, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=delta, minlength=len(ub))
    cnts = np.bincount(inv, minlength=len(ub)).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(ub), (n_boot, len(ub)))
    bs = sums[idx].sum(1) / cnts[idx].sum(1)
    return [float(delta.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def block_signs(delta, blocks):
    """(number of blocks, number of blocks whose mean `delta` is negative): the assumption-light companion of
    the bootstrap interval (how many of the 10 s blocks improve)."""
    ub, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=delta, minlength=len(ub))
    cnts = np.bincount(inv, minlength=len(ub)).astype(float)
    return int(len(ub)), int(np.sum(sums / cnts < 0.0))


def sign_tail(k, n):
    """One-sided tail `P(X >= k)`, `X ~ Binomial(n, 1/2)`: how often a fair coin gives at least `k` improving blocks of `n`
    (a rough guide: the blocks are not independent)."""
    return float(sum(math.comb(int(n), i) for i in range(int(k), int(n) + 1)) / 2.0 ** int(n))


def interval_vs_zero(lo, hi, near=NEAR_ZERO):
    """Where an interval sits relative to 0: 'covers 0'; 'touches 0' when it excludes 0 but an end lies within `near`
    of it (inside the Monte-Carlo error of a bootstrap percentile, so it must not be read as excluding 0);
    otherwise 'excludes 0'."""
    if lo <= 0.0 <= hi:
        return "covers 0"
    return "touches 0" if min(abs(lo), abs(hi)) < near else "excludes 0"


def step_series(z, seqs, key):
    """Per-step `key` of mode `model` concatenated over `seqs`, and the block id of every step."""
    vals, blocks = [], []
    for i, s in enumerate(seqs):
        vals.append(np.asarray(z[f"model|{s}|{key}"], dtype=np.float64))
        blocks.append(i * 10**6 + np.asarray(z[f"model|{s}|end"]) // EX.BLOCK_MS)
    return np.concatenate(vals), np.concatenate(blocks)


def paired_cis(npz_new: Path, npz_ref: Path, seqs, keys=("mpjpe_ra_mm", "root_rot_deg"), n_boot=N_BOOT):
    """`{key: [mean, lo, hi]}` of the per-step difference new - ref plus `blocks`: the number of 10 s blocks and, per
    key, how many of them have a negative mean difference."""
    if not (Path(npz_new).exists() and Path(npz_ref).exists()):
        return None
    zn, zr = np.load(npz_new), np.load(npz_ref)
    out, neg, n_blocks = {}, {}, 0
    for key in keys:
        vn, bn = step_series(zn, seqs, key)
        vr, br = step_series(zr, seqs, key)
        assert np.array_equal(bn, br), "the two evaluations do not cover the same steps"
        out[key] = paired_block_ci(vn - vr, bn, n_boot=n_boot)
        n_blocks, neg[key] = block_signs(vn - vr, bn)
    out["blocks"] = {"n": n_blocks, "negative": neg, "n_boot": int(n_boot)}
    return out


def modes_summary(j: dict) -> dict:
    out = {}
    for mode in ("hold", "noevents", "tf"):
        if mode in j:
            ov = j[mode]["overall"]
            out[mode] = {k: float(ov[k]) for k in ("mpjpe_ra_mm", "root_rot_deg", "mpjpe_abs_mm", "transl_mm")}
    if "amplification" in j:
        out["amplification"] = {k: float(v) for k, v in j["amplification"].items()}
    return out


def perturb_summary(j: dict):
    if "perturb" not in j:
        return None
    return {th: {"trials": int(v["trials"]), "retention_k1_k2_k5_k10_k20": [float(x) for x in v["retention_k1_k2_k5_k10_k20"]],
                 "excess_k1_k2_k5_k10_k20_deg": [float(x) for x in v["excess_k1_k2_k5_k10_k20_deg"]],
                 "half_life_steps": v["half_life_steps"], "div_deg": [float(x) for x in v["div_deg"]]}
            for th, v in j["perturb"].items()}


def magnitude_bound(rho, a):
    """`(1 - a) + a rho`: the first-step retention if the filter added MAGNITUDES. The filter adds rotation vectors,
    so this is an upper bound of the retention (triangle inequality), reached only when the tracker's response is
    parallel to the injected rotation; for a response-free tracker (rho = 0) it is exactly `1 - a`."""
    return (1.0 - a) + a * rho


def filter_law(pert_raw: dict, pert_f: dict, a_root: float) -> dict:
    """First-step retention of a root perturbation of the filtered loop against the magnitude bound, from the
    evaluations alone. With `rho` the raw tracker's first-step retention (its response |f| / theta; taken from
    the (1,1,1) run of the same seed, so it is an approximation of the response inside the filtered loop, whose
    trajectory differs) the retention is at most `(1 - a) + a rho` and, for a tracker that does not oppose the
    perturbation, at least `1 - a`. `gap_measured_minus_bound` is therefore expected to be <= 0; its size is
    set by how far the response is from parallel to the injected rotation (see `law_check`). It is NOT an
    estimate of the tracker's response."""
    out = {}
    for th in pert_f:
        if th not in pert_raw:
            continue
        rho = pert_raw[th]["retention_k1_k2_k5_k10_k20"][0]
        meas = pert_f[th]["retention_k1_k2_k5_k10_k20"][0]
        bound = magnitude_bound(rho, a_root)
        out[th] = {"a_root": a_root, "k1_raw_rho": rho, "k1_measured": meas, "filter_alone_1_minus_a": 1.0 - a_root,
                   "k1_magnitude_bound": bound, "gap_measured_minus_bound": meas - bound}
    return out


# ---------------------------------------------------------------------- first-step law check (CPU replay)
def rotvec_change(rv_new, rv_old):
    """Rotation vector of `R(new) R(old)^-1` (float64): the rotation that carries `old` to `new`, about an axis fixed in
    the camera frame (left composition, the convention of the injected perturbation in `evalx.perturb_trials`)."""
    return (Rot.from_rotvec(np.asarray(rv_new, np.float64)) * Rot.from_rotvec(np.asarray(rv_old, np.float64)).inv()).as_rotvec()


def vector_law(e, f, a):
    """First-order retention `|(1 - a) e + a f| / |e|` of the first fed-back output. `e` is the injected rotation
    vector, `f` the change of the tracker's own output it causes and `a` the root gain: the filter moves `prev`
    toward the tracker output by `a`, so the fed-back output changes by `g = (1 - a) e + a f`. It never exceeds
    `magnitude_bound(|f| / |e|, a)` (triangle inequality; equality iff f is parallel to e, or a is 0 or 1)."""
    e, f = np.asarray(e, np.float64), np.asarray(f, np.float64)
    return float(np.linalg.norm((1.0 - a) * e + a * f) / np.linalg.norm(e))


def trial_plan(run, elapsed, horizon=20, every=20, min_elapsed=1000):
    """The steps `evalx.perturb_trials` branches a perturbed state off at, from a stored evaluation's `run` (segment
    id) and `elapsed` (ms since the segment start) arrays: every `every`-th step of a segment once it is
    `min_elapsed` ms old, provided `horizon` steps remain in the segment."""
    run, elapsed = np.asarray(run), np.asarray(elapsed)
    order = np.arange(len(run))
    out = []
    for rid in np.unique(run):
        idx = order[run == rid]
        for i0 in idx[elapsed[idx] >= min_elapsed][::every]:
            if i0 + horizon > idx[-1] + 1:
                continue
            out.append(int(i0))
    return out


def trial_axes(n, seed=0):
    """The `n` random unit axes `evalx.perturb_trials` draws for one sequence (one `standard_normal(3)` per trial,
    from `default_rng(seed)`; both injected angles share the axis)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ax = rng.standard_normal(3)
        out.append(ax / np.linalg.norm(ax))
    return np.stack(out) if out else np.zeros((0, 3))


class _Tap(nn.Module):
    """Pass-through around the tracker that keeps its last output: the law check needs the tracker's own
    (unfiltered) response next to the filtered output."""

    encoder_name = ""

    def __init__(self, trk):
        super().__init__()
        self.trk = trk
        self.last = None

    def set_hand_context(self, betas, camera_K):
        if hasattr(self.trk, "set_hand_context"):
            self.trk.set_hand_context(betas, camera_K)

    def forward(self, x, prevpos, betas=None, camera_K=None):
        self.last = self.trk(x, prevpos, betas=betas, camera_K=camera_K)
        return self.last


@torch.no_grad()
def law_trial(net, tap, x, prev, es):
    """One packet `x`, the state `prev` (float32 (51,)) fed in, and for every rotation vector in `es` the state with its
    root rotated by it (`Exp(e) R_prev`, as `evalx.perturb_trials`). `net` is the model under test (the filtered
    tracker, or `tap` itself for the raw tracker) built on `tap`. Returns the unperturbed output (51,) and, per `e`,
    the perturbed output `out1`, the change `g` of the fed-back output root and the change `f` of the tracker's own
    output root (rotation vectors)."""
    prev = np.ascontiguousarray(prev, dtype=np.float32)
    out0 = net(x, torch.from_numpy(prev).view(1, -1))[0].numpy().copy()
    o0 = tap.last[0].numpy().copy()
    res = []
    for e in es:
        pp = prev.copy()
        pp[3:6] = EX.aa_compose(np.asarray(e, np.float64), prev[3:6])
        out1 = net(x, torch.from_numpy(pp).view(1, -1))[0].numpy().copy()
        o1 = tap.last[0].numpy().copy()
        res.append({"out1": out1, "g": rotvec_change(out1[3:6], out0[3:6]), "f": rotvec_change(o1[3:6], o0[3:6])})
    return out0, res


def aggregate_law(rows, a):
    """Means over trials of the first-step decomposition. Each row has `e`, `f`, `g` (rotation vectors) and `held`
    (the packet was empty, so the filter returned `prev`: the fed-back change is `e` itself, which is the same as a
    response `f = e`; it is counted as such). Returns the retention measured as |g| / |e| (`k1_cpu_pair`), the
    tracker's response `rho_f = |f| / |e|`, the cosine of the angle between f and e, the vector law
    `|(1 - a) e + a f| / |e|`, the magnitude bound with this loop's own `rho_f`, and how far the vector law is from
    the measurement trial by trial."""
    e = np.array([r["e"] for r in rows], np.float64)
    f = np.array([r["e"] if r.get("held") else r["f"] for r in rows], np.float64)
    g = np.array([r["g"] for r in rows], np.float64)
    ne, nf, ng = (np.linalg.norm(v, axis=1) for v in (e, f, g))
    cos = np.einsum("ij,ij->i", e, f) / np.maximum(ne * nf, 1e-30)
    vec = np.linalg.norm((1.0 - a) * e + a * f, axis=1) / ne
    rho = nf / ne
    return {"n_trials": int(len(rows)), "n_held": int(sum(1 for r in rows if r.get("held"))),
            "k1_cpu_pair": float(np.mean(ng / ne)), "rho_f": float(rho.mean()),
            "cos_f_e_mean": float(cos.mean()), "cos_f_e_median": float(np.median(cos)),
            "k1_vector_law": float(vec.mean()),
            "vector_law_max_abs_trial_diff": float(np.max(np.abs(vec - ng / ne))),
            "k1_magnitude_bound_rho_f": float(magnitude_bound(rho.mean(), a))}


def discover_perturb_npz(in_dir: Path, tag: str):
    """The stored per-step arrays of `tag` that hold the perturbation trials: the evaluation itself when it ran with
    `--perturb`, else its supplementary perturbation-only evaluation (`<tag>_pert`); None when neither exists."""
    for name in (tag, f"{tag}_{SUPP_SUFFIX}"):
        p = Path(in_dir) / f"{name}.npz"
        if p.exists():
            with np.load(p) as z:
                if any(k.startswith("perturb|") for k in z.files):
                    return p
    return None


def law_check_tag(model, gains, cfg, root, seqs, npz_path, thetas=LAW_THETAS):
    """CPU replay of the first step of every stored perturbation trial of one evaluation (`npz_path`): the state fed
    into the branched step is the stored loop's, the packet and the random axes are `evalx.perturb_trials`'s.
    `gains` None is the unwrapped tracker. Returns, per theta, `aggregate_law` plus the comparison with the stored
    GPU retention (`k1_stored_gpu`; `k1_cpu_vs_stored_pred` = the CPU perturbed output against the STORED
    unperturbed output, i.e. evalx's own definition with the CPU network)."""
    z = np.load(npz_path)
    tap = _Tap(model).eval()
    net = tap if gains is None else FilteredTracker(tap, *gains).eval()
    a_root = 1.0 if gains is None else float(gains[0])
    ev_ch = EX.EV.event_channels(cfg)
    rows = {float(t): [] for t in thetas}
    stored_k1 = {float(t): [] for t in thetas}
    vs_pred = {float(t): [] for t in thetas}
    for s, d in seqs:
        events, offsets, aux, _ = EX.ET.load_sequence(root, d, s)
        net.set_hand_context(torch.tensor(aux["betas"], dtype=torch.float32).view(1, -1),
                             torch.tensor(aux["camera_K"], dtype=torch.float32).view(1, 3, 3))
        pred, run_id, end, elapsed = (z[f"model|{s}|{k}"] for k in ("pred", "run", "end", "elapsed"))
        plan = trial_plan(run_id, elapsed)
        axes = trial_axes(len(plan))
        for th in rows:
            key = f"perturb|{s}|{th:g}|div"
            assert len(z[key]) == len(plan), f"{npz_path.name} {s}: {len(z[key])} stored trials, plan has {len(plan)}"
            stored_k1[th] += list(z[key][:, 0] / th)
        for j, i0 in enumerate(plan):
            assert run_id[i0 - 1] == run_id[i0], "a branched step must have a predecessor in its segment"
            x = torch.from_numpy(EX.ET.build_lnes(events, offsets, int(end[i0]), EX.STEP, ev_ch)).unsqueeze(0)
            es = [axes[j] * np.deg2rad(th) for th in rows]
            _, res = law_trial(net, tap, x, pred[i0 - 1], es)
            held = gains is not None and not bool((x.abs().sum() > 0).item())
            for (th, lst), e, r in zip(rows.items(), es, res):
                lst.append({"e": e, "f": r["f"], "g": r["g"], "held": held})
                vs_pred[th].append(float(EX.rot_err_deg(r["out1"][None], pred[i0][None])[0]) / th)
    out = {}
    for th, lst in rows.items():
        agg = aggregate_law(lst, a_root)
        agg["k1_stored_gpu"] = float(np.mean(stored_k1[th]))
        agg["k1_cpu_vs_stored_pred"] = float(np.mean(vs_pred[th]))
        agg["max_abs_trial_diff_vs_stored"] = float(np.max(np.abs(np.array(vs_pred[th]) - np.array(stored_k1[th]))))
        out[f"{th:g}"] = agg
    return out


def cmd_law_check(a):
    if a.threads:
        torch.set_num_threads(a.threads)
    run, in_dir = Path(a.run_dir), Path(a.in_dir)
    device = torch.device(a.device or "cpu")
    if device.type != "cpu":
        raise SystemExit("--law-check replays on the CPU (a few thousand batch-1 passes; the inputs are built on the host): use --device cpu")
    cfg, model, ckpt, step, _ = build_run(run, a.ckpt, device, a.config)
    root = Path(cfg["DATA"]["ROOT"])
    mani = Path(a.manifest) if a.manifest else Path(cfg["DATA"]["SPLITS_MANIFEST"])
    seqs = sequences_for_split(root, a.split, mani)
    tags = [t for t in a.tags.split(",") if t] if a.tags else sorted(
        p.stem for p in in_dir.glob("*.json") if not p.stem.endswith(f"_{SUPP_SUFFIX}") and p.stem != Path(LAW_CHECK_NAME).stem)
    out = {"run": run.name, "ckpt": str(ckpt), "step": step, "split": a.split, "device": str(device),
           "threads": torch.get_num_threads(), "torch": torch.__version__, "generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "md5": {p: md5_of(REPO / p) for p in ("semkine/anchored.py", "tools/tracking/evalx.py", "model/model.py")},
           "method": "teacher-forced replay of the FIRST step of the stored perturbation trials (same steps and random axes as "
                     "evalx.perturb_trials; the state fed in is the stored unperturbed loop's), fp32 batch 1, rotations as "
                     "left rotation vectors", "tags": {}}
    for tag in tags:
        npz = discover_perturb_npz(in_dir, tag)
        j = json.loads((in_dir / f"{tag}.json").read_text()) if (in_dir / f"{tag}.json").exists() else None
        if npz is None or j is None:
            print(f"skip {tag}: no stored perturbation trials", flush=True)
            continue
        assert int(j["step"]) == int(step), f"{tag}: evaluated at step {j['step']}, checkpoint is step {step}"
        g = j.get("gains")
        gains = None if g is None else (g["root"], g["rest"], g["trans"])
        t0 = time.time()
        out["tags"][tag] = {"gains": None if gains is None else list(gains), "npz": npz.name,
                            "per_theta": law_check_tag(model, gains, cfg, root, seqs, npz)}
        print(f"law-check {run.name} {tag}: {time.time() - t0:.0f} s  "
              + "  ".join(f"theta {th}: stored {v['k1_stored_gpu']:.4f} replay {v['k1_cpu_vs_stored_pred']:.4f} "
                          f"vector {v['k1_vector_law']:.4f}" for th, v in out["tags"][tag]["per_theta"].items()), flush=True)
    path = Path(a.law_out) if a.law_out else in_dir / LAW_CHECK_NAME
    _write_atomic(path, json.dumps(out, indent=1))
    print("wrote", path, flush=True)


def _seed_of(run: str):
    mm = re.search(r"_s(\d+)$", run)
    return mm.group(1) if mm else run


def load_eval(in_dir: Path, run: str, tag: str):
    p = Path(in_dir) / run / f"{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def build_report(runs, in_dir, gains, base, primary, semkine_dir):
    """All numbers of the report as one json-able dict (see `render_markdown` for the layout)."""
    in_dir, semkine_dir = Path(in_dir), Path(semkine_dir)
    tags = {g: gain_tag(g) for g in gains}
    rep = {"task": "filter-eval", "model": "rt_cnntrack (ResNet18 / LNES + rendered previous state, delta tracker)",
           "protocol": "evalx.evaluate loop: batch 1, fp32, init = GT + protocol noise (rng seed 0 across the split's "
                       "sequences in order), 50 ms steps, split val_core, checkpoint last",
           "generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "gains_order": "ROOT,REST(fingers),TRANS", "base": list(base), "primary_preregistered": list(primary),
           "sensitivity_descriptive_only": [list(g) for g in gains if g not in (base, primary)],
           "rule": "Filter gains are fixed a priori and never chosen on the evaluation subject; the pre-registered "
                   "primary is (0.5, 1.0, 0.5); the sensitivity rows are descriptive only.",
           "runs": list(runs), "tags": {str(list(g)): t for g, t in tags.items()}, "missing": [],
           "sanity": {}, "per_seed": {}, "mean": {}, "ra_delta_ci": {}, "modes": {}, "perturb": {}, "perturb_source": {},
           "supp_consistency": {}, "filter_law": {}, "law_check": {}, "by_events": {}, "provenance": {},
           "code_now": {p: md5_of(REPO / p) for p in ("semkine/anchored.py", "tools/tracking/evalx.py", "model/model.py")}}
    ev = {}
    for run in runs:
        for g in gains:
            j = load_eval(in_dir, run, tags[g])
            if j is None:
                rep["missing"].append(f"{run}/{tags[g]}")
            else:
                ev[(run, g)] = j
    for run in runs:
        seed = _seed_of(run)
        rec_p = semkine_dir / run / RECORDED_NAME
        rec = json.loads(rec_p.read_text()) if rec_p.exists() else None
        san = {"recorded_json": str(rec_p), "tolerance": SANITY_TOL}
        for name, tag in (("filtered_1_1_1", tags[base]), ("raw_unwrapped", "raw")):
            jn = load_eval(in_dir, run, tag)
            if rec is None or jn is None:
                continue
            ra_new, ra_rec = jn["model"]["overall"]["mpjpe_ra_mm"], rec["model"]["overall"]["mpjpe_ra_mm"]
            san[name] = {"mpjpe_ra_mm_recorded": ra_rec, "mpjpe_ra_mm_reproduced": ra_new,
                         "abs_diff": abs(ra_new - ra_rec), "within_tol": bool(abs(ra_new - ra_rec) <= SANITY_TOL),
                         "all_shared_numeric_leaves": compare_runs(jn, rec),
                         "cuda_visible_devices": (jn.get("env") or {}).get("cuda_visible_devices")}
            npz_new, npz_rec = in_dir / run / f"{tag}.npz", rec_p.with_suffix(".npz")
            if npz_new.exists() and npz_rec.exists():
                san[name]["step_arrays"] = compare_npz(npz_new, npz_rec)
        rep["sanity"][seed] = san
        if (run, base) not in ev:
            continue
        sums = {g: summarize(ev[(run, g)]) for g in gains if (run, g) in ev}
        base_sum = sums[base]
        seqs = seq_names(ev[(run, base)]["model"])
        rep["per_seed"][seed] = {}
        rep["provenance"][seed] = {}
        for g, sm in sums.items():
            role = "base" if g == base else ("primary_preregistered" if g == primary else "sensitivity_descriptive")
            rep["per_seed"][seed][tags[g]] = {"gains": list(g), "role": role, "metrics": sm,
                                              "delta_vs_base": tree_delta(sm, base_sum),
                                              "wall_s": ev[(run, g)].get("wall_s")}
            env = ev[(run, g)].get("env") or {}
            rep["provenance"][seed][tags[g]] = {"cuda_visible_devices": env.get("cuda_visible_devices"),
                                                "gpu": env.get("gpu"), "md5": env.get("md5") or {},
                                                "flags": ev[(run, g)].get("flags")}
            if g != base:
                ci = paired_cis(in_dir / run / f"{tags[g]}.npz", in_dir / run / f"{tags[base]}.npz", seqs)
                if ci:
                    rep["ra_delta_ci"].setdefault(seed, {})[tags[g]] = ci
        rep["by_events"][seed] = {tags[g]: pooled_by_events(ev[(run, g)]) for g in sums}
        rep["modes"][seed] = {tags[g]: modes_summary(ev[(run, g)]) for g in sums if modes_summary(ev[(run, g)])}
        pr, src = {}, {}
        for g in sums:
            if "perturb" in ev[(run, g)]:
                pr[tags[g]], src[tags[g]] = perturb_summary(ev[(run, g)]), "main"
                continue
            js = load_eval(in_dir, run, f"{tags[g]}_{SUPP_SUFFIX}")
            if js is not None and "perturb" in js:
                pr[tags[g]], src[tags[g]] = perturb_summary(js), "supplementary"
                rep["supp_consistency"].setdefault(seed, {})[tags[g]] = compare_runs(js["model"], ev[(run, g)]["model"])
        rep["perturb"][seed], rep["perturb_source"][seed] = pr, src
        if tags[base] in pr:
            for g in sums:
                if g != base and tags[g] in pr:
                    rep["filter_law"].setdefault(seed, {})[tags[g]] = filter_law(pr[tags[base]], pr[tags[g]], g[0])
        lc_p = in_dir / run / LAW_CHECK_NAME
        if lc_p.exists():
            rep["law_check"][seed] = json.loads(lc_p.read_text())
    seeds = list(rep["per_seed"])
    if seeds:
        for g in gains:
            t = tags[g]
            rows = [rep["per_seed"][s][t] for s in seeds if t in rep["per_seed"][s]]
            if len(rows) == len(seeds):
                rep["mean"][t] = {"gains": list(g), "n_seeds": len(rows),
                                  "metrics": tree_mean([r["metrics"] for r in rows]),
                                  "delta_vs_base": tree_mean([r["delta_vs_base"] for r in rows])}
    return rep


# ---- markdown
def _f(v, nd):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def _cell(v, d, nd):
    return _f(v, nd) if d is None else f"{_f(v, nd)} ({d:+.{nd}f})"


def _gtxt(g):
    return "(" + ", ".join(f"{x:g}" for x in g) + ")"


def _role(rep, g):
    g = tuple(g)
    return "reference" if g == tuple(rep["base"]) else ("PRIMARY (pre-registered)" if g == tuple(rep["primary_preregistered"]) else "descriptive")


def _tables(rep, who, rows):
    """Accuracy + motion/jitter tables for one seed (or the mean)."""
    lines = []
    acc_h = ["gains (root, fingers, trans)", "role", "RA all", "RA global", "RA local", "root rot deg", "abs MPJPE mm",
             "transl mm", "fail ep / bad steps"]
    lines += [f"#### {who}: accuracy (value, delta vs (1,1,1))", "", "| " + " | ".join(acc_h) + " |",
              "|" + "---|" * len(acc_h)]
    for t, r in rows.items():
        m, d = r["metrics"], (None if r["gains"] == list(rep["base"]) else r["delta_vs_base"])
        g = lambda k, p, nd: _cell(m[k][p], None if d is None else d[k][p], nd)   # noqa: E731
        fails = m["failures"]
        ftxt = f"{fails['episodes']:g} / {fails['bad_steps']:g}"
        lines.append("| " + " | ".join([_gtxt(r["gains"]), _role(rep, r["gains"]), g("ra_mm", "overall", 3), g("ra_mm", "global", 3),
                                        g("ra_mm", "local", 3), g("root_rot_deg", "overall", 3),
                                        g("abs_mpjpe_mm", "overall", 2), g("transl_mm", "overall", 2), ftxt]) + " |")
    lines += ["", f"#### {who}: motion ratios and jitter (value, delta vs (1,1,1))", ""]
    jk = ["jit_pred_mm", "acc_err_mm", "acc_ratio", "acc_err_only_transl_mm", "acc_err_only_root_mm",
          "acc_err_only_fingers_mm", "rot_acc_pred_deg"]
    head = ["gains", "root speed ratio", "root speed ratio global / local", "finger speed ratio"] + jk
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for t, r in rows.items():
        m, d = r["metrics"], (None if r["gains"] == list(rep["base"]) else r["delta_vs_base"])
        cells = [_gtxt(r["gains"]), _cell(m["root_speed_ratio"]["overall"], None if d is None else d["root_speed_ratio"]["overall"], 3),
                 f"{m['root_speed_ratio']['global']:.3f} / {m['root_speed_ratio']['local']:.3f}",
                 _cell(m["finger_speed_ratio"]["overall"], None if d is None else d["finger_speed_ratio"]["overall"], 3)]
        for k in jk:
            jm = m.get("jitter", {})
            cells.append(_cell(jm[k], None if d is None else d["jitter"][k], 3) if k in jm else "n/a")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _headline(rep):
    t = gain_tag(rep["primary_preregistered"])
    seeds = list(rep["per_seed"])
    if not seeds or t not in rep["mean"]:
        return []
    mean, per = rep["mean"][t], [rep["per_seed"][s][t] for s in seeds]
    mm, dd = mean["metrics"], mean["delta_vs_base"]
    j = "jitter" in dd
    seed_ra = " / ".join(f"{p['delta_vs_base']['ra_mm']['overall']:+.3f}" for p in per)
    line = (f"Pre-registered primary {_gtxt(rep['primary_preregistered'])} against the raw tracker {_gtxt(rep['base'])} "
            f"(two-seed mean, seeds {' / '.join(seeds)}): RA {dd['ra_mm']['overall']:+.3f} mm (per seed {seed_ra}; "
            f"{mm['ra_mm']['overall']:.3f} vs {mm['ra_mm']['overall'] - dd['ra_mm']['overall']:.3f}), root rotation "
            f"{dd['root_rot_deg']['overall']:+.3f} deg, abs MPJPE {dd['abs_mpjpe_mm']['overall']:+.2f} mm, translation "
            f"{dd['transl_mm']['overall']:+.2f} mm")
    if j:
        line += (f"; jitter: jit_pred {dd['jitter']['jit_pred_mm']:+.3f} mm/step, acc_err {dd['jitter']['acc_err_mm']:+.3f} mm/step^2, "
                 f"acc_ratio {mm['jitter']['acc_ratio'] - dd['jitter']['acc_ratio']:.3f} -> {mm['jitter']['acc_ratio']:.3f}, "
                 f"rot_acc_pred {dd['jitter']['rot_acc_pred_deg']:+.3f} deg/step^2")
    line += (f"; root speed ratio {mm['root_speed_ratio']['overall'] - dd['root_speed_ratio']['overall']:.3f} -> "
             f"{mm['root_speed_ratio']['overall']:.3f} (ground truth = 1), finger speed ratio unchanged "
             f"({mm['finger_speed_ratio']['overall']:.3f}, a_rest = 1).")
    flags = []
    for s in seeds:
        c = rep.get("ra_delta_ci", {}).get(s, {}).get(t)
        if c:
            flags.append(f"seed {s} {interval_vs_zero(c['mpjpe_ra_mm'][1], c['mpjpe_ra_mm'][2])} "
                         f"({c['blocks']['negative']['mpjpe_ra_mm']} of {c['blocks']['n']} blocks lower)")
    if flags:
        return [line, "", "The RA gain against sampling noise (paired 10 s block bootstrap, section 1; one subject, evaluation sample only, retraining noise not "
                          "included): " + "; ".join(flags) + ".", ""]
    return [line, ""]


def _rng(vals, nd=3, sign=True):
    """'lo .. hi' of a list of numbers (one number when both ends print the same)."""
    vals = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    if not vals:
        return "n/a"
    f = (lambda v: f"{v:+.{nd}f}") if sign else (lambda v: f"{v:.{nd}f}")
    lo, hi = f(min(vals)), f(max(vals))
    return lo if lo == hi else f"{lo} .. {hi}"


def _law_rows(rep):
    """Flat `(seed, tag, gains, theta, row)` over `rep["filter_law"]`."""
    return [(seed, t, rep["per_seed"][seed][t]["gains"], th, v)
            for seed, d in rep["filter_law"].items() for t, per_th in d.items() for th, v in per_th.items()]


def _law_check_rows(rep):
    """Flat `(seed, tag, gains or None, theta, row)` over the CPU-replay law check, raw first and then in gain order."""
    order = ["raw"] + list(rep["tags"].values())
    out = []
    for seed, lc in (rep.get("law_check") or {}).items():
        for t in sorted(lc["tags"], key=lambda t: order.index(t) if t in order else len(order)):
            for th, v in lc["tags"][t]["per_theta"].items():
                out.append((seed, t, lc["tags"][t]["gains"], th, v))
    return out


def law_check_stats(rep):
    """The ranges quoted in the text of section 3b, computed from `rep["law_check"]` (never typed in)."""
    rows = _law_check_rows(rep)
    filt = [r for r in rows if r[2] is not None and r[2][0] < 1.0]
    raw = [r for r in rows if r[1] == "raw"]
    raw_rho = {(r[0], r[3]): r[4]["rho_f"] for r in raw}
    d_rho = [r[4]["rho_f"] - raw_rho[(r[0], r[3])] for r in filt if (r[0], r[3]) in raw_rho]
    return {"n_rows": len(rows), "n_filtered": len(filt),
            "rho_f_minus_raw": d_rho,
            "rho_f_relative_to_raw": [(r[4]["rho_f"] - raw_rho[(r[0], r[3])]) / raw_rho[(r[0], r[3])] for r in filt if (r[0], r[3]) in raw_rho],
            "replay_vs_stored_mean": max((abs(r[4]["k1_cpu_vs_stored_pred"] - r[4]["k1_stored_gpu"]) for r in rows), default=float("nan")),
            "replay_vs_stored_trial": max((r[4]["max_abs_trial_diff_vs_stored"] for r in rows), default=float("nan")),
            "vector_vs_pair_mean": max((abs(r[4]["k1_vector_law"] - r[4]["k1_cpu_pair"]) for r in filt), default=float("nan")),
            "vector_vs_pair_trial": max((r[4]["vector_law_max_abs_trial_diff"] for r in filt), default=float("nan")),
            "rho_f_filtered": [r[4]["rho_f"] for r in filt], "rho_f_raw": [r[4]["rho_f"] for r in raw],
            "cos_filtered": [r[4]["cos_f_e_mean"] for r in filt],
            "bound_minus_measured": [r[4]["k1_magnitude_bound_rho_f"] - r[4]["k1_cpu_pair"] for r in filt],
            "n_bound_holds": sum(1 for r in filt if r[4]["k1_magnitude_bound_rho_f"] >= r[4]["k1_cpu_pair"] - 1e-9),
            "n_held": sum(r[4]["n_held"] for r in rows)}


def _law_markdown(rep):
    """Sections 3a-3c: the filter law as it holds, the bound against the stored retention, the direct CPU-replay
    measurement and the primary against the response-free law."""
    rows = _law_rows(rep)
    rho_all = [r[4]["k1_raw_rho"] for r in rows]
    L = ["### 3a. What the filter law says", "",
         "Let e be the injected rotation (angle theta) and f the change of the tracker's own output that it causes (both rotation vectors in the "
         "camera frame; the raw tracker's k1 is rho = |f| / theta). The filter moves `prev` toward the tracker output by a_root, so, to first order, "
         "the fed-back output changes by g = (1 - a_root) e + a_root f and the first-step retention is |g| / theta = |(1 - a_root) e + a_root f| / theta. "
         "Rotation vectors add as vectors, not as magnitudes, hence |g| / theta <= (1 - a_root) + a_root * rho (triangle inequality), with equality only "
         "when f is parallel to e. Consequences:", "",
         "- A tracker with no response (f = 0) keeps exactly 1 - a_root at step 1 and (1 - a_root)^k at step k. This is the debug-gate law of the "
         "pre-registration (`docs/DT_RENDER_TRACK_PREREG.md` section 3, 'retention follows (1 - g)^k'). It is a property of the filter module and is "
         "tested as such with a response-free stand-in (`tests/test_dt_filter.py`).",
         "- A tracker whose response is parallel to the perturbation with strength rho gives ((1 - a_root) + a_root * rho)^k (also tested).",
         f"- The real tracker responds (rho = {_rng(rho_all, 2, False)} here) and its response is not parallel to the perturbation, so the measured k1 "
         "lies below the magnitude bound, not on it. The table below is that bound with rho taken from the (1,1,1) run of the same seed; its last column "
         "(measured - bound) is a gap to an upper bound and is NOT an estimate of the tracker's response (an earlier version of this report read it as "
         "one; that reading was wrong)." + (" The response inside the filtered loop is measured directly in 3b." if rep.get("law_check") else ""), ""]
    if rows:
        L += ["| seed | gains | theta | a_root | 1 - a_root (f = 0) | rho (raw k1) | magnitude bound (1-a)+a*rho | measured k1 | measured - bound |",
              "|---|---|---|---|---|---|---|---|---|"]
        for seed, t, g, th, v in rows:
            L.append(f"| {seed} | {_gtxt(g)} | {th} deg | {v['a_root']:g} | {v['filter_alone_1_minus_a']:.3f} | {v['k1_raw_rho']:.3f} | "
                     f"{v['k1_magnitude_bound']:.3f} | {v['k1_measured']:.3f} | {v['gap_measured_minus_bound']:+.3f} |")
        L.append("")
        filt = [r for r in rows if r[4]["a_root"] < 1.0]
        free = [r for r in rows if r[4]["a_root"] >= 1.0]
        prim = [r for r in filt if tuple(r[2]) == tuple(rep["primary_preregistered"])]
        between = sum(1 for r in filt if r[4]["filter_alone_1_minus_a"] <= r[4]["k1_measured"] <= r[4]["k1_magnitude_bound"])
        txt = (f"Reading: in {between} of the {len(filt)} rows with a root filter (a_root < 1) the measured k1 lies between 1 - a_root and the bound; the gap to "
               f"the bound is {_rng([r[4]['gap_measured_minus_bound'] for r in filt])} over these rows")
        if prim:
            txt += f" ({_rng([r[4]['gap_measured_minus_bound'] for r in prim])} for the primary)"
        txt += "."
        if free:
            txt += (f" The {len(free)} rows with a_root = 1 (translation filter only) have no root filter: the bound is rho itself and their gap "
                    f"({_rng([r[4]['gap_measured_minus_bound'] for r in free])}) is only the difference between the tracker's first-step response along two different "
                    "trajectories, i.e. the noise floor of this comparison; gaps of that size are not meaningful.")
        L += [txt, ""]
    lc = _law_check_rows(rep)
    if lc:
        st = law_check_stats(rep)
        L += ["### 3b. Direct measurement of the first step (CPU replay)", "",
              "`tools/dt/filter_eval.py --law-check`: every stored perturbation trial (same steps and random axes as `evalx.perturb_trials`) is replayed for its "
              "first step on CPU (fp32, batch 1), from the stored unperturbed loop's state, once unperturbed and once with the root rotated by e (10 and 20 deg), "
              "with the real network inside the real `FilteredTracker`. Per trial: g = change of the fed-back output root, f = change of the tracker's own "
              "output root (left rotation vectors). `k1 stored` is the section-3 number (GPU); `k1 replay` is evalx's own definition computed with the CPU "
              "network (perturbed CPU output against the stored unperturbed output); `rho_f` = mean |f| / theta, the tracker's response in this loop; "
              "`cos(f, e)` = mean cosine between f and e; `vector law` = mean |(1 - a) e + a f| / theta; `bound` = (1 - a) + a * rho_f. `held` = trials whose "
              "packet was empty (the filter returns `prev`).", "",
              "| seed | gains | theta | trials (held) | k1 stored (GPU) | k1 replay (CPU) | rho_f | cos(f, e) | vector law | bound with rho_f | max trial gap vector law - measured |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for seed, t, g, th, v in lc:
            lab = "unwrapped tracker" if g is None else _gtxt(g)
            L.append(f"| {seed} | {lab} | {th} deg | {v['n_trials']} ({v['n_held']}) | {v['k1_stored_gpu']:.4f} | {v['k1_cpu_vs_stored_pred']:.4f} | "
                     f"{v['rho_f']:.4f} | {v['cos_f_e_mean']:.3f} | {v['k1_vector_law']:.4f} | {v['k1_magnitude_bound_rho_f']:.4f} | "
                     f"{v['vector_law_max_abs_trial_diff']:.1e} |")
        rel = st["rho_f_relative_to_raw"]
        same = ", so the filter hardly changes what the tracker does with a perturbed state" if rel and max(abs(v) for v in rel) < 0.10 else ""
        L += ["", f"Reading: the CPU replay reproduces the stored (GPU) k1 within {st['replay_vs_stored_mean']:.1e} in every row's mean (per trial within "
              f"{st['replay_vs_stored_trial']:.1e}). In the {st['n_filtered']} rows with a root filter the vector law reproduces the retention measured on the "
              f"replay within {st['vector_vs_pair_mean']:.1e} in the mean ({st['vector_vs_pair_trial']:.1e} trial by trial). The tracker's own response in "
              f"the filtered loop is rho_f = {_rng(st['rho_f_filtered'], 3, False)} against {_rng(st['rho_f_raw'], 3, False)} in the raw loop (matched seed and "
              f"theta: filtered minus raw {_rng(st['rho_f_minus_raw'], 3)}, i.e. {_rng([100 * v for v in rel], 1)} %){same}. The mean cosine between that "
              f"response and the injected rotation is {_rng(st['cos_filtered'], 2, False)}: the response is only partly aligned with the perturbation, which "
              f"is why the magnitude bound over-predicts. The bound with the measured rho_f lies above the measured retention in {st['n_bound_holds']} of "
              f"{st['n_filtered']} rows (excess {_rng(st['bound_minus_measured'])}). Empty-packet trials: {st['n_held']}.", ""]
    prim_t = gain_tag(rep["primary_preregistered"])
    a = rep["primary_preregistered"][0]
    pr = [(seed, th, v["retention_k1_k2_k5_k10_k20"]) for seed, d in rep["perturb"].items() if prim_t in d for th, v in d[prim_t].items()]
    if pr:
        above = all(r[i] > (1.0 - a) ** k for _, _, r in pr for i, k in ((0, 1), (1, 2), (2, 5)))
        L += ["### 3c. The primary against the response-free law (1 - a_root)^k", "",
              "The pre-registered debug gate reads 'retention follows (1 - g)^k'. For the filter module this is exact (tested with a response-free stand-in). "
              "For the tracker inside the filter, the tracker's own response keeps part of the injected rotation alive, so the measured retention "
              + ("is above (1 - a_root)^k at k = 1, 2 and 5 (table)" if above else "is not above (1 - a_root)^k at every one of k = 1, 2, 5 (table)")
              + ". Both are given for information; which reading the gate is meant to have is for the registration's owner to decide, and this report "
              "draws no verdict.", "",
              "| seed | theta | measured k1 / k2 / k5 | (1 - a_root)^k for k = 1 / 2 / 5 | measured k20 |", "|---|---|---|---|---|"]
        for seed, th, r in pr:
            L.append(f"| {seed} | {th} deg | {r[0]:.3f} / {r[1]:.3f} / {r[2]:.3f} | {(1 - a):.3f} / {(1 - a) ** 2:.3f} / {(1 - a) ** 5:.3f} | {r[4]:.3f} |")
        L.append("")
    return L


def _criteria_markdown(rep):
    """Section 5: the registered criteria that concern the filter, as they read on this model (information, no verdict)."""
    base_t, prim_t = gain_tag(rep["base"]), gain_tag(rep["primary_preregistered"])
    L = ["## 5. The registered criteria for the filter (`docs/DT_RENDER_TRACK_PREREG.md`), as they read on rt_cnntrack", "",
         "Information for the registration's owner; no verdict is drawn here.", "",
         "| criterion (section) | registered | on rt_cnntrack |", "|---|---|---|",
         f"| gains fixed a priori (2) | root 0.5, fingers 1.0, translation 0.5 | run as the primary {_gtxt(rep['primary_preregistered'])}; the other triples are descriptive only |"]
    n = [f"seed {s}: {v['filtered_1_1_1']['all_shared_numeric_leaves']['n_exact']} / {v['filtered_1_1_1']['all_shared_numeric_leaves']['n_leaves']} json leaves"
         for s, v in rep["sanity"].items() if "filtered_1_1_1" in v]
    if n:
        L.append("| gain 1 is bitwise the bare tracker (3) | yes | section 0 (" + "; ".join(n) + " identical to the recorded evaluation) |")
    L.append("| an empty packet returns `prev` (3) | yes | bitwise for every gain triple of this report, batch 1 and 5 (`tests/test_dt_filter.py`) |")
    pr = [(seed, th, v["retention_k1_k2_k5_k10_k20"]) for seed, d in rep["perturb"].items() if prim_t in d for th, v in d[prim_t].items()]
    if pr:
        a = rep["primary_preregistered"][0]
        L.append(f"| root perturbation retention follows (1 - g)^k (3) | (1 - g)^k | the filter module: exactly, with a response-free stand-in (tests); the tracker "
                 f"inside the filter: k1 {_rng([r[0] for _, _, r in pr], 3, False)}, k2 {_rng([r[1] for _, _, r in pr], 3, False)}, k5 {_rng([r[2] for _, _, r in pr], 3, False)} "
                 f"against {(1 - a):.3f} / {(1 - a) ** 2:.3f} / {(1 - a) ** 5:.3f} (section 3c) |")
    if rep["mean"].get(prim_t) and rep["mean"].get(base_t) and "jitter" in rep["mean"][prim_t]["metrics"]:
        mm, dd = rep["mean"][prim_t]["metrics"], rep["mean"][prim_t]["delta_vs_base"]
        b_acc = rep["mean"][base_t]["metrics"]["jitter"]["acc_err_mm"]
        per = [f"{s}: {100 * rep['per_seed'][s][prim_t]['delta_vs_base']['jitter']['acc_err_mm'] / rep['per_seed'][s][base_t]['metrics']['jitter']['acc_err_mm']:+.1f} %"
               for s in rep["per_seed"]]
        L.append(f"| the filter is effective (6, item 4; registered for the final model, not for rt_cnntrack) | mean delta RA <= 0 and acc_err down >= 15 % | "
                 f"mean delta RA {dd['ra_mm']['overall']:+.3f} mm; acc_err {100 * dd['jitter']['acc_err_mm'] / b_acc:+.1f} % (two-seed mean; per seed {'; '.join(per)}) |")
    L.append("")
    return L


def render_markdown(rep: dict) -> str:
    L = []
    base, prim = rep["base"], rep["primary_preregistered"]
    L += ["# DT filter-eval: constant-gain causal filters on the render-and-compare tracker (zero training)", "",
          f"Generated {rep['generated']} by `tools/dt/filter_eval.py --report` from `outputs/dt/filter/<run>/<tag>.json`.", "",
          f"Model: {rep['model']}; two seeds (3407, 3408), checkpoint `last` (= step 6000).  ",
          f"Protocol: {rep['protocol']}.  ",
          "Filter: `semkine.anchored.FilteredTracker(tracker, a_root, a_rest, a_trans)`: the fed-back state is `prev` moved toward "
          "the tracker's own output by fixed gains (geodesic on the root rotation, linear on the 45 finger parameters and on the "
          "translation); an event-free packet holds `prev`; gains of 1 are the raw tracker. Gains are written ROOT, REST (fingers), TRANS.", "",
          f"**Status of the rows.** The pre-registered primary is {_gtxt(prim)}. All other rows are **descriptive only**: the "
          "project's rule is that filter gains are fixed a priori and never chosen on the evaluation subject (zgz is both the "
          "development and the test subject, AGENTS.md). Nothing in this report selects a gain; a sensitivity row may not be "
          "adopted on the strength of this table.", ""]
    L += _headline(rep)
    if rep["missing"]:
        L += ["**Missing evaluations:** " + ", ".join(rep["missing"]), ""]
    # 0 sanity
    L += ["## 0. Sanity: (1, 1, 1) against the run's own recorded evaluation", "",
          f"Recorded: `outputs/semkine/<run>/{RECORDED_NAME}`, `model.overall.mpjpe_ra_mm`. Required: |difference| <= {SANITY_TOL:g}. "
          "Besides the RA, every numeric leaf the two json files share (controls, teacher forcing, perturbation, CIs, buckets) and every "
          "per-step array of the two npz files is compared.", "",
          "| seed | what was re-run | GPU | recorded RA (mm) | re-run RA (mm) | abs diff | within 1e-6 | shared json leaves identical | per-step arrays bitwise identical |",
          "|---|---|---|---|---|---|---|---|---|"]
    for seed, san in rep["sanity"].items():
        for name, lab in (("filtered_1_1_1", "FilteredTracker (1,1,1)"), ("raw_unwrapped", "unwrapped tracker (evalx path)")):
            if name in san:
                s = san[name]
                a = s["all_shared_numeric_leaves"]
                arr = s.get("step_arrays")
                worst = "" if a["max_abs_diff"] == 0.0 else f"; worst `{a['worst_path']}` {a['max_abs_diff']:.3e}"
                L.append(f"| {seed} | {lab} | {s.get('cuda_visible_devices')} | {s['mpjpe_ra_mm_recorded']:.9f} | {s['mpjpe_ra_mm_reproduced']:.9f} | "
                         f"{s['abs_diff']:.3e} | {'YES' if s['within_tol'] else '**NO**'} | {a['n_exact']} / {a['n_leaves']}{worst} | "
                         + (f"{arr['n_arrays_bitwise_identical']} / {arr['n_arrays_compared']}" if arr else "n/a") + " |")
    L.append("")
    # tables
    L += ["## 1. Per-seed and two-seed-mean results", "",
          "RA = root-aligned MPJPE (mm); global / local = zgz_global / zgz_local sequences (frame-weighted); root rot = geodesic "
          "root-rotation error (deg); abs MPJPE and transl in mm; speed ratios = prediction / ground truth mean step speed "
          "(frame-weighted over the two sequences); fail ep / bad steps = continuous-failure episodes (>= 1 s) / steps with RA > 50 mm "
          "or rotation > 30 deg, summed over both sequences. Jitter keys are `evalx.jitter_decomp` (mm per step / step^2 on the "
          "absolute MANO joints, frame-weighted over the two sequences): `jit_pred_mm` first difference of the output, `acc_err_mm` "
          "second difference of the error (pred - gt), `acc_ratio` second difference of the output over that of the ground truth, "
          "`acc_err_only_<block>_mm` the same with only that block of the state taken from the prediction (the rest from the ground "
          "truth), `rot_acc_pred_deg` root angular acceleration. Lower is better for the error columns (RA, root rot, abs MPJPE, transl, fail counts, "
          "`acc_err*`); for the ratios (speed ratios, `acc_ratio`: 1 is faithful) and for `jit_pred_mm` / `rot_acc_pred_deg` the ground truth's own value "
          "is the reference, not zero (given below the tables).", ""]
    gt_ctx = None
    for seed, rows in rep["per_seed"].items():
        L += _tables(rep, f"seed {seed}", rows)
        jm = rows.get(gain_tag(base), {}).get("metrics", {}).get("jitter")
        if jm and gt_ctx is None:
            gt_ctx = (seed, jm)
    if rep["mean"]:
        L += _tables(rep, "two-seed mean", rep["mean"])
    if gt_ctx:
        jm = gt_ctx[1]
        L += [f"Ground-truth reference for the jitter columns (seed {gt_ctx[0]}; the ground truth is the same for both seeds): "
              f"jit_gt_mm = {jm.get('jit_gt_mm', float('nan')):.3f}, acc_gt_mm = {jm.get('acc_gt_mm', float('nan')):.3f}, "
              f"rot_acc_gt_deg = {jm.get('rot_acc_gt_deg', float('nan')):.3f}.", ""]
    # CIs
    if rep["ra_delta_ci"]:
        some = next(iter(next(iter(rep["ra_delta_ci"].values())).values()))
        nb, n_boot = some["blocks"]["n"], some["blocks"]["n_boot"]
        L += [f"### Paired 10 s block-bootstrap 95% interval of the per-step difference to (1,1,1) (both sequences pooled, {n_boot} resamples, {nb} blocks)", "",
              "| seed | gains | role | delta RA mm [95%] | RA interval vs 0 | blocks with lower RA (sign tail) | delta root rot deg [95%] | root-rot interval vs 0 | blocks with lower root rot (sign tail) |",
              "|---|---|---|---|---|---|---|---|---|"]
        touching = []
        for seed, d in rep["ra_delta_ci"].items():
            for t, ci in d.items():
                g = rep["per_seed"][seed][t]["gains"]
                ra, rr, bl = ci["mpjpe_ra_mm"], ci["root_rot_deg"], ci["blocks"]
                fra, frr = interval_vs_zero(ra[1], ra[2]), interval_vs_zero(rr[1], rr[2])
                if "touches" in fra:
                    touching.append(f"seed {seed} {_gtxt(g)} RA")
                if "touches" in frr:
                    touching.append(f"seed {seed} {_gtxt(g)} root rot")
                nra, nrr = bl["negative"]["mpjpe_ra_mm"], bl["negative"]["root_rot_deg"]
                L.append(f"| {seed} | {_gtxt(g)} | {_role(rep, g)} | {ra[0]:+.3f} [{ra[1]:+.3f}, {ra[2]:+.3f}] | {fra} | {nra} / {bl['n']} ({sign_tail(nra, bl['n']):.3f}) | "
                         f"{rr[0]:+.3f} [{rr[1]:+.3f}, {rr[2]:+.3f}] | {frr} | {nrr} / {bl['n']} ({sign_tail(nrr, bl['n']):.3f}) |")
        L += ["", "The interval covers the evaluation sample only (one subject, 130 s); it does not include retraining noise "
              "(registered replicate floors: 0.4 mm same-config, 1.1 mm across retrainings, `semkine/metrics.py`). It rests on "
              f"{nb} blocks of 10 s (the two sequences' blocks pooled), and a percentile interval from so few blocks is optimistic. An interval "
              f"end within {NEAR_ZERO:g} of 0 is inside the Monte-Carlo error of the percentile (it moves with the resampling seed and the number of "
              "resamples): such a row is marked `touches 0` and must not be read as excluding 0. The count of blocks with a lower error is the "
              "assumption-light companion; the sign tail is the chance that a fair coin gives at least that many (one-sided; the blocks are not "
              "independent, so it is a rough guide only)." + (" Rows marked `touches 0`: " + "; ".join(touching) + "." if touching else ""), ""]
    # modes
    if rep["modes"]:
        L += ["## 2. Teacher forcing and controls", "",
              "Controls (`hold`: the segment's initial state held; `noevents`: empty packets in the model's own loop) were run for (1,1,1) only, "
              "teacher forcing (`tf`: the ground-truth state of the previous step is fed back) for (1,1,1) and the primary only, as specified. "
              "For a filtered model the teacher-forced error is the error of the *filtered* one-step output from the ground-truth state: the filter "
              "pulls that output toward the ground-truth `prev`, so its tf RA and the amplification (closed-loop / tf) are not comparable with the raw "
              "tracker's single-step error.", "",
              "| seed | gains | closed-loop RA | tf RA | amplification (RA) | amplification (rot) | hold RA | no-events RA |", "|---|---|---|---|---|---|---|---|"]
        for seed, d in rep["modes"].items():
            for t, mo in d.items():
                g = rep["per_seed"][seed][t]
                cl = g["metrics"]["ra_mm"]["overall"]
                amp = mo.get("amplification", {})
                L.append(f"| {seed} | {_gtxt(g['gains'])} | {cl:.3f} | {_f(mo.get('tf', {}).get('mpjpe_ra_mm'), 3)} | "
                         f"{_f(amp.get('mpjpe_ra_mm'), 3)} | {_f(amp.get('root_rot_deg'), 3)} | {_f(mo.get('hold', {}).get('mpjpe_ra_mm'), 3)} | "
                         f"{_f(mo.get('noevents', {}).get('mpjpe_ra_mm'), 3)} |")
        L.append("")
    # perturb
    if rep["perturb"]:
        L += ["## 3. Root perturbation retention and the filter law", "",
              "A state rotated by theta about a random axis is rolled 20 steps on the same packets (`evalx.perturb_trials`); retention k = geodesic "
              "angle between the perturbed branch's output and the unperturbed loop's output at step k (k1 = the first output after the perturbed "
              "state), divided by theta. Half-life = first step index k (0-based, k = 0 the first output) at which the divergence is below theta / 2; "
              "0 means the very first output is already below half. Source `main` = the run of section 1; `supp` = a separate perturbation-only "
              "evaluation of the same triple (tag suffix `_pert`), added beyond the specification to look at more than one gain.", "",
              "| seed | gains | source | theta | trials | k1 | k2 | k5 | k10 | k20 | half-life (steps) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for seed, d in rep["perturb"].items():
            for t, pt in d.items():
                g = rep["per_seed"][seed][t]["gains"]
                for th, v in pt.items():
                    r = v["retention_k1_k2_k5_k10_k20"]
                    L.append(f"| {seed} | {_gtxt(g)} | {'main' if rep['perturb_source'][seed][t] == 'main' else 'supp'} | {th} deg | {v['trials']} | "
                             + " | ".join(f"{x:.3f}" for x in r) + f" | {v['half_life_steps']} |")
        L.append("")
        L += _law_markdown(rep)
        if rep["supp_consistency"]:
            worst = max(v["max_abs_diff"] for d in rep["supp_consistency"].values() for v in d.values())
            n = sum(v["n_leaves"] for d in rep["supp_consistency"].values() for v in d.values())
            L += [f"Consistency of the supplementary runs: their closed-loop `model` section equals the main run's of the same triple "
                  f"(max abs difference {worst:.3e} over {n} shared numeric leaves).", ""]
    # by events
    if rep["by_events"]:
        L += ["## 4. Where the primary acts: event-count buckets", "",
              "Steps pooled over both sequences by the number of events in the 50 ms packet (buckets a sequence holds < 20 steps of are dropped by "
              "`evalx.evaluate`); RA (mm) and root rotation error (deg), reference (1,1,1) and the primary with its delta.", "",
              "| seed | events per packet | steps | RA (1,1,1) | RA primary | root rot (1,1,1) | root rot primary |", "|---|---|---|---|---|---|---|"]
        tb, tp = gain_tag(base), gain_tag(prim)
        for seed, d in rep["by_events"].items():
            if tb in d and tp in d:
                for b, v in d[tb].items():
                    w = d[tp].get(b)
                    if w:
                        L.append(f"| {seed} | {b} | {v['n']} | {v['mpjpe_ra_mm']:.3f} | {_cell(w['mpjpe_ra_mm'], w['mpjpe_ra_mm'] - v['mpjpe_ra_mm'], 3)} | "
                                 f"{v['root_rot_deg']:.3f} | {_cell(w['root_rot_deg'], w['root_rot_deg'] - v['root_rot_deg'], 3)} |")
        L.append("")
    L += _criteria_markdown(rep)
    rec_md5 = next(iter(next(iter(rep["provenance"].values()), {}).values()), {}).get("md5", {})
    now_md5 = rep.get("code_now", {})
    changed = [p for p in rec_md5 if p in now_md5 and rec_md5[p] != now_md5[p]]
    L += ["## Notes", "",
          "- Revision: this version replaces the one generated 2026-10-03 07:18, whose perturbation table gave k1 = (1 - a_root) + a_root * rho as "
          "'the law prediction' and read its residual as 'the tracker response implied by the measurement' (wrong: the filter adds rotation "
          "vectors, section 3a). That column is removed; the law is stated as an upper bound, the response inside the filtered loop is measured "
          "directly (3b), and the bootstrap intervals use 20000 resamples with block counts and a `touches 0` flag. The evaluations themselves are unchanged.",
          "- Evaluation sample: val_core = zgz_global (1386 steps, one segment) + zgz_local (1204 steps, many short segments) = 2590 steps of 50 ms; one subject, two "
          "sequences. Closed-loop numbers are chaotic (a 1e-6 change in one forward pass moves a recorded result by ~0.1 mm), which is why (1,1,1) had to reproduce "
          "bitwise (section 0).",
          "- Both seeds are checkpoints of the same configuration (`configs/rt/rt_cnntrack.yaml`, step 6000); two seeds do not make a confidence interval.",
          "- Files: `outputs/dt/filter/<run>/<tag>.json` (+ `.npz`, the per-step arrays) per evaluation, tag = `r<root>_f<fingers>_t<trans>`, `raw` = the unwrapped "
          "tracker, suffix `_pert` = supplementary perturbation-only evaluation, `law_check.json` = the CPU replay of section 3b; this report = "
          "`outputs/dt/reports/filter_cnntrack.{json,md}` (docs copy `docs/DT_FILTER_CNNTRACK_RESULT_20261003.md`). Re-run: `python tools/dt/filter_eval.py "
          "--run-dir outputs/semkine/rt_cnntrack_s<seed> --ckpt last --gains '<ROOT,REST,TRANS;...>' [--controls] [--tf] [--perturb] --out-dir "
          "outputs/dt/filter/rt_cnntrack_s<seed>` (one GPU: `CUDA_VISIBLE_DEVICES=<n>`), then `--law-check --run-dir outputs/semkine/rt_cnntrack_s<seed> "
          "--in-dir outputs/dt/filter/rt_cnntrack_s<seed> --device cpu`, then `--report --runs rt_cnntrack_s3407 rt_cnntrack_s3408 --in-dir outputs/dt/filter "
          "--report-out outputs/dt/reports/filter_cnntrack`.",
          "- Code identity of the evaluations (md5, from the json `env`): " + "; ".join(f"{p} {h}" for p, h in rec_md5.items()) + "."
          + ((" The same files in the checkout that generated this report: " + "; ".join(f"{p} {now_md5[p]}" for p in changed)
              + " (these differ from the recorded ones, i.e. they were edited after the evaluations; the numbers above are those of the recorded versions.)")
             if changed else ""),
          ""]
    return "\n".join(L) + "\n"


def cmd_report(a):
    runs = [Path(r).name for r in a.runs]
    gains = parse_gains(a.gains) if a.gains else [BASE, PRIMARY, *SENSITIVITY]
    base, primary = parse_gains(a.base)[0], parse_gains(a.primary)[0]
    for g in (base, primary):
        if g not in gains:
            gains.insert(0, g)
    rep = build_report(runs, a.in_dir, gains, base, primary, a.semkine_dir)
    out = Path(a.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1))
    md = render_markdown(rep)
    out.with_suffix(".md").write_text(md)
    print("wrote", out.with_suffix(".json"), out.with_suffix(".md"))
    if a.docs_copy:
        Path(a.docs_copy).write_text(md)
        print("wrote", a.docs_copy)
    if rep["missing"]:
        print("MISSING:", ", ".join(rep["missing"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", help="a run directory, e.g. outputs/semkine/rt_cnntrack_s3407")
    ap.add_argument("--ckpt", default="last", help="selected | last | step=N (evalx.find_ckpt)")
    ap.add_argument("--gains", default="", help="'ROOT,REST,TRANS;ROOT,REST,TRANS;...' (report: override the default set)")
    ap.add_argument("--raw", action="store_true", help="also evaluate the unwrapped tracker (no FilteredTracker), tag 'raw'")
    ap.add_argument("--controls", action="store_true", help="also the hold / no-events controls (all triples of this call)")
    ap.add_argument("--tf", action="store_true", help="also the teacher-forced loop")
    ap.add_argument("--perturb", action="store_true", help="also root-rotation perturbation recovery (10 / 20 deg)")
    ap.add_argument("--out-dir", help="one <tag>.json (+ .npz) per triple is written here")
    ap.add_argument("--split", default="val_core")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--config", default=None, help="model config (default: the yaml inside the run directory)")
    ap.add_argument("--suffix", default="", help="appended to the tag, e.g. when the same triple is run with other modes")
    ap.add_argument("--device", default=None, help="default: cuda if available (use CUDA_VISIBLE_DEVICES to pick the GPU)")
    ap.add_argument("--threads", type=int, default=0, help="torch.set_num_threads (0: leave the default)")
    ap.add_argument("--no-npz", action="store_true", help="do not save the per-step arrays")
    ap.add_argument("--report", action="store_true", help="summarise evaluations instead of running them")
    ap.add_argument("--runs", nargs="+", help="report: run names, e.g. rt_cnntrack_s3407 rt_cnntrack_s3408")
    ap.add_argument("--in-dir", default=None, help="report: directory holding <run>/<tag>.json; law-check: the run's own directory "
                                                   "holding <tag>.json / .npz")
    ap.add_argument("--report-out", default=None, help="report: output path without extension (.json and .md are written)")
    ap.add_argument("--docs-copy", default=None, help="report: also write the markdown to this path (e.g. docs/DT_FILTER_CNNTRACK_RESULT.md)")
    ap.add_argument("--semkine-dir", default=str(REPO / "outputs" / "semkine"), help="report: where the recorded evaluations live")
    ap.add_argument("--base", default="1,1,1", help="report: the reference triple")
    ap.add_argument("--primary", default="0.5,1,0.5", help="report: the pre-registered primary")
    ap.add_argument("--law-check", action="store_true", help="CPU replay of the first step of the stored perturbation trials: "
                                                             "splits the first-step retention into the filter's and the tracker's share")
    ap.add_argument("--tags", default="", help="law-check: comma-separated tags (default: every evaluation in --in-dir that has perturbation trials)")
    ap.add_argument("--law-out", default=None, help="law-check: output path (default: <in-dir>/law_check.json)")
    a = ap.parse_args(argv)
    if a.report:
        if not (a.runs and a.in_dir and a.report_out):
            ap.error("--report needs --runs, --in-dir and --report-out")
        cmd_report(a)
    elif a.law_check:
        if not (a.run_dir and a.in_dir):
            ap.error("--law-check needs --run-dir and --in-dir")
        cmd_law_check(a)
    else:
        if not (a.run_dir and a.out_dir):
            ap.error("evaluation needs --run-dir and --out-dir")
        cmd_eval(a)


if __name__ == "__main__":
    main()
