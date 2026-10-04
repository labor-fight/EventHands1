#!/usr/bin/env python3
"""DT3 package C: zero-training INFERENCE-time probes of the dt_dz_l3 tracker (no TF, no perturbation, no controls).

All of them re-run the protocol closed loop (`tools/tracking/evalx.run_sequence`, mode "model": 50 ms steps at the ends
a+49, a+99, ... of every valid segment, prev = the previous output, fp32, batch 1, init = GT + protocol noise from one rng
seeded 0) through `evalx.evaluate`, so the summary json / npz has exactly the structure of an evalx evaluation; the loop is
a faithful copy of the current `evalx.run_sequence` (monkeypatched in, as `tools/dt/lagprev_probe.py` does -- evalx itself is
not touched) with three additions, every one of them off by default:

  --window-ms W        the fixed evidence window (default 50 = the protocol); the evaluated steps never change.
  --clip-run-start     window = min(W, end + 1 - a) (a = the segment start), the way TRAINING clips a window to the run
                       start; the default is evalx's min(W, end + 1) (clipped to the recording start only).
  probe hooks          a multi-pass wrapper model gets the step's context (`begin_step`) so it can build more LNES windows
                       of the same step; its per-step diagnostics are returned next to the arrays.
  --filter-spec F      wrap the probe model in `semkine.anchored.AdaptiveFilter.from_spec` (F: a JSON file / text of one spec,
                       e.g. outputs/dt2/filter_sweep/recommended_spec.json); the loop honours reset_state / get_state /
                       takes_event_count / set_hand_context exactly as evalx does. The filter's `prev` is its own filtered
                       output and the probe model is called with that prev.

Sub-commands (output outputs/dt3/<sub-command>/<run>_<tag>.json + .npz; the evaluate() summary plus a "probe" dict and the
arrays "probe|<seq>|<name>" next to the evalx arrays "model|<seq>|<name>"):

  ens-win --windows 100,200,300   at every step build the LNES at each window, run the underlying model once per window at
        batch 1, SEQUENTIALLY (a batch > 1 changes the cuDNN kernels and adds ~0.03 mm of chaos), same prev; the output is the
        mean of translation and fingers and the tangent-space mean of the root about the FIRST window's output (axis-angle ->
        quaternion -> Log relative to the first -> mean -> Exp). A single window is the plain loop (bit for bit).
        Per step: root / finger / translation spread across the windows, innovation |output - prev|, the 50 ms event count;
        the summary holds Spearman(spread, per-step RA error) within the event-count buckets [0,500) [500,2000) [2000,inf).
  adabn --iters 2 --bs 32         test-time AdaBN (transductive, unlabeled; the validation set is also the test set, so this
        is a dev = test probe): record (LNES x, prev) of every step of the closed loop, reset every nn.BatchNorm2d's running
        stats (momentum = None: a cumulative average) and re-estimate them by forwarding shuffled batches of the recorded
        inputs per sequence (betas / K per sequence via set_hand_context; BN layers in train mode, everything else eval, no
        gradient), re-run the closed loop with the adapted model, `iters` times; the last closed loop is the evaluation.
        `--iters 0` touches nothing (the plain loop). Records the BN statistics shift (mean |delta running_mean| per layer).
  tta-pol                         output = combination of model(x, prev) and model(x with the polarity of every channel pair
        swapped, prev) (batch 1 each, same rule as ens-win); the render channels live inside the model and are not touched.
  seed-ens --run-dirs R1 R2 R3    output = combination of the runs' models (each with its own checkpoint; one prev = the
        ensemble's own previous output); diagnostic only.
  cmp A.npz B.npz                 bit-identity check of two evaluation npz (and, with --json-a/--json-b, of their json numbers).

    CUDA_VISIBLE_DEVICES=6 python tools/dt/infer_probe.py ens-win --run-dir outputs/semkine/dt_dz_l3_s3407 \
        --windows 100,200,300 --window-ms 200
    CUDA_VISIBLE_DEVICES=6 python tools/dt/infer_probe.py adabn --run-dir outputs/semkine/dt_dz_l3_s3407 --iters 2 --window-ms 200
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "tracking")]
import evalx as EX                                                    # noqa: E402
from config import load_config                                        # noqa: E402
from mano_layer import ManoLayer                                      # noqa: E402
from model import MNISTModel                                          # noqa: E402
from semkine import eval_track as ET                                  # noqa: E402
from semkine import events as EV                                      # noqa: E402
from semkine.anchored import AdaptiveFilter, _t_conj, _t_log, _t_qmul, _t_quat   # noqa: E402
from semkine.dataset import sequences_for_split                       # noqa: E402

STEP = EX.STEP
#: the 50 ms event-count buckets of the Spearman report (evalx's EVENT_BUCKETS, without the 2000-10000 split)
SPEARMAN_BUCKETS = ((0, 500), (500, 2000), (2000, 10 ** 9))
MIN_SPEARMAN_N = 20                    # evalx.evaluate's own minimum for a by-events bucket
DIAG_KEYS = ("spread_root_deg", "spread_fing", "spread_transl_mm", "innov_transl_mm", "innov_root_deg", "innov_fing")


# ------------------------------------------------------------------------------------------------ windows
def eff_window(end: int, a: int, win: int, clip_run_start: bool = False) -> int:
    """The evidence window (ms) of the step ending at `end` in a segment starting at `a`. Default: evalx's fixed window,
    `min(win, end + 1)` (clipped to the recording start only). `clip_run_start`: `min(win, end + 1 - a)`, the window
    TRAINING uses (it never reaches before the run start): 50 / 100 / 150 / 200 ... for the first steps of a segment."""
    return int(min(win, end + 1 - a)) if clip_run_start else int(min(win, end + 1))


class StepCtx:
    """What a multi-pass probe model needs at one step to build further LNES windows: the sequence's events / offsets, the
    step's `end`, the segment start `a`, the event-channel set and the primary window's image `x` (window `w`, already on
    the device). `lnes(win)` is the image of window `win` (clipped as the loop clips); the primary window's image is reused
    (the very tensor the loop built), any other is built once per step."""

    def __init__(self, events, offsets, end, a, ev_ch, device, clip_run_start, x, w):
        self.events, self.offsets, self.end, self.a = events, offsets, int(end), int(a)
        self.ev_ch, self.device, self.clip = ev_ch, device, bool(clip_run_start)
        self.x, self.w = x, int(w)
        self._cache = {self.w: x}

    def lnes(self, win=None):
        if win is None:
            return self.x
        w = eff_window(self.end, self.a, win, self.clip)
        if w not in self._cache:
            self._cache[w] = torch.from_numpy(ET.build_lnes(self.events, self.offsets, self.end, w, self.ev_ch)
                                              ).unsqueeze(0).to(self.device)
        return self._cache[w]


# ------------------------------------------------------------------------------------------------ the loop
@torch.no_grad()
def run_sequence(model, cfg, root, d, seq, device, rng, mode="model", wmode="fixed", win=STEP, min_events=0, max_win=300,
                 *, clip_run_start=False, probe=None, sink=None, record=None):
    """`evalx.run_sequence` (mode "model", fixed window) as a copy, with the keyword-only extras below, all default off.

    The signature up to `max_win` is evalx's (so `EX.run_sequence = functools.partial(run_sequence, ...)` is a drop-in);
    `min_events` / `max_win` are unused (fixed windows). A model with `reset_state` is reset at every segment start (no
    random numbers), one with `get_state` has its state recorded before every step in `fstate`, one with
    `takes_event_count` is called `model(x, prev, n_events=n)` with the raw event count of the 50 ms packet.
    clip_run_start  the window is clipped to the segment start (`eff_window`) instead of the recording start.
    probe           a `MultiPass`: `probe.begin_step(StepCtx)` before every step, `probe.pop_diag()` after it; the per-step
                    diagnostics go to `sink[seq]` (arrays, with the effective window `win_ms` and the event `count`).
    sink            dict: filled with `sink[seq] = {name: array}` (the per-step arrays of the "probe|<seq>|<name>" entries).
    record          dict: `record[seq] = {"x": [...], "prev": [...], "betas", "K"}`, the LNES image (H, W, C) float32 and the
                    conditioning state (51,) of every step (CPU), the recorder pass of `adabn`."""
    assert mode == "model" and wmode == "fixed", "infer_probe loops the protocol model mode with a fixed window"
    assert not bool(getattr(model, "encoder_name", "")), "infer_probe reads the dense LNES (a raw-event model is not supported)"
    events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    camera_K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    if hasattr(model, "set_hand_context"):
        model.set_hand_context(betas, camera_K)
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
    ev_ch = EV.event_channels(cfg)
    stateful = hasattr(model, "get_state")
    with_count = bool(getattr(model, "takes_event_count", False))
    preds, gts, ends_all, runs_all, elapsed, counts, prevs, fstate = [], [], [], [], [], [], [], []
    wins, diag = [], {k: [] for k in DIAG_KEYS}
    if record is not None:
        record[seq] = {"x": [], "prev": [], "betas": betas.cpu(), "K": camera_K.cpu()}
    for run_id, (a, b) in enumerate(runs):
        ends = np.arange(a + STEP - 1, b, STEP, dtype=np.int64)
        if not len(ends):
            continue
        prev = pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)
        prev_t = torch.from_numpy(prev).view(1, -1).to(device)
        if hasattr(model, "reset_state"):
            model.reset_state(prev_t)
        for end in ends:
            n_ev = int(offsets[int(end) + 1] - offsets[int(end) - STEP + 1])
            w = eff_window(int(end), int(a), win, clip_run_start)
            prevs.append(prev_t.cpu().numpy()[0])
            if stateful:
                fstate.append(model.get_state())
            x_np = ET.build_lnes(events, offsets, int(end), w, ev_ch)
            x = torch.from_numpy(x_np).unsqueeze(0).to(device)
            if record is not None:
                record[seq]["x"].append(x_np)
                record[seq]["prev"].append(prevs[-1])
            if probe is not None:
                probe.begin_step(StepCtx(events, offsets, int(end), int(a), ev_ch, device, clip_run_start, x, w))
            if with_count:
                pred = model(x, prev_t, n_events=n_ev)
            else:
                pred = model(x, prev_t)
            if probe is not None:
                for k, v in probe.pop_diag().items():
                    diag[k].append(v)
            prev_t = pred
            preds.append(pred.cpu().numpy()[0])
            gts.append(pos51[end])
            ends_all.append(int(end))
            runs_all.append(run_id)
            elapsed.append(int(end - a))
            counts.append(n_ev)
            wins.append(w)
    if sink is not None:
        sink[seq] = {"count": np.asarray(counts), "win_ms": np.asarray(wins)}
        if probe is not None:
            sink[seq].update({k: np.asarray(v, dtype=np.float32) for k, v in diag.items()})
    return {"pred": np.stack(preds).astype(np.float32), "gt": np.stack(gts).astype(np.float32),
            "end": np.asarray(ends_all), "run": np.asarray(runs_all), "elapsed": np.asarray(elapsed),
            "count": np.asarray(counts), "betas": aux["betas"],
            "prev": np.stack(prevs).astype(np.float32), "fstate": fstate}


@contextlib.contextmanager
def patched_loop(**kw):
    """`evalx.evaluate` calls the module-level `run_sequence` positionally; swap in this file's loop for the block."""
    orig = EX.run_sequence
    EX.run_sequence = functools.partial(run_sequence, **kw)
    try:
        yield
    finally:
        EX.run_sequence = orig


# ------------------------------------------------------------------------------------------------ combination rule
def tangent_mean(aa: torch.Tensor):
    """Mean of K rotations (axis-angle `aa`, (K, 3)) in the tangent space about the FIRST one, in float64: q_k = Exp(aa_k),
    rel_k = Log(q_0^-1 q_k) (|rel| <= pi, rel_0 = 0), q = q_0 Exp(mean_k rel_k). Returns `(axis-angle (3,), rel (K, 3))`.
    Identical rotations give back that rotation; two rotations give the geodesic midpoint."""
    aa = aa.double()
    q = _t_quat(aa)
    ref = q[:1].expand_as(q)
    rel = _t_log(_t_qmul(_t_conj(ref), q))
    mean = rel.mean(dim=0, keepdim=True)
    return _t_log(_t_qmul(q[:1], _t_quat(mean)))[0], rel


def rot_angle_deg(aa_a: torch.Tensor, aa_b: torch.Tensor):
    """Geodesic angle (deg) between two sets of axis-angle rotations, (..., 3) float64 -> (...,)."""
    rel = _t_log(_t_qmul(_t_conj(_t_quat(aa_a.double())), _t_quat(aa_b.double())))
    return torch.linalg.norm(rel, dim=-1) * (180.0 / math.pi)


def combine(outs):
    """The ensemble rule for the (1, 51) outputs `outs` of K passes: translation (0:3) and fingers (6:51) = mean, root (3:6) =
    tangent-space mean about the first output (`tangent_mean`), computed in float64 on the host and returned in the first
    output's dtype / device. K = 1 returns that output itself (so a single pass is bit-identical to the plain loop).
    Returns `(out, spread)`, spread = RMS deviation of the K outputs from their mean: `root_deg` (tangent space, degrees),
    `fing` (L2 over the 45 finger angles, rad), `transl_mm`; all 0 for K = 1."""
    if len(outs) == 1:
        return outs[0], {"spread_root_deg": 0.0, "spread_fing": 0.0, "spread_transl_mm": 0.0}
    z = torch.cat([o.detach() for o in outs], dim=0).double().cpu()                      # (K, 51)
    t, f = z[:, :3], z[:, 6:]
    r, rel = tangent_mean(z[:, 3:6])
    rms = lambda v: float(torch.sqrt(((v - v.mean(dim=0, keepdim=True)) ** 2).sum(dim=-1).mean()))   # noqa: E731
    spread = {"spread_root_deg": rms(rel) * 180.0 / math.pi, "spread_fing": rms(f), "spread_transl_mm": rms(t) * 1000.0}
    out = torch.cat([t.mean(dim=0), r, f.mean(dim=0)]).view(1, -1).to(device=outs[0].device, dtype=outs[0].dtype)
    return out, spread


def swap_polarity(x: torch.Tensor) -> torch.Tensor:
    """The LNES with the polarity of every channel pair swapped. `semkine.events.splat_event_image` orders the planes as the
    event channels with polarity (0, 1) adjacent: (1, H, W, 2) for ("last",), (1, H, W, 2k) for k channel sets -> channels
    [1, 0, 3, 2, ...]. Only the event image is touched (the render channels are produced inside the model)."""
    c = x.shape[-1]
    assert c % 2 == 0, "an event image has 2 polarity planes per channel"
    return x[..., [j ^ 1 for j in range(c)]]


# ------------------------------------------------------------------------------------------------ the probe model
class MultiPass(nn.Module):
    """A model made of K forward passes at batch 1, run SEQUENTIALLY and combined by `combine` (so any `prev`-conditioned
    closed loop, or an `AdaptiveFilter` around it, treats it as one tracker). `members` are the underlying models,
    `passes` a list of `(member index, window ms or None, swap polarity)`: a pass with a window reads that window's LNES of
    the step (from the `StepCtx` the loop hands over; None = the loop's primary image), a swapped pass reads the image with
    the polarity of each channel pair swapped. Per call the diagnostics of `combine` plus the innovation `|output - prev|`
    (translation mm, root deg, fingers L2 rad) of the combined output are kept for `pop_diag`."""

    encoder_name = ""

    def __init__(self, members, passes):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.passes = [(int(i), None if w is None else int(w), bool(s)) for i, w, s in passes]
        self._ctx = None
        self._diag = {}

    def set_hand_context(self, betas, camera_K):
        for m in self.members:
            if hasattr(m, "set_hand_context"):
                m.set_hand_context(betas, camera_K)

    def begin_step(self, ctx: StepCtx):
        self._ctx = ctx

    def pop_diag(self):
        d, self._diag = self._diag, {}
        return d

    def forward(self, x, prevpos, betas=None, camera_K=None):
        outs = []
        for mi, win, swap in self.passes:
            if win is None:
                xi = x
            elif self._ctx is None:
                raise RuntimeError("a windowed pass needs the step context: the loop calls begin_step first")
            else:
                xi = self._ctx.lnes(win)
            outs.append(self.members[mi](swap_polarity(xi) if swap else xi, prevpos, betas=betas, camera_K=camera_K))
        out, spread = combine(outs)
        o, p = out.detach().double().cpu()[0], prevpos.detach().double().cpu()[0]
        self._diag = dict(spread, innov_transl_mm=float(torch.linalg.norm(o[:3] - p[:3])) * 1000.0,
                          innov_root_deg=float(rot_angle_deg(p[3:6], o[3:6])),
                          innov_fing=float(torch.linalg.norm(o[6:] - p[6:])))
        return out


# ------------------------------------------------------------------------------------------------ AdaBN
def bn2d_layers(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]


def bn_snapshot(model):
    return {n: (m.running_mean.detach().clone(), m.running_var.detach().clone()) for n, m in bn2d_layers(model)}


def bn_shift(before, after):
    """Per layer: mean |delta running_mean|, mean |delta running_var| / mean running_var(before), and their means over layers."""
    rows = []
    for n in before:
        (m0, v0), (m1, v1) = before[n], after[n]
        rows.append({"layer": n, "d_mean_abs": float((m1 - m0).abs().mean()), "d_var_rel": float((v1 - v0).abs().mean() / v0.abs().mean().clamp_min(1e-12))})
    return {"mean_d_mean_abs": float(np.mean([r["d_mean_abs"] for r in rows])) if rows else 0.0,
            "mean_d_var_rel": float(np.mean([r["d_var_rel"] for r in rows])) if rows else 0.0, "layers": rows}


@torch.no_grad()
def adapt_bn(model, rec, bs, seed, device):
    """Re-estimate the running statistics of every nn.BatchNorm2d of `model` on the recorded inputs `rec` (`run_sequence`'s
    `record`: per sequence the LNES images, the conditioning states and the betas / K): `reset_running_stats()`, momentum
    None (the running stats become the plain average over the forwarded batches), the BN layers in train mode and the rest of
    the model in eval, shuffled batches of `bs` per sequence (a last batch of 1 is skipped), no gradient; afterwards every
    layer is back in eval mode with its momentum restored. The batch statistics of a batch are the very ones training used
    (running_mean = mean of the batch means, running_var = mean of the unbiased batch variances). Returns the number of batches."""
    layers = bn2d_layers(model)
    if not layers:
        return 0
    moms = [m.momentum for _, m in layers]
    model.eval()
    for _, m in layers:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    g = torch.Generator().manual_seed(int(seed))
    n_batches = 0
    for s, r in rec.items():
        n = len(r["x"])
        if not n:
            continue
        if hasattr(model, "set_hand_context"):
            model.set_hand_context(r["betas"].to(device), r["K"].to(device))
        perm = torch.randperm(n, generator=g)
        for i0 in range(0, n, bs):
            sel = perm[i0:i0 + bs].tolist()
            if len(sel) < 2:
                continue
            x = torch.from_numpy(np.stack([r["x"][j] for j in sel])).to(device)
            p = torch.from_numpy(np.stack([r["prev"][j] for j in sel])).to(device)
            model(x, p)
            n_batches += 1
    for (_, m), mo in zip(layers, moms):
        m.eval()
        m.momentum = mo
    return n_batches


# ------------------------------------------------------------------------------------------------ statistics
def _spearman(x, y):
    """Spearman rho of two samples; None when either is constant or shorter than `MIN_SPEARMAN_N`."""
    from scipy.stats import spearmanr
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) < MIN_SPEARMAN_N or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    rho = spearmanr(x, y)[0]
    return None if not np.isfinite(rho) else float(rho)


def spearman_report(arrays, seqs, spread_keys=("spread_root_deg", "spread_fing")):
    """Spearman(per-step spread, per-step RA error `mpjpe_ra_mm`) within the 50 ms event-count buckets, per sequence and
    pooled over the sequences (`all`); each cell `{n, <spread key>: rho or None}`."""
    names = [s for s, _ in seqs]
    out = {}
    for tag, group in [(s, [s]) for s in names] + [("all", names)]:
        cnt = np.concatenate([arrays[f"model|{s}|count"] for s in group])
        err = np.concatenate([arrays[f"model|{s}|mpjpe_ra_mm"] for s in group])
        sp = {k: np.concatenate([arrays[f"probe|{s}|{k}"] for s in group]) for k in spread_keys}
        cells = {}
        for lo, hi in SPEARMAN_BUCKETS + ((0, 10 ** 9),):
            sel = (cnt >= lo) & (cnt < hi)
            key = "all" if (lo, hi) == (0, 10 ** 9) else f"[{lo},{'inf' if hi >= 10 ** 9 else hi})"
            cells[key] = dict({"n": int(sel.sum())}, **{k: _spearman(sp[k][sel], err[sel]) for k in spread_keys})
        out[tag] = cells
    return out


def diag_summary(arrays, seqs):
    """mean / median / p90 / max of every per-step diagnostic, per sequence and pooled."""
    names = [s for s, _ in seqs]
    out = {}
    for tag, group in [(s, [s]) for s in names] + [("all", names)]:
        d = {}
        for k in DIAG_KEYS:
            if f"probe|{group[0]}|{k}" not in arrays:
                continue
            v = np.concatenate([arrays[f"probe|{s}|{k}"] for s in group]).astype(np.float64)
            d[k] = {"mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(v.max())}
        out[tag] = d
    return out


def ra_parts(summary):
    """Overall / global / local `mpjpe_ra_mm` and `root_rot_deg` of an `evaluate` summary (global = sequences without `_local`)."""
    m = summary["model"]
    seqs = [k for k, v in m.items() if isinstance(v, dict) and "n_frames" in v]

    def wm(ss, key):
        n = sum(m[s]["n_frames"] for s in ss)
        return float(sum(m[s][key][0] * m[s]["n_frames"] for s in ss) / n) if n else float("nan")
    glo, loc = [s for s in seqs if "_local" not in s], [s for s in seqs if "_local" in s]
    return {"ra_overall": float(m["overall"]["mpjpe_ra_mm"]), "ra_global": wm(glo, "mpjpe_ra_mm"), "ra_local": wm(loc, "mpjpe_ra_mm"),
            "rot_overall": float(m["overall"]["root_rot_deg"]), "rot_global": wm(glo, "root_rot_deg"), "rot_local": wm(loc, "root_rot_deg")}


# ------------------------------------------------------------------------------------------------ setup / io
def load_one_spec(arg: str):
    """`--filter-spec`: the path of a JSON file or a JSON text holding ONE `AdaptiveFilter.canonical_spec` dict (as
    tools/dt/filter_eval.py reads it; a list of one is accepted)."""
    text = arg
    if arg.lstrip()[:1] not in ("{", "["):
        text = Path(arg).read_text()
    obj = json.loads(text)
    specs = obj if isinstance(obj, list) else [obj]
    if len(specs) != 1:
        raise SystemExit(f"--filter-spec holds {len(specs)} specs; infer_probe takes exactly one")
    return AdaptiveFilter.canonical_spec(specs[0])


def spec_hash_tag(canon: dict) -> str:
    """'af<6 hex>' with the hash of tools/dt/filter_eval.spec_tag (md5 of the canonical spec's sorted json)."""
    return "af" + hashlib.md5(json.dumps(canon, sort_keys=True).encode()).hexdigest()[:6]


def build_run(run: Path, ckpt: str, device, config=None):
    """cfg, model (eval, on `device`), checkpoint path and step of one run, as `evalx.cmd_eval` builds them."""
    cfg_path = config or next(iter(sorted(run.glob("*.yaml"))), None) or json.loads(
        (run / "training_metadata.json").read_text())["config_path"]
    cfg = load_config(cfg_path)
    path, step, _ = EX.find_ckpt(run, ckpt)
    model = MNISTModel.load_from_checkpoint(str(path), cfg=cfg, map_location=device).to(device).eval()
    return cfg, model, path, step


def check_no_leak(run: Path, seqs):
    meta = run / "training_metadata.json"
    trained = json.loads(meta.read_text()).get("train_sequences", []) if meta.exists() else []
    leak = {s.split("_")[0] for s, _ in seqs} & {s.split("_")[0] for s in trained}
    assert not leak, f"validation subjects {sorted(leak)} are in {run.name}'s training set"


def md5_of(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def write_atomic(path: Path, data: bytes):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def eval_namespace(window_ms):
    """The fields `evalx.evaluate` reads from its `a`: the fixed protocol, no controls / TF / perturbation."""
    return argparse.Namespace(controls=False, tf=False, perturb=False, window_mode="fixed", window_ms=int(window_ms),
                              min_events=0, max_window_ms=300)


def evaluate_probe(net, probe, cfg, mano, root, seqs, device, args, label):
    """`evalx.evaluate` with this file's loop; returns `(summary, arrays, sink)`, the probe diagnostics also merged into
    `arrays` as `probe|<seq>|<name>`."""
    sink = {}
    with patched_loop(clip_run_start=args.clip_run_start, probe=probe, sink=sink):
        summary, arrays = EX.evaluate(net, cfg, mano, root, seqs, device, eval_namespace(args.window_ms), label=label)
    for s, dd in sink.items():
        for k, v in dd.items():
            arrays[f"probe|{s}|{k}"] = v
    return summary, arrays, sink


def write_result(args, sub, run_label, tag, ckpts, probe_info, summary, arrays, extra, t0, device):
    out_dir = Path(args.out_dir) if args.out_dir else REPO / "outputs" / "dt3" / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{run_label}_{tag}"
    out = {"run": run_label, "ckpt": ckpts, "split": args.split, "probe": probe_info}
    out.update(summary)
    out["window"] = {"mode": "fixed", "ms": args.window_ms, "clip_run_start": bool(args.clip_run_start), "min_events": 0, "max_ms": 300}
    out.update(extra)
    out["wall_s"] = time.time() - t0
    out["env"] = {"torch": torch.__version__, "device": str(device),
                  "gpu": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "md5": {p: md5_of(REPO / p) for p in ("semkine/anchored.py", "tools/tracking/evalx.py", "model/model.py",
                                                        "tools/dt/infer_probe.py")}}
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    write_atomic(out_dir / f"{name}.npz", buf.getvalue())
    write_atomic(out_dir / f"{name}.json", json.dumps(out, indent=1).encode())
    p = ra_parts(summary)
    print(f"{name}: RA {p['ra_overall']:.4f} (g {p['ra_global']:.3f} l {p['ra_local']:.3f}) rot {p['rot_overall']:.3f} "
          f"(g {p['rot_global']:.2f} l {p['rot_local']:.2f})  wall {out['wall_s']:.0f} s -> {out_dir / (name + '.json')}", flush=True)
    return out


def setup(args):
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def common_context(args, run: Path, device):
    cfg, model, ckpt, step = build_run(run, args.ckpt, device, args.config)
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    mani = Path(args.manifest) if args.manifest else Path(cfg["DATA"]["SPLITS_MANIFEST"])
    seqs = sequences_for_split(root, args.split, mani)
    check_no_leak(run, seqs)
    return cfg, model, ckpt, step, mano, root, seqs


def settings_of(args):
    return {"window_ms": args.window_ms, "clip_run_start": bool(args.clip_run_start),
            "filter": load_one_spec(args.filter_spec) if args.filter_spec else None, "ckpt_rule": args.ckpt, "split": args.split}


def default_tag(args, core):
    if args.tag:
        return args.tag
    tag = f"{core}_pw{args.window_ms}" + ("_clip" if args.clip_run_start else "")
    return tag + (f"_{spec_hash_tag(load_one_spec(args.filter_spec))}" if args.filter_spec else "")


def wrap_filter(args, probe_model, device):
    if not args.filter_spec:
        return probe_model
    return AdaptiveFilter.from_spec(probe_model, load_one_spec(args.filter_spec)).to(device).eval()


# ------------------------------------------------------------------------------------------------ sub-commands
def cmd_multipass(args, sub):
    """ens-win / tta-pol / seed-ens: one MultiPass probe model through the closed loop."""
    t0 = time.time()
    device = setup(args)
    if sub == "seed-ens":
        runs = [Path(r) for r in args.run_dirs]
        built = [common_context(args, runs[0], device)] + [build_run(r, args.ckpt, device, args.config) for r in runs[1:]]
        cfg, m0, ck0, st0, mano, root, seqs = built[0]
        members, ckpts = [b[1] for b in built], [f"{r.name}:{b[2]}" for r, b in zip(runs, built)]
        for r, b in zip(runs[1:], built[1:]):
            assert EV.event_channels(b[0]) == EV.event_channels(cfg) and (b[0].get("TRACK") or {}) == (cfg.get("TRACK") or {}), \
                f"{r.name}: event channels / TRACK noise differ from {runs[0].name}"
            check_no_leak(r, seqs)
        passes = [(i, None, False) for i in range(len(members))]
        names = [r.name for r in runs]
        pre = os.path.commonprefix(names)
        pre = pre[:pre.rfind("_") + 1] if len(set(names)) > 1 else ""                         # dt_dz_l3_s3407+s3408+s3409
        run_label = names[0] + "".join("+" + n[len(pre):] for n in names[1:])
        core, info = "seedens", {"run_dirs": [str(r) for r in runs]}
    else:
        run = Path(args.run_dir)
        cfg, m0, ck0, st0, mano, root, seqs = common_context(args, run, device)
        members, ckpts, run_label = [m0], [str(ck0)], run.name
        if sub == "ens-win":
            windows = [int(w) for w in args.windows.split(",") if w.strip()]
            assert windows, "--windows needs at least one window (ms)"
            passes, core, info = [(0, w, False) for w in windows], "ensw" + "-".join(map(str, windows)), {"windows": windows}
        else:
            passes, core, info = [(0, None, False), (0, None, True)], "ttapol", {"passes": ["plain", "polarity-swapped"]}
    probe = MultiPass(members, passes).to(device).eval()
    net = wrap_filter(args, probe, device)
    tag = default_tag(args, core)
    print(f"{sub}: {run_label} {tag}  passes={passes}  device={device}", flush=True)
    summary, arrays, sink = evaluate_probe(net, probe, cfg, mano, root, seqs, device, args, label=f"{run_label} {tag}")
    info.update({"sub": sub, "settings": settings_of(args), "passes": [list(p) for p in passes],
                 "combination": "mean (translation, fingers); root: tangent-space mean about the first pass's output (float64)",
                 "diag": diag_summary(arrays, seqs), "spearman_spread_vs_ra_err": spearman_report(arrays, seqs),
                 "spearman_buckets": [f"[{lo},{'inf' if hi >= 10 ** 9 else hi})" for lo, hi in SPEARMAN_BUCKETS]})
    write_result(args, sub, run_label, tag, ckpts, info, summary, arrays, {}, t0, device)


def record_pass(net, cfg, root, seqs, device, args, mano=None):
    """One closed loop of `net` recording (x, prev) per step per sequence; returns `(record, overall RA mm)` (None without `mano`)."""
    rng = np.random.default_rng(0)
    rec, n, ra = {}, 0, 0.0
    for s, d in seqs:
        r = run_sequence(net, cfg, root, d, s, device, rng, "model", "fixed", args.window_ms, 0, 300,
                         clip_run_start=args.clip_run_start, record=rec)
        n += len(r["end"])
        if mano is not None:
            ra += float(EX.per_step_metrics(mano, r, device)["mpjpe_ra_mm"].sum())
    return rec, (ra / n if mano is not None else None)


def run_adabn(net, model, cfg, root, seqs, device, args, mano=None, log=print):
    """`args.iters` rounds of: closed loop of `net` (the model, or a filter around it) recording its inputs -> re-estimate the
    BN statistics of `model` on them (`adapt_bn`). `iters = 0` does nothing at all (the model is not even touched).
    Returns `(per-iteration info, recorder RA per iteration, the original BN snapshot)`."""
    orig = bn_snapshot(model)
    iters_info, ra_by_iter = [], []
    for it in range(args.iters):
        before = bn_snapshot(model)
        rec, ra = record_pass(net, cfg, root, seqs, device, args, mano)
        ra_by_iter.append(ra)
        nb = adapt_bn(model, rec, args.bs, args.seed + it, device)
        after = bn_snapshot(model)
        iters_info.append({"iter": it + 1, "recorded_steps": {s: len(r["x"]) for s, r in rec.items()}, "n_batches": nb,
                           "recorder_ra_mm": ra, "shift_vs_previous": bn_shift(before, after), "shift_vs_original": bn_shift(orig, after)})
        log(f"  iter {it + 1}: recorder RA {ra}  BN shift vs original: mean|d mean| "
            f"{iters_info[-1]['shift_vs_original']['mean_d_mean_abs']:.4g}  ({nb} batches)")
        del rec
    return iters_info, ra_by_iter, orig


def cmd_adabn(args):
    t0 = time.time()
    device = setup(args)
    run = Path(args.run_dir)
    cfg, model, ckpt, step, mano, root, seqs = common_context(args, run, device)
    net = wrap_filter(args, model, device)
    tag = default_tag(args, f"adabn_i{args.iters}_bs{args.bs}")
    n_bn1d = sum(isinstance(m, nn.BatchNorm1d) for m in model.modules())
    print(f"adabn: {run.name} {tag}  BN2d layers={len(bn2d_layers(model))} (BN1d left as is: {n_bn1d})  iters={args.iters} bs={args.bs}", flush=True)
    iters_info, ra_by_iter, orig = run_adabn(net, model, cfg, root, seqs, device, args, mano,
                                             log=lambda m: print(f"{m}  ({time.time() - t0:.0f} s)", flush=True))
    summary, arrays, sink = evaluate_probe(net, None, cfg, mano, root, seqs, device, args, label=f"{run.name} {tag}")
    ra_by_iter.append(ra_parts(summary)["ra_overall"])
    info = {"sub": "adabn", "settings": dict(settings_of(args), iters=args.iters, bs=args.bs, seed=args.seed),
            "transductive": "unlabeled test-time statistics from the evaluated sequences themselves (dev = test)",
            "ra_by_iter": ra_by_iter, "iterations": iters_info, "n_bn2d": len(orig), "n_bn1d_untouched": n_bn1d,
            "bn_shift_final": bn_shift(orig, bn_snapshot(model)) if args.iters else None}
    write_result(args, "adabn", run.name, tag, [str(ckpt)], info, summary, arrays, {}, t0, device)


def cmd_cmp(args):
    """Bit-identity of two evaluation npz over the arrays whose key starts with `--prefix` (default `model|`)."""
    A, B = np.load(args.a), np.load(args.b)
    ka = sorted(k for k in A.files if k.startswith(args.prefix))
    kb = sorted(k for k in B.files if k.startswith(args.prefix))
    missing = sorted(set(ka) ^ set(kb))
    same, worst, bad = 0, 0.0, []
    for k in sorted(set(ka) & set(kb)):
        x, y = A[k], B[k]
        if x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y):
            same += 1
        else:
            d = float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()) if x.shape == y.shape else float("inf")
            worst = max(worst, d)
            bad.append((k, d))
    print(f"cmp {args.prefix}*: {len(set(ka) & set(kb))} arrays compared, {same} bit-identical, {len(bad)} differ, "
          f"{len(missing)} keys on one side only; max|diff| over the differing = {worst!r}")
    for k, d in bad[:20]:
        print(f"  differs: {k}  max|diff| = {d!r}")
    for k in missing[:10]:
        print(f"  one side only: {k}")
    if args.json_a and args.json_b:
        ja, jb = json.loads(Path(args.json_a).read_text()), json.loads(Path(args.json_b).read_text())
        ja, jb = ja.get("model", {}), jb.get("model", {})
        ds = list(leaf_diffs(ja, jb))
        worst_leaf = max(ds, key=lambda t: t[1]) if ds else ("", float("nan"))
        print(f"cmp json model/*: {len(ds)} numeric leaves, {sum(1 for _, d in ds if d == 0)} equal, max|diff| = {worst_leaf[1]!r} at {worst_leaf[0]}; "
              f"RA a = {ja['overall']['mpjpe_ra_mm']!r} b = {jb['overall']['mpjpe_ra_mm']!r}")
        if worst_leaf[1]:
            bad = bad + [("json", worst_leaf[1])]
    sys.exit(1 if (bad or missing) else 0)


def leaf_diffs(a, b, path=""):
    """`(path, |a - b|)` of every numeric leaf two nested json structures share (inf for a structural mismatch)."""
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
    elif isinstance(a, (bool, str)) or isinstance(b, (bool, str)):
        return
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        yield path, (0.0 if (math.isnan(a) and math.isnan(b)) else abs(float(a) - float(b)))
    elif (a is None) != (b is None):
        yield path, float("inf")


def main() -> None:
    com = argparse.ArgumentParser(add_help=False)
    com.add_argument("--run-dir", default=None, help="the run (seed-ens: use --run-dirs)")
    com.add_argument("--ckpt", default="last", help="checkpoint rule of evalx.find_ckpt (selected | last | step=N)")
    com.add_argument("--split", default="val_core")
    com.add_argument("--manifest", default=None)
    com.add_argument("--config", default=None, help="model config (default: the yaml copied into the run)")
    com.add_argument("--window-ms", type=int, default=STEP, help="the fixed evidence window of the loop (ms)")
    com.add_argument("--clip-run-start", action="store_true", help="clip the window to the segment start (training semantics) "
                     "instead of the recording start")
    com.add_argument("--filter-spec", default="", help="wrap the probe model in AdaptiveFilter.from_spec (JSON file / text, ONE spec)")
    com.add_argument("--out-dir", default="", help="default: <repo>/outputs/dt3/<sub-command>")
    com.add_argument("--tag", default="", help="output tag (default: derived from the settings)")
    com.add_argument("--device", default="", help="default: cuda if available (pick the GPU with CUDA_VISIBLE_DEVICES)")
    com.add_argument("--threads", type=int, default=0, help="torch.set_num_threads (0: leave)")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("ens-win", parents=[com], help="multi-window ensemble")
    e.add_argument("--windows", required=True, help="comma separated windows (ms), e.g. 100,200,300")
    ad = sub.add_parser("adabn", parents=[com], help="test-time AdaBN (transductive)")
    ad.add_argument("--iters", type=int, default=2)
    ad.add_argument("--bs", type=int, default=32)
    ad.add_argument("--seed", type=int, default=0, help="seed of the batch shuffles")
    sub.add_parser("tta-pol", parents=[com], help="polarity-swap test-time augmentation")
    s = sub.add_parser("seed-ens", parents=[com], help="average of several runs' models (diagnostic)")
    s.add_argument("--run-dirs", nargs="+", required=True)
    c = sub.add_parser("cmp", help="bit-identity of two evaluation npz")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--prefix", default="model|")
    c.add_argument("--json-a", default="")
    c.add_argument("--json-b", default="")
    args = ap.parse_args()
    if args.cmd == "cmp":
        cmd_cmp(args)
        return
    if args.cmd != "seed-ens" and not args.run_dir:
        ap.error(f"{args.cmd} needs --run-dir")
    if args.cmd == "adabn":
        assert args.iters >= 0 and args.bs >= 2
        cmd_adabn(args)
    else:
        cmd_multipass(args, args.cmd)


if __name__ == "__main__":
    main()
