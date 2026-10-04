#!/usr/bin/env python3
"""Fixed-protocol evaluation of trained arms: the recursive protocol, step by step, plus what the main row
does not show.

eval   one checkpoint of one run on one split. The loop is `semkine.eval_track.track_sequence`
       verbatim (batch 1, fp32, init GT + protocol noise from one rng seeded 0 across the split's
       sequences in order, 50 ms steps) but keeps every step, so the selection JSON's RA must be
       reproduced to < 0.05 mm before anything else is reported. Extra per-step quantities:
       root-rotation geodesic error, absolute translation error, event count, segment / elapsed.
       Controls on the same steps: `hold` (the segment's initial state held) and `noevents`
       (the model fed empty packets in its own closed loop).
row    aggregate N seeds of one arm into `<base>/outputs/semkine/<arm>_main_row.json` (the format
       `tools/report_table.py` reads; "two_seed_mean" holds the N-seed mean) and an extended JSON
       with per-sequence / per-bucket / failure / control statistics and 10 s block-bootstrap CIs.

    python tools/tracking/evalx.py eval --run-dir outputs/semkine/rt_s37_s3407 [--ckpt selected|last|step=N]
                                     [--split val_core --manifest M] [--controls]
    python tools/tracking/evalx.py row --arm rt_s37 --runs outputs/semkine/rt_s37_s3407 outputs/semkine/rt_s37_s3408

Evidence-window options (DT3; every one is off by default and the default evaluation is bit for bit what it was):
`--window-ms W` (a longer past window at the same 50 ms step), `--clip-run-start` (clip the window to the segment's start, as
training does, instead of only to the recording's start), `--count-mode norm` (hand a count-aware filter the window's event
count rescaled to a 50 ms rate), `--window-set 150,200,300 --min-events N` (the smallest listed window holding >= N events).
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))
sys.path.insert(0, str(REPO / "tools"))
from config import load_config                                        # noqa: E402
from mano_layer import ManoLayer                                      # noqa: E402
from model import MNISTModel                                          # noqa: E402
from pose_repr import decode_to_mano_inputs                           # noqa: E402
from semkine import eval_track as ET                                  # noqa: E402
from semkine.eval_track import mano_fk                                # noqa: E402
from semkine import events as EV                                      # noqa: E402
from semkine.dataset import sequences_for_split                       # noqa: E402

STEP = 50
BLOCK_MS = 10_000
FAIL_MM, FAIL_ROT_DEG, FAIL_MIN_STEPS = 50.0, 30.0, 20      # >= 1 s above either threshold
EVENT_BUCKETS = (0, 500, 2000, 10000, 10**9)


def find_ckpt(run: Path, which: str, split_tag: str = "val_core"):
    if which == "selected":
        sel = json.loads(sorted(run.glob(f"selection_{split_tag}_step50*.json"))[0].read_text())["selected"]
        return Path(sel["ckpt"]), int(sel["step"]), float(sel["mpjpe_ra_mm"])
    if which == "last":
        steps = sorted(int(re.search(r"step=(\d+)", p.name).group(1)) for p in run.glob("*step=*.ckpt"))
        which = f"step={steps[-1]}"
    st = int(which.split("=")[1])
    return next(run.glob(f"*-step={st}.ckpt")), st, None


def aa_to_R(aa):
    th = aa.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    k = aa / th
    K = torch.zeros(*aa.shape[:-1], 3, 3, dtype=aa.dtype)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    I = torch.eye(3, dtype=aa.dtype).expand_as(K)
    s, c = th.sin()[..., None], th.cos()[..., None]
    return I + s * K + (1 - c) * (K @ K)


def rot_err_deg(p51, g51):
    Ra = aa_to_R(torch.from_numpy(p51[:, 3:6]).double())
    Rb = aa_to_R(torch.from_numpy(g51[:, 3:6]).double())
    tr = (Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(((tr - 1) / 2).clamp(-1, 1).acos()).numpy()


def window_of(offsets, end, wmode, win, min_events, max_win, floor=0, wset=()):
    """Evidence window (ms) ending at `end`. fixed: `win`; adaptive: the shortest window >= `win`
    holding >= `min_events` events, capped at `max_win`. Causal either way (only past events).

    `floor` (ms, default 0 = the recording's start): the earliest ms the window may reach back to; the window is
    clipped to `lim = end + 1 - floor`. Training clips a window to its segment's start (a window never spans the gap
    between two valid runs), `run_sequence(clip_run_start=True)` passes the segment start here. `wset` (default empty):
    a quantized hybrid window -- the smallest member of `sorted(wset)` whose raw event count (clipped to `lim`) reaches
    `min_events`, else the largest member, then clipped to `lim`; it replaces `wmode` / `win` / `max_win`. With the
    defaults this is the original function, bit for bit."""
    lim = end + 1 - floor
    if len(wset):
        ws = sorted(wset)
        w = ws[-1]
        for m in ws:
            if offsets[end + 1] - offsets[end - min(m, lim) + 1] >= min_events:
                w = m
                break
        return int(min(w, lim))
    if wmode == "fixed":
        return int(min(win, lim))
    w = int(min(win, lim))
    while w < max_win and w < lim and offsets[end + 1] - offsets[end - w + 1] < min_events:
        w = min(w + 10, max_win, lim)
    return int(w)


COUNT_MODES = ("last50", "norm")


def parse_window_set(text):
    """`--window-set`: '150,200,300' (or any iterable of ints) -> the sorted tuple of distinct positive windows; '' -> ()."""
    parts = [p for p in str(text).split(",") if p.strip()] if isinstance(text, str) else list(text or ())
    ws = sorted({int(p) for p in parts})
    if any(w <= 0 for w in ws):
        raise ValueError(f"window set {text!r}: every window must be a positive number of ms")
    return tuple(ws)


def window_opts(a):
    """`(clip_run_start, count_mode, window_set)` of an evaluation's args `a` (an argparse Namespace; a field it does not
    have is the default, so a Namespace built before these options existed keeps working)."""
    return (bool(getattr(a, "clip_run_start", False)), getattr(a, "count_mode", "last50") or "last50",
            parse_window_set(getattr(a, "window_set", ()) or ()))


def check_window_opts(a):
    """Raise ValueError when the window options of `a` contradict each other (the CLIs turn it into a parser error)."""
    clip, cmode, wset = window_opts(a)
    if cmode not in COUNT_MODES:
        raise ValueError(f"--count-mode {cmode!r}: choose from {COUNT_MODES}")
    if wset:
        if getattr(a, "min_events", 0) <= 0:
            raise ValueError("--window-set needs --min-events N > 0 (the event count that decides the window)")
        if getattr(a, "window_mode", "fixed") != "fixed" or getattr(a, "window_ms", STEP) != STEP:
            raise ValueError("--window-set replaces --window-mode / --window-ms; give one or the other")


def run_options(a):
    """The keyword arguments `evaluate` adds to its `run_sequence` call: only the non-default window options, so an
    evaluation with the defaults calls `run_sequence` exactly as before (a monkeypatched older signature keeps working)."""
    clip, cmode, wset = window_opts(a)
    kw = {}
    if clip:
        kw["clip_run_start"] = True
    if cmode != "last50":
        kw["count_mode"] = cmode
    if wset:
        kw["wset"] = wset
    return kw


def wset_tag(wset, min_events):
    """'_wset150-200-300_n5000': the file-tag part of a window set (it replaces the fixed / adaptive window tag)."""
    return f"_wset{'-'.join(str(w) for w in wset)}_n{min_events}"


def option_tag(a):
    """'_clip' (segment-clipped windows) and '_cnorm' (rate-normalized event count): the file-tag parts of the other
    window options; '' by default."""
    clip, cmode, _ = window_opts(a)
    return ("_clip" if clip else "") + ("_cnorm" if cmode == "norm" else "")


def eval_tag(a):
    """File stem of `cmd_eval`'s outputs: `evalx_<split>_<ckpt>[_tf][_pert]<window>[_clip][_cnorm][_<suffix>]`, where
    `<window>` is `_wset<a>-<b>-<c>_n<N>` for a window set, else (a window other than the fixed 50 ms) `_w<mode><ms>`
    (+ `_n<min_events>_x<max>` when adaptive). Without the new options this is the historical name."""
    wset = window_opts(a)[2]
    tag = f"evalx_{a.split}_{a.ckpt.replace('=', '')}"
    tag += "_tf" if a.tf else ""
    tag += "_pert" if a.perturb else ""
    if wset:
        tag += wset_tag(wset, a.min_events)
    elif a.window_mode != "fixed" or a.window_ms != STEP:
        tag += f"_w{a.window_mode}{a.window_ms}" + (f"_n{a.min_events}_x{a.max_window_ms}" if a.window_mode == "adaptive" else "")
    tag += option_tag(a)
    tag += f"_{a.suffix}" if a.suffix else ""
    return tag


def window_record(a):
    """The json `window` entry of an evaluation: mode / ms / min_events / max_ms as ever; when any window option is on, also
    `clip_run_start`, `count_mode`, `window_set` (and mode `set` for a window set)."""
    clip, cmode, wset = window_opts(a)
    rec = {"mode": a.window_mode, "ms": a.window_ms, "min_events": a.min_events, "max_ms": a.max_window_ms}
    if clip or cmode != "last50" or wset:
        rec.update(mode="set" if wset else a.window_mode, clip_run_start=clip, count_mode=cmode, window_set=list(wset))
    return rec


@torch.no_grad()
def run_sequence(model, cfg, root, d, seq, device, rng, mode="model", wmode="fixed", win=STEP,
                 min_events=0, max_win=300, clip_run_start=False, count_mode="last50", wset=()):
    """`track_sequence`'s loop, keeping every step. mode: model | hold | noevents | tf.
    `tf` (teacher forcing) feeds the ground truth of the previous evaluated step back instead of the
    prediction (the first step of a segment keeps the protocol's noisy initial state), so its error is
    the single-step error under a clean state. `prev` records the state each step was conditioned on.
    The evaluated steps never change (ends a+49, a+99, ...); only the evidence window may: `wmode` / `win` /
    `min_events` / `max_win` (fixed or adaptive), or `wset` (a quantized set of windows, see `window_of`), and
    `clip_run_start` clips it to the segment's start `a` (training's convention; by default only to the recording's
    start, so a window may reach back over the gap before a segment). The window of every step is returned in `win`.

    A model with `reset_state` is reset at the start of every segment (no random numbers consumed); one with
    `get_state` has its state recorded before every step in the returned `fstate` (a list parallel to `prev`, not
    written to the npz); one with `takes_event_count` is called `model(x, prev, n_events=n)` with the raw event count
    of the 50 ms packet (0 in mode `noevents`); `count_mode="norm"` hands it instead the window's raw count rescaled to a
    50 ms rate, `float(n_win) * (STEP / w)` with `n_win` the events of the window `w` (identical to the 50 ms count at
    w = 50; `r["count"]`, hence the event buckets, stays the 50 ms count and `r["count_f"]` holds the float handed to the
    model). Bare models and `FilteredTracker` are called as before."""
    if count_mode not in COUNT_MODES:
        raise ValueError(f"count_mode {count_mode!r}: choose from {COUNT_MODES}")
    events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
    tsub_path = root / d / f"{seq}_tsub.npy"
    tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
    use_raw = bool(getattr(model, "encoder_name", ""))
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    camera_K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    if hasattr(model, "set_hand_context"):
        model.set_hand_context(betas, camera_K)
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
    ev_ch = EV.event_channels(cfg)
    # A stateful filter (semkine.anchored.AdaptiveFilter): `reset_state(prev)` at each segment start, `get_state()` before
    # each step (the state the step is conditioned on, parallel to `prev`) for `perturb_trials`; a model with
    # `takes_event_count` is also handed the packet's raw event count. Bare models and FilteredTracker have none of these.
    stateful = hasattr(model, "get_state")
    with_count = bool(getattr(model, "takes_event_count", False))
    preds, gts, ends_all, runs_all, elapsed, counts, prevs, fstate = [], [], [], [], [], [], [], []
    wins, counts_f = [], []
    for run_id, (a, b) in enumerate(runs):
        ends = np.arange(a + STEP - 1, b, STEP, dtype=np.int64)
        if not len(ends):
            continue
        floor = int(a) if clip_run_start else 0                 # a window never reaches before the segment (training's rule)
        prev = pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)
        prev_t = torch.from_numpy(prev).view(1, -1).to(device)
        init_t = prev_t.clone()
        if hasattr(model, "reset_state"):
            model.reset_state(prev_t)
        for end in ends:
            n_ev = int(offsets[int(end) + 1] - offsets[int(end) - STEP + 1])
            w = window_of(offsets, int(end), wmode, win, min_events, max_win, floor=floor, wset=wset)
            if count_mode == "norm":
                # the window's events at a 50 ms rate; python float arithmetic on purpose: a device float64 `n.to(float64) * (50.0 / w)`
                # computes the very same product
                n_win = int(offsets[int(end) + 1] - offsets[int(end) - w + 1])
                n_cnt = float(n_win) * (STEP / w)
                counts_f.append(n_cnt)
            else:
                n_cnt = n_ev
            prevs.append(prev_t.cpu().numpy()[0])
            if stateful:
                fstate.append(model.get_state())
            if mode == "hold":
                pred = init_t
            elif use_raw:
                ev5 = ET._window_events(events, offsets, tsub, int(end), w)
                if mode == "noevents":
                    ev5 = ev5[:0]
                pred = model.forward_packet(ET.make_eval_packet(ev5, prev_t, betas, camera_K, w, device))
            else:
                x = torch.from_numpy(ET.build_lnes(events, offsets, int(end), w, ev_ch)).unsqueeze(0).to(device)
                if mode == "noevents":
                    x = torch.zeros_like(x)
                if with_count:
                    pred = model(x, prev_t, n_events=0 if mode == "noevents" else n_cnt)
                else:
                    pred = model(x, prev_t)
            prev_t = pred if mode != "tf" else torch.from_numpy(pos51[int(end)].copy()).view(1, -1).to(device)
            preds.append(pred.cpu().numpy()[0])
            gts.append(pos51[end])
            ends_all.append(int(end))
            runs_all.append(run_id)
            elapsed.append(int(end - a))
            counts.append(n_ev)
            wins.append(w)
    r = {"pred": np.stack(preds).astype(np.float32), "gt": np.stack(gts).astype(np.float32),
         "end": np.asarray(ends_all), "run": np.asarray(runs_all), "elapsed": np.asarray(elapsed),
         "count": np.asarray(counts), "betas": aux["betas"],
         "prev": np.stack(prevs).astype(np.float32), "fstate": fstate, "win": np.asarray(wins, dtype=np.int64)}
    if count_mode != "last50":
        r["count_f"] = np.asarray(counts_f, dtype=np.float64)
    return r


def aa_compose(rot_aa, aa):
    """Axis-angle of `Exp(rot_aa) @ Exp(aa)` (a rotation applied on the left, camera frame)."""
    from scipy.spatial.transform import Rotation as Rot
    return (Rot.from_rotvec(rot_aa) * Rot.from_rotvec(aa)).as_rotvec().astype(np.float32)


@torch.no_grad()
def perturb_trials(model, cfg, root, d, seq, device, r, thetas=(10.0, 20.0), horizon=20, every=20,
                   min_elapsed=1000, seed=0):
    """Perturbation recovery, branched off the protocol closed loop `r` (a `run_sequence` result).

    Every `every` steps (once the segment is `min_elapsed` ms old and `horizon` steps remain), the state
    the closed loop fed into that step is rotated by `theta` degrees about a random axis (left, camera
    frame) and the model is rolled `horizon` steps on the same packets. Returns, per theta, the excess
    root-rotation error over the unperturbed loop at the same steps, `(trials, horizon)` degrees: k = 0
    is the first output after the perturbed state. Returned per theta: `div` -- the geodesic angle
    between the perturbed branch's output and the unperturbed loop's output at the same step, i.e. how
    much of the injected state error is still there (div / theta is the retention curve); and
    `excess` -- the branch's root error minus the loop's (what the perturbation costs in accuracy;
    with a loop error of the same size as theta this understates the retained error).
    """
    events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
    tsub_path = root / d / f"{seq}_tsub.npy"
    tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
    use_raw = bool(getattr(model, "encoder_name", ""))
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    camera_K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    if hasattr(model, "set_hand_context"):
        model.set_hand_context(betas, camera_K)
    ev_ch = EV.event_channels(cfg)
    # a stateful filter (AdaptiveFilter) branches from the loop's own state snapshot `r["fstate"]`; a model that takes
    # the packet's event count gets `r["count_f"]` when the loop handed it a rate-normalized count, else `r["count"]`
    # (the same count the closed loop used); the branch steps use the closed loop's window of the same step `r["win"]`
    # (50 ms when `r` has none)
    stateful = hasattr(model, "set_state")
    if stateful and not r.get("fstate"):
        raise ValueError("a stateful model needs the per-step state snapshots `fstate` of the closed loop `r`")
    with_count = bool(getattr(model, "takes_event_count", False))
    base = rot_err_deg(r["pred"], r["gt"])
    rng = np.random.default_rng(seed)
    out = {float(t): {"div": [], "excess": []} for t in thetas}
    order = np.arange(len(r["end"]))
    for run_id in np.unique(r["run"]):
        idx = order[r["run"] == run_id]
        for i0 in idx[r["elapsed"][idx] >= min_elapsed][::every]:
            if i0 + horizon > idx[-1] + 1:
                continue
            axis = rng.standard_normal(3)
            axis /= np.linalg.norm(axis)
            for th in out:
                prev = r["prev"][i0].copy()
                prev[3:6] = aa_compose(axis * np.deg2rad(th), prev[3:6])
                prev_t = torch.from_numpy(prev).view(1, -1).to(device)
                if stateful:                           # the branch starts from the state the closed loop had at step i0
                    model.set_state(r["fstate"][i0])
                preds = []
                for i in range(i0, i0 + horizon):
                    end = int(r["end"][i])
                    w = int(r["win"][i]) if "win" in r else STEP
                    if use_raw:
                        ev5 = ET._window_events(events, offsets, tsub, end, w)
                        pred = model.forward_packet(ET.make_eval_packet(ev5, prev_t, betas, camera_K, w, device))
                    else:
                        x = torch.from_numpy(ET.build_lnes(events, offsets, end, w, ev_ch)).unsqueeze(0).to(device)
                        if with_count:
                            pred = model(x, prev_t, n_events=float(r["count_f"][i]) if "count_f" in r else int(r["count"][i]))
                        else:
                            pred = model(x, prev_t)
                    prev_t = pred
                    preds.append(pred.cpu().numpy()[0])
                br = np.stack(preds).astype(np.float32)
                out[th]["div"].append(rot_err_deg(br, r["pred"][i0:i0 + horizon]))
                out[th]["excess"].append(rot_err_deg(br, r["gt"][i0:i0 + horizon]) - base[i0:i0 + horizon])
    return {th: {k: np.stack(v) if v else np.zeros((0, horizon)) for k, v in d.items()} for th, d in out.items()}


def motion_drift(r, m):
    """Is a low error just a frozen or over-smoothed output? Root angular speed of prediction and
    ground truth (deg per step, within segments), finger-parameter speed, and the root-error trend
    over a segment's elapsed time (deg per 10 s, least squares, segments >= 10 s)."""
    def ang(a, b):
        return rot_err_deg(a.astype(np.float32), b.astype(np.float32))
    same = r["run"][1:] == r["run"][:-1]
    p, g = r["pred"], r["gt"]
    ps, gs = ang(p[1:], p[:-1])[same], ang(g[1:], g[:-1])[same]
    pf = np.linalg.norm(p[1:, 6:] - p[:-1, 6:], axis=-1)[same]
    gf = np.linalg.norm(g[1:, 6:] - g[:-1, 6:], axis=-1)[same]
    slopes = []
    for rr in np.unique(r["run"]):
        sel = r["run"] == rr
        t = r["elapsed"][sel] / 10_000.0
        if t.max() - t.min() >= 1.0:
            slopes.append(float(np.polyfit(t, m["root_rot_deg"][sel], 1)[0]))
    return {"root_speed_pred_deg": float(ps.mean()), "root_speed_gt_deg": float(gs.mean()),
            "root_speed_ratio": float(ps.mean() / max(gs.mean(), 1e-9)),
            "root_speed_abs_err_deg": float(np.abs(ps - gs).mean()),
            "finger_speed_ratio": float(pf.mean() / max(gf.mean(), 1e-9)),
            "root_err_slope_deg_per_10s": slopes}


#: the three blocks of the 51-D state the jitter decomposition attributes motion to
JIT_PARTS = {"transl": slice(0, 3), "root": slice(3, 6), "fingers": slice(6, 51)}
#: a GT step this small (mm, mean over joints) counts as static (model/eval_track.py's threshold)
STATIC_MM = 0.5


def _rot_second_diff_deg(aa):
    """Angle (deg) between consecutive step rotations: with dR_t = R_t^T R_{t+1}, the geodesic angle of
    dR_{t-1}^T dR_t, i.e. the root's angular acceleration per step^2. `aa` is (N, 3) axis-angle."""
    from semkine.anchored import _qmul, _quat
    q = _quat(aa.astype(np.float64))
    conj = np.array([1.0, -1.0, -1.0, -1.0])
    d = _qmul(q[:-1] * conj, q[1:])
    dd = _qmul(d[:-1] * conj, d[1:])
    return np.rad2deg(2.0 * np.arctan2(np.linalg.norm(dd[:, 1:], axis=-1), np.abs(dd[:, 0])))


def jitter_decomp(mano, r, device, static_mm=STATIC_MM):
    """Where the output's frame-to-frame motion comes from (DT round, docs/DT_RENDER_TRACK_PREREG.md).

    On the absolute (non-aligned) MANO joints of the prediction, the ground truth and counterfactual
    mixes -- the ground truth with one block of the state taken from the prediction (`only_<block>`), and
    the prediction with one block taken back from the ground truth (`allbut_<block>`) -- per step within
    a segment: the first difference (`jit_*`, mm per step, the quantity model/eval_track.py calls jitter)
    and the second difference (`acc_*`, mm per step^2). `acc_err_mm` is the acceleration of the error
    (pred - gt): it is zero for a perfectly tracked motion however fast, so unlike `jit_pred` it does not
    credit the ground truth's own motion. `*_static` restrict to steps where the GT moved < `static_mm`.
    Root angular acceleration (deg per step^2) is reported for the prediction and the ground truth."""
    betas = torch.tensor(r["betas"], dtype=torch.float32, device=device).view(1, -1)
    run = r["run"]
    same1 = run[1:] == run[:-1]
    same2 = same1[1:] & same1[:-1]
    pred, gt = r["pred"].astype(np.float32), r["gt"].astype(np.float32)

    def joints(p51):
        return mano_fk(mano, p51, betas, device)[0]

    def first(j):                                   # (N-1,) mm, mean over joints
        return np.linalg.norm(j[1:] - j[:-1], axis=-1).mean(-1) * 1000

    def second(j):                                  # (N-2,) mm per step^2
        return np.linalg.norm(j[2:] - 2 * j[1:-1] + j[:-2], axis=-1).mean(-1) * 1000

    pj, gj = joints(pred), joints(gt)
    gmove = first(gj)
    static1 = (gmove < static_mm) & same1
    static2 = static1[1:] & static1[:-1]
    m1 = lambda v: float(v[same1].mean()) if same1.any() else float("nan")            # noqa: E731
    m2 = lambda v: float(v[same2].mean()) if same2.any() else float("nan")            # noqa: E731
    out = {"n_steps": int(same1.sum()), "n_static": int(static1.sum()),
           "jit_gt_mm": m1(gmove), "jit_pred_mm": m1(first(pj)),
           "acc_gt_mm": m2(second(gj)), "acc_pred_mm": m2(second(pj)),
           "acc_err_mm": m2(second(pj - gj))}
    out["acc_ratio"] = out["acc_pred_mm"] / max(out["acc_gt_mm"], 1e-9)
    out["jit_ratio"] = out["jit_pred_mm"] / max(out["jit_gt_mm"], 1e-9)
    # Root-aligned counterparts (the main table's convention: joint 0 subtracted from all 21 joints, mean over the 21
    # joints as `per_step_metrics`). Translation drops out of root-aligned joints, so `*_ra_*` measure the jitter of the
    # rotation and the fingers only; `acc_err_ra_only_transl_mm` is ~0 by construction (a sanity check).
    ra = lambda x: x - x[:, :1]                                                       # noqa: E731
    pr, gr = ra(pj), ra(gj)
    out.update({"jit_ra_gt_mm": m1(first(gr)), "jit_ra_pred_mm": m1(first(pr)),
                "acc_ra_gt_mm": m2(second(gr)), "acc_ra_pred_mm": m2(second(pr)), "acc_err_ra_mm": m2(second(pr - gr))})
    out["acc_ratio_ra"] = out["acc_ra_pred_mm"] / max(out["acc_ra_gt_mm"], 1e-9)
    out["jit_ratio_ra"] = out["jit_ra_pred_mm"] / max(out["jit_ra_gt_mm"], 1e-9)
    if static1.any():
        out["jit_pred_static_mm"] = float(first(pj)[static1].mean())
    if static2.any():
        out["acc_err_static_mm"] = float(second(pj - gj)[static2].mean())
    for name, sl in JIT_PARTS.items():
        only = gt.copy()
        only[:, sl] = pred[:, sl]
        allbut = pred.copy()
        allbut[:, sl] = gt[:, sl]
        jo, ja = joints(only), joints(allbut)
        out[f"jit_only_{name}_mm"] = m1(first(jo))
        out[f"acc_err_only_{name}_mm"] = m2(second(jo - gj))
        out[f"acc_err_allbut_{name}_mm"] = m2(second(ja - gj))
        out[f"acc_err_ra_only_{name}_mm"] = m2(second(ra(jo) - gr))
        out[f"acc_err_ra_allbut_{name}_mm"] = m2(second(ra(ja) - gr))
    rp, rg = _rot_second_diff_deg(pred[:, 3:6]), _rot_second_diff_deg(gt[:, 3:6])
    out["rot_acc_pred_deg"], out["rot_acc_gt_deg"] = m2(rp), m2(rg)
    return out


def per_step_metrics(mano, r, device):
    betas = torch.tensor(r["betas"], dtype=torch.float32, device=device).view(1, -1)
    pj, pv = mano_fk(mano, r["pred"], betas, device)
    gj, gv = mano_fk(mano, r["gt"], betas, device)
    ra = lambda x: x - x[..., :1, :]                                            # noqa: E731
    return {"mpjpe_ra_mm": np.linalg.norm(ra(pj) - ra(gj), axis=-1).mean(-1) * 1000,
            "mpvpe_ra_mm": np.linalg.norm(ra(pv) - ra(gv), axis=-1).mean(-1) * 1000,
            "mpjpe_abs_mm": np.linalg.norm(pj - gj, axis=-1).mean(-1) * 1000,
            "mpvpe_abs_mm": np.linalg.norm(pv - gv, axis=-1).mean(-1) * 1000,
            "root_rot_deg": rot_err_deg(r["pred"], r["gt"]),
            "transl_mm": np.linalg.norm(r["pred"][:, :3] - r["gt"][:, :3], axis=-1) * 1000}


def failures(m, run):
    """Continuous tracking failure: >= FAIL_MIN_STEPS consecutive steps (1 s) with RA > 50 mm or
    root rotation > 30 deg, within one segment."""
    bad = (m["mpjpe_ra_mm"] > FAIL_MM) | (m["root_rot_deg"] > FAIL_ROT_DEG)
    eps, cur, cur_run = [], 0, None
    for b, rr in zip(bad, run):
        if rr != cur_run:                       # a new segment closes any open episode
            if cur >= FAIL_MIN_STEPS:
                eps.append(cur)
            cur, cur_run = 0, rr
        if b:
            cur += 1
        else:
            if cur >= FAIL_MIN_STEPS:
                eps.append(cur)
            cur = 0
    if cur >= FAIL_MIN_STEPS:
        eps.append(cur)
    return {"bad_step_frac": float(bad.mean()), "episodes": len(eps),
            "fail_time_frac": float(sum(eps) / max(len(bad), 1)),
            "longest_s": float(max(eps) * STEP / 1000) if eps else 0.0}


def block_ci(vals, end, n_boot=2000, seed=0):
    blocks = end // BLOCK_MS
    ub = np.unique(blocks)
    sums = np.array([vals[blocks == b].sum() for b in ub])
    cnts = np.array([(blocks == b).sum() for b in ub])
    rng = np.random.default_rng(seed)
    bs = [sums[i].sum() / cnts[i].sum() for i in (rng.integers(0, len(ub), len(ub)) for _ in range(n_boot))]
    return [float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def evaluate(model, cfg, mano, root, seqs, device, a, label=""):
    """The protocol loop on `seqs` for any model object (a trained arm or a composite such as
    `semkine.anchored.AnchoredTracker`): the modes `a` asks for (model, controls, tf), perturbation
    recovery, per-sequence CIs / failures / event-rate buckets / motion. Returns `(summary, arrays)`.

    `a` holds `controls`, `tf`, `perturb`, `window_mode`, `window_ms`, `min_events`, `max_window_ms` and, optionally,
    `clip_run_start`, `count_mode`, `window_set` (`window_opts`: a Namespace without them is the default evaluation)."""
    modes = ["model"] + (["hold", "noevents"] if a.controls else []) + (["tf"] if a.tf else [])
    check_window_opts(a)
    wopt = run_options(a)                       # {} unless a DT3 window option is on: run_sequence is then called as ever
    out, t0 = {}, time.time()
    arrays = {}
    for mode in modes:
        rng = np.random.default_rng(0)
        res = {}
        for s, d in seqs:
            r = run_sequence(model, cfg, root, d, s, device, rng, mode, a.window_mode, a.window_ms,
                             a.min_events, a.max_window_ms, **wopt)
            m = per_step_metrics(mano, r, device)
            res[s] = (r, m)
            for k, v in m.items():
                arrays[f"{mode}|{s}|{k}"] = v.astype(np.float32)
            if mode == "model":
                # the per-step window (and the rate-normalized count) are saved only by a run with a DT3 window option on,
                # so every earlier invocation keeps its npz key set
                for k in ("pred", "gt", "end", "run", "elapsed", "count") + (("win",) + (("count_f",) if "count_f" in r else ())
                                                                              if wopt else ()):
                    arrays[f"{mode}|{s}|{k}"] = r[k]
                if a.perturb:
                    for th, dd in perturb_trials(model, cfg, root, d, s, device, r).items():
                        for kind, v in dd.items():
                            arrays[f"perturb|{s}|{th:g}|{kind}"] = v.astype(np.float32)
        n = sum(len(m["mpjpe_ra_mm"]) for _, m in res.values())
        summ = {"n_frames": n, "overall": {k: float(sum(m[k].sum() for _, m in res.values()) / n)
                                           for k in next(iter(res.values()))[1]}}
        for s, (r, m) in res.items():
            e = r["end"] + 0
            ent = {k: block_ci(v, e) for k, v in m.items()}
            ent["n_frames"] = int(len(e))
            ent["failure"] = failures(m, r["run"])
            ent["by_events"] = {}
            for lo, hi in zip(EVENT_BUCKETS[:-1], EVENT_BUCKETS[1:]):
                sel = (r["count"] >= lo) & (r["count"] < hi)
                if sel.sum() >= 20:
                    ent["by_events"][f"[{lo},{hi})"] = {"n": int(sel.sum()),
                                                        "mpjpe_ra_mm": float(m["mpjpe_ra_mm"][sel].mean()),
                                                        "root_rot_deg": float(m["root_rot_deg"][sel].mean())}
            if mode == "model":
                ent["motion"] = motion_drift(r, m)
                ent["jitter"] = jitter_decomp(mano, r, device)
            summ[s] = ent
        if mode == "model":
            # frame-weighted mean of the per-sequence jitter decomposition
            seqs_j = [(summ[s]["n_frames"], summ[s]["jitter"]) for s in summ if isinstance(summ[s], dict)
                      and "jitter" in summ[s]]
            keys = sorted({k for _, j in seqs_j for k, v in j.items() if isinstance(v, float)})
            # each key is averaged over the sequences that have it (a `*_static` key is absent from a
            # sequence without static steps), weighted by those sequences' frames
            summ["jitter"] = {k: float(sum(n_ * j[k] for n_, j in seqs_j if k in j)
                                       / sum(n_ for n_, j in seqs_j if k in j)) for k in keys}
        out[mode] = summ
        print(f"  {label} {mode}: RA={summ['overall']['mpjpe_ra_mm']:.4f} "
              f"rot={summ['overall']['root_rot_deg']:.2f} deg  ({time.time() - t0:.0f} s)", flush=True)
    if a.tf:
        cl, tf = out["model"]["overall"], out["tf"]["overall"]
        out["amplification"] = {k: cl[k] / max(tf[k], 1e-9) for k in ("mpjpe_ra_mm", "root_rot_deg")}
    if a.perturb:
        pert = {}
        for th in sorted({k.split("|")[2] for k in arrays if k.startswith("perturb|")}, key=float):
            cat = {kind: np.concatenate([v for k, v in arrays.items() if k.startswith("perturb|")
                                         and k.endswith(f"|{th}|{kind}")]) for kind in ("div", "excess")}
            div, exc = cat["div"].mean(0), cat["excess"].mean(0)
            half = next((k for k, v in enumerate(div) if v < float(th) / 2), None)
            pert[th] = {"trials": int(len(cat["div"])), "div_deg": [float(v) for v in div],
                        "excess_deg": [float(v) for v in exc],
                        "retention_k1_k2_k5_k10_k20": [float(div[k] / float(th)) for k in (0, 1, 4, 9, 19)],
                        "excess_k1_k2_k5_k10_k20_deg": [float(exc[k]) for k in (0, 1, 4, 9, 19)],
                        "half_life_steps": half}
        out["perturb"] = pert
        print(f"  perturbation retention: { {th: v['retention_k1_k2_k5_k10_k20'] for th, v in pert.items()} }", flush=True)
    return out, arrays


def cmd_eval(a):
    run = Path(a.run_dir)
    cfg_path = a.config or next(iter(sorted(run.glob("*.yaml"))), None) or json.loads(
        (run / "training_metadata.json").read_text())["config_path"]
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, step, sel_ra = find_ckpt(run, a.ckpt)
    model = MNISTModel.load_from_checkpoint(str(ckpt), cfg=cfg, map_location=device).to(device).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    mani = Path(a.manifest) if a.manifest else Path(cfg["DATA"]["SPLITS_MANIFEST"])
    seqs = sequences_for_split(root, a.split, mani)
    trained = json.loads((run / "training_metadata.json").read_text()).get("train_sequences", [])
    leak = {s.split("_")[0] for s, _ in seqs} & {s.split("_")[0] for s in trained}
    assert not leak, f"{a.split} subjects {sorted(leak)} are in {run.name}'s training set"
    out = {"run": run.name, "ckpt": str(ckpt), "step": step, "split": a.split, "manifest": str(mani)}
    summary, arrays = evaluate(model, cfg, mano, root, seqs, device, a, label=f"{run.name} step={step}")
    out.update(summary)
    out["window"] = window_record(a)
    if sel_ra is not None and a.split == "val_core" and out["window"]["mode"] == "fixed" and a.window_ms == STEP:
        drift = abs(out["model"]["overall"]["mpjpe_ra_mm"] - sel_ra)
        out["selection_drift_mm"] = drift
        assert drift < 0.05, f"re-run drift {drift:.4f} mm vs selection"
    tag = eval_tag(a)
    np.savez_compressed(run / f"{tag}.npz", **arrays)
    (run / f"{tag}.json").write_text(json.dumps(out, indent=1))
    print("wrote", run / f"{tag}.json", flush=True)


# ------------------------------------------------------------------------------------ cost
@torch.no_grad()
def latency_anchor(device, iters=300, rounds=8):
    m = MNISTModel(load_config(REPO / "configs/eventhands_abs_full51.yaml")).to(device).eval()
    x = torch.rand(1, 180, 240, 2, device=device)
    prev = torch.randn(1, 51, device=device) * 0.02
    prev[:, 2] += 0.4
    for _ in range(iters):
        m(x, prev)
    torch.cuda.synchronize()
    best = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            m(x, prev)
        torch.cuda.synchronize()
        best.append((time.perf_counter() - t0) / iters * 1e3)
    return min(best)


@torch.no_grad()
def latency_model(model, cfg, device, max_packets=600, passes=3):
    """Mean per-step forward time over real 50 ms packets of `lyq_local`, staged on the GPU (the
    sequence every recorded latency used; a training sequence, irrelevant for timing)."""
    root = Path(cfg["DATA"]["ROOT"])
    events, offsets, aux, pos51 = ET.load_sequence(root, "train", "lyq_local")
    tsub = np.load(root / "train" / "lyq_local_tsub.npy", mmap_mode="r")
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    if hasattr(model, "set_hand_context"):
        model.set_hand_context(betas, K)
    raw = bool(getattr(model, "encoder_name", ""))
    with_count = bool(getattr(model, "takes_event_count", False))
    ev_ch = EV.event_channels(cfg)
    items = []
    for a, b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
        prev = torch.from_numpy(pos51[a].copy()).view(1, -1).to(device)
        for end in np.arange(a + STEP - 1, b, STEP, dtype=np.int64):
            if raw:
                items.append(ET.make_eval_packet(ET._window_events(events, offsets, tsub, int(end), STEP),
                                                 prev, betas, K, STEP, device))
            else:
                items.append((torch.from_numpy(ET.build_lnes(events, offsets, int(end), STEP, ev_ch))
                              .unsqueeze(0).to(device), prev)
                             + ((int(offsets[int(end) + 1] - offsets[int(end) - STEP + 1]),) if with_count else ()))
            if len(items) >= max_packets:
                break
        if len(items) >= max_packets:
            break
    if raw:
        f = lambda it: model.forward_packet(it)                                    # noqa: E731
    elif with_count:                                  # a filter that scales its gains by the packet's raw event count
        f = lambda it: model(it[0], it[1], n_events=it[2])                         # noqa: E731
    else:
        f = lambda it: model(*it)                                                  # noqa: E731
    reset = lambda: model.reset_state(items[0][1]) if hasattr(model, "reset_state") and not raw else None  # noqa: E731
    reset()
    for it in items[:100]:
        f(it)
    torch.cuda.synchronize()
    means = []
    for _ in range(passes):
        reset()
        t0 = time.perf_counter()
        for it in items:
            f(it)
        torch.cuda.synchronize()
        means.append((time.perf_counter() - t0) / len(items) * 1e3)
    return min(means)


def macs_of(cfg):
    from thop import profile
    m = copy.deepcopy(MNISTModel(cfg).eval())
    if not str(cfg["MODEL"].get("ENCODER", "")):
        class _W(torch.nn.Module):
            def __init__(s, m):
                super().__init__()
                s.m = m

            def forward(s, x, p):
                return s.m(x, p)
        macs, _ = profile(_W(m), inputs=(torch.rand(1, 180, 240, 2), torch.zeros(1, 51)), verbose=False)
        return float(macs)
    import make_s36_row as MR
    return MR.macs_forward_packet(cfg)[0]


def aggregate_runs(runs, split, ckpt, variant=""):
    """Per-seed overall / local / global metrics of evaluated runs and their N-seed mean, from each
    run's `evalx_<split>_<ckpt>[_<variant>].json`. Returns `(per_seed, ext, mean, cfg)`."""
    a = argparse.Namespace(runs=runs, split=split, ckpt=ckpt, variant=variant)
    per_seed, ext, cfg = {}, {}, None
    for rd in a.runs:
        run = Path(rd)
        js = json.loads((run / f"evalx_{a.split}_{a.ckpt.replace('=', '')}{'_' + a.variant if a.variant else ''}.json").read_text())
        seed = str(json.loads((run / "training_metadata.json").read_text())["seed"])
        m = js["model"]
        seqs = [k for k in m if k not in ("n_frames", "overall", "jitter")]       # `jitter`: the DT-round aggregate, not a sequence
        loc = [s for s in seqs if "_local" in s]
        glo = [s for s in seqs if "_local" not in s]

        def agg(ss, key):
            n = sum(m[s]["n_frames"] for s in ss)
            return sum(m[s][key][0] * m[s]["n_frames"] for s in ss) / max(n, 1)
        keys = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm", "root_rot_deg", "transl_mm")
        per_seed[seed] = {"step": js["step"], "ckpt": js["ckpt"],
                          "overall": {k: m["overall"][k] for k in keys},
                          "local": {k: agg(loc, k) for k in keys}, "global": {k: agg(glo, k) for k in keys}}
        ext[seed] = js
        cfg = cfg or load_config(next(iter(sorted(run.glob("*.yaml")))))
    keys = per_seed[next(iter(per_seed))]["overall"].keys()
    mean = {part: {k: float(np.mean([per_seed[s][part][k] for s in per_seed])) for k in keys}
            for part in ("overall", "local", "global")}
    return per_seed, ext, mean, cfg


def cmd_row(a):
    device = torch.device("cuda")
    per_seed, ext, mean, cfg = aggregate_runs(a.runs, a.split, a.ckpt, a.variant)
    run0 = Path(a.runs[0])
    ck, _, _ = find_ckpt(run0, a.ckpt)
    model = MNISTModel.load_from_checkpoint(str(ck), cfg=cfg, map_location=device).to(device).eval()
    anchor = latency_anchor(device)
    lat = latency_model(model, cfg, device)
    row = {"arm": a.arm, "split": a.split, "ckpt_rule": a.ckpt, "n_seeds": len(per_seed),
           "two_seed_mean": mean, "per_seed": per_seed,
           "latency_ms_raw": lat, "latency_anchor_ms": anchor, "latency_ms_scaled_full1p75": lat * 1.75 / anchor,
           "macs_forward_packet": macs_of(cfg), "params_total": int(sum(p.numel() for p in model.parameters())),
           "note": "fixed splits_semkine protocol; 'two_seed_mean' is the N-seed mean (key kept for tools/report_table.py)"}
    out = REPO / "outputs" / "semkine"
    out.mkdir(parents=True, exist_ok=True)
    tag = ("" if a.split == "val_core" else f"_{a.split}") + (f"_{a.variant}" if a.variant else "")
    row["eval_variant"] = a.variant
    (out / f"{a.arm}{tag}_main_row.json").write_text(json.dumps(row, indent=1))
    (out / f"{a.arm}{tag}_extended.json").write_text(json.dumps(ext, indent=1))
    print(json.dumps({k: v for k, v in row.items() if k != "per_seed"}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--run-dir", required=True)
    e.add_argument("--ckpt", default="selected")
    e.add_argument("--split", default="val_core")
    e.add_argument("--manifest", default=None)
    e.add_argument("--controls", action="store_true")
    e.add_argument("--config", default=None, help="model config (default: the yaml copied into the run)")
    e.add_argument("--tf", action="store_true", help="also the teacher-forced loop (GT state each step)")
    e.add_argument("--perturb", action="store_true",
                   help="also root-rotation perturbation recovery (10 / 20 deg, 20-step branches)")
    e.add_argument("--window-mode", choices=("fixed", "adaptive"), default="fixed")
    e.add_argument("--window-ms", type=int, default=STEP)
    e.add_argument("--min-events", type=int, default=0)
    e.add_argument("--max-window-ms", type=int, default=300)
    e.add_argument("--clip-run-start", action="store_true",
                   help="clip every evidence window to its segment's start (training's rule), not only to the recording's start "
                        "(file tag _clip)")
    e.add_argument("--count-mode", choices=COUNT_MODES, default="last50",
                   help="the event count handed to a count-aware filter: last50 = the 50 ms packet's (the protocol), norm = the "
                        "window's count rescaled to a 50 ms rate (file tag _cnorm)")
    e.add_argument("--window-set", type=parse_window_set, default=(),
                   help="comma list of windows in ms, e.g. 150,200,300: per step the smallest holding >= --min-events events "
                        "(else the largest), clipped to the available past; replaces --window-mode / --window-ms "
                        "(file tag _wset150-200-300_n<min-events>)")
    e.add_argument("--suffix", default="", help="appended to the output name, e.g. to re-evaluate a run on another "
                   "device without overwriting its recorded evaluation (row: pass it inside --variant)")
    r = sub.add_parser("row")
    r.add_argument("--arm", required=True)
    r.add_argument("--runs", nargs="+", required=True)
    r.add_argument("--split", default="val_core")
    r.add_argument("--ckpt", default="selected")
    r.add_argument("--variant", default="", help="evaluation variant tag, e.g. wadaptive50_n2000_x300")
    a = ap.parse_args()
    if a.cmd == "eval":
        try:
            check_window_opts(a)
        except ValueError as err:
            ap.error(str(err))
    cmd_eval(a) if a.cmd == "eval" else cmd_row(a)


if __name__ == "__main__":
    main()
