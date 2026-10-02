#!/usr/bin/env python3
"""x1001 evaluation of trained arms: the recursive protocol, step by step, plus what the main row
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

    python tools/x1001/evalx.py eval --run-dir RUNS/x1001_s37_s3407 [--ckpt selected|last|step=N]
                                     [--split val_core --manifest M] [--controls]
    python tools/x1001/evalx.py row --arm x1001_s37 --runs RUNS/x1001_s37_s3407 RUNS/x1001_s37_s3408
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


def mano_fk(mano, params, betas, device):
    oj, ov = [], []
    for i0 in range(0, len(params), 2048):
        chunk = torch.from_numpy(params[i0:i0 + 2048]).to(device)
        dec = decode_to_mano_inputs(chunk, "mano_full_axis_angle", mano.hands_components, mano.hands_mean)
        v, j = mano(betas.expand(len(chunk), -1), dec["global_orient"], dec["local_full_aa"], dec["transl"])
        oj.append(j.cpu().numpy())
        ov.append(v.cpu().numpy())
    return np.concatenate(oj), np.concatenate(ov)


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


def window_of(offsets, end, wmode, win, min_events, max_win):
    """Evidence window (ms) ending at `end`. fixed: `win`; adaptive: the shortest window >= `win`
    holding >= `min_events` events, capped at `max_win`. Causal either way (only past events)."""
    if wmode == "fixed":
        return int(min(win, end + 1))
    w = int(min(win, end + 1))
    while w < max_win and w < end + 1 and offsets[end + 1] - offsets[end - w + 1] < min_events:
        w = min(w + 10, max_win, end + 1)
    return int(w)


@torch.no_grad()
def run_sequence(model, cfg, root, d, seq, device, rng, mode="model", wmode="fixed", win=STEP,
                 min_events=0, max_win=300):
    """`track_sequence`'s loop, keeping every step. mode: model | hold | noevents | tf.
    `tf` (teacher forcing) feeds the ground truth of the previous evaluated step back instead of the
    prediction (the first step of a segment keeps the protocol's noisy initial state), so its error is
    the single-step error under a clean state. `prev` records the state each step was conditioned on.
    The evaluated steps never change (ends a+49, a+99, ...); only the evidence window may."""
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
    preds, gts, ends_all, runs_all, elapsed, counts, prevs = [], [], [], [], [], [], []
    for run_id, (a, b) in enumerate(runs):
        ends = np.arange(a + STEP - 1, b, STEP, dtype=np.int64)
        if not len(ends):
            continue
        prev = pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)
        prev_t = torch.from_numpy(prev).view(1, -1).to(device)
        init_t = prev_t.clone()
        for end in ends:
            n_ev = int(offsets[int(end) + 1] - offsets[int(end) - STEP + 1])
            w = window_of(offsets, int(end), wmode, win, min_events, max_win)
            prevs.append(prev_t.cpu().numpy()[0])
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
                pred = model(x, prev_t)
            prev_t = pred if mode != "tf" else torch.from_numpy(pos51[int(end)].copy()).view(1, -1).to(device)
            preds.append(pred.cpu().numpy()[0])
            gts.append(pos51[end])
            ends_all.append(int(end))
            runs_all.append(run_id)
            elapsed.append(int(end - a))
            counts.append(n_ev)
    return {"pred": np.stack(preds).astype(np.float32), "gt": np.stack(gts).astype(np.float32),
            "end": np.asarray(ends_all), "run": np.asarray(runs_all), "elapsed": np.asarray(elapsed),
            "count": np.asarray(counts), "betas": aux["betas"],
            "prev": np.stack(prevs).astype(np.float32)}


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
                preds = []
                for i in range(i0, i0 + horizon):
                    end = int(r["end"][i])
                    if use_raw:
                        ev5 = ET._window_events(events, offsets, tsub, end, STEP)
                        pred = model.forward_packet(ET.make_eval_packet(ev5, prev_t, betas, camera_K, STEP, device))
                    else:
                        x = torch.from_numpy(ET.build_lnes(events, offsets, end, STEP, ev_ch)).unsqueeze(0).to(device)
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
    recovery, per-sequence CIs / failures / event-rate buckets / motion. Returns `(summary, arrays)`."""
    modes = ["model"] + (["hold", "noevents"] if a.controls else []) + (["tf"] if a.tf else [])
    out, t0 = {}, time.time()
    arrays = {}
    for mode in modes:
        rng = np.random.default_rng(0)
        res = {}
        for s, d in seqs:
            r = run_sequence(model, cfg, root, d, s, device, rng, mode, a.window_mode, a.window_ms,
                             a.min_events, a.max_window_ms)
            m = per_step_metrics(mano, r, device)
            res[s] = (r, m)
            for k, v in m.items():
                arrays[f"{mode}|{s}|{k}"] = v.astype(np.float32)
            if mode == "model":
                for k in ("pred", "gt", "end", "run", "elapsed", "count"):
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
            summ[s] = ent
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
    out["window"] = {"mode": a.window_mode, "ms": a.window_ms, "min_events": a.min_events, "max_ms": a.max_window_ms}
    if sel_ra is not None and a.split == "val_core" and a.window_mode == "fixed" and a.window_ms == STEP:
        drift = abs(out["model"]["overall"]["mpjpe_ra_mm"] - sel_ra)
        out["selection_drift_mm"] = drift
        assert drift < 0.05, f"re-run drift {drift:.4f} mm vs selection"
    tag = f"evalx_{a.split}_{a.ckpt.replace('=', '')}"
    tag += "_tf" if a.tf else ""
    tag += "_pert" if a.perturb else ""
    if a.window_mode != "fixed" or a.window_ms != STEP:
        tag += f"_w{a.window_mode}{a.window_ms}" + (f"_n{a.min_events}_x{a.max_window_ms}" if a.window_mode == "adaptive" else "")
    tag += f"_{a.suffix}" if a.suffix else ""
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
                              .unsqueeze(0).to(device), prev))
            if len(items) >= max_packets:
                break
        if len(items) >= max_packets:
            break
    f = (lambda it: model.forward_packet(it)) if raw else (lambda it: model(*it))   # noqa: E731
    for it in items[:100]:
        f(it)
    torch.cuda.synchronize()
    means = []
    for _ in range(passes):
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
        seqs = [k for k in m if k not in ("n_frames", "overall")]
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
           "note": "x1001 protocol; 'two_seed_mean' is the N-seed mean (key kept for tools/report_table.py)"}
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
    e.add_argument("--suffix", default="", help="appended to the output name, e.g. to re-evaluate a run on another "
                   "device without overwriting its recorded evaluation (row: pass it inside --variant)")
    r = sub.add_parser("row")
    r.add_argument("--arm", required=True)
    r.add_argument("--runs", nargs="+", required=True)
    r.add_argument("--split", default="val_core")
    r.add_argument("--ckpt", default="selected")
    r.add_argument("--variant", default="", help="evaluation variant tag, e.g. wadaptive50_n2000_x300")
    a = ap.parse_args()
    cmd_eval(a) if a.cmd == "eval" else cmd_row(a)


if __name__ == "__main__":
    main()
