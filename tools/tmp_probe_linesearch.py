#!/usr/bin/env python3
"""Is the truth the minimum of the event-geometry objective, along the direction of the actual error?

Both plan A (a second, pose-independent observer) and plan D (an analytic factor graph) rest on the
same unstated assumption: that the events in a packet, read against the hand's projected silhouette,
prefer the true pose over the pose the tracker has drifted to. The echo-gain probe gives that
assumption partial support -- mean |sdf| is 3.33 px at the truth and 5.05 px at the drifted state --
but a scalar gap between two isolated points says nothing about whether anything can *descend* it.

So walk the straight line from the drifted state to the truth and evaluate the objective at each
step:

    x(s) = x_drift + s (x_truth - x_drift),   s in [-0.5, 1.5]
    J(s) = mean_i min(|sdf_{x(s)}(u_i)|, BAND)

The clip matters. A band-restricted objective changes its own event set as the pose moves, so it can
fall simply by shedding events; clipping instead of masking keeps one fixed set of events at every s
and makes the values comparable. Three outcomes, three different projects:

  * J falls monotonically to a minimum near s = 1  -> the information is there and the single-step
    Gauss-Newton failure is a solver/damping problem. Plan D is alive, plan A is alive.
  * J is flat (|J(1) - J(0)| small compared to its own scatter) -> the 28 mm drift is not resolvable
    from one packet of events. Both plans are dead at the root, whatever solver is used.
  * J is minimised at s near 0 -> the objective actively prefers the drifted pose, and any corrector
    built on it will push away from the truth.

The paired control `rand` walks the same distance in a random direction: if J falls just as much
there, the fall is an artefact of moving, not evidence about the truth.

Temporary: delete after the verdict lands.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from config import load_config                                    # noqa: E402
from mano_layer import ManoLayer                                  # noqa: E402
from model import MNISTModel                                      # noqa: E402
from semkine import eval_track as ET                              # noqa: E402
from semkine.dataset import sequences_for_split                   # noqa: E402


BAND_PX = 12.0


@torch.no_grad()
def _objective(kssf, betas, K, ev, params, chunk=64):
    """J for a batch of poses against one packet. params (S, 51) -> (S,) clipped mean |sdf|.

    One rasterise call covers many poses, which is what makes a derivative-free search affordable:
    the sdf comes from integer gathers on a rendered grid, so there is no gradient to autograd and
    every direction has to be paid for with a render.
    """
    xy = ev[:, [1, 2]]
    n = xy.shape[0]
    vals = []
    for i in range(0, len(params), chunk):
        p = params[i:i + chunk]
        s = len(p)
        fields = kssf(p, betas.expand(s, -1), K.expand(s, -1, -1), "mano_full_axis_angle")
        bi = torch.arange(s, device=xy.device).repeat_interleave(n)
        q = fields.query(xy.repeat(s, 1), bi)
        vals.append(q["sdf"].abs().clamp_max(BAND_PX).view(s, n).mean(1).cpu().numpy())
    return np.concatenate(vals).astype(np.float64)


@torch.no_grad()
def probe_sequence(model, mano, kssf, cfg, root, legacy_dir, seq, step_ms, device,
                   grid, stride, max_steps):
    events, offsets, aux, pos51 = ET.load_sequence(root, legacy_dir, seq)
    tsub_path = root / legacy_dir / f"{seq}_tsub.npy"
    tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    if hasattr(model, "set_hand_context"):
        model.set_hand_context(betas, K)
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)

    def ra(a, b):
        return float(np.linalg.norm(ET.root_align(a) - ET.root_align(b), axis=-1).mean() * 1000)

    def fk_j(p):
        c = torch.from_numpy(np.ascontiguousarray(np.atleast_2d(p), np.float32)).to(device)
        dec = ET.decode_to_mano_inputs(c, "mano_full_axis_angle",
                                       mano.hands_components, mano.hands_mean)
        _, j = mano(betas.expand(len(c), -1), dec["global_orient"],
                    dec["local_full_aa"], dec["transl"])
        return j.cpu().numpy()

    rng = np.random.default_rng(0)
    rrng = np.random.default_rng(777)
    rows = []
    n_steps = 0
    for a, b in runs:
        ends = np.arange(a + step_ms - 1, b, step_ms, dtype=np.int64)
        if not len(ends):
            continue
        prev = pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)
        prev_t = torch.from_numpy(prev).view(1, -1).to(device)
        for end in ends:
            if max_steps and n_steps >= max_steps:
                break
            ev5 = ET._window_events(events, offsets, tsub, int(end), step_ms)
            pkt = ET.make_eval_packet(ev5, prev_t, betas, K, step_ms, device)
            net = model.forward_packet(pkt).cpu().numpy()[0]
            if (n_steps % stride) == 0 and len(ev5) >= 64:
                tgt = pos51[int(end)]
                d = tgt - prev
                r = rrng.normal(size=51).astype(np.float32)
                r *= float(np.linalg.norm(d) / max(np.linalg.norm(r), 1e-9))
                ev = torch.from_numpy(ev5).to(device)
                ps = np.stack([prev + s * d for s in grid]
                              + [prev + s * r for s in grid]).astype(np.float32)
                pt = torch.from_numpy(ps).to(device)
                J = _objective(kssf, betas, K, ev, pt)
                jt, jr = J[:len(grid)], J[len(grid):]
                j3 = fk_j(np.stack([prev, tgt]))
                rows.append({
                    "n_ev": int(len(ev5)),
                    "err_before_mm": ra(j3[0], j3[1]),
                    "truth": jt.tolist(),
                    "rand": jr.tolist(),
                })
            prev = net
            prev_t = torch.from_numpy(prev).view(1, -1).to(device)
            n_steps += 1
        if max_steps and n_steps >= max_steps:
            break
    return rows, n_steps


@torch.no_grad()
def descend_sequence(model, mano, mano_d, kssf, cfg, root, legacy_dir, seq, step_ms, device,
                     damp, sigma_px, iters, stride, max_steps):
    """Walk down the objective from the drifted state using only events, never the truth.

    The line search proves a monotone descent path exists over the whole drift. This asks whether it
    can be *found*: at every iteration take damped Gauss-Newton steps at several damping values,
    evaluate the objective at each candidate and at halved step lengths, and accept the best one --
    backtracking on J alone. The truth enters nowhere except the error that gets reported.
    """
    import tmp_probe_geom_oracle as GO

    events, offsets, aux, pos51 = ET.load_sequence(root, legacy_dir, seq)
    tsub_path = root / legacy_dir / f"{seq}_tsub.npy"
    tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    if hasattr(model, "set_hand_context"):
        model.set_hand_context(betas, K)
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)

    def fk_j(p):
        c = torch.from_numpy(np.ascontiguousarray(np.atleast_2d(p), np.float32)).to(device)
        dec = ET.decode_to_mano_inputs(c, "mano_full_axis_angle",
                                       mano.hands_components, mano.hands_mean)
        _, j = mano(betas.expand(len(c), -1), dec["global_orient"],
                    dec["local_full_aa"], dec["transl"])
        return j.cpu().numpy()

    def ra(a, b):
        return float(np.linalg.norm(ET.root_align(a) - ET.root_align(b), axis=-1).mean() * 1000)

    rng = np.random.default_rng(0)
    rows = []
    n_steps = 0
    for a, b in runs:
        ends = np.arange(a + step_ms - 1, b, step_ms, dtype=np.int64)
        if not len(ends):
            continue
        prev = pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)
        prev_t = torch.from_numpy(prev).view(1, -1).to(device)
        for end in ends:
            if max_steps and n_steps >= max_steps:
                break
            ev5 = ET._window_events(events, offsets, tsub, int(end), step_ms)
            pkt = ET.make_eval_packet(ev5, prev_t, betas, K, step_ms, device)
            net = model.forward_packet(pkt).cpu().numpy()[0]
            if (n_steps % stride) == 0 and len(ev5) >= 64:
                tgt = pos51[int(end)]
                ev = torch.from_numpy(ev5).to(device)
                x = prev.copy()
                jx = float(_objective(kssf, betas, K, ev,
                                      torch.from_numpy(x[None].astype(np.float32)).to(device))[0])
                jt = fk_j(tgt)[0]
                traj = [ra(fk_j(x)[0], jt)]
                jtraj = [jx]
                for _ in range(iters):
                    steps, _st = GO._gn_step(mano_d, kssf, x, betas, K, ev5, device,
                                             damp, sigma_px)
                    if steps is None:
                        break
                    cand = []
                    for mu in sorted(steps):
                        for frac in (1.0, 0.5, 0.25):
                            cand.append(x + frac * steps[mu])
                    ct = torch.from_numpy(np.stack(cand).astype(np.float32)).to(device)
                    js = _objective(kssf, betas, K, ev, ct)
                    k = int(np.argmin(js))
                    if js[k] >= jx - 1e-4:            # no candidate improves the objective
                        break
                    x, jx = cand[k], float(js[k])
                    traj.append(ra(fk_j(x)[0], jt))
                    jtraj.append(jx)
                rows.append({"err_traj_mm": traj, "J_traj_px": jtraj,
                             "err_net_mm": ra(fk_j(net)[0], jt),
                             "n_iter": len(traj) - 1})
            prev = net
            prev_t = torch.from_numpy(prev).view(1, -1).to(device)
            n_steps += 1
        if max_steps and n_steps >= max_steps:
            break
    return rows, n_steps


# Per-coordinate finite-difference steps: the 51-vector is 48 radians then 3 metres, so one step
# size for all of them would either be numerical noise on the angles or a teleport on the root.
def _fd_h(n=51):
    h = np.full(n, 0.01, np.float32)
    h[48:] = 0.002
    return h


@torch.no_grad()
def fd_descend_sequence(model, mano, kssf, cfg, root, legacy_dir, seq, step_ms, device,
                        iters, stride, max_steps):
    """Is the objective optimisable *without* the truth?

    The line search says the truth is the minimum along the error direction, and the analytic
    Gauss-Newton step says the linearised contour residual does not point there. This separates the
    two: estimate the gradient of the objective itself by finite differences and follow it with a
    backtracking line search. Nothing here sees the ground truth except the reported error.

    If this recovers a large part of the drift, the deficit is that nothing in the system is
    optimising the objective, and the fix is to put it in the loop. If it stalls like Gauss-Newton
    did, then only the exact truth direction descends and there is nothing to exploit.
    """
    events, offsets, aux, pos51 = ET.load_sequence(root, legacy_dir, seq)
    tsub_path = root / legacy_dir / f"{seq}_tsub.npy"
    tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    if hasattr(model, "set_hand_context"):
        model.set_hand_context(betas, K)
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)

    def fk_j(p):
        c = torch.from_numpy(np.ascontiguousarray(np.atleast_2d(p), np.float32)).to(device)
        dec = ET.decode_to_mano_inputs(c, "mano_full_axis_angle",
                                       mano.hands_components, mano.hands_mean)
        _, j = mano(betas.expand(len(c), -1), dec["global_orient"],
                    dec["local_full_aa"], dec["transl"])
        return j.cpu().numpy()

    def ra(a, b):
        return float(np.linalg.norm(ET.root_align(a) - ET.root_align(b), axis=-1).mean() * 1000)

    h = _fd_h()
    scales = np.array([0.25, 0.5, 1.0, 2.0], np.float64)
    rng = np.random.default_rng(0)
    rows = []
    n_steps = 0
    for a, b in runs:
        ends = np.arange(a + step_ms - 1, b, step_ms, dtype=np.int64)
        if not len(ends):
            continue
        prev = pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)
        prev_t = torch.from_numpy(prev).view(1, -1).to(device)
        for end in ends:
            if max_steps and n_steps >= max_steps:
                break
            ev5 = ET._window_events(events, offsets, tsub, int(end), step_ms)
            pkt = ET.make_eval_packet(ev5, prev_t, betas, K, step_ms, device)
            net = model.forward_packet(pkt).cpu().numpy()[0]
            if (n_steps % stride) == 0 and len(ev5) >= 64:
                tgt = pos51[int(end)]
                jt = fk_j(tgt)[0]
                ev = torch.from_numpy(ev5).to(device)
                x = prev.copy().astype(np.float32)
                probe = np.concatenate([x[None], x[None] + np.diag(h)])
                jx = None
                traj, jtraj = [ra(fk_j(x)[0], jt)], []
                for _ in range(iters):
                    J = _objective(kssf, betas, K, ev,
                                   torch.from_numpy(probe).to(device))
                    jx = float(J[0]) if jx is None else jx
                    if not jtraj:
                        jtraj.append(jx)
                    g = (J[1:] - J[0]) / h                       # forward-difference gradient
                    gn = np.linalg.norm(g / np.maximum(1.0 / h, 1e-9))
                    if not np.isfinite(gn) or gn <= 0:
                        break
                    # Trust region in the natural units of each coordinate.
                    d = -(g * h * h)
                    d = d / max(np.linalg.norm(d / h), 1e-9)
                    cand = np.stack([x + float(s) * d for s in scales]).astype(np.float32)
                    Jc = _objective(kssf, betas, K, ev, torch.from_numpy(cand).to(device))
                    k = int(np.argmin(Jc))
                    if Jc[k] >= jx - 1e-4:
                        break
                    x, jx = cand[k], float(Jc[k])
                    traj.append(ra(fk_j(x)[0], jt))
                    jtraj.append(jx)
                    probe = np.concatenate([x[None], x[None] + np.diag(h)])
                rows.append({"err_traj_mm": traj, "J_traj_px": jtraj,
                             "err_net_mm": ra(fk_j(net)[0], jt),
                             "n_iter": len(traj) - 1})
            prev = net
            prev_t = torch.from_numpy(prev).view(1, -1).to(device)
            n_steps += 1
        if max_steps and n_steps >= max_steps:
            break
    return rows, n_steps


@torch.no_grad()
def corr_sequence(model, mano, kssf, cfg, root, legacy_dir, seq, step_ms, device,
                  n_samp, stride, max_steps):
    """Does the objective rank poses by accuracy, or only along the one line to the truth?

    The finite-difference descent lowers the objective by half the gap to the truth's own value and
    moves the joints by 0.03 mm, which says the level sets are large. This measures that directly:
    scatter poses around the drifted state, and correlate each one's objective against its actual
    joint error. A corrector -- analytic, learned, or otherwise -- can only be as good as this
    correlation, because the objective is the only thing it can see.

    The truth direction is included as a positive control: it must show the drop the line search
    already found, or the sampling is broken rather than the objective.
    """
    events, offsets, aux, pos51 = ET.load_sequence(root, legacy_dir, seq)
    tsub_path = root / legacy_dir / f"{seq}_tsub.npy"
    tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    if hasattr(model, "set_hand_context"):
        model.set_hand_context(betas, K)
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)

    def fk_j(p):
        c = torch.from_numpy(np.ascontiguousarray(np.atleast_2d(p), np.float32)).to(device)
        dec = ET.decode_to_mano_inputs(c, "mano_full_axis_angle",
                                       mano.hands_components, mano.hands_mean)
        _, j = mano(betas.expand(len(c), -1), dec["global_orient"],
                    dec["local_full_aa"], dec["transl"])
        return j.cpu().numpy()

    rng = np.random.default_rng(0)
    srng = np.random.default_rng(4242)
    rows = []
    n_steps = 0
    for a, b in runs:
        ends = np.arange(a + step_ms - 1, b, step_ms, dtype=np.int64)
        if not len(ends):
            continue
        prev = pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)
        prev_t = torch.from_numpy(prev).view(1, -1).to(device)
        for end in ends:
            if max_steps and n_steps >= max_steps:
                break
            ev5 = ET._window_events(events, offsets, tsub, int(end), step_ms)
            pkt = ET.make_eval_packet(ev5, prev_t, betas, K, step_ms, device)
            net = model.forward_packet(pkt).cpu().numpy()[0]
            if (n_steps % stride) == 0 and len(ev5) >= 64:
                tgt = pos51[int(end)]
                d = tgt - prev
                nd = float(np.linalg.norm(d))
                # Perturbations spanning the drift's own magnitude, so the cloud covers the region a
                # corrector would have to search, not an infinitesimal neighbourhood.
                D = srng.normal(size=(n_samp, 51)).astype(np.float32)
                D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
                D *= nd * srng.uniform(0.15, 1.2, size=(n_samp, 1)).astype(np.float32)
                cand = np.concatenate([prev[None], tgt[None], prev[None] + D]).astype(np.float32)
                J = _objective(kssf, betas, K, torch.from_numpy(ev5).to(device),
                               torch.from_numpy(cand).to(device))
                j = fk_j(cand)
                jt = ET.root_align(j[1])
                err = np.linalg.norm(ET.root_align(j) - jt, axis=-1).mean(-1) * 1000
                Js, es = J[2:], err[2:]
                rows.append({
                    "pearson": float(np.corrcoef(Js, es)[0, 1]),
                    "spearman": float(np.corrcoef(np.argsort(np.argsort(Js)),
                                                  np.argsort(np.argsort(es)))[0, 1]),
                    # If the best-objective sample is no more accurate than a random one, nothing
                    # that selects on the objective can help.
                    "err_best_J_mm": float(es[int(np.argmin(Js))]),
                    "err_mean_mm": float(es.mean()),
                    "err_best_possible_mm": float(es.min()),
                    "err_drift_mm": float(err[0]),
                    "J_drift": float(J[0]), "J_truth": float(J[1]),
                    "J_min_sampled": float(Js.min()),
                    # How many scattered poses beat the truth's own objective while being wrong.
                    "frac_J_below_truth": float((Js < J[1]).mean()),
                    "err_of_J_below_truth_mm": float(es[Js < J[1]].mean())
                    if bool((Js < J[1]).any()) else float("nan"),
                })
            prev = net
            prev_t = torch.from_numpy(prev).view(1, -1).to(device)
            n_steps += 1
        if max_steps and n_steps >= max_steps:
            break
    return rows, n_steps


def summarise_corr(rows) -> dict:
    keys = [k for k in rows[0]]
    out = {"n_packets": len(rows)}
    for k in keys:
        v = np.array([r[k] for r in rows], np.float64)
        v = v[np.isfinite(v)]
        if len(v):
            out[k] = float(v.mean())
    return out


def summarise_descent(rows) -> dict:
    n_max = max(len(r["err_traj_mm"]) for r in rows)

    def pad(seq, n):
        return list(seq) + [seq[-1]] * (n - len(seq))

    E = np.array([pad(r["err_traj_mm"], n_max) for r in rows], np.float64)
    J = np.array([pad(r["J_traj_px"], n_max) for r in rows], np.float64)
    return {
        "n_packets": int(len(rows)),
        "err_by_iter_mm": E.mean(0).tolist(),
        "J_by_iter_px": J.mean(0).tolist(),
        "err_before_mm": float(E[:, 0].mean()),
        "err_after_mm": float(E[:, -1].mean()),
        "err_net_mm": float(np.mean([r["err_net_mm"] for r in rows])),
        "gain_mm": float((E[:, 0] - E[:, -1]).mean()),
        "frac_descend": float((E[:, -1] < E[:, 0]).mean()),
        "frac_beat_net": float((E[:, -1] < np.array([r["err_net_mm"] for r in rows])).mean()),
        "iters_mean": float(np.mean([r["n_iter"] for r in rows])),
    }


def summarise(rows, grid) -> dict:
    T = np.array([r["truth"] for r in rows], np.float64)          # (N, S)
    R = np.array([r["rand"] for r in rows], np.float64)
    g = np.asarray(grid, np.float64)
    i0 = int(np.argmin(np.abs(g - 0.0)))
    i1 = int(np.argmin(np.abs(g - 1.0)))
    am = g[np.argmin(T, axis=1)]
    out = {
        "n_packets": int(len(rows)),
        "grid": g.tolist(),
        "J_truth_mean_px": T.mean(0).tolist(),
        "J_rand_mean_px": R.mean(0).tolist(),
        "J0_px": float(T[:, i0].mean()),
        "J1_px": float(T[:, i1].mean()),
        # Does walking to the truth beat walking the same distance nowhere in particular?
        "drop_truth_px": float((T[:, i0] - T[:, i1]).mean()),
        "drop_rand_px": float((R[:, i0] - R[:, i1]).mean()),
        "frac_truth_below_start": float((T[:, i1] < T[:, i0]).mean()),
        "frac_rand_below_start": float((R[:, i1] < R[:, i0]).mean()),
        # A usable objective puts its minimum near s = 1; a flat or adversarial one does not.
        "frac_argmin_near_truth": float((am >= 0.75).mean()),
        "frac_argmin_at_start": float((am <= 0.25).mean()),
        "argmin_s_mean": float(am.mean()),
        "argmin_s_median": float(np.median(am)),
        # Scale of the signal against its own packet-to-packet scatter.
        "drop_truth_std_px": float((T[:, i0] - T[:, i1]).std()),
        "err_before_mm": float(np.mean([r["err_before_mm"] for r in rows])),
        "n_ev_mean": float(np.mean([r["n_ev"] for r in rows])),
    }
    d = T[:, i0] - T[:, i1]
    out["snr_drop"] = float(d.mean() / max(d.std(), 1e-9))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--route", default="soft")
    ap.add_argument("--split", default="val_core")
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--grid", default="-0.5,-0.25,0,0.25,0.5,0.75,1.0,1.25,1.5")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--descend", type=int, default=0,
                    help="instead of the line search, run N backtracking Gauss-Newton iterations "
                         "that only ever see the events")
    ap.add_argument("--corr", type=int, default=0,
                    help="sample N poses per packet and correlate objective against joint error")
    ap.add_argument("--fd", type=int, default=0,
                    help="run N finite-difference descent iterations on the objective itself")
    ap.add_argument("--damp", default="0.01,0.1,1")
    ap.add_argument("--sigma-px", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.environ["EVENTHANDS_KEG_ROUTE"] = a.route
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(a.config)
    model = MNISTModel.load_from_checkpoint(a.ckpt, cfg=cfg, map_location=device).to(device).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    kssf = model.kssf_on(device)
    root = Path(cfg["DATA"]["ROOT"])
    grid = [float(x) for x in a.grid.split(",")]

    if a.corr:
        allrows, per = [], []
        for seq, dd in sequences_for_split(root, a.split, None):
            rows, n = corr_sequence(model, mano, kssf, cfg, root, dd, seq, a.step_ms,
                                    device, a.corr, a.stride, a.max_steps)
            if not rows:
                continue
            s = summarise_corr(rows)
            s["seq"] = seq
            per.append(s)
            allrows += rows
            print(f"  {seq}: n={s['n_packets']} rho={s['pearson']:+.3f} "
                  f"spear={s['spearman']:+.3f} drift={s['err_drift_mm']:.1f} "
                  f"bestJ={s['err_best_J_mm']:.1f} mean={s['err_mean_mm']:.1f} "
                  f"floor={s['err_best_possible_mm']:.1f} "
                  f"J<truth={s['frac_J_below_truth']:.2f}", flush=True)
        pooled = summarise_corr(allrows)
        print(json.dumps(pooled, indent=2))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"ckpt": a.ckpt, "mode": "corr",
                                           "pooled": pooled, "per_sequence": per}, indent=2))
        print("wrote", a.out)
        return

    if a.fd:
        allrows, per = [], []
        for seq, d in sequences_for_split(root, a.split, None):
            rows, n = fd_descend_sequence(model, mano, kssf, cfg, root, d, seq, a.step_ms,
                                          device, a.fd, a.stride, a.max_steps)
            if not rows:
                continue
            s = summarise_descent(rows)
            s["seq"] = seq
            per.append(s)
            allrows += rows
            print(f"  {seq}: n={s['n_packets']} {s['err_before_mm']:.2f} -> "
                  f"{s['err_after_mm']:.2f}mm (net {s['err_net_mm']:.2f}) "
                  f"descend={s['frac_descend']:.2f} beat_net={s['frac_beat_net']:.2f} "
                  f"iters={s['iters_mean']:.1f}", flush=True)
        pooled = summarise_descent(allrows)
        print(json.dumps(pooled, indent=2))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"ckpt": a.ckpt, "mode": "fd",
                                           "pooled": pooled, "per_sequence": per}, indent=2))
        print("wrote", a.out)
        return

    if a.descend:
        sys.path.insert(0, str(REPO / "tools"))
        mano_d = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).double().eval()
        damp = [float(x) for x in a.damp.split(",") if x]
        allrows, per = [], []
        for seq, d in sequences_for_split(root, a.split, None):
            rows, n = descend_sequence(model, mano, mano_d, kssf, cfg, root, d, seq, a.step_ms,
                                       device, damp, a.sigma_px, a.descend, a.stride, a.max_steps)
            if not rows:
                continue
            s = summarise_descent(rows)
            s["seq"] = seq
            per.append(s)
            allrows += rows
            print(f"  {seq}: n={s['n_packets']} {s['err_before_mm']:.2f} -> "
                  f"{s['err_after_mm']:.2f}mm (net {s['err_net_mm']:.2f}) "
                  f"descend={s['frac_descend']:.2f} beat_net={s['frac_beat_net']:.2f} "
                  f"iters={s['iters_mean']:.1f}", flush=True)
        pooled = summarise_descent(allrows)
        print(json.dumps(pooled, indent=2))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"ckpt": a.ckpt, "mode": "descend",
                                           "pooled": pooled, "per_sequence": per}, indent=2))
        print("wrote", a.out)
        return

    allrows = []
    per = []
    for seq, d in sequences_for_split(root, a.split, None):
        rows, n = probe_sequence(model, mano, kssf, cfg, root, d, seq, a.step_ms, device,
                                 grid, a.stride, a.max_steps)
        if not rows:
            continue
        s = summarise(rows, grid)
        s["seq"] = seq
        per.append(s)
        allrows += rows
        print(f"  {seq}: n={s['n_packets']} err={s['err_before_mm']:.1f}mm "
              f"J0={s['J0_px']:.3f} J1={s['J1_px']:.3f} "
              f"drop={s['drop_truth_px']:+.3f} (rand {s['drop_rand_px']:+.3f}) "
              f"argmin_s={s['argmin_s_median']:+.2f} near_truth={s['frac_argmin_near_truth']:.2f}",
              flush=True)

    pooled = summarise(allrows, grid)
    print(json.dumps({k: v for k, v in pooled.items()
                      if k not in ("J_truth_mean_px", "J_rand_mean_px")}, indent=2))
    print("J(s) truth:", [round(x, 4) for x in pooled["J_truth_mean_px"]])
    print("J(s) rand :", [round(x, 4) for x in pooled["J_rand_mean_px"]])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"ckpt": a.ckpt, "pooled": pooled,
                                       "per_sequence": per}, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
