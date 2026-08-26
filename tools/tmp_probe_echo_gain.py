#!/usr/bin/env python3
"""Why the recursive fixed point sits so far above the independent-noise one.

`closed_loop_sensitivity_halo_50ms.json` says the s21-halo arm scores 12.11 mm single-step under
teacher forcing, 18.26 mm when 26 mm of *independent* conditioning error is injected, and 27.77 mm
in its own recursion. So its own error hurts roughly twice as much per millimetre as injected error
of the same size. The project calls that the echo excess and attributes it to correlation in time.
Correlation in time is not the only candidate: the self-error also has a *direction*, chosen by the
model itself, and a direction the update operator barely corrects would survive every step no matter
how the curriculum is shaped.

This probe separates the two. At each recursive step, with `c` the model's own conditioning state,
`g` the ground truth it should have been, and `f` the frozen network:

    S_self = d(f(c), f(g)) / d(c, g)          gain along the self-error direction
    S_rand = d(f(g+r), f(g)) / d(g+r, g)      gain along a training-noise direction of the same size

`d` is root-aligned MPJPE through MANO, so both are dimensionless mm/mm and the two directions are
compared at the *same* amplitude -- `r` is rescaled per step to match `d(c, g)`. `S = 1` means the
perturbation passes through the update untouched; `S = 0` means it is fully corrected. The recursive
steady state is governed by these gains, so this is the quantity every one of the four candidate
routes is ultimately trying to move.

Second measurement, same pass: whether the drift is *visible*. If the events, read against the
drifted state, look no more inconsistent than against the truth, then no second observer and no
analytic solver can recover the state, and only a prior or a temporal model can help. Reported as
the KSSF signed-distance statistics of the same events under `c` and under `g`.

Temporary: `/tmp`-class diagnostic per the experiment contract; delete after the verdict lands.
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
from semkine.dataset import sequences_for_split, splits_manifest   # noqa: E402


NOISE_BLOCKS = (0.005, 0.05, 0.05)     # trans / root-rot / pose, the TRACK.PREV_NOISE_* ratios


class Fk:
    """Batched 51D -> (21, 3) joints, and the two distances every gate is written in."""

    def __init__(self, mano, betas, device):
        self.mano, self.betas, self.device = mano, betas, device

    @torch.no_grad()
    def __call__(self, params: np.ndarray) -> np.ndarray:
        p = torch.from_numpy(np.ascontiguousarray(params, np.float32)).to(self.device)
        dec = ET.decode_to_mano_inputs(p, "mano_full_axis_angle",
                                       self.mano.hands_components, self.mano.hands_mean)
        _, j = self.mano(self.betas.expand(len(p), -1), dec["global_orient"],
                         dec["local_full_aa"], dec["transl"])
        return j.cpu().numpy()

    @staticmethod
    def ra(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(ET.root_align(a) - ET.root_align(b), axis=-1).mean() * 1000)

    @staticmethod
    def abs_(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b, axis=-1).mean() * 1000)


def _rand_dir(rng, n=51) -> np.ndarray:
    d = np.zeros(n, np.float32)
    d[0:3] = rng.normal(0, NOISE_BLOCKS[0], 3)
    d[3:6] = rng.normal(0, NOISE_BLOCKS[1], 3)
    d[6:] = rng.normal(0, NOISE_BLOCKS[2], n - 6)
    return d


def _sdf_stats(kssf, params51, betas, K, ev5, device) -> dict:
    """How inconsistent the events look against the surface implied by `params51`."""
    if not len(ev5):
        return {}
    p = torch.from_numpy(params51.astype(np.float32)).view(1, -1).to(device)
    fields = kssf(p, betas, K, "mano_full_axis_angle")
    ev = torch.from_numpy(ev5).to(device)
    q = fields.query(ev[:, [1, 2]], ev[:, 0])
    sdf = q["sdf"].float()
    return {
        "abs_sdf_px": float(sdf.abs().mean()),
        "band_frac": float((sdf.abs() <= 12.0).float().mean()),
        "vis_frac": float((q["visibility"] > 0.5).float().mean()),
        "outside_frac": float((sdf > 0).float().mean()),
    }


@torch.no_grad()
def probe_sequence(model, mano, cfg, root, legacy_dir, seq, step_ms, device,
                   observe_stride, max_steps, kssf, window_ms=None):
    events, offsets, aux, pos51 = ET.load_sequence(root, legacy_dir, seq)
    tsub_path = root / legacy_dir / f"{seq}_tsub.npy"
    tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
    use_raw = bool(getattr(model, "encoder_name", ""))
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    if hasattr(model, "set_hand_context"):
        model.set_hand_context(betas, K)
    fk = Fk(mano, betas, device)
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)

    # Same conditioning stream the frozen recursive evaluator produces.
    rng = np.random.default_rng(0)
    rrng = np.random.default_rng(12345)
    rows, obs_c, obs_g = [], [], []
    n_steps = 0
    # The evidence window and the update interval are separate knobs; LNES normally ties them
    # together, and decoupling them is exactly what an incremental-state network would do.
    win = int(window_ms or step_ms)
    for a, b in runs:
        ends = np.arange(a + max(step_ms, win) - 1, b, step_ms, dtype=np.int64)
        if not len(ends):
            continue
        prev = pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)
        prev_end = max(int(a), int(ends[0]) - step_ms)
        for end in ends:
            if max_steps and n_steps >= max_steps:
                break
            ev5 = ET._window_events(events, offsets, tsub, int(end), win) if use_raw else None
            lnes = None if use_raw else torch.from_numpy(
                ET.build_lnes(events, offsets, int(end), win)).unsqueeze(0).to(device)

            def fwd(state51):
                t = torch.from_numpy(np.ascontiguousarray(state51, np.float32)).view(1, -1).to(device)
                if use_raw:
                    return model.forward_packet(
                        ET.make_eval_packet(ev5, t, betas, K, win, device)).cpu().numpy()[0]
                return model(lnes, t).cpu().numpy()[0]

            c = prev
            g = pos51[prev_end].copy()
            tgt = pos51[int(end)].copy()

            pred_c = fwd(c)                       # the recursive step itself
            measure = (n_steps % observe_stride) == 0
            if measure:
                # One random direction, rescaled so both gains are read at the same amplitude.
                r0 = _rand_dir(rrng)
                j = fk(np.stack([g, c, g + r0]))
                dc, dr0 = fk.ra(j[1], j[0]), fk.ra(j[2], j[0])
                r = r0 * float(np.clip(dc / max(dr0, 1e-6), 0.05, 20.0))
                pred_g, pred_r = fwd(g), fwd(g + r)
                j2 = fk(np.stack([g + r, pred_c, pred_g, pred_r, tgt]))
                dr = fk.ra(j2[0], j[0])
                # Vector bookkeeping for the bias question: is the residual the loop feeds back a
                # fresh draw each step, or the same direction over and over? A geometric series in
                # a coherent bias and a random walk in white noise reach very different steady
                # states from the same per-step magnitude.
                ra = lambda x: ET.root_align(x).reshape(-1)          # noqa: E731
                e_tf = ra(j2[2]) - ra(j2[4])          # TF residual: what the update failed to fix
                e_rec = ra(j2[1]) - ra(j2[4])
                v_gt = ra(j2[4]) - ra(j[0])           # the motion the step had to track
                rows.append({
                    "dc_ra": dc, "dr_ra": dr,
                    "df_self": fk.ra(j2[1], j2[2]), "df_rand": fk.ra(j2[3], j2[2]),
                    "err_rec": fk.ra(j2[1], j2[4]), "err_tf": fk.ra(j2[2], j2[4]),
                    "err_rand": fk.ra(j2[3], j2[4]),
                    "move_rec": fk.ra(j2[1], j[1]), "move_tf": fk.ra(j2[2], j[0]),
                    "need": fk.ra(j[0], j2[4]),
                    "_e_tf": e_tf, "_e_rec": e_rec, "_v_gt": v_gt,
                })
                if kssf is not None and use_raw and len(ev5):
                    obs_c.append(_sdf_stats(kssf, c, betas, K, ev5, device))
                    obs_g.append(_sdf_stats(kssf, g, betas, K, ev5, device))
            prev, prev_end = pred_c, int(end)
            n_steps += 1
        if max_steps and n_steps >= max_steps:
            break
    return rows, obs_c, obs_g, n_steps


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return (a * b).sum(-1) / np.maximum(n, 1e-12)


def _bias_structure(rows) -> dict:
    """Is the fed-back residual a coherent bias or a fresh draw?

    Three readings, all cosines so they are scale free:

    * `autocorr_tf` -- residual against the previous step's residual. Near 1 means the loop adds
      the same vector every step, so the error grows like `n / (1 - G)`; near 0 means a random walk
      and `n / sqrt(1 - G^2)`, which is where the "independent-noise fixed point" comes from.
    * `cos_err_vs_lag` -- residual against minus the ground-truth motion of the step. Near 1 is a
      lag: the update tracks a fraction of the motion and the shortfall points backwards along it.
    * `miss_frac` -- the size of that shortfall relative to the motion.
    """
    if len(rows) < 3:
        return {}
    E = np.stack([r["_e_tf"] for r in rows])
    R = np.stack([r["_e_rec"] for r in rows])
    V = np.stack([r["_v_gt"] for r in rows])
    return {
        "autocorr_tf": float(np.mean(_cos(E[1:], E[:-1]))),
        "autocorr_rec": float(np.mean(_cos(R[1:], R[:-1]))),
        "cos_err_vs_lag_tf": float(np.mean(_cos(E, -V))),
        "cos_err_vs_lag_rec": float(np.mean(_cos(R, -V))),
        "miss_frac_tf": float(np.linalg.norm(E, axis=-1).mean()
                              / max(np.linalg.norm(V, axis=-1).mean(), 1e-12)),
        "mean_bias_frac_tf": float(np.linalg.norm(E.mean(0)) / max(
            np.linalg.norm(E, axis=-1).mean(), 1e-12)),
    }


def _summarise(rows) -> dict:
    if not rows:
        return {}
    out_bias = _bias_structure(rows)
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    A = {k: np.array([r[k] for r in rows], np.float64) for k in rows[0]}
    ok = A["dc_ra"] > 1.0                       # a gain is only defined where there is an error
    okr = A["dr_ra"] > 1.0
    out = {
        "n": int(len(rows)),
        "cond_err_self_mm": float(A["dc_ra"].mean()),
        "cond_err_rand_mm": float(A["dr_ra"].mean()),
        # Pooled gain: ratio of pooled displacements, which is the quantity that governs the
        # steady state. The per-step mean of the ratio is reported next to it because a few
        # near-zero denominators would otherwise dominate.
        "gain_self_pooled": float(A["df_self"][ok].sum() / A["dc_ra"][ok].sum()),
        "gain_rand_pooled": float(A["df_rand"][okr].sum() / A["dr_ra"][okr].sum()),
        "gain_self_median": float(np.median((A["df_self"] / np.maximum(A["dc_ra"], 1e-6))[ok])),
        "gain_rand_median": float(np.median((A["df_rand"] / np.maximum(A["dr_ra"], 1e-6))[okr])),
        "err_recursive_mm": float(A["err_rec"].mean()),
        "err_tf_mm": float(A["err_tf"].mean()),
        "err_rand_mm": float(A["err_rand"].mean()),
        "move_recursive_mm": float(A["move_rec"].mean()),
        "move_tf_mm": float(A["move_tf"].mean()),
        "need_mm": float(A["need"].mean()),
    }
    out["gain_ratio_self_over_rand"] = out["gain_self_pooled"] / max(out["gain_rand_pooled"], 1e-9)
    # The two closed-form steady states this gain implies, for comparison with the measured one.
    g = min(max(out["gain_self_pooled"], 0.0), 0.999)
    out["fixed_point_coherent_mm"] = out["err_tf_mm"] / (1.0 - g)
    out["fixed_point_white_mm"] = out["err_tf_mm"] / float(np.sqrt(1.0 - g * g))
    out.update(out_bias)
    return out


def _mean_dicts(ds) -> dict:
    ds = [d for d in ds if d]
    if not ds:
        return {}
    return {k: float(np.mean([d[k] for d in ds])) for k in ds[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="LABEL=CKPT:CONFIG")
    ap.add_argument("--route", default="soft")
    ap.add_argument("--split", default="val_core")
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--window-ms", type=int, default=0,
                    help="evidence window per update; 0 = tie it to --step-ms as LNES does")
    ap.add_argument("--observe-stride", type=int, default=1,
                    help="measure the gain every Nth recursive step; the chain always runs")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--no-kssf-observability", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["EVENTHANDS_KEG_ROUTE"] = a.route
    result = {"step_ms": a.step_ms, "window_ms": a.window_ms or a.step_ms,
              "split": a.split, "route": a.route, "arms": {}}

    for spec in a.arm:
        label, rest = spec.split("=", 1)
        ckpt, cfg_path = rest.rsplit(":", 1)
        cfg = load_config(cfg_path)
        model = MNISTModel.load_from_checkpoint(ckpt, cfg=cfg, map_location=device).to(device).eval()
        mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
        kssf = None
        if not a.no_kssf_observability and getattr(model, "_kssf_holder", None):
            kssf = model.kssf_on(device)
        root = Path(cfg["DATA"]["ROOT"])

        per, obs = [], {"drifted": [], "truth": []}
        for seq, d in sequences_for_split(root, a.split, splits_manifest(cfg)):
            rows, oc, og, n = probe_sequence(model, mano, cfg, root, d, seq, a.step_ms,
                                             device, a.observe_stride, a.max_steps, kssf,
                                             window_ms=(a.window_ms or None))
            s = _summarise(rows)
            s["seq"] = seq
            s["n_recursive_steps"] = n
            per.append(s)
            obs["drifted"] += oc
            obs["truth"] += og
            print(f"  {label} {seq}: gain_self={s.get('gain_self_pooled', float('nan')):.3f} "
                  f"gain_rand={s.get('gain_rand_pooled', float('nan')):.3f} "
                  f"cond_err={s.get('cond_err_self_mm', float('nan')):.2f}mm "
                  f"err_rec={s.get('err_recursive_mm', float('nan')):.2f} "
                  f"err_tf={s.get('err_tf_mm', float('nan')):.2f}", flush=True)

        w = np.array([p["n"] for p in per], np.float64)
        pooled = {k: float(np.sum([p[k] * wi for p, wi in zip(per, w)]) / w.sum())
                  for k in per[0] if isinstance(per[0][k], float)}
        pooled["gain_ratio_self_over_rand"] = (pooled["gain_self_pooled"]
                                               / max(pooled["gain_rand_pooled"], 1e-9))
        row = {"ckpt": ckpt, "per_sequence": per, "pooled": pooled,
               "observability": {"drifted": _mean_dicts(obs["drifted"]),
                                 "truth": _mean_dicts(obs["truth"])}}
        result["arms"][label] = row
        print(f"{label}: gain_self={pooled['gain_self_pooled']:.3f} "
              f"gain_rand={pooled['gain_rand_pooled']:.3f} "
              f"ratio={pooled['gain_ratio_self_over_rand']:.2f}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
