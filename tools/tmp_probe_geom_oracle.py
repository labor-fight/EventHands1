#!/usr/bin/env python3
"""Plan D's precondition: does the analytic contour residual actually point at the truth?

Plan D hands the well-observed directions to a damped Gauss-Newton solver on the contour-normal
residual and keeps the network for association, outlier scale and the weakly observed directions.
That only makes sense if the residual has descent to give. So before any solver, association head
or block-sparse machinery: take the *frozen* state the tracker is actually sitting on, build the
residual with oracle association (KSSF's own nearest surface, no learning), take one damped step,
and ask whether the joint error went down.

    r_i    = n_i^T (u_i - pi(X_i(x)))         contour-normal residual, S6/S8's own definition
    H      = sum_i J_i^T J_i / sigma^2 + lambda I
    delta  = -H^-1 sum_i J_i^T r_i / sigma^2

Gate (pre-registered in plan/04): one GN step must descend in >80% of observable packets, and the
oracle must beat the network's own step by >1.1 mm, or plan D is closed with no network trained.

Two controls decide whether a descent is real. `random_delta` applies a step of the same norm in a
random direction: any argument that "moving at all helps" has to beat it. `network` is the frozen
model's own step on the same window, which is what plan D would have to improve on.

Temporary: delete after the verdict lands.
"""
from __future__ import annotations

import argparse
import json
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
from semkine import jacobian as JA                                # noqa: E402
from semkine import lie                                           # noqa: E402
from semkine.dataset import sequences_for_split                   # noqa: E402


BAND_PX = 12.0          # the contour band the S6 report and the halo lift both use
MAX_STEP = 0.30         # cap on ||delta||, so a rank-deficient solve cannot teleport the hand


def _gn_step(mano, kssf, p51, betas, K, ev5, device, damp, sigma_px):
    """One damped Gauss-Newton step on the contour-normal residual. Returns (delta51, stats)."""
    p = torch.from_numpy(np.ascontiguousarray(p51, np.float32)).view(1, -1).to(device).double()
    b = betas.double()
    fields = kssf(p.float(), betas, K, "mano_full_axis_angle")
    ev = torch.from_numpy(ev5).to(device)
    q = fields.query(ev[:, [1, 2]], ev[:, 0])

    sdf = q["sdf"].double()
    n = q["sdf_normal"].double()
    # Observable events: inside the contour band and with a defined normal. Outside events are
    # projected onto the contour along the normal, which is the halo lift's own association.
    ok = (sdf.abs() <= BAND_PX) & (n.norm(dim=-1) > 0.5)
    if int(ok.sum()) < 24:
        return None, {"n_used": int(ok.sum())}
    xy = ev[:, [1, 2]].double()[ok]
    sdf_o, n_o = sdf[ok], n[ok]
    proj = xy - sdf_o.clamp_min(0.0).unsqueeze(-1) * n_o
    qp = fields.query(proj.float(), ev[ok][:, 0])
    face = qp["face_id"].long()
    good = face >= 0
    if int(good.sum()) < 24:
        return None, {"n_used": int(good.sum())}

    j0 = (mano.J_regressor @ (mano.v_template
                              + torch.einsum("bl,mkl->bmk", b, mano.shapedirs)))[:, 0]
    state = lie.state_from_51d(p.double(), mano.hands_mean.double(), j0)
    fk = JA.forward_kinematics(mano, state, b)
    bi = torch.zeros(int(good.sum()), dtype=torch.long, device=device)
    pj = JA.query_jacobian(mano, fk, K.double(), kssf.render_scale, bi,
                           face[good], qp["bary"].double()[good],
                           normal=n_o[good], pixel=xy[good])
    r, J = pj.r, pj.J_r                                   # (M,), (M, 51)
    w = 1.0 / (sigma_px ** 2)
    A = (J.T @ J) * w
    g = (J.T @ r) * w
    # Levenberg damping relative to the curvature already present, so `damp` is scale free and a
    # sweep in it means the same thing on a 500-event window and a 20 000-event one. An absolute
    # lambda against a Hessian that grows with the event count is no damping at all.
    scale = float(A.diagonal().mean().clamp_min(1e-12))
    out = {
        "n_used": int(good.sum()),
        "resid_rms_px": float(r.pow(2).mean().sqrt()),
        # The decisive statistic for whether this residual can steer at all: events fire on both
        # sides of a moving edge, so a symmetric band leaves sum_i J_i^T r_i near zero however
        # many events it contains.
        "resid_mean_px": float(r.mean()),
        "resid_symmetry": float(r.mean().abs() / r.abs().mean().clamp_min(1e-12)),
        "outside_frac": float((sdf_o[good] > 0).double().mean()),
    }
    steps = {}
    for mu in damp:
        H = A + (mu * scale) * torch.eye(A.shape[0], device=device, dtype=A.dtype)
        try:
            delta = -torch.linalg.solve(H, g)
        except Exception:
            continue
        nrm = float(delta.norm())
        if not np.isfinite(nrm):
            continue
        if nrm > MAX_STEP:
            delta = delta * (MAX_STEP / nrm)
        steps[mu] = delta.float().cpu().numpy()
        out[f"step_norm_mu{mu:g}"] = nrm
    ev_num = torch.linalg.eigvalsh(A)
    out["cond"] = float(ev_num.max() / ev_num.clamp_min(1e-12).min())
    return (steps or None), out


@torch.no_grad()
def probe_sequence(model, mano, mano_d, kssf, cfg, root, legacy_dir, seq, step_ms, device,
                   damp, sigma_px, stride, max_steps):
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
    rrng = np.random.default_rng(999)
    rows, stats = [], []
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
            if (n_steps % stride) == 0 and len(ev5):
                steps, st = _gn_step(mano_d, kssf, prev, betas, K, ev5, device, damp, sigma_px)
                if steps is not None:
                    mus = sorted(steps)
                    # One random control per damping, matched in norm to that damping's step.
                    rnd = []
                    for mu in mus:
                        rd = rrng.normal(size=51).astype(np.float32)
                        rnd.append(rd * float(np.linalg.norm(steps[mu])
                                              / max(np.linalg.norm(rd), 1e-9)))
                    tgt = pos51[int(end)]
                    j = fk_j(np.stack([prev, net, tgt]
                                      + [prev + steps[mu] for mu in mus]
                                      + [prev + r for r in rnd]))
                    row = {"before": ra(j[0], j[2]), "network": ra(j[1], j[2])}
                    for i, mu in enumerate(mus):
                        row[f"gn_mu{mu:g}"] = ra(j[3 + i], j[2])
                        row[f"rand_mu{mu:g}"] = ra(j[3 + len(mus) + i], j[2])
                    rows.append(row)
                    stats.append(st)
            prev = net
            prev_t = torch.from_numpy(prev).view(1, -1).to(device)
            n_steps += 1
        if max_steps and n_steps >= max_steps:
            break
    return rows, stats, n_steps


@torch.no_grad()
def closed_loop_sequence(model, mano, mano_d, kssf, cfg, root, legacy_dir, seq, step_ms, device,
                         mu, sigma_px, max_steps, conds):
    """Run the recursive protocol with the analytic step in the loop.

    A one-step improvement is not a tracking improvement: the quantity every gate in this project
    is written against is the recursive error, and a corrector that helps once can still raise the
    loop gain. So each condition gets its own full rollout under the frozen protocol, and the only
    difference between them is the update rule.

        net       prev <- f(prev)                    the production arm
        net_gn    prev <- f(prev) + GN(f(prev))      analytic refinement of the network's step
        gn_only   prev <- prev + GN(prev)            no network at all, the negative control
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

    out = {}
    for cond in conds:
        rng = np.random.default_rng(0)          # same initialisation draw for every condition
        preds, gts, n_gn = [], [], 0
        n_steps = 0
        for a, b in runs:
            ends = np.arange(a + step_ms - 1, b, step_ms, dtype=np.int64)
            if not len(ends):
                continue
            prev = pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)
            for end in ends:
                if max_steps and n_steps >= max_steps:
                    break
                ev5 = ET._window_events(events, offsets, tsub, int(end), step_ms)
                if cond == "gn_only":
                    base = prev
                else:
                    pkt = ET.make_eval_packet(
                        ev5, torch.from_numpy(np.ascontiguousarray(prev, np.float32))
                        .view(1, -1).to(device), betas, K, step_ms, device)
                    base = model.forward_packet(pkt).cpu().numpy()[0]
                if cond != "net" and len(ev5):
                    steps, _ = _gn_step(mano_d, kssf, base, betas, K, ev5, device,
                                        [mu], sigma_px)
                    if steps is not None:
                        base = base + steps[mu]
                        n_gn += 1
                prev = base
                preds.append(prev)
                gts.append(pos51[int(end)])
                n_steps += 1
            if max_steps and n_steps >= max_steps:
                break
        pj, gj = fk_j(np.stack(preds)), fk_j(np.stack(gts))
        out[cond] = {
            "n": len(preds),
            "ra_mm": float(np.linalg.norm(ET.root_align(pj) - ET.root_align(gj),
                                          axis=-1).mean() * 1000),
            "abs_mm": float(np.linalg.norm(pj - gj, axis=-1).mean() * 1000),
            "gn_applied_frac": n_gn / max(len(preds), 1),
        }
    return out


def _summarise(rows, stats) -> dict:
    if not rows:
        return {"n": 0}
    keys = set().union(*[set(r) for r in rows])
    A = {k: np.array([r[k] for r in rows if k in r], np.float64) for k in keys}
    out = {
        "n": len(rows),
        "err_before_mm": float(A["before"].mean()),
        "err_after_network_mm": float(A["network"].mean()),
        "frac_network_descends": float((A["network"] < A["before"]).mean()),
    }
    for k in sorted(k for k in keys if k.startswith("gn_mu")):
        mu = k[len("gn_"):]
        if len(A[k]) != len(A["before"]):
            continue
        out[f"err_after_{k}_mm"] = float(A[k].mean())
        out[f"frac_descends_{k}"] = float((A[k] < A["before"]).mean())
        rk = f"rand_{mu}"
        if rk in A and len(A[rk]) == len(A["before"]):
            out[f"err_after_{rk}_mm"] = float(A[rk].mean())
            out[f"frac_descends_{rk}"] = float((A[rk] < A["before"]).mean())
        out[f"frac_{k}_beats_network"] = float((A[k] < A["network"]).mean())
    if stats:
        for k in set().union(*[set(s) for s in stats]):
            v = [s[k] for s in stats if k in s]
            if v and isinstance(v[0], (int, float)):
                out[f"{k}_mean"] = float(np.mean(v))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val_core")
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--damp", default="0.001,0.01,0.1,1,10",
                    help="Levenberg factors relative to mean(diag(H)); swept in one pass")
    ap.add_argument("--sigma-px", type=float, default=3.0)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--closed-loop", action="store_true",
                    help="run full rollouts with the analytic step in the loop instead of the "
                         "single-step descent test")
    ap.add_argument("--cl-mu", type=float, default=0.01)
    ap.add_argument("--cl-conds", default="net,net_gn,gn_only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(a.config)
    model = MNISTModel.load_from_checkpoint(a.ckpt, cfg=cfg, map_location=device).to(device).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    mano_d = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).double().eval()
    kssf = model.kssf_on(device)
    root = Path(cfg["DATA"]["ROOT"])

    damp = [float(x) for x in a.damp.split(",") if x]

    if a.closed_loop:
        conds = [c for c in a.cl_conds.split(",") if c]
        per = []
        for seq, d in sequences_for_split(root, a.split, None):
            row = closed_loop_sequence(model, mano, mano_d, kssf, cfg, root, d, seq, a.step_ms,
                                       device, a.cl_mu, a.sigma_px, a.max_steps, conds)
            row["seq"] = seq
            per.append(row)
            print("  " + seq + ": " + "  ".join(
                f"{c}={row[c]['ra_mm']:.2f}" for c in conds), flush=True)
        pooled = {}
        for c in conds:
            w = np.array([p[c]["n"] for p in per], np.float64)
            pooled[c] = {k: float(np.sum([p[c][k] * wi for p, wi in zip(per, w)]) / w.sum())
                         for k in ("ra_mm", "abs_mm", "gn_applied_frac")}
            pooled[c]["n"] = int(w.sum())
        out = {"ckpt": a.ckpt, "mode": "closed_loop", "mu": a.cl_mu,
               "sigma_px": a.sigma_px, "max_steps": a.max_steps,
               "per_sequence": per, "pooled": pooled}
        print(json.dumps(pooled, indent=2))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2))
        print("wrote", a.out)
        return

    per = []
    for seq, d in sequences_for_split(root, a.split, None):
        rows, stats, n = probe_sequence(model, mano, mano_d, kssf, cfg, root, d, seq, a.step_ms,
                                        device, damp, a.sigma_px, a.stride, a.max_steps)
        s = _summarise(rows, stats)
        s["seq"] = seq
        per.append(s)
        best = min((k for k in s if k.startswith("err_after_gn_mu")), key=lambda k: s[k],
                   default=None)
        print(f"  {seq}: n={s['n']} before={s.get('err_before_mm', float('nan')):.2f} "
              f"net={s.get('err_after_network_mm', float('nan')):.2f} "
              f"best_gn={best}={s.get(best, float('nan')):.2f} "
              f"sym={s.get('resid_symmetry_mean', float('nan')):.3f}", flush=True)

    ok = [p for p in per if p["n"]]
    w = np.array([p["n"] for p in ok], np.float64)
    pooled = {k: float(np.sum([p[k] * wi for p, wi in zip(ok, w)]) / w.sum())
              for k in ok[0] if isinstance(ok[0][k], float)}
    out = {"ckpt": a.ckpt, "damp": a.damp, "sigma_px": a.sigma_px,
           "per_sequence": per, "pooled": pooled}
    print(json.dumps(pooled, indent=2))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
