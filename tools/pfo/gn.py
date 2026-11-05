#!/usr/bin/env python3
"""PFO phase 2k: zero-training kill test of an explicit geometric update (docs/EXPERIMENT_RECORDS.md#pfo-p2k-prereg-20261103).

One Gauss-Newton / Levenberg-Marquardt step on the PFA reprojection residuals replaces the learned update. From the 51-D
start state x, every routed event's template point P_i(x) = sum_a bary_ia V_{f_ia}(x) (the trunk's own MANO FK,
`MNISTModel._fk`) is compared with the event's camera ray d_i (d_z = 1) in normalised image coordinates,
    e_i(x) = P_i,xy / P_i,z - d_i,xy,
and the step is
    x+ = x - (H + mu diag(H) + eps S^-2)^-1 g,   H = sum_i w_i J_i^T J_i / sum_i w_i,   g = sum_i w_i J_i^T e_i / sum_i w_i,
J_i = de_i/dx (forward mode through the FK), S = per-block scales (translation 0.01 m, root and finger axis-angle 0.1 rad),
w_i = 1 (ls) or the Huber weight at HUBER_MM of lateral residual at the start depth (h10). The routed set, faces,
barycentrics and rays do not depend on the state; they are captured once per endpoint from the trunk's own PFA pass at
the GT start (`xyz_keep_aux`). Starts and injected errors are the engine's `pert` task, draw for draw.

    python -m tools.pfo.gn freeze                    write manifests/p2k.json (exclusive)
    python -m tools.pfo.gn run --route R --set S     S = sel (ylf_local) | zgz; R = sr1 | t0p | label; CPU only
    python -m tools.pfo.gn select                    per route, the grid point with the lowest mean post-step MPJPE-RA
                                                     over the eight starts on ylf_local -> gn/selection.json
    python -m tools.pfo.gn decide                    the written kill rule -> gn/decision.json, gn/decision.md
"""
from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pfo.common import COUNTS, OUT, ZGZ, now, read, require, save, sha  # noqa: E402

GN = OUT / "gn"
MANIFEST = OUT / "manifests" / "p2k.json"
PREREG = "docs/EXPERIMENT_RECORDS.md#pfo-p2k-prereg-20261103"
ROUTES = ("sr1", "t0p", "label")
SETS = {"sel": ("ylf_local",), "zgz": ZGZ}
GRID_MU = (0.0, 0.25, 1.0, 4.0)
GRID_EPS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
ROBUST = ("ls", "h10")
HUBER_MM = 10.0
SCALES = (0.01,) * 3 + (0.1,) * 3 + (0.1,) * 45
Z_MIN = 0.05
THREADS = 12
#: the written kill rule (sr1 routes, zgz, the grid point selected on ylf_local)
RHO_MAX = 0.45
B_SEEDS = (3407, 3408, 3409)
FROZEN = ("tools/pfo/gn.py", "tests/test_pfo_gn.py", "tools/pfo/engine.py", "tools/pfo/nets.py", "tools/pfo/common.py",
          "tools/xcl/evaluate.py", "semkine/s45sm_fast.py", "semkine/s45sm.py", "semkine/s45sm_oracle.py",
          "model/model.py", "model/pose_repr.py")


def combos():
    return [(r, mu, eps) for r, mu, eps in itertools.product(ROBUST, GRID_MU, GRID_EPS)]


def combo_name(robust, mu, eps):
    return f"{robust}_mu{mu:g}_eps{eps:g}"


def starts():
    from tools.pfo.engine import PERT_KINDS
    return ("gt",) + tuple(PERT_KINDS)


# --------------------------------------------------------------------------- the step (pure torch, tested on CPU)
def residuals(V, JV, fv, bary, d):
    """Template points of the routed events at each start and their reprojection residuals.

    V (S, 778, 3), JV (S, 778, 3, 51), fv (N, 3) long, bary (N, 3), d (N, 3) -> P (S, N, 3), e (S, N, 2), de (S, N, 2, 51).
    """
    import torch
    b = bary.to(V.dtype)
    P = (b[None, :, :, None] * V[:, fv]).sum(2)
    dP = (b[None, :, :, None, None] * JV[:, fv]).sum(2)
    z = P[..., 2:3]
    e = P[..., :2] / z - d[None, :, :2].to(V.dtype)
    de = (dP[..., :2, :] * z.unsqueeze(-1) - P[..., :2].unsqueeze(-1) * dP[..., 2:3, :]) / (z * z).unsqueeze(-1)
    return P, e, de


def weights(robust, e, P):
    """Per-event weights: 1 (ls) or Huber at HUBER_MM of lateral residual at the start depth (h10); 0 behind Z_MIN."""
    import torch
    front = (P[..., 2] > Z_MIN).to(e.dtype)
    if robust == "ls":
        return front
    r_mm = e.norm(dim=-1) * P[..., 2] * 1000.0
    return front * (HUBER_MM / r_mm.clamp_min(1e-12)).clamp(max=1.0)


def normal_equations(e, de, w):
    """H = sum w J^T J / sum w and g = sum w J^T e / sum w per start; zero where no event carries weight."""
    import torch
    sw = w.sum(-1)
    H = torch.einsum("sn,snkp,snkq->spq", w, de, de)
    g = torch.einsum("sn,snkp,snk->sp", w, de, e)
    scale = torch.where(sw > 0, 1.0 / sw.clamp_min(1e-30), torch.zeros_like(sw))
    return H * scale[:, None, None], g * scale[:, None], sw


def solve_grid(H, g, mus=GRID_MU, epss=GRID_EPS, scales=SCALES):
    """Delta (len(mus), len(epss), S, 51) = -(H + mu diag(H) + eps S^-2)^-1 g."""
    import torch
    s2 = torch.tensor(scales, dtype=H.dtype, device=H.device) ** -2
    eye = torch.diag_embed(s2).expand_as(H)
    dg = torch.diag_embed(torch.diagonal(H, dim1=-2, dim2=-1))
    out = torch.empty(len(mus), len(epss), *g.shape, dtype=H.dtype, device=H.device)
    for a, mu in enumerate(mus):
        for c, eps in enumerate(epss):
            out[a, c] = -torch.linalg.solve(H + mu * dg + eps * eye, g.unsqueeze(-1)).squeeze(-1)
    return out


def block(kind):
    """The parameter block an injected kind perturbs (the engine's `inject` slices)."""
    from tools.pfo.engine import PERT_KINDS
    return {"T": slice(0, 3), "R": slice(3, 6), "F": slice(6, 51), "N": slice(0, 51)}[PERT_KINDS[kind][0]]


def rho(pred, base, delta, sl):
    """The engine's retention <f(x+d) - f(x), d> / |d|^2 on the perturbed block, row by row."""
    p, b, d = (np.asarray(a, np.float64)[..., sl] for a in (pred, base, delta))
    return ((p - b) * d).sum(-1) / np.maximum((d * d).sum(-1), 1e-12)


# --------------------------------------------------------------------------- frozen identity
def freeze():
    files = {f: sha(ROOT / f) for f in FROZEN}
    save(MANIFEST, {"name": "p2k", "frozen_at_utc": now(), "prereg": PREREG, "files": files,
                    "grid": {"mu": GRID_MU, "eps": GRID_EPS, "robust": ROBUST, "huber_mm": HUBER_MM, "scales": "0.01 m x3,"
                             " 0.1 rad x48"}, "rule": {"rho_max": RHO_MAX, "b_seeds": B_SEEDS}}, exclusive=True)
    print("frozen", MANIFEST)


def verify():
    m = read(MANIFEST)
    bad = [f for f, h in m["files"].items() if sha(ROOT / f) != h]
    require(not bad, f"frozen files changed: {bad}")


# --------------------------------------------------------------------------- one route x one set
def run(route, which, limit=None):
    import torch
    from tools.pfo import engine as E
    from tools.pfo import nets as N
    from tools.xcl import evaluate as XE
    require(route in ROUTES and which in SETS, "unknown route / set")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", "CPU only (CUDA_VISIBLE_DEVICES='')")
    if limit is None:
        verify()
        if which == "zgz":
            require((GN / "selection.json").exists(), "selection on ylf_local comes first")
    XE.configure_environment()
    XE.set_policy()
    torch.set_num_threads(THREADS)
    base = GN if limit is None else Path(os.environ["PFO_SMOKE_DIR"])
    out = base / route / which
    require(not out.exists(), "preserve existing reading: " + str(out))
    out.mkdir(parents=True)
    device = torch.device("cpu")
    net = N.load("pfasrdz", route=route, device=device)
    net.model.xyz_keep_aux = True
    names = SETS[which] if limit is None else tuple(os.environ["PFO_SMOKE_SEQS"].split(","))
    selected = read(GN / "selection.json")[route]["combo"] if (limit is None and which == "zgz") else None
    launch = {"route": route, "set": which, "sequences": list(names), "identity": net.identity,
              "manifest_sha256": sha(MANIFEST) if limit is None else None, "gn_sha256": sha(Path(__file__)),
              "grid": [combo_name(*c) for c in combos()], "selected": selected, "device": "cpu", "threads": THREADS,
              "limit": limit, "started_utc": now()}
    save(out / "launch.json", launch, exclusive=True)
    from mano_layer import ManoLayer
    mano = ManoLayer(net.cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root, seqs = E.sequences(net.cfg, names)
    prior = E.prior_noise_fn(net.cfg)
    kinds = tuple(E.PERT_KINDS)
    grid = combos()
    where = {c: j for j, c in enumerate(grid)}
    S = len(kinds) + 1
    arrays, blocks = {}, {}
    for seq, directory in seqs:
        ctx = N.sequence(root, directory, seq, device)
        net.bind(ctx)
        rows = E.endpoint_rows(ctx.aux)
        if seq in COUNTS and limit is None:
            require(len(rows) == COUNTS[seq], "endpoint denominator changed")
        n = len(rows) if limit is None else min(len(rows), limit)
        beta = torch.tensor(np.asarray(ctx.aux["betas"], np.float32), device=device).view(1, -1)
        fk1 = lambda t: net.model._fk(t.view(1, -1), ctx.betas)[0][0]  # noqa: E731
        jac = torch.func.vmap(torch.func.jacfwd(fk1))
        x0 = np.zeros((S, n, 51), np.float32)
        dlt = np.zeros((S, n, 51), np.float64)
        tgt = np.zeros((n, 51), np.float32)
        pred = np.zeros((len(grid), S, n, 51), np.float32)
        routed = np.zeros(n, np.int64)
        v2 = 0.0
        for i in range(n):
            _, end, prev_ms, _ = rows[i]
            truth = np.asarray(ctx.gt[int(prev_ms)], np.float32).copy()
            tgt[i] = np.asarray(ctx.gt[int(end)], np.float32)
            x0[0, i] = truth
            for k, kind in enumerate(kinds):
                x, d_, _ = E.inject(truth, kind, E.rng_for(seq, int(end), E.SALT["pert"] + k), prior)
                x0[k + 1, i], dlt[k + 1, i] = x, d_
            with torch.no_grad():
                net.model.xyz_last = None                                    # an empty packet returns before writing it
                net.step(ctx, end, truth)                                   # the trunk's own PFA pass at the GT start
                last = net.model.xyz_last
                if not last or "assoc" not in last:
                    pred[:, :, i] = x0[:, i][None]
                    continue
                valid = last["assoc"]["valid"][0]
                fv = last["assoc"]["vid"][0, :, :3][valid]
                bary = last["assoc"]["A"][0, :, :3][valid].double()
                ray = last["d"][0][valid].double()
                X = torch.from_numpy(np.ascontiguousarray(x0[:, i]))
                V = net.model._fk(X, ctx.betas.expand(S, -1))[0]
                v2 = max(v2, float((V[0] - last["verts"][0]).abs().max()))
            routed[i] = int(valid.sum())
            if routed[i] == 0:
                pred[:, :, i] = x0[:, i][None]
                continue
            JV = jac(X).double()
            P, e, de = residuals(V.double(), JV, fv, bary, ray)
            for a, rob in enumerate(ROBUST):
                H, g, _ = normal_equations(e, de, weights(rob, e, P))
                step = solve_grid(H, g)                                     # (mu, eps, S, 51)
                for m, mu in enumerate(GRID_MU):
                    for c, eps in enumerate(GRID_EPS):
                        pred[where[(rob, mu, eps)], :, i] = (X.double() + step[m, c]).float().numpy()
        require(np.isfinite(pred).all(), f"nonfinite step retained as failure: {seq}")
        blk = {"n": n, "routed_median": float(np.median(routed)), "v2_max_abs_m": v2, "combos": {}}
        blk["identity"] = {k: float(v.mean()) for k, v in E.score(mano, x0[0], tgt, beta, device).items()}
        ra = np.zeros((len(grid), S, n), np.float32)
        rh = np.zeros((len(grid), S - 1, n), np.float32)
        for j, cb in enumerate(grid):
            entry = {"kinds": {}}
            for s in range(S):
                m = E.score(mano, pred[j, s], tgt, beta, device)
                ra[j, s] = m["mpjpe_ra_mm"]
                if s == 0:
                    entry["gt"] = {k: float(v.mean()) for k, v in m.items()}
                    continue
                kind = kinds[s - 1]
                rh[j, s - 1] = rho(pred[j, s], pred[j, 0], dlt[s], block(kind))
                entry["kinds"][kind] = {"mpjpe_ra_mm": float(m["mpjpe_ra_mm"].mean()),
                                        "rho_median": float(np.median(rh[j, s - 1])), "rho_mean": float(rh[j, s - 1].mean())}
            entry["mean8"] = float(ra[j].mean(-1).mean())
            blk["combos"][combo_name(*cb)] = entry
        blocks[seq] = blk
        arrays.update({f"{seq}/mpjpe_ra": ra, f"{seq}/rho": rh, f"{seq}/routed": routed, f"{seq}/start": x0,
                       f"{seq}/delta": dlt, f"{seq}/target": tgt})
        if selected is not None:
            arrays[f"{seq}/prediction_selected"] = pred[[combo_name(*c) for c in grid].index(selected)]
        save(out / "progress.json", {"done": seq, "at": now()})
    np.savez_compressed(out / "arrays.npz", **arrays)
    overall = {}
    if all(s in blocks for s in ZGZ):
        nn_ = sum(blocks[s]["n"] for s in ZGZ)
        overall = {name: {"gt_mpjpe_ra_mm": sum(blocks[s]["combos"][name]["gt"]["mpjpe_ra_mm"] * blocks[s]["n"]
                                                for s in ZGZ) / nn_} for name in blocks[ZGZ[0]]["combos"]}
        overall["identity_mpjpe_ra_mm"] = sum(blocks[s]["identity"]["mpjpe_ra_mm"] * blocks[s]["n"] for s in ZGZ) / nn_
    save(out / "report.json", {**launch, "status": "COMPLETE", "finished_utc": now(), "per_sequence": blocks,
                               "overall": overall, "arrays_sha256": sha(out / "arrays.npz")}, exclusive=True)
    print("COMPLETE", out, flush=True)
    return out


# --------------------------------------------------------------------------- selection and the kill rule
def select():
    verify()
    sel = {"rule": "lowest mean post-step MPJPE-RA over the eight starts (GT + seven injected kinds) on ylf_local",
           "selected_at_utc": now()}
    for route in ROUTES:
        rep = read(GN / route / "sel" / "report.json")
        c = rep["per_sequence"]["ylf_local"]["combos"]
        best = min(c, key=lambda k: c[k]["mean8"])
        sel[route] = {"combo": best, "mean8": c[best]["mean8"], "report_sha256": sha(GN / route / "sel" / "report.json")}
    save(GN / "selection.json", sel, exclusive=True)
    print({r: sel[r]["combo"] for r in ROUTES})


def b_comparator():
    e3, r15, f15 = [], [], []
    for seed in B_SEEDS:
        tag = f"pfo_b_s{seed}"
        e3.append(read(OUT / "readings" / tag / "e3" / "report.json")["overall"]["mpjpe_ra_mm"])
        p = read(OUT / "readings" / tag / "pert" / "report.json")["per_sequence"]["zgz_local"]
        r15.append(p["R15"]["rho_median"]); f15.append(p["F15"]["rho_median"])
    return {"e3_mpjpe_ra_mm": e3, "e3_mean": float(np.mean(e3)), "local_R15_rho": r15, "local_F15_rho": f15}


def decide():
    verify()
    sel = read(GN / "selection.json")
    B = b_comparator()
    routes = {}
    for route in ROUTES:
        rep = read(GN / route / "zgz" / "report.json")
        name = sel[route]["combo"]
        loc = rep["per_sequence"]["zgz_local"]["combos"][name]
        glb = rep["per_sequence"]["zgz_global"]["combos"][name]
        routes[route] = {"combo": name, "e3_overall_mpjpe_ra_mm": rep["overall"][name]["gt_mpjpe_ra_mm"],
                         "identity_overall_mpjpe_ra_mm": rep["overall"]["identity_mpjpe_ra_mm"],
                         "e3_local": loc["gt"]["mpjpe_ra_mm"], "e3_global": glb["gt"]["mpjpe_ra_mm"],
                         "local": {k: v for k, v in loc["kinds"].items()}, "global": {k: v for k, v in glb["kinds"].items()}}
        if route == "sr1":
            ok_any = [k for k in rep["overall"] if k != "identity_mpjpe_ra_mm"
                      and rep["overall"][k]["gt_mpjpe_ra_mm"] <= B["e3_mean"]
                      and rep["per_sequence"]["zgz_local"]["combos"][k]["kinds"]["R15"]["rho_median"] < RHO_MAX
                      and rep["per_sequence"]["zgz_local"]["combos"][k]["kinds"]["F15"]["rho_median"] < RHO_MAX]
            routes[route]["descriptive_grid_points_meeting_both"] = ok_any
    s = routes["sr1"]
    k1 = s["e3_overall_mpjpe_ra_mm"] > B["e3_mean"]
    k2 = s["local"]["R15"]["rho_median"] >= RHO_MAX or s["local"]["F15"]["rho_median"] >= RHO_MAX
    verdict = "KILL" if (k1 or k2) else "PASS"
    dec = {"prereg": PREREG, "decided_at_utc": now(), "verdict": verdict, "K1_bias_exceeds_B": k1,
           "K2_retention_not_below": k2, "rho_max": RHO_MAX, "B": B, "routes": routes,
           "selection_sha256": sha(GN / "selection.json")}
    save(GN / "decision.json", dec, exclusive=True)
    lines = [f"# PFO 2k decision: {verdict}", "",
             f"B (learned update, seeds {', '.join(map(str, B_SEEDS))}): E3 overall MPJPE-RA mean {B['e3_mean']:.2f} mm; "
             f"local rho R15 {np.mean(B['local_R15_rho']):.2f}, F15 {np.mean(B['local_F15_rho']):.2f}", "",
             "| route | grid point | E3 overall | identity | E3 local / global | local rho R15 / F15 / T30 | global rho R15 / F15 |",
             "|---|---|---|---|---|---|---|"]
    for route, r in routes.items():
        lines.append(f"| {route} | {r['combo']} | {r['e3_overall_mpjpe_ra_mm']:.2f} | {r['identity_overall_mpjpe_ra_mm']:.2f} | "
                     f"{r['e3_local']:.2f} / {r['e3_global']:.2f} | {r['local']['R15']['rho_median']:.2f} / "
                     f"{r['local']['F15']['rho_median']:.2f} / {r['local']['T30']['rho_median']:.2f} | "
                     f"{r['global']['R15']['rho_median']:.2f} / {r['global']['F15']['rho_median']:.2f} |")
    lines += ["", f"K1 (sr1 E3 > B): {k1}; K2 (sr1 local R15 or F15 rho >= {RHO_MAX}): {k2}"]
    (GN / "decision.md").write_text("\n".join(lines) + "\n")
    print(verdict, {"K1": k1, "K2": k2})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    sub.add_parser("freeze")
    r = sub.add_parser("run")
    r.add_argument("--route", required=True, choices=ROUTES)
    r.add_argument("--set", required=True, choices=tuple(SETS))
    r.add_argument("--limit", type=int, help="engineering smoke only: first endpoints, PFO_SMOKE_DIR / PFO_SMOKE_SEQS")
    sub.add_parser("select")
    sub.add_parser("decide")
    a = p.parse_args(argv)
    if a.verb == "freeze":
        freeze()
    elif a.verb == "run":
        run(a.route, a.set, a.limit)
    elif a.verb == "select":
        select()
    else:
        decide()


if __name__ == "__main__":
    main()
