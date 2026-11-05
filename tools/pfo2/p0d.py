#!/usr/bin/env python3
"""PFO2 P0-D, does first-moment (torque) pooling of the base's H3 carry closed-loop state error that zeroth-order pooling
loses? (docs/EXPERIMENT_RECORDS.md#pfo2-p0-prereg-20261103; zero training, on B2's three seeds)

    python -m tools.pfo2.p0d run --arm A --seed S     one GPU; readings/pfo2_<A>_s<S>/cap_mom/ (fresh-only)
    python -m tools.pfo2.p0d decide --arm A           CPU; decisions/p0d.json, analysis/p0d.md

Capture: B2 stepped from its own closed-loop incoming states (its tl_e1 reading on the nine training local recordings =
fit, its e1 reading on zgz = test), every endpoint; at the LBS slot pooling: H3 (V x C), propagated support S, the
previous-state vertices and joints (J_regressor), the previous root rotation R. Per joint j with a_vj = W_vj S_v:
  zeroth_j = sum_v a_vj H_v / sum_v a_vj                                   (the existing slot's mean channel)
  first_j  = sum_v a_vj H_v (x) R^T (v - J_j) / L / sum_v a_vj, L = 1 cm  (C x 3)
plus the existing slots (17 x 258) as the reference. Targets: the incoming state's error against the GT previous state,
root (log(R_in R_gt^T), 3) and fingers (45), separate ridges, lambda by leave-one-recording-out CV. Score: the incoming
state corrected by the decoded error, MPJPE-RA against the GT previous state (zgz, frame-weighted overall).
Feature sets (all with prev48): slots / zeroth / zeroth + first. Gate: RA(zeroth) - RA(zeroth + first) >= 0.3 mm
(three-seed mean) -> OPEN (arm b); otherwise no moment-pooling arm.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pfo2.common import COUNTS, OUT, SEEDS, TRAIN_LOCAL, ZGZ, now, read, require, save, sha  # noqa: E402

L_M = 0.01
GATE = 0.3
SETS = ("slots", "zeroth", "zeroth+first")


def moments(H, S, W, verts, joints, R):
    """(zeroth (B, J, C), first (B, J, C, 3)); H (B, V, C), S (B, V), W (V, J), verts (B, V, 3), joints (B, J, 3), R (B, 3, 3)."""
    import torch
    a = W.unsqueeze(0) * S.unsqueeze(-1)                                                  # (B, V, J)
    den = a.sum(1).clamp_min(1e-6)                                                         # (B, J)
    zeroth = torch.einsum("bvj,bvc->bjc", a, H) / den.unsqueeze(-1)
    rel = verts.unsqueeze(2) - joints.unsqueeze(1)                                         # (B, V, J, 3)
    rel = torch.einsum("bik,bvji->bvjk", R, rel) / L_M                                     # R^T (v - J_j)
    first = torch.einsum("bvj,bvc,bvjk->bjck", a, H, rel) / den.view(*den.shape, 1, 1)
    return zeroth, first


class MomentCapture:
    """Captures the moment features of the latest trunk call (B2's trunk: semkine/pfo_trunk4.py)."""

    def __init__(self, model, npz):
        import torch
        from semkine import xyz_mesh as XM
        with np.load(str(npz), allow_pickle=False) as z:
            self.Jreg = torch.tensor(np.asarray(z["J_regressor"], np.float32))
        self.calls, self.last, self._pending = 0, None, None
        native_pool = XM.lbs_slot_pool
        xyz = model.xyz
        native = xyz.forward
        cap = self

        def pool(H, S, O, W, hard):
            cap._pending = (H.detach().float(), S.detach().float(), W.detach().float())
            return native_pool(H, S, O, W, hard)

        def forward(tok, d, mask, verts, assoc, prev48, *args, **kwargs):
            from semkine.pfo_trunk2 import rodrigues as rod
            cap._pending = None
            delta, stats, aux = native(tok, d, mask, verts, assoc, prev48, *args, **kwargs)
            require(cap._pending is not None, "trunk did not pool")
            H, S, Wl = cap._pending
            v = verts.float()
            J = torch.einsum("jv,bvk->bjk", cap.Jreg.to(v.device), v)
            R = rod(prev48[:, 0:3].float())
            z0, z1 = moments(H, S, Wl, v, J, R)
            cap.calls += 1
            cap.last = (aux["slots"].detach().float().cpu().numpy()[0], z0.cpu().numpy()[0], z1.cpu().numpy()[0],
                        prev48.detach().float().cpu().numpy()[0])
            return delta, stats, aux
        XM.lbs_slot_pool = pool
        xyz.forward = forward


def run(arm, seed):
    import torch
    from tools.pfo import engine as E
    from tools.pfo import nets as N
    from tools.pfo2 import run as R
    from tools.xcl import evaluate as XE
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, range(2, 8))), "one GPU among 2-7")
    XE.configure_environment()
    XE.set_policy()
    tag, sp = R.arm_spec(arm, seed, 6000)
    R.prepare(sp)
    out = OUT / "readings" / tag / "cap_mom"
    require(not out.exists(), "captured once: " + str(out))
    out.mkdir(parents=True)
    s = read(sp)
    device = torch.device("cuda")
    net = N.load(tag, route=None, tau=None, device=device,
                 spec={**s, "checkpoint": Path(s["checkpoint"]), "config": Path(s["config"])})
    cap = MomentCapture(net.model, net.cfg["MANO"]["NPZ"])
    save(out / "launch.json", {"arm": arm, "seed": seed, "tag": tag, "identity": net.identity, "tool_sha256": sha(__file__),
                               "started_utc": now()}, exclusive=True)
    arrays = {}
    with torch.no_grad():
        for group, names, task in (("fit", TRAIN_LOCAL, "tl_e1"), ("test", ZGZ, "e1")):
            src = np.load(OUT / "readings" / tag / task / "states.npz")
            root, seqs = E.sequences(net.cfg, names)
            for seq, directory in seqs:
                ctx = N.sequence(root, directory, seq, device)
                net.bind(ctx)
                rows = E.endpoint_rows(ctx.aux)
                if seq in COUNTS:
                    require(len(rows) == COUNTS[seq], "endpoint denominator changed")
                inc, gtp = src[f"{seq}/incoming"], src[f"{seq}/gt_previous"]
                require(len(inc) == len(rows), "own-loop endpoint count differs")
                rec = {k: [] for k in ("slots", "zeroth", "first", "prev48", "incoming", "gt_previous", "ok")}
                for i in range(len(rows)):
                    end = int(rows[i][1])
                    k0 = cap.calls
                    net.step(ctx, end, inc[i])
                    c = cap.last if cap.calls == k0 + 1 else None
                    rec["ok"].append(c is not None)
                    rec["slots"].append((c[0] if c else np.zeros((17, 258), np.float32)).astype(np.float16))
                    rec["zeroth"].append((c[1] if c else np.zeros((16, 128), np.float32)).astype(np.float16))
                    rec["first"].append((c[2] if c else np.zeros((16, 128, 3), np.float32)).astype(np.float16))
                    rec["prev48"].append(c[3] if c else np.asarray(inc[i][3:51], np.float32))
                    rec["incoming"].append(inc[i]); rec["gt_previous"].append(gtp[i])
                for k, v in rec.items():
                    arrays[f"{seq}/{k}"] = np.stack(v)
                arrays[f"{seq}/betas"] = np.asarray(ctx.aux["betas"], np.float32)
                save(out / "progress.json", {"done": seq, "at": now()})
    np.savez_compressed(out / "states.npz", **arrays)
    save(out / "report.json", {"status": "COMPLETE", "finished_utc": now(), "states_sha256": sha(out / "states.npz")},
         exclusive=True)


def state_error(inc, gt):
    from tools.s45loc.common import log_so3, rodrigues
    root = np.stack([log_so3(rodrigues(a) @ rodrigues(b).T) for a, b in zip(inc[:, 3:6], gt[:, 3:6])])
    return root, inc[:, 6:51] - gt[:, 6:51]


def corrected(inc, d_root, d_fing):
    from tools.s45loc.common import log_so3, rodrigues
    out = np.array(inc, np.float64, copy=True)
    out[:, 3:6] = np.stack([log_so3(rodrigues(-d) @ rodrigues(a)) for d, a in zip(d_root, inc[:, 3:6])])
    out[:, 6:51] = inc[:, 6:51] - d_fing
    return out


def design(z, seqs, fset):
    X, G, inc, gt, ok_all = [], [], [], [], []
    for gi, s in enumerate(seqs):
        ok = z[f"{s}/ok"].astype(bool)
        n = len(ok)
        parts = [z[f"{s}/prev48"].astype(np.float32).reshape(n, -1)]
        if fset == "slots":
            parts.append(z[f"{s}/slots"].astype(np.float32).reshape(n, -1))
        else:
            parts.append(z[f"{s}/zeroth"].astype(np.float32).reshape(n, -1))
            if fset == "zeroth+first":
                parts.append(z[f"{s}/first"].astype(np.float32).reshape(n, -1))
        X.append(np.concatenate(parts, 1)); G.append(np.full(n, gi))
        inc.append(z[f"{s}/incoming"]); gt.append(z[f"{s}/gt_previous"]); ok_all.append(ok)
    return (np.concatenate(X).astype(np.float64), np.concatenate(G), np.concatenate(inc).astype(np.float64),
            np.concatenate(gt).astype(np.float64), np.concatenate(ok_all))


def fit_predict2(X, targets, groups, tests, lambdas=None):
    """tools/pfo2/p0c.py's CV ridge for several targets on the same features: one eigendecomposition per fold serves
    every target and lambda; per target the lambda with the lowest pooled CV error; refit on all, predict the tests."""
    from tools.pfo2.p0c import LAMBDAS, ridge_fit, standardise
    lambdas = lambdas or LAMBDAS
    err = {t: {lam: 0.0 for lam in lambdas} for t in targets}
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        Xa, Xb = standardise(X[tr], X[te])
        evals, V = np.linalg.eigh(Xa.T @ Xa)
        Q = Xb @ V
        for t, Y in targets.items():
            ym = Y[tr].mean(0)
            P = V.T @ (Xa.T @ (Y[tr] - ym))
            for lam in lambdas:
                err[t][lam] += float(((Q @ (P / (evals + lam)[:, None]) + ym - Y[te]) ** 2).sum())
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-6, sd, np.inf)
    Xs = (X - mu) / sd
    out = {}
    for t, Y in targets.items():
        lam = min(err[t], key=err[t].get)
        ym = Y.mean(0)
        W = ridge_fit(Xs.copy(), Y - ym, lam)
        out[t] = ([((Xt - mu) / sd) @ W + ym for Xt in tests], lam)
    return out


def decide(arm):
    import torch
    from mano_layer import ManoLayer
    from tools.xcl.evaluate import score
    dest = OUT / "decisions" / "p0d.json"
    require(not dest.exists(), "decided once")
    mano = ManoLayer(str(ROOT / "assets/mano_right.npz"), add_mean=False).eval()
    res = {}
    for seed in SEEDS:
        z = np.load(OUT / "readings" / f"pfo2_{arm}_s{seed}" / "cap_mom" / "states.npz")
        for fset in SETS:
            Xf, gf, incf, gtf, okf = design(z, TRAIN_LOCAL, fset)
            rf, ff = state_error(incf, gtf)
            tests = {s: design(z, (s,), fset) for s in ZGZ}
            fit = fit_predict2(Xf[okf], {"root": rf[okf], "fingers": ff[okf]}, gf[okf], [tests[s][0] for s in ZGZ])
            (pr, lam_r), (pf, lam_f) = fit["root"], fit["fingers"]
            row = {"lambda_root": lam_r, "lambda_fingers": lam_f}
            for s, dr, df in zip(ZGZ, pr, pf):
                _, _, inc, gt, ok = tests[s]
                dr, df = np.where(ok[:, None], dr, 0.0), np.where(ok[:, None], df, 0.0)
                beta = torch.tensor(z[f"{s}/betas"]).view(1, -1)
                corr = corrected(inc, dr, df).astype(np.float32)
                row[s] = {"ra_corrected": float(score(mano, corr, gt.astype(np.float32), beta,
                                                      torch.device("cpu"))["mpjpe_ra_mm"].mean()),
                          "ra_incoming": float(score(mano, inc.astype(np.float32), gt.astype(np.float32), beta,
                                                     torch.device("cpu"))["mpjpe_ra_mm"].mean()), "n": int(len(inc))}
            n = sum(row[s]["n"] for s in ZGZ)
            row["overall"] = sum(row[s]["ra_corrected"] * row[s]["n"] for s in ZGZ) / n
            res[f"{seed}/{fset}"] = row
            print(seed, fset, round(row["overall"], 3), flush=True)
    mean = {f: float(np.mean([res[f"{s}/{f}"]["overall"] for s in SEEDS])) for f in SETS}
    gain = mean["zeroth"] - mean["zeroth+first"]
    out = {"arm": arm, "per_seed_set": res, "mean_overall": mean, "first_moment_gain": gain, "gate": GATE,
           "verdict": "OPEN" if gain >= GATE else "CLOSED", "decided_at_utc": now()}
    save(dest, out, exclusive=True)
    lines = ["## P0-D 力矩池化（B2 三种子；按解码误差纠正入态后的 MPJPE-RA，mm）", "",
             f"零阶 {mean['zeroth']:.2f}，零阶 + 一阶 {mean['zeroth+first']:.2f}，现有槽 {mean['slots']:.2f}；一阶增益 {gain:.2f}"
             f"（门 {GATE}）→ {out['verdict']}", "", "| 种子 / 特征 | local | global | overall | 入态（不纠正）local / global |",
             "|---|---|---|---|---|"]
    for k, r in res.items():
        lines.append(f"| {k} | {r['zgz_local']['ra_corrected']:.2f} | {r['zgz_global']['ra_corrected']:.2f} | "
                     f"{r['overall']:.2f} | {r['zgz_local']['ra_incoming']:.2f} / {r['zgz_global']['ra_incoming']:.2f} |")
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / "p0d.md").write_text("\n".join(lines) + "\n")
    print(out["verdict"], round(gain, 3), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True)
    r.add_argument("--seed", required=True, type=int)
    d = sub.add_parser("decide")
    d.add_argument("--arm", required=True)
    a = p.parse_args(argv)
    run(a.arm, a.seed) if a.verb == "run" else decide(a.arm)


if __name__ == "__main__":
    main()
