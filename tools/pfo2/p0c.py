#!/usr/bin/env python3
"""PFO2 P0-C, correction information (docs/EXPERIMENT_RECORDS.md#pfo2-p0-prereg-20261103): can the injected error be
decoded from what F's readout receives at the injected state?

    python -m tools.pfo2.p0c run --seed S      captures (tools/pfo2/engine.py cap_inj) for F seed S, deployed router
    python -m tools.pfo2.p0c decide            rho_feat per kind / seed, the network's own corrected fraction, the gate;
                                               decisions/p0c.json, analysis/p0c.md

Per kind (R15: delta[3:6]; F15: delta[6:51]) and seed: features = the 17 x 258 slots and prev48 (the readout's inputs
before its compression), z-scored on the fit set; ridge, lambda by leave-one-recording-out CV over the nine training
local recordings (every endpoint); refit on all of them; zgz test:
  rho_feat = 1 - sum |delta - delta_hat|^2 / sum |delta|^2          (decodable fraction)
  c_net    = 1 - sum <f(x + delta) - f(x), delta> / sum |delta|^2   (the network's own corrected fraction)
Gate: mean over the two kinds and three seeds of rho_feat on zgz_local >= 0.75 -> OPEN (A2).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pfo2.common import OUT, SEEDS, TRAIN_LOCAL, ZGZ, now, require, save  # noqa: E402

KINDS = {"R15": slice(3, 6), "F15": slice(6, 51)}
LAMBDAS = tuple(10.0 ** k for k in range(-1, 6))
GATE = 0.75


def ridge_fit(X, Y, lam):
    """Ridge on centred data (X, Y already centred): W = (X'X + lam I)^-1 X'Y."""
    G = X.T @ X
    G[np.diag_indices_from(G)] += lam
    return np.linalg.solve(G, X.T @ Y)


def standardise(Xf, Xt):
    mu, sd = Xf.mean(0), Xf.std(0)
    sd = np.where(sd > 1e-6, sd, np.inf)
    return (Xf - mu) / sd, (Xt - mu) / sd if Xt is not None else None


def cv_lambda(X, Y, groups, lambdas=LAMBDAS):
    """Leave-one-group-out CV: the lambda with the lowest pooled squared error (X, Y raw; folds standardise themselves;
    one eigendecomposition of the fold's Gram matrix serves every lambda)."""
    err = {lam: 0.0 for lam in lambdas}
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        Xa, Xb = standardise(X[tr], X[te])
        ym = Y[tr].mean(0)
        evals, V = np.linalg.eigh(Xa.T @ Xa)
        P = V.T @ (Xa.T @ (Y[tr] - ym))
        Q = Xb @ V
        for lam in lambdas:
            pred = Q @ (P / (evals + lam)[:, None]) + ym
            err[lam] += float(((pred - Y[te]) ** 2).sum())
    best = min(err, key=err.get)
    return best, err


def fit_predict(Xf, Yf, gf, tests):
    """CV lambda on the fit set, refit on all of it, predict every test matrix."""
    lam, err = cv_lambda(Xf, Yf, gf)
    mu, sd = Xf.mean(0), Xf.std(0)
    sd = np.where(sd > 1e-6, sd, np.inf)
    ym = Yf.mean(0)
    W = ridge_fit((Xf - mu) / sd, Yf - ym, lam)
    return [((Xt - mu) / sd) @ W + ym for Xt in tests], lam, err


def rho(y, yhat):
    return 1.0 - float(((y - yhat) ** 2).sum()) / float((y ** 2).sum())


def load(seed):
    z = np.load(OUT / "readings" / f"pfo_f1_s{seed}" / "cap_inj" / "states.npz")
    return z


def arrays(z, seqs, kind):
    X, Y, G, Fp, Bp = [], [], [], [], []
    sl = KINDS[kind]
    for gi, s in enumerate(seqs):
        ok = z[f"{s}/{kind}/captured"].astype(bool)
        slots = z[f"{s}/{kind}/slots"].astype(np.float32).reshape(len(ok), -1)
        X.append(np.concatenate([slots, z[f"{s}/{kind}/prev48"].astype(np.float32)], 1)[ok])
        Y.append(z[f"{s}/{kind}/delta"][:, sl][ok].astype(np.float64))
        G.append(np.full(int(ok.sum()), gi))
        Fp.append(z[f"{s}/{kind}/prediction"][:, sl][ok].astype(np.float64))
        Bp.append(z[f"{s}/prediction"][:, sl][ok].astype(np.float64))
    return (np.concatenate(X).astype(np.float64), np.concatenate(Y), np.concatenate(G), np.concatenate(Fp),
            np.concatenate(Bp))


def decide():
    dest = OUT / "decisions" / "p0c.json"
    require(not dest.exists(), "decided once")
    res = {}
    for seed in SEEDS:
        z = load(seed)
        for kind in KINDS:
            Xf, Yf, gf, _, _ = arrays(z, TRAIN_LOCAL, kind)
            tests = {s: arrays(z, (s,), kind) for s in ZGZ}
            preds, lam, err = fit_predict(Xf, Yf, gf, [tests[s][0] for s in ZGZ])
            row = {"n_fit": int(len(Yf)), "lambda": lam, "cv_error": {f"{k:g}": v for k, v in err.items()}}
            for s, yhat in zip(ZGZ, preds):
                _, Yt, _, Fp, Bp = tests[s]
                c_net = 1.0 - float(((Fp - Bp) * Yt).sum()) / float((Yt ** 2).sum())
                row[s] = {"rho_feat": rho(Yt, yhat), "c_net": c_net, "n": int(len(Yt))}
            res[f"{seed}/{kind}"] = row
            print(seed, kind, {s: round(row[s]["rho_feat"], 3) for s in ZGZ}, flush=True)
    local = [res[f"{s}/{k}"]["zgz_local"]["rho_feat"] for s in SEEDS for k in KINDS]
    stat = float(np.mean(local))
    out = {"per_seed_kind": res, "rho_feat_local_mean": stat,
           "c_net_local_mean": float(np.mean([res[f"{s}/{k}"]["zgz_local"]["c_net"] for s in SEEDS for k in KINDS])),
           "gate": GATE, "verdict": "OPEN" if stat >= GATE else "CLOSED", "decided_at_utc": now()}
    save(dest, out, exclusive=True)
    lines = ["## P0-C 纠错信息（F 三种子；zgz_local / zgz_global）", "",
             f"ρ_feat（local，两类三种子均值）= {stat:.3f}，门 {GATE} → {out['verdict']}；网络自身纠正份额 c_net = "
             f"{out['c_net_local_mean']:.3f}", "", "| 种子 / 注入 | ρ_feat local | ρ_feat global | c_net local | c_net global |",
             "|---|---|---|---|---|"]
    for k, r in res.items():
        lines.append(f"| {k} | {r['zgz_local']['rho_feat']:.3f} | {r['zgz_global']['rho_feat']:.3f} | "
                     f"{r['zgz_local']['c_net']:.3f} | {r['zgz_global']['c_net']:.3f} |")
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / "p0c.md").write_text("\n".join(lines) + "\n")
    print(out["verdict"], round(stat, 3), flush=True)


def run(seed):
    from tools.pfo2 import run as R
    from tools.pfo2 import engine as E2
    tag, sp = R.pfo_spec("f1", seed, 6000)
    R.prepare(sp)
    E2.run(tag, "cap_inj", None, str(sp))


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seed", required=True, type=int)
    sub.add_parser("decide")
    a = p.parse_args(argv)
    run(a.seed) if a.verb == "run" else decide()


if __name__ == "__main__":
    main()
