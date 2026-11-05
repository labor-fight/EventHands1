#!/usr/bin/env python3
"""PFO2 P0-A, a previous-state-free root anchor (docs/EXPERIMENT_RECORDS.md#pfo2-p0-prereg-20261103): how well do a
frozen router's features alone give the packet-end root rotation?

    python -m tools.pfo2.p0a extract --router r_s2d|r_wide     one GPU; readings/p0a/<router>/features.npz
    python -m tools.pfo2.p0a decide                            CPU; decisions/p0a.json, analysis/p0a.md

Features per window (50 ms, and 200 ms clipped at the segment start), from the frozen router on the packet:
  FPN map global mean and max (2 x fpn), per part k (16) with soft weights w_ik = softmax(part)_ik * sigmoid(onhand_i)
  over the live nodes: mass (sum w / n), weighted centroid (x / W, y / H) and scatter (var x, var y, cov xy), then
  log1p(#nodes with sigmoid(onhand) >= 0.5) and log1p(#live nodes).
Fit: every 4th endpoint of the 72 training sequences (9 subjects); test: every zgz endpoint. Targets: the packet-end root
rotation (6D: the first two columns) and translation, z-scored. Regressors: ridge (lambda by leave-one-subject-out CV)
and an MLP (2 x 256, epochs by 3-fold subject CV); per feature set the one with the lower CV error is used.
Score: root-only RA = MPJPE-RA of (GT translation, anchor root rotation, GT fingers) against GT; F's closed-loop
root-only RA = F's E1 predictions with GT fingers (three-seed mean, PFO's frozen readings).
Gate (r_s2d; feature set "50" or "50+200"): zgz_local <= F - 1.5 mm and zgz_global < F -> PASS.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pfo2.common import OUT, PFO_OUT, SEEDS, ZGZ, now, read, require, save, sha  # noqa: E402

ROUTERS = {"r_s2d": PFO_OUT / "routers/r_s2d/r_s2d_final.ckpt", "r_wide": PFO_OUT / "routers/r_wide/r_wide_final.ckpt"}
CFG = ROOT / "configs/pfo_20261103/pfo_f1_s3407.yaml"
FIT_STRIDE = 4
WINDOWS = (50, 200)
MARGIN = 1.5
LAMBDAS = tuple(10.0 ** k for k in range(-2, 5))


def features(net, b, H=180, W=240):
    import torch
    from semkine.encoder import EV_X, EV_Y
    from semkine.s45rt_router import event_image
    out = net(b.events, b.ptr, b.delta_t_s)
    fmap = net.trunk(event_image(b.events, b.ptr, b.delta_t_s).to(net.stem[0].weight.dtype)).float()
    g = torch.cat([fmap.mean((2, 3)), fmap.amax((2, 3))], 1)[0]
    mask = out["mask"][0]
    n = int(mask.sum())
    parts = out["part"].shape[-1]
    if n == 0 or b.events.shape[0] == 0:
        return torch.cat([g, torch.zeros(parts * 6 + 2, device=g.device)]).cpu().numpy()
    ev = b.events[out["src"][0][mask]]
    xy = torch.stack([ev[:, EV_X].float() / W, ev[:, EV_Y].float() / H], 1)
    on = torch.sigmoid(out["onhand"][0][mask].float())
    w = torch.softmax(out["part"][0][mask].float(), -1) * on.unsqueeze(1)                 # (n, 16)
    m = w.sum(0)
    c = (w.unsqueeze(-1) * xy.unsqueeze(1)).sum(0) / m.clamp_min(1e-6).unsqueeze(-1)       # (16, 2)
    d = xy.unsqueeze(1) - c.unsqueeze(0)
    vxx = (w * d[..., 0] ** 2).sum(0) / m.clamp_min(1e-6)
    vyy = (w * d[..., 1] ** 2).sum(0) / m.clamp_min(1e-6)
    vxy = (w * d[..., 0] * d[..., 1]).sum(0) / m.clamp_min(1e-6)
    per = torch.stack([m / n, c[:, 0], c[:, 1], vxx, vyy, vxy], 1).reshape(-1)
    cnt = torch.tensor([np.log1p(float((on >= 0.5).sum())), np.log1p(n)], device=g.device)
    return torch.cat([g, per, cnt]).cpu().numpy()


def extract(router):
    import torch
    from config import load_config
    from semkine import eval_track as ET
    from semkine import pfo_router as PR
    from semkine.dataset import sequences_for_split, splits_manifest
    from tools.pfo import engine as E
    from tools.pfo import nets as N
    from tools.xcl import evaluate as XE
    import os
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, range(2, 8))), "one GPU among 2-7")
    XE.configure_environment()
    XE.set_policy()
    out = OUT / "readings" / "p0a" / router
    require(not out.exists(), "extracted once: " + str(out))
    out.mkdir(parents=True)
    device = torch.device("cuda")
    net, _ = PR.load(ROUTERS[router], map_location="cpu")
    net = net.to(device).eval()
    cfg = load_config(str(CFG))
    root = Path(cfg["DATA"]["ROOT"])
    train = [s for s, _ in sequences_for_split(root, "train", splits_manifest(cfg))]
    save(out / "launch.json", {"router": router, "checkpoint_sha256": sha(ROUTERS[router]), "tool_sha256": sha(__file__),
                               "fit_sequences": train, "started_utc": now()}, exclusive=True)
    arrays = {}
    with torch.no_grad():
        for group, names in (("fit", train), ("test", ZGZ)):
            r, seqs = E.sequences(cfg, names)
            for seq, directory in seqs:
                ctx = N.sequence(r, directory, seq, device)
                rows = E.endpoint_rows(ctx.aux)
                runs = np.asarray(ctx.aux["valid_runs_ms"], np.int64).reshape(-1, 2)
                stride = FIT_STRIDE if group == "fit" else 1
                feats = {w: [] for w in WINDOWS}
                gt, ends, el = [], [], []
                prev = torch.zeros(1, 51, device=device)
                for i in range(0, len(rows), stride):
                    rid, end = int(rows[i][0]), int(rows[i][1])
                    a = int(runs[rid][0])
                    for w in WINDOWS:
                        win = min(w, end - a + 1)
                        ev = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, end, win)
                        b = ET.make_eval_packet(ev, prev, ctx.betas, ctx.K, win, device)
                        feats[w].append(features(net, b))
                    gt.append(np.asarray(ctx.gt[end], np.float32)); ends.append(end); el.append(end - a)
                for w in WINDOWS:
                    arrays[f"{seq}/w{w}"] = np.stack(feats[w]).astype(np.float32)
                arrays[f"{seq}/gt"] = np.stack(gt)
                arrays[f"{seq}/end_ms"] = np.asarray(ends)
                arrays[f"{seq}/elapsed_ms"] = np.asarray(el)
                arrays[f"{seq}/betas"] = np.asarray(ctx.aux["betas"], np.float32)
                save(out / "progress.json", {"done": seq, "at": now()})
    np.savez_compressed(out / "features.npz", **arrays)
    save(out / "report.json", {"status": "COMPLETE", "finished_utc": now(), "states_sha256": sha(out / "features.npz")},
         exclusive=True)


# --------------------------------------------------------------------------- fitting (CPU)
def six_d(aa):
    from tools.s45loc.common import rodrigues
    return np.stack([rodrigues(v)[:, :2].T.reshape(-1) for v in aa])


def from_six_d(x):
    """Gram-Schmidt of the two predicted columns -> axis-angle."""
    from tools.s45loc.common import log_so3
    out = []
    for v in x:
        a, b = np.array(v[0:3], np.float64), np.array(v[3:6], np.float64)      # columns 0 and 1 of R (`six_d`)
        a = a / max(np.linalg.norm(a), 1e-9)
        b = b - a * (a @ b)
        b = b / max(np.linalg.norm(b), 1e-9)
        R = np.stack([a, b, np.cross(a, b)], 1)
        out.append(log_so3(R))
    return np.asarray(out, np.float64)


def design(z, seqs, fset):
    keys = ("w50",) if fset == "50" else ("w50", "w200")
    X = np.concatenate([np.concatenate([z[f"{s}/{k}"] for k in keys], 1) for s in seqs]).astype(np.float64)
    gt = np.concatenate([z[f"{s}/gt"] for s in seqs]).astype(np.float64)
    Y = np.concatenate([six_d(gt[:, 3:6]), gt[:, 0:3]], 1)
    groups = np.concatenate([np.full(len(z[f"{s}/gt"]), sorted({q.split("_")[0] for q in seqs}).index(s.split("_")[0]))
                             for s in seqs])
    return X, Y, gt, groups


def zscore(Xf, *rest):
    mu, sd = Xf.mean(0), Xf.std(0)
    sd = np.where(sd > 1e-8, sd, np.inf)
    return [(X - mu) / sd for X in (Xf, *rest)]


def ridge_cv(X, Y, groups):
    err = {lam: 0.0 for lam in LAMBDAS}
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        Xa, Xb = zscore(X[tr], X[te])
        ym, ys = Y[tr].mean(0), Y[tr].std(0)
        Ya = (Y[tr] - ym) / ys
        ev, V = np.linalg.eigh(Xa.T @ Xa)
        P, Q = V.T @ (Xa.T @ Ya), Xb @ V
        Yb = (Y[te] - ym) / ys
        for lam in LAMBDAS:
            resid = Q @ (P / (ev + lam)[:, None]) - Yb
            err[lam] += float((resid ** 2).sum())
    lam = min(err, key=err.get)
    return lam, err[lam] / len(Y)


def ridge_fit_predict(X, Y, Xt, lam):
    Xa, Xb = zscore(X, Xt)
    ym, ys = Y.mean(0), Y.std(0)
    G = Xa.T @ Xa
    G[np.diag_indices_from(G)] += lam
    W = np.linalg.solve(G, Xa.T @ ((Y - ym) / ys))
    return (Xb @ W) * ys + ym


def mlp_train(X, Y, epochs, seed=0):
    import torch
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 256),
                              torch.nn.ReLU(), torch.nn.Linear(256, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, Yt = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    snaps = {}
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), 512):
            j = perm[i:i + 512]
            opt.zero_grad()
            torch.nn.functional.mse_loss(net(Xt[j]), Yt[j]).backward()
            opt.step()
        if ep % 10 == 0:
            snaps[ep] = {k: v.clone() for k, v in net.state_dict().items()}
    return net, snaps


def mlp_cv(X, Y, groups, max_epochs=200):
    import torch
    subjects = np.unique(groups)
    folds = [subjects[i::3] for i in range(3)]
    err = {}
    for f in folds:
        te = np.isin(groups, f)
        Xa, Xb = zscore(X[~te], X[te])
        ym, ys = Y[~te].mean(0), Y[~te].std(0)
        net, snaps = mlp_train(Xa, (Y[~te] - ym) / ys, max_epochs)
        for ep, sd in snaps.items():
            net.load_state_dict(sd)
            with torch.no_grad():
                p = net(torch.tensor(Xb, dtype=torch.float32)).numpy()
            err[ep] = err.get(ep, 0.0) + float(((p - (Y[te] - ym) / ys) ** 2).sum())
    ep = min(err, key=err.get)
    return ep, err[ep] / len(Y)


def mlp_fit_predict(X, Y, Xt, epochs):
    import torch
    Xa, Xb = zscore(X, Xt)
    ym, ys = Y.mean(0), Y.std(0)
    net, _ = mlp_train(Xa, (Y - ym) / ys, epochs)
    with torch.no_grad():
        return net(torch.tensor(Xb, dtype=torch.float32)).numpy().astype(np.float64) * ys + ym


def root_only(mano, aa, gt, betas):
    import torch
    from tools.xcl.evaluate import score
    pred = gt.copy()
    pred[:, 3:6] = aa
    beta = torch.tensor(betas, dtype=torch.float32).view(1, -1)
    return score(mano, pred.astype(np.float32), gt.astype(np.float32), beta, torch.device("cpu"))["mpjpe_ra_mm"]


def f_root_only(mano):
    """F's closed-loop root-only RA (E1 predictions with GT fingers), per sequence, three-seed mean."""
    import torch
    from tools.xcl.evaluate import score
    out = {s: [] for s in ZGZ}
    feats = np.load(OUT / "readings/p0a/r_s2d/features.npz")
    for seed in SEEDS:
        z = np.load(PFO_OUT / "readings" / f"pfo_f1_s{seed}" / "e1" / "states.npz")
        for s in ZGZ:
            pred, tgt = z[f"{s}/prediction"].copy(), z[f"{s}/target"]
            pred[:, 6:51] = tgt[:, 6:51]
            beta = torch.tensor(feats[f"{s}/betas"]).view(1, -1)
            out[s].append(float(score(mano, pred, tgt, beta, torch.device("cpu"))["mpjpe_ra_mm"].mean()))
    return {s: float(np.mean(v)) for s, v in out.items()}


def decide():
    from mano_layer import ManoLayer
    dest = OUT / "decisions" / "p0a.json"
    require(not dest.exists(), "decided once")
    mano = ManoLayer(str(ROOT / "assets/mano_right.npz"), add_mean=False).eval()
    F = f_root_only(mano)
    res = {"F_root_only": F}
    for router in ROUTERS:
        z = np.load(OUT / "readings" / "p0a" / router / "features.npz")
        train = read(OUT / "readings" / "p0a" / router / "launch.json")["fit_sequences"]
        for fset in ("50", "50+200"):
            X, Y, _, groups = design(z, train, fset)
            lam, e_r = ridge_cv(X, Y, groups)
            ep, e_m = mlp_cv(X, Y, groups)
            use = "ridge" if e_r <= e_m else "mlp"
            row = {"cv_error": {"ridge": e_r, "mlp": e_m}, "lambda": lam, "epochs": ep, "used": use}
            for s in ZGZ:
                Xt, _, gt, _ = design(z, [s], fset)
                pred = ridge_fit_predict(X, Y, Xt, lam) if use == "ridge" else mlp_fit_predict(X, Y, Xt, ep)
                ra = root_only(mano, from_six_d(pred[:, :6]), gt, z[f"{s}/betas"])
                row[s] = {"root_only_ra": float(ra.mean()),
                          "trans_err_mm": float(np.linalg.norm(pred[:, 6:9] - gt[:, 0:3], axis=1).mean() * 1000)}
            row["pass"] = bool(row["zgz_local"]["root_only_ra"] <= F["zgz_local"] - MARGIN and
                               row["zgz_global"]["root_only_ra"] < F["zgz_global"])
            res[f"{router}/{fset}"] = row
            print(router, fset, use, {s: round(row[s]["root_only_ra"], 2) for s in ZGZ}, "pass", row["pass"], flush=True)
    passed = [f for f in ("50", "50+200") if res[f"r_s2d/{f}"]["pass"]]
    res.update(verdict="PASS" if passed else "FAIL", a1_window_set=passed[0] if passed else None,
               rule="r_s2d, set 50 or 50+200: local <= F - 1.5 and global < F", decided_at_utc=now())
    save(dest, res, exclusive=True)
    lines = ["## P0-A 前态无关根锚（只根 RA，mm；local / global）", "",
             f"F 闭环只根 RA（三种子均值）{F['zgz_local']:.2f} / {F['zgz_global']:.2f}；门：local ≤ "
             f"{F['zgz_local'] - MARGIN:.2f} 且 global < {F['zgz_global']:.2f} → {res['verdict']}", "",
             "| 路由器 / 窗口 | 回归器 | 只根 RA local | 只根 RA global | 平移误差 local / global (mm) | 通过 |",
             "|---|---|---|---|---|---|"]
    for router in ROUTERS:
        for fset in ("50", "50+200"):
            r = res[f"{router}/{fset}"]
            lines.append(f"| {router} / {fset} | {r['used']} | {r['zgz_local']['root_only_ra']:.2f} | "
                         f"{r['zgz_global']['root_only_ra']:.2f} | {r['zgz_local']['trans_err_mm']:.1f} / "
                         f"{r['zgz_global']['trans_err_mm']:.1f} | {r['pass']} |")
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / "p0a.md").write_text("\n".join(lines) + "\n")
    print(res["verdict"], flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="verb", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--router", required=True, choices=tuple(ROUTERS))
    sub.add_parser("decide")
    a = p.parse_args(argv)
    extract(a.router) if a.verb == "extract" else decide()


if __name__ == "__main__":
    main()
