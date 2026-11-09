#!/usr/bin/env python3
"""PMR S0 zero-training proxy (docs/EXPERIMENT_RECORDS.md#pmr-prereg-20261109): does the design's propagation have to
carry >= 1.0 mm? No network is trained; linear readouts of the design's fixed node statistics stand in for the arms.

    python -m tools.pmr.proxy capture --set tl|zgz --design d1      (one GPU among 2-7)
        B2 (PFO2 fgs seed 3407, r_s2d + PFA) stepped on the 9 training local recordings (tl) / zgz with two priors:
        gt = the GT previous state, loop = B2's own closed-loop incoming state (its tl_e1 / e1 reading). Per packet the
        design's fixed node statistics (semkine/pmr_trunk.py `geometry` + `statistics`, float64) and the 51-D
        increment target - prior; proxy/capture_<design>_<set>.npz
    python -m tools.pmr.proxy analyze --design d1                   (CPU)
        standardised ridge readouts with the connectivity of the arms:
          own   d theta_j from node j's statistics + its prev48 slice (j = 1..15); [d transl, d root] from the hand node's
                whole-hand statistics + residual share / presence + root slice  (= R-noprop's information)
          all   every output from the 17 nodes together                         (= R's receptive field after >= 2 rounds,
                                                                                 the hub joins every pair of parts)
        fitted on tl (gt and loop priors pooled; lambda by leave-recordings-out on tl), applied to zgz; prediction =
        prior + increment, scored with PFO's engine score (MPJPE-RA, mm). Also the zgz cross-fitted version
        (even / odd packets, lambda by 50-endpoint blocks inside the fit half) as the gshare probe did.
        proxy/<design>.json and .md (its `verdict` field = the first, tl-fitted reading of the gate; see `decide`)
    python -m tools.pmr.proxy decide --design d1                    (CPU)
        the registered S0 gate (docs/EXPERIMENT_RECORDS.md#pmr-prereg-20261109): on the zgz cross-fitted readouts, the
        loop-prior gap own - all >= 1.0 mm, with power: the `all` readout improves on the identity by >= 1.0 mm (a proxy
        that cannot improve on doing nothing by the gate's size cannot show a gap of that size) -> TRAIN, else
        NEXT_DESIGN; decisions/s0_<design>.json. The tl-fitted version written first (fit on the training recordings)
        has no such power on d1 (its readouts improve on the identity by <= 0.17 mm, and are worse than it under the GT
        prior), so it cannot measure a gap; disclosed in the registration.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pmr.common import OUT, PFO2_OUT, B2_TAG, TRAIN_LOCAL, ZGZ, now, read, require, save  # noqa: E402

GATE_MM = 1.0
LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
SETS = {"tl": ("tl_e1", TRAIN_LOCAL), "zgz": ("e1", ZGZ)}


def out_dir():
    d = OUT / "proxy"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- capture (GPU)
def capture(set_name, design):
    import torch
    from tools.pfo import engine as E
    from tools.pfo import nets as N
    from tools.xcl import evaluate as XE
    from tools.pmr import run as R
    from tools.pmr.common import b2_spec, check_b2
    require(design == "d1", "the proxy statistics of d1 (semkine/pmr_trunk.py)")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, range(2, 8))), "one GPU among 2-7")
    dst = out_dir() / f"capture_{design}_{set_name}.npz"
    require(not dst.exists(), "written once: " + str(dst))
    XE.configure_environment()
    XE.set_policy()
    check_b2()
    sp = read(b2_spec())
    R.prepare(b2_spec())                                            # B2's trunk (gshare) and the router kind
    device = torch.device("cuda")
    net = N.load(B2_TAG, route=None, tau=None, device=device,
                 spec={**sp, "checkpoint": Path(sp["checkpoint"]), "config": Path(sp["config"])})
    model = net.model
    model.xyz_keep_aux = True
    from semkine import xyz_mesh as XM
    from semkine import pmr_trunk as PM
    stat = type("PMRStat", (PM.PMRNet,), {"variant": "r"})(model.mano.weights, model.mano.f).to(device).eval()
    task, names = SETS[set_name]
    states = np.load(PFO2_OUT / "readings" / B2_TAG / task / "states.npz", allow_pickle=False)
    rec = {k: [] for k in ("sp", "sh", "Mp", "rs", "prior", "target", "seq", "cond", "end_ms")}
    root, seqs = E.sequences(net.cfg, names)
    t0 = time.time()
    with torch.no_grad():
        for si, (seq, directory) in enumerate(seqs):
            ctx = N.sequence(root, directory, seq, device)
            net.bind(ctx)
            inc, tgt, gtp, ends = (states[f"{seq}/{k}"] for k in ("incoming", "target", "gt_previous", "end_ms"))
            for i in range(len(ends)):
                for cond, prior in ((0, gtp[i]), (1, inc[i])):
                    net.step(ctx, int(ends[i]), prior)
                    L = model.xyz_last
                    prev48 = torch.from_numpy(np.asarray(prior, np.float32)).view(1, -1).to(device)[:, 3:51]
                    g = stat.geometry(L["mask"], L["d"], L["verts"], L["assoc"], prev48, False, torch.float64)
                    Mp, spv, Mh, sh = stat.statistics(g)
                    live = g["mask_f"].sum(1).clamp_min(1.0)
                    rec["sp"].append(spv[0].float().cpu().numpy())
                    rec["sh"].append(sh[0].float().cpu().numpy())
                    rec["Mp"].append(Mp[0].float().cpu().numpy())
                    rec["rs"].append(np.array([float(g["r"].sum() / live), float(g["r"].sum() > 0)], np.float32))
                    rec["prior"].append(np.asarray(prior, np.float32))
                    rec["target"].append(np.asarray(tgt[i], np.float32))
                    rec["seq"].append(si)
                    rec["cond"].append(cond)
                    rec["end_ms"].append(int(ends[i]))
                if i % 400 == 0:
                    print(seq, i, len(ends), round(time.time() - t0, 1), flush=True)
    arrays = {k: np.stack(v) if k in ("sp", "sh", "Mp", "rs", "prior", "target") else np.asarray(v) for k, v in rec.items()}
    arrays["seq_names"] = np.asarray([s for s, _ in seqs])
    np.savez_compressed(dst, **arrays)
    save(out_dir() / f"capture_{design}_{set_name}.json", {"set": set_name, "design": design, "rows": int(len(rec["seq"])),
                                                           "b2_checkpoint": sp["checkpoint"], "written_utc": now(),
                                                           "seconds": round(time.time() - t0, 1)}, exclusive=True)
    print("DONE", dst, flush=True)


# --------------------------------------------------------------------------- analysis (CPU)
def node_features(z):
    """own-node inputs: parts (n, 16, 28) = [statistics 25 | prev slice 3]; hand (n, 27)."""
    n = len(z["seq"])
    p3 = z["prior"][:, 3:51].reshape(n, 16, 3)
    part = np.concatenate([z["sp"], p3], -1)
    hub = np.concatenate([z["sh"], z["rs"], p3[:, 0]], -1)
    return part.astype(np.float64), hub.astype(np.float64)


def design_matrix(part, hub, out, conn):
    """Inputs for output group `out` (0 = [transl, root] from the hand node, j = 1..15 = joint j)."""
    if conn == "all":
        return np.concatenate([part.reshape(len(part), -1), hub], 1)
    return hub if out == 0 else part[:, out]


def targets(z, out):
    d = (z["target"] - z["prior"]).astype(np.float64)
    return d[:, 0:6] if out == 0 else d[:, 6 + 3 * (out - 1): 9 + 3 * (out - 1)]


def ridge_fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Xs = (X - mu) / sd
    ym = Y.mean(0)
    A = Xs.T @ Xs + lam * len(X) * np.eye(X.shape[1])
    return mu, sd, ym, np.linalg.solve(A, Xs.T @ (Y - ym))


def ridge_pred(m, X):
    mu, sd, ym, B = m
    return ((X - mu) / sd) @ B + ym


def choose_lambda(X, Y, groups):
    """Leave-groups-out (two folds of the group ids) squared error."""
    ug = np.unique(groups)
    fold = np.isin(groups, ug[::2])
    best, bl = None, None
    for lam in LAMS:
        e = 0.0
        for f in (fold, ~fold):
            m = ridge_fit(X[~f], Y[~f], lam)
            e += ((ridge_pred(m, X[f]) - Y[f]) ** 2).sum()
        if best is None or e < best:
            best, bl = e, lam
    return bl


def predict(ztr, zte, conn, groups_tr):
    """Increment predictions (n_te, 51) from ridge fits on ztr."""
    ptr, htr = node_features(ztr)
    pte, hte = node_features(zte)
    out = np.zeros((len(zte["seq"]), 51))
    lams = {}
    for o in range(16):
        Xtr, Xte = design_matrix(ptr, htr, o, conn), design_matrix(pte, hte, o, conn)
        Ytr = targets(ztr, o)
        lam = choose_lambda(Xtr, Ytr, groups_tr)
        lams[o] = lam
        P = ridge_pred(ridge_fit(Xtr, Ytr, lam), Xte)
        if o == 0:
            out[:, 0:6] = P
        else:
            out[:, 6 + 3 * (o - 1): 9 + 3 * (o - 1)] = P
    return out, lams


def scorer():
    import torch
    import yaml
    from mano_layer import ManoLayer
    from tools.pfo.engine import score, sequences
    mano = ManoLayer(str(ROOT / "assets/mano_right.npz"), add_mean=False).eval()
    cfg = yaml.safe_load((ROOT / "configs/pfo2_20261103/eval/pfo2_fgs_s3407.yaml").read_text())
    root, seqs = sequences(cfg, ZGZ)
    betas = {}
    for s, d in seqs:
        aux = np.load(Path(root) / d / f"{s}_aux.npz", allow_pickle=True)
        betas[s] = torch.tensor(np.asarray(aux["betas"], np.float32)).view(1, -1)

    def ra(z, inc, mask):
        tot, n, per = 0.0, 0, {}
        names = [str(x) for x in z["seq_names"]]
        for si, s in enumerate(names):
            sel = mask & (z["seq"] == si)
            if not sel.any():
                continue
            pred = z["prior"][sel] + inc[sel].astype(np.float32)
            m = score(mano, pred, z["target"][sel], betas[s], torch.device("cpu"))["mpjpe_ra_mm"]
            per[s] = float(m.mean())
            tot += float(m.sum())
            n += len(m)
        per["overall"] = tot / max(n, 1)
        return per
    return ra


def analyze(design):
    d = out_dir()
    dst = d / f"{design}.json"
    require(not dst.exists(), "written once: " + str(dst))
    tl = dict(np.load(d / f"capture_{design}_tl.npz", allow_pickle=False))
    zg = dict(np.load(d / f"capture_{design}_zgz.npz", allow_pickle=False))
    ra = scorer()
    res = {}
    # ---- fitted on the training recordings (both priors), applied to zgz
    for conn in ("own", "all"):
        inc, lams = predict(tl, zg, conn, tl["seq"])
        res[f"tl_fit/{conn}"] = {c: ra(zg, inc, zg["cond"] == ci) for ci, c in ((0, "gt"), (1, "loop"))}
        res[f"tl_fit/{conn}/lambda"] = {str(k): v for k, v in lams.items()}
    zero = np.zeros((len(zg["seq"]), 51))
    res["identity"] = {c: ra(zg, zero, zg["cond"] == ci) for ci, c in ((0, "gt"), (1, "loop"))}
    # ---- zgz cross-fitted (even / odd packets within each condition)
    pid = np.zeros(len(zg["seq"]), np.int64)
    for ci in (0, 1):
        sel = np.nonzero(zg["cond"] == ci)[0]
        pid[sel] = np.arange(len(sel))
    for conn in ("own", "all"):
        inc = np.zeros((len(zg["seq"]), 51))
        for fold in (0, 1):
            fit = pid % 2 == fold
            sub = lambda m: {k: (v[m] if k != "seq_names" else v) for k, v in zg.items()}  # noqa: E731
            p, _ = predict(sub(fit), sub(~fit), conn, (pid[fit] // 50))
            inc[~fit] = p
        res[f"zgz_cross/{conn}"] = {c: ra(zg, inc, zg["cond"] == ci) for ci, c in ((0, "gt"), (1, "loop"))}
    gap = {k: {c: res[f"{k}/own"][c]["overall"] - res[f"{k}/all"][c]["overall"] for c in ("gt", "loop")}
           for k in ("tl_fit", "zgz_cross")}
    verdict = "TRAIN" if gap["tl_fit"]["loop"] >= GATE_MM else "NEXT_DESIGN"
    report = {"design": design, "results": res, "gap_own_minus_all": gap, "gate_mm": GATE_MM, "verdict": verdict,
              "rows": {"tl": int(len(tl["seq"])), "zgz": int(len(zg["seq"]))}, "written_utc": now()}
    save(dst, report, exclusive=True)
    L = [f"# PMR S0 proxy, design {design} (zero training; ridge readouts of the fixed node statistics; MPJPE-RA mm, single step)", "",
         "| fit | readout | zgz gt prior (local / global / all) | zgz B2-loop prior (local / global / all) |", "|---|---|---|---|"]
    fmt = lambda r: f"{r['zgz_local']:.2f} / {r['zgz_global']:.2f} / {r['overall']:.2f}"  # noqa: E731
    L.append(f"| – | identity (no increment) | {fmt(res['identity']['gt'])} | {fmt(res['identity']['loop'])} |")
    for k in ("tl_fit", "zgz_cross"):
        for conn in ("own", "all"):
            L.append(f"| {k} | {conn} | {fmt(res[f'{k}/{conn}']['gt'])} | {fmt(res[f'{k}/{conn}']['loop'])} |")
    L += ["", f"gap own − all: tl-fit gt {gap['tl_fit']['gt']:.2f} / loop {gap['tl_fit']['loop']:.2f} mm; "
          f"zgz cross-fit gt {gap['zgz_cross']['gt']:.2f} / loop {gap['zgz_cross']['loop']:.2f} mm.",
          f"S0 gate (tl-fit, loop prior, >= {GATE_MM} mm): **{verdict}**"]
    (d / f"{design}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def decide(design):
    """The registered S0 gate on the zgz cross-fitted readouts (power: `all` improves on the identity by >= GATE_MM)."""
    rep = read(out_dir() / f"{design}.json")
    res = rep["results"]
    own, alll, ident = (res[k]["loop"]["overall"] for k in ("zgz_cross/own", "zgz_cross/all", "identity"))
    gap = own - alll
    power = ident - alll >= GATE_MM
    verdict = "TRAIN" if (power and gap >= GATE_MM) else "NEXT_DESIGN"
    dst = OUT / "decisions" / f"s0_{design}.json"
    save(dst, {"design": design, "rule": "zgz cross-fit, loop prior: own - all >= 1.0 mm and identity - all >= 1.0 mm",
               "own": own, "all": alll, "identity": ident, "gap": gap, "power": bool(power), "verdict": verdict,
               "gt_prior_gap": res["zgz_cross/own"]["gt"]["overall"] - res["zgz_cross/all"]["gt"]["overall"],
               "tl_fit_gap_loop": rep["gap_own_minus_all"]["tl_fit"]["loop"], "tl_fit_power": bool(
                   ident - res["tl_fit/all"]["loop"]["overall"] >= GATE_MM), "written_utc": now()}, exclusive=True)
    print(design, verdict, f"gap {gap:.2f} mm, all {alll:.2f} vs identity {ident:.2f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("capture")
    s.add_argument("--set", required=True, choices=tuple(SETS))
    s.add_argument("--design", default="d1")
    s = sub.add_parser("analyze")
    s.add_argument("--design", default="d1")
    s = sub.add_parser("decide")
    s.add_argument("--design", default="d1")
    a = p.parse_args(argv)
    if a.verb == "capture":
        return capture(a.set, a.design)
    if a.verb == "decide":
        return decide(a.design)
    return analyze(a.design)


if __name__ == "__main__":
    main()
