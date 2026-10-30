#!/usr/bin/env python3
"""OBS-A OA2 of docs/EXPERIMENT_RECORDS.md#obs-prereg-20261026: model-side check of the event clock.

Frozen S39 (3 seeds, step 6000), E3 single step (W200 event image clipped to the segment start, prev = GT(end - 50);
`tools/mv2/common.py::SeqSamples`), the 20 base streams, every 4th endpoint of the evaluator's 50 ms grid (each segment's
first step skipped). For every sample, the root-aligned joint error (RA, mm, the evaluator's `evalx.mano_fk`) between the
output and the label shifted to end + alpha * e(end) + beta (rounded to 1 ms), with e the U -> H offset of the stream
(alpha = 0: events on the label clock H; alpha = 1: events on the average-frame-rate clock U). Only samples whose every
shifted target is a valid ms enter. (alpha_hat, beta_hat) minimise the mean error over samples (seed-averaged), pooled
and per stream; the 95 % interval of alpha_hat from 2,000 (sequence, 1 s) block bootstrap replicates; the paired
difference err(1, beta_hat) - err(0, beta_hat); the null replaces e by the e of another base stream.

    python tools/obs/oa_shift.py --out outputs/obs/oa2 [--workers 20 --threads 4]
    python tools/obs/oa_shift.py --smoke
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from obs import common as C  # noqa: E402

ALPHAS = np.array([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
BETAS = np.arange(-20.0, 20.0 + 1e-9, 2.5)
EVERY = 4
BATCH = 32
N_BOOT = 2000


def _mano():
    sys.path.insert(0, str(C.REPO / "tools" / "tracking"))
    import evalx as EX
    from config import load_config
    from mano_layer import ManoLayer
    cfg = load_config(C.REPO / "outputs" / "semkine" / "dt_dz_l3_s3407" / "config_resolved.yaml")
    return EX, ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).eval()


def ra_joints(EX, mano, P: np.ndarray, betas: np.ndarray) -> np.ndarray:
    import torch
    j, _ = EX.mano_fk(mano, np.asarray(P, np.float32), torch.tensor(betas, dtype=torch.float32).view(1, -1),
                      torch.device("cpu"))
    return j - j[..., :1, :]


def seq_job(item) -> dict:
    seq, seeds, threads, max_n, foreign = item
    import torch
    torch.set_num_threads(threads)
    try:
        EX, mano = _mano()
        S = C.MC.SeqSamples(seq, every=EVERY, max_n=max_n)
        aux = C.load_aux(seq, C.MC.seq_dir(seq))
        c_h, c_u = C.clocks_us(aux)
        valid = np.asarray(aux["valid_ms"], dtype=bool)
        n_ms = len(valid)
        ends = S.ends.astype(np.float64)
        e = C.e_ms_at(ends, c_h, c_u)
        fa = C.load_aux(foreign, C.MC.seq_dir(foreign))
        fh, fu = C.clocks_us(fa)
        dur_f_ms = float(fh[-1]) / 1e3                      # e vanishes at both ends: wrap the foreign clock periodically
        e_null = C.e_ms_at(np.mod(ends, dur_f_ms), fh, fu)
        tgt = np.rint(ends[:, None, None] + ALPHAS[None, :, None] * e[:, None, None] + BETAS[None, None, :]).astype(np.int64)
        tgt_null = np.rint(ends[:, None] + 1.0 * e_null[:, None] + BETAS[None, :]).astype(np.int64)

        def ok_t(t):
            inside = (t >= 0) & (t < n_ms)
            good = np.zeros(t.shape, bool)
            good[inside] = valid[t[inside]]
            return good
        keep = ok_t(tgt).reshape(len(ends), -1).all(1) & ok_t(tgt_null).all(1)
        idx = np.flatnonzero(keep)
        betas = S.betas
        uniq = np.unique(np.concatenate([tgt[idx].ravel(), tgt_null[idx].ravel()]))
        Pt = np.stack([np.asarray(S.pos51[int(t)], np.float32) for t in uniq]) if len(uniq) else np.zeros((0, 51), np.float32)
        Jt = ra_joints(EX, mano, Pt, betas) if len(uniq) else np.zeros((0, 21, 3), np.float32)
        pos = {int(t): k for k, t in enumerate(uniq)}
        errs, errs_null = {}, {}
        for seed in seeds:
            model, _ = C.MC.load_s39(seed, torch.device("cpu"))
            # the per-sequence hand context, exactly as evalx.run_sequence sets it (betas (1, 10), K (1, 3, 3))
            model.set_hand_context(torch.tensor(S.betas, dtype=torch.float32).view(1, -1),
                                   torch.tensor(S.K, dtype=torch.float32).view(1, 3, 3))
            outs = []
            with torch.no_grad():
                for i in range(0, len(idx), BATCH):
                    b = idx[i:i + BATCH]
                    X = torch.from_numpy(S.lnes(b))
                    P = torch.from_numpy(S.prev[b])
                    outs.append(model(X, P).float().numpy())
            out = np.concatenate(outs) if outs else np.zeros((0, 51), np.float32)
            Jo = ra_joints(EX, mano, out, betas)
            k_t = np.vectorize(pos.get)(tgt[idx]) if len(idx) else np.zeros((0, len(ALPHAS), len(BETAS)), int)
            E = np.linalg.norm(Jo[:, None, None] - Jt[k_t], axis=-1).mean(-1) * 1000.0         # (n, A, B)
            k_n = np.vectorize(pos.get)(tgt_null[idx]) if len(idx) else np.zeros((0, len(BETAS)), int)
            En = np.linalg.norm(Jo[:, None] - Jt[k_n], axis=-1).mean(-1) * 1000.0             # (n, B)
            errs[str(seed)] = E.astype(np.float32)
            errs_null[str(seed)] = En.astype(np.float32)
        return {"seq": seq, "ends": S.ends[idx].tolist(), "e_ms": e[idx].tolist(), "n_samples_all": int(len(ends)),
                "n_kept": int(len(idx)), "foreign": foreign, "errs": {k: v.tolist() for k, v in errs.items()},
                "errs_null": {k: v.tolist() for k, v in errs_null.items()}}
    except Exception as ex:                                                     # noqa: BLE001
        import traceback
        return {"seq": seq, "error": f"{type(ex).__name__}: {ex}", "tb": traceback.format_exc()}


def argmin_ab(E: np.ndarray):
    m = E.mean(0)
    k = int(np.argmin(m))
    ia, ib = np.unravel_index(k, m.shape)
    return float(ALPHAS[ia]), float(BETAS[ib]), m


def fit_block(E: np.ndarray, groups: np.ndarray, seed: int = 20261026):
    """(alpha_hat, beta_hat, mean grid, alpha 95 % interval over (sequence, 1 s) block bootstrap replicates)."""
    a_hat, b_hat, m = argmin_ab(E)
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    gidx = {g: np.flatnonzero(groups == g) for g in ug}
    sums = np.stack([E[gidx[g]].sum(0) for g in ug])
    cnts = np.array([len(gidx[g]) for g in ug], dtype=np.float64)
    a_boot = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(ug), len(ug))
        mm = sums[pick].sum(0) / cnts[pick].sum()
        ia, _ = np.unravel_index(int(np.argmin(mm)), mm.shape)
        a_boot.append(float(ALPHAS[ia]))
    return a_hat, b_hat, m, [float(np.percentile(a_boot, 2.5)), float(np.percentile(a_boot, 97.5))]


def read_alpha(lo: float, hi: float) -> str:
    """H: the interval holds 0 and not 1; U: holds 1 and not 0; U attenuated: lies above 0 without reaching 1 (S39 was
    trained on stamp-time labels, so under U its outputs may follow the events only partly); otherwise undetermined."""
    if lo <= 0.0 <= hi and not (lo <= 1.0 <= hi):
        return "supports_H"
    if lo <= 1.0 <= hi and not (lo <= 0.0 <= hi):
        return "supports_U"
    if lo > 0.0 and hi < 1.0:
        return "supports_U_attenuated"
    return "cannot_discriminate"


def aggregate(rows) -> dict:
    ok = [r for r in rows if "error" not in r and r["n_kept"] > 0]
    Es = {r["seq"]: np.mean([np.asarray(r["errs"][s]) for s in r["errs"]], 0) for r in ok}           # (n, A, B)
    Ens = {r["seq"]: np.mean([np.asarray(r["errs_null"][s]) for s in r["errs_null"]], 0) for r in ok}  # (n, B)
    gr = {r["seq"]: np.asarray([f"{r['seq']}|{int(e) // 1000}" for e in r["ends"]]) for r in ok}
    out = {"alphas": ALPHAS.tolist(), "betas": BETAS.tolist(), "errors": [r for r in rows if "error" in r]}
    i0, i1 = int(np.flatnonzero(ALPHAS == 0.0)[0]), int(np.flatnonzero(ALPHAS == 1.0)[0])
    for name, sel in (("all", [r["seq"] for r in ok]), ("zgz", [r["seq"] for r in ok if r["seq"] in C.ZGZ]),
                      ("train", [r["seq"] for r in ok if r["seq"] in C.BASE_TRAIN])):
        if not sel:
            continue
        E = np.concatenate([Es[s] for s in sel])
        En = np.concatenate([Ens[s] for s in sel])
        g = np.concatenate([gr[s] for s in sel])
        a_hat, b_hat, m, (lo, hi) = fit_block(E, g)
        ib = int(np.flatnonzero(BETAS == b_hat)[0])
        d = E[:, i1, ib] - E[:, i0, ib]
        dn = En[:, ib] - E[:, i0, ib]
        out[name] = {"alpha_hat": a_hat, "beta_hat_ms": b_hat, "alpha_ci95": [lo, hi], "n_samples": int(len(E)),
                     "mean_err_grid": m.tolist(), "reading": read_alpha(lo, hi),
                     "diff_alpha1_minus_alpha0": {"mean": float(d.mean()), "ci95": list(C.block_ci(d, g))},
                     "null_diff_foreign_e": {"mean": float(dn.mean()), "ci95": list(C.block_ci(dn, g))}}
    out["per_seq"] = {}
    for s in Es:
        a_r, b_r, _ = argmin_ab(Es[s])
        out["per_seq"][s] = {"alpha_hat": a_r, "beta_hat": b_r, "n": int(len(Es[s])),
                             "err_alpha0_beta0": float(Es[s][:, i0, int(np.flatnonzero(BETAS == 0.0)[0])].mean())}
    out["verdict"] = out["all"]["reading"] if "all" in out else "no_data"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    C.force_cpu()
    base = list(C.BASE)
    foreign = {s: base[(i + 7) % len(base)] for i, s in enumerate(base)}
    if a.smoke:
        t0 = time.time()
        r = seq_job((base[0], (C.SEEDS[0],), a.threads, 12, foreign[base[0]]))
        if "error" in r:
            print(r["tb"])
            raise SystemExit(r["error"])
        E = np.asarray(r["errs"][str(C.SEEDS[0])])
        import torch
        S = C.MC.SeqSamples(base[0], every=EVERY, max_n=2)
        m, _ = C.MC.load_s39(C.SEEDS[0], torch.device("cpu"))
        X, P = torch.from_numpy(S.lnes([0, 1])), torch.from_numpy(S.prev[:2])
        with torch.no_grad():
            z0 = m(X, P)
            m.set_hand_context(torch.tensor(S.betas).view(1, -1), torch.tensor(S.K).view(1, 3, 3))
            z1 = m(X, P)
        print(f"[smoke] hand context active (output changes when set): {bool((z0 != z1).any())}")
        print(f"[smoke] {r['seq']}: kept {r['n_kept']}/{r['n_samples_all']}, err grid {E.shape}, "
              f"null {np.asarray(r['errs_null'][str(C.SEEDS[0])]).shape}, {time.time() - t0:.1f} s (shapes only)")
        return
    if not a.out:
        raise SystemExit("--out required")
    C.verify_protocol()
    out = Path(a.out)
    if (out / "result.json").exists():
        raise SystemExit(f"{out}/result.json exists; preserve it")
    t0 = time.time()
    items = [(s, tuple(C.SEEDS), a.threads, None, foreign[s]) for s in base]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        rows = pool.map(seq_job, items, chunksize=1)
    res = {"probe": "OA2", "decision": aggregate(rows), "rows_summary": [
        {k: r.get(k) for k in ("seq", "n_kept", "n_samples_all", "foreign", "error")} for r in rows],
        "wall_s": time.time() - t0}
    C.save_json(out / "result.json", res)
    np.savez_compressed(out / "errs.npz", **{f"{r['seq']}|{s}": np.asarray(r["errs"][s], np.float32)
                                             for r in rows if "error" not in r for s in r["errs"]})
    C.receipt(out, {"probe": "OA2"}, scripts=["tools/obs/oa_shift.py", "tools/obs/common.py"])
    print(f"[OA2] done in {time.time() - t0:.0f} s -> {out}/result.json")


if __name__ == "__main__":
    main()
