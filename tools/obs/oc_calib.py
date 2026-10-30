#!/usr/bin/env python3
"""OBS-C of docs/EXPERIMENT_RECORDS.md#obs-prereg-20261026: the individual-calibration upper bound of S39.

Stage `run`: the frozen S39 trackers (3 seeds, step 6000) through the full S39 stack -- `AdaptiveFilter` with the ramp1000
spec, W200 window clipped to the segment start, Count50 (`evalx.run_sequence`'s closed loop) -- on CPU, with the bare
network wrapped by `GaugeNet(sigma)`: each step moves the previous state rigidly along its wrist ray to w / sigma (the
event gauge), runs the network, and moves the output back along its own wrist ray (w_e -> sigma * w_e); sigma = 1 is the
bare network bit for bit. Sigma grid 0.90 .. 1.06 step 0.01. The two zgz recordings run in the evaluator's order under one
`default_rng(0)` (the archived S39 rows' initial-state noise); each training base recording runs alone under
`default_rng(0)`. Per (seed, sigma, recording) the per-step pred / gt / end / run / count are stored.

Stage `analyze`: calibration sources Cal-A (first 100 evaluated steps of the recording), Cal-B (all steps of the subject's
other recording), Oracle-R (the evaluated steps themselves), Oracle-S (per segment), Oracle-sigma* (Stage 0 sigma*, the
nearest grid value, no output-side offsets); parameters in order sigma (min RA on the calibration steps) -> constant root
rotation offset (left or right, chordal mean, side by calibration RA) -> 45-d finger offset (mean residual x n / (n + 50)).
Every source is scored on the same steps (each recording's steps after its first 100) against the uncalibrated baseline;
single-parameter versions too. Gate on zgz (pre-registered), training recordings descriptive.

    python tools/obs/oc_calib.py run --out outputs/obs/oc [--workers 60 --threads 2]
    python tools/obs/oc_calib.py analyze --out outputs/obs/oc [--workers 24]
    python tools/obs/oc_calib.py smoke
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from obs import common as C  # noqa: E402

SIGMAS = tuple(round(0.90 + 0.01 * k, 2) for k in range(17))          # 0.90 .. 1.06
CAL_A_STEPS = 100
SHRINK_N0 = 50.0
SPEC_PATH = C.REPO / "outputs" / "dt3" / "gate_specs" / "ramp1000.json"
ARCHIVE = C.REPO / "outputs" / "dt3" / "gate_eval" / "ramp1000_w200" / "dt_dz_l3_s{seed}"
WIN, MAX_WIN = 200, 300
TOL_REPRO_MM = 0.30
MDE = {"local": 1.149, "global": 0.793}
SD = {"local": 0.379, "global": 0.261}


def _imports():
    sys.path.insert(0, str(C.REPO / "tools" / "tracking"))
    sys.path.insert(0, str(C.REPO / "tools" / "dt"))
    import evalx as EX
    import filter_eval as FE
    from semkine.anchored import AdaptiveFilter
    return EX, FE, AdaptiveFilter


def gauge_net_class():
    import torch
    import torch.nn as nn

    class GaugeNet(nn.Module):
        """The bare tracker run in the event gauge sigma (see the module docstring); sigma = 1 is the tracker itself."""

        def __init__(self, net, sigma: float):
            super().__init__()
            self.net = net
            self.sigma = float(sigma)

        def set_hand_context(self, betas, camera_K):
            if hasattr(self.net, "set_hand_context"):
                self.net.set_hand_context(betas, camera_K)

        def wrist(self, P, betas=None, camera_K=None):
            betas_f, _ = self.net._resolve_betas_K(P, betas, camera_K)
            _, j = self.net._fk(P.float(), betas_f.float())
            return j[:, 0].to(P.dtype)

        def forward(self, x, prevpos, betas=None, camera_K=None):
            if self.sigma == 1.0:
                return self.net(x, prevpos, betas=betas, camera_K=camera_K)
            s = self.sigma
            w = self.wrist(prevpos, betas, camera_K)
            p_e = prevpos.clone()
            p_e[:, :3] = prevpos[:, :3] + w * (1.0 / s - 1.0)
            z_e = self.net(x, p_e, betas=betas, camera_K=camera_K)
            w_e = self.wrist(z_e, betas, camera_K)
            z = z_e.clone()
            z[:, :3] = z_e[:, :3] + w_e * (s - 1.0)
            return z

    return GaugeNet


def zgz_order():
    from semkine.dataset import sequences_for_split
    return [(s, d) for s, d in sequences_for_split(C.DATA_ROOT, "val_core")]


def groups():
    """Run groups: the zgz pair in the evaluator's order (one rng), then each training base recording alone."""
    out = [("zgz", zgz_order())]
    for s in C.BASE_TRAIN:
        out.append((s, [(s, C.MC.seq_dir(s))]))
    return out


def run_path(out: Path, seed: int, sigma: float, seq: str) -> Path:
    return out / "runs" / f"s{seed}" / f"sigma{sigma:.2f}" / f"{seq}.npz"


def run_job(item):
    seed, sigma, gname, seqs, out, threads, max_steps = item
    import torch
    torch.set_num_threads(threads)
    try:
        EX, FE, AdaptiveFilter = _imports()
        GaugeNet = gauge_net_class()
        model, cfg = C.MC.load_s39(seed, torch.device("cpu"))
        spec = AdaptiveFilter.canonical_spec(FE.load_filter_specs(str(SPEC_PATH))[0])
        net = AdaptiveFilter.from_spec(GaugeNet(model, sigma), spec).eval()
        rng = np.random.default_rng(0)
        t0 = time.time()
        done = []
        for seq, d in seqs:
            p = run_path(Path(out), seed, sigma, seq)
            with torch.no_grad():
                r = EX.run_sequence(net, cfg, C.DATA_ROOT, d, seq, torch.device("cpu"), rng, "model", "fixed", WIN,
                                    0, MAX_WIN, clip_run_start=True, count_mode="last50", wset=())
            if max_steps is not None:
                return {"seed": seed, "sigma": sigma, "group": gname, "steps": int(len(r["end"])),
                        "wall_s": time.time() - t0, "keys": sorted(r.keys())}
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, pred=np.asarray(r["pred"], np.float32), gt=np.asarray(r["gt"], np.float32),
                                end=np.asarray(r["end"], np.int64), run=np.asarray(r["run"], np.int64),
                                count=np.asarray(r["count"], np.int64), betas=np.asarray(r["betas"], np.float32))
            done.append(seq)
        return {"seed": seed, "sigma": sigma, "group": gname, "done": done, "wall_s": time.time() - t0}
    except Exception as ex:                                                     # noqa: BLE001
        import traceback
        return {"seed": seed, "sigma": sigma, "group": gname, "error": f"{type(ex).__name__}: {ex}",
                "tb": traceback.format_exc()}


# ----------------------------------------------------------------------------------------------- calibration
def _rot():
    from scipy.spatial.transform import Rotation as R
    return R


def proj_so3(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def fit_root(pred: np.ndarray, gt: np.ndarray):
    """(left, right) constant offsets: R_b = proj(sum R_gt R_pred^T) (left) and proj(sum R_pred^T R_gt) (right)."""
    R = _rot()
    Rp = R.from_rotvec(pred[:, 3:6].astype(np.float64)).as_matrix()
    Rg = R.from_rotvec(gt[:, 3:6].astype(np.float64)).as_matrix()
    left = proj_so3(np.einsum("nij,nkj->ik", Rg, Rp))
    right = proj_so3(np.einsum("nji,njk->ik", Rp, Rg))
    return left, right


def apply_root(pred: np.ndarray, side: str, Rb: np.ndarray) -> np.ndarray:
    R = _rot()
    Rp = R.from_rotvec(pred[:, 3:6].astype(np.float64)).as_matrix()
    Rc = np.einsum("ij,njk->nik", Rb, Rp) if side == "left" else np.einsum("nij,jk->nik", Rp, Rb)
    out = pred.copy()
    out[:, 3:6] = R.from_matrix(Rc).as_rotvec().astype(np.float32)
    return out


class Scorer:
    def __init__(self):
        import torch
        EX, _, _ = _imports()
        from config import load_config
        from mano_layer import ManoLayer
        cfg = load_config(C.REPO / "outputs" / "semkine" / "dt_dz_l3_s3407" / "config_resolved.yaml")
        self.EX, self.torch = EX, torch
        self.mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).eval()

    def metrics(self, pred, gt, betas) -> dict:
        return self.EX.per_step_metrics(self.mano, {"pred": np.asarray(pred, np.float32), "gt": np.asarray(gt, np.float32),
                                                    "betas": betas}, self.torch.device("cpu"))

    def ra(self, pred, gt, betas) -> np.ndarray:
        return self.metrics(pred, gt, betas)["mpjpe_ra_mm"]


def ra_cache(sc: Scorer, runs: dict) -> dict:
    """Per-step RA of the uncorrected predictions of every sigma run (FK once per run)."""
    return {s: sc.ra(r["pred"], r["gt"], r["betas"]) for s, r in runs.items()}


def calibrate(sc: Scorer, runs: dict, cache: dict, cal_sel, use=("sigma", "root", "fingers"), sigma_fixed=None):
    """Fit the calibration on the steps `cal_sel` (bool mask) of the recording whose runs {sigma: arrays} and RA cache
    are given. Returns {"sigma", "root": (side, Rb) or None, "fingers": vec or None}."""
    out = {"sigma": 1.0, "root": None, "fingers": None}
    if sigma_fixed is not None:
        out["sigma"] = float(sigma_fixed)
    elif "sigma" in use:
        vals = [(float(cache[s][cal_sel].mean()), s) for s in SIGMAS]
        out["sigma"] = min(vals)[1]
    r = runs[out["sigma"]]
    pred, gt, betas = r["pred"][cal_sel], r["gt"][cal_sel], r["betas"]
    if "root" in use and len(pred):
        left, right = fit_root(pred, gt)
        cand = {"left": apply_root(pred, "left", left), "right": apply_root(pred, "right", right)}
        sc_l = float(sc.ra(cand["left"], gt, betas).mean())
        sc_r = float(sc.ra(cand["right"], gt, betas).mean())
        side = "left" if sc_l <= sc_r else "right"
        out["root"] = (side, left if side == "left" else right)
        pred = cand[side]
    if "fingers" in use and len(pred):
        n = len(pred)
        out["fingers"] = (gt[:, 6:] - pred[:, 6:]).mean(0) * (n / (n + SHRINK_N0))
    return out


def apply_cal(r: dict, cal: dict, sel) -> np.ndarray:
    pred = r["pred"][sel].copy()
    if cal["root"] is not None:
        pred = apply_root(pred, cal["root"][0], cal["root"][1])
    if cal["fingers"] is not None:
        pred[:, 6:] = pred[:, 6:] + cal["fingers"][None, :]
    return pred


def partner(seq: str) -> str:
    s, dom = seq.split("_")
    return f"{s}_{'global' if dom == 'local' else 'local'}"


def load_runs(out: Path, seed: int, seq: str) -> dict:
    runs = {}
    for s in SIGMAS:
        z = np.load(run_path(out, seed, s, seq))
        runs[s] = {k: z[k] for k in z.files}
    return runs


def sigma_star() -> dict:
    j = json.loads((C.REPO / "outputs" / "sba" / "p0v" / "result.json").read_text())
    return j


def analyze_job(item):
    seed, seq, out, sstar = item
    try:
        out = Path(out)
        sc = Scorer()
        runs = load_runs(out, seed, seq)
        pruns = load_runs(out, seed, partner(seq))
        cache, pcache = ra_cache(sc, runs), ra_cache(sc, pruns)
        base = runs[1.0]
        n = len(base["end"])
        ev = np.arange(n) >= CAL_A_STEPS                       # every source is scored on these steps
        allp = np.ones(len(pruns[1.0]["end"]), bool)
        res = {"seed": seed, "seq": seq, "n_steps": int(n), "n_eval": int(ev.sum())}
        keys = ("mpjpe_ra_mm", "mpvpe_ra_mm", "root_rot_deg", "mpjpe_abs_mm", "transl_mm")

        def score(cal, src_runs=None):
            r = runs[cal["sigma"]]
            pred = apply_cal(r, cal, ev)
            m = sc.metrics(pred, r["gt"][ev], r["betas"])
            return {k: float(m[k].mean()) for k in keys}, m["mpjpe_ra_mm"]

        bm, bra = score({"sigma": 1.0, "root": None, "fingers": None})
        res["baseline"] = bm
        blocks = np.asarray([f"{seq}|{int(e) // 1000}" for e in base["end"][ev]])
        variants = {}
        first = np.arange(n) < CAL_A_STEPS
        specs = {
            "calA_full": (runs, cache, first, ("sigma", "root", "fingers")),
            "calB_full": (pruns, pcache, allp, ("sigma", "root", "fingers")),
            "oracleR_full": (runs, cache, ev, ("sigma", "root", "fingers")),
            "calB_sigma": (pruns, pcache, allp, ("sigma",)),
            "calB_root": (pruns, pcache, allp, ("root",)),
            "calB_fingers": (pruns, pcache, allp, ("fingers",)),
            "oracleR_sigma": (runs, cache, ev, ("sigma",)),
            "oracleR_root": (runs, cache, ev, ("root",)),
            "oracleR_fingers": (runs, cache, ev, ("fingers",)),
            "calA_sigma": (runs, cache, first, ("sigma",)),
        }
        for name, (src, cch, sel, use) in specs.items():
            cal = calibrate(sc, src, cch, sel, use=use)
            m, ra = score(cal)
            d = ra - bra
            variants[name] = {"sigma": cal["sigma"], "root_side": None if cal["root"] is None else cal["root"][0],
                              "root_deg": None if cal["root"] is None else float(np.degrees(np.arccos(
                                  np.clip((np.trace(cal["root"][1]) - 1) / 2, -1, 1)))),
                              "metrics": m, "delta_ra": float(d.mean()), "delta_ra_ci95": list(C.block_ci(d, blocks))}
        # Oracle-S: per segment of the evaluated steps
        seg = base["run"]
        pred_s = np.zeros_like(base["pred"][ev])
        gt_s = base["gt"][ev]
        ra_s = np.zeros(int(ev.sum()))
        ev_idx = np.flatnonzero(ev)
        for sid in np.unique(seg[ev]):
            m_seg = np.zeros(n, bool)
            m_seg[ev_idx[seg[ev] == sid]] = True
            cal = calibrate(sc, runs, cache, m_seg, use=("sigma", "root", "fingers"))
            r = runs[cal["sigma"]]
            p = apply_cal(r, cal, m_seg)
            loc = np.flatnonzero(seg[ev] == sid)
            pred_s[loc] = p
            gt_s[loc] = r["gt"][m_seg]
        ms = sc.metrics(pred_s, gt_s, base["betas"])
        d = ms["mpjpe_ra_mm"] - bra
        variants["oracleS_full"] = {"metrics": {k: float(ms[k].mean()) for k in keys}, "delta_ra": float(d.mean()),
                                    "delta_ra_ci95": list(C.block_ci(d, blocks))}
        # Oracle-sigma*: Stage 0 sigma*, nearest grid value, no offsets
        if sstar is not None:
            s_near = min(SIGMAS, key=lambda s: abs(s - float(sstar)))
            m, ra = score({"sigma": s_near, "root": None, "fingers": None})
            d = ra - bra
            variants["oracle_sigma_star"] = {"sigma_star": float(sstar), "sigma": s_near, "metrics": m,
                                             "delta_ra": float(d.mean()), "delta_ra_ci95": list(C.block_ci(d, blocks))}
        res["variants"] = variants
        # reproduction of the full-recording S39 RA at sigma = 1 (validity, zgz only)
        res["baseline_all_steps_ra"] = float(cache[1.0].mean())
        return res
    except Exception as ex:                                                     # noqa: BLE001
        import traceback
        return {"seed": seed, "seq": seq, "error": f"{type(ex).__name__}: {ex}", "tb": traceback.format_exc()}


def archived_ra(seed: int, seq: str) -> float:
    f = next(iter(sorted(Path(str(ARCHIVE).format(seed=seed)).glob("af9f41a0*_w200_clip.json"))))
    j = json.loads(f.read_text())
    return float(j["model"][seq]["mpjpe_ra_mm"][0])


def decide(rows) -> dict:
    ok = [r for r in rows if "error" not in r]
    out = {"errors": [r for r in rows if "error" in r]}
    repro = {}
    for r in ok:
        if r["seq"] in C.ZGZ:
            a = archived_ra(r["seed"], r["seq"])
            repro[f"{r['seq']}|{r['seed']}"] = {"cpu": r["baseline_all_steps_ra"], "archived": a,
                                               "abs_diff": abs(r["baseline_all_steps_ra"] - a)}
    zgz_err = [r for r in out["errors"] if r.get("seq") in C.ZGZ]
    valid = bool(repro) and len(repro) == 6 and all(v["abs_diff"] <= TOL_REPRO_MM for v in repro.values()) and not zgz_err
    out["reproduction"] = repro
    out["validity"] = valid
    gate = {}
    for seq, dom in (("zgz_local", "local"), ("zgz_global", "global")):
        rs = [r for r in ok if r["seq"] == seq]
        calb = [-r["variants"]["calB_full"]["delta_ra"] for r in rs]               # improvement (positive = better)
        orr = [-r["variants"]["oracleR_full"]["delta_ra"] for r in rs]
        cala = [-r["variants"]["calA_full"]["delta_ra"] for r in rs]
        gate[dom] = {"calB_improvement_per_seed": calb, "oracleR_improvement_per_seed": orr,
                     "calA_improvement_per_seed": cala,
                     "calB_lever": bool(len(calb) == 3 and all(v >= MDE[dom] for v in calb)),
                     "oracleR_below_sd": bool(len(orr) == 3 and float(np.mean(orr)) < SD[dom]),
                     "calA_lever_descriptive": bool(len(cala) == 3 and all(v >= MDE[dom] for v in cala))}
    lever = gate["local"]["calB_lever"] or gate["global"]["calB_lever"]
    close = gate["local"]["oracleR_below_sd"] and gate["global"]["oracleR_below_sd"]
    out["flag_lever_and_close_both_hold"] = bool(lever and close)
    if not valid:
        verdict = "tool_defect"
    elif lever and close:
        verdict = "intermediate"                                      # contradictory conditions: registered precedence
    elif lever:
        verdict = "calibration_is_a_lever"
    elif close:
        verdict = "close_calibration_path"
    else:
        verdict = "intermediate"
    out["gate"] = gate
    out["verdict"] = verdict
    # descriptive pooled training recordings (in-sample)
    tr = [r for r in ok if r["seq"] in C.BASE_TRAIN]
    if tr:
        out["train_insample_mean_delta_ra"] = {k: float(np.mean([r["variants"][k]["delta_ra"] for r in tr]))
                                               for k in tr[0]["variants"]}
    return out


# ----------------------------------------------------------------------------------------------- main
def smoke(threads: int):
    import torch
    t0 = time.time()
    torch.set_num_threads(threads)
    GaugeNet = gauge_net_class()
    model, cfg = C.MC.load_s39(C.SEEDS[0], torch.device("cpu"))
    g1, g2 = GaugeNet(model, 1.0).eval(), GaugeNet(model, 0.95).eval()
    x = torch.zeros(2, 180, 240, 2)
    x[:, 80:100, 100:140, 0] = 0.5
    p = torch.zeros(2, 51)
    p[:, 2] = 0.6
    p[:, 3:6] = torch.tensor([2.2, 0.1, -0.2])
    with torch.no_grad():
        assert torch.equal(g1(x, p), model(x, p)), "sigma = 1 must be the bare network"
        z = g2(x, p)
    print(f"[smoke] GaugeNet sigma=1 bitwise ok; sigma=0.95 output {tuple(z.shape)} finite {bool(torch.isfinite(z).all())}")
    print(f"[smoke] groups: {[g for g, _ in groups()][:3]} ... ({len(groups())}); zgz order {[s for s, _ in zgz_order()]}")
    r = run_job((C.SEEDS[0], 0.95, "smoke", [(C.BASE_TRAIN[0], C.MC.seq_dir(C.BASE_TRAIN[0]))], "/tmp/none", threads, 1))
    if "error" in r:
        print(r["tb"])
        raise SystemExit(r["error"])
    print(f"[smoke] one full recording at sigma 0.95: {r['steps']} steps in {r['wall_s']:.0f} s "
          f"({1000 * r['wall_s'] / max(r['steps'], 1):.1f} ms/step), keys {r['keys']} (no metrics); total {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("run", "analyze", "smoke"))
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    C.force_cpu()
    if a.stage == "smoke":
        return smoke(a.threads)
    if not a.out:
        raise SystemExit("--out required")
    C.verify_protocol()
    out = Path(a.out)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    if a.stage == "run":
        if (out / "runs").exists():
            raise SystemExit(f"{out}/runs exists; preserve it")
        items = [(seed, s, g, seqs, str(out), a.threads, None) for g, seqs in groups() for seed in C.SEEDS for s in SIGMAS]
        items.sort(key=lambda it: (it[2] != "zgz",))                     # zgz jobs (two recordings) first
        with ctx.Pool(a.workers) as pool:
            rows = pool.map(run_job, items, chunksize=1)
        bad = [r for r in rows if "error" in r]
        C.save_json(out / "run_manifest.json", {"rows": rows, "n_jobs": len(rows), "n_failed": len(bad),
                                               "sigmas": SIGMAS, "wall_s": time.time() - t0})
        print(f"[OC run] {len(rows)} jobs, {len(bad)} failed, {time.time() - t0:.0f} s")
        if bad:
            raise SystemExit(1)
        return
    if (out / "result.json").exists():
        raise SystemExit(f"{out}/result.json exists; preserve it")
    recs = sigma_star().get("recordings", {})

    def sstar_of(seq):
        v = recs.get(seq, {})
        return float(v["sigma_star_q0_median"]) if "sigma_star_q0_median" in v else None
    items = [(seed, seq, str(out), sstar_of(seq)) for seq in C.BASE for seed in C.SEEDS]
    with ctx.Pool(a.workers) as pool:
        rows = pool.map(analyze_job, items, chunksize=1)
    res = {"probe": "OC", "rows": rows, "decision": decide(rows), "wall_s": time.time() - t0}
    C.save_json(out / "result.json", res)
    C.receipt(out, {"probe": "OC"}, scripts=["tools/obs/oc_calib.py", "tools/obs/common.py"])
    print(f"[OC analyze] done in {time.time() - t0:.0f} s -> {out}/result.json")


if __name__ == "__main__":
    main()
