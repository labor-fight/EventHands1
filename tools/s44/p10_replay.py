#!/usr/bin/env python3
"""S44 P0.10: frozen S39 (or S43) closed loop with the event fast path, no training
(docs/EXPERIMENT_RECORDS.md#s44-prereg-20261019 section 2).

The loop is `tools/tracking/evalx.run_sequence` in mode "model" for the S39 stack (W200 clipped to the segment start,
ramp1000 `AdaptiveFilter`, Count50, protocol initial noise from one rng seeded 0 over val_core in order), written out
so that the step can also see the raw events. Per step:

    z   = net(LNES(end, w), prev)                    the frozen network, fed the OLD state, exactly as S39
    x-  = prev ;  fingers(x-) += dtheta_f             (mode fast_fingers; + root in fast_root; nothing in control)
    out = AdaptiveFilter(ramp1000) blending x- toward z     (x- in place of prev: with dtheta = 0 this IS the S39 stack)

dtheta = rate * 50 ms, the posterior mean of `evflow.solve_rate` from the normal flow of this step's events, deployable
owners and the Jacobian at prev, hyperparameters from `--hyper` (the training-subject selection). `--tf` instead runs
the network from the GT of the previous step (teacher forcing; diagnostic).

    python tools/s44/p10_replay.py --run outputs/semkine/dt_dz_l3_s3407 --mode control --out outputs/s44/p10/s39_s3407_control.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "dt"))
from s44 import evflow as EF  # noqa: E402
import filter_eval as FE  # noqa: E402
from mano_layer import ManoLayer  # noqa: E402
from semkine import eval_track as ET  # noqa: E402
from semkine import events as EV  # noqa: E402
from semkine.anchored import AdaptiveFilter  # noqa: E402
from semkine.async_tracker import _Fixed  # noqa: E402
from semkine.kinematic_gate import deadband_state  # noqa: E402

EX = FE.EX                    # the very evalx module filter_eval (hence the S39 main row) uses
SPEC = REPO / "outputs" / "dt3" / "gate_specs" / "ramp1000.json"
WIN = 200
WRITE_DEG = 1.0


def fast_increment(sq, run_a, end, prev64, betas, intr, mano, prec, hp, dev, stats):
    meas = EF.step_measurements(sq, run_a, end, hp["r"], hp["dfit_ms"], dev, mem_ms=hp.get("mem_ms", EF.W_MEM))
    inc = torch.zeros(51, dtype=torch.float64, device=dev)
    stats["n_meas"].append(meas["n_meas"])
    if meas["n_valid"] == 0:
        stats["n_used"].append(0)
        return inc
    _, uv, z, J = mano.jacobian(prev64, betas, intr)
    vis = EF.visible(uv[0], z[0])
    ids, w = EF.owners_deploy(meas["px"], uv[0], vis)
    A, mass = EF.rows_A(J[0], meas["n"], ids, w)
    sel = (meas["eps_px"] <= hp["eps_px"]) & (mass > 1e-6)
    stats["n_used"].append(int(sel.sum()))
    if int(sel.sum()) == 0:
        return inc
    th, _ = EF.solve_rate(A[sel], meas["s"][sel], mass[sel], hp["kappa"], hp["lam_p"], prec)
    return th * (EF.STEP * 1e-3)


@torch.no_grad()
def run(a):
    dev = torch.device("cuda")
    cfg, model, ckpt, step, _ = FE.build_run(Path(a.run), "last", dev)
    spec = json.loads(SPEC.read_text())
    filt = AdaptiveFilter.from_spec(_Fixed(), spec).to(dev).eval()
    ev_ch = EV.event_channels(cfg)
    mano32 = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(dev).eval()
    hp = json.loads(Path(a.hyper).read_text()) if a.hyper else None
    if a.mode != "control" and hp is None:
        raise SystemExit("--hyper is required for a fast-path mode")
    mano64 = EF.Mano64(dev) if a.mode != "control" else None
    prec = EF.rate_prior(dev)[0] if a.mode != "control" else None
    rng = np.random.default_rng(0)
    out, summ = {}, {}
    t0 = time.time()
    for s, d in EF.split_seqs("val_core"):
        events, offsets, aux, pos51 = ET.load_sequence(EF.DATA_ROOT, d, s)
        sq = EF.Seq(s, d) if a.mode != "control" else None
        betas = torch.tensor(aux["betas"], dtype=torch.float32, device=dev).view(1, -1)
        camera_K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=dev).view(1, 3, 3)
        model.set_hand_context(betas, camera_K)
        betas64 = betas.double()
        intr = sq.intrinsics() if sq is not None else None
        runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
        rec = {k: [] for k in ("pred", "prev", "gt", "end", "run", "elapsed", "count", "win", "writes", "inc")}
        stats = {"n_meas": [], "n_used": []}
        for run_id, (ra, rb) in enumerate(runs):
            ends = np.arange(ra + EF.STEP - 1, rb, EF.STEP, dtype=np.int64)
            if not len(ends):
                continue
            prev = pos51[ra].copy() + ET.sample_init_noise(cfg, rng, 1.0)
            prev_t = torch.from_numpy(prev).view(1, -1).to(dev)
            filt.reset_state(prev_t)
            for end in ends:
                end = int(end)
                n_ev = int(offsets[end + 1] - offsets[end - EF.STEP + 1])
                w = EX.window_of(offsets, end, "fixed", WIN, 0, 300, floor=int(ra))
                x = torch.from_numpy(ET.build_lnes(events, offsets, end, w, ev_ch)).unsqueeze(0).to(dev)
                rec["prev"].append(prev_t.cpu().numpy()[0])
                z = model(x, prev_t)
                inc = torch.zeros(51, dtype=torch.float64, device=dev)
                if a.mode != "control" and n_ev > 0:
                    inc = fast_increment(sq, int(ra), end, prev_t.double(), betas64, intr, mano64, prec, hp, dev, stats)
                xm = prev_t.clone()
                if a.mode in ("fast_fingers", "fast_root"):
                    xm[:, EF.FINGER] = (prev_t[:, EF.FINGER].double() + inc[EF.FINGER]).to(prev_t.dtype)
                if a.mode == "fast_root":
                    xm[:, 3:6] = (prev_t[:, 3:6].double() + inc[3:6]).to(prev_t.dtype)
                filt.trk_model.z = z
                pred = filt(x, xm, n_events=n_ev)
                if a.deadband and n_ev > 0:                       # S43 deviation 7 dead band (after the filter)
                    ef, er = (np.deg2rad(float(v)) for v in a.deadband.split(","))
                    pred = deadband_state(prev_t.to(pred.dtype), pred, ef, er)
                fj = inc[EF.FINGER].view(15, 3).norm(dim=-1)
                rec["writes"].append(int((fj >= np.deg2rad(WRITE_DEG)).sum()))
                rec["inc"].append(inc.float().cpu().numpy())
                prev_t = pred if not a.tf else torch.from_numpy(pos51[end].copy()).view(1, -1).to(dev)
                rec["pred"].append(pred.cpu().numpy()[0])
                rec["gt"].append(pos51[end])
                rec["end"].append(end)
                rec["run"].append(run_id)
                rec["elapsed"].append(end - int(ra))
                rec["count"].append(n_ev)
                rec["win"].append(w)
        r = {k: np.asarray(v) for k, v in rec.items()}
        r["pred"], r["gt"], r["prev"] = r["pred"].astype(np.float32), r["gt"].astype(np.float32), r["prev"].astype(np.float32)
        r["betas"] = aux["betas"]
        m = EX.per_step_metrics(mano32, r, dev)
        for k, v in r.items():
            if k != "betas":
                out[f"{s}|{k}"] = v
        for k, v in m.items():
            out[f"{s}|{k}"] = v.astype(np.float32)
        summ[s] = {"n": int(len(r["end"])), **{k: float(v.mean()) for k, v in m.items()},
                   "writes_per_query": float(r["writes"].mean()),
                   "meas_per_query": float(np.mean(stats["n_meas"])) if stats["n_meas"] else 0.0,
                   "used_per_query": float(np.mean(stats["n_used"])) if stats["n_used"] else 0.0}
        print(f"{s}: RA {summ[s]['mpjpe_ra_mm']:.4f}  writes {summ[s]['writes_per_query']:.2f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
    n = sum(v["n"] for v in summ.values())
    summ["overall"] = {k: sum(summ[s][k] * summ[s]["n"] for s in ("zgz_global", "zgz_local")) / n
                       for k in ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm", "root_rot_deg",
                                 "transl_mm", "writes_per_query")}
    meta = {"run": str(a.run), "ckpt": str(ckpt), "step": step, "mode": a.mode, "tf": bool(a.tf), "hyper": hp,
            "deadband": a.deadband,
            "spec": str(SPEC), "window": WIN, "clip_run_start": True, "wall_s": time.time() - t0, "summary": summ}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(summ["overall"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", choices=("control", "fast_fingers", "fast_root"), default="control")
    ap.add_argument("--hyper", default="")
    ap.add_argument("--tf", action="store_true")
    ap.add_argument("--deadband", default="", help="'EPS_F_DEG,EPS_R_DEG': update dead band after the filter")
    ap.add_argument("--out", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
