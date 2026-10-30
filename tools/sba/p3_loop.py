#!/usr/bin/env python3
"""SBA P3 (docs/EXPERIMENT_RECORDS.md#sba-prereg-20261022 P3): closed loop on zgz, zero training.

    standalone  SBA alone. Each valid run starts at GT + the protocol initial noise (ET.sample_init_noise, one rng
                seeded --init-seed over val_core in order); per 50 ms step the start knot is the previous output
                under a Gaussian state prior (5 mm, 0.05 rad, 0.05 rad), the end knot starts there, the 50 ms
                increment prior and (--w-abs) a weak absolute finger-pose prior fitted on the training GT apply.
                An event-free step holds.
    fused       S39's loop exactly as tools/xvm/pi_fuse.py control (W200 LNES clipped to the run start, ramp1000
                AdaptiveFilter, Count50, protocol noise rng 0), then the SBA posterior with the start knot at the
                fed-back state (state prior), the end knot anchored on S39's output (0.02 m, 0.10 rad, 0.15 rad,
                fixed a priori) and the increment prior; the result is fed back. --control: S39 alone (must reproduce
                8.911 / 8.527 / 8.690).

Outputs per arm: outputs/sba/p3/<arm>.npz (per-step metrics per sequence, the format tools/sba/make_row.py reads)
and <arm>.json (summary).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python tools/sba/p3_loop.py standalone --init-seed 0
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 python tools/sba/p3_loop.py fused --seed 3407
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
sys.path[:0] = [str(REPO / "tools"), str(REPO / "tools" / "dt"), str(REPO), str(REPO / "model")]
from sba import common as CM  # noqa: E402
from sba.common import SB, EF, ET  # noqa: E402

OUT = CM.OUT / "p3"
KEYS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm", "root_rot_deg", "transl_mm")


def sba_step(ctx, model, sd, prev_end, end, x_prev, end_anchor=None, pose=None):
    win = CM.window(ctx, sd, prev_end, end, x_prev, x_prev if end_anchor is None else end_anchor)
    init_end = x_prev if end_anchor is None else end_anchor
    if win.n_events == 0:
        return init_end, win.n_events
    ev = SB.Evaluator(ctx.geo, win, model, sd.betas, sd.K)
    pri = SB.Priors(inc_prec=ctx.inc_prec(end - prev_end), start_mean=x_prev,
                    start_prec=CM.sig_prec(CM.STATE_SIG, ctx.dev))
    if end_anchor is not None:
        pri.end_mean, pri.end_prec = end_anchor, CM.sig_prec(CM.S39_SIG, ctx.dev)
    if pose is not None:
        pri.pose_mean, pri.pose_prec = pose
    return SB.solve(ev, x_prev, init_end, pri, pools=CM.POOLS_TRACK).x1, win.n_events


def finish(rec_all, mano32, dev, EX, name, meta):
    summ, arrays = {}, {}
    for s, rec in rec_all.items():
        r = {k: np.asarray(v) for k, v in rec.items()}
        r["pred"], r["gt"], r["prev"] = r["pred"].astype(np.float32), r["gt"].astype(np.float32), r["prev"].astype(np.float32)
        mm = EX.per_step_metrics(mano32, r, dev)
        summ[s] = {"n": int(len(r["end"])), **{k: float(v.mean()) for k, v in mm.items()}}
        for k, v in mm.items():
            arrays[f"{s}|{k}"] = np.asarray(v)
        arrays[f"{s}|pred"] = r["pred"]
        arrays[f"{s}|end"] = r["end"]
    n = sum(summ[s]["n"] for s in CM.ZGZ)
    summ["overall"] = {k: sum(summ[s][k] * summ[s]["n"] for s in CM.ZGZ) / n for k in KEYS}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", **arrays)
    (OUT / f"{name}.json").write_text(json.dumps({**meta, "summary": summ}, indent=1))
    CM.receipt(OUT / name, {"probe": "P3", "arm": name}, scripts=(__file__,))
    print(name, json.dumps({k: round(summ[k]["mpjpe_ra_mm"], 3) for k in ("zgz_local", "zgz_global", "overall")}),
          flush=True)


@torch.no_grad()
def standalone(a, dev):
    import filter_eval as FE
    ctx = CM.Ctx(dev)
    model = CM.load_model()
    if json.loads(CM.CALIB.read_text()).get("kappa_mode") == "profiled":
        model.profile = True
    mu, prec = CM.finger_prior(dev)
    pose = (mu, prec * a.w_abs) if a.w_abs > 0 else None
    rng = np.random.default_rng(a.init_seed)
    rec_all = {}
    t0 = time.time()
    for s, d in EF.split_seqs("val_core"):
        sd = CM.SeqData(s, d, dev)
        rec = {k: [] for k in ("pred", "prev", "gt", "end", "run", "count")}
        for run_id, (ra, rb) in enumerate(sd.runs):
            ends = np.arange(ra + EF.STEP - 1, rb, EF.STEP, dtype=np.int64)
            if not len(ends):
                continue
            x = torch.tensor(np.asarray(sd.pos51[ra], np.float32) + ET.sample_init_noise(CM.INIT_CFG, rng, 1.0),
                             device=dev)
            prev_end = int(ra)
            for end in ends:
                end = int(end)
                rec["prev"].append(x.cpu().numpy())
                x, n_ev = sba_step(ctx, model, sd, prev_end, end, x, pose=pose)
                prev_end = end
                rec["pred"].append(x.cpu().numpy()); rec["gt"].append(np.asarray(sd.pos51[end]))
                rec["end"].append(end); rec["run"].append(run_id); rec["count"].append(n_ev)
            print(s, run_id, len(rec["end"]), f"{time.time() - t0:.0f} s", flush=True)
        rec["betas"] = sd.sq.betas.astype(np.float32)
        rec_all[s] = rec
    finish(rec_all, ctx.mano, dev, FE.EX, f"standalone_w{a.w_abs:g}_i{a.init_seed}",
           {"arm": "standalone", "init_seed": a.init_seed, "w_abs": a.w_abs, "model": model.__dict__,
            "wall_s": time.time() - t0})


@torch.no_grad()
def fused(a, dev):
    import filter_eval as FE
    from semkine import events as EV
    from semkine.anchored import AdaptiveFilter
    from semkine.async_tracker import _Fixed
    from mano_layer import ManoLayer
    from xvm.p0d_gain import SPEC, WIN
    EX = FE.EX
    ctx = CM.Ctx(dev)
    model = CM.load_model()
    if json.loads(CM.CALIB.read_text()).get("kappa_mode") == "profiled":
        model.profile = True
    cfg, net, ckpt, step, _ = FE.build_run(REPO / f"outputs/semkine/dt_dz_l3_s{a.seed}", "last", dev)
    filt = AdaptiveFilter.from_spec(_Fixed(), json.loads(SPEC.read_text())).to(dev).eval()
    mano32 = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(dev).eval()
    ev_ch = EV.event_channels(cfg)
    rng = np.random.default_rng(0)
    rec_all = {}
    t0 = time.time()
    for s, d in EF.split_seqs("val_core"):
        events, offsets, aux, pos51 = ET.load_sequence(EF.DATA_ROOT, d, s)
        sd = CM.SeqData(s, d, dev)
        betas = torch.tensor(aux["betas"], dtype=torch.float32, device=dev).view(1, -1)
        camera_K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=dev).view(1, 3, 3)
        net.set_hand_context(betas, camera_K)
        runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
        rec = {k: [] for k in ("pred", "prev", "gt", "end", "run", "count")}
        for run_id, (ra, rb) in enumerate(runs):
            ends = np.arange(ra + EF.STEP - 1, rb, EF.STEP, dtype=np.int64)
            if not len(ends):
                continue
            prev = pos51[ra].copy() + ET.sample_init_noise(cfg, rng, 1.0)
            prev_t = torch.from_numpy(prev).view(1, -1).to(dev)
            filt.reset_state(prev_t)
            prev_end = int(ra)
            for end in ends:
                end = int(end)
                n_ev = int(offsets[end + 1] - offsets[end - EF.STEP + 1])
                w = EX.window_of(offsets, end, "fixed", WIN, 0, 300, floor=int(ra))
                x = torch.from_numpy(ET.build_lnes(events, offsets, end, w, ev_ch)).unsqueeze(0).to(dev)
                rec["prev"].append(prev_t.cpu().numpy()[0])
                z = net(x, prev_t)
                filt.trk_model.z = z
                out = filt(x, prev_t, n_events=n_ev)
                if not a.control:
                    xf, _ = sba_step(ctx, model, sd, prev_end, end, prev_t[0].float(), end_anchor=out[0].float())
                    out = xf.view(1, -1).to(out.dtype)
                prev_t = out
                prev_end = end
                rec["pred"].append(out.cpu().numpy()[0]); rec["gt"].append(pos51[end])
                rec["end"].append(end); rec["run"].append(run_id); rec["count"].append(n_ev)
            if run_id % 10 == 0:
                print(s, run_id, len(rec["end"]), f"{time.time() - t0:.0f} s", flush=True)
        rec["betas"] = aux["betas"]
        rec_all[s] = rec
    name = f"{'control' if a.control else 'fused'}_s{a.seed}"
    finish(rec_all, mano32, dev, EX, name, {"arm": name, "seed": a.seed, "ckpt": str(ckpt), "step": int(step),
                                             "model": model.__dict__, "wall_s": time.time() - t0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=("standalone", "fused"))
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--w-abs", type=float, default=CM.W_ABS)
    ap.add_argument("--seed", type=int, choices=(3407, 3408, 3409), default=3407)
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args()
    dev = CM.require_gpu()
    torch.set_grad_enabled(False)
    if a.arm == "standalone":
        standalone(a, dev)
    else:
        fused(a, dev)


if __name__ == "__main__":
    main()
