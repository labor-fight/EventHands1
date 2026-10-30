#!/usr/bin/env python3
"""SBA round 2 ladder (docs/EXPERIMENT_RECORDS.md#sba-p4-prereg-20261022): P1', P2', P3' with the P4 emission model and
nuisance-orthogonal inference (semkine/sba_emission.py solve_orth: per-window part-group emission multipliers and
outlier rate profiled at every iteration, the scale gauge log sigma a state dimension with a 0.05 prior).

Same packets, priors, schedules and scoring as round 1 (tools/sba/p1_e3.py, p2_xdi.py, p3_loop.py); only the
observation model and the inference change. Outputs under outputs/sba/{p1o,p2o,p3o}/.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python tools/sba/orth_ladder.py p1 --seq zgz_local --h 50
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python tools/sba/orth_ladder.py p2 --seq zgz_local --shard 0 --shards 2
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python tools/sba/orth_ladder.py p3 --arm standalone --init-seed 0
    python tools/sba/orth_ladder.py summarize --stage p1
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
from sba import p1_e3 as P1, p2_xdi as P2, p3_loop as P3  # noqa: E402
from semkine import sba_emission as EM  # noqa: E402

P4 = CM.OUT / "p4"
P1.OUT, P2.OUT, P3.OUT = CM.OUT / "p1o", CM.OUT / "p2o", CM.OUT / "p3o"


class Orth:
    def __init__(self, ctx):
        fit = json.loads((P4 / "fit_raw_b1.json").read_text())
        self.ctx = ctx
        self.patch = torch.from_numpy(np.load(P4 / "patches.npy")).to(ctx.dev)
        self.n_p = int(fit["meta"]["n_patches"])
        self.theta = torch.tensor(fit["selected"]["theta"], dtype=torch.float32, device=ctx.dev)
        self.blur = float(fit["meta"]["blur"])
        self.group_of = EM.chan_groups(ctx.geo, self.patch, self.n_p)

    def evaluator(self, sd, win):
        return EM.OrthEvaluator(self.ctx.geo, win, sd.betas, sd.K, self.theta, self.group_of, self.patch, self.n_p,
                                self.blur)


def p1(a, dev):
    ctx = CM.Ctx(dev)
    om = Orth(ctx)
    sd = CM.SeqData(a.seq, "val", dev)
    rows = CM.endpoint_rows(sd)
    idx = list(range(len(rows)))[a.shard::a.shards]
    out = {k: [] for k in ("i", "start", "end", "x", "n_events", "accepted", "sign", "s", "sigma")}
    t0 = time.time()
    for i in idx:
        rid, end, prev, first = rows[i]
        start = prev if a.h == 50 else max(sd.run_start(end), end - a.h)
        x0 = sd.gt(start)
        win = CM.window(ctx, sd, start, end, x0, x0)
        ts = time.time()
        if win.n_events == 0:
            x, acc, sg = x0, 0, 1.0
        else:
            r = EM.solve_orth(om.evaluator(sd, win), x0, x0, SB.Priors(inc_prec=ctx.inc_prec(end - start)),
                              pools=CM.POOLS_TRACK)
            x, acc, sg = r.x1, r.accepted, r.sigma
        out["i"].append(i); out["start"].append(start); out["end"].append(end); out["x"].append(x.cpu().numpy())
        out["n_events"].append(win.n_events); out["accepted"].append(acc); out["sign"].append(1)
        out["s"].append(time.time() - ts); out["sigma"].append(sg)
        if len(out["i"]) % 100 == 0:
            print(json.dumps({"seq": a.seq, "h": a.h, "done": len(out["i"]), "of": len(idx),
                              "s": round(time.time() - t0)}), flush=True)
    path = P1.OUT / "raw" / f"{a.seq}_h{a.h}__{a.shard:02d}of{a.shards:02d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in out.items()})
    CM.receipt(P1.OUT / "raw" / f"{a.seq}_h{a.h}__{a.shard:02d}", {"probe": "P1'", "wall_s": time.time() - t0},
               scripts=(__file__, REPO / "semkine" / "sba_emission.py"))


def p2(a, dev):
    ctx = CM.Ctx(dev)
    om = Orth(ctx)
    sd = CM.SeqData(a.seq, "val", dev)
    rows = CM.bank_rows(CM.bank(), a.seq)
    idx = list(range(len(rows)))[a.shard::a.shards]
    prec0 = CM.sig_prec(CM.XDI_SIG, dev)
    rec = {k: [] for k in ("i", "x_in", "x_out", "gt_prev", "gt_next", "wrong_in", "wrong_out", "n_events", "has_donor")}
    t0 = time.time()

    def step(win, xin):
        if win.n_events == 0:
            return xin
        pri = SB.Priors(inc_prec=ctx.inc_prec(50), start_mean=xin, start_prec=prec0)
        return EM.solve_orth(om.evaluator(sd, win), xin, xin, pri, pools=CM.POOLS_WIDE).x1

    for i in idx:
        row = rows[i]
        prev, end = int(row["previous_end_ms"]), int(row["end_ms"])
        gp, gn = sd.gt(prev), sd.gt(end)
        xs, outs, n_ev = [], [], 0
        for name in P2.NAMES:
            kind, val = CM.PERTURB[name]
            xin = CM.perturb(ctx, gp, kind, val, i % 15, sd.betas)
            win = CM.window(ctx, sd, prev, end, xin, xin)
            n_ev = win.n_events
            xs.append(xin.cpu().numpy()); outs.append(step(win, xin).cpu().numpy())
        dr = CM.donor_row(ctx, sd, rows, i, sd.betas)
        wi, wo = [], []
        for name in P2.WRONG:
            kind, val = CM.PERTURB[name]
            xin = CM.perturb(ctx, gp, kind, val, i % 15, sd.betas)
            if dr is None:
                wi.append(xin.cpu().numpy()); wo.append(xin.cpu().numpy())
                continue
            evms = CM.retime(sd.events(int(dr["previous_end_ms"]), int(dr["end_ms"])), int(dr["previous_end_ms"]),
                             prev, end)
            win = CM.window(ctx, sd, prev, end, xin, xin, events=evms)
            wi.append(xin.cpu().numpy()); wo.append(step(win, xin).cpu().numpy())
        rec["i"].append(i); rec["x_in"].append(np.stack(xs)); rec["x_out"].append(np.stack(outs))
        rec["gt_prev"].append(gp.cpu().numpy()); rec["gt_next"].append(gn.cpu().numpy())
        rec["wrong_in"].append(np.stack(wi)); rec["wrong_out"].append(np.stack(wo))
        rec["n_events"].append(n_ev); rec["has_donor"].append(dr is not None)
        if len(rec["i"]) % 25 == 0:
            print(json.dumps({"seq": a.seq, "shard": a.shard, "done": len(rec["i"]), "of": len(idx),
                              "s": round(time.time() - t0)}), flush=True)
    path = P2.OUT / "raw" / f"{a.seq}__{a.shard:02d}of{a.shards:02d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in rec.items()}, names=np.array(P2.NAMES))
    CM.receipt(P2.OUT / "raw" / f"{a.seq}__{a.shard:02d}", {"probe": "P2'", "wall_s": time.time() - t0},
               scripts=(__file__, REPO / "semkine" / "sba_emission.py"))


def orth_step(ctx, om, sd, prev_end, end, x_prev, sigma, end_anchor=None, pose=None):
    win = CM.window(ctx, sd, prev_end, end, x_prev, x_prev if end_anchor is None else end_anchor)
    init_end = x_prev if end_anchor is None else end_anchor
    if win.n_events == 0:
        return init_end, sigma, 0
    pri = SB.Priors(inc_prec=ctx.inc_prec(end - prev_end), start_mean=x_prev,
                    start_prec=CM.sig_prec(CM.STATE_SIG, ctx.dev))
    if end_anchor is not None:
        pri.end_mean, pri.end_prec = end_anchor, CM.sig_prec(CM.S39_SIG, ctx.dev)
    if pose is not None:
        pri.pose_mean, pri.pose_prec = pose
    r = EM.solve_orth(om.evaluator(sd, win), x_prev, init_end, pri, sigma0=sigma, pools=CM.POOLS_TRACK)
    return r.x1, r.sigma, win.n_events


@torch.no_grad()
def p3(a, dev):
    import filter_eval as FE
    ctx = CM.Ctx(dev)
    om = Orth(ctx)
    t0 = time.time()
    rec_all = {}
    if a.arm == "standalone":
        mu, prec = CM.finger_prior(dev)
        pose = (mu, prec * a.w_abs) if a.w_abs > 0 else None
        rng = np.random.default_rng(a.init_seed)
        for s, d in EF.split_seqs("val_core"):
            sd = CM.SeqData(s, d, dev)
            rec = {k: [] for k in ("pred", "prev", "gt", "end", "run", "count", "sigma")}
            for run_id, (ra, rb) in enumerate(sd.runs):
                ends = np.arange(ra + EF.STEP - 1, rb, EF.STEP, dtype=np.int64)
                if not len(ends):
                    continue
                x = torch.tensor(np.asarray(sd.pos51[ra], np.float32) + ET.sample_init_noise(CM.INIT_CFG, rng, 1.0),
                                 device=dev)
                sig, prev_end = 1.0, int(ra)
                for end in ends:
                    end = int(end)
                    rec["prev"].append(x.cpu().numpy())
                    x, sig, n_ev = orth_step(ctx, om, sd, prev_end, end, x, sig, pose=pose)
                    prev_end = end
                    rec["pred"].append(x.cpu().numpy()); rec["gt"].append(np.asarray(sd.pos51[end]))
                    rec["end"].append(end); rec["run"].append(run_id); rec["count"].append(n_ev)
                    rec["sigma"].append(sig)
                print(s, run_id, len(rec["end"]), f"{time.time() - t0:.0f} s", flush=True)
            rec["betas"] = sd.sq.betas.astype(np.float32)
            rec_all[s] = {k: v for k, v in rec.items() if k != "sigma"}
            rec_all[s]["sigma_median"] = float(np.median(rec["sigma"]))
        sig_med = {s: rec_all[s].pop("sigma_median") for s in rec_all}
        P3.finish(rec_all, ctx.mano, dev, FE.EX, f"standalone_w{a.w_abs:g}_i{a.init_seed}",
                  {"arm": "standalone-orth", "init_seed": a.init_seed, "w_abs": a.w_abs, "sigma_median": sig_med,
                   "wall_s": time.time() - t0})
        return
    from semkine import events as EV
    from semkine.anchored import AdaptiveFilter
    from semkine.async_tracker import _Fixed
    from mano_layer import ManoLayer
    from xvm.p0d_gain import SPEC, WIN
    EX = FE.EX
    cfg, net, ckpt, step, _ = FE.build_run(REPO / f"outputs/semkine/dt_dz_l3_s{a.seed}", "last", dev)
    filt = AdaptiveFilter.from_spec(_Fixed(), json.loads(SPEC.read_text())).to(dev).eval()
    mano32 = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(dev).eval()
    ev_ch = EV.event_channels(cfg)
    rng = np.random.default_rng(0)
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
            sig, prev_end = 1.0, int(ra)
            for end in ends:
                end = int(end)
                n_ev = int(offsets[end + 1] - offsets[end - EF.STEP + 1])
                w = EX.window_of(offsets, end, "fixed", WIN, 0, 300, floor=int(ra))
                x = torch.from_numpy(ET.build_lnes(events, offsets, end, w, ev_ch)).unsqueeze(0).to(dev)
                rec["prev"].append(prev_t.cpu().numpy()[0])
                z = net(x, prev_t)
                filt.trk_model.z = z
                out = filt(x, prev_t, n_events=n_ev)
                xf, sig, _ = orth_step(ctx, om, sd, prev_end, end, prev_t[0].float(), sig, end_anchor=out[0].float())
                out = xf.view(1, -1).to(out.dtype)
                prev_t = out
                prev_end = end
                rec["pred"].append(out.cpu().numpy()[0]); rec["gt"].append(pos51[end])
                rec["end"].append(end); rec["run"].append(run_id); rec["count"].append(n_ev)
        rec["betas"] = aux["betas"]
        rec_all[s] = rec
    P3.finish(rec_all, mano32, dev, EX, f"fused_s{a.seed}", {"arm": "fused-orth", "seed": a.seed, "ckpt": str(ckpt),
                                                              "wall_s": time.time() - t0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("p1", "p2", "p3", "summarize"))
    ap.add_argument("--seq", choices=CM.ZGZ)
    ap.add_argument("--h", type=int, default=50)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--arm", choices=("standalone", "fused"), default="standalone")
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--w-abs", type=float, default=CM.W_ABS)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--of", choices=("p1", "p2"), default="p1")
    a = ap.parse_args()
    if a.stage == "summarize":
        (P1.summarize if a.of == "p1" else P2.summarize)()
        return
    dev = CM.require_gpu()
    torch.set_grad_enabled(False)
    {"p1": p1, "p2": p2, "p3": p3}[a.stage](a, dev)


if __name__ == "__main__":
    main()
