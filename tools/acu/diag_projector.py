#!/usr/bin/env python3
"""ACU D5 (docs/EXPERIMENT_RECORDS.md#acu-diag-20261104): would blocking the learned update in directions the packet does
not constrain help, before any retraining? Zero training, CPU, read-only on existing E3 readings (single step from the GT
previous state; the closed loop needs a rerun and is not measured here).

    python -m tools.acu.diag_projector run --seq zgz_local --part 0 --nparts 4
    python -m tools.acu.diag_projector table

Per packet (every 4th zgz endpoint), whitened information Hw = S H S of point-to-point correspondences (sig 3 px), H from
  lab   XAG label points (exact; each event on the GT mesh at its own ms)
  rt    r_s2d's routed points on the same meshes (what a deployed geometry would see; only the Jacobians enter, not the
        residuals)
The network's own step d = f(prev) - prev (51-D) is replaced by S W S^-1 d with the projector W = (Hw + lam I)^-1 Hw
(~identity where the packet constrains a direction, ~0 where it does not; no learned parameter). Scores: E3 MPJPE-RA of
prev + d and of prev + W d against the target, per recording, for lam in LAMS.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.acu.diag_update import NETS, READ_ROOTS  # noqa: E402

OUT = ROOT / "outputs/acu/20261104/diag/d5"
STRIDE = 4
SIG = 3.0
SCALE = np.array([0.01] * 3 + [0.1] * 48)
LAMS = (0.3, 1.0, 3.0)
READ = ("B2 3407", "B2 3408", "B2-gt label", "B2 3407 +label test", "B2-gt r_s2d", "S43 3407")


def run(seq, part, nparts):
    import torch
    from semkine import eval_track as ET
    from semkine.xyz_mesh import stride_sample
    from tools.acu import diag_interface as D
    from tools.pfo import engine as E
    from tools.pfo.nets import sequence
    cfg, mano, SM = D.setup()
    dev = torch.device("cpu")
    net = SM._net(dev)
    native = SM._NATIVE["sample_sources"]
    root, seqs = E.sequences(cfg, (seq,))
    ctx = sequence(root, seqs[0][1], seq, dev)
    rows = E.endpoint_rows(ctx.aux)
    betas = ctx.betas.double()
    K = ctx.K[0].double()
    s = 0.375
    fx, fy = float(K[0, 0] * s), float(K[1, 1] * s)
    faces = mano.f.long()
    jacV = torch.func.vmap(torch.func.jacfwd(lambda p: D.fk(mano, p.view(1, 51), betas)[0][0]))
    e3 = {}
    for name in READ:
        r, tag = NETS[name]
        p = READ_ROOTS[r] / tag / "e3" / "states.npz"
        if p.exists():
            z = np.load(p)
            e3[name] = {k: z[f"{seq}/{k}"] for k in ("prediction", "gt_previous", "target", "end_ms")}
    sel = list(range(0, len(rows), STRIDE))[part::nparts]
    recs = []
    with torch.no_grad():
        for i in sel:
            _, end, prev_ms, _ = rows[i]
            ev5 = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, int(end), 50)
            if len(ev5) < 50:
                continue
            b = ET.make_eval_packet(ev5, torch.zeros(1, 51), ctx.betas, ctx.K, 50, dev)
            b = ET.attach_xag_sources(b, cfg, seq, ctx.offsets, int(end), 50, dev)
            src, mask = stride_sample(b.ptr, 2048)
            _, routed, rface, rbary, _ = SM.route_nodes(net, SM._STATE["router"], b.events, b.ptr, b.delta_t_s, 0.5)
            lab = native(b, src, mask)
            m = mask[0]
            F_ = faces.shape[0]
            lv = lab["xag_valid"][0][m] & (lab["xag_face"][0][m] >= 0) & (lab["xag_face"][0][m] < F_)
            lb = lab["xag_bary"][0][m].double()
            lv = lv & torch.isfinite(lb).all(-1) & (lb >= 0).all(-1) & (lb.sum(-1) > 0)
            lb = lb / lb.sum(-1, keepdim=True).clamp_min(1e-12)
            lf = lab["xag_face"][0][m].clamp(0, F_ - 1).long()
            rv = routed[0][m]
            rf, rb = rface[0][m].long(), rbary[0][m].double()
            if lv.sum() < 20 or rv.sum() < 20:
                continue
            start = torch.tensor(np.asarray(ctx.gt[int(prev_ms)], np.float64))
            JV = jacV(start.view(1, 51))[0]
            V, _ = D.fk(mano, start.view(1, 51), betas)
            Hs = {}
            for src_name, use, fsel, bsel in (("lab", lv, lf, lb), ("rt", rv, rf, rb)):
                idx = use.nonzero().flatten()
                tri = faces[fsel[idx]]
                X = (bsel[idx].unsqueeze(-1) * V[0][tri]).sum(1)
                dX = (bsel[idx][:, :, None, None] * JV[tri]).sum(1)
                J = D.proj_jac(X, fx, fy) @ dX
                H = torch.einsum("nkp,nkq->pq", J, J).numpy() / SIG ** 2
                Hs[src_name] = SCALE[:, None] * H * SCALE[None, :]
            rec = {"end": int(end)}
            for name, z in e3.items():
                k = np.nonzero(z["end_ms"] == int(end))[0]
                if not len(k):
                    continue
                k = int(k[0])
                prev = z["gt_previous"][k].astype(np.float64)
                d = z["prediction"][k].astype(np.float64) - prev
                rec[name] = {"prev": prev.tolist(), "target": z["target"][k].tolist(), "d": d.tolist()}
                for src_name, Hw in Hs.items():
                    for lam in LAMS:
                        W = np.linalg.solve(Hw + lam * np.eye(51), Hw)
                        dp = SCALE * (W @ (d / SCALE))
                        rec[name][f"{src_name}|{lam:g}"] = dp.tolist()
            recs.append(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{seq}_p{part}of{nparts}.json").write_text(json.dumps(recs))
    print(seq, part, len(recs), flush=True)


def table():
    import torch
    from tools.acu.diag_update import context
    from tools.pfo.engine import score
    c = context()
    agg = {}
    for f in sorted(OUT.glob("*_p*of*.json")):
        agg.setdefault(f.name.split("_p")[0], []).extend(json.loads(f.read_text()))
    L = ["# ACU D5: the learned E3 step with its unconstrained directions removed (test-time, zero training)", "",
         "E3 MPJPE-RA (mm, mean over every 4th zgz endpoint): the network's own step vs the step projected by "
         "W = (Hw + lam I)^-1 Hw, Hw from label points (lab) or from r_s2d's routed points (rt); identity = no step.", ""]
    for seq, recs in agg.items():
        cols = [f"{s}|{lam:g}" for s in ("lab", "rt") for lam in LAMS]
        shr = (0.25, 0.5, 0.75)
        L += [f"## {seq} ({len(recs)} packets)", "", "| network | own step | identity | " + " | ".join(cols) + " | "
              + " | ".join(f"x{c:g}" for c in shr) + " | mean |Wd|/|d| (lab, lam 1) |",
              "|---|---|---|" + "---|" * (len(cols) + len(shr) + 1)]
        for name in READ:
            v = [r[name] for r in recs if name in r]
            if not v:
                continue
            prev = np.array([x["prev"] for x in v], np.float32)
            tgt = np.array([x["target"] for x in v], np.float32)
            d = np.array([x["d"] for x in v])
            sc = lambda st: float(score(c["mano"], st.astype(np.float32), tgt, c["betas"][seq], torch.device("cpu"))["mpjpe_ra_mm"].mean())  # noqa: E731
            row = [sc(prev + d), sc(prev)]
            for col in cols:
                row.append(sc(prev + np.array([x[col] for x in v])))
            for c_ in shr:
                row.append(sc(prev + c_ * d))
            wd = np.array([x["lab|1"] for x in v])
            ratio = float(np.mean(np.linalg.norm(wd / SCALE, axis=1) / np.maximum(np.linalg.norm(d / SCALE, axis=1), 1e-12)))
            L.append(f"| {name} | " + " | ".join(f"{x:.2f}" for x in row) + f" | {ratio:.2f} |")
        L.append("")
    (OUT.parent / "d5.md").write_text("\n".join(L))
    print("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seq", required=True)
    r.add_argument("--part", type=int, default=0)
    r.add_argument("--nparts", type=int, default=1)
    sub.add_parser("table")
    a = p.parse_args(argv)
    if a.verb == "run":
        if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
            raise SystemExit("CPU only: set CUDA_VISIBLE_DEVICES=''")
        run(a.seq, a.part, a.nparts)
    else:
        table()


if __name__ == "__main__":
    main()
