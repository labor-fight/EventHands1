#!/usr/bin/env python3
"""XVM PI (post hoc, docs/EXPERIMENT_RECORDS.md#xvm-pi-addendum-20261021): S39 stack + XVM measurement as an
information-form correction, zero training.

Loop = tools/xvm/p0d_gain.py control (bit-identical when the correction is off). Per step, after S39's filter gives
x_S39, the frozen P4 head measures (m_i, Lambda_i) on the visible vertices of the incoming state `prev`; with the
projected Jacobian B at prev,

    x = x_S39 + (sum_i B_i^T Lt_i B_i + lam_S P)^-1 sum_i B_i^T Lt_i (mt_i - B_i (x_S39 - prev))

    python tools/xvm/pi_fuse.py --seed 3407 --out outputs/xvm/s3/pi/s3407.json
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
from xvm import common as C  # noqa: E402
from xvm import p4_train as T  # noqa: E402
from xvm.p4_data import P4Data, collate  # noqa: E402
from xvm.p5_solve import load_head  # noqa: E402
from xvm.p0d_gain import EX, FE, SPEC, WIN  # noqa: E402
from s44 import evflow as EF  # noqa: E402
from mano_layer import ManoLayer  # noqa: E402
from semkine import eval_track as ET  # noqa: E402
from semkine import events as EV  # noqa: E402
from semkine import xvm as XV  # noqa: E402
from semkine.anchored import AdaptiveFilter  # noqa: E402
from semkine.async_tracker import _Fixed  # noqa: E402

LAMS = (1.0, 4.0, 16.0)
HEAD_RUN, HEAD = REPO / "outputs/xvm/s2/p4/A_m600", "patch_r2"


@torch.no_grad()
def loop(model, filt, cfg, dev, mano32, lam, ctx):
    head, geo, P, cache = ctx
    ev_ch = EV.event_channels(cfg)
    rng = np.random.default_rng(0)
    grng = np.random.default_rng(12345)
    summ = {}
    for s, d in EF.split_seqs("val_core"):
        events, offsets, aux, pos51 = cache[s]["raw"]
        ds = cache[s]["ds"]
        sq = ds.seqs[0]
        intr = sq.intrinsics()
        betas64 = torch.tensor(sq.betas, dtype=torch.float64, device=dev)[None]
        betas = torch.tensor(aux["betas"], dtype=torch.float32, device=dev).view(1, -1)
        camera_K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=dev).view(1, 3, 3)
        model.set_hand_context(betas, camera_K)
        runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
        rec = {k: [] for k in ("pred", "prev", "gt", "end", "run", "elapsed", "count", "win")}
        i = 0
        for run_id, (ra, rb) in enumerate(runs):
            ends = np.arange(ra + EF.STEP - 1, rb, EF.STEP, dtype=np.int64)
            if not len(ends):
                continue
            prev = pos51[ra].copy() + ET.sample_init_noise(cfg, rng, 1.0)
            prev_t = torch.from_numpy(prev).view(1, -1).to(dev)
            filt.reset_state(prev_t)
            for end in ends:
                end = int(end)
                assert ds.index[i][2] == end
                n_ev = int(offsets[end + 1] - offsets[end - EF.STEP + 1])
                w = EX.window_of(offsets, end, "fixed", WIN, 0, 300, floor=int(ra))
                x = torch.from_numpy(ET.build_lnes(events, offsets, end, w, ev_ch)).unsqueeze(0).to(dev)
                rec["prev"].append(prev_t.cpu().numpy()[0])
                z = model(x, prev_t)
                filt.trk_model.z = z
                out = filt(x, prev_t, n_events=n_ev)
                if lam is not None and n_ev > 0:
                    item = collate([ds[i]])
                    item["x0"] = prev_t.double().cpu()
                    g = geo(item, "A", grng)
                    m, L, rho, _ = head(g) if isinstance(head, T.KernelHead) else head(g, item)
                    mt, Lt = XV.to_image(m.float(), L.float(), g["R"].float(), rho)
                    _, _, _, _, J2, _ = C.fd_jacobian(geo.mano, prev_t.double(), betas64, intr)
                    vid = g["vis"][0].nonzero().flatten()
                    Bv = J2[0, vid]
                    W = Lt[0, vid].double()
                    pred_disp = (Bv @ (out.double() - prev_t.double())[0].view(1, 51, 1)).squeeze(-1)
                    r = mt[0, vid].double() - pred_disp
                    H = torch.einsum("nka,nkl,nlb->ab", Bv, W, Bv) + lam * P
                    gg = torch.einsum("nka,nkl,nl->a", Bv, W, r)
                    out = out + torch.linalg.solve(H, gg).to(out.dtype).view(1, -1)
                prev_t = out
                i += 1
                rec["pred"].append(out.cpu().numpy()[0])
                rec["gt"].append(pos51[end])
                rec["end"].append(end)
                rec["run"].append(run_id)
                rec["elapsed"].append(end - int(ra))
                rec["count"].append(n_ev)
                rec["win"].append(w)
        r = {k: np.asarray(v) for k, v in rec.items()}
        r["pred"], r["gt"], r["prev"] = r["pred"].astype(np.float32), r["gt"].astype(np.float32), r["prev"].astype(np.float32)
        r["betas"] = aux["betas"]
        mm = EX.per_step_metrics(mano32, r, dev)
        summ[s] = {"n": int(len(r["end"])), **{k: float(v.mean()) for k, v in mm.items()}}
    n = sum(v["n"] for v in summ.values())
    summ["overall"] = {k: sum(summ[s][k] * summ[s]["n"] for s in C.ZGZ) / n
                       for k in ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "root_rot_deg", "transl_mm")}
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=(3407, 3408, 3409))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = C.require_gpu()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; preserve it")
    cfg, model, ckpt, step, _ = FE.build_run(REPO / f"outputs/semkine/dt_dz_l3_s{a.seed}", "last", dev)
    filt = AdaptiveFilter.from_spec(_Fixed(), json.loads(SPEC.read_text())).to(dev).eval()
    mano32 = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(dev).eval()
    head, mem = load_head(HEAD_RUN, HEAD, dev)
    geo = T.Geo(dev)
    P = C.increment_precision(dev)
    cache = {s: {"raw": ET.load_sequence(EF.DATA_ROOT, d, s), "ds": P4Data([(s, d)], mem, e3=True, with_sae=True)}
             for s, d in EF.split_seqs("val_core")}
    res = {"seed": a.seed, "ckpt": str(ckpt), "head": f"{HEAD_RUN}:{HEAD}", "runs": {}}
    t0 = time.time()
    for lam in (None, *LAMS):
        key = "control" if lam is None else f"lam{lam:g}"
        res["runs"][key] = loop(model, filt, cfg, dev, mano32, lam, (head, geo, P, cache))
        o = res["runs"][key]
        print(a.seed, key, f"local {o['zgz_local']['mpjpe_ra_mm']:.3f} global {o['zgz_global']['mpjpe_ra_mm']:.3f} "
              f"RA {o['overall']['mpjpe_ra_mm']:.3f} transl {o['overall']['transl_mm']:.1f} ({time.time() - t0:.0f} s)",
              flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    C.receipt(out.parent, {"probe": "pi"}, scripts=(__file__,))


if __name__ == "__main__":
    main()
