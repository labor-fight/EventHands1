#!/usr/bin/env python3
"""XVM P5: learned per-vertex measurement + fixed kinematic transport, no further training
(docs/EXPERIMENT_RECORDS.md#xvm-prereg-20261021 §3).

The P4-selected kernel (trained on the seven training subjects, arm A) measures (m_i, Lambda_i) on every visible vertex
of the anchor mesh; the information-weighted Gauss-Newton of P2 (3 iterations, 51-D additive, prior lam * P with the
50 ms increment precision) turns them into the state. Modes: e3 (anchor = GT previous endpoint, the gate) and e1 (own
closed loop from the XCL A0FOmega-o E1 initial states; descriptive). `--obs-only`: only vertices with an event of the
last 50 ms within 2 cm (diagnostic).

    python tools/xvm/p5_solve.py --run outputs/xvm/s2/p4/A_m200 --head xvm_r2 --mode e3 --out outputs/xvm/s3/p5/e3.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xvm import common as C  # noqa: E402
from xvm.common import EF  # noqa: E402
from xvm.p4_data import P4Data, collate  # noqa: E402
from xvm import p4_train as T  # noqa: E402
from semkine import xvm as XV  # noqa: E402

E1_TRACE = C.REPO / "outputs/xcl/20261019/evaluations/xcl_a0foo_s3407/e1_step6000_repeat1_none/states.npz"


def load_head(run: Path, head: str, dev):
    ck = torch.load(run / "heads.pt", map_location=dev)
    mem = int(ck["args"]["mem"])
    r = float(head.split("_r")[1])
    h = (T.PatchHead(r) if head.startswith("patch") else T.KernelHead(r, mem)).to(dev)
    sd = {k[len(head) + 1:]: v for k, v in ck["heads"].items() if k.startswith(head + ".")}
    h.load_state_dict(sd)
    return h.eval(), mem


def solve(ctx_mano, P, g, b, mt, Lt, wmask, betas, intr):
    """3 GN iterations for sample b. mt (V, 2) px, Lt (V, 2, 2), wmask (V,) bool."""
    x0 = g["x0"][b:b + 1]
    vid = wmask.nonzero().flatten()
    x = x0.clone()
    if len(vid) == 0:
        return x[0]
    v0, dX0, uv0, z0, J20, _ = C.fd_jacobian(ctx_mano, x0, betas, intr)
    m = mt[vid].double()
    W = Lt[vid].double()
    for it in range(C.GN_ITERS):
        if it == 0:
            uv, J2 = uv0, J20
        else:
            _, _, uv, _, J2, _ = C.fd_jacobian(ctx_mano, x, betas, intr)
        Q = torch.eye(2, dtype=torch.float64, device=x.device).expand(len(vid), 2, 2)
        x[0] += C.gn_step(J2[0, vid], uv[0, vid] - uv0[0, vid], Q, m, W, P, C.LAM_P, x[0] - x0[0])
    return x[0]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--mode", choices=("e3", "e1"), required=True)
    ap.add_argument("--obs-only", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = C.require_gpu()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; preserve it")
    head, mem = load_head(Path(a.run), a.head, dev)
    geo = T.Geo(dev)
    P = C.increment_precision(dev)
    m32 = C.mano32(dev)
    res = {"run": a.run, "head": a.head, "mem": mem, "mode": a.mode, "obs_only": a.obs_only, "per_seq": {}}
    t0 = time.time()
    rng = np.random.default_rng(0)
    trace = np.load(E1_TRACE) if a.mode == "e1" else None
    for s, d in C.zgz_seqs():
        ds = P4Data([(s, d)], mem, e3=True, with_sae=a.head.startswith("patch"))
        sq = ds.seqs[0]
        intr = sq.intrinsics()
        betas = torch.tensor(sq.betas, dtype=torch.float64, device=dev)[None]
        X, X0, GE = [], [], []
        if a.mode == "e3":
            loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=12, collate_fn=collate)
            for batch in loader:
                g = geo(batch, "A", rng)
                m, L, rho, _ = head(g) if isinstance(head, T.KernelHead) else head(g, batch)
                mt, Lt = XV.to_image(m.float(), L.float(), g["R"].float(), rho)
                wm = g["vis"] & (g["obs2"] if a.obs_only else torch.ones_like(g["vis"]))
                for b in range(len(m)):
                    X.append(solve(geo.mano, P, g, b, mt[b], Lt[b], wm[b], betas, intr))
                X0.append(g["x0"]); GE.append(batch["gt"].to(dev))
            X, X0, GE = torch.stack(X), torch.cat(X0), torch.cat(GE)
        else:
            inc = trace[f"{s}/incoming"].astype(np.float64)
            rows = C.endpoint_grid(sq.runs)
            x = None
            for i in range(len(ds)):
                item = collate([ds[i]])
                if rows[i][3]:
                    x = torch.as_tensor(inc[i], dtype=torch.float64, device=dev)
                item["x0"] = x[None].cpu()
                g = geo(item, "A", rng)
                m, L, rho, _ = head(g) if isinstance(head, T.KernelHead) else head(g, item)
                mt, Lt = XV.to_image(m.float(), L.float(), g["R"].float(), rho)
                wm = g["vis"] & (g["obs2"] if a.obs_only else torch.ones_like(g["vis"]))
                X0.append(x.clone())
                x = solve(geo.mano, P, g, 0, mt[0], Lt[0], wm[0], betas, intr)
                X.append(x.clone()); GE.append(item["gt"][0].to(dev))
            X, X0, GE = torch.stack(X), torch.stack(X0), torch.stack(GE)
        sc = C.score_ra(m32, X.cpu().numpy(), GE.cpu().numpy(), sq.betas, dev)
        ea, ez, _, _ = C.r_art_terms(geo.mano, X0, X, GE, betas)
        res["per_seq"][s] = {"n": int(len(X)), "mpjpe_ra_mm": float(sc["mpjpe_ra_mm"].mean()),
                             "mpvpe_ra_mm": float(sc["mpvpe_ra_mm"].mean()), "wrist_mm": float(sc["wrist_mm"].mean()),
                             "R_art": float(ea.sum() / ez.sum())}
        print(s, json.dumps(res["per_seq"][s]), f"{time.time() - t0:.0f} s", flush=True)
    res["wall_s"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    C.receipt(out.parent, {"probe": "p5"}, scripts=(__file__,))


if __name__ == "__main__":
    main()
