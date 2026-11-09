#!/usr/bin/env python3
"""TPD asynchronous-equivalence check of a deployed closed-loop network (docs/EXPERIMENT_RECORDS.md#tpd-causal-prereg-20261105;
ACU's two tests, tools/acu/asynccheck.py, with the packet-based router exempt): CPU, one network, receipts/async_<name>.json.

    python -m tools.tpd.asynccheck --name N --spec SPEC [--causal] [--lc CKPT]

On sampled zgz e1 packets (the six with the most events and six at random per sequence, prior = the GT at the window start):
  prefix  for nodes m at 1/4, 1/2 and 3/4 of the node list: every window event after node m's event is replaced by random
          events (another count, times after node m's), the routes of nodes <= m are held at the original packet's
          (router exempt); the node event indices, tokens, association records and ray directions of nodes <= m must be
          bitwise unchanged
  order   the readout with the node order randomly permuted (each node keeps its quantities) equals the readout in arrival
          order (max |difference| <= 1e-5 in the 51-D output): the per-vertex aggregation is a sum / max / mass, so any
          partition of the stream into computation chunks accumulates to the same readout at the same query
PASS iff every sampled packet passes both. With --causal the trunk's nodes come from semkine/tpd_causal.py; without it,
B2's count-dependent stride is expected to fail the prefix test on packets above the 2048-node cap.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.tpd.common import OUT, ZGZ, now, read, require, save, sha  # noqa: E402

TOL = 1e-5
FRACS = (0.25, 0.5, 0.75)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--lc")
    p.add_argument("--out-root")
    a = p.parse_args(argv)
    import torch
    from semkine import eval_track as ET
    from semkine import tpd_causal as TC
    from semkine import xag as XAG
    from tools.pfo import engine as E
    from tools.pfo import nets as N
    from tools.tpd import run_rol as RR
    root_out = Path(a.out_root) if a.out_root else OUT
    out = root_out / "receipts" / f"async_{a.name}.json"
    require(not out.exists(), "checked once: " + str(out))
    torch.set_num_threads(16)
    torch.use_deterministic_algorithms(True)
    if a.causal:
        TC.install()
    s = RR.prepare(a.spec)
    dev = torch.device("cpu")
    net = N.load(a.name, device=dev, spec={**s, "checkpoint": Path(s["checkpoint"]), "config": Path(s["config"])})
    TC.retoken()                                                    # after the engine's FAST.install
    if a.lc:
        from semkine import tpd_lc as LC
        LC.install(a.lc)
        XAG.sample_sources = LC.lc_sources
    model = net.model
    model.xyz_keep_aux = True
    xyz = model.xyz
    native_fwd = type(xyz).forward
    perm = {"p": None}

    def fwd(self, tok, d, mask, verts, assoc, prev48, **kw):
        q = perm["p"]
        if q is not None:
            tok, d, mask = tok[:, q], d[:, q], mask[:, q]
            assoc = {k: (v[:, q] if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == q.numel() else v)
                     for k, v in assoc.items()}
        return native_fwd(self, tok, d, mask, verts, assoc, prev48, **kw)
    type(xyz).forward = fwd
    native_sources = XAG.sample_sources
    held = {"src": None, "n": 0}

    def sources(batch, src, mask, **kw):
        out_ = native_sources(batch, src, mask, **kw)
        if held["src"] is not None:
            n = held["n"]
            for k in out_:
                out_[k] = out_[k].clone()
                out_[k][:, :n] = held["src"][k][:, :n]
        return out_
    XAG.sample_sources = sources
    rng = np.random.default_rng(3407)
    rows = []
    with torch.no_grad():
        for seq in ZGZ:
            root, seqs = E.sequences(net.cfg, (seq,))
            ctx = N.sequence(root, seqs[0][1], seq, dev)
            net.bind(ctx)
            grid = E.endpoint_rows(ctx.aux)
            counts = np.array([int(ctx.offsets[r[1] + 1] - ctx.offsets[r[1] - 49]) for r in grid])
            pick = list(np.argsort(-counts)[:6]) + list(rng.choice(len(grid), 6, replace=False))
            for i in pick:
                _, end, prev_ms, _ = grid[int(i)]
                end = int(end)
                prev = torch.from_numpy(np.asarray(ctx.gt[int(prev_ms)], np.float32)).view(1, -1)
                ev5 = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, end, 50)
                k = TC.k_from_prev(TC.n_prev_events(ctx.offsets, end))

                def run(ev, keep_routes=None, n_keep=0, permute=None):
                    TC.set_k(k.view(1))
                    held.update(src=keep_routes, n=n_keep)
                    perm["p"] = permute
                    b = ET.make_eval_packet(np.ascontiguousarray(ev, np.float32), prev, ctx.betas, ctx.K, 50, dev)
                    y = model.forward_packet(b)
                    last = {kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in model.xyz_last.items()}
                    held.update(src=None, n=0)
                    perm["p"] = None
                    return y, last, b
                y0, L0, b0 = run(ev5)
                src0, mask0 = L0["src"][0], L0["mask"][0]
                nodes = int(mask0.sum())
                srcs = {kk: L0[kk] for kk in ("xag_part", "xag_face", "xag_bary", "xag_valid", "xag_source_index",
                                              "xag_source_ms") if kk in L0}
                row = {"seq": seq, "end": end, "events": int(len(ev5)), "nodes": nodes, "k": int(k), "prefix": {}}
                ok_prefix = True
                for f in FRACS:
                    m = max(int(nodes * f) - 1, 0)
                    em = int(src0[m])
                    tail = len(ev5) - em - 1
                    n_new = max(int(tail * (0.6 if rng.random() < 0.5 else 1.5)), 1)
                    t0, t1 = float(ev5[em, 3]), 0.05
                    new = np.zeros((n_new, 5), np.float32)
                    new[:, 1] = rng.integers(0, 240, n_new)
                    new[:, 2] = rng.integers(0, 180, n_new)
                    new[:, 3] = np.sort(rng.uniform(t0, t1, n_new)).astype(np.float32)
                    new[:, 4] = rng.integers(0, 2, n_new)
                    ev2 = np.concatenate([ev5[:em + 1], new], 0)
                    _, L1, _ = run(ev2, keep_routes=srcs, n_keep=m + 1)
                    same = bool(torch.equal(L1["src"][0, :m + 1], src0[:m + 1]))
                    for kk in ("tok", "d"):
                        same = same and bool(torch.equal(L1[kk][0, :m + 1], L0[kk][0, :m + 1]))
                    for kk in ("vid", "A", "q", "z", "valid", "hit"):
                        same = same and bool(torch.equal(L1["assoc"][kk][0, :m + 1], L0["assoc"][kk][0, :m + 1]))
                    row["prefix"][str(f)] = same
                    ok_prefix = ok_prefix and same
                q = torch.randperm(L0["mask"].shape[1], generator=torch.Generator().manual_seed(int(end)))
                yp, _, _ = run(ev5, permute=q)
                diff = float((yp - y0).abs().max())
                row.update(prefix_ok=ok_prefix, order_max_abs=diff, order_ok=diff <= TOL)
                rows.append(row)
                print(row, flush=True)
    status = "PASS" if all(r["prefix_ok"] and r["order_ok"] for r in rows) else "FAIL"
    save(out, {"name": a.name, "spec": a.spec, "causal": a.causal, "lc": a.lc, "rows": rows, "status": status,
               "checkpoint_sha256": sha(s["checkpoint"]), "checked_at_utc": now()}, exclusive=True)
    print(status, flush=True)


if __name__ == "__main__":
    main()
