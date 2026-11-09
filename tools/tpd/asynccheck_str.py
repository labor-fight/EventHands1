#!/usr/bin/env python3
"""TPD stage STR asynchronous-equivalence check (docs/EXPERIMENT_RECORDS.md#tpd-str-prereg-20261106): tools/tpd/asynccheck.py's
two tests on the streaming input form (semkine/tpd_str.py, router exempt) plus a partition test; CPU, one network,
receipts/async_<name>.json.

    python -m tools.tpd.asynccheck_str --name N --spec SPEC

On the sampled zgz e1 packets of tools/tpd/asynccheck.py (six with the most events and six at random per sequence, prior =
the GT at the window start; each packet with the raw 50 ms before it and the recording's event counter):
  prefix     as asynccheck.py: the events after node m's event replaced (another count, later times), routes of nodes <= m
             held; node indices, tokens, association records and ray directions of nodes <= m bitwise unchanged
  order      as asynccheck.py: the readout with the node order permuted equals arrival order (max |diff| <= 1e-5)
  partition  the packet's events fed to `semkine.tpd_str.Stream` one by one (the first 1500, then the rest), in chunks of
             max(37, n / 300) and in 60 random chunks: the kept events and their tokens equal the batched form's nodes and
             tokens bitwise
PASS iff every sampled packet passes all three.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.tpd.common import OUT, ZGZ, now, require, save, sha  # noqa: E402

TOL = 1e-5
FRACS = (0.25, 0.5, 0.75)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--out-root")
    a = p.parse_args(argv)
    import torch
    from semkine import eval_track as ET
    from semkine import tpd_str as TS
    from semkine import xag as XAG
    from tools.pfo import engine as E
    from tools.pfo import nets as N
    from tools.tpd import run_rol as RR
    root_out = Path(a.out_root) if a.out_root else OUT
    out = root_out / "receipts" / f"async_{a.name}.json"
    require(not out.exists(), "checked once: " + str(out))
    torch.set_num_threads(16)
    torch.use_deterministic_algorithms(True)
    TS.install()
    s = RR.prepare(a.spec)
    dev = torch.device("cpu")
    net = N.load(a.name, device=dev, spec={**s, "checkpoint": Path(s["checkpoint"]), "config": Path(s["config"])})
    TS.retoken()                                                    # after the engine's FAST.install
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

                def run(ev, keep_routes=None, n_keep=0, permute=None):
                    held.update(src=keep_routes, n=n_keep)
                    perm["p"] = permute
                    b = ET.make_eval_packet(np.ascontiguousarray(ev, np.float32), prev, ctx.betas, ctx.K, 50, dev)
                    TS.prepare_packet(b, ctx.events, ctx.offsets, ctx.tsub, end)
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
                row = {"seq": seq, "end": end, "events": int(len(ev5)), "nodes": nodes, "prefix": {}, "partition": {}}
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
                start = end - 49
                hist = torch.from_numpy(TS.history_events(ctx.events, ctx.offsets, ctx.tsub, start))
                ok_part = True
                n_ev = len(ev5)
                for name in ("1", "37", "random"):
                    if name == "random":
                        cuts = np.sort(rng.choice(np.arange(1, n_ev), min(60, max(n_ev - 1, 0)), replace=False))
                    elif name == "1":
                        cuts = np.arange(1, min(n_ev, 1500))         # the first 1500 one by one, then the rest at once
                    else:
                        cuts = np.arange(37, n_ev, max(37, n_ev // 300))
                    st = TS.Stream(180, 240, counter=int(ctx.offsets[start]), history=hist)
                    for chunk in np.split(ev5, cuts):
                        st.ingest(torch.from_numpy(np.ascontiguousarray(chunk)))
                    kept, tok = st.update()
                    same = bool(torch.equal(kept, src0[:nodes])) and bool(torch.equal(tok, L0["tok"][0, :nodes]))
                    row["partition"][name] = same
                    ok_part = ok_part and same
                row.update(prefix_ok=ok_prefix, order_max_abs=diff, order_ok=diff <= TOL, partition_ok=ok_part)
                rows.append(row)
                print(row, flush=True)
    status = "PASS" if all(r["prefix_ok"] and r["order_ok"] and r["partition_ok"] for r in rows) else "FAIL"
    save(out, {"name": a.name, "spec": a.spec, "form": "stream", "rows": rows, "status": status,
               "checkpoint_sha256": sha(s["checkpoint"]), "checked_at_utc": now()}, exclusive=True)
    print(status, flush=True)


if __name__ == "__main__":
    main()
