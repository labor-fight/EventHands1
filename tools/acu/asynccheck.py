#!/usr/bin/env python3
"""ACU stage-3 asynchronous-equivalence check of the deployed form (docs/EXPERIMENT_RECORDS.md#acu-s3-prereg-20261104),
CPU, the final configuration's trained weights (seed 3407), receipts/asynccheck.json; run once.

On sampled zgz evaluation packets (with their stream history):
  prefix     for nodes m at 1/4, 1/2 and 3/4 of the window: randomising every window event after node m leaves every
             per-node quantity of nodes <= m bitwise unchanged (node set, PR routes, association, CA geometry, tokens)
  order      the readout with the node order randomly permuted (each node keeps its quantities) equals the readout in
             arrival order (max |difference| <= 1e-5 in the 51-D output): the per-vertex aggregation is a sum / max / mass,
             so any partition of the stream into computation chunks accumulates to the same readout at the same query
PASS iff every packet passes both.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.acu import arms_s2 as A  # noqa: E402
from tools.acu.common import OUT, now, read, require, save, sha  # noqa: E402

N_PACKETS = 12
TOL = 1e-5


def main():
    import torch
    from semkine import acu_causal as AC
    from tools.acu import run_s2 as R
    from tools.acu.decide_s3 import chosen
    from tools.pfo import engine as E
    from tools.pfo import nets as N
    out = OUT / "receipts" / "asynccheck.json"
    require(not out.exists(), "checked once")
    torch.set_num_threads(16)
    w, t = chosen()
    name, sp = R.spec(w, 3407, t)
    R.install(w, t)
    s = read(sp)
    dev = torch.device("cpu")
    net = N.load(name, device=dev, spec={**s, "checkpoint": Path(s["checkpoint"]), "config": Path(s["config"])})
    net.model.xyz_keep_aux = True
    xyz = net.model.xyz
    native_fwd = type(xyz).forward
    perm_state = {"perm": None}

    def fwd(self, tok, d, mask, verts, assoc, prev48, **kw):
        p = perm_state["perm"]
        if p is not None:
            tok, d, mask = tok[:, p], d[:, p], mask[:, p]
            assoc = {k: (v[:, p] if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == p.numel() else v)
                     for k, v in assoc.items()}
        return native_fwd(self, tok, d, mask, verts, assoc, prev48, **kw)
    type(xyz).forward = fwd
    rows = []
    for seq in ("zgz_local", "zgz_global"):
        root, seqs = E.sequences(net.cfg, (seq,))
        ctx = N.sequence(root, seqs[0][1], seq, dev)
        net.bind(ctx)
        ends = E.endpoint_rows(ctx.aux)
        for i in np.linspace(5, len(ends) - 1, N_PACKETS // 2).astype(int):
            _, end, prev_ms, _ = ends[i]
            prev = torch.from_numpy(np.asarray(ctx.gt[int(prev_ms)], np.float32)).view(1, -1)
            b = AC.eval_batch(ctx, int(end), AC.run_start(ctx.aux, int(end)), prev, dev)
            with torch.no_grad():
                y0 = net.model.forward_packet(b)
                full = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in net.model.xyz_last.items()}
                n = int(full["mask"].sum())
                if n < 8:
                    continue
                prefix_ok = True
                for frac in (0.25, 0.5, 0.75):
                    m = int(n * frac)
                    cut = int(full["src"][0, m])
                    ev2 = b.events.clone()
                    later = torch.arange(ev2.shape[0]) > cut
                    g = torch.Generator().manual_seed(int(end) + m)
                    ev2[later, 1] = torch.randint(0, 240, (int(later.sum()),), generator=g).float()
                    ev2[later, 2] = torch.randint(0, 180, (int(later.sum()),), generator=g).float()
                    net.model.forward_packet(type(b)(**{**b.__dict__, "events": ev2}))
                    alt = net.model.xyz_last
                    keep = torch.zeros_like(full["mask"])
                    keep[0, : m + 1] = True
                    same = [torch.equal(full["src"][keep], alt["src"][keep]), torch.equal(full["tok"][keep], alt["tok"][keep])]
                    same += [torch.equal(full[k][keep], alt[k][keep]) for k in ("xag_part", "xag_face", "xag_bary", "xag_valid")]
                    same += [torch.equal(full["assoc"][k][keep], alt["assoc"][k][keep]) for k in full["assoc"]
                             if torch.is_tensor(full["assoc"][k]) and full["assoc"][k].dim() >= 2]
                    prefix_ok = prefix_ok and all(same)
                Nn = full["mask"].shape[1]
                perm_state["perm"] = torch.randperm(Nn, generator=torch.Generator().manual_seed(int(end)))
                y1 = net.model.forward_packet(b)
                perm_state["perm"] = None
                diff = float((y1 - y0).abs().max())
            rows.append({"seq": seq, "end": int(end), "nodes": n, "prefix_ok": bool(prefix_ok), "order_max_abs_diff": diff,
                         "ok": bool(prefix_ok and diff <= TOL)})
    type(xyz).forward = native_fwd
    receipt = {"check": "acu_async_equivalence", "winner": w, "trunk": t, "checkpoint_sha256": sha(s["checkpoint"]),
               "rows": rows, "tolerance": TOL, "status": "PASS" if rows and all(r["ok"] for r in rows) else "FAIL",
               "checked_at_utc": now()}
    save(out, receipt, exclusive=True)
    print(receipt["status"], [(r["nodes"], r["prefix_ok"], round(r["order_max_abs_diff"], 9)) for r in rows], flush=True)
    require(receipt["status"] == "PASS", "deployed form is not asynchronously equivalent")


if __name__ == "__main__":
    main()
