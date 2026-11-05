#!/usr/bin/env python3
"""EVF cost (docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104).

    python -m tools.evf.cost macs --arm A            CPU, before any training of A: executed MACs per packet on the
                                                     600 lyq_local packets of the main-row rule and on a full-load
                                                     packet (EVF.MAX_NODES nodes); refuses (exit 3) above 2x S43
    python -m tools.evf.cost run --arm A --seed S --phase P --step N [--policy det|default] [--tag T]
                                                     GPU cost reading of a trained arm in an exclusive block: PFO's
                                                     rules (tools/pfo/cost.py: params, MACs on the actual tensors, row
                                                     latency with the x 1.75 / EventHands-Full anchor, end-to-end own
                                                     closed loop p50 / p90 / p99, peak memory), plus the full-load packet
MACs: every Linear layer on its actual input (node counts vary per packet) + MANO FK (analytic, PFO's count) + the
pooling contraction; the rasteriser / distance transforms are comparisons, reported apart as element operations.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.evf import arms as A  # noqa: E402
from tools.evf.common import OUT, now, read, require, save, sha  # noqa: E402

S43_EXEC_MACS = 128.3e6          # outputs/pfo/20261103/costs/s43_s3407__exec__det/report.json, median over the packets
BUDGET = 2.0 * S43_EXEC_MACS
N_PACKETS = 600


class LinearMacs:
    def __init__(self, model):
        import torch
        self.total, self.by = 0, {}
        self.handles = []
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                self.handles.append(m.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def f(m, inp, out):
            n = int(inp[0].numel() // m.in_features) * m.in_features * m.out_features
            self.total += n
            key = ".".join(name.split(".")[:2])
            self.by[key] = self.by.get(key, 0) + n
        return f

    def reset(self):
        self.total, self.by = 0, {}

    def close(self):
        for h in self.handles:
            h.remove()


def build_model(arm, seed=3407, device="cpu"):
    import torch
    from semkine.evf_model import EVFModel
    torch.manual_seed(seed)
    cfg = A.build_config(arm, seed, "formal")
    cfg["MANO"]["NPZ"] = str(ROOT / cfg["MANO"]["NPZ"]) if not os.path.isabs(cfg["MANO"]["NPZ"]) else cfg["MANO"]["NPZ"]
    return EVFModel(cfg).to(device).eval(), cfg


def lyq_items(cfg, device):
    from tools.pfo.cost import lyq_packets
    return lyq_packets(cfg, device)


def full_load_packet(model, ctx, device):
    """A packet with more than EVF.MAX_NODES nodes in distinct cells (the cap keeps MAX_NODES of them)."""
    import torch
    from semkine import eval_track as ET
    o = model.evf
    H, W, v = model.render_h, model.render_w, int(o["VOXEL_PX"])
    need = int(o["MAX_NODES"]) + 100
    cells = []
    for tb in range(int(o["TIME_BINS"])):
        for p in (0, 1):
            for gy in range(H // v):
                for gx in range(W // v):
                    cells.append((tb, p, gy * v, gx * v))
                    if len(cells) >= need * 4:
                        break
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(cells), need, replace=False))
    dt = 0.05
    rows = [(cells[i][3], cells[i][2], (cells[i][0] + 0.5) / int(o["TIME_BINS"]) * dt, cells[i][1]) for i in pick]
    rows.sort(key=lambda r: r[2])
    ev = np.zeros((len(rows), 5), np.float32)
    ev[:, 1:] = np.asarray(rows, np.float32)
    return ev


def macs(arm):
    import torch
    out = OUT / "costs" / f"macs_{arm}.json"
    require(not out.exists(), "measured once: " + str(out))
    from semkine import eval_track as ET
    from tools.pfo.cost import mano_fk_macs
    model, cfg = build_model(arm)
    device = torch.device("cpu")
    ctx, items = lyq_items(cfg, device)
    model.set_hand_context(ctx.betas, ctx.K)
    counter = LinearMacs(model)
    fk = mano_fk_macs(model.mano)
    per, nodes, by = [], [], {}
    t0 = time.time()
    with torch.no_grad():
        for end, prev_np, _ in items:
            ev5 = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, end, 50)
            pk = ET.make_eval_packet(ev5, torch.from_numpy(prev_np).view(1, -1), ctx.betas, ctx.K, 50, device)
            counter.reset()
            model.forward_packet(pk)
            k = int(model.evf_stats["kept"][0])
            pool = k * (model.net.att.out_features * model.net.val.out_features)
            per.append(counter.total + (fk + pool) * int(model.evf["ITERS"]))
            nodes.append(k)
            for key, v in counter.by.items():
                by[key] = by.get(key, 0) + v
        ev5 = full_load_packet(model, ctx, device)
        pk = ET.make_eval_packet(ev5, torch.from_numpy(items[0][1]).view(1, -1), ctx.betas, ctx.K, 50, device)
        counter.reset()
        model.forward_packet(pk)
        kf = int(model.evf_stats["kept"][0])
        full = counter.total + (fk + kf * model.net.att.out_features * model.net.val.out_features) * int(model.evf["ITERS"])
    counter.close()
    per = np.asarray(per, np.float64)
    params = int(sum(p.numel() for p in model.net.parameters()))
    rep = {"arm": arm, "evf": model.evf, "params_net": params, "n_packets": len(per),
           "macs": {"median": float(np.median(per)), "mean": float(per.mean()), "p90": float(np.percentile(per, 90)),
                    "max": float(per.max())},
           "macs_full_load": float(full), "nodes_full_load": kf,
           "nodes": {"median": float(np.median(nodes)), "p90": float(np.percentile(nodes, 90)), "max": int(max(nodes))},
           "macs_by_module_mean": {k: v / len(per) for k, v in by.items()}, "macs_mano_fk": fk,
           "s43_exec_macs": S43_EXEC_MACS, "budget_2x": BUDGET,
           "ratio_median": float(np.median(per)) / S43_EXEC_MACS, "ratio_full_load": float(full) / S43_EXEC_MACS,
           "within_budget": bool(np.median(per) <= BUDGET and full <= BUDGET), "cpu_seconds": time.time() - t0,
           "measured_at_utc": now()}
    save(out, rep, exclusive=True)
    print("MACs median %.1f M (x%.2f S43), full load %.1f M (x%.2f), nodes median %d, params %d" % (
        rep["macs"]["median"] / 1e6, rep["ratio_median"], full / 1e6, rep["ratio_full_load"], rep["nodes"]["median"],
        params), "WITHIN" if rep["within_budget"] else "OVER BUDGET", flush=True)
    sys.exit(0 if rep["within_budget"] else 3)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    m = sub.add_parser("macs")
    m.add_argument("--arm", required=True)
    a = p.parse_args(argv)
    if a.verb == "macs":
        macs(a.arm)


if __name__ == "__main__":
    main()
