#!/usr/bin/env python3
r"""G5: the cost accounting, per event and per window.

Registered as report-only. The claim KEG makes is algorithmic, not engineering: the recurrence is
`O(1)` per event and exact under re-discretisation, so a deployment *can* consume events as they
arrive. This repository does not ship such a runtime -- there is no CUDA kernel that updates one
node on one event -- and pretending otherwise by quoting a latency would be measuring PyTorch's
dispatch overhead rather than the method. So the table reports two separable things:

* **Analytic multiply-accumulates**, split into a per-event term and a per-window term. This is the
  quantity the architecture determines, and it is where the comparison is meaningful: a dense
  trunk spends the same 1.5 GMAC on a window holding 300 events as on one holding 29 000, because
  the cost is set by the pixel grid and not by the evidence.
* **Measured throughput** on this machine, batched, which is what actually bounds training.

The rasteriser is excluded from both arms. `PREV_RENDER` is on for the dense baseline and KSSF is
on for KEG, and both are a per-window pass over the same 1538 triangles, so including them would
add a shared constant that hides the term under test. It is reported separately instead.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from semkine.frontends import build_frontend             # noqa: E402
from semkine.keg import KSSF_CHANNELS, N_GROUPS, N_NODES, ROUTE_SLOTS, TOKEN_DIM  # noqa: E402

#: Measured events per 50 ms window, `ARCHITECTURE_AUDIT.md` §7: the sparsest and densest
#: validation sequences. One number would hide the whole point, which is that the dense cost does
#: not move between them.
EVENTS_PER_50MS = {"zgz_local (sparsest)": 1_512, "zgz_global (densest)": 29_255}


def keg_macs(hidden: int, node_dim: int, feat_dim: int, n_layers: int, kssf: bool,
             routes: int = ROUTE_SLOTS):
    """Per-event and per-window multiply-accumulates of the KEG frontend.

    `routes` is how many nodes an event is shared between: four under the KSSF skinning routing,
    one under the state-independent grid fallback. The embedding is computed once and reused across
    slots, so only the accumulation scales with it.
    """
    in_dim = TOKEN_DIM + (KSSF_CHANNELS if kssf else 0)
    slots = routes if kssf else 1
    per_event = in_dim * hidden            # the per-event embedding, shared across routing slots
    # Per slot: the decay weight, the weighted value, the mass, and the peak comparison.
    per_event += slots * 4 * hidden
    per_window = N_GROUPS * (3 * hidden + 1) * node_dim          # node projection
    per_window += n_layers * 2 * N_NODES * node_dim * node_dim    # TreeConv self + message
    per_window += 3 * node_dim * feat_dim + feat_dim * feat_dim   # readout
    return per_event, per_window


def resnet18_macs(h: int, w: int, in_ch: int) -> int:
    """Multiply-accumulates of the published trunk on an `(h, w, in_ch)` window.

    Counted from the layer shapes rather than quoted from the literature, because the input here is
    180x240 with a `conv1` adapter rather than the 224x224x3 every published figure refers to.
    """
    total = in_ch * 3 * 3 * 3 * h * w                       # the conv1 adapter
    hh, ww = h // 2, w // 2                                 # resnet18 stem, stride 2
    total += 3 * 64 * 7 * 7 * hh * ww
    hh, ww = hh // 2, ww // 2                               # max pool
    for ch, blocks, first_stride in ((64, 2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2)):
        cin = ch if first_stride == 1 else ch // 2
        for b in range(blocks):
            s = first_stride if b == 0 else 1
            oh, ow = hh // s, ww // s
            total += cin * ch * 9 * oh * ow + ch * ch * 9 * oh * ow
            if s != 1:
                total += cin * ch * oh * ow                 # the downsample projection
            cin, hh, ww = ch, oh, ow
    return total


def measure(module, n_events: int, batch: int, extra_ch: int, device, iters: int = 10) -> float:
    g = torch.Generator().manual_seed(0)
    ev, ptr = [], [0]
    for b in range(batch):
        t = torch.sort(torch.rand(n_events, generator=g) * 0.05).values
        ev.append(torch.stack([torch.full((n_events,), float(b)),
                               torch.randint(0, 240, (n_events,), generator=g).float(),
                               torch.randint(0, 180, (n_events,), generator=g).float(),
                               t,
                               torch.randint(0, 2, (n_events,), generator=g).float()], -1))
        ptr.append(ptr[-1] + n_events)
    ev = torch.cat(ev).to(device)
    ptr = torch.tensor(ptr).to(device)
    dt = torch.full((batch,), 0.05, device=device)
    extra = torch.zeros(ev.shape[0], extra_ch, device=device) if extra_ch else None
    with torch.no_grad():
        module(ev, ptr, dt, extra)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            module(ev, ptr, dt, extra)
        if device.type == "cuda":
            torch.cuda.synchronize()
        el = time.perf_counter() - t0
    return iters * ev.shape[0] / el


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--node-dim", type=int, default=64)
    ap.add_argument("--feat-dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--height", type=int, default=180)
    ap.add_argument("--width", type=int, default=240)
    ap.add_argument("--out", default="outputs/semkine/s22_efficiency.json")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pe, pw = keg_macs(a.hidden, a.node_dim, a.feat_dim, a.layers, kssf=True)
    dense = resnet18_macs(a.height, a.width, in_ch=4)

    rows = []
    print(f"KEG    {pe:,} MAC/event + {pw:,} MAC/window")
    print(f"LNES   {dense:,} MAC/window, independent of the event count\n")
    print(f"{'sequence':24s} {'events':>8s} {'KEG MMAC':>10s} {'LNES MMAC':>10s} {'ratio':>8s}")
    for name, n in EVENTS_PER_50MS.items():
        keg = pe * n + pw
        rows.append({"sequence": name, "events_per_50ms": n, "keg_mac": keg, "lnes_mac": dense,
                     "lnes_over_keg": dense / keg})
        print(f"{name:24s} {n:>8,} {keg / 1e6:>10.2f} {dense / 1e6:>10.2f} "
              f"{dense / keg:>7.1f}x")

    thr = {}
    m = build_frontend("keg", hidden=a.hidden, feat_dim=a.feat_dim, node_dim=a.node_dim,
                       extra_channels=KSSF_CHANNELS).to(device).eval()
    for name, n in EVENTS_PER_50MS.items():
        thr[name] = measure(m, n, 16, KSSF_CHANNELS, device)
        print(f"measured  {name:24s} {thr[name] / 1e6:8.1f} M events/s (batch 16)")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "note": "report-only; no asynchronous runtime is claimed or shipped",
        "keg_mac_per_event": pe, "keg_mac_per_window": pw,
        "lnes_mac_per_window": dense,
        "rasteriser_excluded": "both arms run one pass over 1538 MANO triangles per window",
        "rows": rows, "measured_events_per_s": thr,
    }, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
