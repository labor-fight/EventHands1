#!/usr/bin/env python3
"""EvHandTrack cost block (docs/EXPERIMENT_RECORDS.md#evhandtrack-str-20261109) through PFO's frozen cost harness
(tools/pfo/cost.py: the first 600 lyq_local packets, parameters incl. the router, executed Linear / Conv MACs on the actual
tensors, row-form and end-to-end latency), output root outputs/evhandtrack/costs/<tag>/; one GPU, nothing else on it.

    python -m tools.evhandtrack.cost run --net s43 --policy default       S43's sparse executor (the comparator)
    python -m tools.evhandtrack.cost run --net packet --policy default    EvHandTrack 3407, ordinary packet form
    python -m tools.evhandtrack.cost run --net stream --policy default    EvHandTrack 3407, streaming form

The streaming form (semkine/tpd_str.py): the harness cuts each packet from the raw arrays (`semkine.eval_track.
_window_events` + `make_eval_packet`) and calls `model.forward_packet` outside PFO's engine, so the engine hooks of
tpd_str do not fire. Here every packet gets its streaming state (the 50 ms history, the event counter, the kept events) in
the host part of the step, right after the window cut (so the end-to-end latency includes it), carried on the packet and
set just before the trunk runs (the staged packets of the row form keep their own state), as tools/tpd/cost.py does for
the causal form's stride.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo.common import now, read, require, save, sha  # noqa: E402

OUT = ROOT / "outputs/evhandtrack"
SPEC = ROOT / "outputs/tpd/20261105/specs/tpd_rol5_s3407.json"
RECEIPT = ROOT / "outputs/tpd/20261105/receipts/tpd_rol5_s3407.json"
POLICIES = ("default", "det")
_NODES = []


def stream_cost_hooks():
    from semkine import eval_track as ET
    from semkine import tpd_str as TS
    from tools.pfo import nets as NM
    native_win, native_make, native_load = ET._window_events, ET.make_eval_packet, NM.load
    pending = {}

    def window_events(events, offsets, tsub, end, window):
        require(int(window) == TS.WINDOW_MS, "streaming form: 50 ms update intervals only")
        pending["cut"] = (events, offsets, tsub, int(end))
        return native_win(events, offsets, tsub, end, window)

    def make_eval_packet(*args, **kwargs):
        require("cut" in pending, "streaming form: a packet without its window cut")
        batch = native_make(*args, **kwargs)
        batch.str_state = TS.prepare_packet(batch, *pending.pop("cut"))
        _NODES.append(batch.str_state["nodes"])
        return batch

    def load(*args, **kwargs):
        net = native_load(*args, **kwargs)
        TS.retoken()
        forward = net.model.forward_packet

        def forward_packet(batch, *a, **k):
            TS._STATE["pending"] = batch.str_state
            return forward(batch, *a, **k)
        net.model.forward_packet = forward_packet
        return net
    ET._window_events, ET.make_eval_packet, NM.load = window_events, make_eval_packet, load


def run(net, policy):
    from tools.pfo import cost as C
    C.OUT = OUT
    if net == "s43":
        C.run("s43_s3407", "exec", None, f"s43_s3407__exec__{policy}", policy)
        return
    from tools.tpd import run_rol as RR
    sp = read(SPEC)
    require(sha(sp["checkpoint"]) == read(RECEIPT)["checkpoint_sha256_by_step"]["1500"], "rol5 3407 checkpoint changed")
    RR.prepare(SPEC)
    if net == "stream":
        from semkine import tpd_str as TS
        TS.install()
        stream_cost_hooks()
    tag = f"evhandtrack_s3407__{net}__{policy}"
    C.run("tpd_rol5_s3407", "fast", str(SPEC), tag, policy)
    if net == "stream":
        save(OUT / "costs" / tag / "stream.json",
             {"stream": True, "budget": TS.BUDGET, "window_ms": TS.WINDOW_MS, "n_packets_built": len(_NODES),
              "nodes_mean": sum(_NODES) / max(len(_NODES), 1), "nodes_max": max(_NODES, default=0),
              "written_utc": now()}, exclusive=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--net", required=True, choices=("s43", "packet", "stream"))
    r.add_argument("--policy", required=True, choices=POLICIES)
    a = p.parse_args(argv)
    return run(a.net, a.policy)


if __name__ == "__main__":
    main()
