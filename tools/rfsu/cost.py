#!/usr/bin/env python3
"""Cost of route-first EvHandTrack on the simulated data (docs/EXPERIMENT_RECORDS.md#rfsu-20261113).

tools/real/rfs_cost.py on the simulated data: the same harness (tools/pfo/cost.py: the first 600 50 ms lyq_local packets,
parameters incl. the router, executed Linear / Conv MACs, row-form and end-to-end latency; PMR's stage breakdown; the
streaming form; the fast trunk semkine/pmr_fast.py; default kernels), the same deployable route-first step (the router
once per packet over every event, the history flags carried over, the nodes' routes read from the router output;
tools/real/rfs_cost.py `rfs_select` / `cached_sources`) and the same check before timing (deployable step vs the
readings' implementation, fast vs reference trunk; <tag>/check.json), with tools/rfsu/sim.py's arms and NEAR_PX:

  --arm d1r_rol5  the original EvHandTrack (PMR run pmr_d1r_rol5_s3407, the paper's simulated row): --select staging
                  (the published definition, the node selection built with the packet outside the timed step) or
                  step (tpd_str's selection inside the step), a same-block re-measure
  --arm rfs_rol5  route-first EvHandTrack (tools/rfsu/sim.py run pmr_rfs_rol5_s3407): --select step only; --ref = the
                  readings' implementation (semkine/rfs.py install_stream), descriptive
  --arm rfsr_rol5 the same with the simulated r_s2dx2 (the paper's row since the user's 2026-11-13 decision)

    CUDA_VISIBLE_DEVICES=<4..7> python tools/rfsu/cost.py --arm d1r_rol5|rfs_rol5|rfsr_rol5 --select staging|step [--ref] [--smoke]
Tags under outputs/rfsu/20261113/sim/costs/: pmr_rfs_rol5_s3407__str (deployable), ..__str__ref,
pmr_d1r_rol5_s3407__str__<select>_rfsblk. --smoke: 120 packets, CHECK 20, under outputs/rfsu/20261113/sim/smoke/costs_rfs/.
One GPU among 4-7 with nothing else running on it.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

SEED = 3407
ARMS = ("d1r_rol5", "rfs_rol5", "rfsr_rol5")
PMR = ROOT / "outputs/pmr/20261109"
ORIG_CHECK = ROOT / "outputs/cmp/20261110/costs/evhandtrack_s3407__fast_check.json"


def run(arm, select, ref, smoke):
    from tools.real import rfs_cost as RC
    from tools.rfsu import sim as S
    RC._S["X"] = S                                       # save_json / require for RC.check
    S.require(arm in ARMS, "arm")
    rfs = arm != "d1r_rol5"
    S.require(select == "step" or not rfs, "route-first runs place the selection inside the step (--select step)")
    S.require(not ref or (rfs and select == "step"), "--ref is the route-first readings' implementation")
    from tools.pmr import arms_any6  # noqa: F401
    from semkine import rdr_hooks as RH
    from tools.pmr import ablrt as RT
    RH.enable()
    RT.patch_loader()
    from tools.cmp.common import now, sha
    from tools.pfo import cost as C
    from tools.pfo import nets as NM
    from tools.pmr import cost2 as PC2
    from tools.pmr import run as R
    from semkine import eval_track as ET
    from semkine import pmr_fast as F
    from semkine import tpd_str as TS
    from semkine import xag as XAG
    rn = f"pmr_{arm}_s{SEED}"
    home = S.OUT if rfs else PMR
    spec = home / "specs" / f"{rn}.json"
    sp = S.read_json(spec)
    rec = S.read_json(home / "receipts" / f"{rn}.json")
    S.require(rec["status"] == "PASS" and sha(sp["checkpoint"]) == sp["checkpoint_sha256"]
              == rec["checkpoint_sha256_by_step"]["1500"], rn + ": checkpoint changed since acceptance")
    R.prepare(spec)
    import yaml
    from semkine import s45sm as SM
    router_ck = str(yaml.safe_load(Path(sp["config"]).read_text())["S45SM"]["ROUTER_CHECKPOINT"])
    root = S.OUT / "smoke" / "costs_rfs" if smoke else S.OUT
    C.OUT = root
    n_check = 20 if smoke else RC.CHECK
    if smoke:
        C.N_PACKETS = 120

    def breakdown(net, model, *a, **k):
        RC._select = a[3].wrap("select", RC._select)
        return PC2.pmr_stage_breakdown(net, model, *a, **k)
    C.stage_breakdown = breakdown
    TS.install()
    if rfs:
        from semkine import rfs as RF
        RF.install(S.TAU, S.NEAR_PX, packet=False)
        RF.install_stream()
    native_win, native_make, native_load = ET._window_events, ET.make_eval_packet, NM.load
    pending = {}
    tag = (f"{rn}__str" + ("__ref" if ref else "")) if rfs else f"{rn}__str__{select}_rfsblk"
    if smoke:
        tag += "__" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    def window_events(events, offsets, tsub, end, window):
        S.require(int(window) == TS.WINDOW_MS, "streaming form: 50 ms update intervals only")
        pending["cut"] = (events, offsets, tsub, int(end))
        return native_win(events, offsets, tsub, end, window)

    def make_eval_packet(*args, **kwargs):
        S.require("cut" in pending, "streaming form: a packet without its window cut")
        batch = native_make(*args, **kwargs)
        cut = pending.pop("cut")
        if select == "staging":
            batch.str_state = TS.prepare_packet(batch, *cut)
            RC._S["nodes"].append(int(batch.str_state["nodes"]))
        else:
            batch.str_cut = cut
        return batch

    def load(*args, **kwargs):
        net = native_load(*args, **kwargs)
        TS.retoken()
        forward = net.model.forward_packet

        def forward_packet(batch, *a, **k):
            if select == "staging":
                TS._STATE["pending"] = batch.str_state
            else:
                RC._select(batch)
            return forward(batch, *a, **k)
        net.model.forward_packet = forward_packet
        S.require(str(SM._STATE.get("checkpoint")) == router_ck, "loaded router is not the eval config's")
        if rfs:
            RC._S["native_sources"] = XAG.sample_sources
            RC.check(net, C.OUT / "costs" / tag, n_check, rn)
            RC._use("ref" if ref else "dep")
        else:
            c = S.read_json(ORIG_CHECK)
            S.require(c["run"] == rn and c["max_abs_diff"] < RC.FAST_MAX, "the packet run's fast check")
            RC._use("orig")
            F.fastify(net.model.xyz)
        return net
    ET._window_events, ET.make_eval_packet, NM.load = window_events, make_eval_packet, load
    C.run(rn, "fast", str(spec), tag, "default")
    nodes, elig = RC._S["nodes"], RC._S["elig"]
    S.save_json(root / "costs" / tag / "stream.json",
                {"stream": True, "select": select, "route_first": rfs, "implementation": (
                    "readings" if ref else "deployable") if rfs else "tpd_str",
                 "router_checkpoint": router_ck, "router_checkpoint_sha256": sha(router_ck), "near_px": S.NEAR_PX,
                 "budget": TS.BUDGET, "window_ms": TS.WINDOW_MS,
                 "n_selections": len(nodes), "nodes_mean": sum(nodes) / max(len(nodes), 1),
                 "nodes_max": max(nodes, default=0),
                 "eligible_share_mean": (sum(e / max(n, 1) for e, n in elig) / len(elig)) if elig else None,
                 "events_mean": (sum(n for _, n in elig) / len(elig)) if elig else None,
                 "written_utc": now()}, exclusive=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--select", required=True, choices=("staging", "step"))
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    run(a.arm, a.select, a.ref, a.smoke)


if __name__ == "__main__":
    main()
