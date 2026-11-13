#!/usr/bin/env python3
"""Cost of route-first EvHandTrack on the real data (docs/EXPERIMENT_RECORDS.md#real-rfs-20261112).

tools/real/cost.py `evht --net stream --fast` (tools/pfo/cost.py's harness on the first 600 50 ms packets of the real
lyq_local recording, PMR's stage breakdown, the streaming form, the fast trunk semkine/pmr_fast.py, default kernels) for
the route-first runs, with the streaming node selection placed by --select:

  staging   the published definition of EvHandTrack's streaming row: the node selection is built with the packet,
            outside the timed step. Original EvHandTrack (d1r_rol5) only: a same-block re-measure of its row.
  step      the node selection runs inside the timed step (and inside every other per-packet reading of the harness).
            d1r_rol5: tpd_str's streaming selection, unchanged.
            rfs_rol5 / rfsr_rol5, deployable form: the router runs once per packet over every event (its dense trunk
            once, its head over every event); on-hand = p >= TAU; eligible = on-hand events and the events within
            NEAR_PX canvas px of one (square window); tpd_str's streaming rule over the eligible events. The history
            window's flags are the previous packet's (the 50 ms before a packet are exactly the previous packet;
            computed afresh after a gap), and the nodes' routes are read from the per-event router output (no second
            router pass). --ref: the readings' implementation instead (semkine/rfs.py install_stream: the router on
            the packet, again on the history window, again on the nodes), descriptive.
Before timing, a route-first run checks, on the first CHECK packets in order (prev = GT at the run start, the harness's
row form), the deployable step against the readings' implementation (max |state difference|, packets whose node set
differs) and the fast trunk against the reference trunk; <tag>/check.json; it stops when a state differs by FAST_MAX.

    CUDA_VISIBLE_DEVICES=<4..7> python tools/real/rfs_cost.py --arm d1r_rol5|rfs_rol5|rfsr_rol5 --select staging|step
        [--ref] [--smoke]
Tags under outputs/real/20261110/costs/: pmr_<arm>_s3407__str (route-first, deployable step; tools/real/table5.py reads
it), pmr_<arm>_s3407__str__ref (--ref), pmr_d1r_rol5_s3407__str__<select>_rfsblk (the original, same block).
--smoke: 120 packets, CHECK 20, under outputs/real/20261110/smoke/costs_rfs/ (engineering only, not a reading).
One GPU among 4-7 with nothing else running on it.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

SEED = 3407
ARMS = ("d1r_rol5", "rfs_rol5", "rfsr_rol5")
CHECK = 100
FAST_MAX = 1e-3
_S = {"mode": None, "cache": {}, "last": None, "nodes": [], "elig": [], "native_sources": None, "X": None}


# --------------------------------------------------------------------------- the deployable route-first step
def router_all(batch):
    """The router over every event of the packet, once: `(z (N, Z), on (N,) bool)` (semkine/s45sm_fast.py
    `router_forward` with every event as a node; its dense trunk is the same call)."""
    import torch
    from semkine import s45sm as SM
    from semkine import s45sm_fast as FAST
    from semkine.encoder import EV_BATCH, EV_X, EV_Y
    events, ptr, dt = batch.events, batch.ptr, batch.delta_t_s
    net = SM._net(events.device)
    if events.shape[0] == 0:
        return None, torch.zeros(0, dtype=torch.bool, device=events.device)
    tok = FAST.event_tokens(events, ptr, dt, SM.HEIGHT, SM.WIDTH)
    fmap = FAST.trunk(net, FAST.event_image(events, ptr, dt).to(net.stem[0].weight.dtype))
    h4, w4 = fmap.shape[-2:]
    yy = (events[:, EV_Y].long() * h4 // SM.HEIGHT).clamp(0, h4 - 1)
    xx = (events[:, EV_X].long() * w4 // SM.WIDTH).clamp(0, w4 - 1)
    feat = fmap[events[:, EV_BATCH].long(), :, yy, xx]
    z = net.head(torch.cat([feat.to(tok.dtype), tok], -1))
    on = torch.sigmoid(net.onhand(z).squeeze(-1).float()) >= float(SM._STATE["onhand_tau"])
    return z, on


def vicinity(events, ptr, on, near):
    """semkine/rfs.py `eligible`'s second half: on-hand events and the events within `near` px of one (square)."""
    import torch
    import torch.nn.functional as F
    from semkine import s45sm as SM
    from semkine.encoder import EV_BATCH, EV_X, EV_Y
    if not near or near <= 0 or events.shape[0] == 0:
        return on
    Hh, Ww = int(SM.HEIGHT), int(SM.WIDTH)
    B = int(ptr.numel() - 1)
    b = events[:, EV_BATCH].long()
    x = events[:, EV_X].long().clamp(0, Ww - 1)
    y = events[:, EV_Y].long().clamp(0, Hh - 1)
    k = 2 * int(near) + 1
    img = torch.zeros(B, 1, Hh, Ww, device=events.device)
    img[b[on], 0, y[on], x[on]] = 1.0
    img = F.max_pool2d(img, (1, k), stride=1, padding=(0, int(near)))
    img = F.max_pool2d(img, (k, 1), stride=1, padding=(int(near), 0))
    return on | (img[b, 0, y, x] > 0)


def rfs_select(batch):
    """semkine/rfs.py install_stream's `prepare_packet` with the router once and the history flags carried over."""
    import torch
    from semkine import rfs as RF
    from semkine import tpd_str as TF
    from semkine.encoder import EV_T
    events, offsets, tsub, end = batch.str_cut
    window = TF.WINDOW_MS
    start = int(end) - int(window) + 1
    dev = batch.events.device
    hist = torch.from_numpy(TF.history_events(events, offsets, tsub, start)).to(dev)
    hp = torch.tensor([0, hist.shape[0]], dtype=torch.long, device=dev)
    c0 = torch.tensor([int(offsets[start])], dtype=torch.long, device=dev)
    z, on = router_all(batch)
    fp = vicinity(batch.events, batch.ptr, on, RF._STATE["near"])
    c = _S["cache"]
    if c.get("end") == int(end) - int(window) and min(int(window), start) == int(window) and c["n"] == hist.shape[0]:
        fh = c["elig"]                                         # the history window = the previous packet
    elif hist.shape[0]:
        hev = hist.clone()
        hev[:, EV_T] = hev[:, EV_T] + float(window) * 1e-3     # the window's own time origin (as the readings)
        fh, _ = RF.eligible(hev, hp, batch.delta_t_s[:1])
    else:
        fh = torch.zeros(0, dtype=torch.bool, device=dev)
    c.update(end=int(end), elig=fp, n=int(batch.events.shape[0]))
    idx = torch.nonzero(fp).squeeze(1)
    ev_on = batch.events[idx]
    ptr_on = torch.tensor([0, int(idx.numel())], dtype=torch.long, device=dev)
    hist_on = hist[fh]
    hp_on = torch.tensor([0, int(hist_on.shape[0])], dtype=torch.long, device=dev)
    s_on, mask, info = TF.stream_nodes(ev_on, ptr_on, hist_on, hp_on, c0, TF.BUDGET, TF.WINDOW_MS, None)
    src = idx[s_on] if idx.numel() else torch.zeros_like(s_on)
    src = torch.where(mask, src, torch.zeros_like(src))
    _S["last"] = {"ptr": batch.ptr, "z": z}
    _S["elig"].append((int(idx.numel()), int(batch.events.shape[0])))
    TF._STATE["pending"] = {"ptr": batch.ptr, "events": batch.events, "hist": hist, "hist_ptr": hp, "c0": c0,
                            "src": src, "mask": mask, "nodes": int(info["nodes_max"])}
    return TF._STATE["pending"]


def cached_sources(batch, src, mask, *, require_depth=False):
    """semkine/s45sm_fast.py `predicted_sources` with the nodes' router output read from `router_all`'s."""
    import torch
    from semkine import s45sm as SM
    from semkine import s45sm_fast as FAST
    last = _S["last"]
    if last is None or last["ptr"] is not batch.ptr or last["z"] is None:
        return _S["native_sources"](batch, src, mask, require_depth=require_depth)
    if require_depth:
        raise RuntimeError("S45-SM routes faces; full_oracle depth is not defined for it")
    st = SM._STATE
    net = SM._net(src.device)
    B, n = src.shape
    z = last["z"][src.reshape(-1)].view(B, n, -1) * mask.unsqueeze(-1).to(last["z"].dtype)
    out = {"z": z, "coord": net.coord(z), "onhand": net.onhand(z).squeeze(-1), "src": src, "mask": mask}
    out["patch" if hasattr(net, "patch") else "part"] = (net.patch if hasattr(net, "patch") else net.part)(z)
    routed = mask & (torch.sigmoid(out["onhand"].float()) >= st["onhand_tau"])
    face, bary = FAST.route(net, st["router"], out, routed)
    zero = torch.zeros_like(face)
    return {"xag_source_index": zero, "xag_source_ms": zero, "xag_part": net.face_part[face], "xag_face": face,
            "xag_bary": bary.to(torch.float32), "xag_valid": routed}


def _select(batch):
    from semkine import tpd_str as TS
    if _S["mode"] == "dep":
        p = rfs_select(batch)
    else:                                       # "orig": tpd_str's rule; "ref": the readings' route-first rule
        _S["last"] = None
        p = TS.prepare_packet(batch, *batch.str_cut)
    _S["nodes"].append(int(p["nodes"]))
    return p


def _use(mode):
    from semkine import xag as XAG
    _S["mode"], _S["last"] = mode, None
    _S["cache"].clear()
    if _S["native_sources"] is not None:
        XAG.sample_sources = cached_sources if mode == "dep" else _S["native_sources"]


# --------------------------------------------------------------------------- checks
def check(net, out_dir, n_check, rn):
    """Deployable step vs the readings' implementation, then the fast trunk vs the reference trunk (in place)."""
    import numpy as np
    import torch
    from semkine import pmr_fast as F
    from semkine import tpd_str as TS
    from tools.cmp.common import now
    from tools.pfo import cost as C
    X = _S["X"]
    ctx, items = C.lyq_packets(net.cfg, torch.device("cuda"))
    net.bind(ctx)
    items = items[:n_check]

    def pass_(mode):
        _use(mode)
        ys, nodes = [], []
        with torch.no_grad():
            for end, gt_a, _ in items:
                ys.append(net.step(ctx, end, gt_a))
                p = TS._STATE["pending"]
                nodes.append(p["src"][p["mask"]].cpu().numpy())
        return ys, nodes
    t0 = time.time()
    y_ref, n_ref = pass_("ref")
    y_dep, n_dep = pass_("dep")
    F.fastify(net.model.xyz)                    # the timed trunk from here on
    y_fast, _ = pass_("dep")
    d_dep = [float(np.abs(a - b).max()) for a, b in zip(y_ref, y_dep)]
    d_fast = [float(np.abs(a - b).max()) for a, b in zip(y_dep, y_fast)]
    same = [bool(a.shape == b.shape and (a == b).all()) for a, b in zip(n_ref, n_dep)]
    rec = {"run": rn, "packets": len(items),
           "deployable_vs_readings": {"max_abs_state_diff": max(d_dep), "mean_max_abs_state_diff": float(np.mean(d_dep)),
                                      "packets_same_nodes": int(sum(same)),
                                      "packets_other_nodes": [int(e) for (e, _, _), s in zip(items, same) if not s]},
           "fast_vs_reference_trunk": {"max_abs_state_diff": max(d_fast), "mean_max_abs_state_diff": float(np.mean(d_fast))},
           "limit": FAST_MAX, "seconds": time.time() - t0, "written_utc": now()}
    X.save_json(out_dir / "check.json", rec, exclusive=True)
    print("CHECK", rec, flush=True)
    X.require(max(d_dep) < FAST_MAX, f"deployable step differs from the readings' by {max(d_dep)}")
    X.require(max(d_fast) < FAST_MAX, f"fast trunk differs from the reference by {max(d_fast)}")
    _S["nodes"].clear()
    _S["elig"].clear()


# --------------------------------------------------------------------------- the block
def run(arm, select, ref, smoke):
    from tools.real import rfs as RFS
    X = RFS.extend()
    _S["X"] = X
    from tools.real import ns
    X.require(arm in ARMS, "arm")
    rfs = arm != "d1r_rol5"
    X.require(select == "step" or not rfs, "route-first runs place the selection inside the step (--select step)")
    X.require(not ref or (rfs and select == "step"), "--ref is the route-first readings' implementation")
    X.enter()
    from tools.pmr import arms_any6  # noqa: F401
    from semkine import rdr_hooks as RH
    from tools.pmr import ablrt as RT
    RH.enable()
    RT.patch_loader()
    X.patch_engine()                                      # lyq_local is looked up in the real split manifest
    from tools.cmp.common import now, sha
    from tools.pfo import cost as C
    from tools.pfo import nets as NM
    from tools.pmr import cost2 as PC2
    from tools.pmr import run as R
    from semkine import eval_track as ET
    from semkine import pmr_fast as F
    from semkine import tpd_str as TS
    from semkine import xag as XAG
    rn = X.run_name(arm, SEED)
    spec = X.OUT / "specs" / f"{rn}.json"
    sp = X.read_json(spec)
    rec = X.read_json(X.OUT / "receipts" / f"{rn}.json")
    X.require(rec["status"] == "PASS" and sha(sp["checkpoint"]) == sp["checkpoint_sha256"]
              == rec["checkpoint_sha256_by_step"]["1500"], rn + ": checkpoint changed since acceptance")
    R.prepare(spec)
    import yaml
    from semkine import s45sm as SM
    router_ck = str(yaml.safe_load(Path(sp["config"]).read_text())["S45SM"]["ROUTER_CHECKPOINT"])
    root = ns.OUT / "smoke" / "costs_rfs" if smoke else ns.OUT
    C.OUT = root
    n_check = 20 if smoke else CHECK
    if smoke:
        C.N_PACKETS = 120

    def breakdown(net, model, *a, **k):
        global _select
        st = a[3]
        _select = st.wrap("select", _select)
        return PC2.pmr_stage_breakdown(net, model, *a, **k)
    C.stage_breakdown = breakdown
    TS.install()
    if rfs:
        from semkine import rfs as RF
        RF.install(RFS.TAU, RFS.NEAR_PX, packet=False)
        RF.install_stream()                               # TS.prepare_packet = the readings' route-first selection
    orig_check = ns.OUT / "costs" / f"{X.run_name('d1r_rol5', SEED)}__fast_check.json"
    native_win, native_make, native_load = ET._window_events, ET.make_eval_packet, NM.load
    pending = {}

    def window_events(events, offsets, tsub, end, window):
        X.require(int(window) == TS.WINDOW_MS, "streaming form: 50 ms update intervals only")
        pending["cut"] = (events, offsets, tsub, int(end))
        return native_win(events, offsets, tsub, end, window)

    def make_eval_packet(*args, **kwargs):
        X.require("cut" in pending, "streaming form: a packet without its window cut")
        batch = native_make(*args, **kwargs)
        cut = pending.pop("cut")
        if select == "staging":
            batch.str_state = TS.prepare_packet(batch, *cut)
            _S["nodes"].append(int(batch.str_state["nodes"]))
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
                _select(batch)
            return forward(batch, *a, **k)
        net.model.forward_packet = forward_packet
        X.require(str(SM._STATE.get("checkpoint")) == router_ck, "loaded router is not the eval config's")
        if rfs:
            _S["native_sources"] = XAG.sample_sources            # semkine/s45sm_fast.py predicted_sources
            check(net, C.OUT / "costs" / tag, n_check, rn)       # fastifies the trunk
            _use("ref" if ref else "dep")
        else:
            c = X.read_json(orig_check)
            X.require(c["run"] == X.run_name("d1r_rol5", SEED) and c["max_abs_diff"] < FAST_MAX,
                      "the packet run's fast check")
            _use("orig")
            F.fastify(net.model.xyz)
        return net
    ET._window_events, ET.make_eval_packet, NM.load = window_events, make_eval_packet, load
    if rfs:
        tag = f"{rn}__str" + ("__ref" if ref else "")
    else:
        tag = f"{rn}__str__{select}_rfsblk"
    if smoke:
        tag += "__" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    C.run(rn, "fast", str(spec), tag, "default")
    nodes, elig = _S["nodes"], _S["elig"]
    X.save_json(root / "costs" / tag / "stream.json",
                {"stream": True, "select": select, "route_first": rfs, "implementation": (
                    "readings" if ref else "deployable") if rfs else "tpd_str",
                 "router_checkpoint": router_ck, "router_checkpoint_sha256": sha(router_ck),
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
