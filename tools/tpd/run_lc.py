#!/usr/bin/env python3
"""TPD problem 1 LC stage: router acceptance, identity probes, the zero-training swap into the frozen L1 trunk and the
checks (docs/EXPERIMENT_RECORDS.md#tpd-lc-prereg-20261105). Readings go through PFO's frozen engine (tools/pfo/engine.py,
tools/pfo/nets.py, unchanged) with the output root redirected to outputs/tpd/20261105 and LC behind the routing hooks
(semkine/tpd_lc.py `engine_hooks`).

    python -m tools.tpd.run_lc init                       lc_init = LC built from r_s2d (history columns zero), saved once
    python -m tools.tpd.run_lc accept --router R --phase P
    python -m tools.tpd.run_lc probe --router R --seq S   R in r_s2d / lc_init / lc / lc0: accuracy against the XAG labels
                                                          on every 2nd e1 endpoint (PCR's probe protocol): nodes both routed
                                                          (on-hand >= 0.5) and labelled; part / finger-group (palm = a group)
    python -m tools.tpd.run_lc read --router R --task T   the frozen IDP L1 trunk (seed 3407, step 6000) with R's routes;
                                                          tag l1__<R>
    python -m tools.tpd.run_lc check                      K_LC: lc_init's zgz_local probe equals r_s2d's (node count; part /
                                                          finger accuracy within 1e-3)
    python -m tools.tpd.run_lc check2                     K_LC2: l1__lc_init e1 RA within 0.05 mm of IDP's idp_l1_s3407 e1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.tpd.common import IDP_OUT, OUT, ZGZ, now, read, require, save, sha  # noqa: E402
from tools.tpd import router_lc as RL  # noqa: E402

LC_OUT = OUT / "lc"
L1_SPEC = IDP_OUT / "specs" / "idp_l1_s3407.json"
L1_DEPLOYED_E1 = IDP_OUT / "readings" / "idp_l1_s3407" / "e1"
ROUTERS = ("r_s2d", "lc_init", "lc", "lc0")
TASKS = ("e1", "e3", "pert", "f1")
INIT = OUT / "routers" / "lc_init" / "lc_init_final.ckpt"


def router_ckpt(name):
    if name == "lc_init":
        require(INIT.exists(), "lc_init not written")
        return INIT
    receipt = read(OUT / "receipts" / f"router_{name}.json")
    require(receipt["status"] == "PASS", "router not accepted: " + name)
    ck = RL.checkpoint(name)
    require(sha(ck) == receipt["checkpoint_sha256"], "router checkpoint changed since acceptance")
    return ck


def init():
    from semkine import tpd_lc as LC
    require(not INIT.exists(), "written once")
    INIT.parent.mkdir(parents=True, exist_ok=True)
    net = LC.from_r_s2d(RL.R_S2D)
    LC.save_checkpoint(net, INIT, init=str(RL.R_S2D), init_sha256=sha(RL.R_S2D), note="r_s2d, history columns zero")
    save(INIT.with_suffix(".json"), {"status": "COMPLETE", "checkpoint_sha256": sha(INIT), "written_utc": now()},
         exclusive=True)


def accept(name, phase):
    require(name in RL.ROUTERS, "router")
    out = OUT / "receipts" / f"router_{name}{'_preflight' if phase == 'preflight' else ''}.json"
    require(not out.exists(), "accepted once")
    d = RL.router_dir(name, phase)
    launch, done = read(d / "launch.json"), read(RL.checkpoint(name, phase).with_suffix(".json"))
    ck = RL.checkpoint(name, phase)
    import torch
    obj = torch.load(str(ck), map_location="cpu")
    ok = {"complete": done["status"] == "COMPLETE" and done["step"] == RL.STEPS[phase],
          "checkpoint": ck.exists() and done["checkpoint_sha256"] == sha(ck),
          "launch": launch["router"] == name and launch["phase"] == phase and launch["steps"] == RL.STEPS[phase]
          and launch["init_sha256"] == sha(RL.R_S2D) and launch["module_sha256"] == sha(ROOT / "semkine/tpd_lc.py")
          and launch["trainer_sha256"] == sha(ROOT / "tools/tpd/router_lc.py"),
          "meta": obj["kind"] == "tpd_lc" and obj["empty_history"] == RL.ROUTERS[name]["empty"] and obj["heldout"] == "ylf"
          and obj["hist_ms"] == 200 and obj["packet_ms"] == 50,
          "params": int(done["params"]) == 137380}
    receipt = {"router": name, "phase": phase, "checks": ok, "status": "PASS" if all(ok.values()) else "FAIL",
               "checkpoint_sha256": sha(ck), "accepted_at_utc": now()}
    save(out, receipt, exclusive=True)
    print(receipt["status"], ok, flush=True)
    require(receipt["status"] == "PASS", "acceptance failed")


def probe(router, seq):
    import torch
    import yaml
    from semkine import eval_track as ET
    from semkine import pfo_router as PR
    from semkine import s45sm as SM
    from semkine import s45sm_fast as FAST
    from semkine import tpd_lc as LC
    from semkine.xyz_mesh import stride_sample
    from tools.pfo.common import XAG_SOURCE_ROOT, require_one_gpu
    from tools.pfo.engine import endpoint_rows, sequences
    from tools.pfo.nets import sequence
    require_one_gpu()
    require(router in ROUTERS and seq in ZGZ, "router / seq")
    out_path = LC_OUT / "probes" / f"{router}__{seq}.json"
    require(not out_path.exists(), "probe written once")
    dev = torch.device("cuda")
    if router == "r_s2d":
        net, _ = PR.load(RL.R_S2D)
    else:
        net, obj = LC.load(router_ckpt(router))
        LC._DEPLOY["empty"] = bool(obj.get("empty_history", False))
    net = net.to(dev).eval()
    cfg = yaml.safe_load(Path(read(L1_SPEC)["config"]).read_text())
    cfg["DATA"]["XAG_SOURCE_ROOT"] = str(XAG_SOURCE_ROOT)
    root, seqs = sequences(cfg, (seq,))
    ctx = sequence(root, seqs[0][1], seq, dev)
    rows = endpoint_rows(ctx.aux)
    g = lambda q: torch.where(q == 0, torch.zeros_like(q), (q - 1) // 3 + 1)  # noqa: E731
    tot = {"n": 0, "part": 0, "finger": 0, "palm_finger": 0, "cross_finger": 0, "onhand_routed": 0}
    with torch.no_grad():
        for i in range(0, len(rows), 2):
            _, end, prev_ms, _ = rows[i]
            ev5 = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, int(end), 50)
            if len(ev5) == 0:
                continue
            b = ET.make_eval_packet(ev5, torch.zeros(1, 51, device=dev), ctx.betas, ctx.K, 50, dev)
            b = ET.attach_xag_sources(b, cfg, seq, ctx.offsets, int(end), 50, dev)
            src, mask = stride_sample(b.ptr, SM.MAX_NODES)
            if router == "r_s2d":
                out = FAST.router_forward(net, b.events, b.ptr, b.delta_t_s, src, mask)
            else:
                hist, hptr = LC.history_events(ctx, int(end), 50, net.hist_ms, dev)
                out = net(b.events, b.ptr, b.delta_t_s, hist, hptr)
            routed = mask & (torch.sigmoid(out["onhand"].float()) >= 0.5)
            face, _ = FAST.route(net, "sr", out, routed)
            lab = SM._NATIVE["sample_sources"](b, src, mask)
            F_ = net.faces.shape[0]
            lv = lab["xag_valid"] & mask & (lab["xag_face"] >= 0) & (lab["xag_face"] < F_)
            lb = lab["xag_bary"].float()
            lv = lv & torch.isfinite(lb).all(-1) & (lb >= 0).all(-1) & (lb.sum(-1) > 0)
            lf = lab["xag_face"].clamp(0, F_ - 1)
            both = routed & lv
            lp, rp = net.face_part[lf], net.face_part[face]
            lg, rg = g(lp), g(rp)
            tot["n"] += int(both.sum())
            tot["part"] += int((both & (lp == rp)).sum())
            tot["finger"] += int((both & (lg == rg)).sum())
            tot["palm_finger"] += int((both & (lg != rg) & ((lg == 0) | (rg == 0))).sum())
            tot["cross_finger"] += int((both & (lg != rg) & (lg > 0) & (rg > 0)).sum())
            tot["onhand_routed"] += int(routed.sum())
    n = max(tot["n"], 1)
    res = {"router": router, "seq": seq, "nodes": tot["n"], "onhand_routed": tot["onhand_routed"],
           "part_acc": tot["part"] / n, "finger_acc": tot["finger"] / n, "palm_finger_err": tot["palm_finger"] / n,
           "cross_finger_err": tot["cross_finger"] / n, "written_utc": now()}
    save(out_path, res, exclusive=True)
    print(json.dumps(res), flush=True)


def prepare(sp):
    import yaml
    s = read(sp)
    cfg = yaml.safe_load(Path(s["config"]).read_text())
    require(cfg["S45SM"]["ROUTER_KIND"] == "pfr", "L1's evaluation config names r_s2d")
    from semkine import pfo_router as PR
    PR.register()
    from tools.pfo2.train import install_trunk
    install_trunk(s["module"], s["trunk"])
    return s


def reading(router, task):
    from semkine import tpd_lc as LC
    from tools.pfo import engine as E
    require(router in ("lc_init", "lc", "lc0") and task in TASKS, "router / task")
    sp = L1_SPEC
    s = read(sp)
    require(sha(s["checkpoint"]) == s["checkpoint_sha256"], "L1 checkpoint changed")
    prepare(sp)
    ck = router_ckpt(router)
    route = LC.install(ck)
    if router != "lc_init":
        import torch
        LC._DEPLOY["empty"] = bool(torch.load(str(ck), map_location="cpu")["empty_history"])
        route["empty_history"] = LC._DEPLOY["empty"]
    LC.engine_hooks()
    E.OUT = OUT
    out = E.run(f"l1__{router}", task, None, None, (), str(sp))
    save(Path(out) / "lc_route.json", {**route, "checkpoint_sha256": sha(ck), "trunk_spec": str(sp),
                                        "written_utc": now()}, exclusive=True)


def check():
    a, b = read(LC_OUT / "probes" / "lc_init__zgz_local.json"), read(LC_OUT / "probes" / "r_s2d__zgz_local.json")
    ok = {"nodes": a["nodes"] == b["nodes"], "part": abs(a["part_acc"] - b["part_acc"]) <= 1e-3,
          "finger": abs(a["finger_acc"] - b["finger_acc"]) <= 1e-3}
    receipt = {"check": "k_lc", "lc_init": a, "r_s2d": b, "rows": ok, "status": "PASS" if all(ok.values()) else "FAIL",
               "checked_at_utc": now()}
    save(OUT / "receipts" / "k_lc.json", receipt, exclusive=True)
    print(receipt["status"], ok, flush=True)
    require(receipt["status"] == "PASS", "LC init does not route like r_s2d")


def check2():
    a = read(OUT / "readings" / "l1__lc_init" / "e1" / "report.json")["overall"]["mpjpe_ra_mm"]
    b = read(L1_DEPLOYED_E1 / "report.json")["overall"]["mpjpe_ra_mm"]
    ok = abs(a - b) <= 0.05
    receipt = {"check": "k_lc2", "l1__lc_init_ra": a, "idp_l1_deployed_ra": b, "status": "PASS" if ok else "FAIL",
               "checked_at_utc": now()}
    save(OUT / "receipts" / "k_lc2.json", receipt, exclusive=True)
    print(receipt["status"], a, b, flush=True)
    require(ok, "the LC reading path does not reproduce L1 + r_s2d")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    sub.add_parser("init")
    s = sub.add_parser("accept")
    s.add_argument("--router", required=True, choices=tuple(RL.ROUTERS))
    s.add_argument("--phase", required=True, choices=tuple(RL.STEPS))
    s = sub.add_parser("probe")
    s.add_argument("--router", required=True, choices=ROUTERS)
    s.add_argument("--seq", required=True, choices=ZGZ)
    s = sub.add_parser("read")
    s.add_argument("--router", required=True, choices=("lc_init", "lc", "lc0"))
    s.add_argument("--task", required=True, choices=TASKS)
    sub.add_parser("check")
    sub.add_parser("check2")
    a = p.parse_args(argv)
    if a.verb == "init":
        return init()
    if a.verb == "accept":
        return accept(a.router, a.phase)
    if a.verb == "probe":
        return probe(a.router, a.seq)
    if a.verb == "read":
        return reading(a.router, a.task)
    return check() if a.verb == "check" else check2()


if __name__ == "__main__":
    main()
