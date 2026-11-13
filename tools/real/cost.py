#!/usr/bin/env python3
"""Real-study cost block (docs/EXPERIMENT_RECORDS.md#real-prereg-20261110 "成本列", #real-amend5-20261111): the frozen cost
tools on the real canvas, written under outputs/real/20261110/costs/<tag>/ (read by tools/real/table.py / table2.py; the
EvRealHands tables reuse them: same networks on the same tensors). Packets: the first 600 50 ms packets of the real
lyq_local recording (a training recording, as the CMP cost rule's lyq_local), prev = GT at the run start.

  cmp   tools/cmp/cost.py `run` unchanged inside the real study's namespace (tools/real/ns.py; the evaluator's frame set to
        the arm's canvas): EventHands (346 x 260) and FastMETRO-Event (346 x 260 padded to 352 x 288)
  evht  tools/evhandtrack/cost2.py's procedure (PFO's harness + PMR's stage breakdown; --fast = semkine/pmr_fast.py after
        the reference-vs-fast check on the packet run; the streaming form's hooks) on EvHandTrack's real run
        pmr_d1r_rol5_s3407 (step 1500, the accepted checkpoint and its reading spec), with the router frame of the real
        canvas (tools/real/evht.py `enter`)

    python tools/real/cost.py cmp --arm eho|fme [--seed 3407] [--policy default]
    python tools/real/cost.py evht --net packet|stream [--fast] [--policy default]
One GPU among 4-7 with nothing else running on it.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

SEED = 3407
ARM = "d1r_rol5"


def cmp_cost(arm, seed, policy):
    import yaml
    from tools.real import ns
    ns.enter()
    from tools.cmp import cost as C
    from tools.cmp.arms import config_path
    ns.set_frame(yaml.safe_load(Path(config_path(arm, seed)).read_text()))
    C.run(arm, seed, policy)


def evht_cost(net, fast, policy):
    from tools.real import evht as X
    from tools.real import ns
    X.enter()
    from tools.pmr import arms_any6  # noqa: F401
    from semkine import rdr_hooks as RH
    from tools.pmr import ablrt as RT
    RH.enable()
    RT.patch_loader()
    X.patch_engine()                                      # lyq_local is looked up in the real split manifest
    from tools.cmp.common import now, sha
    from tools.evhandtrack import cost2 as EC2
    from tools.pfo import cost as C
    from tools.pfo import nets as NM
    from tools.pmr import cost2 as PC2
    from tools.pmr import run as R
    out = ns.OUT
    C.OUT = out
    rn = X.run_name(ARM, SEED)
    spec = X.OUT / "specs" / f"{rn}.json"
    sp = X.read_json(spec)
    rec = X.read_json(X.OUT / "receipts" / f"{rn}.json")
    X.require(rec["status"] == "PASS" and sha(sp["checkpoint"]) == sp["checkpoint_sha256"]
              == rec["checkpoint_sha256_by_step"]["1500"], rn + ": checkpoint changed since acceptance")
    R.prepare(spec)
    C.stage_breakdown = PC2.pmr_stage_breakdown
    check = out / "costs" / f"{rn}__fast_check.json"
    if fast:
        from semkine import pmr_fast as F
        if net == "stream":
            c = X.read_json(check)
            X.require(c["run"] == rn and c["max_abs_diff"] < EC2.FAST_MAX, "the packet run's fast check")
        native = NM.load

        def load(*a, **k):
            n = native(*a, **k)
            if net == "packet":
                EC2.check_path = lambda _out: check
                d = EC2.fast_check(n, out, rn)
                X.require(d < EC2.FAST_MAX, f"fast path differs from the reference by {d}")
            F.fastify(n.model.xyz)
            return n
        NM.load = load
    if net == "stream":
        from semkine import tpd_str as TS
        from tools.evhandtrack import cost as EC
        TS.install()
        EC.stream_cost_hooks()
    tag = f"{rn}__{'str' if net == 'stream' else 'packet'}" + ("" if fast else "__ref") + (
        "" if policy == "default" else f"__{policy}")
    C.run(rn, "fast", str(spec), tag, policy)
    if net == "stream":
        nodes = EC._NODES
        X.save_json(out / "costs" / tag / "stream.json",
                    {"stream": True, "budget": TS.BUDGET, "window_ms": TS.WINDOW_MS, "n_packets_built": len(nodes),
                     "nodes_mean": sum(nodes) / max(len(nodes), 1), "nodes_max": max(nodes, default=0),
                     "written_utc": now()}, exclusive=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    c = sub.add_parser("cmp")
    c.add_argument("--arm", required=True, choices=("eho", "fme"))
    c.add_argument("--seed", type=int, default=SEED)
    c.add_argument("--policy", default="default", choices=("default", "det"))
    e = sub.add_parser("evht")
    e.add_argument("--net", required=True, choices=("packet", "stream"))
    e.add_argument("--fast", action="store_true")
    e.add_argument("--policy", default="default", choices=("default", "det"))
    a = ap.parse_args(argv)
    if a.verb == "cmp":
        cmp_cost(a.arm, a.seed, a.policy)
    else:
        evht_cost(a.net, a.fast, a.policy)


if __name__ == "__main__":
    main()
