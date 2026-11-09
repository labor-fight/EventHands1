#!/usr/bin/env python3
"""TPD cost block (docs/EXPERIMENT_RECORDS.md#tpd-prereg-20261105, deliverable: cost table) through PFO's frozen cost
harness (tools/pfo/cost.py: the first 600 lyq_local packets, parameters, executed Linear / Conv MACs on the actual
tensors, MANO FK apart, row-form and end-to-end latency, memory) with its output root redirected to
outputs/tpd/20261105/costs/<tag>/; every job exclusive on GPU 4.

    python -m tools.tpd.cost run --net s43 --impl exec|dense --policy default|det
    python -m tools.tpd.cost run --net b2 --policy ...             B2 (PFO2 fgs 3407, step 6000), B2's node sampler
    python -m tools.tpd.cost run --net b2c --policy ...            the same weights in the causal deployment form
    python -m tools.tpd.cost run --arm A --seed S --policy ...     a TPD synthesis arm (accepted step 6000) in its
                                                                   deployment form (causal arms: the causal form)
    python -m tools.tpd.cost jobs --name N --after KEY [...] [--arm A:S ...] --out FILE
        the block: PyTorch-default flags first (the table's policy), then the deterministic ones; S43 sparse executor,
        the arms, B2 causal, B2, S43 masked dense; chained one after another after the given jobs

The causal form: the harness cuts each packet from the raw arrays (`semkine.eval_track._window_events` +
`make_eval_packet`) and calls `model.forward_packet`, outside PFO's engine, so the engine hooks of semkine/tpd_causal.py
do not fire. Here every packet carries its own stride k = max(1, ceil(n_prev / 2048)) (n_prev = events in the 50 ms
before the packet, `semkine.tpd_causal.n_prev_events`), computed in the host part of the step (with the window cut, so
the end-to-end latency includes it) and set just before the trunk runs (the staged packets of the row form keep their
own k). Tags final__<name>__<impl>__<policy>; the causal rows add `causal.json` (the k of every packet).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.tpd.common import OUT, PY, now, require, save  # noqa: E402

POLICIES = ("default", "det")
_KS = []


def causal_cost_hooks():
    """Per-packet k for the cost harness's own packet builder; `_KS` records the k of every packet built."""
    from semkine import eval_track as ET
    from semkine import tpd_causal as TC
    from tools.pfo import nets as NM
    native_win, native_make, native_load = ET._window_events, ET.make_eval_packet, NM.load
    pending = {}

    def window_events(events, offsets, tsub, end, window):
        require(int(window) == TC.PACKET_MS, "causal form: 50 ms packets only")
        pending["k"] = int(TC.k_from_prev(TC.n_prev_events(offsets, end)))
        return native_win(events, offsets, tsub, end, window)

    def make_eval_packet(*args, **kwargs):
        require("k" in pending, "causal form: a packet without its window cut")
        batch = native_make(*args, **kwargs)
        batch.causal_k = pending.pop("k")
        _KS.append(batch.causal_k)
        return batch

    def load(*args, **kwargs):
        net = native_load(*args, **kwargs)
        TC.retoken()
        forward = net.model.forward_packet

        def forward_packet(batch, *a, **k):
            TC.set_k([batch.causal_k])
            return forward(batch, *a, **k)
        net.model.forward_packet = forward_packet
        return net
    ET._window_events, ET.make_eval_packet, NM.load = window_events, make_eval_packet, load


def run(net, impl, policy, arm=None, seed=None):
    from tools.pfo import cost as C
    C.OUT = OUT
    causal = False
    if net == "s43":
        require(impl in ("exec", "dense"), "S43 implementations: exec | dense")
        name, sp = "s43_s3407", None
    elif net in ("b2", "b2c"):
        from tools.pfo2 import run as R
        from tools.pfo2 import run2 as R2
        name, sp = R2.arm_spec("fgs", 3407, 6000)
        R.prepare(sp)
        impl, causal = "fast", net == "b2c"
        name += "__causal" if causal else ""
    else:
        from tools.tpd import arms_final as A
        from tools.tpd import run_final as RF
        from tools.tpd import run_rol as RR
        name, sp = RF.arm_spec(arm, seed)
        RR.prepare(sp)
        impl, causal = "fast", bool(A.ARMS[arm]["causal"])
    if causal:
        from semkine import tpd_causal as TC
        TC.install()
        causal_cost_hooks()
    tag = f"final__{name}__{impl}__{policy}"
    C.run(name, impl, str(sp) if sp else None, tag, policy)
    if causal:
        ks = list(_KS)
        save(OUT / "costs" / tag / "causal.json",
             {"causal": True, "cap": TC.CAP, "packet_ms": TC.PACKET_MS, "n_packets_built": len(ks),
              "k_hist": {str(k): ks.count(k) for k in sorted(set(ks))}, "written_utc": now()}, exclusive=True)


def jobs(name, after, arms):
    out, last = [], None
    for policy in POLICIES:
        cmds = [("s43:exec", ["run", "--net", "s43", "--impl", "exec"])]
        for item in arms:
            arm, seed = item.split(":")
            cmds.append((f"{arm}:s{seed}", ["run", "--net", "arm", "--arm", arm, "--seed", seed]))
        cmds += [("b2c", ["run", "--net", "b2c"]), ("b2", ["run", "--net", "b2"]),
                 ("s43:dense", ["run", "--net", "s43", "--impl", "dense"])]
        for label, args in cmds:
            key = f"{name}:cost:{label}:{policy}"
            out.append({"key": key, "kind": "exclusive", "command": [PY, "-m", "tools.tpd.cost", *args, "--policy", policy],
                        "deps": [last] if last else list(after), "mem_gib": 15, "note": "TPD final cost block"})
            last = key
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--net", required=True, choices=("s43", "b2", "b2c", "arm"))
    r.add_argument("--impl", default="fast")
    r.add_argument("--arm")
    r.add_argument("--seed", type=int)
    r.add_argument("--policy", required=True, choices=POLICIES)
    j = sub.add_parser("jobs")
    j.add_argument("--name", required=True)
    j.add_argument("--after", action="append", required=True)
    j.add_argument("--arm", action="append", default=[])
    j.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.verb == "run":
        require((a.net == "arm") == (a.arm is not None and a.seed is not None), "--arm and --seed go with --net arm")
        return run(a.net, a.impl, a.policy, a.arm, a.seed)
    o = Path(a.out)
    require(not o.exists(), "job file exists")
    J = jobs(a.name, a.after, a.arm)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(J, indent=1) + "\n")
    print(len(J), "jobs ->", o)


if __name__ == "__main__":
    main()
