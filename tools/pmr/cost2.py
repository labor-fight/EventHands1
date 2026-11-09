#!/usr/bin/env python3
"""PMR cost readings, engineering amendment 1 (docs/EXPERIMENT_RECORDS.md#pmr-amend1-20261109): tools/pmr/cost.py
unchanged for B2 and S43; for a PMR arm PFO's frozen harness (tools/pfo/cost.py) runs with its stage breakdown replaced
by the same timers on the PMR trunk's own modules (the frozen breakdown names B2's submodules rel / vin / edge / layers /
slot_proj / fc1 / fc2 and raises on the PMR trunk before the report is written). Parameters, MACs, row-form and
end-to-end latency and memory are the harness's own code, unchanged. A directory left by a failed earlier attempt of
the same tag (launch.json without report.json) is moved to costs/attempt_failed/<tag>_<utc>/ first (preserved), so the
reading keeps the tag tools/pmr/analyze.py reads. GPU 4 exclusive (the driver's `exclusive` kind).

    python -m tools.pmr.cost2 run --arm A [--seed S] --policy default|det
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr.common import OUT, SEED, require  # noqa: E402

POLICIES = ("default", "det")


def pmr_stage_breakdown(net, model, kind, impl, staged, st, forward, s43_exec, exe):
    """tools/pfo/cost.py `stage_breakdown` for a routed network, with the PMR trunk's modules as the trunk stages."""
    from semkine import encoder as ENC
    from semkine import s45sm_fast as FAST
    from semkine import xag as XAG
    from semkine import xyz_mesh as XM
    require(kind != "s43", "the PMR breakdown is for routed PMR networks")
    ENC.event_tokens = st.wrap("tokens", ENC.event_tokens)
    FAST.event_image = st.wrap("router_image", FAST.event_image)
    FAST.trunk = st.wrap("router_cnn", FAST.trunk)
    XAG.sample_sources = st.wrap("router_total", XAG.sample_sources)
    FAST.route = st.wrap("route_decode", FAST.route)
    model._fk = st.wrap("mano_fk", model._fk)
    XAG.route_association = st.wrap("pfa", XAG.route_association)
    XM.associate_rays = st.wrap("assoc_empty", XM.associate_rays)
    x = model.xyz
    st.module("xyz_total", x)
    x.node_inputs = st.wrap("node_inputs", x.node_inputs)          # geometry, statistics, phi and the pooling
    st.module("phi", x.phi)
    for name in ("enc_part", "enc_hub"):
        st.module("encode", getattr(x, name))
    if hasattr(x, "gates"):
        x.gates = st.wrap("gates", x.gates)
    if hasattr(x, "rounds"):
        x.propagate = st.wrap("propagate", x.propagate)
    if hasattr(x, "head_j1"):
        x.readout = st.wrap("readout", x.readout)
    for name in ("fc1", "fc2"):
        if hasattr(x, name):
            st.module("readout", getattr(x, name))
    st.on = True
    for it in staged:
        with st.time("step_total"):
            forward(it)
    st.on = False
    return {k: v / len(staged) for k, v in st.t.items()}


def archive_failed(tag):
    d = OUT / "costs" / tag
    if d.exists() and not (d / "report.json").exists():
        dst = OUT / "costs" / "attempt_failed" / f"{tag}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), str(dst))
        print("archived failed attempt", d, "->", dst, flush=True)


def run(arm, seed, policy):
    from tools.pfo import cost as C
    from tools.pmr import run as R
    C.OUT = OUT
    C.stage_breakdown = pmr_stage_breakdown
    name, sp = R.arm_spec(arm, seed)
    R.prepare(sp)
    tag = f"{name}__fast__{policy}"
    archive_failed(tag)
    C.run(name, "fast", str(sp), tag, policy)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True)
    r.add_argument("--seed", type=int, default=SEED)
    r.add_argument("--policy", required=True, choices=POLICIES)
    a = p.parse_args(argv)
    return run(a.arm, a.seed, a.policy)


if __name__ == "__main__":
    main()
