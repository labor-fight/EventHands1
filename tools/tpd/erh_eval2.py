#!/usr/bin/env python3
"""TPD EvRealHands readings, engineering amendment 1 (docs/EXPERIMENT_RECORDS.md#tpd-erh-amend1-20261106):
tools/tpd/erh_eval.py unchanged except where the causal rol5 network comes from. Seeds 3408 / 3409 were trained and read
inside amendment 3's namespace, so their evaluation configs carry its keys and the frozen runner, which regenerates the
config in the plain ROL namespace, is refused by the write-once check. Here every seed's network is the spec its zgz
readings used (outputs/tpd/20261105/specs/tpd_rol5_s<seed>.json), its checkpoint checked against the acceptance receipt.

    python -m tools.tpd.erh_eval2 run --net rol5c --seed S --tier 3d [--mask gt]
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.tpd import erh_eval as E  # noqa: E402
from tools.tpd.common import OUT, read, require, sha  # noqa: E402

STEP = "1500"
_native = E.load_net


def load_net(net, seed, arm=None, device=None):
    if net != "rol5c":
        return _native(net, seed, arm, device)
    from semkine import tpd_causal as TC
    from tools.pfo import nets as N
    from tools.tpd import ns_seed
    ns_seed.enter()                                          # before tools.tpd.arms_rol is imported (seed 3409)
    from tools.tpd import run_rol as RR
    sp = OUT / "specs" / f"tpd_rol5_s{seed}.json"
    s = read(sp)
    receipt = read(OUT / "receipts" / f"tpd_rol5_s{seed}.json")
    require(receipt["status"] == "PASS" and sha(s["checkpoint"]) == receipt["checkpoint_sha256_by_step"][STEP]
            and s["checkpoint_sha256"] == receipt["checkpoint_sha256_by_step"][STEP], "rol5 checkpoint not the accepted one")
    RR.prepare(sp)
    TC.install()
    tag = f"tpd_rol5_s{seed}__causal"
    net_ = N.load(tag, device=device, spec=E._spec_obj(sp))
    TC.retoken()
    return tag, net_, True


E.load_net = load_net

if __name__ == "__main__":
    sys.exit(E.main())
