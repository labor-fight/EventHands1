#!/usr/bin/env python3
"""T1A-UB rounds >= 4: the frozen round-1 training wrapper (tools/t1aub/train.py, unchanged) plus training-only mechanisms
that the round-1 registry fields cannot express, declared per arm in tools/t1aub/train2_r<k>.json ({"round": k, "arms":
{arm: {"n2": lambda}}}):
  n2   semkine/s45tr_n2.py: the same packet fed a second time with the first prediction (gradient stopped) as the prior,
       the second prediction also scored against the target, loss = L1 + lambda L2 (S45-TR N2); evaluation unchanged.
The arm itself is a round-1-format arm of tools/t1aub/registry_r<k>.json: its config, packets, initial tensors and
acceptance are the frozen layer's; the mechanism is not in the config, so this wrapper writes `train2_binding.json` in the
run directory and a formal run requires its preflight's binding to name the same mechanism. Lightning re-executes this file
for DDP rank 1 (sys.argv[0]), so both ranks install the mechanism before the model is built.

    python tools/t1aub/train2.py --arm A --phase {preflight,formal} --config C
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]


def mechanisms(arm):
    from tools.t1aub import common as C
    found = {}
    for f in sorted(C.PACKAGE.glob("train2_r*.json")):
        r = json.loads(f.read_text())
        for name, spec in (r.get("arms") or {}).items():
            C.require(name not in found, "mechanism declared twice: " + name)
            C.require(set(spec) <= {"n2", "note"}, "unknown train2 mechanism in " + name)
            found[name] = {**spec, "round": int(r["round"])}
    C.require(arm in found, "no train2 mechanism declared for " + arm)
    C.require(found[arm]["round"] == C.arm_spec(arm)["round"], "train2 mechanism and arm registered in different rounds")
    return {k: v for k, v in found[arm].items() if k not in ("note", "round")}


def main():
    from tools.t1aub import common as C
    arm = os.environ.get("T1AUB_ARM")
    phase = os.environ.get("T1AUB_PHASE")
    if "--arm" in sys.argv:
        arm = sys.argv[sys.argv.index("--arm") + 1]
    if "--phase" in sys.argv:
        phase = sys.argv[sys.argv.index("--phase") + 1]
    mech = mechanisms(arm)
    binding = {"arm": arm, "phase": phase, "mechanisms": mech, "train2_sha256": C.sha(__file__),
               "n2_module_sha256": C.sha(ROOT / "semkine/s45tr_n2.py")}
    if phase == "formal":
        pre = C.read(C.output_dir(arm, "preflight") / "train2_binding.json")
        C.require(pre["mechanisms"] == mech and pre["train2_sha256"] == binding["train2_sha256"],
                  "the accepted preflight ran another train2 mechanism or wrapper")
    if "n2" in mech:
        from semkine import s45tr_n2
        s45tr_n2.install(float(mech["n2"]))
    if os.environ.get("LOCAL_RANK") in (None, "0"):
        out = C.output_dir(arm, phase)

        import tools.t1aub.train as W
        native_claim = W.fresh_claim

        def claim(options, cfg):
            native_claim(options, cfg)
            C.save(out / "train2_binding.json", {**binding, "recorded_at_utc": C.now()}, exclusive=True)
        W.fresh_claim = claim
    from tools.t1aub import train as W
    W.main()


if __name__ == "__main__":
    main()
