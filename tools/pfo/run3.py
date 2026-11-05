#!/usr/bin/env python3
"""PFO router swaps (docs/EXPERIMENT_RECORDS.md#pfo-p2r-prereg-20261103): a trained trunk read with another events-only
router at test time (zero training), through the frozen phase-1 engine.

    python -m tools.pfo.run3 swap --trunk pfasrdz|<arm>:<seed> --router <PFO router> --task T

The evaluation config is the trunk's own config with the routing block pointing at the router (kind "pfr", registered
first; hard PFA; threshold 0.5), written once under configs/pfo_20261103/swap/. Reading name `<trunk>+<router>`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

import yaml  # noqa: E402

from tools.pfo.common import CONFIGS, NETS, OUT, read, require, save, sha  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("swap")
    s.add_argument("--trunk", required=True)
    s.add_argument("--router", required=True)
    s.add_argument("--task", required=True)
    a = p.parse_args(argv)
    from tools.pfo.router_train import ROUTERS, checkpoint
    require(a.router in ROUTERS, "unregistered PFO router")
    ck = checkpoint(a.router)
    done = read(ck.with_suffix(".json"))
    require(done["status"] == "COMPLETE" and done["checkpoint_sha256"] == sha(ck), "router not complete")
    trunk_variant = None
    if ":" in a.trunk:
        from tools.pfo.arms import ARMS, run_name
        from tools.pfo.run2 import spec_file
        arm, seed = a.trunk.split(":")
        base_spec = read(spec_file(arm, int(seed), 6000))
        trunk_variant = ARMS[arm]["trunk"]
        name = run_name(arm, int(seed))
    else:
        require(a.trunk in NETS and NETS[a.trunk]["kind"] == "routed", "a routed trunk")
        base_spec = {k: str(v) for k, v in NETS[a.trunk].items()}
        name = a.trunk
    cfg = yaml.safe_load(Path(base_spec["config"]).read_text())
    cfg["S45SM"] = {"ARM": f"{name}+{a.router}", "ROUTER": a.router, "ROUTER_KIND": "pfr",
                    "ROUTER_CHECKPOINT": str(ck), "ASSOC": "pfa", "ONHAND_TAU": 0.5, "FAST": True}
    path = CONFIGS / "swap" / f"{name}+{a.router}.yaml"
    text = yaml.safe_dump(cfg, sort_keys=False)
    if path.exists():
        require(path.read_text() == text, "swap config differs")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    spec = {"kind": "routed", "checkpoint": base_spec["checkpoint"], "config": str(path), "trunk": trunk_variant}
    sp = OUT / "specs" / f"{name}+{a.router}.json"
    if sp.exists():
        require(read(sp) == spec, "spec differs")
    else:
        save(sp, spec)
    from semkine import pfo_router as PR
    PR.register()
    if trunk_variant:
        from semkine import pfo_trunk as PT
        PT.install(trunk_variant)
    from tools.pfo.engine import run
    run(f"{name}+{a.router}", a.task, None, None, (), str(sp))


if __name__ == "__main__":
    main()
