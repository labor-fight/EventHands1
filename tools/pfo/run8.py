#!/usr/bin/env python3
"""PFO router swaps on a composed-trunk arm (tools/pfo/arms3.py arms trained by tools/pfo/train4.py): the arm's accepted
step-6000 trunk read with another registered PFO router at test (zero training), through the frozen engine.

    python -m tools.pfo.run8 swap --arm A --seed S --router R --task T

Evaluation config = the arm's evaluation config (tools/pfo/run4.py `eval_config`) with the routing block pointing at the
router (kind "pfr"), written once under configs/pfo_20261103/swap/. Reading name `pfo_<arm>_s<seed>+<router>`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

import yaml  # noqa: E402

from tools.pfo import arms3 as A3  # noqa: E402
from tools.pfo import run4 as R4  # noqa: E402
from tools.pfo.arms import run_name  # noqa: E402
from tools.pfo.common import CONFIGS, OUT, read, require, save, sha  # noqa: E402

R4.ARMS2, R4.build_config, R4.spec = A3.ARMS3, A3.build_config, A3.spec


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("swap")
    s.add_argument("--arm", required=True)
    s.add_argument("--seed", required=True, type=int)
    s.add_argument("--router", required=True)
    s.add_argument("--task", required=True)
    a = p.parse_args(argv)
    from tools.pfo import routers3 as R3
    from tools.pfo.router_train import checkpoint
    require(a.router in R3.ROUTERS3, "unregistered PFO router")
    ck = checkpoint(a.router)
    done = read(ck.with_suffix(".json"))
    require(done["status"] == "COMPLETE" and done["checkpoint_sha256"] == sha(ck), "router not complete")
    base = read(R4.spec_file(a.arm, a.seed, 6000))
    cfg = yaml.safe_load(Path(base["config"]).read_text())
    name = run_name(a.arm, a.seed)
    cfg["S45SM"] = {"ARM": f"{name}+{a.router}", "ROUTER": a.router, "ROUTER_KIND": "pfr", "ROUTER_CHECKPOINT": str(ck),
                    "ASSOC": "pfa", "ONHAND_TAU": 0.5, "FAST": True}
    path = CONFIGS / "swap" / f"{name}+{a.router}.yaml"
    text = yaml.safe_dump(cfg, sort_keys=False)
    if path.exists():
        require(path.read_text() == text, "swap config differs")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    spec = {"kind": "routed", "checkpoint": base["checkpoint"], "config": str(path), "trunk": base["trunk"]}
    sp = OUT / "specs" / f"{name}+{a.router}.json"
    if sp.exists():
        require(read(sp) == spec, "spec differs")
    else:
        save(sp, spec)
    from semkine import pfo_router as PR
    from semkine import pfo_trunk4 as PT
    from tools.pfo.train4 import trunk4_variant
    PR.register()
    PT.install(trunk4_variant(base["trunk"]))
    from tools.pfo.engine import run
    run(f"{name}+{a.router}", a.task, None, None, (), str(sp))


if __name__ == "__main__":
    main()
