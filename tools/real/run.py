#!/usr/bin/env python3
"""Real-study readings (docs/EXPERIMENT_RECORDS.md#real-prereg-20261110): CMP's accept and PFO's frozen E1 engine with
the CMP adapters, inside the real study's namespace.

    python tools/real/run.py accept --arm A --seed S
    python tools/real/run.py e1 --arm A --seed S [--form main] [--phase smoke --limit N]

e1 reads the test sequences (kqj, lqh: global / local x normal / dark) and then the dev sequences (wh) in one pass, in
that fixed order (the engine's shared initial-noise rng runs over the sequences in order); the engine's GT + initial
noise at every segment start and its 50 ms grid are unchanged. Two engine globals are replaced in this process only: the
sequence list (ZGZ -> test + dev, endpoint counts unchecked, sequences looked up in val / test / train of
splits_real.json) and the "one GPU among 2-7" check, which also accepts the single borrowed card gpu-borrow hands over
(CUDA_VISIBLE_DEVICES = its UUID). --phase smoke reads a smoke checkpoint (no receipt) into outputs/real/20261110/smoke/.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.real import ns  # noqa: E402

ns.enter()

from tools.cmp import run as R  # noqa: E402
from tools.cmp.arms import ARMS, checkpoint, config_path  # noqa: E402
from tools.cmp.common import OUT, read, require, save, sha  # noqa: E402


def patch_engine():
    from tools.pfo import common as PC
    from tools.pfo import engine as E
    from tools.pfo import nets as PN
    from tools.cmp import nets as CN
    E.OUT = PC.OUT = OUT
    PN.load = CN.load
    E.ZGZ = ns.TEST + ns.DEV
    E.COUNTS = {}
    native_require = E.require

    def require_(cond, msg):
        if msg == "one GPU among 2-7" and os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-") \
                and "," not in os.environ["CUDA_VISIBLE_DEVICES"]:
            return
        native_require(cond, msg)
    E.require = require_

    def sequences(cfg, names):
        from semkine.dataset import sequences_for_split, splits_manifest
        root = Path(cfg["DATA"]["ROOT"])
        require(root in [c["root"] for c in ns.CANVAS.values()], "not a real data root")
        found = {}
        for split in ("val", "test", "train"):
            found.update(dict(sequences_for_split(root, split, splits_manifest(cfg))))
        require(all(s in found for s in names), "missing sequences")
        return root, [(s, found[s]) for s in names]
    E.sequences = sequences
    return E


def e1_fme(seed, form="main", phase="formal", limit=None):
    """FastMETRO-Event (amendment 2): its own frame-wise reading (tools/cmp/ext_fme.py `e1`, mesh output) on the same
    test + dev endpoint grid; the engine globals it reads at call time (sequence list, endpoint counts, sequence lookup)
    and its "one GPU among 4-7" check (which also accepts the single borrowed card) are replaced in this process."""
    import numpy as np
    import yaml
    from tools.cmp import ext_fme as F
    from tools.pfo import common as PC
    from tools.xcl.evaluate import endpoint_grid
    E = patch_engine()
    cfg = yaml.safe_load(Path(config_path("fme", seed, phase)).read_text())
    ns.set_frame(cfg)
    names = ns.TEST + ns.DEV
    root, seqs = E.sequences(cfg, names)
    PC.ZGZ = names
    PC.COUNTS = {s: len(endpoint_grid(np.load(root / d / f"{s}_aux.npz", allow_pickle=True))) for s, d in seqs}
    native_require = F.require

    def require_(cond, msg):
        if msg == "one GPU among 4-7" and os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-") \
                and "," not in os.environ["CUDA_VISIBLE_DEVICES"]:
            return
        native_require(cond, msg)
    F.require = require_
    return F.e1("fme", seed, form, smoke=phase == "smoke", limit=limit)


def e1(arm, seed, form="main", phase="formal", limit=None):
    if arm == "fme":
        return e1_fme(seed, form, phase, limit)
    E = patch_engine()
    model = ARMS[arm]["model"]
    if phase == "formal":
        rec = read(OUT / "receipts" / f"cmp_{arm}_s{seed}.json")
        require(rec["status"] == "PASS" and sha(rec["checkpoint"]) == rec["checkpoint_sha256"], "unaccepted checkpoint")
        ck, cfg = rec["checkpoint"], rec["config"]
        spec_dir = OUT / "specs"
    else:
        from tools.cmp.arms import SMOKE_STEPS
        ck, cfg = str(checkpoint(arm, seed, SMOKE_STEPS, "smoke")), str(config_path(arm, seed, "smoke"))
        require(Path(ck).exists(), "missing smoke checkpoint " + ck)
        spec_dir = OUT / "smoke" / "specs"
        os.environ["PFO_SMOKE_DIR"] = str(OUT / "smoke")
    import yaml
    ns.set_frame(yaml.safe_load(Path(cfg).read_text()))
    spec = {"kind": "cmp", "model": model, "checkpoint": ck, "config": cfg, "form": form}
    tag = R.tag(arm, seed, form) + ("" if phase == "formal" else "_smoke")
    sp = save(spec_dir / f"{tag}.json", spec)
    E.run(tag, "e1", spec=str(sp), limit=limit)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("accept", "e1"))
    p.add_argument("--arm", required=True, choices=ns.ARMS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--form", default="main")
    p.add_argument("--phase", default="formal", choices=("formal", "smoke"))
    p.add_argument("--limit", type=int)
    a = p.parse_args(argv)
    if a.verb == "accept":
        R.accept(a.arm, a.seed)
    else:
        require((a.phase == "smoke") == (a.limit is not None), "--limit only (and always) with --phase smoke")
        e1(a.arm, a.seed, a.form, a.phase, a.limit)


if __name__ == "__main__":
    main()
