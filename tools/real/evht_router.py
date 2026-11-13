#!/usr/bin/env python3
"""EvHandTrack's router on the real data (docs/EXPERIMENT_RECORDS.md#real-amend5-20261111): r_s2d's recipe
(tools/pfo/router_train.py, frozen: the slim SRNet semkine/pfo_router.py with stem "s2d", 137,252 parameters; sr1's
recipe through tools/s45sm/router_train.py -- S45's training data contract and augmentation in depth scale mode, the XAG
per-event sources, two-GPU DDP x 512 packets, Adam 1e-3, 500-step warm-up then constant, BF16, seed 3407, 6000 steps;
loss = part CE + point smooth-L1 + 0.5 on-hand BCE) on the real training data: S45's config with the real data root,
split manifest and canvas n346 (346 x 260; the domain-randomisation shift scaled by 346 / 240 as for every arm of this
study), the router image at that canvas (tools/real/evht.py `enter`), the real XAG sources (tools/real/xag_build.py),
and all 12 training subjects (the simulated r_s2d held one of its nine subjects out for a router-only readout this study
does not make; every EvHandTrack trunk then trains on the 12 subjects with it). Fresh-only.

    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=a,b python tools/real/evht_router.py --phase smoke|formal
Lightning relaunches rank 1 with this same argv.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.real import evht as X  # noqa: E402
from tools.real import ns  # noqa: E402

X.enter()                                                          # before tools.s45sm.router_train binds the frame

STEPS = {"smoke": 20, "formal": 6000}
DESIGN = {"stem": "s2d"}                                           # tools/pfo/router_train.py ROUTERS["r_s2d"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=tuple(STEPS))
    ap.add_argument("--workers", type=int, default=16, help="data workers per rank (the recipe's loader: 16)")
    a = ap.parse_args()
    import json
    import pytorch_lightning as pl
    import torch
    from config import load_config
    from semkine import pfo_router as PR
    from semkine import s45sm as SM
    from tools.s45sm import router_train as S
    from tools.s45sm.paths import S45_CONFIG
    from tools.cmp.common import now, sha
    X.require((SM.HEIGHT, SM.WIDTH) == (X.H, X.W) and (S.RT_H, S.RT_W) == (X.H, X.W), "router frame")
    out = X.ROUTER_DIR if a.phase == "formal" else ns.OUT / "routers" / "r_s2d_smoke"
    target = out / "r_s2d_final.ckpt"
    rank0 = os.environ.get("LOCAL_RANK") in (None, "0")
    if rank0:
        X.require(not out.exists(), "fresh-only output already exists: " + str(out))
        out.mkdir(parents=True)
    pl.seed_everything(3407, workers=True)
    cfg = load_config(str(S45_CONFIG))
    cfg["AUG"]["DOMRAND"]["SCALE_MODE"] = "depth"                    # r_s2d's `dz`
    X.require(float(cfg["AUG"]["DOMRAND"]["SHIFT_PX"]) == ns.B2_SHIFT_PX, "S45's shift")
    cfg["AUG"]["DOMRAND"]["SHIFT_PX"] = float(cfg["AUG"]["DOMRAND"]["SHIFT_PX"]) * X.PIX
    cfg["DATA"].update(ROOT=str(ns.CANVAS[X.CANVAS]["root"]), SPLITS_MANIFEST=str(ns.splits(X.CANVAS)), HEIGHT=X.H,
                       WIDTH=X.W, XAG_SOURCE_ROOT=str(X.XAG))
    cfg["MODEL"].update(RENDER_H=X.H, RENDER_W=X.W, RENDER_SCALE=X.W / 640.0)
    import numpy as np
    from semkine.dataset import build_dataset
    components = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    ds = build_dataset(cfg, "train", components, train=True)
    subjects = sorted({s.split("_")[0] for s, _ in ds.sequences})
    import json as _json
    want = _json.loads(ns.splits(X.CANVAS).read_text())["train"]["trials"]
    X.require(sorted(s for s, _ in ds.sequences) == sorted(want), "the training split of the manifest")
    if a.workers == 16:
        loader = S._loader(ds)
    else:                                                          # S._loader with fewer workers (host memory)
        from torch.utils.data import DataLoader
        from semkine.events import collate_packets
        loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=a.workers, pin_memory=True,
                            persistent_workers=True, prefetch_factor=2, drop_last=True, collate_fn=collate_packets)
    torch.manual_seed(3407)
    net = PR.build(cfg["MANO"]["NPZ"], **DESIGN)
    module = S.SRModule(net, None, None, None)
    meta = {"heldout": None, "design": net.design, "kd": None, "router": "r_s2d", "data": "real n346",
            "section": X.SECTION}
    if rank0:
        (out / "launch.json").write_text(json.dumps({
            "router": "r_s2d", "phase": a.phase, "design": DESIGN, "steps": STEPS[a.phase], "subjects": subjects,
            "workers_per_rank": a.workers,
            "sequences": len(ds.sequences), "xag_sources": str(X.XAG), "frame": [X.W, X.H],
            "domrand_shift_px": cfg["AUG"]["DOMRAND"]["SHIFT_PX"], "trainer_sha256": sha(__file__),
            "recipe_sha256": {"tools/pfo/router_train.py": sha(ROOT / "tools/pfo/router_train.py"),
                              "tools/s45sm/router_train.py": sha(ROOT / "tools/s45sm/router_train.py"),
                              "semkine/pfo_router.py": sha(ROOT / "semkine/pfo_router.py")},
            "xag_build_summary_sha256": sha(X.XAG / "build_summary.json"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "started_utc": now()}, indent=1) + "\n")
    S._fit(module, loader, out, STEPS[a.phase], S.Finish(target, a.phase, PR.save_checkpoint, meta))


if __name__ == "__main__":
    main()
