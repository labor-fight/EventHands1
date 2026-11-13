#!/usr/bin/env python3
"""Real-study training entry (docs/EXPERIMENT_RECORDS.md#real-prereg-20261110, #real-amend1-20261110,
#real-amend2-20261110): tools/cmp/train.py inside the real study's namespace (tools/real/ns.py: per-arm canvas and data
root, real output and config directories, EvHandPose's real crop-frame translation mean).

  fresh    no run directory yet: tools/cmp/train.py `main` unchanged (fresh-only checks, cmp_launch.json).
  resume   (amendment 2, user 2026-11-10: "允许从检查点续训") the run directory exists because a gpu-borrow
           preemption killed the training: continue from its newest loadable checkpoint (the highest global_step among
           `*-step=N.ckpt` and `last.ckpt` that torch.load reads) with semkine/train.py `--resume` (Lightning restores
           weights, optimizer, LR schedule and step; the data order and RNG streams restart, so a resumed run is not
           bit-identical to an uninterrupted one). The same policy as the fresh path (deterministic flags, Trainer
           policy, CMP model class, datasets); cmp_launch.json of the first start is kept and each resume adds
           cmp_resume_<k>.json. A run directory without a loadable checkpoint is archived to pack/preempted/ and the
           training starts fresh.

    python tools/real/train.py --config configs/real_20261110/cmp_<arm>_s<seed>[_smoke].yaml
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.real import ns  # noqa: E402

ns.enter()

from tools.cmp import train as CT  # noqa: E402


def newest_checkpoint(out):
    import torch
    cands = []
    for p in out.glob("*.ckpt"):
        m = re.search(r"step=(\d+)\.ckpt$", p.name)
        cands.append((int(m.group(1)) if m else -1, p))
    best = None
    for _, p in sorted(cands, key=lambda c: c[0], reverse=True):
        try:
            step = int(torch.load(str(p), map_location="cpu", weights_only=False).get("global_step", -1))
        except Exception as e:                                         # noqa: BLE001  (a file cut by the kill)
            print(f"unreadable checkpoint {p.name}: {type(e).__name__}", flush=True)
            continue
        if best is None or step > best[0]:
            best = (step, p)
    return best


def resume_main(path, cfg, ckpt, step):
    """tools/cmp/train.py `main` without the fresh-only checks, ending in semkine/train.py --resume."""
    import torch
    from tools.cmp.arms import EXT
    from tools.cmp.common import now, save, sha
    c = cfg["CMP"]
    out = Path(cfg["TRAIN"]["OUTPUT_DIR"])
    det = c.get("DETERMINISTIC", True)
    CT.require(det in (True, "warn"), "CMP.DETERMINISTIC is true or warn")
    torch.use_deterministic_algorithms(True, warn_only=det == "warn")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    CT.install_policy(det)
    from semkine import train as T
    from semkine.cmp_models import model_class
    T.MNISTModel = model_class(c["MODEL"])

    class Throughput(T.ThroughputCallback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            super().on_train_batch_end(trainer, pl_module, outputs,
                                       (batch["lnes"],) if isinstance(batch, dict) else batch, batch_idx)
    T.ThroughputCallback = Throughput
    ext = EXT.get(c["ARM"])
    if ext is not None and hasattr(ext, "build_dataset"):
        native_build = T.build_dataset

        def build(cfg_, split, components, train=None, **kwargs):
            return ext.build_dataset(native_build(cfg_, split, components, train=train, **kwargs), cfg_, split)
        T.build_dataset = build
    elif c["STAGE"] == "seq":
        from semkine import cmp_seq as S
        native_build = T.build_dataset
        s = c["SEQ"]

        def build(cfg_, split, components, train=None, **kwargs):
            ds = native_build(cfg_, split, components, train=train, **kwargs)
            if split != "train":
                return ds
            if "CROP" in c:
                from semkine import cmp_seq_crop as SC
                return SC.upgrade(ds, int(s["K"]), int(s["PACKET_MS"]), int(s["VOXEL_BINS"]), int(c["CROP"]["SIZE"]),
                                  float(c["CROP"]["RATE"]), float(c["CROP"]["SHIFT"]))
            return S.upgrade(ds, int(s["K"]), int(s["PACKET_MS"]), int(s.get("VOXEL_BINS", 0)))
        T.build_dataset = build
    k = len(list(out.glob("cmp_resume_*.json"))) + 1
    save(out / f"cmp_resume_{k}.json", {"cmp": c, "config": str(path), "config_sha256": sha(path),
                                         "resume_from": str(ckpt), "resume_from_sha256": sha(ckpt),
                                         "resume_global_step": step,
                                         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                         "pid": os.getpid(), "started_utc": now(),
                                         "note": "#real-amend2-20261110: resumed after a gpu-borrow preemption"},
         exclusive=True)
    if (out / "cmp_timing.json").exists():                         # the timing callback rewrites it from row 0
        shutil.copy2(out / "cmp_timing.json", out / f"cmp_timing_before_resume_{k}.json")
    sys.argv = [sys.argv[0], "--config", str(path), "--resume", str(ckpt)]
    T.main()


def main():
    import yaml
    from tools.cmp.arms import ARMS, build_config, config_path
    CT.require(len(sys.argv) == 3 and sys.argv[1] == "--config", "usage: train.py --config <registered config>")
    path = Path(sys.argv[2]).resolve()
    cfg = yaml.safe_load(path.read_text())
    ns.set_frame(cfg)
    c = cfg["CMP"]
    CT.require(c["ARM"] in ARMS, "unregistered arm")
    CT.require(path == config_path(c["ARM"], c["SEED"], c["PHASE"]).resolve(), "unregistered config path")
    CT.require(cfg == build_config(c["ARM"], c["SEED"], c["PHASE"]), "config differs from the registry")
    out = Path(cfg["TRAIN"]["OUTPUT_DIR"])
    if out.exists() and c["PHASE"] == "formal":
        best = newest_checkpoint(out)
        if best is not None and (out / "cmp_launch.json").exists():
            step, ckpt = best
            print(f"resume {out.name} from {ckpt.name} (global_step {step})", flush=True)
            return resume_main(path, cfg, ckpt, step)
        dst = ns.OUT / "pack" / "preempted" / f"{out.name}__{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out), str(dst))
        print(f"no loadable checkpoint in {out.name}: archived to {dst}, fresh start", flush=True)
    CT.main()


if __name__ == "__main__":
    main()
