#!/usr/bin/env python3
"""T1A-UB router variants: training (two GPUs among 2-7; fresh-only outputs; seed 3407) under srdwkd's recipe, unchanged.

  router.py train  --name N --phase {preflight,formal}     the variant registered as N in tools/t1aub/routers.json
  router.py anneal --name N --phase {preflight,formal}     continue the variant `anneal_of` for `anneal_steps` steps

The recipe is S45-SM srdwkd's (tools/s45sm/registry_r8.json; tools/s45sm/router_train.py components, imported unchanged):
S45's training data and augmentation with S45-AT dz's depth scale mode and the XAG per-event sources, minus every sequence of
ylf; two-GPU DDP, 512 packets per GPU, Adam 1e-3, 500-step warmup then constant, BF16, 6000 steps (50 in a preflight);
T0P's patch / rest-template point / on-hand loss plus distillation from the frozen T0P (279-patch KL at T 2, lambda 1 / 0.5 /
0.5). Anneal = srdwkda's: from the variant's final weights, fresh Adam, LR warmed up over 100 steps to 1e-3 and decayed
linearly to 0 at `anneal_steps` (2000). Seed placement is srdwkd's: teacher, output directory, seed_everything(3407), data,
manual_seed(3407), build. The only difference from srdwkd is the variant's one change (semkine/t1aub_router.py).
Writes outputs/t1aub/20261102/routers/<N>[_preflight]/<N>_final.ckpt and its completion JSON.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REGISTRY = Path(__file__).resolve().parent / "routers.json"
KD = {"teacher": "t0p", "T": 2.0, "lambda_patch": 1.0, "lambda_coord": 0.5, "lambda_onhand": 0.5}


def spec(name):
    from tools.t1aub import common as C
    routers = json.loads(REGISTRY.read_text())["routers"]
    C.require(name in routers, "unregistered T1A-UB router " + repr(name))
    return routers[name]


def out_dir(name, phase="formal"):
    from tools.t1aub import common as C
    return C.OUT / "routers" / (name + ("_preflight" if phase == "preflight" else ""))


def checkpoint(name, phase="formal"):
    return out_dir(name, phase) / f"{name}_final.ckpt"


def ready(name):
    """The router checkpoint arms / cells read: this study's completed formal training (verified by SHA)."""
    from tools.t1aub import common as C
    ck = checkpoint(name)
    done = C.read(ck.with_suffix(".json"))
    C.require(done["status"] == "COMPLETE" and done["phase"] == "formal" and done["checkpoint_sha256"] == C.sha(ck),
              "router training not complete: " + name)
    return ck


def _fresh(name, phase):
    from tools.t1aub import common as C
    out = out_dir(name, phase)
    if os.environ.get("LOCAL_RANK") in (None, "0"):
        C.require(not out.exists(), "Fresh-only output already exists: " + str(out))
        out.mkdir(parents=True)
    return out, checkpoint(name, phase)


def train(name, phase):
    import pytorch_lightning as pl
    import torch
    from tools.t1aub import common as C
    from tools.s45sm import router_train as RTR
    from semkine import t1aub_router as V
    s = spec(name)
    C.require("anneal_of" not in s and s["kind"] in V.KINDS and s.get("recipe") == "srdwkd", "a variant trained from scratch")
    C.verify_protocol()
    teacher = RTR._teacher({"kd": KD})
    out, target = _fresh(name, phase)
    pl.seed_everything(3407, workers=True)
    cfg = RTR._data({"dz": True})
    loader = RTR._loader(RTR.dataset(cfg, "ylf"))
    torch.manual_seed(3407)
    net = V.build(cfg["MANO"]["NPZ"], s["kind"])
    C.require(sum(p.numel() for p in net.parameters()) == int(s["params"]), "parameter count differs from the registration")
    steps = 6000 if phase == "formal" else 50
    module = RTR.PatchModule(net, None, teacher, KD)
    RTR._fit(module, loader, out, steps, RTR.Finish(target, phase, V.save_checkpoint,
                                                    {"heldout": "ylf", "kd": KD, "recipe": "srdwkd"}))


def anneal(name, phase):
    import pytorch_lightning as pl
    from tools.t1aub import common as C
    from tools.s45sm import router_train as RTR
    from semkine import t1aub_router as V
    s = spec(name)
    base = spec(s["anneal_of"])
    C.require(base["kind"] == s["kind"] and int(s["anneal_steps"]) > RTR.ANNEAL_WARMUP, "anneal of a trained variant")
    C.verify_protocol()
    base_ckpt = ready(s["anneal_of"])
    net, _ = V.load(base_ckpt)
    net.train()
    for q in net.parameters():
        q.requires_grad_(True)
    teacher = RTR._teacher({"kd": KD})
    out, target = _fresh(name, phase)
    pl.seed_everything(3407, workers=True)
    cfg = RTR._data({"dz": True})
    loader = RTR._loader(RTR.dataset(cfg, "ylf"))
    steps = int(s["anneal_steps"]) if phase == "formal" else 50
    module = RTR.PatchAnneal(net, None, teacher, KD)
    module.anneal_steps = steps
    RTR._fit(module, loader, out, steps, RTR.Finish(target, phase, V.save_checkpoint,
                                                    {"anneal_of": s["anneal_of"], "base_checkpoint_sha256": C.sha(base_ckpt),
                                                     "heldout": "ylf"}, anneal=True))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("train", "anneal"))
    p.add_argument("--name", required=True)
    p.add_argument("--phase", required=True, choices=("preflight", "formal"))
    a = p.parse_args(argv)
    (train if a.verb == "train" else anneal)(a.name, a.phase)


if __name__ == "__main__":
    main()
