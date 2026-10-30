#!/usr/bin/env python3
"""AURA C2: background injection (docs/EXPERIMENT_RECORDS.md#aura-prereg-20261024), descriptive, evaluation only.

E3-trained probes (MR@c1, MR@surf, MR@b32, MR@mM, MR@aura10, M@p2dh, loaded from their Stage C processes) are
evaluated on all zgz endpoints after adding uniformly distributed background events (pixel, time in the window and
polarity uniform; 10 % / 30 % of each packet's own count, rng 9090 + endpoint) merged in time order. The injected rows
carry no source (index -1, `xag_valid` False). Readout: the single-step RA change, i.e. whether the aura keeps
background off the mesh.

    CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID python tools/aura/c_background.py --out outputs/aura/c/background/run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.agg778 import common as C  # noqa: E402
from tools.agg778 import b_task as BT  # noqa: E402
from tools.aura import common as AC  # noqa: E402
from tools.aura import c_task as CT  # noqa: E402

TAKE = {"e3_aura1": ("MR@c1", "MR@surf", "MR@b32"), "e3_aura2": ("MR@mM", "MR@aura10"), "e3_geo": ("M@p2dh",)}
FRACS = (0.0, 0.1, 0.3)


def inject(item, frac, rng):
    ev, src = item["ev5"], item["src"]
    k = int(round(frac * len(ev)))
    if k == 0:
        return item
    add = np.zeros((k, 5), np.float32)
    add[:, 1] = rng.integers(0, 240, k)
    add[:, 2] = rng.integers(0, 180, k)
    add[:, 3] = np.round(rng.uniform(0.0, 0.05, k) * 1e6) * 1e-6
    add[:, 4] = rng.integers(0, 2, k)
    ev2 = np.concatenate([ev, add])
    order = np.argsort(ev2[:, 3], kind="stable")
    s2 = {}
    for key, v in src.items():
        v = np.asarray(v)
        fill = {"xag_source_index": -1, "xag_source_ms": -1, "xag_part": -1, "xag_face": -1, "xag_valid": False,
                "xag_depth": np.nan}.get(key, 0)
        pad = np.full((k,) + v.shape[1:], fill, dtype=v.dtype)
        s2[key] = np.concatenate([v, pad])[order]
    return dict(item, ev5=ev2[order], src=s2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--probe-root", default=str(AC.OUT / "c" / "probe"))
    a = ap.parse_args()
    dev = torch.device("cpu") if a.cpu else AC.require_gpu()
    if not a.smoke:
        AC.verify_protocol("C")
    C.set_policy()
    out = Path(a.out)
    if (out / "result.json").exists():
        raise SystemExit(f"{out} holds a result; preserve it")
    out.mkdir(parents=True, exist_ok=True)
    frozen, cfg = C.load_frozen("A0", dev)
    mano = C.mano32(cfg, dev)
    arms, names = [], []
    for proc, wanted in TAKE.items():
        st = torch.load(Path(a.probe_root) / proc / "arms.pt", map_location=dev)
        for sd, (arm, s) in zip(st["arms"], st["names"]):
            if arm not in wanted:
                continue
            m = CT.Arm(arm, frozen).to(dev)
            m.load_state_dict(sd)
            m.eval()
            arms.append(m)
            names.append((arm, s))
    need = set()
    for arm, _ in names:
        need |= CT.needs({arm: 1})
    ds = C.Endpoints(C.zgz_seqs())
    if a.smoke:
        ds.index = ds.index[:16]
    res_all = {}
    from tools.xcl.evaluate import score
    for frac in FRACS:
        preds = [[] for _ in arms]
        tgt, meta = [], []
        for items in C.loader(ds, 32, False, 0, workers=0 if a.cpu else a.workers):
            items = [inject(x, frac, np.random.default_rng(9090 + x["i"])) for x in items]
            anchors = np.stack([x["gt_prev"] for x in items])
            with torch.no_grad():
                inp = CT.batch_inputs(frozen, items, anchors, dev, 0, need)
                for k, m in enumerate(arms):
                    preds[k].append((inp["batch"].prev_state + m(inp)).float().cpu().numpy())
            tgt.append(np.stack([x["gt_end"] for x in items]))
            meta += [(x["seq"], x["si"], x["k"], x["rid"], x["blk"]) for x in items]
        res = BT.finish(preds, None, tgt, meta, ds, mano, dev, score)
        np.savez_compressed(out / f"eval_frac{int(frac * 100)}.npz", **res)
        for dom in ("local", "global"):
            sel = np.array([("_" + dom) in s for s in res["seq"]])
            if not sel.any():
                continue
            res_all[f"frac{int(frac * 100)}/{dom}"] = {f"{n}/{s}": float(res["mpjpe_ra_mm"][j, sel].mean())
                                                      for j, (n, s) in enumerate(names)}
    C.save_json(out / "result.json", {"names": [f"{n}/{s}" for n, s in names], "summary": res_all,
                                      "fracs": FRACS, "smoke": bool(a.smoke)}, exclusive=False)
    AC.receipt(out, {"stage": "C2", "smoke": bool(a.smoke)}, scripts=(__file__,))
    print("COMPLETE", out, json.dumps(res_all)[:300], flush=True)


if __name__ == "__main__":
    main()
