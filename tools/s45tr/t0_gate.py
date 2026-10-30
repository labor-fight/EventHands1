#!/usr/bin/env python3
"""S45-TR T0 gate (docs/EXPERIMENT_RECORDS.md#s45tr-prereg-20261030): read once, on the held-out subject ylf only.

Packets: 8192 samples drawn once (rng 3407, without replacement) from every ylf sequence of S45's training dataset in
training mode, i.e. S45's own augmentation and mixed prior-noise contract, with the XAG per-event sources. On the same
stride-sampled events:
  * T0's correspondence (argmax face + projected barycentrics -> MANO rest-template point);
  * S45's own ray association (the frozen S45, its prior = the packet's noisy prior): template point
    sum_r A_r v_tmpl[vid_r] / sum_r A_r of the events it matched (q > 0, valid);
  * the XAG source template point.
Gate set = on-hand events (XAG valid) that S45's association matched. PASS iff T0's median template error on the gate
set is <= 8.0 mm and below the association's median on the same set; otherwise T1b is not trained (exit code 3).
Also reported (descriptive): T0 on every on-hand event, split by association hit (surface) / near-band (contour);
face / part accuracy; on-hand AUC.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.s45tr.common import (OUT, S45_CHECKPOINT, S45_CHECKPOINT_SHA256, S45_CONFIG, T0_CHECKPOINT, XAG_SOURCES,  # noqa: E402
                                now, read, require, save, sha, verify_protocol)

N_SAMPLES, BATCH = 8192, 128
GATE_MM = 8.0


def auc(score, label):
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = label.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--phase", choices=("preflight", "formal"), default="formal")
    a = p.parse_args()
    verify_protocol()
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, range(2, 8))), "one GPU among 2-7")
    from config import load_config
    from model import MNISTModel
    from semkine import s45tr_t0 as T0
    from semkine.dataset import build_dataset
    from semkine.events import collate_packets
    ckpt = T0_CHECKPOINT if a.phase == "formal" else OUT / "t0_preflight" / "t0_final.ckpt"
    target = (T0_CHECKPOINT.parent if a.phase == "formal" else OUT / "t0_preflight") / "gate.json"
    require(not target.exists(), "the gate is read once: " + str(target))
    require(sha(S45_CHECKPOINT) == S45_CHECKPOINT_SHA256, "S45 checkpoint changed")
    done = read(ckpt.with_suffix(".json"))
    require(done["status"] == "COMPLETE" and done["checkpoint_sha256"] == sha(ckpt), "T0 training not complete")
    dev = torch.device("cuda")
    cfg = load_config(str(S45_CONFIG))
    model = MNISTModel.load_from_checkpoint(str(S45_CHECKPOINT), cfg=cfg, map_location=dev, strict=True).to(dev).eval()
    model.xyz_keep_aux = True
    net, meta = T0.load(ckpt)
    net = net.to(dev)
    dcfg = load_config(str(S45_CONFIG))
    dcfg["DATA"]["XAG_SOURCE_ROOT"] = str(XAG_SOURCES)
    components = np.load(dcfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    ds = build_dataset(dcfg, "train", components, train=True)
    held = np.array([s.startswith(meta["heldout"] + "_") for s, _ in ds.sequences])
    rows = np.flatnonzero(held[ds.index[:, 0]])
    require(held.sum() == 8 and rows.size > N_SAMPLES, "held-out selection changed")
    pick = np.sort(np.random.default_rng(3407).choice(rows, N_SAMPLES, replace=False))
    acc = {k: [] for k in ("t0_err", "assoc_err", "gate", "matched", "valid", "hit", "prob", "face_ok", "part_ok")}
    with torch.no_grad():
        for start in range(0, N_SAMPLES, BATCH):
            batch = collate_packets([ds[int(i)] for i in pick[start:start + BATCH]]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model.forward_packet(batch)
                out = net(batch.events, batch.ptr, batch.delta_t_s)
            last = model.xyz_last
            src, mask, assoc = last["src"], last["mask"], last["assoc"]
            require(torch.equal(out["mask"], mask) and torch.equal(out["src"][mask], src[mask]), "node mismatch")
            tgt = T0.targets(net, batch, src, mask)
            face, bary, point = net.correspondence(out)
            w = (assoc["A"].float() * (assoc["q"].float().unsqueeze(-1) > 0)).clamp_min(0)
            apoint = (w.unsqueeze(-1) * net.tmpl[assoc["vid"]]).sum(-2) / w.sum(-1, keepdim=True).clamp_min(1e-12)
            matched = mask & (assoc["q"] > 0) & assoc["valid"] & (w.sum(-1) > 0)
            live = mask
            acc["t0_err"].append(((point - tgt["point"]).norm(dim=-1) * 1000)[live].cpu())
            acc["assoc_err"].append(((apoint - tgt["point"]).norm(dim=-1) * 1000)[live].cpu())
            acc["valid"].append(tgt["valid"][live].cpu())
            acc["matched"].append(matched[live].cpu())
            acc["gate"].append((tgt["valid"] & matched)[live].cpu())
            acc["hit"].append(assoc["hit"][live].cpu())
            acc["prob"].append(torch.sigmoid(out["onhand"].float())[live].cpu())
            acc["face_ok"].append((face == tgt["face"])[live].cpu())
            acc["part_ok"].append((out["part"].argmax(-1) == tgt["part"])[live].cpu())
    c = {k: torch.cat(v).numpy() for k, v in acc.items()}
    g, v = c["gate"], c["valid"]
    t0_g, as_g = float(np.median(c["t0_err"][g])), float(np.median(c["assoc_err"][g]))
    passed = bool(t0_g <= GATE_MM and t0_g < as_g)
    result = {
        "status": "PASS" if passed else "FAIL", "phase": a.phase, "read_at_utc": now(),
        "t0_checkpoint_sha256": sha(ckpt), "s45_checkpoint_sha256": S45_CHECKPOINT_SHA256,
        "heldout": meta["heldout"], "samples": N_SAMPLES, "live_events": int(len(v)),
        "gate_events": int(g.sum()), "onhand_events": int(v.sum()),
        "gate": {"t0_median_mm": t0_g, "assoc_median_mm": as_g, "threshold_mm": GATE_MM},
        "descriptive": {
            "t0_onhand_median_mm": float(np.median(c["t0_err"][v])),
            "t0_onhand_p75_mm": float(np.percentile(c["t0_err"][v], 75)),
            "t0_gate_hit_median_mm": float(np.median(c["t0_err"][g & c["hit"]])),
            "t0_gate_near_median_mm": float(np.median(c["t0_err"][g & ~c["hit"]])) if (g & ~c["hit"]).any() else None,
            "assoc_gate_hit_median_mm": float(np.median(c["assoc_err"][g & c["hit"]])),
            "assoc_gate_near_median_mm": float(np.median(c["assoc_err"][g & ~c["hit"]])) if (g & ~c["hit"]).any() else None,
            "face_accuracy_onhand": float(c["face_ok"][v].mean()), "part_accuracy_onhand": float(c["part_ok"][v].mean()),
            "onhand_auc": auc(c["prob"], v), "onhand_share": float(v.mean()),
            "assoc_matched_share_of_onhand": float((g.sum()) / max(v.sum(), 1)),
            "t0_prob_ge_tau_share_of_gate": float((c["prob"][g] >= 0.5).mean()),
        },
    }
    save(target, result, exclusive=True)
    print(result["status"], result["gate"], flush=True)
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
