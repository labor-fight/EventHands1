#!/usr/bin/env python3
"""S45-RT T0P gate (docs/EXPERIMENT_RECORDS.md#s45rt-prereg-20261031): read once, on the held-out subject ylf only.

Packets and gate set exactly as S45-TR's T0 gate (tools/s45tr/t0_gate.py): 8192 samples drawn once (rng 3407, without
replacement) from the ylf sequences of S45's training dataset in training mode with the XAG sources; gate set = on-hand
events (XAG valid) that the frozen S45's own association matched (q > 0, valid). On the same stride-sampled events,
with the same definitions for both routers:
  T0   routed where its on-hand probability >= 0.5 (S45-TR T1b's tau); face = argmax face, projected barycentrics
  T0P  routed where on-hand >= 0.5 and top patch probability >= 0.5; face = nearest face of the predicted patch
  coverage            routed / gate set
  part-cross share    routed face's part != the XAG face's part (part = majority skinning part of the face)
  face geodesic error median (template face-centroid geodesic, mm); template-point error median (mm)
PASS iff T0P's part-cross share <= T0's - 0.05 and T0P's coverage >= 0.5 (formal: exit code 3 otherwise, so T1p is
not trained). The preflight runs the same code on the 50-step router as an engineering check and never fails on the
criterion. Descriptive: S45 association template error on the gate set, on-hand AUC, sigma16 anchors.
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

from tools.s45tr.common import S45_CHECKPOINT, S45_CHECKPOINT_SHA256, XAG_SOURCES  # noqa: E402
from tools.s45rt.common import (PHYSICAL_GPUS, S45_CONFIG, T0_CHECKPOINT, now, read, require, save, sha,  # noqa: E402
                                verify_protocol)
from tools.s45rt.plan2 import HELDOUT, ROUTER_CHECKPOINT, ROUTER_DIR, ROUTER_PREFLIGHT_DIR, TAUS  # noqa: E402

N_SAMPLES, BATCH = 8192, 128
T0_TAU = 0.5
MARGIN, MIN_COVERAGE = 0.05, 0.5
SIGMA16 = {"face_geodesic_median_mm": 17.72, "part_cross_share": 0.271}


def auc(score, label):
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = label.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def metrics(routed, face, point, gt_face, gt_point, gate, face_part, geo):
    sel = gate & routed
    return {"coverage": float(sel.sum() / max(gate.sum(), 1)),
            "part_cross_share": float((face_part[face[sel]] != face_part[gt_face[sel]]).mean()) if sel.any() else None,
            "face_geodesic_median_mm": float(np.median(geo[face[sel], gt_face[sel]]) * 1000) if sel.any() else None,
            "template_median_mm": float(np.median(np.linalg.norm(point[sel] - gt_point[sel], axis=-1)) * 1000)
            if sel.any() else None,
            "routed_events": int(sel.sum())}


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--phase", choices=("preflight", "formal"), default="formal")
    a = p.parse_args()
    verify_protocol()
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, PHYSICAL_GPUS)), "one GPU among 2-7")
    from config import load_config
    from model import MNISTModel
    from semkine import s45rt_router as RT
    from semkine import s45tr_t0 as T0
    from semkine.dataset import build_dataset
    from semkine.events import collate_packets
    from semkine.s45ct_route import face_geodesic
    ckpt = ROUTER_CHECKPOINT if a.phase == "formal" else ROUTER_PREFLIGHT_DIR / "t0p_final.ckpt"
    target = (ROUTER_DIR if a.phase == "formal" else ROUTER_PREFLIGHT_DIR) / "gate.json"
    require(not target.exists(), "the gate is read once: " + str(target))
    require(sha(S45_CHECKPOINT) == S45_CHECKPOINT_SHA256, "S45 checkpoint changed")
    done = read(ckpt.with_suffix(".json"))
    require(done["status"] == "COMPLETE" and done["checkpoint_sha256"] == sha(ckpt), "T0P training not complete")
    dev = torch.device("cuda")
    cfg = load_config(str(S45_CONFIG))
    model = MNISTModel.load_from_checkpoint(str(S45_CHECKPOINT), cfg=cfg, map_location=dev, strict=True).to(dev).eval()
    model.xyz_keep_aux = True
    t0, _ = T0.load(T0_CHECKPOINT)
    t0 = t0.to(dev)
    net, meta = RT.load(ckpt)
    net = net.to(dev)
    require(meta["heldout"] == HELDOUT, "T0P held-out subject differs")
    face_part = net.face_part.cpu().numpy()
    geo = face_geodesic(net.tmpl.cpu().numpy(), net.faces.cpu().numpy())
    dcfg = load_config(str(S45_CONFIG))
    dcfg["DATA"]["XAG_SOURCE_ROOT"] = str(XAG_SOURCES)
    components = np.load(dcfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    ds = build_dataset(dcfg, "train", components, train=True)
    held = np.array([s.startswith(HELDOUT + "_") for s, _ in ds.sequences])
    rows = np.flatnonzero(held[ds.index[:, 0]])
    require(held.sum() == 8 and rows.size > N_SAMPLES, "held-out selection changed")
    pick = np.sort(np.random.default_rng(3407).choice(rows, N_SAMPLES, replace=False))
    keys = ("gate", "valid", "gt_face", "gt_point", "assoc_point", "t0_routed", "t0_face", "t0_point",
            "p_routed", "p_face", "p_point", "p_onhand")
    acc = {k: [] for k in keys}
    with torch.no_grad():
        for start in range(0, N_SAMPLES, BATCH):
            batch = collate_packets([ds[int(i)] for i in pick[start:start + BATCH]]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model.forward_packet(batch)
                o0 = t0(batch.events, batch.ptr, batch.delta_t_s)
                op = net(batch.events, batch.ptr, batch.delta_t_s)
            last = model.xyz_last
            src, mask, assoc = last["src"], last["mask"], last["assoc"]
            for o in (o0, op):
                require(torch.equal(o["mask"], mask) and torch.equal(o["src"][mask], src[mask]), "node mismatch")
            tgt = RT.targets(net, batch, src, mask)
            w = (assoc["A"].float() * (assoc["q"].float().unsqueeze(-1) > 0)).clamp_min(0)
            apoint = (w.unsqueeze(-1) * net.tmpl[assoc["vid"]]).sum(-2) / w.sum(-1, keepdim=True).clamp_min(1e-12)
            matched = mask & (assoc["q"] > 0) & assoc["valid"] & (w.sum(-1) > 0)
            f0, _, pt0 = t0.correspondence(o0)
            r0 = torch.sigmoid(o0["onhand"].float()) >= T0_TAU
            on = torch.sigmoid(op["onhand"].float())
            pprob = torch.softmax(op["patch"].float(), -1).max(-1).values
            rp = (on >= TAUS["ONHAND_TAU"]) & (pprob >= TAUS["PATCH_TAU"])
            fp, bp, _ = net.route(op, mask)
            ptp = (bp.unsqueeze(-1) * net.tmpl[net.faces[fp]]).sum(-2)
            live = mask
            for k, v in (("gate", tgt["valid"] & matched), ("valid", tgt["valid"]), ("gt_face", tgt["face"]),
                         ("gt_point", tgt["point"]), ("assoc_point", apoint), ("t0_routed", r0), ("t0_face", f0),
                         ("t0_point", pt0), ("p_routed", rp), ("p_face", fp), ("p_point", ptp), ("p_onhand", on)):
                acc[k].append(v[live].float().cpu() if v.dtype.is_floating_point else v[live].cpu())
    c = {k: torch.cat(v).numpy() for k, v in acc.items()}
    g = c["gate"].astype(bool)
    m0 = metrics(c["t0_routed"].astype(bool), c["t0_face"], c["t0_point"], c["gt_face"], c["gt_point"], g, face_part, geo)
    mp = metrics(c["p_routed"].astype(bool), c["p_face"], c["p_point"], c["gt_face"], c["gt_point"], g, face_part, geo)
    passed = bool(mp["part_cross_share"] is not None and m0["part_cross_share"] is not None and
                  mp["part_cross_share"] <= m0["part_cross_share"] - MARGIN and mp["coverage"] >= MIN_COVERAGE)
    every = metrics(np.ones_like(g), c["p_face"], c["p_point"], c["gt_face"], c["gt_point"], g, face_part, geo)
    result = {
        "status": ("PASS" if passed else "FAIL") if a.phase == "formal" else "ENGINEERING_ONLY",
        "criterion_met": passed, "phase": a.phase, "read_at_utc": now(),
        "t0p_checkpoint_sha256": sha(ckpt), "t0_checkpoint_sha256": sha(T0_CHECKPOINT),
        "s45_checkpoint_sha256": S45_CHECKPOINT_SHA256, "heldout": HELDOUT, "samples": N_SAMPLES,
        "live_events": int(len(g)), "gate_events": int(g.sum()),
        "rule": {"margin": MARGIN, "min_coverage": MIN_COVERAGE, "t0_tau": T0_TAU, **TAUS},
        "t0": m0, "t0p": mp,
        "descriptive": {
            "t0p_every_gate_event_unfiltered": every,
            "assoc_template_median_mm": float(np.median(np.linalg.norm(c["assoc_point"][g] - c["gt_point"][g], axis=-1)) * 1000),
            "t0p_onhand_auc": auc(c["p_onhand"], c["valid"].astype(bool)),
            "sigma16_anchor": SIGMA16, "n_patches": int(net.n_patches),
        },
    }
    save(target, result, exclusive=True)
    print(result["status"], {"t0": m0, "t0p": mp}, flush=True)
    if a.phase == "formal" and not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
