#!/usr/bin/env python3
"""S45-CT step-0 proxy reproduction: the training hook's corruptions measured with S45-RF P2's informativeness I.

Same windows, priors, common event set E*, innovations, truth and formula as P2 on the decision domain (ylf, S45's
own closed-loop states from the N4 bank; tools/s45rf/p2.py functions, unchanged), with three sources: the exact XAG
face (must equal P2's face_oracle), and the face / barycentrics corrupted by semkine/s45ct_route.py `corrupt` for
sigma16 and shuffle. Registered pass: I(sigma16) inside P2's noise16 95 % interval widened by 0.02 on each side,
I(shuffle) inside P2's shuffled interval widened the same way, |I(face) - P2 face_oracle| <= 1e-6.

  proxy.py run --seq S   one GPU among 2-7        proxy.py analyze   CPU -> acceptance/proxy.json
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.s45ct.common import OUT, PHYSICAL_GPUS, now, read, require, save, sha, verify_protocol  # noqa: E402
from tools.s45rf import p2 as P2  # noqa: E402

PROXY = OUT / "acceptance" / "proxy"
P2_SUMMARY = ROOT / "outputs/s45rf/20261030/p2/summary_ylf.json"
SOURCES = ("face_oracle", "sigma16", "shuffle")
WIDEN = 0.02


def ylf_sequences():
    return [s for s, prior in P2.domain_jobs("ylf") if prior == "loop"]


def run(seq):
    import torch
    from config import load_config
    from model import MNISTModel
    from semkine import eval_track as ET
    from semkine import s45ct_route as H
    from semkine import xag as XAG
    from semkine.dataset import sequences_for_split, splits_manifest
    from tools.s45tr.common import N4_BANK, S45_CHECKPOINT, S45_CHECKPOINT_SHA256, S45_CONFIG, XAG_SOURCES
    from tools.xcl import evaluate as XE
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, PHYSICAL_GPUS)), "one GPU among 2-7")
    XE.configure_environment()
    XE.set_policy()
    verify_protocol()
    require(seq in ylf_sequences(), "unregistered proxy sequence")
    out = PROXY / seq
    require(not out.exists(), "preserve existing proxy job: " + str(out))
    require(sha(S45_CHECKPOINT) == S45_CHECKPOINT_SHA256, "S45 checkpoint changed")
    device = torch.device("cuda")
    cfg = load_config(str(S45_CONFIG))
    model = MNISTModel.load_from_checkpoint(str(S45_CHECKPOINT), cfg=cfg, map_location=device, strict=True).to(device).eval()
    model.xyz_keep_aux = True
    cfg_x = copy.deepcopy(cfg)
    cfg_x["DATA"]["XAG_SOURCE_ROOT"] = str(XAG_SOURCES)
    npz = cfg["MANO"]["NPZ"]
    faces, lbs = model.mano.f.long().to(device), model.xyz.lbs_w.to(device)
    vparts, fparts = lbs.argmax(-1), XAG.face_parts(lbs, faces)
    root = Path(cfg["DATA"]["ROOT"])
    found = dict(sequences_for_split(root, "train", splits_manifest(cfg)))
    events, offsets, aux, gt = ET.load_sequence(root, found[seq], seq)
    tsub_path = root / found[seq] / f"{seq}_tsub.npy"
    tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
    betas = torch.tensor(np.asarray(aux["betas"], np.float32), device=device).view(1, -1)
    K = torch.tensor(np.asarray(aux["camera_K"], np.float32), device=device).view(1, 3, 3)
    model.set_hand_context(betas, K)
    rows = XE.endpoint_grid(aux)
    with np.load(N4_BANK / f"{seq}.npz", allow_pickle=False) as z:
        require(np.array_equal(z["end_ms"], [r[1] for r in rows]), "bank grid differs")
        loop = np.concatenate([np.zeros((1, 51), np.float32), z["state"][:-1]])
    out.mkdir(parents=True)
    rec = {k: [] for k in ("end_ms", "D", *(f"cnt_{s}" for s in SOURCES), *(f"sum_{s}" for s in SOURCES))}
    with torch.no_grad():
        for i, (rid, end, prev_ms, first) in enumerate(rows):
            if first:
                continue
            prev = torch.from_numpy(np.ascontiguousarray(loop[i], np.float32)).view(1, -1).to(device)
            packet = ET.make_eval_packet(ET._window_events(events, offsets, tsub, int(end), 50), prev, betas, K, 50, device)
            packet = ET.attach_xag_sources(packet, cfg_x, seq, offsets, int(end), 50, device)
            model.xyz_last = None
            model.forward_packet(packet)
            last = model.xyz_last
            target = torch.from_numpy(np.asarray(gt[int(end)], np.float32)).view(1, -1).to(device)
            betas_f, _ = model._resolve_betas_K(prev, packet.betas, packet.camera_K)
            Vp, Ve = model._fk(prev.float(), betas_f)[0][0], model._fk(target.float(), betas_f)[0][0]
            rec["end_ms"].append(int(end))
            rec["D"].append(P2.part_truth(Vp, Ve, vparts).cpu().numpy())
            estar = None
            if last is not None and "assoc" in last:
                src, mask, assoc, d = last["src"], last["mask"], last["assoc"], last["d"]
                s = XAG.sample_sources(packet, src, mask)
                known = s["xag_valid"] & (s["xag_face"] >= 0) & (s["xag_face"] < len(faces))
                known = known & torch.isfinite(s["xag_bary"]).all(-1) & (s["xag_bary"] >= 0).all(-1) & (s["xag_bary"].sum(-1) > 0)
                matched = mask & (assoc["q"] > 0) & assoc["valid"] & (assoc["A"].clamp_min(0).sum(-1) > 0)
                estar = known & matched
            if estar is None or not bool(estar.any()):
                for k in SOURCES:
                    rec[f"cnt_{k}"].append(np.zeros(P2.N_PARTS, np.int64)); rec[f"sum_{k}"].append(np.zeros((P2.N_PARTS, 2)))
                continue
            face, bary, rows_ = s["xag_face"].long(), s["xag_bary"].float(), s["xag_source_index"]
            variants = {"face_oracle": (face, bary), "sigma16": H.corrupt("sigma16", face, bary, estar, rows_, npz),
                        "shuffle": H.corrupt("shuffle", face, bary, estar, rows_, npz)}
            ray_xy = d[0][estar[0]][:, :2].double()
            for k, (f, b) in variants.items():
                ff, bb = f[0][estar[0]].long(), b[0][estar[0]].float()
                cnt, total = P2.pool(ray_xy - P2.project(P2.face_point(Vp, faces, ff, bb).double()), fparts[ff])
                rec[f"cnt_{k}"].append(cnt.cpu().numpy()); rec[f"sum_{k}"].append(total.cpu().numpy())
    np.savez_compressed(out / "states.npz", **{k: np.asarray(v) for k, v in rec.items()})
    save(out / "report.json", {"seq": seq, "status": "COMPLETE", "windows": len(rec["end_ms"]), "finished_utc": now()},
         exclusive=True)


def analyze():
    verify_protocol()
    target = OUT / "acceptance" / "proxy.json"
    require(not target.exists(), "proxy reproduction is read once")
    seqs = ylf_sequences()
    parts = []
    for seq in seqs:
        with np.load(PROXY / seq / "states.npz", allow_pickle=False) as z:
            parts.append({k: z[k] for k in z.files})
    group = np.concatenate([np.full(len(p["end_ms"]), i) for i, p in enumerate(parts)])
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    vis = cat["cnt_face_oracle"] >= P2.M_MIN
    I = {s: P2.informativeness(*P2.window_terms(cat["D"], cat[f"cnt_{s}"], cat[f"sum_{s}"], vis, group)[:3])[0]
         for s in SOURCES}
    ref = read(P2_SUMMARY)["priors"]["loop"]["sources"]
    lo16, hi16 = ref["noise16"]["I_ci95"]
    losh, hish = ref["shuffled"]["I_ci95"]
    checks = {"face_oracle_reproduced": abs(I["face_oracle"] - ref["face_oracle"]["I"]) <= 1e-6,
              "sigma16_in_noise16_interval": lo16 - WIDEN <= I["sigma16"] <= hi16 + WIDEN,
              "shuffle_in_shuffled_interval": losh - WIDEN <= I["shuffle"] <= hish + WIDEN}
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "I": I, "checks": checks,
              "p2_reference": {"face_oracle": ref["face_oracle"]["I"], "noise16": ref["noise16"], "shuffled": ref["shuffled"]},
              "widen": WIDEN, "p2_summary_sha256": sha(P2_SUMMARY), "computed_utc": now()}
    save(target, result, exclusive=True)
    print(result, flush=True)
    require(result["status"] == "PASS", "proxy reproduction failed")


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("action", choices=("run", "analyze"))
    p.add_argument("--seq")
    a = p.parse_args(argv)
    if a.action == "run":
        require(a.seq is not None, "--seq required")
        run(a.seq)
    else:
        analyze()


if __name__ == "__main__":
    main()
