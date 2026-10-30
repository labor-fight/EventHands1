#!/usr/bin/env python3
"""S45-TR T-series routing accuracy at the arm's own closed-loop states (prereg: "T1b、T1d 闭环状态下的路由准确率").

For T1b / T1d on zgz_local (descriptive; zgz XAG sources are read for scoring only, never for training or selection):
replay every endpoint's packet from the arm's own E1 loop state (readouts/<arm>/zgz_e1, `incoming_used`), capture the
ray association before and after routing (`semkine.xag.route_association` wrapped), and score both against the XAG
source of each sampled event on the MANO rest template:
  routed share       routed events / live events, and / on-hand events
  routed accuracy    face agreement and template-point error (mm) of the routed correspondences
  ray association    template-point error of the same events under the unrouted (S45) association
One GPU among 2-7; writes readouts/<arm>/route_accuracy.json.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.s45tr.common import OUT, XAG_SOURCES, now, require, save, sha, verify_protocol  # noqa: E402


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--net", required=True, choices=("t1b", "t1d"))
    a = p.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, range(2, 8))), "one GPU among 2-7")
    import torch
    from semkine import eval_track as ET
    from semkine import xag as XAG
    from semkine.dataset import sequences_for_split, splits_manifest
    from semkine.xag_sources import load_sources
    from tools.s45tr.readouts import load
    from tools.xcl import evaluate as XE
    XE.configure_environment()
    XE.set_policy()
    verify_protocol()
    target = OUT / "readouts" / a.net / "route_accuracy.json"
    require(not target.exists(), "read once: " + str(target))
    device = torch.device("cuda")
    model, cfg, checkpoint = load(a.net, device)
    model.xyz_keep_aux = True
    with np.load(cfg["MANO"]["NPZ"], allow_pickle=False) as z:
        tmpl = torch.tensor(np.asarray(z["v_template"], np.float32), device=device)
        faces = torch.tensor(np.asarray(z["f"], np.int64), device=device)
    seen = {}
    native_route = XAG.route_association

    def spy(mode, assoc, *args, **kwargs):
        out, audit = native_route(mode, assoc, *args, **kwargs)
        seen.update(before=assoc, after=out, replaced=audit["xag_replaced"])
        return out, audit
    XAG.route_association = spy
    loop = np.load(OUT / "readouts" / a.net / "zgz_e1" / "states.npz", allow_pickle=False)
    root = Path(cfg["DATA"]["ROOT"])
    seq = "zgz_local"
    found = dict(sequences_for_split(root, "val_core", splits_manifest(cfg)))
    events, offsets, aux, gt = ET.load_sequence(root, found[seq], seq)
    tsub_path = root / found[seq] / f"{seq}_tsub.npy"
    tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
    betas = torch.tensor(np.asarray(aux["betas"], np.float32), device=device).view(1, -1)
    K = torch.tensor(np.asarray(aux["camera_K"], np.float32), device=device).view(1, 3, 3)
    model.set_hand_context(betas, K)
    rows = XE.endpoint_grid(aux)
    incoming = loop[f"{seq}/incoming_used"]
    require(len(incoming) == len(rows), "loop endpoint count differs")

    def point(vid, w):
        w = w.float().clamp_min(0)
        return (w.unsqueeze(-1) * tmpl[vid]).sum(-2) / w.sum(-1, keepdim=True).clamp_min(1e-12)
    acc = {k: [] for k in ("live", "onhand", "routed", "routed_onhand", "face_ok", "err_routed", "err_ray")}
    with torch.no_grad():
        for i, (_, end, _, _) in enumerate(rows):
            ev5 = ET._window_events(events, offsets, tsub, int(end), 50)
            prev = torch.from_numpy(np.ascontiguousarray(incoming[i], np.float32)).view(1, -1).to(device)
            packet = ET.make_eval_packet(ev5, prev, betas, K, 50, device)
            packet = ET.attach_xag_sources(packet, cfg, seq, offsets, int(end), 50, device)
            seen.clear()
            model.forward_packet(packet)
            last = model.xyz_last
            if not seen or last is None:
                continue
            src, mask = last["src"], last["mask"]
            start = int(end) - 49
            row_ids = np.arange(int(offsets[start]), int(offsets[int(end) + 1]), dtype=np.int64)
            event_ms = np.repeat(np.arange(start, int(end) + 1, dtype=np.int64),
                                 np.diff(offsets[start:int(end) + 2]).astype(np.int64))
            gtsrc = load_sources(XAG_SOURCES, seq, row_ids, event_ms)
            flat = src.reshape(-1).cpu().numpy()
            g_face = torch.from_numpy(np.asarray(gtsrc["xag_face"])[flat]).view(src.shape).to(device)
            g_bary = torch.from_numpy(np.asarray(gtsrc["xag_bary"], np.float32)[flat]).view(*src.shape, 3).to(device)
            g_valid = torch.from_numpy(np.asarray(gtsrc["xag_valid"])[flat]).view(src.shape).to(device) & mask
            g_valid = g_valid & (g_face >= 0) & (g_face < len(faces))
            g_point = point(faces[g_face.clamp(0, len(faces) - 1)], g_bary)
            routed = seen["replaced"] & mask
            after, before = seen["after"], seen["before"]
            routed_face_vid = after["vid"][..., :3]
            face_ok = (routed_face_vid.sort(-1).values == faces[g_face.clamp(0, len(faces) - 1)].sort(-1).values).all(-1)
            e_routed = (point(after["vid"], after["A"]) - g_point).norm(dim=-1) * 1000
            e_ray = (point(before["vid"], before["A"]) - g_point).norm(dim=-1) * 1000
            sel = routed & g_valid
            acc["live"].append(int(mask.sum())); acc["onhand"].append(int(g_valid.sum()))
            acc["routed"].append(int(routed.sum())); acc["routed_onhand"].append(int(sel.sum()))
            acc["face_ok"].append(face_ok[sel].cpu().numpy())
            acc["err_routed"].append(e_routed[sel].cpu().numpy()); acc["err_ray"].append(e_ray[sel].cpu().numpy())
    XAG.route_association = native_route
    fo, er, ey = (np.concatenate(acc[k]) for k in ("face_ok", "err_routed", "err_ray"))
    result = {
        "net": a.net, "sequence": seq, "loop_states": "readouts/" + a.net + "/zgz_e1 incoming_used",
        "checkpoint_sha256": sha(checkpoint), "script_sha256": sha(__file__), "read_at_utc": now(),
        "endpoints": len(acc["live"]),
        "routed_share_of_live": float(sum(acc["routed"]) / max(sum(acc["live"]), 1)),
        "routed_share_of_onhand": float(sum(acc["routed_onhand"]) / max(sum(acc["onhand"]), 1)),
        "routed_face_agreement": float(fo.mean()) if fo.size else None,
        "routed_template_error_mm_median": float(np.median(er)) if er.size else None,
        "ray_template_error_mm_median_same_events": float(np.median(ey)) if ey.size else None,
        "zgz_labels_scoring_only": True,
    }
    save(target, result, exclusive=True)
    print(result)


if __name__ == "__main__":
    main()
