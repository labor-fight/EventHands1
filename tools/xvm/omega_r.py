#!/usr/bin/env python3
"""XVM Omega-R: does the fixed record -> vertex mean lose displacement? (docs/EXPERIMENT_RECORDS.md#xvm-prereg-20261021 §2)

Frozen XCL A0FOmega-o (step 6000). Its H3 is Linear(3->128)((V_gt - V_prev) / L_REL) on the directly observed vertices O.
Conditions (all with the A3 full-oracle routing: GT source face / barycentrics and GT per-event depth), only the
vector fed to that projection changes:

    R0_base  original base routing, native GT offsets           (harness sanity: must reproduce XCL E1 4.568 local)
    R0       A3 routing, native GT vertex offsets on O_A3
    R1       A3 routing, per-vertex weighted mean of the A3 record dXYZ = z_gt d - V_prev[vid]   (what A3F0 sees)
    R2       A3 routing, per-vertex weighted mean of the record's material-point motion z_gt d - sum_j A_j V_prev[vid_j]

Modes: e1 (own closed loop, protocol initial noise, rng 0) and e3 (GT previous endpoint each step).
Writes outputs/xvm/s1/omega_r/<cond>_<mode>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools")]
from xvm import common as C  # noqa: E402

RUN = REPO / "outputs/xcl/20261019/runs/xcl_a0foo_s3407"
CONDS = ("R0_base", "R0", "R1", "R2")


def record_offsets(d, mask, verts, assoc, cond):
    """(B, V, 3) per-vertex weighted mean of the records' displacement (metres) and the (B, V) mass."""
    B, N, R = assoc["A"].shape
    V = verts.shape[1]
    w = assoc["q"].unsqueeze(-1) * assoc["A"] * mask.float().unsqueeze(-1)
    z = assoc["z"]
    X = z.unsqueeze(-1) * d.float()                                                    # (B, N, 3) GT point
    vid = assoc["vid"]
    Vr = torch.gather(verts.float().unsqueeze(1).expand(B, N, V, 3), 2,
                      vid.unsqueeze(-1).expand(B, N, R, 3))                           # (B, N, R, 3)
    if cond == "R1":
        disp = X.unsqueeze(2) - Vr
    else:  # R2: motion of the material point the event sits on
        An = assoc["A"] / assoc["A"].sum(-1, keepdim=True).clamp_min(1e-12)
        Xp = (An.unsqueeze(-1) * Vr).sum(2)                                            # (B, N, 3)
        disp = (X - Xp).unsqueeze(2).expand(B, N, R, 3)
    key = (torch.arange(B, device=d.device).view(B, 1, 1) * V + vid).reshape(-1)
    wf = w.reshape(-1).double()
    mass = torch.zeros(B * V, dtype=torch.float64, device=d.device).index_add_(0, key, wf)
    num = torch.zeros(B * V, 3, dtype=torch.float64, device=d.device).index_add_(
        0, key, wf.unsqueeze(-1) * disp.reshape(-1, 3).double())
    est = (num / mass.clamp_min(1e-12).unsqueeze(-1)).view(B, V, 3).float()
    return est, mass.view(B, V)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=CONDS, required=True)
    ap.add_argument("--mode", choices=("e1", "e3"), required=True)
    a = ap.parse_args()
    dev = C.require_gpu()
    out = C.OUT / "s1" / "omega_r" / f"{a.cond}_{a.mode}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; preserve it")
    from tools.xcl.evaluate import set_policy, score, endpoint_grid
    from tools.xcl.interventions import apply, adapt_config
    set_policy()
    from config import load_config
    from mano_layer import ManoLayer
    from model import MNISTModel
    from semkine import eval_track as ET
    from semkine.dataset import sequences_for_split, splits_manifest
    cfg = load_config(str(RUN / "config_resolved.yaml"))
    ckpt = RUN / "last.ckpt"
    model = MNISTModel.load_from_checkpoint(str(ckpt), cfg=cfg, map_location=dev, strict=True).to(dev).eval()
    route = "none" if a.cond == "R0_base" else "route_a3"
    if route != "none":
        cfg = adapt_config(cfg, route)
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(dev).eval()
    xyz = model.xyz
    native_xyz = xyz.forward
    stat = {"n": 0, "obs": 0.0}

    def patched(tok, d, mask, verts, assoc, prev48, **kw):
        if a.cond in ("R1", "R2"):
            est, mass = record_offsets(d, mask, verts, assoc, a.cond)
            vg = kw["verts_gt"]
            kw["verts_gt"] = torch.where((mass > 0).unsqueeze(-1), verts.float() + est, vg.float()).to(vg.dtype)
        return native_xyz(tok, d, mask, verts, assoc, prev48, **kw)

    xyz.forward = patched
    seqs = sequences_for_split(Path(cfg["DATA"]["ROOT"]), "val_core", splits_manifest(cfg))
    native_fp = model.forward_packet
    rng = np.random.default_rng(0)
    res = {"cond": a.cond, "mode": a.mode, "ckpt": str(ckpt), "ckpt_sha256": C.sha(ckpt), "per_seq": {}}
    t0 = time.time()
    with apply(model, route):
        for s, d in seqs:
            _, _, aux, gt = ET.load_sequence(Path(cfg["DATA"]["ROOT"]), d, s)
            rows = endpoint_grid(aux)
            preds, targets = [], []

            def capture(packet):
                i = len(preds)
                rid, end, previous_ms, first = rows[i]
                if a.mode == "e3":
                    packet = replace(packet, prev_state=torch.from_numpy(np.asarray(gt[previous_ms], np.float32)).view(1, -1).to(dev))
                packet = replace(packet, target=torch.from_numpy(np.asarray(gt[end], np.float32)).view(1, -1).to(dev))
                pred = native_fp(packet)
                preds.append(pred.detach().cpu().numpy()[0].copy())
                targets.append(np.asarray(gt[end], np.float32).copy())
                return pred

            model.forward_packet = capture
            try:
                r = ET.track_sequence(model, mano, cfg, Path(cfg["DATA"]["ROOT"]), d, s, 50, dev, rng, 1.0, 1.0, None)
            finally:
                model.forward_packet = native_fp
            assert r["n_frames"] == len(preds) == len(rows)
            beta = torch.tensor(np.array(aux["betas"]), device=dev, dtype=torch.float32).view(1, -1)
            m = score(mano, np.stack(preds), np.stack(targets), beta, dev)
            res["per_seq"][s] = {k: float(v.mean()) for k, v in m.items()}
            print(a.cond, a.mode, s, json.dumps(res["per_seq"][s]), f"{time.time() - t0:.0f} s", flush=True)
    xyz.forward = native_xyz
    res["wall_s"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    C.receipt(out.parent, {"probe": "omega_r"}, scripts=(__file__,))


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
