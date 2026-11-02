#!/usr/bin/env python3
"""T1A-UB latency breakdown (descriptive; GPU 4 exclusive, after a block's main rows; never trains or scores).

    python tools/t1aub/breakdown.py --net N --block B      -> outputs/t1aub/20261102/evaluations/<N>/breakdown/report.json

Same network / implementation as the block's main row (`t1aref_<B>` = T1a original, `t1arp_<B>` = T1a deployment form,
arms and cells in deployment form), same 600 lyq_local packets. Each pass times every packet end to end
(CUDA-synchronised) and, in a separate pass, every wrapped stage inclusively with a synchronisation on entry and exit:
  tokens (semkine.encoder.event_tokens), sample (stride_sample), fk (MNISTModel._fk), graph (EventGNN._edges: kNN
  construction), event_features (node_features / reparam_features), route (XAG sample_sources: the router, or the label
  sources), assoc (associate_rays: ray casting, empty under PFA), route_assoc (XAG route_association: PFA / face_route),
  trunk (XYZMeshNet.forward: records, phi, 778 pooling, MeshGNN, slots, readout; includes event_features).
The stage pass's own synchronisations inflate its total; the end-to-end pass is reported next to it. Raw ms and ms scaled
by the EventHands-Full anchor (as make_s36_row) are both written.
"""
from __future__ import annotations

import argparse
import functools
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.t1aub import common as C  # noqa: E402


def packets_for(model, cfg, device, max_packets=600):
    import torch
    from tools import make_s36_row as R
    from semkine import eval_track as ET
    from semkine.dataset import sequences_for_split, splits_manifest
    root = Path(cfg["DATA"]["ROOT"])
    seq, d = next((s, d) for split in ("val_core", "train")
                  for s, d in sequences_for_split(root, split, splits_manifest(cfg)) if s == "lyq_local")
    events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
    tsub_p = root / d / f"{seq}_tsub.npy"
    tsub = np.load(tsub_p, mmap_mode="r") if tsub_p.exists() else None
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    model.set_hand_context(betas, K)
    out = []
    for a, b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
        prev = torch.from_numpy(pos51[a].copy()).view(1, -1).to(device)
        for end in np.arange(a + R.STEP_MS - 1, b, R.STEP_MS, dtype=np.int64):
            ev5 = ET._window_events(events, offsets, tsub, int(end), R.STEP_MS)
            p = ET.make_eval_packet(ev5, prev, betas, K, R.STEP_MS, device)
            out.append(ET.attach_xag_sources(p, cfg, seq, offsets, int(end), R.STEP_MS, device))
            if len(out) >= max_packets:
                return out
    return out


def main(argv=None):
    import torch
    from tools.xcl import evaluate as XE
    from tools import make_s36_row as R
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--net", required=True)
    p.add_argument("--block", required=True)
    a = p.parse_args(argv)
    C.require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(C.LATENCY_GPU), "breakdowns are measured on GPU 4 only")
    XE.configure_environment()
    XE.set_policy()
    C.verify_protocol()
    ref, rp = a.net == f"t1aref_{a.block}", a.net == f"t1arp_{a.block}"
    net = "t1a" if ref or rp else a.net
    checkpoint, _, _, _ = C.resolve(net)
    cfg = C.load_config_for(net)
    deploy = not ref
    state = C.install_for_config(cfg, evaluation=True, fast=deploy, reparam=deploy and bool(cfg["T1AUB"]["EG"]))
    out = C.OUT / "evaluations" / a.net / "breakdown"
    C.require(not out.exists(), "preserve existing breakdown: " + str(out))
    out.mkdir(parents=True)
    device = torch.device("cuda")
    from model import MNISTModel
    model = MNISTModel.load_from_checkpoint(str(checkpoint), cfg=cfg, map_location=device, strict=True).to(device).eval()
    packets = packets_for(model, cfg, device)
    anchor = R.latency_anchor(device)
    totals = {}
    on = {"v": False}

    def timed(name, fn):
        @functools.wraps(fn)
        def wrap(*args, **kwargs):
            if not on["v"]:
                return fn(*args, **kwargs)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = fn(*args, **kwargs)
            torch.cuda.synchronize()
            totals[name] = totals.get(name, 0.0) + (time.perf_counter() - t0) * 1e3
            return r
        return wrap
    from semkine import encoder as ENC, xag as XAG, xyz_mesh as XM, event_gnn as EG, s45tr_t1a as T1A, t1aub as M
    patches = [(ENC, "event_tokens"), (XM, "stride_sample"), (XM, "associate_rays"), (XAG, "sample_sources"),
               (XAG, "route_association"), (T1A, "node_features"), (M, "reparam_features")]
    saved = [(mod, attr, getattr(mod, attr)) for mod, attr in patches]
    names = {"event_tokens": "tokens", "stride_sample": "sample", "associate_rays": "assoc", "sample_sources": "route",
             "route_association": "route_assoc", "node_features": "event_features", "reparam_features": "event_features"}
    for mod, attr, fn in saved:
        setattr(mod, attr, timed(names[attr], fn))
    edges = EG.EventGNN._edges
    EG.EventGNN._edges = timed("graph", edges)
    model._fk = timed("fk", model._fk)
    model.xyz.forward = timed("trunk", model.xyz.forward)
    try:
        with torch.no_grad():
            for b_ in packets[:100]:
                model.forward_packet(b_)
            torch.cuda.synchronize()
            e2e = []
            for _ in range(3):
                t = []
                for b_ in packets:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    model.forward_packet(b_)
                    torch.cuda.synchronize()
                    t.append((time.perf_counter() - t0) * 1e3)
                e2e.append(t)
            on["v"] = True
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for b_ in packets:
                model.forward_packet(b_)
            torch.cuda.synchronize()
            staged_total = (time.perf_counter() - t0) * 1e3
            on["v"] = False
    finally:
        for mod, attr, fn in saved:
            setattr(mod, attr, fn)
        EG.EventGNN._edges = edges
    n = len(packets)
    best = np.asarray(e2e).min(0)
    scale = R.FULL_PUBLISHED_MS / anchor
    report = {"net": a.net, "block": a.block, "installed": state, "deployment_form": deploy, "packets": n,
              "anchor_raw_ms": anchor, "end_to_end_raw_ms": {"mean": float(best.mean()), "p50": float(np.percentile(best, 50)),
                                                            "p99": float(np.percentile(best, 99))},
              "end_to_end_scaled_ms_mean": float(best.mean()) * scale,
              "staged_pass_raw_ms_per_packet": staged_total / n,
              "stages_raw_ms_per_packet": {k: v / n for k, v in sorted(totals.items())},
              "stages_scaled_ms_per_packet": {k: v / n * scale for k, v in sorted(totals.items())},
              "checkpoint_sha256": C.sha(checkpoint), "script_sha256": C.sha(__file__), "finished_utc": C.now()}
    C.save(out / "report.json", report, exclusive=True)
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
