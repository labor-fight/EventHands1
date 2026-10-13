#!/usr/bin/env python3
"""(Server side, not shipped.) Export the S39 deployment package's data from the training repository:
weights/s39_s{seed}.pt, configs/s39.json, configs/template_init.json, tests/zgz_clip.npz (raw zgz events + the official
S39 evaluation's initial states and outputs, computed here with tools/tracking/evalx.py run_sequence itself).
    python server_export.py --out /path/EventHands_DAVIS346
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/data1/lyq/code/mesh/EventHands1")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "model")); sys.path.insert(0, str(REPO / "tools"))
from config import load_config                                        # noqa: E402
from model import MNISTModel                                          # noqa: E402
from semkine import eval_track as ET                                  # noqa: E402
from semkine.anchored import AdaptiveFilter                           # noqa: E402
from semkine.dataset import sequences_for_split                       # noqa: E402
from tracking import evalx                                            # noqa: E402

SEEDS = (3407, 3408, 3409)
SPEC = REPO / "outputs/dt3/gate_specs/ramp1000.json"
DATA = Path("/data1/lyq/code/mesh/EventHands/data/hand_data51")
RAW = Path("/data1/lyq/data/hand_data/hand_data/val")
N_GLOBAL_STEPS = 40
LOCAL_OFFSET_MS = 100_000          # local events are appended after global ones on one session clock


def ckpt(seed):
    return REPO / f"outputs/semkine/dt_dz_l3_s{seed}/dt_dz_l3_s{seed}-step=6000.ckpt"


def export_weights(out: Path):
    (out / "weights").mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        p = ckpt(s)
        ck = torch.load(str(p), map_location="cpu")
        sd = ck["state_dict"]
        keep = {k: v.clone() for k, v in sd.items() if k.startswith(("conv1.", "rn.", "prev_mlp.", "mano."))}
        assert set(keep) == set(sd), sorted(set(sd) - set(keep))
        n_par = sum(v.numel() for k, v in keep.items() if not k.startswith("mano.") and "running" not in k
                    and "num_batches" not in k)
        meta = {"arm": "S39: dt_dz_l3, W=200 + adaptive filter + evidence gate ramp1000", "seed": s, "step": 6000,
                "source": str(p), "source_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "params": n_par}
        torch.save({"state_dict": keep, "meta": meta}, out / "weights" / f"s39_s{s}.pt")
        print("weights", s, n_par)


def export_configs(out: Path):
    (out / "configs").mkdir(parents=True, exist_ok=True)
    spec = AdaptiveFilter.canonical_spec(json.loads(SPEC.read_text()))
    row = json.loads((REPO / "outputs/semkine/dt3_dz_l3_w200cfg_main_row.json").read_text())
    m = row["two_seed_mean"]
    cfg = {"name": "S39: dt_dz_l3, W=200 + adaptive filter + evidence gate ramp1000",
           "window_ms": 200, "count_window_ms": 50, "step_ms": 50,
           "filter": {"a_root": spec["a_root"], "a_rest": spec["a_rest"], "a_trans": spec["a_trans"]},
           "input": {"height": 180, "width": 240, "render_scale": 0.375, "lnes": "last timestamp per pixel/polarity, (ms - start) / w"},
           "official_zgz_3seed_mean_mm": {"mpjpe_local": m["local"]["mpjpe_ra_mm"], "mpjpe_global": m["global"]["mpjpe_ra_mm"],
                                           "mpvpe_local": m["local"]["mpvpe_ra_mm"], "mpvpe_global": m["global"]["mpvpe_ra_mm"],
                                           "ra_mpjpe_overall": m["overall"]["mpjpe_ra_mm"]},
           "count_reference": dict(json.loads(Path("/tmp/count_ref.json").read_text()),
                                   events_per_active_pixel=json.loads(Path("/tmp/epap_ref.json").read_text())),
           "training_camera": "v2e-simulated events from RealSense D435 RGB, 640 x 480, K = [[603.45, 0, 325.09], [0, 602.96, 242.10], [0, 0, 1]]"}
    (out / "configs" / "s39.json").write_text(json.dumps(cfg, indent=1))
    d = dict(sequences_for_split(DATA, "train"))
    _, _, aux, pos = ET.load_sequence(DATA, d["lyq_local"], "lyq_local")
    a = int(np.asarray(aux["valid_runs_ms"]).reshape(-1, 2)[0, 0])
    tpl = {"state51": [float(v) for v in np.asarray(pos[a], np.float32)],
           "note": "right hand, open, palm towards the camera, fingers up (first frame of training sequence lyq_local); "
                   "the tracker re-places it at the image centre, ~72 px wide in the 240 x 180 image"}
    (out / "configs" / "template_init.json").write_text(json.dumps(tpl, indent=1))


def raw_events(seq):
    from dv import AedatFile
    t, x, y, p = [], [], [], []
    with AedatFile(str(RAW / seq / "event" / f"{seq}_events.aedat4")) as f:
        for pk in f["events"].numpy():
            t.append(pk["timestamp"].astype(np.int64)); x.append(pk["x"].astype(np.uint16))
            y.append(pk["y"].astype(np.uint16)); p.append(pk["polarity"].astype(np.uint8))
    return np.concatenate(t), np.concatenate(x), np.concatenate(y), np.concatenate(p)


def export_tests(out: Path, device):
    (out / "tests").mkdir(parents=True, exist_ok=True)
    seqs = sequences_for_split(DATA, "val_core")
    runs = {}
    for s in SEEDS:
        cfg = load_config(str(REPO / f"outputs/semkine/dt_dz_l3_s{s}/config_resolved.yaml"))
        model = MNISTModel.load_from_checkpoint(str(ckpt(s)), cfg=cfg, map_location=device).to(device).eval()
        net = AdaptiveFilter.from_spec(model, AdaptiveFilter.canonical_spec(json.loads(SPEC.read_text()))).to(device).eval()
        rng = np.random.default_rng(0)
        with torch.no_grad():
            runs[s] = {seq: evalx.run_sequence(net, cfg, DATA, d, seq, device, rng, mode="model", wmode="fixed", win=200,
                                               clip_run_start=True, count_mode="last50") for seq, d in seqs}
        # the official npz of the restored S39 must match
        import glob
        z = np.load(glob.glob(str(REPO / f"outputs/trk6/s39_repro/dt_dz_l3_s{s}/*.npz"))[0])
        for seq, _ in seqs:
            assert np.array_equal(z[f"model|{seq}|pred"], runs[s][seq]["pred"]), (s, seq)
        print("official run reproduced", s)
    T, X, Y, P = [], [], [], []
    seg_a, seg_b, seg_init, end_ms, seg_id, gt, seq_name, refs = [], [], [], [], [], [], [], {s: [] for s in SEEDS}
    _, _, aux_l, _ = ET.load_sequence(DATA, dict(seqs)["zgz_local"], "zgz_local")
    betas = np.asarray(aux_l["betas"], np.float32)
    K = np.asarray(aux_l["camera_K"], np.float32).reshape(3, 3)
    for seq, d in seqs:
        _, _, aux, _ = ET.load_sequence(DATA, d, seq)
        assert np.allclose(np.asarray(aux["betas"]), betas) and np.allclose(np.asarray(aux["camera_K"]).reshape(3, 3), K)
        off = 0 if seq == "zgz_global" else LOCAL_OFFSET_MS
        r0 = runs[SEEDS[0]][seq]
        n = N_GLOBAL_STEPS if seq == "zgz_global" else len(r0["end"])
        valid = np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2)
        lo_ms, hi_ms = [], []
        for rid in np.unique(r0["run"][:n]):
            idx = np.where(r0["run"][:n] == rid)[0]
            a = int(r0["end"][idx[0]] - r0["elapsed"][idx[0]])
            k = len(seg_a)
            seg_a.append(a + off); seg_b.append(int(valid[rid, 1]) + off)
            seg_init.append(r0["prev"][idx[0]])
            for s in SEEDS:
                assert np.array_equal(runs[s][seq]["prev"][idx[0]], r0["prev"][idx[0]])
            end_ms.extend((r0["end"][idx] + off).tolist()); seg_id.extend([k] * len(idx)); gt.extend(r0["gt"][idx])
            seq_name.extend([seq] * len(idx))
            for s in SEEDS:
                refs[s].extend(runs[s][seq]["pred"][idx])
            lo_ms.append(a); hi_ms.append(int(r0["end"][idx[-1]]))
        t, x, y, p = raw_events(seq)
        ms = t // 1000
        keep = np.zeros(len(t), bool)
        for lo, hi in zip(lo_ms, hi_ms):
            keep |= (ms >= lo) & (ms <= hi)
        T.append(t[keep] + off * 1000); X.append(x[keep]); Y.append(y[keep]); P.append(p[keep])
        print(seq, "segments", len(lo_ms), "steps", n, "events", int(keep.sum()))
    t = np.concatenate(T)
    assert np.all(np.diff(t) >= 0)
    out_npz = out / "tests" / "zgz_clip.npz"
    np.savez_compressed(out_npz, t_us=t.astype(np.uint32), x=np.concatenate(X), y=np.concatenate(Y), p=np.concatenate(P),
                        width=640, height=480, name="zgz (v2e 640x480 training camera)", K=K, betas=betas,
                        seg_a=np.asarray(seg_a), seg_b=np.asarray(seg_b), seg_init=np.asarray(seg_init, np.float32),
                        end_ms=np.asarray(end_ms), seg_id=np.asarray(seg_id), gt=np.asarray(gt, np.float32),
                        seq_name=np.asarray(seq_name), **{f"ref_state_s{s}": np.asarray(refs[s], np.float32) for s in SEEDS})
    print("tests ->", out_npz, out_npz.stat().st_size / 1e6, "MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = Path(a.out)
    export_weights(out)
    export_configs(out)
    export_tests(out, torch.device(a.device))


if __name__ == "__main__":
    main()
