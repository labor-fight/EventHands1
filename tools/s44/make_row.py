#!/usr/bin/env python3
"""S44 main rows for the P0.10 closed-loop arms (`tools/s44/p10_replay.py` outputs), in the format
`tools/report_table.py` reads (`outputs/semkine/<arm>_main_row.json`).

Accuracy: per seed from the replay's per-step arrays (frame-weighted over zgz_global + zgz_local, exactly as
`evalx.aggregate_runs`); "two_seed_mean" is the N-seed mean. Latency: mean per-step time of the composite step (frozen
network forward + the fast path: SAE of the step, plane fit, Jacobian, owners, solve) over the first 600 protocol packets
of `lyq_local`, scaled by `evalx.latency_anchor` to the 1.75 ms full-model anchor, the main-row convention. MACs: the
network's thop count (copied from its own main row) plus the fast path's counted multiply-accumulates (dominated by the
103 MANO forward passes of the central-difference Jacobian, about 0.6 MMAC each). Params: the network's (the fast path
has no learned parameters; its fixed rate prior is 51 x 51 statistics of the training GT).

    python tools/s44/make_row.py --arm s44p10_s39ff --mode fast_fingers --hyper outputs/s44/p01_select/selected.json \
        --npz outputs/s44/p10/s39_s{seed}_fast_fingers.npz --base-row outputs/semkine/dt3_dz_l3_w200cfg_main_row.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "dt"))
from s44 import evflow as EF  # noqa: E402
from s44 import p10_replay as PR  # noqa: E402
import filter_eval as FE  # noqa: E402
from semkine import eval_track as ET  # noqa: E402
from semkine import events as EV  # noqa: E402

EX = FE.EX
KEYS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm", "root_rot_deg", "transl_mm")
MANO_FK_MACS = 0.6e6


def seed_metrics(npz):
    z = np.load(npz)
    out = {}
    for part, seqs in (("overall", ("zgz_global", "zgz_local")), ("local", ("zgz_local",)), ("global", ("zgz_global",))):
        n = sum(len(z[f"{s}|mpjpe_ra_mm"]) for s in seqs)
        out[part] = {k: float(sum(z[f"{s}|{k}"].sum() for s in seqs) / n) for k in KEYS}
    return out


@torch.no_grad()
def composite_latency(run, mode, hp, dev, max_packets=600, passes=3):
    cfg, model, _, _, _ = FE.build_run(Path(run), "last", dev)
    spec = json.loads(PR.SPEC.read_text())
    filt = PR.AdaptiveFilter.from_spec(PR._Fixed(), spec).to(dev).eval()
    sq = EF.Seq("lyq_local", "train")
    events, offsets, aux, pos51 = ET.load_sequence(EF.DATA_ROOT, "train", "lyq_local")
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=dev).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=dev).view(1, 3, 3)
    model.set_hand_context(betas, K)
    mano = EF.Mano64(dev)
    prec, _ = EF.rate_prior(dev)
    ev_ch = EV.event_channels(cfg)
    items = []
    for a, b in sq.runs:
        for end in np.arange(a + EF.STEP - 1, b, EF.STEP):
            items.append((int(a), int(end)))
            if len(items) >= max_packets:
                break
        if len(items) >= max_packets:
            break
    stats = {"n_meas": [], "n_used": []}

    def step(a, end):
        w = EX.window_of(offsets, end, "fixed", PR.WIN, 0, 300, floor=a)
        x = torch.from_numpy(ET.build_lnes(events, offsets, end, w, ev_ch)).unsqueeze(0).to(dev)
        prev = torch.from_numpy(pos51[a].copy()).view(1, -1).to(dev)
        z = model(x, prev)
        n_ev = int(offsets[end + 1] - offsets[end - EF.STEP + 1])
        xm = prev.clone()
        if mode != "control" and n_ev > 0:
            inc = PR.fast_increment(sq, a, end, prev.double(), betas.double(), sq.intrinsics(), mano, prec, hp, dev, stats)
            xm[:, EF.FINGER] = (prev[:, EF.FINGER].double() + inc[EF.FINGER]).float()
        filt.trk_model.z = z
        return filt(x, xm, n_events=n_ev)

    for a, e in items[:50]:
        step(a, e)
    torch.cuda.synchronize()
    best = []
    for _ in range(passes):
        t0 = time.perf_counter()
        for a, e in items:
            step(a, e)
        torch.cuda.synchronize()
        best.append((time.perf_counter() - t0) / len(items) * 1e3)
    used = float(np.mean(stats["n_used"])) if stats["n_used"] else 0.0
    return min(best), used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--hyper", required=True)
    ap.add_argument("--npz", required=True, help="path with {seed}")
    ap.add_argument("--runs", default="outputs/semkine/dt_dz_l3_s{seed}")
    ap.add_argument("--base-row", required=True)
    ap.add_argument("--seeds", default="3407,3408,3409")
    a = ap.parse_args()
    dev = torch.device("cuda")
    hp = json.loads(Path(a.hyper).read_text())
    seeds = [int(s) for s in a.seeds.split(",")]
    per_seed = {}
    for sd in seeds:
        m = seed_metrics(a.npz.format(seed=sd))
        meta = json.loads(Path(a.npz.format(seed=sd)).with_suffix(".json").read_text())
        per_seed[str(sd)] = {"step": meta["step"], "ckpt": meta["ckpt"], **m}
    mean = {part: {k: float(np.mean([per_seed[s][part][k] for s in per_seed])) for k in KEYS}
            for part in ("overall", "local", "global")}
    base = json.loads(Path(a.base_row).read_text())
    anchor = EX.latency_anchor(dev)
    raw, used = composite_latency(a.runs.format(seed=seeds[0]), a.mode, hp, dev)
    p = max(used, 1.0)
    fast_macs = 103 * MANO_FK_MACS + p * (25 * 12 + 4 * 2 * 51 + 51 * 51 * (EF.IRLS_ITERS + 1)) + 51 ** 3
    row = {"arm": a.arm, "split": "val_core", "ckpt_rule": "last", "n_seeds": len(seeds), "two_seed_mean": mean,
           "per_seed": per_seed, "latency_ms_raw": raw, "anchor_full_raw_ms": anchor,
           "latency_ms_scaled_full1p75": raw * 1.75 / anchor, "latency_packets": 600,
           "macs_network": base["macs_forward_packet"], "macs_fast_path": fast_macs,
           "macs_forward_packet": base["macs_forward_packet"] + fast_macs,
           "macs_forward_packet_note": "network thop + counted fast-path MACs (central-difference Jacobian: 103 MANO FK)",
           "params_total": base["params_total"], "mode": a.mode, "hyper": hp,
           "note": "S44 P0.10 zero-training composite: frozen network + event fast path; docs/EXPERIMENT_RECORDS.md#s44-prereg-20261019"}
    out = REPO / "outputs" / "semkine" / f"{a.arm}_main_row.json"
    out.write_text(json.dumps(row, indent=1))
    print(out, json.dumps({k: round(v["mpjpe_ra_mm"], 4) for k, v in mean.items()}), f"latency {row['latency_ms_scaled_full1p75']:.2f} ms")


if __name__ == "__main__":
    main()
