#!/usr/bin/env python3
"""S43 main rows: `tools/trk6/make_rows.sh` (the S39-stack evaluations aggregated by `evalx.py row --variant w200cfg`), then
the cost columns of the sparse trunk replaced by measurements of the sparse implementation (pre-registration G5):

    macs_forward_packet          median multiply-accumulates the active-site executor performs per query on the zgz
                                 evaluation steps (S39 stack, from tools/s43/async_eval.py at hop 50); the thop count of the
                                 masked dense training form is kept as `macs_dense_equiv_thop`
    latency_ms_raw               mean per-step forward (render of the previous state + trunk on active sites + prev_mlp) over
                                 the 600 lyq_local packets `evalx.latency_model` times, same anchor scaling; the masked dense
                                 forward's figure is kept as `latency_ms_raw_masked_dense`

    python tools/s43/make_row.py --gpu 7 --arm s43_sp --runs s43_sp_s3407 s43_sp_s3408 s43_sp_s3409 \
        --async-json outputs/s43/async/s43_sp_s3407_off_h50.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "tracking"), str(REPO / "tools" / "dt")]


@torch.no_grad()
def sparse_latency(model, cfg, device, max_packets=600, passes=3):
    from semkine import eval_track as ET
    from semkine import events as EV
    from sparse_trunk import SparseExecutor
    ex = SparseExecutor(model.conv1, model.rn)
    root = Path(cfg["DATA"]["ROOT"])
    events, offsets, aux, pos51 = ET.load_sequence(root, "train", "lyq_local")
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    model.set_hand_context(betas, K)
    ev_ch = EV.event_channels(cfg)
    items = []
    for a, b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
        prev = torch.from_numpy(pos51[a].copy()).view(1, -1).to(device)
        for end in np.arange(a + 49, b, 50, dtype=np.int64):
            items.append((torch.from_numpy(ET.build_lnes(events, offsets, int(end), 50, ev_ch)).to(device), prev))
            if len(items) >= max_packets:
                break
        if len(items) >= max_packets:
            break

    def f(it):
        lnes, prev = it
        rend = model._render_prev(prev.float(), betas, K)[0]
        logits, _, _ = ex.run(torch.cat([lnes, rend], dim=-1))
        z = prev + logits[None] + model.prev_mlp(prev)
        if model.transl_head is not None:
            t = prev[:, :3] + model.transl_head(torch.cat([ex.last_feats[None], prev[:, :3]], dim=1))
            z = torch.cat([t, z[:, 3:]], dim=1)
        return z
    for it in items[:100]:
        f(it)
    torch.cuda.synchronize()
    means = []
    for _ in range(passes):
        t0 = time.perf_counter()
        for it in items:
            f(it)
        torch.cuda.synchronize()
        means.append((time.perf_counter() - t0) / len(items) * 1e3)
    return min(means)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--eval-root", default="outputs/s43/full")
    ap.add_argument("--async-json", required=True, help="tools/s43/async_eval.py output of the arm's first seed (hop 50)")
    ap.add_argument("--variant", default="w200cfg")
    ap.add_argument("--eval-json", default=None, help="file name under <eval-root>/<run>/ for a non-default variant")
    a = ap.parse_args()
    os.chdir(REPO)
    if a.variant == "w200cfg":
        subprocess.run(["tools/trk6/make_rows.sh", a.gpu, a.eval_root, a.arm] + a.runs, check=True)
    else:
        # another S39-stack evaluation of the same checkpoints (e.g. the deviation 7 dead band): link `--eval-json` of each
        # run as evalx_val_core_last_<variant>.json, then the very same `evalx.py row`
        dirs = []
        for run in a.runs:
            src = (REPO / a.eval_root / run / a.eval_json).resolve()
            if not src.exists():
                raise SystemExit(f"missing {src}")
            dst = REPO / "outputs" / "semkine" / run / f"evalx_val_core_last_{a.variant}.json"
            if dst.resolve() != src:                     # the protocol evaluation already sits under its own name
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(src)
            dirs.append(f"outputs/semkine/{run}")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu)
        subprocess.run([sys.executable, "tools/tracking/evalx.py", "row", "--arm", a.arm, "--runs", *dirs, "--ckpt", "last",
                        "--variant", a.variant], check=True, env=env)
    path = REPO / "outputs" / "semkine" / f"{a.arm}_{a.variant}_main_row.json"
    row = json.loads(path.read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    from config import load_config
    from model import MNISTModel
    import evalx as EX
    device = torch.device("cuda")
    run0 = REPO / "outputs" / "semkine" / a.runs[0]
    cfg = load_config(str(next(iter(sorted(run0.glob("*.yaml"))))))
    ck, _, _ = EX.find_ckpt(run0, "last")
    model = MNISTModel.load_from_checkpoint(str(ck), cfg=cfg, map_location=device).to(device).eval()
    lat = sparse_latency(model, cfg, device)
    asy = json.loads((REPO / a.async_json).read_text())
    row["latency_ms_raw_masked_dense"] = row["latency_ms_raw"]
    row["latency_ms_scaled_masked_dense"] = row["latency_ms_scaled_full1p75"]
    row["latency_ms_raw"] = lat
    row["latency_ms_scaled_full1p75"] = lat * 1.75 / row["latency_anchor_ms"]
    row["macs_dense_equiv_thop"] = row["macs_forward_packet"]
    row["macs_forward_packet"] = asy["macs"]["median"]
    row["macs_sparse_zgz"] = asy["macs"]
    row["s43_cost_note"] = ("S43 sparse trunk: latency = render + active-site executor + prev_mlp (eager, batch 1, fp32, "
                            "lyq_local 50 ms packets, x1.75 / anchor); MACs = median active-site MACs per query on the zgz "
                            f"S39-stack steps ({a.async_json}); masked dense / thop figures kept under *_masked_dense / "
                            "macs_dense_equiv_thop")
    path.write_text(json.dumps(row, indent=1))
    print(json.dumps({k: row[k] for k in ("latency_ms_raw", "latency_ms_scaled_full1p75", "latency_ms_raw_masked_dense",
                                          "latency_ms_scaled_masked_dense", "macs_forward_packet", "macs_dense_equiv_thop",
                                          "params_total")}, indent=1))


if __name__ == "__main__":
    main()
