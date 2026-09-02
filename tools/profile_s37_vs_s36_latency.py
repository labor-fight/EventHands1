#!/usr/bin/env python3
"""H2 probe: S37 (FK-direct) against S36 (render) on the same real 50 ms packets.

Both models are timed in the same process on the same 600 lyq_local packets (batch=1,
min of 3 pass-means after 100-packet warmup), so the comparison cannot be confounded by
machine state. The conditioning stage is additionally timed alone for each arm:
S36 pays MANO FK + rasterize + per-event image query, S37 pays MANO FK + analytic
distances on the sampled nodes only.

    CUDA_VISIBLE_DEVICES=0 python tools/profile_s37_vs_s36_latency.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from config import load_config                       # noqa: E402
from model import MNISTModel                         # noqa: E402
from semkine import eval_track as ET                 # noqa: E402
from semkine.dataset import sequences_for_split      # noqa: E402
from semkine.encoder import query_render             # noqa: E402

STEP_MS = 50
#: published EventHands-Full latency the row scales to, and its raw measurement from
#: `outputs/semkine/s36_main_row.json` on this same machine
FULL_PUBLISHED_MS = 1.75


def build_packets(cfg, root, device, max_packets=600):
    seqs = sequences_for_split(root, "val_core", root / "_retired_splits_semkine_5v2v3.json")
    seq, d = next((s, d) for s, d in seqs if s == "lyq_local")
    events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
    tsub_p = root / d / f"{seq}_tsub.npy"
    tsub = np.load(tsub_p, mmap_mode="r") if tsub_p.exists() else None
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    packets = []
    for a, b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
        prev = torch.from_numpy(pos51[a].copy()).view(1, -1).to(device)
        for end in np.arange(a + STEP_MS - 1, b, STEP_MS, dtype=np.int64):
            ev5 = ET._window_events(events, offsets, tsub, int(end), STEP_MS)
            packets.append(ET.make_eval_packet(ev5, prev, betas, K, STEP_MS, device))
            if len(packets) >= max_packets:
                return packets, betas, K
    return packets, betas, K


def timeit(fn, items, passes=3, warm=100):
    for it in items[:warm]:
        fn(it)
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(passes):
        t0 = time.perf_counter()
        for it in items:
            fn(it)
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / len(items) * 1e3)
    return best


def load_arm(run_dir, device):
    run = REPO / run_dir
    sel = json.loads((run / "selection_val_core_step50"
                            "__retired_splits_semkine_5v2v3.json").read_text())["selected"]
    cfg = load_config(json.loads((run / "training_metadata.json").read_text())["config_path"])
    model = MNISTModel.load_from_checkpoint(sel["ckpt"], cfg=cfg,
                                            map_location=device).to(device).eval()
    return model, cfg


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda")
    s36, cfg36 = load_arm("outputs/semkine/s36_eventgnn_s3407", device)
    s37, _ = load_arm("outputs/semkine/s37_fkdirect_s3407", device)
    packets, betas, K = build_packets(cfg36, Path(cfg36["DATA"]["ROOT"]), device)
    for m in (s36, s37):
        m.set_hand_context(betas, K)
    n_ev = int(np.mean([int(b.events.shape[0]) for b in packets]))
    print(f"{len(packets)} packets, mean {n_ev} events")

    def cond36(b):
        bf, kf = s36._resolve_betas_K(b.prev_state, b.betas, b.camera_K)
        rend = s36._render_prev(b.prev_state.float(), bf, kf)
        return query_render(rend.to(dtype=b.events.dtype), b.events)

    def cond37(b):
        bf, kf = s37._resolve_betas_K(b.prev_state, b.betas, b.camera_K)
        verts_p, joints_p = s37._fk(b.prev_state.float(), bf)
        return s37._fk_extra(b.events, b.ptr, verts_p, joints_p, kf)

    t36_full = timeit(s36.forward_packet, packets)
    t37_full = timeit(s37.forward_packet, packets)
    t36_cond = timeit(cond36, packets)
    t37_cond = timeit(cond37, packets)

    scale = None
    row = REPO / "outputs/semkine/s36_main_row.json"
    if row.exists():
        r = json.loads(row.read_text())
        scale = FULL_PUBLISHED_MS / float(r["anchor_full_raw_ms"])

    print(f"\n{'arm':<28}{'full ms':>9}{'cond ms':>9}")
    print(f"{'S36 render+query':<28}{t36_full:9.3f}{t36_cond:9.3f}")
    print(f"{'S37 fk_direct':<28}{t37_full:9.3f}{t37_cond:9.3f}")
    if scale:
        print(f"\nscaled to Full={FULL_PUBLISHED_MS}ms anchor: "
              f"S36 {t36_full * scale:.2f} ms, S37 {t37_full * scale:.2f} ms "
              f"(S36 row recorded {json.loads(row.read_text())['latency_ms_scaled_full1p75']:.2f})")

    # #region agent log
    import time as _t
    dbg = {"sessionId": "ea00d3", "runId": "s37", "hypothesisId": "H2",
           "location": "tools/profile_s37_vs_s36_latency.py",
           "message": "latency probe", "timestamp": int(_t.time() * 1000),
           "data": {"s36_full_ms": t36_full, "s37_full_ms": t37_full,
                    "s36_cond_ms": t36_cond, "s37_cond_ms": t37_cond,
                    "scale_to_full": scale}}
    with open(REPO / ".cursor/debug-ea00d3.log", "a") as f:
        f.write(json.dumps(dbg) + "\n")
    # #endregion


if __name__ == "__main__":
    main()
